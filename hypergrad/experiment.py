"""Run one (config, seed) and write its result JSON.

Each result file records the fully resolved config, the seed, the code version, the
device, wall-clock time and peak memory, the final hyperparameters and their trajectory,
and the metrics of the shared final-training protocol (``train.train_final``).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml

from .data import load_fashion_mnist
from .hyperparams import build_hparams, readable
from .models import build_model, param_groups
from .train import TrainConfig, train_final

log = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[1]


# --- config -------------------------------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path, quick: bool = False) -> dict:
    """Read a YAML config; in quick mode its ``quick:`` section overrides the rest."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    quick_override = cfg.pop("quick", {}) or {}
    if quick:
        cfg = deep_merge(cfg, quick_override)
    cfg["quick"] = quick
    return cfg


def result_path(cfg: dict, seed: int, results_dir: str | Path | None = None) -> Path:
    """results/<name>_seed<k>.json; quick runs go to results/quick/ so they never mix with
    full runs."""
    base = Path(results_dir) if results_dir else REPO / "results"
    if cfg["quick"]:
        base = base / "quick"
    return base / f"{cfg['name']}_seed{seed}.json"


# --- provenance and resources ---------------------------------------------------------------


def code_version() -> dict:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "git_hash": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"git_hash": None, "git_dirty": None}


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def peak_memory(device: torch.device) -> dict:
    """GPU: torch.cuda.max_memory_allocated (tensors only). CPU: peak resident set size of
    the whole process (includes the dataset and the Python runtime)."""
    if device.type == "cuda":
        return {
            "peak_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "memory_kind": "cuda_max_memory_allocated",
        }
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)  # Windows
    if peak is None:
        import resource  # Unix: ru_maxrss is in KiB on Linux

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {"peak_memory_mb": peak / 2**20, "memory_kind": "process_peak_rss"}


def set_determinism(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


# --- methods ------------------------------------------------------------------------------


def method_fixed(cfg: dict, data, device: torch.device, seed: int) -> dict:
    """No search: the hyperparameters in the config are used as they are."""
    groups = param_groups(build_model(cfg["model"]))
    hp = build_hparams(cfg["hyperparams"], max(groups.values()) + 1, len(data.train), device)
    return {"hparams": hp, "trajectory": [], "search_steps": 0}


METHODS = {"fixed": method_fixed}


# --- entry point --------------------------------------------------------------------------


def run_experiment(
    cfg: dict, seed: int, device: str = "auto", results_dir: str | Path | None = None
) -> Path:
    dev = resolve_device(device)
    set_determinism(seed)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()

    dcfg = cfg["data"]
    data = load_fashion_mnist(
        root=str(REPO / "data"),
        n_train=dcfg["n_train"],
        n_val=dcfg["n_val"],
        n_test=dcfg.get("n_test"),
        split_seed=dcfg.get("split_seed", 0),
        noise_rate=dcfg.get("noise_rate", 0.0),
        noise_kind=dcfg.get("noise_kind", "symmetric"),
        noise_seed=dcfg.get("noise_seed", 1000),
        device=dev,
    )
    log.info(
        "data: train %d  val %d  test %d  corrupted %d",
        len(data.train),
        len(data.val),
        len(data.test),
        len(data.corrupted),
    )

    method = METHODS[cfg["method"]]
    t_search = time.perf_counter()
    search = method(cfg, data, dev, seed)
    search_time = time.perf_counter() - t_search
    log.info("hyperparameters: %s", readable(search["hparams"]))

    t_train = time.perf_counter()
    final = train_final(
        cfg["model"], search["hparams"], data, TrainConfig(**cfg["train"]), seed, dev
    )
    train_time = time.perf_counter() - t_train
    log.info(
        "test_acc %.4f  test_nll %.4f  (selected epoch %s)",
        final["test_acc"],
        final["test_nll"],
        final["best_epoch"],
    )

    result = {
        "name": cfg["name"],
        "method": cfg["method"],
        "seed": seed,
        "quick": cfg["quick"],
        "config": cfg,
        "code": code_version(),
        "env": {
            "device": str(dev),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "hyperparams": readable(search["hparams"]),
        "hyperparam_trajectory": search["trajectory"],
        "search_steps": search["search_steps"],
        "final": final,
        "corrupted_indices": data.corrupted.tolist(),
        "time_s": {
            "search": search_time,
            "final_train": train_time,
            "total": time.perf_counter() - t0,
        },
        **peak_memory(dev),
    }
    for key in ("cleaning",):
        if key in search:
            result[key] = search[key]

    path = result_path(cfg, seed, results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    log.info("wrote %s", path)
    return path
