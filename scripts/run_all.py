"""Run every config x seed, skipping (config, seed) pairs whose result file already exists.

Each run is a separate process, so that peak-memory measurements do not leak across runs.

    python scripts/run_all.py --quick
    python scripts/run_all.py --configs experiments/configs/baseline_*.yaml --seeds 0 1 2 3 4
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hypergrad.experiment import load_config, result_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--configs", nargs="+", default=[str(REPO / "experiments/configs/*.yaml")])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dry-run", action="store_true", help="only list what would run")
    args = parser.parse_args()

    paths = sorted({p for pattern in args.configs for p in glob.glob(pattern)})
    if not paths:
        sys.exit("no config matched")
    failed = []
    for path in paths:
        cfg = load_config(path, quick=args.quick)
        for seed in args.seeds:
            out = result_path(cfg, seed, args.results_dir)
            if out.exists():
                print(f"skip  {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
                continue
            cmd = [
                sys.executable,
                str(REPO / "scripts/run.py"),
                "--config",
                path,
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            if args.quick:
                cmd.append("--quick")
            if args.results_dir:
                cmd += ["--results-dir", args.results_dir]
            print("run  ", " ".join(cmd[1:]))
            if not args.dry_run and subprocess.run(cmd).returncode != 0:
                failed.append((path, seed))
    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {failed}")


if __name__ == "__main__":
    main()
