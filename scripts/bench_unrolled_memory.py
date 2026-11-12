"""Peak GPU memory and wall-clock time of one unrolled hypergradient on SmallCNN.

Grid: inner batch size x unroll length K x checkpoint_every c (0 = full autograd graph).
Each cell runs in its own process: deterministic kernels, one warm-up call, then
``torch.cuda.reset_peak_memory_stats`` and ``torch.cuda.synchronize`` around every timed
call. Memory is ``torch.cuda.max_memory_allocated`` and includes the data set resident on
the GPU (reported separately as ``base_mb``). A cell that does not fit is reported as OOM.

    python scripts/bench_unrolled_memory.py
    python scripts/bench_unrolled_memory.py --batch 128 --k 200 --every 1 14

Writes results/benchmarks/unrolled_memory.json and prints a markdown table.
"""

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def measure(batch: int, k: int, every: int, repeats: int) -> dict:
    import torch

    from hypergrad.data import BatchStream, load_fashion_mnist
    from hypergrad.experiment import set_determinism
    from hypergrad.functional import get_params, make_forward
    from hypergrad.inner import DiffSGD
    from hypergrad.methods import unrolled_hypergrad
    from hypergrad.models import SmallCNN, param_groups
    from hypergrad.problem import classification_problem

    set_determinism(0)
    dev = torch.device("cuda")
    data = load_fashion_mnist(root=str(REPO / "data"), device=dev)
    model = SmallCNN().to(dev)
    problem = classification_problem(make_forward(model), param_groups(model))
    opt = DiffSGD(problem, momentum=0.9)
    hp = {
        "log_lr": torch.full((4,), math.log(0.01), device=dev),
        "log_wd": torch.full((4,), math.log(1e-4), device=dev),
    }
    params = get_params(model)
    stream = BatchStream(data.train, batch, seed=0)
    val = BatchStream(data.val, 500, seed=1).take(1)

    def call(n_steps: int):
        return unrolled_hypergrad(
            opt,
            params,
            opt.init_state(params),
            hp,
            stream.take(n_steps),
            val,
            checkpoint_every=every,
        )

    row = {"batch": batch, "K": k, "checkpoint_every": every}
    try:
        call(min(k, 2))  # warm-up: cuDNN/cuBLAS initialisation
        torch.cuda.synchronize()
        times, peaks = [], []
        for _ in range(repeats):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(dev)
            base = torch.cuda.memory_allocated(dev)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = call(k)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            peaks.append(torch.cuda.max_memory_allocated(dev))
            del res
        row.update(
            status="ok",
            time_s=statistics.median(times),
            times_s=times,
            peak_mb=max(peaks) / 2**20,
            base_mb=base / 2**20,
        )
    except torch.OutOfMemoryError:
        row.update(status="OOM")
    row["gpu"] = torch.cuda.get_device_name(dev)
    row["gpu_total_mb"] = torch.cuda.get_device_properties(dev).total_memory / 2**20
    return row


def table(rows: list[dict]) -> str:
    everys = sorted({r["checkpoint_every"] for r in rows})
    ks = sorted({r["K"] for r in rows})
    lines = [
        "| batch | c | " + " | ".join(f"K={k}" for k in ks) + " |",
        "|---|---|" + "---|" * len(ks),
    ]
    for b in sorted({r["batch"] for r in rows}, reverse=True):
        for c in everys:
            cells = []
            for k in ks:
                r = next(
                    (r for r in rows if (r["batch"], r["checkpoint_every"], r["K"]) == (b, c, k)),
                    None,
                )
                if r is None:
                    cells.append("")
                elif r["status"] == "OOM":
                    cells.append("OOM")
                else:
                    cells.append(f"{r['peak_mb'] / 1024:.2f} GB, {r['time_s']:.2f} s")
            label = "full graph" if c == 0 else str(c)
            lines.append(f"| {b} | {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", nargs="+", type=int, default=[128, 64, 32])
    parser.add_argument("--k", nargs="+", type=int, default=[10, 50, 100, 200])
    parser.add_argument("--every", nargs="+", type=int, default=[0, 1, 14])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=str(REPO / "results/benchmarks/unrolled_memory.json"))
    parser.add_argument("--cell", nargs=3, type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.cell:  # child process
        print(json.dumps(measure(*args.cell, repeats=args.repeats)))
        return

    rows = []
    for b in args.batch:
        for c in args.every:
            for k in args.k:
                cmd = [
                    sys.executable,
                    __file__,
                    "--cell",
                    str(b),
                    str(k),
                    str(c),
                    "--repeats",
                    str(args.repeats),
                ]
                out = subprocess.run(cmd, capture_output=True, text=True)
                lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
                if not lines:
                    row = {
                        "batch": b,
                        "K": k,
                        "checkpoint_every": c,
                        "status": "error",
                        "stderr": out.stderr[-2000:],
                    }
                else:
                    row = json.loads(lines[-1])
                rows.append(row)
                shown = (
                    f"{row['peak_mb']:.0f} MB, {row['time_s']:.2f} s"
                    if row["status"] == "ok"
                    else row["status"]
                )
                print(f"batch {b:4d}  c {c:3d}  K {k:4d}  {shown}", flush=True)

    import torch

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        from hypergrad.experiment import code_version

        json.dump({"torch": torch.__version__, "code": code_version(), "rows": rows}, f, indent=1)
    print()
    print(table(rows))


if __name__ == "__main__":
    main()
