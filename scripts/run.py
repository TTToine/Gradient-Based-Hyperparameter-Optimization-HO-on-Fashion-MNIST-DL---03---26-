"""Run one experiment config for one seed and write results/<name>_seed<k>.json.

python scripts/run.py --config experiments/configs/baseline_default.yaml --seed 0 --quick
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hypergrad.experiment import load_config, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="small CPU-sized run")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    cfg = load_config(args.config, quick=args.quick)
    run_experiment(cfg, args.seed, device=args.device, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
