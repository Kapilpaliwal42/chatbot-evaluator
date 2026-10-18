"""Prepare, freeze, run, and compare the local judge benchmark."""

import argparse
import json
from pathlib import Path
import sys

from evaluation.benchmarking import compare_runs, freeze_dataset, make_blind_review, run_benchmark, select_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    blind = sub.add_parser("blind", help="Prepare label-free inputs for independent agent review")
    blind.add_argument("--sources", nargs="+", type=Path, required=True)
    blind.add_argument("--dataset", type=Path, default=Path("evaluation/benchmark/v1"))
    freeze = sub.add_parser("freeze", help="Freeze independently agreed labels and disjoint splits")
    freeze.add_argument("--dataset", type=Path, default=Path("evaluation/benchmark/v1"))
    freeze.add_argument("--reviews", type=Path, required=True)
    run = sub.add_parser("run", help="Run or resume a fixed local model/split")
    run.add_argument("--dataset", type=Path, default=Path("evaluation/benchmark/v1"))
    run.add_argument("--split", choices=("calibration", "holdout"), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--timeout", type=float, default=240)
    run.add_argument("--base-url", default="http://127.0.0.1:11434")
    run.add_argument("--retry-errors", action="store_true")
    compare = sub.add_parser("compare")
    compare.add_argument("--results", nargs="+", type=Path, required=True)
    compare.add_argument("--out", type=Path, default=Path("evaluation/output/benchmark-comparison"))
    select = sub.add_parser("select", help="Record the preferred model using calibration results before holdout")
    select.add_argument("--results", nargs="+", type=Path, required=True)
    select.add_argument("--out", type=Path, default=Path("evaluation/benchmark/v1/selection.json"))
    args = parser.parse_args(argv)
    try:
        if args.action == "blind":
            print(json.dumps(make_blind_review(args.sources, args.dataset), indent=2))
        elif args.action == "freeze":
            print(json.dumps(freeze_dataset(args.dataset, args.reviews), indent=2))
        elif args.action == "run":
            return run_benchmark(args.dataset, args.split, args.model, args.out, args.base_url, args.timeout, args.retry_errors)
        elif args.action == "select":
            print(json.dumps(select_calibration(args.results, args.out), indent=2))
        else:
            print(json.dumps(compare_runs(args.results, args.out), indent=2))
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
