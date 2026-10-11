#!/usr/bin/env python3
"""Explicit run/summarize commands; no experiment is launched without arguments."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser(description="Oracle-corruption diagnostic, not end-to-end prompt recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume an explicitly configured local evaluation")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset", required=True, help="Local JSONL with string id and text fields")
    run.add_argument("--output", required=True)
    run.add_argument("--modes", nargs="+", choices=("encoder", "projected"))
    summary = commands.add_parser("summarize", help="Regenerate aggregate CSV without loading weights")
    summary.add_argument("--results", required=True)
    summary.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        from prompt_inversion.evaluate import run_evaluation

        run_evaluation(args.config, args.dataset, args.output, modes=args.modes)
    else:
        from prompt_inversion.results import summarize, write_timings

        rows = summarize(args.results, args.output)
        write_timings(args.results, Path(args.output).with_name("timings.csv"))
        print(f"Wrote {len(rows)} summary groups to {args.output}")


if __name__ == "__main__":
    main()
