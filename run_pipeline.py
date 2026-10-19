#!/usr/bin/env python3
"""Launch Stage A -> Stage B from any working directory using installed packages."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STAGE_A = ROOT / "stage-a-backbone-inversion"


def default_python():
    for directory in (STAGE_A, ROOT / "stage-b-prompt-inversion", ROOT):
        candidate = directory / ".venv/bin/python"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Preserve the virtualenv path: resolving its symlink loses the venv.
            return str(candidate)
    return sys.executable


def main():
    parser = argparse.ArgumentParser(
        description="Run continuous Stage A recovery followed by discrete Stage B recovery.",
        epilog="Requires the installed Stage A/B packages and cached pinned ELF weights. "
               "Defaults to the included synthetic example. User-supplied relative paths "
               "are relative to your current working directory.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--example", type=Path, help="Synthetic example JSON with text or token_ids")
    source.add_argument("--observation", type=Path, help="Existing full hidden-state .npy file")
    parser.add_argument("--config", type=Path, default=STAGE_A / "configs/first_pass.yaml")
    parser.add_argument("--output", type=Path, help="New results directory; defaults to a timestamped Stage A output")
    parser.add_argument("--noise-seed", type=int, help="Synthetic observation seed (default: 100)")
    parser.add_argument("--python", default=default_python(), help="Python executable; defaults to a repository virtualenv")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without loading weights or running either stage")
    args = parser.parse_args()
    if args.observation is not None and args.noise_seed is not None:
        parser.error("--noise-seed applies only to synthetic examples")
    if args.noise_seed is not None and not 0 <= args.noise_seed < 2**32:
        parser.error("--noise-seed must be between 0 and 2**32 - 1")

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output = STAGE_A / "outputs" / f"pipeline-{stamp}"
    command = [args.python, "-B", "-u", "-m", "backbone_inversion.experiment",
               "invert" if args.observation is not None else "synthetic",
               "--config", str(args.config.expanduser().absolute()),
               "--output", str(output.expanduser().absolute())]
    if args.observation is not None:
        command += ["--observation", str(args.observation.expanduser().absolute())]
    else:
        example = args.example if args.example is not None else STAGE_A / "examples/prompt.json"
        command += ["--example", str(example.expanduser().absolute()),
                    "--noise-seed", str(100 if args.noise_seed is None else args.noise_seed)]
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except OSError as error:
        parser.error(f"Could not launch Python: {error}")
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
