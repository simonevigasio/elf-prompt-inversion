#!/usr/bin/env python3
"""Run the full experiment: ELF hidden state -> Stage A -> Stage B.

    python3 run_pipeline.py --config configs/pipeline/layer_3.yaml
    python3 run_pipeline.py --config configs/pipeline/layer_3.yaml --observation data/observations/<name>
    python3 run_pipeline.py --help

Works from any directory. Re-runs itself with the repository's .venv interpreter
when one exists. Details: experiments/pipeline.py.
"""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"


if __name__ == "__main__":
    venv_python = VENV / "bin/python"
    if venv_python.is_file() and Path(sys.prefix).resolve() != VENV.resolve():
        # Keep the unresolved venv path: resolving its symlink would leave the venv.
        os.execv(venv_python, [str(venv_python), *sys.argv])
    sys.path.insert(0, str(ROOT))
    from experiments.pipeline import main

    sys.exit(main())
