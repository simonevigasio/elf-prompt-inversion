# elf-prompt-inversion

- [Stage A: continuous backbone inversion](stage-a-backbone-inversion/README.md)
  estimates prompt/noise bottleneck codes from a declared ELF hidden-state observation.
- [Stage B: discrete prompt inversion](stage-b-prompt-inversion/README.md)
  recovers tokens from encoder or projected targets, including Stage A's recovered target.

The implemented Stage A package is `stage-a-backbone-inversion/`.

Run both stages with the repository-level launcher after installing the packages
and caching the pinned weights (see the Stage A README):

```bash
python3 run_pipeline.py
```

To observe the state after the third complete backbone block:

```bash
python3 run_pipeline.py --config stage-a-backbone-inversion/configs/layer_3.yaml
```

It selects a repository virtual environment, generates the included synthetic
observation, runs Stage A followed by Stage B, and saves results in a new
timestamped directory under `stage-a-backbone-inversion/outputs/`.

```bash
# Supply your own observation and its matching configuration.
python3 run_pipeline.py --observation /path/to/hidden-state.npy \
  --config /path/to/experiment.yaml --output /path/to/new-results

# Inspect the command without running the model.
python3 run_pipeline.py --dry-run
```

The launcher also works from another directory when invoked by its full path.
Use `--example` for a custom synthetic JSON, `--python` to select an interpreter,
and `--help` for all options. It reuses the integrated runner and its single loaded
model instance. Weights are loaded offline; downloads are a separate setup step.
