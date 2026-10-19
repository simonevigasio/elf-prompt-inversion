"""Experiment setup: datasets, synthetic hidden states, and experiment runners.

Runs on CPU unless JAX_PLATFORMS is already set. This must happen before JAX
is imported, so it lives here rather than in each command.
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
