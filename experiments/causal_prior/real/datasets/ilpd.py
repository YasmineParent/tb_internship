"""Indian Liver Patient (ILPD) loader, via openml.

n=583, clinical: age + gender (categorical) + liver-function labs (mostly
continuous). target 1 = liver patient (the positive/event class), 2 = not. a
larger, near-continuous clinical benchmark.
"""
from experiments.causal_prior.real.datasets._openml import load_mixed


def load(args=None):
    return load_mixed('ilpd', 1, '1')
