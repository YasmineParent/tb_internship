"""Synthetic q-vector constructors for §6.1 baselines and probes.

These are analysis-time q sources used as soft-prior inputs to the modified
FasterRisk, constructed from ground-truth knowledge of S_star / confounded
(only available in synthetic settings). Real discovery sources (PC, GES,
bootstrap-L1) live in src_tb/causal_discovery/priors.py.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def oracle_q(p: int, S_star: Iterable[int], sigma: float = 0.0,
             rng: np.random.Generator | None = None) -> np.ndarray:
    """Ground-truth indicator on S_star with optional Gaussian noise clipped to [0,1]."""
    q = np.zeros(p)
    q[list(S_star)] = 1.0
    if sigma > 0:
        if rng is None:
            rng = np.random.default_rng()
        q = q + rng.normal(0.0, sigma, size=p)
    return np.clip(q, 0.0, 1.0)


def uniform_q(p: int, value: float = 0.5) -> np.ndarray:
    """Constant q; contract check (at mu>0 with uniform q, optimal support is vanilla)."""
    return np.full(p, value)


def adversarial_q(p: int, confounded: Iterable[int]) -> np.ndarray:
    """Indicator on the confounded-correlate set; deliberately wrong q."""
    q = np.zeros(p)
    q[list(confounded)] = 1.0
    return q
