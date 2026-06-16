"""causal feature-selection baselines used for comparison.

pyCausalFS fisher-z markov blankets (IAMB, HITON-MB) and a soft prior built from
the same fisher-z search, for the soft-vs-hard ablation. the conditional-gaussian
blanket `bnlearn_mb` (the valid mixed-data selector) lives in priors.py.

fisher-z assumes joint gaussianity and is not valid on mixed data; these are a
predictive-filter reference, not a valid causal selector there. see §6.2b.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# pyCausalFS is vendored (not a pip package); put it on the path for the lazy
# CBD imports inside _cfs_fisherz.
_PYCFS = Path(__file__).resolve().parents[2] / 'external' / 'pyCausalFS'
if _PYCFS.is_dir() and str(_PYCFS) not in sys.path:
    sys.path.insert(0, str(_PYCFS))


def _cfs_fisherz(algo, X, y, alpha):
    """pyCausalFS Markov blanket with Fisher-Z (off-the-shelf, Gaussian CI test)."""
    from CBD.MBs.IAMB import IAMB
    from CBD.MBs.HITON.HITON_MB import HITON_MB
    fn = {'iamb': IAMB, 'hiton_mb': HITON_MB}[algo]
    data = pd.DataFrame(np.column_stack([X, (y > 0).astype(int)]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mb, _ = fn(data, X.shape[1], alpha, False)
    return sorted(int(j) for j in mb)


def iamb_soft_q(X, y, alpha, B, rng, algo='iamb'):
    """soft prior built from the SAME fisher-z markov-blanket search as the cfs
    arms: q_j = fraction of B subsamples in which feature j lands in the iamb
    blanket. identical form to the ges_cg stability-selection q, so feeding it into
    the soft prior isolates soft-vs-hard use of one fixed information source."""
    p = X.shape[1]
    counts = np.zeros(p)
    n = len(y)
    for _ in range(B):
        idx = rng.choice(n, size=max(20, n // 2), replace=False)  # subsample_fraction=0.5
        mb = _cfs_fisherz(algo, X[idx], y[idx], alpha)
        if mb:
            counts[mb] += 1
    return counts / B
