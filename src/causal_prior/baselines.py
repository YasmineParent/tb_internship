"""naive off-the-shelf causal feature-selection baselines (pyCausalFS).

cfs_fisherz runs pyCausalFS markov-blanket discovery (IAMB, HITON-MB) with the
fisher-z ci test. IAMB and HITON-MB are causal markov-blanket algorithms; the
fisher-z test assumes joint gaussianity and is invalid on mixed/one-hot data
(pyCausalFS ships no mixed-data test), so on these benchmarks they are a naive
off-the-shelf reference, not a valid mixed-data selector. these are the cfs_iamb
and cfs_hiton_mb hard baselines (the requested CFS comparison).

the VALID mixed-data selectors live in priors.py: bnlearn_mb (iamb + mi-cg, the
hard cfs_cg baseline) and bnlearn_mb_stability_q (iamb + mi-cg, used softly as the
method's iamb_soft prior). see §6.2b.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# pyCausalFS is vendored (not a pip package); put it on the path for the lazy
# CBD imports inside cfs_fisherz.
_PYCFS = Path(__file__).resolve().parents[2] / 'external' / 'pyCausalFS'
if _PYCFS.is_dir() and str(_PYCFS) not in sys.path:
    sys.path.insert(0, str(_PYCFS))


def cfs_fisherz(algo, X, y, alpha):
    """pyCausalFS Markov blanket with Fisher-Z (off-the-shelf, Gaussian CI test)."""
    try:
        from CBD.MBs.IAMB import IAMB
        from CBD.MBs.HITON.HITON_MB import HITON_MB
    except ImportError as e:
        raise ImportError(
            f'pyCausalFS not found at {_PYCFS}. it is not on pypi; clone it there:\n'
            '  git clone https://github.com/wt-hu/pyCausalFS '
            f'{_PYCFS.parent}/pyCausalFS-repo && '
            f'cp -r {_PYCFS.parent}/pyCausalFS-repo/pyCausalFS {_PYCFS}') from e
    fn = {'iamb': IAMB, 'hiton_mb': HITON_MB}[algo]
    data = pd.DataFrame(np.column_stack([X, (y > 0).astype(int)]))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            mb, _ = fn(data, X.shape[1], alpha, False)
    except Exception:
        # fisher-z inverts a covariance that is singular on collinear one-hot data;
        # the honest outcome is "no usable blanket", not a crashed run.
        return []
    return sorted(int(j) for j in mb)


def iamb_fisherz_stability_q(X, y, alpha, B, rng, algo='iamb'):
    """soft prior built from the SAME pyCausalFS fisher-z iamb search as the cfs_iamb
    hard baseline: q_j = fraction of B subsamples in which feature j lands in the
    fisher-z markov blanket. this is an ABLATION CONTROL, not the deployed method:
    its only job is to isolate soft-vs-hard at a FIXED ci test (fisher-z), matched to
    cfs_iamb, so the soft-vs-hard contrast is not confounded with the ci test. the
    method ships the mixed-data-valid mi-cg variant (priors.bnlearn_mb_stability_q).
    on one-hot data fisher-z is singular and cfs_fisherz returns [], so q collapses
    toward zero there; that is the honest ci-test effect, not a bug."""
    p = X.shape[1]
    counts = np.zeros(p)
    n = len(y)
    for _ in range(B):
        idx = rng.choice(n, size=max(20, n // 2), replace=False)  # subsample_fraction=0.5
        mb = cfs_fisherz(algo, X[idx], y[idx], alpha)
        if mb:
            counts[mb] += 1
    return counts / B
