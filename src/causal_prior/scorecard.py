"""fasterrisk fit/eval helpers and the causal-discovery q dispatcher.

these are the generic, dataset-agnostic glue between the binarized design matrix,
the causal q sources in priors.py, and a FasterRisk scorecard. dataset-specific
loaders stay in the experiment scripts.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def import_fasterrisk():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
    return FasterRisk


def ece(y01, p, n_bins=10):
    """expected calibration error: weighted mean |accuracy - confidence| over bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y01[m].mean() - p[m].mean())
    return float(e)


def fit_fr(X: np.ndarray, y: np.ndarray, k: int, mu: float, q: np.ndarray | None):
    """Fit FasterRisk(k, mu, freq=q) on (X, y); return the fitted model."""
    FR = import_fasterrisk()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fr = FR(k=k, mu=float(mu), freq=None if q is None else q.astype(float))
        fr.fit(X, y)
    return fr


def score_auc(fr, X: np.ndarray, y: np.ndarray, model_idx: int = 0) -> float:
    """Held-out AUC from a fitted FasterRisk model; y may be signed or binary."""
    from sklearn.metrics import roc_auc_score
    yb = (y > 0).astype(int)
    if yb.sum() in (0, len(yb)):
        return float('nan')
    p = np.clip(fr.predict_proba(X, model_idx=model_idx), 1e-7, 1 - 1e-7)
    return float(roc_auc_score(yb, p))


def fit_eval(FasterRisk, X_tr, y_tr, X_te, y_te, mu, q, k, return_card=False):
    fr = FasterRisk(k=k, mu=float(mu), freq=q.astype(float) if q is not None else None)
    fr.fit(X_tr, y_tr)
    p = np.clip(fr.predict_proba(X_te), 1e-7, 1 - 1e-7)
    y01 = (y_te > 0).astype(int)
    out = {
        'auc': roc_auc_score(y01, p),
        'brier': brier_score_loss(y01, p),
        'ece': ece(y01, p),
        'nfeat': int(np.count_nonzero(fr.betas_[0])),
    }
    if return_card:  # full fitted scorecard for recording: betas, intercept, multiplier
        out['card'] = {'betas': np.asarray(fr.betas_[0]).tolist(),
                       'intercept': float(fr.beta0_[0]),
                       'multiplier': float(fr.multipliers_[0])}
    return out
