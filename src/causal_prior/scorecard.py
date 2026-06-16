"""fasterrisk fit/eval helpers and the causal-discovery q dispatcher.

these are the generic, dataset-agnostic glue between the binarized design matrix,
the causal q sources in priors.py, and a FasterRisk scorecard. dataset-specific
loaders stay in the experiment scripts.
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score

from .priors import pc_stability_q, ges_stability_q, bnlearn_stability_q


def _import_fasterrisk():
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


def discover_q(qsrc, X, y, b, seed):
    """dispatch a stability-selection q over the original features.

    pc/ges are gaussian (pcalg); pc_cg/ges_cg are conditional-gaussian (bnlearn
    mi-cg / bic-cg) for mixed data. returns q in [0,1]^p.
    """
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if qsrc == 'ges':
            q, _ = ges_stability_q(X, y, B=b, subsample_fraction=0.5, rng=rng)
        elif qsrc == 'pc_cg':
            q = bnlearn_stability_q(X, y, method='mi-cg', B=b,
                                    subsample_fraction=0.5, rng=rng)
        elif qsrc == 'ges_cg':
            q = bnlearn_stability_q(X, y, method='bic-cg', B=b,
                                    subsample_fraction=0.5, rng=rng)
        else:
            q = pc_stability_q(X, y, B=b, subsample_fraction=0.5,
                               alpha=0.1, m_max=5, rng=rng)
    return q
