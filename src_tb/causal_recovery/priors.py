"""Causal-evidence prior sources for the §6.1 mechanism test.

Each function returns q in [0,1]^p, the evidence vector over the p features
(excluding the target). At mu=0 in the modified FasterRisk objective, q is
irrelevant; at mu>0 it biases support selection toward high-q features.

The synthetic generator orders columns as 'x_0', ..., 'x_{p-1}', 'y'; the PC
and GES sources assume the target is the last column of the input matrix.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


def _patch_causallearn_bic_for_numpy2() -> None:
    """causal-learn's local_score_BIC_from_cov calls float(...) on a (1,1) array,
    which raises on numpy >= 2. Replace with an .item()-based scalar extraction.
    Strict improvement; safe to apply globally."""
    import causallearn.score.LocalScoreFunction as _lsf
    import causallearn.score.LocalScoreFunctionClass as _lsfc

    def _fixed_bic_from_cov(Data, i, PAi, parameters=None):
        cov, n = Data
        lam = 0.5 if parameters is None else parameters.get('lambda_value', 0.5)
        sigma = cov[i, i]
        if len(PAi) > 0:
            yX = cov[np.ix_([i], PAi)]
            XX = cov[np.ix_(PAi, PAi)]
            try:
                XX_inv = np.linalg.inv(XX)
            except np.linalg.LinAlgError:
                XX_inv = np.linalg.pinv(XX)
            sigma = float(np.asarray(cov[i, i] - yX @ XX_inv @ yX.T).item())
        if sigma <= 0:
            sigma = np.finfo(float).eps
        return -0.5 * n * (1 + np.log(sigma)) - lam * (len(PAi) + 1) * np.log(n)

    _fixed_bic_from_cov.__name__ = 'local_score_BIC_from_cov'
    _lsf.local_score_BIC_from_cov = _fixed_bic_from_cov
    _lsfc.local_score_BIC_from_cov = _fixed_bic_from_cov


_patch_causallearn_bic_for_numpy2()

from causallearn.search.ConstraintBased.PC import pc  # noqa: E402
from causallearn.search.ScoreBased.GES import ges  # noqa: E402


def _subsample_indices(n: int, fraction: float,
                       rng: np.random.Generator) -> np.ndarray:
    m = int(np.floor(fraction * n))
    return rng.choice(n, size=m, replace=False)


def oracle_q(p: int, S_star, sigma: float = 0.0,
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


def adversarial_q(p: int, confounded) -> np.ndarray:
    """Indicator on the confounded-correlate set; deliberately wrong q."""
    q = np.zeros(p)
    q[list(confounded)] = 1.0
    return q


def pc_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                   B: int = 100, subsample_fraction: float = 0.5,
                   alpha: float = 0.05, indep_test: str = 'fisherz',
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability PC: q_j = freq over B runs that x_j is adjacent to y."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    y_idx = p  # target is appended as the last column
    counts = np.zeros(p, dtype=int)

    for _ in range(B):
        idx = _subsample_indices(n, subsample_fraction, rng)
        data = np.column_stack([X[idx], y_continuous[idx]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            cg = pc(data, alpha=alpha, indep_test=indep_test, show_progress=False)
        adj = cg.G.graph
        for j in range(p):
            # adjacency in either direction; PC may orient edges away from y
            # when CI tests are underpowered, and adjacency is the support-relevant signal
            if adj[y_idx, j] != 0 or adj[j, y_idx] != 0:
                counts[j] += 1

    return counts / B


def ges_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                    B: int = 100, subsample_fraction: float = 0.5,
                    score_func: str = 'local_score_BIC',
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability GES with Gaussian BIC; same adjacency definition as pc_stability_q."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    y_idx = p
    counts = np.zeros(p, dtype=int)

    for _ in range(B):
        idx = _subsample_indices(n, subsample_fraction, rng)
        data = np.column_stack([X[idx], y_continuous[idx]])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            record = ges(data, score_func=score_func, maxP=None)
        adj = record['G'].graph
        for j in range(p):
            if adj[y_idx, j] != 0 or adj[j, y_idx] != 0:
                counts[j] += 1

    return counts / B


def bootstrap_l1_q(X: np.ndarray, y: np.ndarray,
                   B: int = 100, subsample_fraction: float = 0.5,
                   C: float | None = None, cv: int = 5,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability L1-logistic: q_j = freq over B runs that beta_hat_j != 0.

    X is standardized per subsample so the L1 penalty is scale-comparable across
    features (otherwise larger-scale columns are selected preferentially).

    If C is None (default), the regularization is tuned once on the full
    standardized (X, y) via 5-fold LogisticRegressionCV, then held fixed across
    the B subsamples. This is the Meinshausen-Bühlmann recipe: frequency reflects
    sample variation, not regularization variation. Pass an explicit float to
    override.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape

    if C is None:
        X_full = StandardScaler().fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tuner = LogisticRegressionCV(
                penalty='l1', solver='liblinear',
                cv=cv, fit_intercept=True, max_iter=1000,
                scoring='neg_log_loss',
            )
            tuner.fit(X_full, y)
        C = float(tuner.C_[0])

    counts = np.zeros(p, dtype=int)
    for _ in range(B):
        idx = _subsample_indices(n, subsample_fraction, rng)
        Xb = StandardScaler().fit_transform(X[idx])
        yb = y[idx]
        clf = LogisticRegression(
            penalty='l1', solver='liblinear',
            C=C, fit_intercept=True, max_iter=1000,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf.fit(Xb, yb)
        counts += (np.abs(clf.coef_.ravel()) > 0).astype(int)

    return counts / B
