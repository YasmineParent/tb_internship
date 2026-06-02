"""Discovery-source q vectors via Meinshausen-Bühlmann stability selection.

Each function returns q in [0,1]^p over the p features (target excluded),
where q_j is the frequency over B subsamples that feature j is judged adjacent
to (or selected for) the target by the underlying discovery procedure (PC,
GES, or L1-logistic). These are the realistic inputs to the modified
FasterRisk soft prior; analysis-time synthetic q sources (oracle, uniform,
adversarial) live in src_tb/support_recovery/q_sources.py.

The synthetic generator orders columns as 'x_0', ..., 'x_{p-1}', 'y'; the PC
and GES sources assume the target is the last column of the input matrix.
"""

from __future__ import annotations

import signal
import warnings

import numpy as np
from joblib import Parallel, delayed
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


def _y_adjacency_from_graph(adj: np.ndarray, p: int) -> np.ndarray:
    """1 if x_j is adjacent to y (in either direction) in causal-learn's graph encoding."""
    y_idx = p  # target is appended as the last column
    return (
        (adj[y_idx, :p] != 0) | (adj[:p, y_idx] != 0)
    ).astype(int)


def _pc_one_subsample(idx: np.ndarray, X: np.ndarray, y_continuous: np.ndarray,
                      p: int, alpha: float, indep_test: str) -> np.ndarray:
    data = np.column_stack([X[idx], y_continuous[idx]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cg = pc(data, alpha=alpha, indep_test=indep_test, show_progress=False)
    return _y_adjacency_from_graph(cg.G.graph, p)


def pc_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                   B: int = 100, subsample_fraction: float = 0.5,
                   alpha: float = 0.05, indep_test: str = 'fisherz',
                   n_jobs: int = -1,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability PC: q_j = freq over B runs that x_j is adjacent to y.

    Bootstrap iterations run in parallel via joblib (n_jobs=-1 by default).
    Subsample indices are drawn sequentially from rng before dispatch, so the
    output is bit-identical regardless of n_jobs.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    adjacencies = Parallel(n_jobs=n_jobs)(
        delayed(_pc_one_subsample)(idx, X, y_continuous, p, alpha, indep_test)
        for idx in indices
    )
    return np.sum(adjacencies, axis=0) / B


class _GESTimeout(Exception):
    pass


def _ges_one_subsample(idx: np.ndarray, X: np.ndarray, y_continuous: np.ndarray,
                       p: int, score_func: str, max_parents: int | None,
                       timeout_seconds: float) -> tuple[np.ndarray, bool]:
    """Returns (adjacency, timed_out). On timeout the adjacency is zeros.

    Uses SIGALRM; safe inside loky workers since each runs in its own process
    and the GES call holds the worker's main thread.
    """
    data = np.column_stack([X[idx], y_continuous[idx]])

    def _on_alarm(signum, frame):
        raise _GESTimeout

    prev_handler = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            record = ges(data, score_func=score_func, maxP=max_parents)
        adj = _y_adjacency_from_graph(record['G'].graph, p)
        return adj, False
    except _GESTimeout:
        return np.zeros(p, dtype=int), True
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev_handler)


def ges_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                    B: int = 100, subsample_fraction: float = 0.5,
                    score_func: str = 'local_score_BIC',
                    max_parents: int | None = 10,
                    timeout_seconds: float = 1800.0,
                    n_jobs: int = -1,
                    rng: np.random.Generator | None = None
                    ) -> tuple[np.ndarray, int]:
    """Subsample-stability GES with Gaussian BIC; returns (q, n_timeouts).

    A per-subsample SIGALRM timeout (1800s default) bounds each call so a slow
    GES run cannot stall the whole bootstrap. Timed-out subsamples contribute
    zeros to the count, so q remains a clean frequency. At p<=20 the timeout
    rarely fires; at p=30 dense cells some subsamples will saturate it
    (intended; ges_n_timeouts is recorded per cell).

    Bootstrap iterations run in parallel via joblib (n_jobs=-1 by default).
    Subsample indices are drawn sequentially from rng before dispatch, so the
    output is bit-identical regardless of n_jobs.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_ges_one_subsample)(
            idx, X, y_continuous, p, score_func, max_parents, timeout_seconds,
        )
        for idx in indices
    )
    adjacencies = [adj for adj, _ in results]
    n_timeouts = sum(1 for _, timed_out in results if timed_out)
    return np.sum(adjacencies, axis=0) / B, n_timeouts


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
