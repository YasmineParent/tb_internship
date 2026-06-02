"""Discovery-source q vectors via Meinshausen-Bühlmann stability selection.

Each function returns q in [0,1]^p over the p features (target excluded),
where q_j is the frequency over B subsamples that feature j is judged adjacent
to (or selected for) the target by the underlying discovery procedure (PC,
GES, or L1-logistic). These are the realistic inputs to the modified
FasterRisk soft prior; analysis-time synthetic q sources (oracle, uniform,
adversarial) live in src/causal_prior/q_sources.py.

The synthetic generator orders columns as 'x_0', ..., 'x_{p-1}', 'y'; the PC
and GES sources assume the target is the last column of the input matrix.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


_R_INITIALIZED = False
_R_CONVERTER = None


def _init_R_pcalg() -> None:
    """Lazy one-shot init of rpy2 + pcalg. Sets up the numpy<->R converter and
    loads pcalg in the global R session. Idempotent."""
    global _R_INITIALIZED, _R_CONVERTER
    if _R_INITIALIZED:
        return
    # set user library before importing rpy2 so R picks it up
    os.environ.setdefault('R_LIBS_USER', os.path.expanduser('~/R/library'))
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.conversion import Converter
    cv = Converter('numpy')
    cv += numpy2ri.converter
    _R_CONVERTER = cv
    ro.r('.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ro.r('suppressPackageStartupMessages(library(pcalg))')
    _R_INITIALIZED = True


def _subsample_indices(n: int, fraction: float,
                       rng: np.random.Generator) -> np.ndarray:
    m = int(np.floor(fraction * n))
    return rng.choice(n, size=m, replace=False)


def _pc_one_subsample_R(idx: np.ndarray, X: np.ndarray, y_continuous: np.ndarray,
                         p: int, alpha: float, m_max: int) -> np.ndarray:
    """One pcalg::pc call on a subsample. gaussCItest = Fisher Z on partial
    correlations. m.max caps the size of conditioning sets, recovering power
    on dense DAGs where larger separators lose Fisher Z power."""
    _init_R_pcalg()
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    data = np.column_stack([X[idx], y_continuous[idx]])
    with localconverter(_R_CONVERTER):
        ro.globalenv['X_sub'] = data
    ro.r('n_sub <- nrow(X_sub); p_sub <- ncol(X_sub)')
    ro.r('suffStat <- list(C = cor(X_sub), n = n_sub)')
    ro.r(f'res <- pc(suffStat, indepTest=gaussCItest, alpha={alpha}, '
         f'p=p_sub, m.max={m_max}, verbose=FALSE)')
    with localconverter(_R_CONVERTER):
        adj = np.array(ro.r('as(res@graph, "matrix")'))
    y_idx = p  # target is appended as the last column
    return ((adj[y_idx, :p] != 0) | (adj[:p, y_idx] != 0)).astype(int)


def pc_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                   B: int = 100, subsample_fraction: float = 0.5,
                   alpha: float = 0.1, m_max: int = 5,
                   n_jobs: int = 1,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability PC via pcalg (R backend); q_j = freq over B runs
    that x_j is adjacent to y.

    alpha=0.1 (looser than the textbook 0.05) and m.max=5 (cap conditioning
    set size) are tuned to recover PC's power on dense linear-Gaussian DAGs.
    Both are standard pcalg defaults in recent dense-DAG benchmarks; the
    underlying CI test is unchanged (Fisher Z on partial correlations).

    Each pc() call is ~0.1s at p=31, n=300, so we run B subsamples
    sequentially in a single R session. n_jobs is ignored.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    _init_R_pcalg()
    adjacencies = [_pc_one_subsample_R(idx, X, y_continuous, p, alpha, m_max)
                   for idx in indices]
    return np.sum(adjacencies, axis=0) / B


def _ges_one_subsample_R(idx: np.ndarray, X: np.ndarray, y_continuous: np.ndarray,
                          p: int) -> np.ndarray:
    """One pcalg::ges call on a subsample. Returns binary length-p adjacency to y.

    pcalg's essgraph adjacency M has M[i,j]=1 if i -> j (with both M[i,j] and
    M[j,i] set for undirected edges). We mark x_j as adjacent to y if either
    direction has an edge.
    """
    _init_R_pcalg()
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    data = np.column_stack([X[idx], y_continuous[idx]])
    with localconverter(_R_CONVERTER):
        ro.globalenv['X_sub'] = data
    ro.r('score <- new("GaussL0penObsScore", as.matrix(X_sub))')
    ro.r('res <- ges(score, verbose=FALSE)')
    with localconverter(_R_CONVERTER):
        adj = np.array(ro.r('as(res$essgraph, "matrix")'))
    y_idx = p  # target is appended as the last column
    return ((adj[y_idx, :p] != 0) | (adj[:p, y_idx] != 0)).astype(int)


def ges_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                    B: int = 100, subsample_fraction: float = 0.5,
                    n_jobs: int = 1,
                    rng: np.random.Generator | None = None
                    ) -> tuple[np.ndarray, int]:
    """Subsample-stability GES via pcalg (R backend); returns (q, n_timeouts).

    Each ges() call is ~0.03s at p=31, n=300 (vs causal-learn's minutes-to-
    timeout for dense cells), so we run B subsamples sequentially in a single
    R session by default. n_timeouts is always 0 with this backend; kept for
    signature compat with the legacy causal-learn version.

    n_jobs is ignored (left for compat); running R in joblib workers requires
    per-worker rpy2/pcalg init which dominates the runtime of the actual fit.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    _init_R_pcalg()
    adjacencies = [_ges_one_subsample_R(idx, X, y_continuous, p) for idx in indices]
    return np.sum(adjacencies, axis=0) / B, 0


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
