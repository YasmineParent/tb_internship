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
_BNLEARN_INITIALIZED = False


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
                         p: int, alpha: float, m_max: int,
                         skel_method: str = 'original') -> np.ndarray:
    """One pcalg::pc call on a subsample. gaussCItest = Fisher Z on partial
    correlations. m.max caps the size of conditioning sets, recovering power
    on dense DAGs where larger separators lose Fisher Z power. skel_method
    'stable' is the order-independent skeleton (Colombo & Maathuis 2014)."""
    _init_R_pcalg()
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    data = np.column_stack([X[idx], y_continuous[idx]])
    with localconverter(_R_CONVERTER):
        ro.globalenv['X_sub'] = data
    ro.r('n_sub <- nrow(X_sub); p_sub <- ncol(X_sub)')
    ro.r('suffStat <- list(C = cor(X_sub), n = n_sub)')
    ro.r(f'res <- pc(suffStat, indepTest=gaussCItest, alpha={alpha}, '
         f'p=p_sub, m.max={m_max}, skel.method="{skel_method}", verbose=FALSE)')
    with localconverter(_R_CONVERTER):
        adj = np.array(ro.r('as(res@graph, "matrix")'))
    y_idx = p  # target is appended as the last column
    return ((adj[y_idx, :p] != 0) | (adj[:p, y_idx] != 0)).astype(int)


def pc_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                   B: int = 100, subsample_fraction: float = 0.5,
                   alpha: float = 0.1, m_max: int = 5,
                   stable: bool = False, n_jobs: int = 1,
                   rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability PC via pcalg (R backend); q_j = freq over B runs
    that x_j is adjacent to y. stable=True uses the order-independent skeleton
    (pc-stable, Colombo & Maathuis 2014), the standard fix to vanilla PC's
    order-dependence and instability.

    alpha=0.1 (looser than the textbook 0.05) and m.max=5 (cap conditioning
    set size) are tuned to recover PC's power on dense linear-Gaussian DAGs.
    Each pc() call is ~0.1s at p=31, n=300, run sequentially; n_jobs ignored.
    """
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    skel = 'stable' if stable else 'original'
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    _init_R_pcalg()
    adjacencies = [_pc_one_subsample_R(idx, X, y_continuous, p, alpha, m_max, skel)
                   for idx in indices]
    return np.sum(adjacencies, axis=0) / B


def dagma_stability_q(X: np.ndarray, y_continuous: np.ndarray,
                      B: int = 100, subsample_fraction: float = 0.5,
                      lambda1: float = 0.02, w_threshold: float = 0.3,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability DAGMA (continuous-optimization discovery; Bello et al.
    2022, the modern NOTEARS); q_j = freq over B runs that x_j is adjacent to y.

    features are standardized per subsample to defuse the varsortability artifact
    (Reisach et al. 2021) that lets continuous-opt methods read off the variance
    order on simulated DAGs. y is appended as the last node."""
    import contextlib
    import io
    from dagma.linear import DagmaLinear
    from sklearn.preprocessing import StandardScaler
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    counts = np.zeros(p)
    for _ in range(B):
        idx = _subsample_indices(n, subsample_fraction, rng)
        data = StandardScaler().fit_transform(
            np.column_stack([X[idx], y_continuous[idx]]))
        try:
            sink = io.StringIO()
            with warnings.catch_warnings(), contextlib.redirect_stderr(sink):
                warnings.simplefilter('ignore')
                W = DagmaLinear(loss_type='l2').fit(
                    data, lambda1=lambda1, w_threshold=w_threshold)
        except Exception:
            continue
        counts += ((np.abs(W[p, :p]) > 0) | (np.abs(W[:p, p]) > 0)).astype(int)
    return counts / B


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


def _init_R_bnlearn() -> None:
    """Lazy load of bnlearn into the R session (reuses the pcalg init for the
    numpy<->R converter and library path). Idempotent."""
    global _BNLEARN_INITIALIZED
    if _BNLEARN_INITIALIZED:
        return
    _init_R_pcalg()
    import rpy2.robjects as ro
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ro.r('suppressPackageStartupMessages(library(bnlearn))')
    _BNLEARN_INITIALIZED = True


def _bnlearn_one_subsample(idx: np.ndarray, X: np.ndarray, y01: np.ndarray,
                           disc_cols: list[int], method: str,
                           max_sx: int) -> np.ndarray:
    """One bnlearn structure-learn on a subsample; binary adjacency of features to y.

    disc_cols are 1-based positions of discrete (factor) feature columns; the
    target y is always a factor. method 'bic-cg' (hc) or 'mi-cg' (pc.stable)."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    p = X.shape[1]
    with localconverter(_R_CONVERTER):
        ro.globalenv['Xmat'] = X[idx]
        ro.globalenv['yvec'] = y01[idx].astype(float)
        ro.globalenv['disc'] = np.asarray(disc_cols, dtype=float)
    ro.r('''
        df <- as.data.frame(Xmat)
        names(df) <- paste0("x", seq_len(ncol(df)) - 1L)
        for (j in as.integer(disc)) df[[j]] <- as.factor(df[[j]])
        df[["y"]] <- as.factor(yvec)
    ''')
    learn = ('hc(df, score="bic-cg")' if method == 'bic-cg'
             else f'pc.stable(df, test="mi-cg", max.sx={max_sx}, undirected=FALSE)')
    ro.r(f'''
        adj_to_y <- tryCatch({{
            g <- {learn}
            a <- arcs(g)
            unique(c(a[a[, 2] == "y", 1], a[a[, 1] == "y", 2]))
        }}, error = function(e) character(0))
    ''')
    with localconverter(_R_CONVERTER):
        adj = list(ro.r('adj_to_y'))
    out = np.zeros(p, dtype=int)
    for nm in adj:
        if isinstance(nm, str) and nm.startswith('x') and nm[1:].isdigit():
            out[int(nm[1:])] = 1
    return out


def bnlearn_stability_q(X: np.ndarray, y: np.ndarray, method: str = 'mi-cg',
                        B: int = 100, subsample_fraction: float = 0.5,
                        max_sx: int = 3,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Subsample-stability q via bnlearn conditional-Gaussian discovery, for mixed
    data (continuous features + categorical target/indicators). q_j = freq over B
    subsamples that feature j is adjacent to the target.

    method='mi-cg' uses pc.stable (constraint-based, CG mutual-information CI test);
    method='bic-cg' uses hc (score-based conditional-linear-Gaussian BIC, the GES
    analog). Binary feature columns (e.g. missingness indicators) are passed as
    factors; the target is always a factor. max_sx caps the PC conditioning-set
    size (mi-cg only), the analog of m.max on the Gaussian path. Runs sequentially
    in one R session (~0.7s/subsample for bic-cg, ~6.5s for mi-cg at n~5k, p~24)."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    y01 = (np.asarray(y) > 0).astype(int)
    # treat columns whose observed values are a subset of {0,1} as factors (the
    # missingness indicators). edge case: a genuinely continuous column that by
    # chance only takes values in {0,1}, or an all-constant column, is also typed
    # as a factor here. harmless on the current data (continuous features are
    # credit scores/ratios, never literally {0,1}); revisit if adding datasets
    # where a continuous feature could be coincidentally binary.
    disc_cols = [j + 1 for j in range(p)
                 if set(np.unique(X[:, j]).tolist()) <= {0.0, 1.0}]
    _init_R_bnlearn()
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    adjacencies = [_bnlearn_one_subsample(idx, X, y01, disc_cols, method, max_sx)
                   for idx in indices]
    return np.sum(adjacencies, axis=0) / B


def bnlearn_mb(X: np.ndarray, y: np.ndarray, method: str = 'iamb',
               test: str = 'mi-cg', alpha: float = 0.05) -> list[int]:
    """Markov blanket of the target via bnlearn's constraint-based MB learner with a
    conditional-Gaussian CI test (mi-cg), i.e. a causal-feature-selection baseline
    (IAMB-style) that uses the *correct* mixed-data test for continuous features and
    a binary target. Returns sorted original-feature indices; single run, not
    stability-aggregated."""
    y01 = (np.asarray(y) > 0).astype(int)
    p = X.shape[1]
    disc_cols = [j + 1 for j in range(p)
                 if set(np.unique(X[:, j]).tolist()) <= {0.0, 1.0}]
    _init_R_bnlearn()
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    with localconverter(_R_CONVERTER):
        ro.globalenv['Xmat'] = X
        ro.globalenv['yvec'] = y01.astype(float)
        ro.globalenv['disc'] = np.asarray(disc_cols, dtype=float)
    ro.r('''
        df <- as.data.frame(Xmat)
        names(df) <- paste0("x", seq_len(ncol(df)) - 1L)
        for (j in as.integer(disc)) df[[j]] <- as.factor(df[[j]])
        df[["y"]] <- as.factor(yvec)
    ''')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ro.r(f'mb <- tryCatch(learn.mb(df, "y", method="{method}", test="{test}", '
             f'alpha={alpha}), error=function(e) character(0))')
    with localconverter(_R_CONVERTER):
        mb = list(ro.r('mb'))
    return sorted(int(nm[1:]) for nm in mb
                  if isinstance(nm, str) and nm.startswith('x') and nm[1:].isdigit())


def _bnlearn_mb_one_subsample(idx: np.ndarray, X: np.ndarray, y01: np.ndarray,
                              disc_cols: list[int], test: str,
                              alpha: float) -> np.ndarray:
    """one bnlearn iamb markov-blanket learn on a subsample; length-p indicator of
    the blanket of y. mirrors _bnlearn_one_subsample but uses learn.mb (iamb) with
    the mixed-data ci test, for the soft iamb prior."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    p = X.shape[1]
    with localconverter(_R_CONVERTER):
        ro.globalenv['Xmat'] = X[idx]
        ro.globalenv['yvec'] = y01[idx].astype(float)
        ro.globalenv['disc'] = np.asarray(disc_cols, dtype=float)
    ro.r('''
        df <- as.data.frame(Xmat)
        names(df) <- paste0("x", seq_len(ncol(df)) - 1L)
        for (j in as.integer(disc)) df[[j]] <- as.factor(df[[j]])
        df[["y"]] <- as.factor(yvec)
    ''')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ro.r(f'mb <- tryCatch(learn.mb(df, "y", method="iamb", test="{test}", '
             f'alpha={alpha}), error=function(e) character(0))')
    with localconverter(_R_CONVERTER):
        mb = list(ro.r('mb'))
    out = np.zeros(p, dtype=int)
    for nm in mb:
        if isinstance(nm, str) and nm.startswith('x') and nm[1:].isdigit():
            out[int(nm[1:])] = 1
    return out


def bnlearn_mb_stability_q(X: np.ndarray, y: np.ndarray, test: str = 'mi-cg',
                           alpha: float = 0.05, B: int = 100,
                           subsample_fraction: float = 0.5,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """subsample-stability q from bnlearn's iamb markov-blanket learner with the
    mixed-data conditional-gaussian ci test (mi-cg). q_j = freq over B subsamples
    that feature j lands in the blanket of y.

    this is the soft iamb prior used by the method's iamb_soft arm: the SAME
    iamb+mi-cg discovery as the cfs_cg hard baseline, used softly, so cfs_cg vs
    iamb_soft is a clean same-source soft-vs-hard contrast on a valid mixed-data
    test. replaces the earlier pycausalfs fisher-z source (invalid on mixed data);
    pycausalfs stays only for the naive hard cfs_iamb/cfs_hiton baselines."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    y01 = (np.asarray(y) > 0).astype(int)
    disc_cols = [j + 1 for j in range(p)
                 if set(np.unique(X[:, j]).tolist()) <= {0.0, 1.0}]
    _init_R_bnlearn()
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    blankets = [_bnlearn_mb_one_subsample(idx, X, y01, disc_cols, test, alpha)
                for idx in indices]
    return np.sum(blankets, axis=0) / B


def _iamb_gauss_one_subsample(idx: np.ndarray, X: np.ndarray, y_cont: np.ndarray,
                              test: str, alpha: float, method: str = 'iamb') -> np.ndarray:
    """one bnlearn markov-blanket learn on a continuous (gaussian) subsample; length-p
    indicator of the blanket of the latent continuous target. method is any bnlearn
    learn.mb algorithm (iamb, fast.iamb, inter.iamb, iamb.fw, gs). all columns numeric,
    fisher's-z ci test, for the synthetic recovery sweep."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter

    p = X.shape[1]
    with localconverter(_R_CONVERTER):
        ro.globalenv['Xmat'] = X[idx]
        ro.globalenv['yvec'] = y_cont[idx].astype(float)
    ro.r('''
        df <- as.data.frame(Xmat)
        names(df) <- paste0("x", seq_len(ncol(df)) - 1L)
        df[["y"]] <- as.numeric(yvec)
    ''')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ro.r(f'mb <- tryCatch(learn.mb(df, "y", method="{method}", test="{test}", '
             f'alpha={alpha}), error=function(e) character(0))')
    with localconverter(_R_CONVERTER):
        mb = list(ro.r('mb'))
    out = np.zeros(p, dtype=int)
    for nm in mb:
        if isinstance(nm, str) and nm.startswith('x') and nm[1:].isdigit():
            out[int(nm[1:])] = 1
    return out


def iamb_stability_q(X: np.ndarray, y_continuous: np.ndarray, test: str = 'zf',
                     alpha: float = 0.1, B: int = 100,
                     subsample_fraction: float = 0.5,
                     rng: np.random.Generator | None = None,
                     method: str = 'iamb') -> np.ndarray:
    """subsample-stability q from bnlearn's iamb markov-blanket learner on continuous
    gaussian data with fisher's-z, for the synthetic recovery sweep. q_j = freq over
    B subsamples that feature j lands in the blanket of the latent continuous target.

    uses y_continuous to match the pc/ges gaussian sources, and alpha defaults to 0.1
    to match the pc source (pc_stability_q), so the constraint-based backends are
    compared on equal footing (the source axis). the real-data method uses the
    mixed-data mi-cg variant instead (bnlearn_mb_stability_q, alpha 0.05)."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    y_cont = np.asarray(y_continuous, dtype=float)
    _init_R_bnlearn()
    indices = [_subsample_indices(n, subsample_fraction, rng) for _ in range(B)]
    blankets = [_iamb_gauss_one_subsample(idx, X, y_cont, test, alpha, method) for idx in indices]
    return np.sum(blankets, axis=0) / B


# full recovered cpdags: whole graph, not just adjacency to y.
# the *_stability_q sources above collapse each run to a length-p adjacency-to-y
# vector (all q needs). for orientation scoring and the consensus-graph picture
# we need the entire (p+1)x(p+1) recovered cpdag. amat convention (pcalg): a
# nonzero amat[i,j] is a mark from i into j; amat[i,j]=amat[j,i]=1 is undirected
# (i--j); amat[i,j]=1, amat[j,i]=0 is the directed edge i->j (arrowhead at j).
# the last node (index p) is the target.

def _cpdag_pc_R(data: np.ndarray, alpha: float, m_max: int) -> np.ndarray:
    """Full PC CPDAG (Fisher-Z) on a stacked [X | y] matrix; returns the amat."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    with localconverter(_R_CONVERTER):
        ro.globalenv['X_sub'] = data
    ro.r('n_sub <- nrow(X_sub); p_sub <- ncol(X_sub)')
    ro.r('suffStat <- list(C = cor(X_sub), n = n_sub)')
    ro.r(f'res <- pc(suffStat, indepTest=gaussCItest, alpha={alpha}, '
         f'p=p_sub, m.max={m_max}, verbose=FALSE)')
    with localconverter(_R_CONVERTER):
        return np.array(ro.r('as(res@graph, "matrix")'))


def _cpdag_ges_R(data: np.ndarray) -> np.ndarray:
    """Full GES CPDAG (Gaussian-BIC) on a stacked [X | y] matrix; returns the amat."""
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    with localconverter(_R_CONVERTER):
        ro.globalenv['X_sub'] = data
    ro.r('score <- new("GaussL0penObsScore", as.matrix(X_sub))')
    ro.r('res <- ges(score, verbose=FALSE)')
    with localconverter(_R_CONVERTER):
        return np.array(ro.r('as(res$essgraph, "matrix")'))


def pc_full_cpdag(X: np.ndarray, y_continuous: np.ndarray,
                  alpha: float = 0.1, m_max: int = 5) -> np.ndarray:
    """Recovered PC CPDAG on the full sample; (p+1)x(p+1) amat, y is the last node."""
    _init_R_pcalg()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _cpdag_pc_R(np.column_stack([X, y_continuous]), alpha, m_max)


def ges_full_cpdag(X: np.ndarray, y_continuous: np.ndarray) -> np.ndarray:
    """Recovered GES CPDAG on the full sample; (p+1)x(p+1) amat, y is the last node."""
    _init_R_pcalg()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return _cpdag_ges_R(np.column_stack([X, y_continuous]))


def dag_to_cpdag_amat(adj_dag: np.ndarray) -> np.ndarray:
    """True DAG (amat[i,j]=1 iff i->j) -> its CPDAG amat via pcalg::dag2cpdag.

    PC/GES can only recover a Markov-equivalence class, so the fair comparison
    target is the *CPDAG* of the true DAG, not the DAG: edges the data cannot
    orient (no v-structure forces them) are left undirected in both, and so are
    not charged as orientation errors.
    """
    _init_R_pcalg()
    import rpy2.robjects as ro
    from rpy2.robjects.conversion import localconverter
    with localconverter(_R_CONVERTER):
        ro.globalenv['amat_dag'] = adj_dag.astype(float)
    ro.r('rownames(amat_dag) <- colnames(amat_dag) <- '
         'as.character(seq_len(nrow(amat_dag)) - 1L)')
    ro.r('amat_cp <- as(dag2cpdag(as(amat_dag, "graphNEL")), "matrix")')
    with localconverter(_R_CONVERTER):
        return np.array(ro.r('amat_cp'))


def edge_stability_matrix(X: np.ndarray, y_continuous: np.ndarray,
                          method: str = 'ges', B: int = 100,
                          subsample_fraction: float = 0.5,
                          alpha: float = 0.1, m_max: int = 5,
                          rng: np.random.Generator | None = None) -> np.ndarray:
    """Per-edge stability: freq[i,j] = fraction of B subsamples whose recovered
    CPDAG carries a mark from i into j. Skeleton stability of pair {i,j} is
    max(freq[i,j], freq[j,i]); orientation confidence is freq[i,j] vs freq[j,i].
    Drives the consensus-CPDAG picture (face validity of the recovered edges)."""
    if rng is None:
        rng = np.random.default_rng()
    n, p = X.shape
    _init_R_pcalg()
    acc = np.zeros((p + 1, p + 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(B):
            idx = _subsample_indices(n, subsample_fraction, rng)
            data = np.column_stack([X[idx], y_continuous[idx]])
            amat = (_cpdag_ges_R(data) if method == 'ges'
                    else _cpdag_pc_R(data, alpha, m_max))
            acc += (amat != 0).astype(float)
    return acc / B


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
