import tempfile
import time
from pathlib import Path

import networkx as nx
import pyagrum as gum
import numpy as np
import pandas as pd
from rpy2.rinterface_lib.embedded import RRuntimeError

import rpy2.robjects as ro

import src  # noqa: F401  extends src.__path__ with external/cmm/src
from src.exp.algos import CD


def _drop_sparse_binary_cols(X: np.ndarray, forbidden_edges: set, is_binary: np.ndarray,
                              min_count: int) -> tuple[np.ndarray, set, list[int]]:
    """Drop binary columns with fewer than min_count positives. Returns (X_fit, remapped_forbidden, kept_indices)."""
    keep = [k for k in range(X.shape[1]) if not is_binary[k] or X[:, k].sum() >= min_count]
    old_to_new = {old: new for new, old in enumerate(keep)}
    remapped = {(old_to_new[s], old_to_new[t]) for s, t in forbidden_edges
                if s in old_to_new and t in old_to_new}
    return X[:, keep], remapped, keep


def run_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], use_logistic: bool = True,
            max_parents: int | None = None, k_max: int = 5, min_cluster_count: int = 5,
            noise_seed: int = 0, noise_std: float = 0.05,
            cont_noise_std: float = 1e-3, max_retries: int = 10):
    """Run CMM on X, retrying with fresh R seeds (and jittered data) on FLXfit failure.

    Binary columns with fewer than min_cluster_count * k_max positives are dropped before
    fitting — they cannot support logistic mixture with k_max components without degenerate
    clusters.

    Retry behaviour:
      - use_logistic=False: small Gaussian noise (std=noise_std) is added to binary cols on
        every attempt, treating them as continuous.
      - use_logistic=True: attempt 0 fits the un-noised data. On retries, tiny Gaussian noise
        (std=cont_noise_std, default 1e-3) is added to continuous cols to break FLXfit
        degeneracies — the full-data analog of subsample_cmm's row-resampling. This is the
        jittering / stochastic-EM convention (Diebolt & Celeux 1993; Bishop 1995),
        unbiased and formally equivalent to small ridge regularization.

    max_parents: optional cap on in-degree per node (None = unconstrained)."""
    is_binary = np.array([np.all(np.isin(X[:, k], [0, 1])) for k in range(X.shape[1])])
    if use_logistic:
        X, forbidden_edges, _ = _drop_sparse_binary_cols(
            X, forbidden_edges, is_binary, min_count=min_cluster_count * k_max)
        # recompute is_binary for the now-smaller X
        is_binary = np.array([np.all(np.isin(X[:, k], [0, 1])) for k in range(X.shape[1])])
    rng = np.random.default_rng(noise_seed)
    binary_cols = [j for j in range(X.shape[1]) if is_binary[j]]
    continuous_cols = [j for j in range(X.shape[1]) if not is_binary[j]]

    for attempt in range(max_retries):
        X_fit = X.astype(float).copy()
        if not use_logistic and binary_cols:
            X_fit[:, binary_cols] += rng.normal(0, noise_std, (X_fit.shape[0], len(binary_cols)))
        if use_logistic and attempt > 0 and continuous_cols:
            X_fit[:, continuous_cols] += rng.normal(0, cont_noise_std,
                                                    (X_fit.shape[0], len(continuous_cols)))
        ro.r(f'set.seed({noise_seed + attempt})')
        try:
            cmm = CD.CausalMixtures.get_method()
            cmm.fit(X_fit, forbidden_edges=forbidden_edges, use_logistic=use_logistic,
                    max_parents=max_parents, k_max=k_max)
            return cmm
        except RRuntimeError:
            continue
    raise RRuntimeError(f'run_cmm: {max_retries} consecutive attempts failed')


def refit_with_stable_graph(X: np.ndarray, stable_edges: list[tuple[int, int]],
                             use_logistic: bool = True, k_max: int = 6, max_parents: int = 4,
                             min_cluster_count: int = 4, seed: int = 0,
                             cont_noise_std: float = 1e-3, max_retries: int = 10) -> tuple:
    """Refit CMM on full X with topic_graph fixed to stable_edges (post-selection inference).

    Skips structure learning by passing oracle_G=True with the stable DAG as truths['true_g'].
    _fit_Z_nodes runs as normal and fills betas, gammas, idls, pprobas. Sparse binary columns
    are dropped per the run_cmm convention; stable edges referencing dropped columns are
    silently filtered out.

    stable_edges: iterable of (src_idx, tgt_idx) int pairs over columns of X.

    Returns (cmm, kept_indices). kept_indices maps new-column-index -> old-column-index."""
    is_binary = np.array([np.all(np.isin(X[:, k], [0, 1])) for k in range(X.shape[1])])
    if use_logistic:
        X_keep, _, keep = _drop_sparse_binary_cols(
            X, set(), is_binary, min_count=min_cluster_count * k_max)
    else:
        X_keep, keep = X, list(range(X.shape[1]))
    old_to_new = {old: new for new, old in enumerate(keep)}
    edges_remapped = [(old_to_new[s], old_to_new[t]) for s, t in stable_edges
                      if s in old_to_new and t in old_to_new]

    g = nx.DiGraph()
    g.add_nodes_from(range(len(keep)))
    g.add_edges_from(edges_remapped)

    rng = np.random.default_rng(seed)
    is_bin_new = np.array([np.all(np.isin(X_keep[:, k], [0, 1])) for k in range(X_keep.shape[1])])
    continuous_cols = [j for j in range(X_keep.shape[1]) if not is_bin_new[j]]

    for attempt in range(max_retries):
        X_fit = X_keep.astype(float).copy()
        if use_logistic and attempt > 0 and continuous_cols:
            X_fit[:, continuous_cols] += rng.normal(0, cont_noise_std,
                                                    (X_fit.shape[0], len(continuous_cols)))
        ro.r(f'set.seed({seed + attempt})')
        try:
            cmm = CD.CausalMixtures.get_method()
            cmm.fit(X_fit, oracle_G=True, truths={'true_g': g.copy()},
                    use_logistic=use_logistic, max_parents=max_parents, k_max=k_max)
            return cmm, keep
        except RRuntimeError:
            continue
    raise RRuntimeError(f'refit_with_stable_graph: {max_retries} consecutive attempts failed')


def subsample_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], n_runs: int,
                  use_logistic: bool = True, max_parents: int | None = None, k_max: int = 5,
                  subsample_frac: float = 0.8, min_cluster_count: int = 5,
                  features: list[str] | None = None,
                  seed: int = 0, max_retries: int = 3) -> tuple[list, list]:
    """Stability selection via subsampling without replacement (Meinshausen-Buhlmann style).

    Each run subsamples subsample_frac of rows without replacement, then drops any binary
    column with fewer than min_cluster_count * k_max positives in that subsample before
    fitting. The product is the threshold below which a logistic mixture with k_max
    components would collapse on a sparse cluster — defaults give 5 * 5 = 25 positives.

    Returns (cmm_list, features_per_run). If features (names) is provided, features_per_run
    is a list of name-lists (one per run). Otherwise a list of index-lists. Pass features_per_run
    to edge_stability for correct per-edge denominators."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    m = max(1, int(round(subsample_frac * n)))
    is_binary = np.array([np.all(np.isin(X[:, k], [0, 1])) for k in range(p)])
    col_threshold = min_cluster_count * k_max

    cmm_list = []
    features_per_run = []
    for run_i in range(n_runs):
        t0 = time.perf_counter()
        for attempt in range(max_retries):
            idx = rng.choice(n, size=m, replace=False)
            X_sub = X[idx]
            X_fit, run_forbidden, keep = _drop_sparse_binary_cols(
                X_sub, forbidden_edges, is_binary, min_count=col_threshold)
            noise_seed = int(rng.integers(2**31))
            try:
                cmm = run_cmm(X_fit, run_forbidden, use_logistic=use_logistic,
                              max_parents=max_parents, k_max=k_max,
                              min_cluster_count=0,  # already filtered above
                              noise_seed=noise_seed)
                cmm_list.append(cmm)
                features_per_run.append([features[k] for k in keep] if features else keep)
                dt = time.perf_counter() - t0
                print(f"run {run_i+1}/{n_runs} done in {dt:.1f}s ({len(keep)} features kept, attempt {attempt+1})", flush=True)
                break
            except RRuntimeError:
                continue
        else:
            raise RuntimeError(f"subsample_cmm: {max_retries} consecutive RRuntimeErrors at run {run_i+1}")
    return cmm_list, features_per_run


def edge_stability(cmm_list: list, features_per_run: list) -> pd.DataFrame:
    """Count how often each named edge appears across runs.

    features_per_run: either a flat list[str] (same features every run, old behaviour —
    frequency = count / n_runs) or a list[list[str]] (per-run feature sets, new behaviour —
    frequency = count / n_eligible where n_eligible is the number of runs in which both
    endpoints were present)."""
    flat = features_per_run and isinstance(features_per_run[0], str)
    if flat:
        per_run = [features_per_run] * len(cmm_list)
    else:
        per_run = features_per_run

    edge_counts = {}
    for cmm, run_feats in zip(cmm_list, per_run):
        for i, j in cmm.dag.edges:
            edge = (run_feats[i], run_feats[j])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    run_sets = [set(f) for f in per_run]
    rows = []
    for (src, tgt), count in edge_counts.items():
        n_eligible = sum(1 for s in run_sets if src in s and tgt in s)
        rows.append({'source': src, 'target': tgt, 'count': count,
                     'n_eligible': n_eligible, 'frequency': count / n_eligible})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['source', 'target', 'count', 'n_eligible', 'frequency'])


def per_node_k_summary(cmm_list: list, features_per_run: list) -> pd.DataFrame:
    """Aggregate the best k chosen per node across stability-selection runs.

    Reads cmm.model.div_measures[i]['k'] (set by mixing.fit_functional_mixture) for each
    node i present in a given run. Returns a DataFrame with one row per feature:
    n_runs_present, k1_count..kK_count, mean_k, median_k, mode_k. Sorted by mean_k desc."""
    flat = features_per_run and isinstance(features_per_run[0], str)
    per_run = [features_per_run] * len(cmm_list) if flat else features_per_run

    feature_ks: dict[str, list[int]] = {}
    for cmm, run_feats in zip(cmm_list, per_run):
        div = getattr(cmm.model, 'div_measures', {})
        for i, feat in enumerate(run_feats):
            entry = div.get(i)
            if entry is not None and 'k' in entry:
                feature_ks.setdefault(feat, []).append(int(entry['k']))

    if not feature_ks:
        return pd.DataFrame(columns=['feature', 'n_runs_present', 'mean_k', 'median_k', 'mode_k'])

    all_ks = sorted({k for ks in feature_ks.values() for k in ks})
    rows = []
    for feat, ks in feature_ks.items():
        row = {'feature': feat, 'n_runs_present': len(ks)}
        for k in all_ks:
            row[f'k{k}_count'] = sum(1 for x in ks if x == k)
        row['mean_k'] = float(np.mean(ks))
        row['median_k'] = float(np.median(ks))
        vals, counts = np.unique(ks, return_counts=True)
        row['mode_k'] = int(vals[np.argmax(counts)])
        rows.append(row)
    return pd.DataFrame(rows).sort_values('mean_k', ascending=False).reset_index(drop=True)


def build_stable_bn(stability: pd.DataFrame | str | Path, threshold: float = 0.5,
                    continuous_features: list[str] | None = None) -> gum.BayesNet:
    """Build a BayesNet from edges with frequency >= threshold.

    stability: edge_stability DataFrame (or path to its CSV) with columns source, target, frequency.
    Nodes in the BN are the union of source/target labels among kept edges. Names in
    continuous_features become RangeVariables; everything else is binary LabelizedVariables."""
    if not isinstance(stability, pd.DataFrame):
        stability = pd.read_csv(stability)
    # add the strongest edges first so that if the thresholded set has a cycle (stability
    # selection is not itself acyclic), the weaker cycle-closing edge is the one dropped.
    stable = stability[stability['frequency'] >= threshold].sort_values('frequency', ascending=False)
    nodes = sorted(set(stable['source']) | set(stable['target']))
    continuous = set(continuous_features or [])
    bn = gum.BayesNet()
    for feature in nodes:
        if feature in continuous:
            bn.add(gum.RangeVariable(feature, feature, 0, 1))
        else:
            bn.add(gum.LabelizedVariable(feature, feature, 2))
    for _, row in stable.iterrows():
        try:
            bn.addArc(row['source'], row['target'])
        except gum.InvalidDirectedCycle:
            pass  # drop the weaker edge that would close a cycle (greedy DAG, highest freq first)
    return bn


# node-type fills for the stable-graph renderer (bio-readable: outcome vs mutation vs background)
_STABLE_BN_COLORS = {'mic': '#e8896b', 'mutation': '#9ec5e0', 'burden': '#f0a35e',
                     'lineage': '#cfcfcf', 'type': '#ddd0a0'}


def _stable_node_class(name: str, continuous: set) -> str:
    if name in continuous:
        return 'mic'
    if name.startswith('burden_'):
        return 'burden'
    if name.startswith('lineage'):
        return 'lineage'
    if name.startswith('type'):
        return 'type'
    return 'mutation'


def visualize_stable_bn(stability: pd.DataFrame | str | Path, threshold: float = 0.5,
                        size: str = "30", continuous_features: list[str] | None = None,
                        into_only: bool = False):
    """Visualize stable edges as a coloured causal diagram. Nodes are filled by type (outcome MIC,
    mutation, rare-variant burden, lineage, resistance type) and each arc is labelled and widened
    by its selection frequency. Rendered to PNG so the output embeds reliably in .ipynb. Needs the
    graphviz `dot` binary on PATH.

    into_only: keep only edges that point into an outcome MIC (continuous_features), dropping the
    mutation-mutation / mutation-lineage co-occurrence clutter, for a clean drivers -> MIC view."""
    import os
    import shutil
    import pyagrum.lib.bn2graph as b2g
    from IPython.display import display, Image
    if shutil.which('dot') is None:  # pick up a homebrew graphviz if it is not already on PATH
        for _p in ('/opt/homebrew/bin', '/usr/local/bin'):
            if (Path(_p) / 'dot').exists():
                os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')
                break
    if not isinstance(stability, pd.DataFrame):
        stability = pd.read_csv(stability)
    continuous = set(continuous_features or [])
    if into_only:
        stability = stability[stability['target'].isin(continuous)]
    bn = build_stable_bn(stability, threshold, continuous_features)
    name2id = {bn.variable(i).name(): i for i in bn.nodes()}
    stable = stability[stability['frequency'] >= threshold]
    arc_label, arc_width = {}, {}
    for s, t, f in zip(stable['source'], stable['target'], stable['frequency']):
        a = (name2id.get(s), name2id.get(t))
        if a[0] is not None and a[1] is not None and bn.dag().existsArc(*a):
            arc_label[a] = f'{f:.2f}'
            arc_width[a] = 1 + 4 * f
    g = b2g.BN2dot(bn, size=size, arcLabel=arc_label, arcWidth=arc_width)
    for node in g.get_nodes():
        nm = node.get_name().strip('"')
        if nm not in name2id:
            continue
        node.set_style('filled')
        node.set_shape('box')
        node.set_fillcolor(_STABLE_BN_COLORS[_stable_node_class(nm, continuous)])
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        g.write_png(path)
        with open(path, 'rb') as fp:
            data = fp.read()
    finally:
        Path(path).unlink(missing_ok=True)
    display(Image(data=data, format='png'))


# notebook helpers for the real-TB stability-selection results: keep notebook cells to one-liners
SUBS = Path(__file__).resolve().parents[2] / 'results' / 'mixed_cmm' / 'subsampling'


def show_graph(rel: str, mics: list[str], threshold: float = 0.3, size: str = '13',
               into_only: bool = True):
    """Render the stable causal graph for results subdir `rel` (mics = continuous outcome nodes).
    into_only (default True) keeps only the drivers -> MIC edges, dropping co-occurrence clutter;
    pass into_only=False for the complete-structure view."""
    visualize_stable_bn(pd.read_csv(SUBS / rel / 'edge_stability.csv'),
                        threshold=threshold, continuous_features=mics, size=size, into_only=into_only)


def parents_of(rel: str, mic: str) -> pd.DataFrame:
    """Selection frequency of each feature as a parent of `mic`, descending."""
    e = pd.read_csv(SUBS / rel / 'edge_stability.csv')
    return (e[e['target'] == mic][['source', 'frequency']]
            .sort_values('frequency', ascending=False).round(2).reset_index(drop=True))


def parents_across(rels: dict[str, str], mic: str) -> pd.DataFrame:
    """Parent frequencies for `mic` across several result dirs ({label: subdir}), one column each,
    sorted by the last column. Use for the ablation progression and old-vs-corrected tables."""
    cols = {}
    for label, rel in rels.items():
        e = pd.read_csv(SUBS / rel / 'edge_stability.csv')
        cols[label] = e[e['target'] == mic].set_index('source')['frequency']
    out = pd.DataFrame(cols).fillna(0)
    return out.sort_values(out.columns[-1], ascending=False).round(2)


def plot_parents_across(rels: dict[str, str], mic: str, threshold: float = 0.5, min_show: float = 0.1):
    """Grouped-bar comparison of each feature's selection frequency into `mic` across result dirs."""
    import numpy as np
    import matplotlib.pyplot as plt
    df = parents_across(rels, mic)
    df = df[df.max(axis=1) >= min_show]
    conds = list(df.columns)
    x = np.arange(len(df))
    w = 0.8 / max(len(conds), 1)
    _, ax = plt.subplots(figsize=(max(7, 0.9 * len(df)), 4))
    for i, c in enumerate(conds):
        ax.bar(x + i * w, df[c].to_numpy(), w, label=c)
    ax.axhline(threshold, ls='--', color='grey', lw=0.8, label=f'threshold {threshold}')
    ax.set_xticks(x + (len(conds) - 1) / 2 * w)
    ax.set_xticklabels(df.index, rotation=45, ha='right')
    ax.set_ylabel(f'selection frequency into {mic}')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_parents_panels(rels: dict[str, str], mic: str, top: int = 7, threshold: float = 0.5,
                        min_show: float = 0.1, ncols: int | None = None):
    """Small-multiple `features -> mic` graphs, one panel per condition in `rels` (dict order kept).
    Every panel shows the same feature set (union of the `top` strongest) so edge widths, which are
    proportional to selection frequency, read as an evolution across the progressive-adjustment
    conditions. Dark blue = at or above `threshold`, light blue = seen (>= min_show), grey = near 0."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    df = parents_across(rels, mic)
    df = df[df.max(axis=1) >= min_show]
    df = df.reindex(df.max(axis=1).sort_values(ascending=False).index).head(top)
    feats, conds = list(df.index), list(df.columns)
    rows = len(feats)
    ypos = {f: rows - 1 - i for i, f in enumerate(feats)}
    ymid = (rows - 1) / 2
    n = len(conds); ncols = ncols or n; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 0.62 * rows * nrows + 1.1),
                             sharey=True)
    axes = np.atleast_1d(axes).ravel()
    arr_x0, mic_x = 1.6, 3.05
    for ax, cond in zip(axes, conds):
        ax.set_title(cond, fontsize=10, fontweight='bold')
        ax.scatter([mic_x], [ymid], s=620, color='#f4a3a3', edgecolors='black',
                   linewidths=0.6, zorder=3)
        ax.text(mic_x, ymid, mic.replace('_mic', ''), ha='center', va='center', fontsize=7.5, zorder=4)
        for f in feats:
            fr = float(df.loc[f, cond])
            if fr >= threshold:
                col, alpha, tcol = '#2C5F9E', 1.0, 'black'
            elif fr >= min_show:
                col, alpha, tcol = '#7FA6D6', 0.95, 'black'
            else:
                col, alpha, tcol = '#c9c9c9', 0.5, '#9a9a9a'
            ax.text(1.5, ypos[f], f, ha='right', va='center', fontsize=7, color=tcol)
            ax.add_patch(FancyArrowPatch((arr_x0, ypos[f]), (mic_x - 0.22, ymid), arrowstyle='-|>',
                         mutation_scale=9, lw=0.5 + 6.5 * fr, color=col, alpha=alpha, zorder=2,
                         shrinkA=0, shrinkB=0))
            ax.text(arr_x0 + 0.06, ypos[f] + 0.16, f'{fr:.2f}', fontsize=6.5, color=col, ha='left',
                    va='bottom', fontweight='bold' if fr >= threshold else 'normal')
        ax.set_xlim(-1.4, 3.5); ax.set_ylim(-0.8, rows - 0.2)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(f'parents of {mic}: edge width proportional to selection frequency '
                 f'(dark blue >= {threshold})', fontsize=9)
    plt.tight_layout()
    plt.show()
