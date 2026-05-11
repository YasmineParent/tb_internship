import tempfile
import time
from pathlib import Path

import pyagrum as gum
import numpy as np
import pandas as pd
from rpy2.rinterface_lib.embedded import RRuntimeError

import rpy2.robjects as ro

import src_tb  # ensures external/cmm is on sys.path
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
            noise_seed: int = 0, noise_std: float = 0.05, max_retries: int = 3):
    """Run CMM on X, retrying with different R seeds on FLXfit failure.

    Binary columns with fewer than min_cluster_count * k_max positives are dropped before
    fitting — they cannot support logistic mixture with k_max components without degenerate
    clusters. use_logistic=False adds Gaussian noise to binary cols instead.
    max_parents: optional cap on in-degree per node (None = unconstrained)."""
    is_binary = np.array([np.all(np.isin(X[:, k], [0, 1])) for k in range(X.shape[1])])
    if use_logistic:
        X, forbidden_edges, _ = _drop_sparse_binary_cols(
            X, forbidden_edges, is_binary, min_count=min_cluster_count * k_max)
    X_fit = X.astype(float).copy()
    if not use_logistic:
        rng = np.random.default_rng(noise_seed)
        binary_cols = [j for j in range(X_fit.shape[1]) if is_binary[j]]
        if binary_cols:
            X_fit[:, binary_cols] += rng.normal(0, noise_std, (X_fit.shape[0], len(binary_cols)))
    for attempt in range(max_retries):
        ro.r(f'set.seed({noise_seed + attempt})')
        try:
            cmm = CD.CausalMixtures.get_method()
            cmm.fit(X_fit, forbidden_edges=forbidden_edges, use_logistic=use_logistic,
                    max_parents=max_parents, k_max=k_max)
            return cmm
        except RRuntimeError:
            continue
    raise RRuntimeError(f'run_cmm: {max_retries} consecutive R seed attempts failed')


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
    stable = stability[stability['frequency'] >= threshold]
    nodes = sorted(set(stable['source']) | set(stable['target']))
    continuous = set(continuous_features or [])
    bn = gum.BayesNet()
    for feature in nodes:
        if feature in continuous:
            bn.add(gum.RangeVariable(feature, feature, 0, 1))
        else:
            bn.add(gum.LabelizedVariable(feature, feature, 2))
    for _, row in stable.iterrows():
        bn.addArc(row['source'], row['target'])
    return bn


def visualize_stable_bn(stability: pd.DataFrame | str | Path, threshold: float = 0.5,
                        size: str = "30", continuous_features: list[str] | None = None):
    """Visualize stable edges as a BN. Rendered to PNG so the output embeds
    in .ipynb in a format that renders reliably on GitHub (gnb.showBN's SVG
    output is flaky in GitHub's notebook renderer)."""
    import pyagrum.lib.image as gimg
    from IPython.display import display, Image
    bn = build_stable_bn(stability, threshold, continuous_features)
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        path = f.name
    try:
        gimg.export(bn, path, size=size)
        with open(path, 'rb') as fp:
            data = fp.read()
    finally:
        Path(path).unlink(missing_ok=True)
    display(Image(data=data, format='png'))
