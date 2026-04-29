import os
from datetime import datetime
from pathlib import Path

import pyagrum as gum
import pyagrum.lib.notebook as gnb
import numpy as np
import pandas as pd
from rpy2.rinterface_lib.embedded import RRuntimeError

import src_tb  # ensures external/cmm is on sys.path
from src.exp.algos import CD

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], use_logistic: bool = True, noise_seed: int = 0, noise_std: float = 0.05):
    """Run CMM on X.
    use_logistic=True: FLXMRglm(family='binomial') used for binary targets, no noise needed.
    use_logistic=False: Gaussian driver throughout; binary columns auto-detected and noise added to prevent variance collapse."""
    X_fit = X.astype(float).copy()
    if not use_logistic:
        rng = np.random.default_rng(noise_seed)
        binary_cols = [j for j in range(X_fit.shape[1]) if np.all(np.isin(X[:, j], [0, 1]))]
        if binary_cols:
            X_fit[:, binary_cols] += rng.normal(0, noise_std, (X_fit.shape[0], len(binary_cols)))
    cmm = CD.CausalMixtures.get_method()
    cmm.fit(X_fit, forbidden_edges=forbidden_edges, use_logistic=use_logistic)
    return cmm


def bootstrap_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], n_runs: int,
                  use_logistic: bool = True, seed: int = 0, max_retries: int = 20) -> list:
    """Vary patients (resample with replacement). Measures sampling variability."""
    rng = np.random.default_rng(seed)
    cmm_list = []
    for _ in range(n_runs):
        for _ in range(max_retries):
            X_boot = X[rng.choice(len(X), size=len(X), replace=True)]
            try:
                cmm = run_cmm(X_boot, forbidden_edges, use_logistic=use_logistic)
                cmm_list.append(cmm)
                break
            except RRuntimeError:
                continue
        else:
            raise RuntimeError(f"bootstrap_cmm: {max_retries} consecutive RRuntimeErrors")
    return cmm_list


def edge_stability(cmm_list: list, features: list[str]) -> pd.DataFrame:
    """Count how often each named edge appears across runs. Returns DataFrame with source, target, count, frequency."""
    edge_counts = {}
    for cmm in cmm_list:
        for i, j in cmm.dag.edges:
            edge = (features[i], features[j])
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    df_edges = pd.DataFrame([(src, tgt, count) for (src, tgt), count in edge_counts.items()], columns=['source', 'target', 'count'])
    df_edges['frequency'] = df_edges['count'] / len(cmm_list)
    return df_edges


def get_stable_edges(cmm_list: list, features: list[str], threshold: float = 0.5) -> pd.DataFrame:
    """Return edges present in more than threshold fraction of runs."""
    df = edge_stability(cmm_list, features)
    return df[df['frequency'] > threshold].sort_values('frequency', ascending=False).reset_index(drop=True)


def build_stable_bn(cmm_list: list, features: list[str], threshold: float = 0.5, continuous_features: list[str] = None) -> gum.BayesNet:
    """Build a BayesNet from edges present in more than threshold fraction of runs."""
    stable_edges = get_stable_edges(cmm_list, features, threshold)
    feature_set = set(features)
    unknown = {e for row in stable_edges.itertuples() for e in (row.source, row.target) if e not in feature_set}
    if unknown:
        raise ValueError(f"stable edges reference unknown features: {sorted(unknown)}")
    continuous_features = set(continuous_features or [])
    bn = gum.BayesNet()
    for feature in features:
        if feature in continuous_features:
            bn.add(gum.RangeVariable(feature, feature, 0, 1))
        else:
            bn.add(gum.LabelizedVariable(feature, feature, 2))
    for _, row in stable_edges.iterrows():
        bn.addArc(row['source'], row['target'])
    return bn


def visualize_stable_bn(cmm_list: list, features: list[str], threshold: float = 0.5, size: str = "30", continuous_features: list[str] = None):
    """Visualize stable edges as a BN."""
    gnb.showBN(build_stable_bn(cmm_list, features, threshold, continuous_features=continuous_features), size=size)


def save_bootstrap_results(cmm_list: list, features: list[str], threshold: float = 0.5, size: str = "60", continuous_features: list[str] = None) -> tuple[str, pd.DataFrame]:
    """Save edge stability CSVs and stable graph to a timestamped results folder. Visualizes stable BN."""
    output_dir = str(REPO_ROOT / 'results' / f'bootstrap_{datetime.now().strftime("%Y%m%d_%H%M")}')
    os.makedirs(output_dir, exist_ok=True)

    df_stability = edge_stability(cmm_list, features)
    df_stability.to_csv(os.path.join(output_dir, 'edge_stability.csv'), index=False)

    stable = get_stable_edges(cmm_list, features, threshold=threshold)
    stable.to_csv(os.path.join(output_dir, 'stable_edges.csv'), index=False)

    bn = build_stable_bn(cmm_list, features, threshold=threshold, continuous_features=continuous_features)
    gum.saveBN(bn, os.path.join(output_dir, 'stable_graph.bifxml'))
    gnb.showBN(bn, size=size)

    return output_dir, df_stability
