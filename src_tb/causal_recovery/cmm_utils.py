import os
from datetime import datetime
import networkx as nx
import pyagrum as gum
import pyagrum.lib.notebook as gnb
import numpy as np
import pandas as pd
from rpy2.rinterface_lib.embedded import RRuntimeError
from src.exp.algos import CD


def topic_graph_to_bn(topic_graph: nx.DiGraph, node_names: list) -> gum.BayesNet:
    """Convert a TopologicalCausalMixture topic_graph to a pyAgrum BayesNet for visualization."""
    bn = gum.BayesNet()
    for i in topic_graph.nodes:
        bn.add(gum.LabelizedVariable(str(node_names[i]), str(node_names[i]), 2))
    for i, j in topic_graph.edges:
        bn.addArc(str(node_names[i]), str(node_names[j]))
    return bn


def run_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], binary_indices: list[int], noise_seed: int, noise_std: float = 0.05):
    """Run CMM with added noise to binary columns. Returns fitted model."""
    rng = np.random.default_rng(noise_seed)
    X_fit = X.astype(float).copy()
    X_fit[:, binary_indices] += rng.normal(0, noise_std, (X_fit.shape[0], len(binary_indices)))
    cmm = CD.CausalMixtures.get_method()
    cmm.fit(X_fit, forbidden_edges=forbidden_edges)
    return cmm


def bootstrap_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], binary_indices: list[int], n_runs: int, noise_std: float = 0.05, noise_seed: int = 42) -> list:
    """Vary patients (resample with replacement), fix noise. Measures sampling variability."""
    rng = np.random.default_rng(0)
    cmm_list = []
    for _ in range(n_runs):
        while True:
            X_boot = X[rng.choice(len(X), size=len(X), replace=True)]
            try:
                cmm = run_cmm(X_boot, forbidden_edges, binary_indices, noise_seed=noise_seed, noise_std=noise_std)
                cmm_list.append(cmm)
                break
            except RRuntimeError:
                continue
    return cmm_list


def noise_robustness_cmm(X: np.ndarray, forbidden_edges: set[tuple[int, int]], binary_indices: list[int], n_runs: int, noise_std: float = 0.05) -> list:
    """Fix patients, vary noise seed. Measures sensitivity to binary column noise."""
    cmm_list = []
    for seed in range(n_runs):
        cmm = run_cmm(X, forbidden_edges, binary_indices, noise_seed=seed, noise_std=noise_std)
        cmm_list.append(cmm)
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


def build_stable_bn(cmm_list: list, features: list[str], threshold: float = 0.5) -> gum.BayesNet:
    """Build a BayesNet from edges present in more than threshold fraction of runs."""
    stable_edges = get_stable_edges(cmm_list, features, threshold)
    bn = gum.BayesNet()
    for feature in features:
        bn.add(gum.LabelizedVariable(feature, feature, 2))
    for _, row in stable_edges.iterrows():
        bn.addArc(row['source'], row['target'])
    return bn


def visualize_stable_bn(cmm_list: list, features: list[str], threshold: float = 0.5, size: str = "30"):
    """Visualize stable edges as a BN. Edges present in more than threshold fraction of runs are included."""
    gnb.showBN(build_stable_bn(cmm_list, features, threshold), size=size)


def run_and_save(X: np.ndarray, features: list[str], forbidden_edges: set[tuple[int, int]], binary_indices: list[int], noise_seed: int, save: bool = False, output_dir: str = '../results/preliminary', size: str = "60", noise_std: float = 0.05):
    """Run CMM and return BN visualization. Optionally save edges to CSV and graph to bifxml."""
    cmm = run_cmm(X, forbidden_edges, binary_indices, noise_seed=noise_seed, noise_std=noise_std)
    bn = topic_graph_to_bn(cmm.dag, features)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        edges = [(features[i], features[j]) for i, j in cmm.dag.edges()]
        pd.DataFrame(edges, columns=['source', 'target']).to_csv(os.path.join(output_dir, 'edges.csv'), index=False)
        gum.saveBN(bn, os.path.join(output_dir, 'causal_graph.bifxml'))

    gnb.showBN(bn, size=size)
    return cmm


def bootstrap_and_save(X: np.ndarray, features: list[str], forbidden_edges: set[tuple[int, int]], binary_indices: list[int], n_runs: int, threshold: float = 0.5, size: str = "60", noise_std: float = 0.05):
    """Run bootstrap, save edge stability CSVs and stable graph, visualize stable BN."""
    output_dir = f'../results/bootstrap_{datetime.now().strftime("%Y%m%d_%H%M")}'
    cmm_list = bootstrap_cmm(X, forbidden_edges, binary_indices, n_runs=n_runs, noise_std=noise_std)

    os.makedirs(output_dir, exist_ok=True)
    df_stability = edge_stability(cmm_list, features)
    df_stability.to_csv(os.path.join(output_dir, 'edge_stability.csv'), index=False)

    stable = get_stable_edges(cmm_list, features, threshold=threshold)
    stable.to_csv(os.path.join(output_dir, 'stable_edges.csv'), index=False)

    bn = build_stable_bn(cmm_list, features, threshold=threshold)
    gum.saveBN(bn, os.path.join(output_dir, 'stable_graph.bifxml'))
    gnb.showBN(bn, size=size)
    return cmm_list, df_stability
