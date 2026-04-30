import numpy as np
import networkx as nx


def eval_recovery(stable_set: set, true_edges: set) -> tuple[float, float, float]:
    """Compute precision, recall, F1 of recovered stable edges vs true edges."""
    tp = len(stable_set & true_edges)
    fp = len(stable_set - true_edges)
    fn = len(true_edges - stable_set)
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return pr, re, f1


def compute_graph_metrics(recovered: set, true_edges: set, features: list) -> dict:
    """Compute SD, SHD, SC, F1, TPR, FPR using the CMM paper's graph metric library."""
    from src.mixtures.util.util import nxdigraph_to_lmg, compare_lmg_DAG
    name_to_idx = {f: i for i, f in enumerate(features)}
    nodes = list(range(len(features)))

    def _build(edges, label):
        unknown = {n for s, t in edges for n in (s, t) if n not in name_to_idx}
        if unknown:
            raise ValueError(f"{label} edges reference unknown features: {sorted(unknown)}")
        g = nx.DiGraph()
        g.add_nodes_from(nodes)
        for s, t in edges:
            g.add_edge(name_to_idx[s], name_to_idx[t])
        return g

    true_g = _build(true_edges, 'true')
    rec_g  = _build(recovered, 'recovered')
    return compare_lmg_DAG(nxdigraph_to_lmg(true_g), nxdigraph_to_lmg(rec_g))


def evaluate_method(
    method_name: str,
    bootstrap_results: list[set[tuple[str, str]]],
    true_edges: set[tuple[str, str]],
    features: list[str],
    stability_threshold: float = 0.5,
) -> dict:
    """Aggregate metrics across bootstrap runs for a single method."""
    edge_counts = {}
    for run in bootstrap_results:
        for edge in run:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    n_runs = len(bootstrap_results)
    stable_edges = {e for e, c in edge_counts.items() if c / n_runs >= stability_threshold}

    pr, re, f1 = eval_recovery(stable_edges, true_edges)
    graph_metrics = compute_graph_metrics(stable_edges, true_edges, features)
    per_run_f1 = [eval_recovery(run, true_edges)[2] for run in bootstrap_results]

    return {
        'method': method_name,
        'n_bootstrap': n_runs,
        'stability_threshold': stability_threshold,
        'n_stable_edges': len(stable_edges),
        'stable_precision': pr,
        'stable_recall': re,
        'stable_f1': f1,
        'mean_per_run_f1': round(float(np.mean(per_run_f1)), 3),
        'std_per_run_f1':  round(float(np.std(per_run_f1)), 3),
        **graph_metrics,
    }
