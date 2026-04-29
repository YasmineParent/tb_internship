import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def eval_recovery(stable_set: set, true_edges: set) -> tuple[float, float, float]:
    """Compute precision, recall, F1 of recovered stable edges vs true edges."""
    tp = len(stable_set & true_edges)
    fp = len(stable_set - true_edges)
    fn = len(true_edges - stable_set)
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return round(pr, 3), round(re, 3), round(f1, 3)


def compute_graph_metrics(recovered: set, true_edges: set, features: list) -> dict:
    """Compute SD, SHD, SC, F1, TPR, FPR using the CMM paper's graph metric library."""
    from src.mixtures.util.util import nxdigraph_to_lmg, compare_lmg_DAG
    name_to_idx = {f: i for i, f in enumerate(features)}
    nodes = list(range(len(features)))

    true_g = nx.DiGraph()
    true_g.add_nodes_from(nodes)
    for s, t in true_edges:
        true_g.add_edge(name_to_idx[s], name_to_idx[t])

    rec_g = nx.DiGraph()
    rec_g.add_nodes_from(nodes)
    for s, t in recovered:
        if s in name_to_idx and t in name_to_idx:
            rec_g.add_edge(name_to_idx[s], name_to_idx[t])

    return compare_lmg_DAG(nxdigraph_to_lmg(true_g), nxdigraph_to_lmg(rec_g))


def plot_comparison(records_old: list[dict], records_new: list[dict], metrics: list[str], labels: list[str],
                    label_old: str = 'Gaussian (noise)', label_new: str = 'Logistic (FLXMRglm)'):
    """Grouped bar chart comparing any set of metrics between two models."""
    df_old = pd.DataFrame(records_old)
    df_new = pd.DataFrame(records_new)

    means_old = [df_old[m].mean() for m in metrics]
    means_new = [df_new[m].mean() for m in metrics]
    stds_old  = [df_old[m].std()  for m in metrics]
    stds_new  = [df_new[m].std()  for m in metrics]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(2 + 2 * len(labels), 4))
    ax.bar(x - width / 2, means_old, width, yerr=stds_old, label=label_old, capsize=4, color='steelblue')
    ax.bar(x + width / 2, means_new, width, yerr=stds_new, label=label_new, capsize=4, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.show()
