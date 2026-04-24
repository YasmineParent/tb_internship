import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from src_tb.data.synthetic import SyntheticData


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
    """Compute SD, SHD, SC, F1, TPR, MCC using the CMM paper's graph metric library."""
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


def eval_all(stable_set: set, data: SyntheticData) -> dict:
    """Compute all graph recovery metrics broken down by edge type."""
    pr_d,  re_d,  f1_d  = eval_recovery(stable_set, data.true_direct)
    pr_bb, re_bb, f1_bb = eval_recovery(stable_set, data.true_bin_to_bin)
    pr_cc, re_cc, f1_cc = eval_recovery(stable_set, data.true_chain_cont)

    fp_independent = len({(s, t) for s, t in stable_set
                          if s in data.independent_features or t in data.independent_features})

    graph = compute_graph_metrics(stable_set, data.true_edges, data.features)

    return {
        'direct_P': pr_d,    'direct_R': re_d,    'direct_F1': f1_d,
        'bin_bin_P': pr_bb,  'bin_bin_R': re_bb,  'bin_bin_F1': f1_bb,
        'chain_P': pr_cc,    'chain_R': re_cc,    'chain_F1': f1_cc,
        'fp_independent': fp_independent,
        'sd':  round(graph.get('sd',  float('nan')), 3),
        'shd': round(graph.get('shd', float('nan')), 3),
        'sc':  round(graph.get('sc',  float('nan')), 3),
        'f1':  round(graph.get('f1',  float('nan')), 3),
        'tpr': round(graph.get('tpr', float('nan')), 3),
        'mcc': round(graph.get('mcc', float('nan')), 3),
    }


def summary_table(records: list[dict]) -> pd.DataFrame:
    """Mean ± std across seeds for all numeric metrics."""
    df = pd.DataFrame(records).select_dtypes(include='number')
    table = pd.DataFrame({'mean': df.mean().round(3), 'std': df.std().round(3)})
    table['mean ± std'] = table.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)
    return table[['mean ± std']]


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
