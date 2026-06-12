"""Consensus recovered CPDAG on FICO, for face validity of the orientations.

No ground truth on real data, so this is the qualitative check: render the edges
the discovery recovers stably and see whether the *oriented* ones are sensible
(e.g. a demographic pointing into a financial feature is plausible; an arrow into
a demographic, or feature->target reversed, is a red flag). Edge opacity = how
often the edge survives subsampling; an arrowhead is drawn only when one
direction dominates the other by --orient-margin (else the edge is left
undirected, honestly reflecting that the data did not orient it).

The features are continuous, so Gaussian PC/GES is appropriate for the
feature-subgraph orientations shown here; only the *target* node carries the
binary-Y caveat (use --background to pin it as a sink, or read its edges with
the CG caveat). For deciding q you still use the CG sources in fico_parity; this
script is purely the picture.

Usage:
    python experiments/causal_prior/real/cpdag_plot.py --method ges --sentinel-nan
    python experiments/causal_prior/real/cpdag_plot.py --method pc --skeleton-thresh 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments._io import new_run_dir  # noqa: E402
from src.causal_prior.priors import edge_stability_matrix  # noqa: E402
from experiments.causal_prior.real.fico_parity import (  # noqa: E402
    DATA, TARGET, POS_LABEL, load_features)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=['pc', 'ges'], default='ges')
    p.add_argument('--b', type=int, default=100, help='subsamples for edge stability')
    p.add_argument('--skeleton-thresh', type=float, default=0.5,
                   help='draw an edge when max(freq[i,j],freq[j,i]) exceeds this')
    p.add_argument('--orient-margin', type=float, default=0.15,
                   help='draw an arrowhead only when the two directions differ by '
                        'more than this in stability (else undirected)')
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b = 10
    return args


def consensus_edges(freq, names, skel_thresh, margin):
    """Yield (u, v, weight, directed) per surviving unordered pair: directed True
    means a -> b with u=a; directed False means undirected (u, v unordered)."""
    p = len(names)
    edges = []
    for i in range(p):
        for j in range(i + 1, p):
            fij, fji = freq[i, j], freq[j, i]
            skel = max(fij, fji)
            if skel < skel_thresh:
                continue
            if abs(fij - fji) >= margin:
                u, v = (i, j) if fij >= fji else (j, i)
                edges.append((names[u], names[v], skel, True))
            else:
                edges.append((names[i], names[j], skel, False))
    return edges


def plot_cpdag(edges, names, target_name, method, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(names)
    for u, v, w, directed in edges:
        G.add_edge(u, v, weight=w, directed=directed)

    pos = nx.spring_layout(G, seed=0, k=1.4)
    fig, ax = plt.subplots(figsize=(11, 9))
    node_colors = ['#d98880' if n == target_name else '#aed6f1' for n in names]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=900, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    for u, v, d in G.edges(data=True):
        kw = dict(G=G, pos=pos, edgelist=[(u, v)], ax=ax,
                  width=1.0 + 2.0 * d['weight'],
                  alpha=min(1.0, 0.25 + 0.75 * d['weight']),
                  edge_color='#2c3e50')
        if d['directed']:  # arrow kwargs only apply to FancyArrowPatch edges
            kw.update(arrows=True, arrowstyle='-|>', arrowsize=16,
                      connectionstyle='arc3,rad=0.02')
        else:
            kw.update(arrows=False)
        nx.draw_networkx_edges(**kw)
    ax.set_title(f'FICO consensus CPDAG ({method.upper()}, edge opacity = stability); '
                 f'target = {target_name}')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, args.sentinel_nan)
    node_names = names + [TARGET]
    print(f'FICO: n={X_orig.shape[0]}  features={X_orig.shape[1]}  '
          f'edge-stability {args.method.upper()} B={args.b}...', flush=True)

    freq = edge_stability_matrix(
        X_orig, y.astype(float), method=args.method, B=args.b,
        rng=np.random.default_rng(args.seed))

    edges = consensus_edges(freq, node_names, args.skeleton_thresh, args.orient_margin)
    n_dir = sum(d for *_, d in edges)
    print(f'{len(edges)} edges above skeleton-thresh {args.skeleton_thresh} '
          f'({n_dir} oriented, {len(edges) - n_dir} undirected)', flush=True)

    suffix = f'fico_{args.method}' + ('_sent' if args.sentinel_nan else '')
    out_dir = new_run_dir(
        REPO_ROOT / 'results' / 'causal_prior' / 'cpdag' / suffix, vars(args))
    pd.DataFrame(freq, index=node_names, columns=node_names).to_csv(
        out_dir / 'edge_stability.csv')
    pd.DataFrame(edges, columns=['from', 'to', 'stability', 'directed']).to_csv(
        out_dir / 'consensus_edges.csv', index=False)
    plot_cpdag(edges, node_names, TARGET, args.method, out_dir / 'cpdag.png')

    print(f'Done. Graph + figure in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
