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
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fico')
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


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    node_names = list(names) + ['target']
    print(f'{args.dataset}: n={X_orig.shape[0]}  features={X_orig.shape[1]}  '
          f'edge-stability {args.method.upper()} B={args.b}...', flush=True)

    freq = edge_stability_matrix(
        X_orig, y.astype(float), method=args.method, B=args.b,
        rng=np.random.default_rng(args.seed))

    edges = consensus_edges(freq, node_names, args.skeleton_thresh, args.orient_margin)
    n_dir = sum(d for *_, d in edges)
    print(f'{len(edges)} edges above skeleton-thresh {args.skeleton_thresh} '
          f'({n_dir} oriented, {len(edges) - n_dir} undirected)', flush=True)

    suffix = f'{args.dataset}_{args.method}' + ('_sent' if args.sentinel_nan else '')
    out_dir = new_run_dir(
        REPO_ROOT / 'results' / 'causal_prior' / 'cpdag' / suffix, vars(args))
    pd.DataFrame(freq, index=node_names, columns=node_names).to_csv(
        out_dir / 'edge_stability.csv')
    pd.DataFrame(edges, columns=['from', 'to', 'stability', 'directed']).to_csv(
        out_dir / 'consensus_edges.csv', index=False)

    print(f'Done. Graph + figure in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
