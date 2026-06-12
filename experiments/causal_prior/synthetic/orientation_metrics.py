"""Orientation quality of PC vs GES on the §6.1 synthetic, with ground truth.

Answers Nataliya's question directly and with numbers, not a vibe: for each cell
we know the true DAG, so we convert it to its CPDAG (the most any observational
method can recover) and score the recovered PC/GES CPDAG against it, on the full
sample, per seed. Reports SHD, arrowhead precision/recall/F1 (did we orient the
edges right) and adjacency precision/recall/F1 (did we get the skeleton right),
aggregated mean+-std over seeds.

Reading: adjacency_recall says how much of the true skeleton is found;
arrowhead_precision says, of the edges we *did* orient, how many point the right
way. Edges the data cannot orient are undirected in the true CPDAG too, so they
are never charged here.

Usage:
    python experiments/causal_prior/synthetic/orientation_metrics.py            # p_edge sweep
    python experiments/causal_prior/synthetic/orientation_metrics.py --sweep n
    python experiments/causal_prior/synthetic/orientation_metrics.py --methods ges --n-seeds 5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.causal_prior.synthetic.config import SWEEPS, build_cells  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from src.data.synthetic_lingauss import LinGaussSyntheticData  # noqa: E402
from src.causal_prior.graph_metrics import all_scores  # noqa: E402
from src.causal_prior.priors import (  # noqa: E402
    pc_full_cpdag, ges_full_cpdag, dag_to_cpdag_amat)


def true_cpdag_amat(data: LinGaussSyntheticData) -> np.ndarray:
    """CPDAG amat of the generator's true DAG (nodes 0..p, y is the last)."""
    p = data.X.shape[1]
    dag_amat = nx.to_numpy_array(data.dag, nodelist=list(range(p + 1)))
    return dag_to_cpdag_amat(dag_amat)


def recover(method: str, data: LinGaussSyntheticData) -> np.ndarray:
    if method == 'pc':
        return pc_full_cpdag(data.X, data.y_continuous)
    if method == 'ges':
        return ges_full_cpdag(data.X, data.y_continuous)
    raise ValueError(f'unknown method {method!r}')


def plot_orientation(df: pd.DataFrame, axis: str, methods: list[str],
                     out_png: Path) -> None:
    """Two panels vs the swept axis: skeleton recovery (adjacency_recall) and
    orientation quality (arrowhead_f1), one line per method, error bands = std
    over seeds. Reads the same story as the q result: GES recovers and orients
    where PC collapses on dense graphs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    panels = [('adjacency_recall', 'skeleton recovery (adjacency recall)'),
              ('arrowhead_f1', 'orientation quality (arrowhead F1)')]
    colors = {'pc': 'C0', 'ges': 'C1'}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for ax, (metric, title) in zip(axes, panels):
        for method in methods:
            g = (df[df['method'] == method].groupby(axis)[metric]
                 .agg(['mean', 'std']).sort_index())
            std = g['std'].fillna(0.0).to_numpy()
            ax.plot(g.index, g['mean'], marker='o', color=colors.get(method),
                    label=method.upper())
            ax.fill_between(g.index, g['mean'] - std, g['mean'] + std,
                            color=colors.get(method), alpha=0.18)
        ax.set_xlabel(axis)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--sweep', choices=list(SWEEPS) + ['all'], default='p_edge')
    p.add_argument('--methods', default='pc,ges')
    p.add_argument('--n-seeds', type=int, default=None,
                   help='override per-sweep seed count (faster smoke runs)')
    args = p.parse_args()
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    cells = build_cells(args.sweep, n_seeds=args.n_seeds)
    print(f'sweep={args.sweep}, methods={methods}, {len(cells)} cells', flush=True)

    records = []
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        data = LinGaussSyntheticData(
            p=cell.p, n_samples=cell.n, k_star=cell.k_star,
            p_edge=cell.p_edge, noise_scale=cell.noise_scale, seed=cell.seed)
        true_cp = true_cpdag_amat(data)
        for method in methods:
            est_cp = recover(method, data)
            sc = all_scores(true_cp, est_cp)
            records.append({'method': method, 'p_edge': cell.p_edge, 'p': cell.p,
                            'n': cell.n, 'k_star': cell.k_star, 'seed': cell.seed,
                            **sc})
        if i % 10 == 0 or i == len(cells):
            print(f'  [{i}/{len(cells)}] {time.time() - t0:.0f}s', flush=True)

    df = pd.DataFrame(records)
    metrics = ['shd', 'adjacency_recall', 'adjacency_precision',
               'arrowhead_recall', 'arrowhead_precision', 'arrowhead_f1']
    axis = 'p_edge' if args.sweep in ('p_edge', 'all') else args.sweep
    summary = (df.groupby(['method', axis])[metrics]
               .agg(['mean', 'std']).round(3))

    config = {**vars(args), 'methods': methods, 'sweep': args.sweep}
    out_dir = new_run_dir(
        REPO_ROOT / 'results' / 'causal_prior' / 'synthetic' / 'orientation'
        / f'orientation_{args.sweep}', config)
    df.to_csv(out_dir / 'per_cell.csv', index=False)
    summary.to_csv(out_dir / 'summary.csv')
    plot_orientation(df, axis, methods, out_dir / 'orientation_curves.png')

    print(summary.to_string(), flush=True)
    print(f'Done. Results in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
