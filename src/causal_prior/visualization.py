"""Plot functions for the §6.1 figures. Accept tidy DataFrames, plot to axes.

Conventions matching src_tb/causal_discovery/visualization.plot_sweep:
- tab10 color palette
- mean line + SEM band on each curve
- y-axis [0, 1] for recovery metrics
- per-source line style: causal/predictive solid, oracle dashed-black,
  uniform dashed-gray, adversarial dashed-red

Caller controls figure size and layout via plt.subplots; we draw onto axes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


# Source palette: tab10 for the discovery + L1 sources; black/gray/red for the
# synthetic-q sources so the eye separates "real" from "constructed".
SOURCE_COLORS: dict[str, str] = {
    'oracle':       'black',
    'uniform':      'gray',
    'adversarial':  'crimson',
    'pc':           plt.cm.tab10(0),
    'ges':          plt.cm.tab10(2),
    'bootstrap_l1': plt.cm.tab10(1),
}

SOURCE_LINESTYLES: dict[str, str] = {
    'oracle':       '--',
    'uniform':      ':',
    'adversarial':  '--',
    'pc':           '-',
    'ges':          '-',
    'bootstrap_l1': '-',
}

SOURCE_ORDER: list[str] = ['oracle', 'ges', 'pc', 'bootstrap_l1', 'uniform', 'adversarial']

SOURCE_LABELS: dict[str, str] = {
    'oracle':       'oracle',
    'uniform':      'uniform',
    'adversarial':  'adversarial',
    'pc':           'PC',
    'ges':          'GES',
    'bootstrap_l1': 'bootstrap-$L_1$',
}


def _filter_soft(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the hard-threshold baseline rows (q_source like 'pc_hard_t0.3')."""
    return df[~df['q_source'].str.contains('_hard_', na=False)]


def plot_recovery_vs_mu(
    df: pd.DataFrame,
    p_edge: float,
    metric: str = 'S_precision',
    ax: Axes | None = None,
    show_vanilla_anchor: bool = True,
) -> Axes:
    """Single-panel recovery vs mu_relative at one p_edge.

    One line per q_source, mean across seeds with SEM band. Adversarial,
    uniform, oracle styled distinctly. The mu_relative = 0 point is the
    vanilla anchor (same value across sources) and is drawn as a horizontal
    dashed line if show_vanilla_anchor. If ax is None, creates a figure.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    soft = _filter_soft(df[df['p_edge'] == p_edge])
    if soft.empty:
        raise ValueError(f'no rows at p_edge={p_edge}')

    # one mean per (q_source, mu_relative); SEM across seeds
    g = soft.groupby(['q_source', 'mu_relative'])[metric].agg(['mean', 'sem']).reset_index()
    # vanilla anchor = the shared mu=0 value across sources
    if show_vanilla_anchor:
        vanilla = g[g['mu_relative'] == 0.0]['mean'].mean()
        ax.axhline(vanilla, color='lightgray', linestyle='-', linewidth=1, zorder=0,
                   label=f'vanilla (μ=0): {vanilla:.2f}')

    sources_present = [s for s in SOURCE_ORDER if s in g['q_source'].unique()]
    for src in sources_present:
        sub = g[g['q_source'] == src].sort_values('mu_relative')
        # drop mu=0 from the curve (it's the anchor, shared across sources)
        sub = sub[sub['mu_relative'] > 0]
        if sub.empty:
            continue
        x = sub['mu_relative'].values
        m = sub['mean'].values
        s = sub['sem'].fillna(0).values
        color = SOURCE_COLORS[src]
        ls = SOURCE_LINESTYLES[src]
        ax.plot(x, m, color=color, linestyle=ls, marker='o', markersize=3,
                label=SOURCE_LABELS[src])
        ax.fill_between(x, m - s, m + s, color=color, alpha=0.15)

    ax.set_xscale('log')
    ax.set_xlabel(r'$\mu_{\mathrm{relative}}$')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_ylim(0, 1)
    ax.set_title(f'$p_{{\\mathrm{{edge}}}} = {p_edge}$', fontsize=10)
    ax.grid(True, alpha=0.3)
    return ax


def plot_recovery_vs_mu_facet(
    df: pd.DataFrame,
    metric: str = 'S_precision',
    p_edges: list[float] | None = None,
    show_legend_in: int = 0,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
):
    """Multi-panel facet: one panel per p_edge value, one row of figures.

    Creates the figure internally and returns it. p_edges defaults to the
    sorted unique p_edge values in df. The legend is drawn in the panel at
    index show_legend_in.
    """
    if p_edges is None:
        p_edges = sorted(df['p_edge'].unique())
    nrows = (len(p_edges) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows),
                             sharey=True)
    flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, pe in enumerate(p_edges):
        plot_recovery_vs_mu(df, p_edge=pe, metric=metric, ax=flat[i])
        if i != show_legend_in:
            leg = flat[i].get_legend()
            if leg is not None:
                leg.remove()
    # hide unused panels
    for j in range(len(p_edges), len(flat)):
        flat[j].set_visible(False)
    flat[show_legend_in].legend(fontsize=8, loc='best')
    fig.tight_layout()
    return fig


def plot_soft_vs_hard(
    df: pd.DataFrame,
    q_source: str,
    metric: str = 'S_precision',
    mu_relative: float = 1.0,
    oracle_mu_per_seed: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Soft prior (at chosen mu_relative) vs hard pre-selection (threshold sweep).

    Lines: soft prior at the given mu_relative (default 1.0, a fixed point in
    the headroom region) and hard at each threshold in the data. Both as a
    function of p_edge.

    oracle_mu_per_seed=True instead picks the per-(p_edge, seed) argmax mu on
    the same data; this is an oracle upper bound on soft-prior performance and
    only fair as a sensitivity check, not a baseline comparison.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    soft = _filter_soft(df[df['q_source'] == q_source])
    hard = df[df['q_source'].str.startswith(f'{q_source}_hard_t')]
    if hard.empty:
        raise ValueError(f'no hard-threshold rows for {q_source}')

    if oracle_mu_per_seed:
        soft_best = soft.loc[soft.groupby(['p_edge', 'seed'])[metric].idxmax()]
        soft_label = f'{SOURCE_LABELS[q_source]} soft (oracle best $\\mu$ per seed)'
    else:
        mu_vals = sorted(soft['mu_relative'].unique())
        mu_pick = min(mu_vals, key=lambda v: abs(v - mu_relative))
        soft_best = soft[soft['mu_relative'] == mu_pick]
        soft_label = f'{SOURCE_LABELS[q_source]} soft ($\\mu_{{\\mathrm{{rel}}}}={mu_pick:.2f}$)'
    soft_curve = soft_best.groupby('p_edge')[metric].agg(['mean', 'sem']).reset_index()
    ax.plot(soft_curve['p_edge'], soft_curve['mean'],
            marker='o', color=SOURCE_COLORS[q_source], linewidth=2,
            label=soft_label)
    ax.fill_between(soft_curve['p_edge'],
                    soft_curve['mean'] - soft_curve['sem'].fillna(0),
                    soft_curve['mean'] + soft_curve['sem'].fillna(0),
                    color=SOURCE_COLORS[q_source], alpha=0.15)

    # hard: one curve per threshold
    threshold_styles = {'0.3': ':', '0.5': '--', '0.7': '-.'}
    for src in sorted(hard['q_source'].unique()):
        t_str = src.split('_t')[-1]
        sub = hard[hard['q_source'] == src].groupby('p_edge')[metric].agg(['mean', 'sem']).reset_index()
        ax.plot(sub['p_edge'], sub['mean'],
                linestyle=threshold_styles.get(t_str, '--'),
                color=SOURCE_COLORS[q_source], alpha=0.7,
                marker='x', markersize=6,
                label=f'hard $t={t_str}$')

    ax.set_xlabel(r'$p_{\mathrm{edge}}$')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_ylim(0, 1)
    ax.set_title(f'{SOURCE_LABELS[q_source]}: soft prior vs hard pre-selection', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    return ax
