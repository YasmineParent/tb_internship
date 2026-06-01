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
    ax: Axes,
    p_edge: float,
    metric: str = 'S_precision',
    show_vanilla_anchor: bool = True,
) -> Axes:
    """Single-panel recovery vs mu_relative at one p_edge.

    One line per q_source, mean across seeds with SEM band. Adversarial,
    uniform, oracle styled distinctly. The mu_relative = 0 point is the
    vanilla anchor (same value across sources) and is drawn as a horizontal
    dashed line if show_vanilla_anchor.
    """
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
    axes,
    metric: str = 'S_precision',
    p_edges: list[float] | None = None,
    show_legend_in: int = 0,
) -> None:
    """Multi-panel facet: one panel per p_edge value.

    axes: array-like of matplotlib Axes (flat order matches p_edges).
    p_edges: defaults to sorted unique p_edge values in df.
    show_legend_in: which panel index gets the legend (default 0).
    """
    if p_edges is None:
        p_edges = sorted(df['p_edge'].unique())
    if len(axes) < len(p_edges):
        raise ValueError(f'need >= {len(p_edges)} axes, got {len(axes)}')
    for i, pe in enumerate(p_edges):
        ax = axes[i]
        plot_recovery_vs_mu(df, ax, p_edge=pe, metric=metric)
        if i != show_legend_in:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    axes[show_legend_in].legend(fontsize=8, loc='best')


def plot_recovery_vs_selectivity(
    merged: pd.DataFrame,
    ax: Axes,
    mu_relative_ref: float,
    metric: str = 'S_precision',
    color_by: str = 'p_edge',
) -> Axes:
    """Scatter: x = sel(q), y = recovery at fixed mu_relative.

    merged must contain columns sel_q, q_source, p_edge, mu_relative, <metric>.
    One point per (q_source, p_edge, seed) at mu_relative_ref. Colored by p_edge
    by default; switch to color_by='q_source' to color by source family.
    """
    soft = _filter_soft(merged)
    # find the mu_relative grid value closest to the target
    mu_vals = sorted(soft['mu_relative'].unique())
    mu_pick = min(mu_vals, key=lambda v: abs(v - mu_relative_ref))
    sub = soft[soft['mu_relative'] == mu_pick].copy()
    # finite sel only (drop NaN and inf)
    sub = sub[np.isfinite(sub['sel_q'])]

    if color_by == 'p_edge':
        norm = plt.Normalize(sub['p_edge'].min(), sub['p_edge'].max())
        cmap = plt.cm.viridis
        for src in sub['q_source'].unique():
            s_sub = sub[sub['q_source'] == src]
            ax.scatter(s_sub['sel_q'], s_sub[metric],
                       c=cmap(norm(s_sub['p_edge'])),
                       edgecolor='black', linewidth=0.4, s=30,
                       marker={'oracle':'*','uniform':'s','adversarial':'X',
                               'pc':'o','ges':'^','bootstrap_l1':'D'}.get(src, 'o'),
                       label=SOURCE_LABELS.get(src, src))
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=r'$p_{\mathrm{edge}}$')
    elif color_by == 'q_source':
        for src in sub['q_source'].unique():
            s_sub = sub[sub['q_source'] == src]
            ax.scatter(s_sub['sel_q'], s_sub[metric],
                       color=SOURCE_COLORS.get(src, 'k'),
                       edgecolor='black', linewidth=0.4, s=30,
                       label=SOURCE_LABELS.get(src, src))
    else:
        raise ValueError(f'color_by must be p_edge or q_source, got {color_by}')

    ax.set_xscale('log')
    ax.set_xlabel(r'selectivity ratio $\bar q_C / \bar q_{S^*}$')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_title(f'recovery vs selectivity at $\\mu_{{\\mathrm{{rel}}}} \\approx {mu_pick:.2f}$',
                 fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    return ax


def plot_soft_vs_hard(
    df: pd.DataFrame,
    ax: Axes,
    q_source: str,
    metric: str = 'S_precision',
    mu_relative: float = 1.0,
    oracle_mu_per_seed: bool = False,
) -> Axes:
    """Soft prior (at chosen mu_relative) vs hard pre-selection (threshold sweep).

    Lines: soft prior at the given mu_relative (default 1.0, a fixed point in
    the headroom region) and hard at each threshold in the data. Both as a
    function of p_edge.

    oracle_mu_per_seed=True instead picks the per-(p_edge, seed) argmax mu on
    the same data; this is an oracle upper bound on soft-prior performance and
    only fair as a sensitivity check, not a baseline comparison.
    """
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
