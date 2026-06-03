"""Plot functions for the §6.1 figures. Accept tidy DataFrames, plot to axes.

Conventions matching src/causal_discovery/visualization.plot_sweep:
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
    'oracle':       plt.cm.tab10(3),  # red
    'ges':          plt.cm.tab10(2),  # green
    'pc':           plt.cm.tab10(0),  # blue
    'bootstrap_l1': plt.cm.tab10(1),  # orange
    'uniform':      'gray',
    'adversarial':  'black',
}

SOURCE_LINESTYLES: dict[str, str] = {
    'oracle':       '-',
    'ges':          '-',
    'pc':           '-',
    'bootstrap_l1': '-',
    'uniform':      ':',
    'adversarial':  '--',
}

SOURCE_ORDER: list[str] = ['oracle', 'ges', 'pc', 'bootstrap_l1', 'uniform', 'adversarial']

SOURCE_LABELS: dict[str, str] = {
    'oracle':       'oracle',
    'uniform':      'vanilla',
    'adversarial':  'adversarial',
    'pc':           'PC',
    'ges':          'GES',
    'bootstrap_l1': 'bootstrap-$L_1$',
}


def _filter_soft(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the hard-threshold baseline rows (q_source like 'pc_hard_t0.3')."""
    return df[~df['q_source'].str.contains('_hard_', na=False)]


FACET_LABELS: dict[str, str] = {
    'p_edge': r'p_{\mathrm{edge}}',
    'n':      'n',
    'p':      'p',
    'k_star': r'k^{*}',
}


def plot_recovery_vs_mu(
    df: pd.DataFrame,
    metric: str = 'S_precision',
    ax: Axes | None = None,
    title: str | None = None,
    show_vanilla_anchor: bool = True,
) -> Axes:
    """Single-panel recovery vs mu_relative for a pre-filtered cell.

    Caller is responsible for filtering df to one cell (one combination of
    p, n, k_star, p_edge across seeds). One line per q_source, mean across
    seeds with SEM band. If ax is None, creates a figure. If title is None,
    builds one from the unique cell ids in df.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    soft = _filter_soft(df)
    if soft.empty:
        raise ValueError('no rows in df')

    g = soft.groupby(['q_source', 'mu_relative'])[metric].agg(['mean', 'sem']).reset_index()
    # uniform q is mathematically equivalent to vanilla (mu=0); skip the
    # redundant horizontal line when uniform is in the plot, since the
    # uniform curve already shows the vanilla level (and the overlap is a
    # meaningful invariance, not noise).
    uniform_in_plot = 'uniform' in g['q_source'].unique()
    if show_vanilla_anchor and not uniform_in_plot:
        vanilla = g[g['mu_relative'] == 0.0]['mean'].mean()
        ax.axhline(vanilla, color='lightgray', linestyle='-', linewidth=1, zorder=0,
                   label=f'vanilla (μ=0): {vanilla:.2f}')

    sources_present = [s for s in SOURCE_ORDER if s in g['q_source'].unique()]
    for src in sources_present:
        sub = g[g['q_source'] == src].sort_values('mu_relative')
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
    if title is None:
        # default title: list the cell dims that are unique in df
        parts = [f'{col}={df[col].iloc[0]}' for col in ('p', 'n', 'k_star', 'p_edge')
                 if col in df.columns and df[col].nunique() == 1]
        title = ', '.join(parts)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    return ax


def plot_recovery_vs_mu_facet(
    df: pd.DataFrame,
    facet_col: str,
    metric: str = 'S_precision',
    facet_values: list | None = None,
    show_legend_in: int = 0,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
):
    """Multi-panel facet along facet_col (e.g. 'p_edge', 'n', 'p', 'k_star').

    Caller pre-filters df to a single sweep (one column varying, others fixed).
    Creates the figure internally and returns it. facet_values defaults to the
    sorted unique values of facet_col in df. Legend is drawn in panel show_legend_in.
    """
    if facet_col not in df.columns:
        raise ValueError(f'facet_col {facet_col!r} not in df')
    if facet_values is None:
        facet_values = sorted(df[facet_col].unique())
    label = FACET_LABELS.get(facet_col, facet_col)
    nrows = (len(facet_values) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows),
                             sharey=True)
    flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, fv in enumerate(facet_values):
        sub = df[df[facet_col] == fv]
        plot_recovery_vs_mu(sub, metric=metric, ax=flat[i],
                            title=f'${label} = {fv}$')
        if i != show_legend_in:
            leg = flat[i].get_legend()
            if leg is not None:
                leg.remove()
    for j in range(len(facet_values), len(flat)):
        flat[j].set_visible(False)
    flat[show_legend_in].legend(fontsize=8, loc='best')
    fig.tight_layout()
    return fig


def plot_recovery_vs_axis(
    df: pd.DataFrame,
    axis_col: str,
    mu_relative: float = 1.0,
    metric: str = 'S_precision',
    ax: Axes | None = None,
) -> Axes:
    """Single-panel: metric vs axis_col (e.g. 'n', 'p', 'k_star') at fixed mu_relative.

    One line per q_source, mean across seeds with SEM band. The vanilla curve
    (q-source-agnostic baseline at mu=0) is plotted separately in light gray.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    soft = _filter_soft(df)
    if soft.empty:
        raise ValueError('no rows in df')

    mu_vals = sorted(soft['mu_relative'].unique())
    mu_pick = min([m for m in mu_vals if m > 0],
                  key=lambda v: abs(v - mu_relative), default=mu_relative)

    at_mu = soft[soft['mu_relative'] == mu_pick]
    sources_present = [s for s in SOURCE_ORDER if s in at_mu['q_source'].unique()]

    # uniform q at mu>0 is mathematically equivalent to vanilla (mu=0); skip
    # the redundant vanilla curve when uniform is present.
    if 'uniform' not in sources_present:
        vanilla = soft[soft['mu_relative'] == 0.0].groupby(axis_col)[metric].agg(['mean', 'sem']).reset_index()
        ax.plot(vanilla[axis_col], vanilla['mean'], color='lightgray', marker='s',
                markersize=4, label=f'vanilla (μ=0)', zorder=0)
        ax.fill_between(vanilla[axis_col],
                        vanilla['mean'] - vanilla['sem'].fillna(0),
                        vanilla['mean'] + vanilla['sem'].fillna(0),
                        color='lightgray', alpha=0.3, zorder=0)
    for src in sources_present:
        g = at_mu[at_mu['q_source'] == src].groupby(axis_col)[metric].agg(['mean', 'sem']).reset_index()
        if g.empty:
            continue
        color = SOURCE_COLORS[src]
        ls = SOURCE_LINESTYLES[src]
        ax.plot(g[axis_col], g['mean'], color=color, linestyle=ls,
                marker='o', markersize=4, label=SOURCE_LABELS[src])
        ax.fill_between(g[axis_col],
                        g['mean'] - g['sem'].fillna(0),
                        g['mean'] + g['sem'].fillna(0),
                        color=color, alpha=0.15)

    label = FACET_LABELS.get(axis_col, axis_col)
    ax.set_xlabel(f'${label}$')
    ax.set_ylabel(metric.replace('_', ' '))
    ax.set_ylim(0, 1)
    ax.set_title(f'${label}$ sweep ($\\mu_{{\\mathrm{{rel}}}}={mu_pick:.2f}$)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')
    return ax


def plot_mu_star_vs_axis(
    cv: pd.DataFrame,
    axis_col: str,
    K_multiplier: float | None = None,
    ax: Axes | None = None,
) -> Axes:
    """CV-picked mu_star (relative) vs axis_col, one line per q_source.

    Reads CSVs produced by recovery_sweep_cv.py. The §6.4 diagnostic:
    when mu_hat > 0, the prior is doing meaningful work; when mu_hat = 0,
    CV collapsed to vanilla (and the precision should match the
    no-prior baseline). Log-scaled y so the spread across orders of
    magnitude is readable; mu_hat = 0 rows are clipped to the grid floor
    and marked with an open marker to disambiguate.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    df = cv[~cv['q_source'].astype(str).str.contains('_hard_', na=False)]
    if K_multiplier is not None:
        df = df[df['K_multiplier'] == K_multiplier]
    if df.empty:
        raise ValueError('no rows after filter')

    # log-floor; the smallest positive mu_grid point is ~1e-2, so 1e-3 is below
    floor = 1e-3
    df = df.copy()
    df['mu_plot'] = df['mu_star_relative'].clip(lower=floor)
    df['is_zero'] = df['mu_star_relative'] == 0.0

    sources_present = [s for s in SOURCE_ORDER if s in df['q_source'].unique()]
    for src in sources_present:
        sub = df[df['q_source'] == src]
        g = sub.groupby(axis_col).agg(
            mu_mean=('mu_plot', 'mean'),
            mu_sem=('mu_plot', 'sem'),
            zero_frac=('is_zero', 'mean'),
        ).reset_index()
        color = SOURCE_COLORS[src]
        ls = SOURCE_LINESTYLES[src]
        ax.plot(g[axis_col], g['mu_mean'], color=color, linestyle=ls,
                marker='o', markersize=4, label=SOURCE_LABELS[src])
        ax.fill_between(g[axis_col],
                        (g['mu_mean'] - g['mu_sem'].fillna(0)).clip(lower=floor),
                        g['mu_mean'] + g['mu_sem'].fillna(0),
                        color=color, alpha=0.15)
        # open markers where >= 50% of seeds had mu_hat = 0 (CV collapsed to vanilla)
        collapsed = g[g['zero_frac'] >= 0.5]
        if not collapsed.empty:
            ax.scatter(collapsed[axis_col], collapsed['mu_mean'],
                       facecolors='white', edgecolors=color, s=60, zorder=5)

    label = FACET_LABELS.get(axis_col, axis_col)
    ax.set_xlabel(f'${label}$')
    ax.set_ylabel(r'$\hat{\mu}_{\mathrm{rel}}$ (CV)')
    ax.set_yscale('log')
    ax.axhline(floor, color='lightgray', linestyle=':', linewidth=0.8, zorder=0)
    title = f'CV-picked $\\mu$ vs ${label}$'
    if K_multiplier is not None:
        title += f' ($K = {K_multiplier} \\cdot k^*$)'
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=8, loc='best')
    return ax


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


def plot_soft_vs_hard_facet(
    df: pd.DataFrame,
    q_sources: list[str] = ('pc', 'ges', 'bootstrap_l1'),
    metric: str = 'S_precision',
    mu_relative: float = 1.0,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
):
    """Multi-panel soft vs hard, one panel per q_source.

    Sources without hard-threshold rows in df are skipped silently. Returns the
    Figure. Use this for the §6.4 comparison across all causal/predictive q's.
    """
    available = [s for s in q_sources
                 if df['q_source'].str.startswith(f'{s}_hard_t').any()]
    if not available:
        raise ValueError('no hard-threshold rows for any of: '
                         f'{q_sources}')
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per_panel[0] * ncols,
                                      figsize_per_panel[1] * nrows),
                             sharey=True)
    flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for i, src in enumerate(available):
        plot_soft_vs_hard(df, q_source=src, metric=metric,
                          mu_relative=mu_relative, ax=flat[i])
    for j in range(len(available), len(flat)):
        flat[j].set_visible(False)
    fig.tight_layout()
    return fig
