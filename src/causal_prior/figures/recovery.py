"""fig3 recovery panels (separate titleless files, aggregated in latex).

shared source colours come from a standalone legend strip (fig3_legend), placed
once below the assembled group; the panels carry no source legend.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from src.causal_prior import figstyle
from src.causal_prior.figures import figure
from src.causal_prior.loading import load_recovery_csvs

SYN = figstyle.ROOT / 'results/causal_prior/synthetic'
GRID = SYN / 'recovery_p30_headline'
K_GRID = SYN / 'recovery_p30_K_ablation'

FIXED_MU = 1.0
ANCHOR = 'p == 30 and n == 300 and k_star == 5 and p_edge == 0.2'
AXES = [('n', 'p == 30 and k_star == 5 and p_edge == 0.2'),
        ('p', 'n == 300 and k_star == 5 and p_edge == 0.2'),
        ('p_edge', 'p == 30 and n == 300 and k_star == 5'),
        ('k_star', 'p == 30 and n == 300 and p_edge == 0.2')]
SRC_MECH = ['uniform', 'iamb', 'ges', 'bootstrap_l1', 'adversarial']
PANEL = figstyle.PANEL


@lru_cache(maxsize=None)
def _grid():
    df = load_recovery_csvs(GRID, add_causal=False)
    return df[df['noise_scale'] == 1.0]


@lru_cache(maxsize=None)
def _kgrid():
    df = load_recovery_csvs(K_GRID, add_causal=False)
    return df[df['noise_scale'] == 1.0]


def _nearest_mu(df, target):
    mus = np.array(sorted(df['mu_relative'].unique()))
    return float(mus[np.argmin(np.abs(mus - target))])


def _line(ax, x, y, src, ms=3, **kw):
    _, colour, ls = figstyle.SOURCES[src]
    ax.plot(x, y, marker='o', ms=ms, color=colour, ls=ls, **kw)


@figure('fig3a_mechanism', kind='main')
def fig3a_mechanism():
    d = _grid().query(ANCHOR)
    fig, ax = plt.subplots(figsize=PANEL)
    for src in SRC_MECH:
        g = d[d.q_source == src].groupby('mu_relative')['S_precision'].mean()
        g = g[g.index > 0]
        _line(ax, g.index, g.values, src)
    ax.set_xscale('log')
    ax.axvline(FIXED_MU, color='0.7', lw=0.8, ls=':')
    ax.set_xlabel(r'prior strength $\mu/\mu_{\mathrm{scale}}$')
    ax.set_ylabel(r'support recovery $S_\mathrm{prec}$')
    return fig


@figure('fig3b_robustness', kind='main')
def fig3b_robustness():
    df = _grid()
    mu = _nearest_mu(df, FIXED_MU)
    d = df.query(AXES[2][1])
    fig, ax = plt.subplots(figsize=PANEL)
    for src, marker in [('iamb', 'o'), ('ges', '^')]:
        _, colour, _ = figstyle.SOURCES[src]
        per_mu = (d[d.q_source == src].groupby(['p_edge', 'mu_relative'])['S_precision']
                  .mean().reset_index())
        best = per_mu.groupby('p_edge')['S_precision'].max()
        fixed = per_mu[np.isclose(per_mu['mu_relative'], mu)].set_index('p_edge')['S_precision']
        ax.plot(best.index, best.values, marker=marker, ms=4, color=colour, ls='--', alpha=0.6)
        ax.plot(fixed.index, fixed.values, marker=marker, ms=4, color=colour, ls='-')
    ax.set_xlabel('p_edge (confounding density)')
    ax.set_ylabel(r'support recovery $S_\mathrm{prec}$')
    ax.legend(handles=[Line2D([], [], color='0.4', ls='-', label=rf'fixed $\mu$={mu:g}'),
                       Line2D([], [], color='0.4', ls='--', alpha=0.6, label=r'best-case $\mu$')],
              loc='lower left')
    return fig


@figure('fig3c_kbudget', kind='main')
def fig3c_kbudget():
    df_k = _kgrid()
    mu = _nearest_mu(df_k, FIXED_MU)
    at = df_k[np.isclose(df_k['mu_relative'], mu)]
    fig, ax = plt.subplots(figsize=PANEL)
    for src in ['iamb', 'ges', 'bootstrap_l1', 'uniform']:
        g = at[at.q_source == src].groupby('K_multiplier')['S_precision'].mean()
        _line(ax, g.index, g.values, src, ms=4)
    ax.set_xlabel(r'sparsity budget $K/k^\star$')
    ax.set_ylabel(rf'$S_\mathrm{{prec}}$ @ $\mu_\mathrm{{rel}}$={mu:g}')
    return fig


@figure('fig3d_regime', kind='main')
def fig3d_regime():
    df = _grid()
    mu = _nearest_mu(df, FIXED_MU)
    at_mu = df[np.isclose(df['mu_relative'], mu)]
    floor = df[np.isclose(df['mu_relative'], 0.0)]
    fig, axes = plt.subplots(1, 4, figsize=(figstyle.TEXT_W, 2.4), sharey=True)
    for ax, (col, slc) in zip(axes, AXES):
        for src in ['iamb', 'ges', 'bootstrap_l1']:
            g = at_mu.query(slc)
            g = g[g.q_source == src].groupby(col)['S_precision'].mean()
            _line(ax, g.index, g.values, src, ms=4)
        fl = floor.query(slc)
        fl = fl[fl.q_source == 'uniform'].groupby(col)['S_precision'].mean()
        _line(ax, fl.index, fl.values, 'uniform', ms=4)
        ax.set_xlabel(col)
        if col == 'n':
            ax.set_xscale('log')
        elif col == 'p':
            ax.set_xticks(sorted(at_mu.query(slc)[col].unique()))
    axes[0].set_ylabel(rf'$S_\mathrm{{prec}}$ @ $\mu_\mathrm{{rel}}$={mu:g}')
    return fig


@figure('fig3_legend', kind='main')
def fig3_legend():
    handles = [Line2D([], [], color=figstyle.SOURCES[s][1], ls=figstyle.SOURCES[s][2],
                      marker='o', ms=3, label=figstyle.SOURCES[s][0]) for s in SRC_MECH]
    fig = plt.figure(figsize=(figstyle.TEXT_W, 0.35))
    fig.legend(handles=handles, loc='center', ncol=len(handles))
    return fig
