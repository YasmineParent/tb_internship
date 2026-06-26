"""c1 theory-validation panels (separate titleless files, aggregated in latex)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.causal_prior import figstyle
from src.causal_prior.figures import figure, latest_run

SYN = figstyle.ROOT / 'results/causal_prior/synthetic'
PANEL = figstyle.PANEL


@figure('c1a_radii', kind='appendix')
def c1a_radii():
    df = pd.read_csv(latest_run(SYN / 'exact_radii', 'exact_radii_p12_') / 'theorem1_radii.csv')
    ratio = 2 * df['eps_star'] / df['Delta']          # eps*/eta*, eta* = Delta/2
    c = float(np.median(ratio * df['mu_rel']))
    fig, ax = plt.subplots(figsize=PANEL)
    ax.loglog(df['mu_rel'], c / df['mu_rel'], '--', color='0.5', lw=1, label=r'$\propto 1/\mu$')
    ax.loglog(df['mu_rel'], ratio, 'o-', color='C0', ms=3, label=r'exact $\varepsilon^\star/\eta^\star$')
    ax.set_xlabel(r'prior strength $\mu/\mu_{\mathrm{scale}}$')
    ax.set_ylabel(r'radius ratio $\varepsilon^\star/\eta^\star$')
    ax.legend()
    return fig


@figure('c1b_flip', kind='appendix')
def c1b_flip():
    df = pd.read_csv(latest_run(SYN / 'integer_radii', 'int_p12_k3_C5_seeds') / 'flip_sweep.csv')
    g = df.groupby('eps_rel')['flipped'].mean().sort_index()
    fig, ax = plt.subplots(figsize=PANEL)
    ax.plot(g.index, 100 * g.values, '-', color='C0')
    ax.axvline(1.0, color='0.5', ls='--', lw=1, label=r'$\varepsilon^\star$')
    ax.set_xlabel(r'perturbation $\varepsilon/\varepsilon^\star$')
    ax.set_ylabel('integer MAP flipped (% cases)')
    ax.legend()
    return fig


@figure('c1c_pool', kind='appendix')
def c1c_pool():
    s = pd.read_csv(latest_run(SYN / 'beam_gap', 'beam_gap_p30_') / 'summary.csv')
    s = s[s['parent_size'] == s['parent_size'].max()]
    bars = [('vanilla', 'vanilla', '0.5'), ('ges@2.0', 'GES', 'C9'), ('oracle@2.0', 'oracle', '0.2')]
    fig, ax = plt.subplots(figsize=PANEL)
    for i, (name, lab, colour) in enumerate(bars):
        v = float(s.loc[s['q_name'] == name, 'in_pool_pct'].iloc[0])
        ax.bar(i, v, color=colour)
        ax.text(i, v + 1.5, f'{v:.0f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[1] for b in bars])
    ax.set_ylim(0, 108)
    ax.set_ylabel('exact MAP in beam pool (% cells)')
    return fig
