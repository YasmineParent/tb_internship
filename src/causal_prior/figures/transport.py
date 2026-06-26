"""fig2 out-of-environment transport panels (separate titleless files)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.causal_prior import figstyle
from src.causal_prior.figures import figure

SHIFT = figstyle.ROOT / 'results/causal_prior/synthetic/recovery_shift/headline/shift.csv'
FOLK = figstyle.ROOT / 'results/causal_prior/folktables_transport'
PANEL = figstyle.PANEL

# transport gap is reported to two held-out states; distinguish by marker/linestyle.
TARGETS = {'PR': ('o', '-'), 'SD': ('^', '--')}


def _gap_vs_mu(ax, summary_csv):
    df = pd.read_csv(summary_csv)
    causal = df[df['arm'] == 'causal'].sort_values('mu_rel')
    vanilla = df[df['arm'] == 'vanilla'].iloc[0]
    for t, (mk, ls) in TARGETS.items():
        ax.plot(causal['mu_rel'], causal[f'gap_{t}'], marker=mk, ms=4, color='C0', ls=ls,
                label=f'causal $\\to$ {t}')
        ax.axhline(vanilla[f'gap_{t}'], color='0.5', ls=ls, lw=1, label=f'vanilla $\\to$ {t}')
    ax.set_xscale('log')
    ax.set_xlabel(r'prior strength $\mu/\mu_{\mathrm{scale}}$')
    ax.set_ylabel(r'transport gap (AUC$_\mathrm{src}-$AUC$_\mathrm{tgt}$)')
    ax.legend()


@figure('fig2a_synthetic', kind='main')
def fig2a_synthetic():
    d = pd.read_csv(SHIFT)
    d = d[d['test_gamma'] == -1.0].dropna(subset=['correlate_inclusion', 'delta_auc'])
    bins = [-0.01, 0.001, 0.1, 0.2, 0.3, 1.01]
    labels = ['0', '(0,.1]', '(.1,.2]', '(.2,.3]', '>.3']
    binned = pd.cut(d['correlate_inclusion'], bins, labels=labels)
    grp = d.groupby(binned, observed=True)['delta_auc']
    mean, sem = grp.mean(), grp.std() / np.sqrt(grp.count())
    r = np.corrcoef(d['correlate_inclusion'], d['delta_auc'])[0, 1]

    fig, ax = plt.subplots(figsize=PANEL)
    ax.bar(range(len(mean)), mean.values, yerr=sem.values, capsize=3, color='C0', edgecolor='white')
    ax.axhline(0, color='k', lw=0.7)
    ax.set_xticks(range(len(mean)))
    ax.set_xticklabels(mean.index)
    ax.set_xlabel('correlate reliance of fitted support')
    ax.set_ylabel(r'transport gap (AUC$_\mathrm{in}-$AUC$_\mathrm{shift}$)')
    ax.text(0.05, 0.95, f'$r={r:.2f}$', transform=ax.transAxes, va='top')
    return fig


@figure('fig2b_semisynthetic', kind='main')
def fig2b_semisynthetic():
    fig, ax = plt.subplots(figsize=PANEL)
    _gap_vs_mu(ax, FOLK / 'inject' / 'summary.csv')
    return fig


@figure('fig2c_real', kind='main')
def fig2c_real():
    fig, ax = plt.subplots(figsize=PANEL)
    _gap_vs_mu(ax, FOLK / 'plain' / 'summary.csv')
    return fig
