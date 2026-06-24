"""recovery figures from the fixed-mu grid sweep (recovery_p30_headline, 20 seeds).

no cv: on balanced synthetic data the predictive cv criteria (log_loss, auc) are
mu-flat, so a cv-picked mu is arbitrary w.r.t. recovery. instead we characterise
recovery as a function of mu directly. three panels:

  1. mechanism (anchor cell): S_precision vs mu, all sources. causal rises and
     plateaus; predictive declines below the floor; uniform flat (do-no-harm);
     adversarial collapses; oracle is the ceiling.
  2. across-regime: S_precision vs {n, p, p_edge, k_star} at a single fixed mu_rel
     above the do-no-harm threshold. shows where the prior helps.
  3. mu-robustness: recovery at the fixed mu vs the per-regime oracle mu
     (post-hoc argmax over the grid). they nearly coincide, so precise mu tuning
     buys little - the fixed choice is not cherry-picked.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.causal_prior.loading import load_recovery_csvs   # noqa: E402

GRID_DIR = ROOT / 'results/causal_prior/synthetic/recovery_p30_headline'
OUT = ROOT / 'results/causal_prior/synthetic/recovery_figs'

FIXED_MU = 1.0   # operating mu_rel, above the do-no-harm threshold, in the flat band

# source styling: floor grey, causal blues, predictive red, controls
STYLE = {
    'uniform':      ('vanilla (uniform $q$)', '0.5', '-'),
    'iamb':         ('IAMB (MB, deployed)',   'C0',  '-'),
    'ges':          ('GES (global causal)',   'C9',  '-'),
    'bootstrap_l1': ('bootstrap-$L_1$ (predictive)', 'C3', '-'),
    'oracle':       ('oracle (ceiling)',      'k',   '--'),
    'adversarial':  ('adversarial (control)', 'C1',  ':'),
}
AXES = [('n', 'p == 30 and k_star == 5 and p_edge == 0.2'),
        ('p', 'n == 300 and k_star == 5 and p_edge == 0.2'),
        ('p_edge', 'p == 30 and n == 300 and k_star == 5'),
        ('k_star', 'p == 30 and n == 300 and p_edge == 0.2')]
ANCHOR = 'p == 30 and n == 300 and k_star == 5 and p_edge == 0.2'


def _nearest_mu(df, target):
    """grid mu_relative closest to target (1.0 is an exact grid point)."""
    mus = np.array(sorted(df['mu_relative'].unique()))
    return float(mus[np.argmin(np.abs(mus - target))])


def panel_mechanism(df, ax):
    """S_precision vs mu_relative at the anchor cell, every source."""
    d = df.query(ANCHOR)
    for src, (label, color, ls) in STYLE.items():
        g = d[d.q_source == src].groupby('mu_relative')['S_precision'].mean()
        g = g[g.index > 0]  # drop mu=0 (log-x); it equals the uniform floor anyway
        ax.plot(g.index, g.values, marker='o', ms=3, color=color, ls=ls, label=label)
    ax.set_xscale('log')
    ax.axvline(FIXED_MU, color='0.7', lw=0.8, ls=':')
    ax.set_xlabel('prior strength  $\\mu/\\mu_{\\mathrm{scale}}$')
    ax.set_ylabel('support recovery (S_precision)')
    ax.set_title('(1) mechanism: recovery vs $\\mu$ (anchor cell)', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.85)


def panel_regime(df, axes_row):
    """S_precision vs each axis at the fixed operating mu, causal vs predictive vs floor."""
    mu = _nearest_mu(df, FIXED_MU)
    at_mu = df[np.isclose(df['mu_relative'], mu)]
    floor = df[np.isclose(df['mu_relative'], 0.0)]
    for ax, (col, slc) in zip(axes_row, AXES):
        for src in ['iamb', 'ges', 'bootstrap_l1']:
            label, color, ls = STYLE[src]
            g = at_mu.query(slc)
            g = g[g.q_source == src].groupby(col)['S_precision'].mean()
            ax.plot(g.index, g.values, marker='o', ms=4, color=color, ls=ls,
                    label=label if ax is axes_row[0] else None)
        # vanilla floor (uniform at mu=0)
        fl = floor.query(slc)
        fl = fl[fl.q_source == 'uniform'].groupby(col)['S_precision'].mean()
        ax.plot(fl.index, fl.values, marker='s', ms=3, color='0.5', ls='--',
                label='vanilla floor' if ax is axes_row[0] else None)
        ax.set_xlabel(col)
        if col in ('n', 'p'):
            ax.set_xscale('log')
    axes_row[0].set_ylabel(f'S_precision @ $\\mu_{{\\mathrm{{rel}}}}$={mu:g}')
    axes_row[0].legend(fontsize=7, framealpha=0.85)


def panel_robustness(df, ax):
    """fixed-mu vs per-regime oracle-mu recovery across p_edge, for the deployed source."""
    mu = _nearest_mu(df, FIXED_MU)
    d = df.query(AXES[2][1])  # p_edge slice
    for src, marker in [('iamb', 'o'), ('ges', '^')]:
        label, color, _ = STYLE[src]
        s = d[d.q_source == src]
        # per p_edge: mean S_precision per mu (over seeds), oracle = max over mu, fixed = at mu
        per_mu = s.groupby(['p_edge', 'mu_relative'])['S_precision'].mean().reset_index()
        oracle = per_mu.groupby('p_edge')['S_precision'].max()
        fixed = (per_mu[np.isclose(per_mu['mu_relative'], mu)]
                 .set_index('p_edge')['S_precision'])
        ax.plot(oracle.index, oracle.values, marker=marker, ms=4, color=color, ls='--',
                alpha=0.6, label=f'{label} — oracle $\\mu$')
        ax.plot(fixed.index, fixed.values, marker=marker, ms=4, color=color, ls='-',
                label=f'{label} — fixed $\\mu$={mu:g}')
    ax.set_xlabel('p_edge (confounding density)')
    ax.set_ylabel('support recovery (S_precision)')
    ax.set_title('(3) fixed $\\mu$ $\\approx$ oracle $\\mu$: tuning buys little', fontsize=10)
    ax.legend(fontsize=7, framealpha=0.85)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_recovery_csvs(GRID_DIR)
    df = df[df['noise_scale'] == 1.0]

    # composite: mechanism (1) + robustness (3) side by side
    fig, (a1, a3) = plt.subplots(1, 2, figsize=(13, 4.5))
    panel_mechanism(df, a1)
    panel_robustness(df, a3)
    fig.tight_layout()
    fig.savefig(OUT / 'recovery_mechanism_and_robustness.png', dpi=130, bbox_inches='tight')

    # across-regime row (2)
    fig2, axs2 = plt.subplots(1, len(AXES), figsize=(4 * len(AXES), 4))
    panel_regime(df, axs2)
    fig2.suptitle(f'(2) support recovery across regimes at fixed $\\mu_{{\\mathrm{{rel}}}}$={FIXED_MU:g}',
                  fontsize=12)
    fig2.tight_layout()
    fig2.savefig(OUT / 'recovery_vs_regime.png', dpi=130, bbox_inches='tight')

    print('sources:', sorted(df['q_source'].unique()))
    print(f'fixed mu_rel = {_nearest_mu(df, FIXED_MU)}')
    print(f'Done. Figures in {OUT}/')


if __name__ == '__main__':
    main()
