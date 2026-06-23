"""Figures for the environment-shift experiment (recovery_shift.py output).

Fig 1 (mechanism): out-of-environment transport gap vs how much the fitted
support relies on non-causal correlates. Pooled over all sources/seeds/p_edge.
Fig 2 (provenance): per-source in-distribution AUC parity vs out-of-environment
transport gap, showing causal discovery (ges) routes to the low-reliance regime.

Usage:
    python experiments/causal_prior/synthetic/recovery_shift_figs.py --csv <shift_sweep.csv>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = ROOT / 'results/causal_prior/synthetic/recovery_shift/headline/shift.csv'

BINS = [-0.01, 0.001, 0.1, 0.2, 0.3, 1.01]
BINLAB = ['0', '(0,.1]', '(.1,.2]', '(.2,.3]', '>.3']
OP_MU = {'vanilla': 0.0}   # vanilla operates at mu=0; all others at the strong prior
TOP_MU = 2.0
SRC_ORDER = ['oracle', 'ges', 'iamb', 'gs', 'pc', 'vanilla', 'bootstrap_l1', 'adversarial']


def load(csv=DEFAULT_CSV):
    """read a recovery_shift output csv (defaults to the headline run)."""
    return pd.read_csv(csv)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', type=Path, default=DEFAULT_CSV,
                   help='recovery_shift.py output CSV to plot')
    p.add_argument('--out-dir', type=Path, default=None,
                   help='where to write the PNGs (default: alongside --csv)')
    return p.parse_args()


def fig_mechanism(df, path):
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for k, g in enumerate((0.0, -1.0)):
        d = df[df.test_gamma == g].dropna(subset=['correlate_inclusion', 'delta_auc']).copy()
        d['bin'] = pd.cut(d['correlate_inclusion'], BINS, labels=BINLAB)
        grp = d.groupby('bin', observed=True)['delta_auc']
        mean, sem = grp.mean(), grp.std() / np.sqrt(grp.count())
        r = np.corrcoef(d['correlate_inclusion'], d['delta_auc'])[0, 1]
        ax[k].bar(range(len(mean)), mean.values, yerr=sem.values, capsize=3,
                  color='#c44', alpha=0.85)
        ax[k].axhline(0, color='k', lw=0.6)
        ax[k].set_xticks(range(len(mean)))
        ax[k].set_xticklabels(mean.index, rotation=0)
        ax[k].set_xlabel('correlate_inclusion of the fitted support')
        ax[k].set_ylabel('transport gap  (AUC$_{in}$ - AUC$_{shift}$)')
        title = 'correlates -> noise' if g == 0.0 else 'correlates reversed'
        ax[k].set_title(f'$\\gamma$={g:g} ({title})   Pearson $r$={r:.2f}')
    fig.suptitle('Transport failure is governed by correlate reliance (pooled, 60 cells)')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f'wrote {path}', flush=True)


def fig_provenance(df, path):
    rows = []
    for src in SRC_ORDER:
        mu = OP_MU.get(src, TOP_MU)
        d = df[(df.q_source == src) & (df.mu_rel == mu)]
        ind = d[d.test_gamma == 1.0]
        gn = d[d.test_gamma == -1.0]
        rows.append((src, ind.auc.mean(), ind.correlate_inclusion.mean(),
                     gn.delta_auc.mean(), gn.delta_auc.std() / np.sqrt(len(gn))))
    t = pd.DataFrame(rows, columns=['src', 'auc_ind', 'corr_incl', 'gap', 'gap_sem'])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(t))
    ax[0].bar(x, t['auc_ind'], color='#48a', alpha=0.85)
    ax[0].set_ylim(0.95, 1.0)
    ax[0].set_xticks(x); ax[0].set_xticklabels(t['src'], rotation=30, ha='right')
    ax[0].set_ylabel('in-distribution AUC')
    ax[0].set_title(f'i.i.d. parity (spread {t.auc_ind.max()-t.auc_ind.min():.3f})')

    ax[1].bar(x, t['gap'], yerr=t['gap_sem'], capsize=3, color='#c44', alpha=0.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(t['src'], rotation=30, ha='right')
    ax[1].set_ylabel('transport gap @ $\\gamma$=-1')
    ax[1].set_title(f'out-of-environment (spread {t.gap.max()-t.gap.min():.3f})')
    for i, ci in enumerate(t['corr_incl']):
        ax[1].annotate(f'c={ci:.2f}', (i, t['gap'].iloc[i]), ha='center',
                       va='bottom', fontsize=8)
    fig.suptitle('Same models, indistinguishable in-distribution, split out-of-environment')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    print(f'wrote {path}', flush=True)


def main():
    args = parse_args()
    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    fig_mechanism(df, out_dir / 'fig_mechanism.png')
    fig_provenance(df, out_dir / 'fig_provenance.png')
    print(f'Done. Figures in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
