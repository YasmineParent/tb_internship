"""Fig 1: integer scorecard comparison (a) and rashomon pool cloud (b) on heart."""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.binarize import fit_binarizer, apply_binarizer   # noqa: E402
from src.causal_prior.priors import discover_q                          # noqa: E402
from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid             # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk                # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset         # noqa: E402

POOL_CSV = ROOT / 'results/causal_prior/rashomon/heart_k10/pool.csv'
OUT_DEFAULT = ROOT / 'results' / 'causal_prior' / 'real' / 'fig_scorecard.png'

ARM_COLOR = {'vanilla': '#888', 'causal': '#2171b5'}
SWAP_COLOR = '#fcae91'  # highlight for features in one card but not the other


def fit_cards(k=10, b=100, n_mu=12, n_cv=5, n_thresholds=4, seed=0):
    """return top-1 card dict (rows, intercept, auc) for vanilla and causal on heart."""
    class _A:
        sentinel_nan = False

    X_orig, names, y = load_dataset('heart', _A())
    names = np.asarray(names)
    rest, disc_idx = train_test_split(np.arange(len(y)), test_size=0.3,
                                      stratify=(y > 0), random_state=seed)
    tr_idx, te_idx = train_test_split(rest, test_size=0.3,
                                      stratify=(y[rest] > 0), random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q('ges_cg', X_orig[disc_idx], y[disc_idx].astype(float), b, seed)
    spec, bin_names_raw, parent = fit_binarizer(X_orig[tr_idx], names, n_thresholds)
    bin_names = [str(b) for b in bin_names_raw]  # ensure plain str, not numpy str_
    q_bin = q[parent]
    Xtr = apply_binarizer(X_orig[tr_idx], spec)
    Xte = apply_binarizer(X_orig[te_idx], spec)
    ytr, yte = y[tr_idx], (y[te_idx] > 0).astype(int)
    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, n_mu)
    FR = import_fasterrisk()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = FR(k=k, mu=0.0, freq=None)
        van.fit(Xtr, ytr)
        mu_star = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin,
                             n_splits=n_cv, criterion='log_loss',
                             rng=np.random.default_rng(seed)).mu_star
        cau = FR(k=k, mu=float(mu_star), freq=q_bin.astype(float))
        cau.fit(Xtr, ytr)

    def top_card(fr):
        b0 = float(fr.beta0_[0])
        betas = np.asarray(fr.betas_[0])
        nz = np.nonzero(betas)[0]
        p = np.clip(fr.predict_proba(Xte), 1e-7, 1 - 1e-7)
        auc = float(roc_auc_score(yte, p))
        rows = [{'feature': bin_names[j], 'points': int(betas[j]), 'q': float(q_bin[j])}
                for j in sorted(nz, key=lambda i: abs(betas[i]), reverse=True)]
        return {'rows': rows, 'intercept': b0, 'auc': auc}

    return top_card(van), top_card(cau)


def _table_ax(ax, card, title, swap_feats, arm):
    """render scorecard as a matplotlib table; highlight swapped features."""
    rows = card['rows']
    feat_col = [r['feature'] for r in rows]
    pts_col = [f"{r['points']:+d}" for r in rows]
    q_col = [f"{r['q']:.2f}" for r in rows]
    cell_text = [[f, p, q] for f, p, q in zip(feat_col, pts_col, q_col)]
    cell_colors = []
    for r in rows:
        bg = SWAP_COLOR if r['feature'] in swap_feats else 'white'
        cell_colors.append([bg, bg, bg])
    col_labels = ['feature (binarized)', 'points', 'q']
    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   colLabels=col_labels, loc='center', cellLoc='left')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width([0, 1, 2])
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ccc')
        if row == 0:
            cell.set_facecolor(ARM_COLOR[arm])
            cell.set_text_props(color='white', fontweight='bold')
    ax.axis('off')
    auc_str = f"AUC {card['auc']:.3f}  intercept {card['intercept']:+.0f}"
    ax.set_title(f"{title}\n{auc_str}", fontsize=9, pad=6, color=ARM_COLOR[arm],
                 fontweight='bold')


def panel_a(axv, axc):
    """panel (a): side-by-side vanilla vs causal scorecards."""
    print('fitting heart scorecards...', flush=True)
    van_card, cau_card = fit_cards()
    van_feats = {r['feature'] for r in van_card['rows']}
    cau_feats = {r['feature'] for r in cau_card['rows']}
    swap = (van_feats - cau_feats) | (cau_feats - van_feats)
    print(f"  vanilla AUC={van_card['auc']:.3f}  causal AUC={cau_card['auc']:.3f}")
    print(f"  swapped features ({len(swap)}): {sorted(swap)}")
    _table_ax(axv, van_card, 'vanilla', swap, 'vanilla')
    _table_ax(axc, cau_card, 'causal (ours)', swap, 'causal')
    patch = mpatches.Patch(color=SWAP_COLOR, label='feature differs between cards')
    axv.legend(handles=[patch], loc='lower left', fontsize=7, framealpha=0.8)


def panel_b(ax):
    """panel (b): rashomon pool cloud — q-mass vs AUC per pool member."""
    if not POOL_CSV.exists():
        ax.text(0.5, 0.5, 'pool.csv not found\nrun rashomon.py first',
                ha='center', va='center', transform=ax.transAxes)
        return
    pool = pd.read_csv(POOL_CSV)
    for arm, color, label, marker in [('vanilla', ARM_COLOR['vanilla'], 'vanilla', 'o'),
                                       ('causal', ARM_COLOR['causal'], 'causal (ours)', 's')]:
        d = pool[pool['arm'] == arm]
        top = d.iloc[0]
        rest = d.iloc[1:]
        ax.scatter(rest['q_mass'], rest['auc'], c=color, alpha=0.4, s=22,
                   marker=marker, label=f'{label}  (n={len(d)})')
        ax.scatter(top['q_mass'], top['auc'], c=color, s=90, marker='*',
                   edgecolors='k', linewidths=0.5, zorder=5,
                   label=f'{label} deployed card')
    ax.set_xlabel('support q-mass (mean causal evidence of selected features)')
    ax.set_ylabel('test AUC')
    ax.set_title('rashomon pool: vanilla scattered at low q-mass,\ncausal concentrated at high q-mass')
    ax.legend(fontsize=7, framealpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT_DEFAULT)
    ap.add_argument('--no-panel-a', action='store_true',
                    help='skip scorecard fit (faster, pool cloud only)')
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.no_panel_a:
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        panel_b(ax)
    else:
        fig = plt.figure(figsize=(16, 7))
        axv = fig.add_subplot(1, 3, 1)
        axc = fig.add_subplot(1, 3, 2)
        axb = fig.add_subplot(1, 3, 3)
        panel_a(axv, axc)
        panel_b(axb)
        fig.suptitle('Heart disease: causal scorecard vs vanilla (integer point scoring)',
                     fontsize=11, y=1.01)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f'wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
