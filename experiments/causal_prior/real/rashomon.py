"""fico rashomon-set navigation: does the prior steer the diverse pool toward causal supports.

faster risk returns a diverse near-optimal pool (collectsparsediversepool, the rashomon
set at the chosen sparsity). this fits vanilla and causal faster risk, extracts the whole
pool, and compares each member's support q-mass (mean causal evidence of its selected
features) within a matched-accuracy band. the claim (theory target #2): the vanilla
rashomon set already contains high-q members at equal auc, so a causally-grounded scorecard
exists at no accuracy cost, and the prior reliably selects it.

leakage-free: three disjoint sets, q discovered on a held-out discovery set, pools fit on a
train set, auc evaluated on a test set; binarization thresholds fit train-only.

usage:
    python experiments/causal_prior/real/fico_rashomon.py --qsrc ges_cg --sentinel-nan
    python experiments/causal_prior/real/fico_rashomon.py --train-n 3000 --mu-rel 0.1
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.priors import discover_q  # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='fico')
    ap.add_argument('--qsrc', choices=['pc', 'ges', 'pc_cg', 'ges_cg'], default='ges_cg')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--n_cv', type=int, default=5)
    ap.add_argument('--n_mu', type=int, default=12)
    ap.add_argument('--b', type=int, default=100, help='discovery stability subsamples')
    ap.add_argument('--n_thresholds', type=int, default=4)
    ap.add_argument('--discovery-frac', type=float, default=0.3)
    ap.add_argument('--test_size', type=float, default=0.3)
    ap.add_argument('--auc-band', type=float, default=0.01,
                    help='matched-accuracy band: compare q-mass only among pool members '
                         'within this auc of the best member across both pools')
    ap.add_argument('--train-n', type=int, default=0,
                    help='subsample training rows for speed (0 = full train split)')
    ap.add_argument('--mu-rel', type=float, default=-1.0,
                    help='fixed relative mu; >=0 skips the inner cv (fast path)')
    ap.add_argument('--sentinel-nan', action='store_true')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        args.b = 10
    return args


def pool_rows(fr, q_bin, Xte, yte, arm, parent, names):
    """one row per diverse-pool member: support size, support q-mass, test auc, and the
    original-feature support (json list of names) so pool members can be compared at the
    feature level (the predictive-multiplicity / one-card-resolution figure)."""
    rows = []
    for m in range(len(fr.betas_)):
        supp = np.nonzero(np.asarray(fr.betas_[m]))[0]
        if len(supp) == 0:
            continue
        feats = sorted(set(np.asarray(names)[parent[supp]].tolist()))
        p = np.clip(fr.predict_proba(Xte, model_idx=m), 1e-7, 1 - 1e-7)
        rows.append({'arm': arm, 'member': m, 'nfeat': int(len(supp)),
                     'q_mass': float(np.mean(q_bin[supp])),
                     'auc': float(roc_auc_score(yte, p)),
                     'support': json.dumps(feats)})
    return rows


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    n = len(y)

    # three disjoint sets: discovery (q), train (fit pools), test (eval auc)
    rest, disc_idx = train_test_split(np.arange(n), test_size=args.discovery_frac,
                                      stratify=(y > 0), random_state=args.seed)
    tr_idx, te_idx = train_test_split(rest, test_size=args.test_size,
                                      stratify=(y[rest] > 0), random_state=args.seed)
    if args.train_n and args.train_n < len(tr_idx):
        tr_idx = np.random.default_rng(args.seed).choice(tr_idx, args.train_n, replace=False)
    print(f'{args.dataset} n={n}: discovery={len(disc_idx)} (held out), train={len(tr_idx)}, '
          f'test={len(te_idx)}', flush=True)

    print(f'{args.qsrc.upper()} discovery (B={args.b}) on held-out set...', flush=True)
    q_orig = discover_q(args.qsrc, X_orig[disc_idx], y[disc_idx].astype(float), args.b, args.seed)
    spec, _, parent = fit_binarizer(X_orig[tr_idx], names, args.n_thresholds)
    q_bin = q_orig[parent]
    Xtr, Xte = apply_binarizer(X_orig[tr_idx], spec), apply_binarizer(X_orig[te_idx], spec)
    ytr, yte = y[tr_idx], (y[te_idx] > 0).astype(int)

    FR = import_fasterrisk()
    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, args.n_mu)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = FR(k=args.k, mu=0.0, freq=None); van.fit(Xtr, ytr)
        if args.mu_rel >= 0:
            mu_star = args.mu_rel * mu_scale
        else:
            mu_star = cv_pick_mu(Xtr, ytr, K=args.k, mu_grid=mu_grid, q=q_bin,
                                 n_splits=args.n_cv, criterion='log_loss',
                                 rng=np.random.default_rng(args.seed)).mu_star
        cau = FR(k=args.k, mu=float(mu_star), freq=q_bin.astype(float)); cau.fit(Xtr, ytr)
    mu_hat_rel = mu_star / (mu_scale or 1.0)

    pool = pd.DataFrame(pool_rows(van, q_bin, Xte, yte, 'vanilla', parent, names)
                        + pool_rows(cau, q_bin, Xte, yte, 'causal', parent, names))
    # matched-accuracy band: only members within --auc-band of the best member
    # across both pools, so the q-mass comparison is not confounded by an auc shift
    auc_floor = pool['auc'].max() - args.auc_band
    matched = pool[pool['auc'] >= auc_floor]

    def _summ(p):
        return (p.groupby('arm')
                .agg(n=('member', 'size'), q_mass_mean=('q_mass', 'mean'),
                     q_mass_top=('q_mass', 'max'), auc_min=('auc', 'min'),
                     auc_max=('auc', 'max'))
                .reindex(['vanilla', 'causal']).round(4))
    summ_full, summ_matched = _summ(pool), _summ(matched)

    config = {**vars(args), 'mu_hat_rel': mu_hat_rel, 'auc_floor': float(auc_floor),
              'discovery_n': int(len(disc_idx)), 'train_n': int(len(tr_idx)),
              'test_n': int(len(te_idx)), 'leakage_free': True}
    out = new_run_dir(ROOT / 'results' / 'causal_prior' / 'rashomon' / f'{args.dataset}_k{args.k}',
                      config)
    pool.to_csv(out / 'pool.csv', index=False)
    summ_full.to_csv(out / 'summary_full_pool.csv')
    summ_matched.to_csv(out / 'summary_matched_band.csv')

    print('full pool:\n' + summ_full.to_string(), flush=True)
    print(f"\nmatched-accuracy band (auc >= {auc_floor:.4f}):\n" + summ_matched.to_string(),
          flush=True)
    print(f"\nat matched accuracy the prior shifts pool mean q-mass "
          f"{summ_matched.loc['vanilla','q_mass_mean']:.3f} -> "
          f"{summ_matched.loc['causal','q_mass_mean']:.3f} "
          f"(mu_hat_rel={mu_hat_rel:.3f}).", flush=True)
    print(f"Done (leakage-free). Results + figure in {out}/", flush=True)


if __name__ == '__main__':
    main()
