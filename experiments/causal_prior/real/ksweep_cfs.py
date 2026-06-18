"""k-sweep for every arm, FasterRisk Figure-3 style: test AUC vs model size k, for
vanilla / our method / the CFS baselines, on one benchmark.

discovery (the expensive mi-cg step) is done once per resample and reused across k;
the prior strength mu is picked once per resample (cv at the largest k) and reused
across k. so a full k-sweep costs barely more than a single fixed-k run. output is a
long csv [rep, arm, k, auc, n, p] that plots straight into the FasterRisk-style grid.

usage:
    python experiments/causal_prior/real/ksweep_cfs.py --dataset heart --reps 25 --b 40
    python experiments/causal_prior/real/ksweep_cfs.py --dataset fico --sentinel-nan --reps 10 --b 15
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.priors import bnlearn_mb, bnlearn_mb_stability_q, discover_q  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk  # noqa: E402
from src.causal_prior.baselines import cfs_fisherz, iamb_fisherz_stability_q  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402

K_GRID = [2, 4, 6, 8, 10]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='heart')
    p.add_argument('--qsrc', default='ges_cg')
    p.add_argument('--reps', type=int, default=25)
    p.add_argument('--b', type=int, default=40)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n-thresholds', type=int, default=4)
    p.add_argument('--n-mu', type=int, default=8)
    p.add_argument('--n-cv', type=int, default=5)
    p.add_argument('--test-frac', type=float, default=0.3)
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.reps, args.b = 3, 10
    return args


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    print(f'{args.dataset}: n={len(y)}, features={len(names)}, positive={(y > 0).mean():.0%}', flush=True)
    FR = import_fasterrisk()
    k_ref = max(K_GRID)
    rows = []

    for r in range(args.reps):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac,
                                     random_state=args.seed + r)
        (tr, te), = sss.split(X_orig, (y > 0).astype(int))
        Xtr_o, ytr = X_orig[tr], y[tr]
        Xte_o, yte = X_orig[te], (y[te] > 0).astype(int)

        # discover once per resample (leakage-free: train rows only)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = discover_q(args.qsrc, Xtr_o, ytr.astype(float), args.b, args.seed)
            qi_cg = bnlearn_mb_stability_q(Xtr_o, ytr, test='mi-cg', alpha=args.alpha,
                                           B=args.b, rng=np.random.default_rng((args.seed, r, 9)))
            qi_fz = iamb_fisherz_stability_q(Xtr_o, ytr, args.alpha, args.b,
                                             np.random.default_rng((args.seed, r, 8)))
            mbs = {'cfs_iamb': cfs_fisherz('iamb', Xtr_o, ytr, args.alpha),
                   'cfs_hiton_mb': cfs_fisherz('hiton_mb', Xtr_o, ytr, args.alpha),
                   'cfs_cg': bnlearn_mb(Xtr_o, ytr, method='iamb', test='mi-cg', alpha=args.alpha)}

        # binarize on train rows only
        spec, _, parent = fit_binarizer(Xtr_o, names.tolist(), args.n_thresholds)
        Xtr, Xte = apply_binarizer(Xtr_o, spec), apply_binarizer(Xte_o, spec)
        allc = np.arange(Xtr.shape[1])
        mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ytr)))
        mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, args.n_mu)]) * mu_scale

        def auc(fr, cols):
            return float(roc_auc_score(yte, np.clip(fr.predict_proba(Xte[:, cols]), 1e-7, 1 - 1e-7)))

        # pick mu once per soft arm (cv at the largest k), reuse across k
        soft = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for arm, qv, tag in [('causal', q, 1), ('iamb_soft_cg', qi_cg, 2), ('iamb_soft_fz', qi_fz, 3)]:
                qb = qv[parent]
                mu = cv_pick_mu(Xtr, ytr, K=k_ref, mu_grid=mu_grid, q=qb, n_splits=args.n_cv,
                                criterion='log_loss',
                                rng=np.random.default_rng((args.seed, r, tag))).mu_star
                soft[arm] = (qb, float(mu))

            for k in K_GRID:
                v = FR(k=k, mu=0.0, freq=None); v.fit(Xtr, ytr)
                rows.append((r, 'vanilla', k, auc(v, allc)))
                for arm, (qb, mu) in soft.items():
                    fr = FR(k=k, mu=mu, freq=qb.astype(float)); fr.fit(Xtr, ytr)
                    rows.append((r, arm, k, auc(fr, allc)))
                for arm, mb in mbs.items():
                    cols = allc[np.isin(parent, mb)] if mb else allc[:0]
                    if len(cols) == 0:
                        rows.append((r, arm, k, float('nan'))); continue
                    fr = FR(k=min(k, len(cols)), mu=0.0, freq=None); fr.fit(Xtr[:, cols], ytr)
                    rows.append((r, arm, k, auc(fr, cols)))
        print(f'  rep {r + 1}/{args.reps} done', flush=True)

    df = pd.DataFrame(rows, columns=['rep', 'arm', 'k', 'auc'])
    df['n'], df['p'] = len(tr), len(names)
    out = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'ksweep_cfs' / args.dataset, vars(args))
    df.to_csv(out / 'ksweep_arms.csv', index=False)
    print(f'saved {out}', flush=True)


if __name__ == '__main__':
    main()
