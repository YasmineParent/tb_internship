"""FICO: causal-prior FasterRisk vs causal feature-selection (CFS) baselines.

Nataliya's ask: compare against established CFS. Honest positioning: "not worse on
accuracy, more stable selection, and you get an interpretable integer scorecard a
raw feature set does not give you".

All feature-selection decisions (your q, and every CFS Markov blanket) are made on
the SAME held-out discovery set, so no method is starved of data relative to
another. Then, leakage-free, each split fits a scorecard at matched sparsity k and
is scored on held-out test rows.

Arms:
  - vanilla       : FasterRisk on all binarized columns (mu=0)
  - causal        : causal-prior FasterRisk (q from conditional-Gaussian discovery)
  - cfs_iamb      : pyCausalFS IAMB, Fisher-Z (Gaussian) CI test  [off-the-shelf]
  - cfs_hiton_mb  : pyCausalFS HITON-MB, Fisher-Z                 [off-the-shelf]
  - cfs_cg        : bnlearn IAMB with mi-cg, the mixed-data CI test (fair baseline)
then FasterRisk on the selected features' columns. Reports test AUC, sparsity, and
cross-split support stability (original-feature-level Jaccard).

usage:
    python experiments/causal_prior/real/fico_cfs.py --sentinel-nan --n-jobs 8
    python experiments/causal_prior/real/fico_cfs.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'external' / 'pyCausalFS'))

from src.causal_prior.cv_mu import cv_pick_mu  # noqa: E402
from src.causal_prior.priors import bnlearn_mb  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.fico_parity import (  # noqa: E402
    DATA, TARGET, POS_LABEL, load_features, fit_binarizer, apply_binarizer,
    discover_q, fit_eval, _import_fasterrisk)


def _cfs_fisherz(algo, X, y, alpha):
    """pyCausalFS Markov blanket with Fisher-Z (off-the-shelf, Gaussian CI test)."""
    from CBD.MBs.IAMB import IAMB
    from CBD.MBs.HITON.HITON_MB import HITON_MB
    fn = {'iamb': IAMB, 'hiton_mb': HITON_MB}[algo]
    data = pd.DataFrame(np.column_stack([X, (y > 0).astype(int)]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mb, _ = fn(data, X.shape[1], alpha, False)
    return sorted(int(j) for j in mb)


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def _stability(supports):
    pairs = [_jaccard(a, b) for a, b in combinations(supports, 2)]
    return float(np.mean(pairs)) if pairs else float('nan')


def _nogueira(supports, d):
    """nogueira & brown (jmlr 2017) chance-corrected selection stability.

    1 = identical selections across resamples, ~0 = no better than random, can go
    negative. unlike raw jaccard it corrects for chance and stays comparable across
    methods that select different numbers of features (cfs restricts to a small
    blanket, the causal arm roams), which is exactly the asymmetry here. d is the
    size of the full feature universe; features never selected contribute zero
    variance but still count toward d.
    """
    m = len(supports)
    if m < 2 or d == 0:
        return float('nan')
    sizes = np.array([len(s) for s in supports], dtype=float)
    kbar = sizes.mean()
    if kbar == 0 or kbar == d:
        return float('nan')  # denominator undefined (every resample selects all / none)
    cnt = Counter()
    for s in supports:
        cnt.update(s)
    phat = np.array(list(cnt.values()), dtype=float) / m
    var = (m / (m - 1)) * phat * (1 - phat)  # unobserved features add 0 to the sum
    num = var.sum() / d
    den = (kbar / d) * (1 - kbar / d)
    return float(1 - num / den)


def _split_unit(s, tr, te, X_orig, y, q_orig, mbs, names, k, n_thresholds, n_mu, n_cv, seed):
    """One train/test split: fit a scorecard for every arm (selections precomputed),
    score on test. Pure-Python (no R here) so it pickles for joblib."""
    FR = _import_fasterrisk()
    spec, _, parent = fit_binarizer(X_orig[tr], names.tolist(), n_thresholds)
    q_bin = q_orig[parent]
    Xtr, Xte = apply_binarizer(X_orig[tr], spec), apply_binarizer(X_orig[te], spec)
    ytr, yte = y[tr], (y[te] > 0).astype(int)
    all_cols = np.arange(Xtr.shape[1])
    mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ytr)))
    mu_grid = np.concatenate([[0.0], np.logspace(-2, 1, n_mu)]) * mu_scale

    def orig_supp(betas, cols):
        nz = np.nonzero(np.asarray(betas))[0]
        return frozenset(names[parent[cols[i]]] for i in nz)

    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = fit_eval(FR, Xtr, ytr, Xte, yte, 0.0, None, k, return_card=True)
        out['vanilla'] = (van['auc'], van['nfeat'], orig_supp(van['card']['betas'], all_cols))
        cv = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=q_bin, n_splits=n_cv,
                        criterion='log_loss', rng=np.random.default_rng(seed + s))
        cau = fit_eval(FR, Xtr, ytr, Xte, yte, cv.mu_star, q_bin, k, return_card=True)
        out['causal'] = (cau['auc'], cau['nfeat'], orig_supp(cau['card']['betas'], all_cols))
        for arm, mb in mbs.items():
            cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
            if len(cols) == 0:
                out[arm] = (float('nan'), 0, frozenset())
                continue
            fr = fit_eval(FR, Xtr[:, cols], ytr, Xte[:, cols], yte, 0.0, None, k, return_card=True)
            out[arm] = (fr['auc'], fr['nfeat'], orig_supp(fr['card']['betas'], cols))
    return s, out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--qsrc', default='ges_cg')
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--splits', type=int, default=10)
    p.add_argument('--test_size', type=float, default=0.3)
    p.add_argument('--discovery-frac', type=float, default=0.3)
    p.add_argument('--alpha', type=float, default=0.05, help='CI-test alpha for the CFS baselines')
    p.add_argument('--n_cv', type=int, default=5)
    p.add_argument('--n-mu', type=int, default=12)
    p.add_argument('--b', type=int, default=100)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.splits = 10, 3
    return args


def main():
    args = parse_args()
    if not DATA.exists():
        sys.exit(f'FICO CSV not found at {DATA}')

    df = pd.read_csv(DATA)
    y = np.where(df[TARGET].astype(str).str.strip() == POS_LABEL, 1, -1).astype(int)
    X_orig, names = load_features(df, args.sentinel_nan)
    names = np.asarray(names)
    n = len(y)

    pool_idx, disc_idx = train_test_split(np.arange(n), test_size=args.discovery_frac,
                                          stratify=(y > 0), random_state=args.seed)
    Xd, yd = X_orig[disc_idx], y[disc_idx]
    print(f'FICO n={n}: discovery={len(disc_idx)} (held out), eval pool={len(pool_idx)}', flush=True)

    # every feature-selection decision is made on the SAME held-out discovery set
    print(f'selections on held-out set: {args.qsrc} q (B={args.b}), iamb, hiton_mb, cg...', flush=True)
    q_orig = discover_q(args.qsrc, Xd, yd.astype(float), args.b, args.seed)
    mbs = {
        'cfs_iamb': _cfs_fisherz('iamb', Xd, yd, args.alpha),
        'cfs_hiton_mb': _cfs_fisherz('hiton_mb', Xd, yd, args.alpha),
        'cfs_cg': bnlearn_mb(Xd, yd, method='iamb', test='mi-cg', alpha=args.alpha),
    }
    for a, mb in mbs.items():
        print(f'  {a}: |MB|={len(mb)} -> {[names[j] for j in mb]}', flush=True)
    arms = ['vanilla', 'causal'] + list(mbs)

    sss = StratifiedShuffleSplit(n_splits=args.splits, test_size=args.test_size,
                                 random_state=args.seed)
    jobs = [(s, pool_idx[trp], pool_idx[tep])
            for s, (trp, tep) in enumerate(sss.split(pool_idx, (y[pool_idx] > 0).astype(int)))]
    ua = (X_orig, y, q_orig, mbs, names, args.k, args.n_thresholds, args.n_mu, args.n_cv, args.seed)
    if args.n_jobs == 1:
        results = [_split_unit(s, tr, te, *ua) for s, tr, te in jobs]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(_split_unit)(s, tr, te, *ua) for s, tr, te in jobs)

    rows, supports = [], {a: [] for a in arms}
    for s, out in results:
        for arm in arms:
            auc, nfeat, supp = out[arm]
            rows.append({'split': s, 'arm': arm, 'auc': auc, 'nfeat': nfeat})
            supports[arm].append(supp)

    df_rows = pd.DataFrame(rows)
    summ = (df_rows.groupby('arm')
            .agg(auc_mean=('auc', 'mean'), auc_std=('auc', 'std'), nfeat_mean=('nfeat', 'mean'))
            .reindex(arms))
    summ['mb_size'] = [np.nan, np.nan] + [len(mbs[a]) for a in list(mbs)]
    summ = summ.round(4)
    # note: this script is the downstream ACCURACY comparison (selections made once
    # on the held-out set). selection stability is measured separately, with every
    # method re-selecting per resample (fico_cfs_stability.py); measuring it here
    # would trivially give CFS stability 1.0 since its blanket is fixed.

    out_dir = new_run_dir(REPO_ROOT / 'results' / 'causal_prior' / 'fico_cfs' / f'fico_k{args.k}',
                          {**vars(args), 'mb_sizes': {a: len(m) for a, m in mbs.items()},
                           'leakage_free': True})
    df_rows.to_csv(out_dir / 'splits.csv', index=False)
    summ.to_csv(out_dir / 'summary.csv')
    print(summ.to_string(), flush=True)
    print(f'Done. Results in {out_dir}/', flush=True)


if __name__ == '__main__':
    main()
