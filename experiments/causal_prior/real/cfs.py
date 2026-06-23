"""Causal-prior FasterRisk vs causal feature selection (CFS) on real benchmarks.

Arms (2x2 of ci-test x soft/hard, plus vanilla):
  vanilla       mu=0
  causal        soft prior, GES-CG q                       [deployed method]
  iamb_soft_cg  soft prior, IAMB mi-cg q                  [deployed method, IAMB source]
  iamb_soft_fz  soft prior, IAMB Fisher-Z q               [ablation: invalid ci-test]
  cfs_iamb      hard IAMB + Fisher-Z                       [ablation: hard selection]
  cfs_hiton_mb  hard HITON-MB + Fisher-Z
  cfs_cg        hard IAMB mi-cg                            [valid hard CFS baseline]

the 2x2 keeps soft-vs-hard orthogonal to the ci-test choice.

q-mode: with --n-grid q is discovered once on a held-out split (leakage-free);
without it q is re-discovered per resample on each split's train.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.cv_mu import cv_pick_mu, make_mu_grid  # noqa: E402
from src.causal_prior.priors import bnlearn_mb, bnlearn_mb_stability_q, discover_q  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk  # noqa: E402
from src.causal_prior.stability import mean_pairwise_jaccard, nogueira  # noqa: E402
from src.causal_prior.baselines import cfs_fisherz, iamb_fisherz_stability_q  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402
from experiments.causal_prior.real.datasets import load_dataset  # noqa: E402

ARMS = ['vanilla', 'causal', 'iamb_soft_cg', 'iamb_soft_fz',
        'cfs_iamb', 'cfs_hiton_mb', 'cfs_cg']


def _blankets(Xs, ys, alpha):
    """re-select all CFS blankets on this (sub)sample."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return {'cfs_iamb': cfs_fisherz('iamb', Xs, ys, alpha),
                'cfs_hiton_mb': cfs_fisherz('hiton_mb', Xs, ys, alpha),
                'cfs_cg': bnlearn_mb(Xs, ys, method='iamb', test='mi-cg', alpha=alpha)}


def _eval_split(Xtr_o, ytr, Xte_o, yte01, q_orig, qi_cg_orig, qi_fz_orig, mbs, names, k,
                n_thresholds, n_mu, mu_rel, n_cv, seed_tuple):
    """fit all arms on one train/test split; return per-arm auc + support + mu_hat_rel."""
    FR = import_fasterrisk()
    spec, _, parent = fit_binarizer(Xtr_o, names.tolist(), n_thresholds)
    Xtr, Xte = apply_binarizer(Xtr_o, spec), apply_binarizer(Xte_o, spec)
    all_cols = np.arange(Xtr.shape[1])
    mu_scale, mu_grid = make_mu_grid(Xtr, ytr, n_mu)

    def supp(cols, betas):
        nz = np.nonzero(np.asarray(betas))[0]
        return frozenset(names[parent[cols[nz]]].tolist())

    def auc(fr, cols):
        return float(roc_auc_score(yte01, np.clip(fr.predict_proba(Xte[:, cols]), 1e-7, 1 - 1e-7)))

    def soft_arm(qv, tag):
        qb = qv[parent]
        if mu_rel >= 0:
            mu = mu_rel * mu_scale
        else:
            mu = cv_pick_mu(Xtr, ytr, K=k, mu_grid=mu_grid, q=qb, n_splits=n_cv,
                            criterion='log_loss',
                            rng=np.random.default_rng(seed_tuple + (tag,))).mu_star
        fr = FR(k=k, mu=float(mu), freq=qb.astype(float)); fr.fit(Xtr, ytr)
        return auc(fr, all_cols), supp(all_cols, fr.betas_[0]), mu / (mu_scale or 1.0)

    rec = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        van = FR(k=k, mu=0.0, freq=None); van.fit(Xtr, ytr)
        rec['auc_vanilla'], rec['supp_vanilla'] = auc(van, all_cols), supp(all_cols, van.betas_[0])
        a, s, mh = soft_arm(q_orig, 1)
        rec['auc_causal'], rec['supp_causal'], rec['mu_hat_rel'] = a, s, mh
        a, s, _ = soft_arm(qi_cg_orig, 2)
        rec['auc_iamb_soft_cg'], rec['supp_iamb_soft_cg'] = a, s
        a, s, _ = soft_arm(qi_fz_orig, 3)
        rec['auc_iamb_soft_fz'], rec['supp_iamb_soft_fz'] = a, s
        for arm, mb in mbs.items():
            cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
            if len(cols) == 0:
                rec[f'auc_{arm}'], rec[f'supp_{arm}'] = float('nan'), frozenset()
                continue
            fr = FR(k=k, mu=0.0, freq=None); fr.fit(Xtr[:, cols], ytr)
            rec[f'auc_{arm}'], rec[f'supp_{arm}'] = auc(fr, cols), supp(cols, fr.betas_[0])
    return rec


def _unit_heldout(n, r, train_pool, test_abs, X_orig, y, q_orig, qi_cg_orig, qi_fz_orig, names, args):
    """one resample (heldout-q mode): q fixed, blankets re-selected on the subsample."""
    sub = np.random.default_rng((args.seed, n, r)).choice(train_pool, size=n, replace=False)
    Xs, ys = X_orig[sub], y[sub]
    mbs = _blankets(Xs, ys, args.alpha)
    rec = _eval_split(Xs, ys, X_orig[test_abs], (y[test_abs] > 0).astype(int),
                      q_orig, qi_cg_orig, qi_fz_orig, mbs, names, args.k, args.n_thresholds, args.n_mu,
                      args.mu_rel, args.n_cv, (args.seed, n, r))
    rec['n'], rec['rep'] = n, r
    return rec


def _unit_resample(n, r, X_orig, y, names, args):
    """one resample (per-resample mode): fresh split, all q-sources and blankets re-discovered."""
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_frac, random_state=args.seed + r)
    (tr, te), = sss.split(X_orig, (y > 0).astype(int))
    if n is not None and n < len(tr):
        tr = np.random.default_rng((args.seed, n, r)).choice(tr, size=n, replace=False)
    Xtr_o, ytr = X_orig[tr], y[tr]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q(args.qsrc, Xtr_o, ytr.astype(float), args.b, args.seed)
        qi_cg = bnlearn_mb_stability_q(Xtr_o, ytr, test='mi-cg', alpha=args.alpha, B=args.b,
                                       rng=np.random.default_rng((args.seed, n or 0, r, 9)))
        qi_fz = iamb_fisherz_stability_q(Xtr_o, ytr, args.alpha, args.b,
                                         np.random.default_rng((args.seed, n or 0, r, 8)))
        mbs = _blankets(Xtr_o, ytr, args.alpha)
    rec = _eval_split(Xtr_o, ytr, X_orig[te], (y[te] > 0).astype(int), q, qi_cg, qi_fz, mbs, names,
                      args.k, args.n_thresholds, args.n_mu, args.mu_rel, args.n_cv,
                      (args.seed, n or 0, r))
    rec['n'], rec['rep'] = (n if n else len(tr)), r
    return rec


def _summary(res, names):
    d = len(names)
    ns = sorted({x['n'] for x in res})
    rows = []
    for n in ns:
        rn = [x for x in res if x['n'] == n]
        row = {'n': n, 'mu_nonzero_frac': float(np.mean([x['mu_hat_rel'] > 0 for x in rn]))}
        for arm in ARMS:
            supps = [x[f'supp_{arm}'] for x in rn]
            row[f'auc_{arm}'] = float(np.nanmean([x[f'auc_{arm}'] for x in rn]))
            row[f'stab_{arm}'] = mean_pairwise_jaccard(supps)
            row[f'nog_{arm}'] = nogueira(supps, d)
        rows.append(row)
    return pd.DataFrame(rows)


def _contrasts(res):
    """paired causal-vs-arm and the 2x2 soft-vs-hard contrasts over shared resamples."""
    from scipy import stats
    long = pd.DataFrame([{f'auc_{a}': x[f'auc_{a}'] for a in ARMS} for x in res])

    def paired(a, b):
        diff = (long[f'auc_{a}'] - long[f'auc_{b}']).to_numpy()
        diff = diff[~np.isnan(diff)]
        p = stats.wilcoxon(diff).pvalue if np.any(diff) else float('nan')
        return float(np.mean(diff)), p

    causal = {a: paired('causal', a) for a in ARMS if a != 'causal'}
    # iamb ci-test x use 2x2: each contrast varies exactly one of {soft/hard, ci test}
    ablation = {'iamb_soft_fz - cfs_iamb (soft vs hard, fisher-z)': paired('iamb_soft_fz', 'cfs_iamb'),
                'iamb_soft_cg - cfs_cg (soft vs hard, mi-cg)': paired('iamb_soft_cg', 'cfs_cg'),
                'iamb_soft_fz - iamb_soft_cg (fisher-z vs mi-cg, both soft)': paired('iamb_soft_fz', 'iamb_soft_cg'),
                'cfs_iamb - cfs_cg (fisher-z vs mi-cg, both hard)': paired('cfs_iamb', 'cfs_cg')}
    return long, causal, ablation


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='fico')
    p.add_argument('--qsrc', default='ges_cg')
    p.add_argument('--n-grid', default=None, help='scarcity sweep sizes, e.g. 150,300,600,1200')
    p.add_argument('--q-mode', choices=['auto', 'heldout', 'per-resample'], default='auto')
    p.add_argument('--reps', type=int, default=20)
    p.add_argument('--k', type=int, default=10)
    p.add_argument('--discovery-frac', type=float, default=0.3)
    p.add_argument('--test-n', type=int, default=2500, help='held-out mode: fixed test size')
    p.add_argument('--test-frac', type=float, default=0.3, help='per-resample mode: test fraction')
    p.add_argument('--n-mu', type=int, default=8)
    p.add_argument('--mu-rel', type=float, default=-1.0, help='fixed relative mu; >=0 skips inner cv')
    p.add_argument('--b', type=int, default=50)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_cv', type=int, default=5)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--sentinel-nan', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.b, args.reps = 10, 4
        if args.dataset == 'fico' and args.n_grid is None:
            args.n_grid = '300,600'
    args.n_grid = [int(x) for x in args.n_grid.split(',')] if args.n_grid else None
    if args.q_mode == 'auto':
        args.q_mode = 'heldout' if args.n_grid else 'per-resample'
    return args


def main():
    args = parse_args()
    X_orig, names, y = load_dataset(args.dataset, args)
    print(f'{args.dataset}: n={len(y)}, features={len(names)}, '
          f'positive={(y > 0).mean():.0%}, q-mode={args.q_mode}', flush=True)

    grid = args.n_grid or [None]
    if args.q_mode == 'heldout':
        rest, disc = train_test_split(np.arange(len(y)), test_size=args.discovery_frac,
                                      stratify=(y > 0), random_state=args.seed)
        pool, test_abs = train_test_split(rest, test_size=args.test_n,
                                          stratify=(y[rest] > 0), random_state=args.seed)
        if max(grid) > len(pool):
            sys.exit(f'largest n ({max(grid)}) exceeds train pool ({len(pool)})')
        print(f'held-out discovery={len(disc)}, pool={len(pool)}, test={len(test_abs)}; '
              f'{args.qsrc} q (B={args.b})...', flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q_orig = discover_q(args.qsrc, X_orig[disc], y[disc].astype(float), args.b, args.seed)
            qi_cg_orig = bnlearn_mb_stability_q(X_orig[disc], y[disc], test='mi-cg',
                                                alpha=args.alpha, B=args.b,
                                                rng=np.random.default_rng((args.seed, 99)))
            qi_fz_orig = iamb_fisherz_stability_q(X_orig[disc], y[disc], args.alpha, args.b,
                                                  np.random.default_rng((args.seed, 98)))
        jobs = [(n, r) for n in grid for r in range(args.reps)]
        unit = lambda n, r: _unit_heldout(n, r, pool, test_abs, X_orig, y,
                                          q_orig, qi_cg_orig, qi_fz_orig, names, args)
    else:
        jobs = [(n, r) for n in grid for r in range(args.reps)]
        unit = lambda n, r: _unit_resample(n, r, X_orig, y, names, args)

    print(f'{len(jobs)} resamples on n_jobs={args.n_jobs}', flush=True)
    if args.n_jobs == 1:
        import time
        res, t0 = [], time.time()
        for i, (n, r) in enumerate(jobs, 1):
            res.append(unit(n, r))
            el = time.time() - t0
            print(f'  [{i}/{len(jobs)}] n={n} rep={r}  elapsed={el/60:.1f}m  '
                  f'eta={el/i*(len(jobs)-i)/60:.1f}m', flush=True)
    else:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=args.n_jobs, backend='loky', verbose=5)(
            delayed(unit)(n, r) for n, r in jobs)

    summ = _summary(res, names).round(4)
    long, causal, ablation = _contrasts(res)

    out = new_run_dir(ROOT / 'results' / 'causal_prior' / 'cfs' / f'{args.dataset}_k{args.k}',
                      {**vars(args), 'leakage_free': True})
    summ.to_csv(out / 'summary.csv', index=False)
    long.to_csv(out / 'resamples.csv', index=False)

    print(summ.to_string(index=False), flush=True)
    print('\npaired causal - arm (mean delta, wilcoxon p):', flush=True)
    for a, (md, p) in causal.items():
        print(f'  vs {a:14s} delta={md:+.4f}  p={p:.3f}', flush=True)
    print('\n2x2 ablation contrasts (mean delta, wilcoxon p):', flush=True)
    for lab, (md, p) in ablation.items():
        print(f'  {lab:48s} delta={md:+.4f}  p={p:.3f}', flush=True)
    print(f'Done. Results in {out}/', flush=True)


if __name__ == '__main__':
    main()
