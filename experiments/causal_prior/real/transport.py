"""Folktables transport: does the causal prior buy out-of-state transport that
predictive feature selection does not?

Real-data version of the §6.4 synthetic transport story, and the axis the FICO
comparison cannot test (FICO is one stationary distribution). Train on a source
state, test in-distribution (held-out source rows) and on shifted target states.

Two design choices learned the hard way:
  - the feature set must contain non-causal *predictive* correlates for the prior
    to have anything to prune. we one-hot POBP (place of birth): it predicts income
    within a state via state-specific composition but should not transport. without
    such features (and with more scorecard slots than features) the prior is inert.
  - mu is SWEPT, not cv-picked. cv on the source rewards the spurious correlates
    (they help in-distribution), so it zeroes the prior. §6.4 sweeps mu_rel and
    reads transport as a function of prior strength; we do the same.

Arms: vanilla (= causal at mu=0), causal at a grid of mu_rel, and the cfs
baselines (cfs_cg = valid mi-cg; cfs_iamb/hiton = naive fisher-z, invalid on mixed
data). Selection (q, cfs blankets) is done once on a held-out source discovery
split; seeds resample the source train/test. Discriminating quantity per arm:
    transport_gap(target) = auc(source in-distribution) - auc(target).
Win signature: as mu_rel grows the causal source auc dips slightly but the target
auc holds or rises, so its transport gap falls below vanilla's.

Run under `caffeinate -i`; first run downloads the ACS PUMS files (~per state).

usage:
    python experiments/causal_prior/real/folktables_transport.py --source CA --targets PR,SD
    python experiments/causal_prior/real/folktables_transport.py --smoke
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.causal_prior.priors import bnlearn_mb, discover_q  # noqa: E402
from src.causal_prior.binarize import fit_binarizer, apply_binarizer  # noqa: E402
from src.causal_prior.scorecard import import_fasterrisk  # noqa: E402
from src.causal_prior.baselines import cfs_fisherz  # noqa: E402
from experiments._io import new_run_dir  # noqa: E402

DATA_ROOT = ROOT / 'data' / 'real' / 'folktables'


def _task_cfg(task):
    """feature layout + target per folktables task. income: POBP is the spurious
    state-correlate. pubcov: PINCP (income) is the policy-dependent predictor whose
    relationship to coverage flips between Medicaid-expansion and non-expansion states,
    so it should fail an invariance test while genuine eligibility features (DIS, AGEP,
    CIT) stay stable. ST is dropped (constant within a state)."""
    from folktables import ACSIncome, ACSPublicCoverage
    if task == 'income':
        return dict(prob=ACSIncome, cont=['AGEP', 'WKHP', 'SCHL'], cat=['SEX', 'RAC1P'],
                    high=['POBP'], target=lambda v: np.where(v > 50000, 1, -1))
    if task == 'pubcov':
        return dict(prob=ACSPublicCoverage, cont=['AGEP', 'SCHL', 'PINCP'],
                    cat=['SEX', 'DIS', 'CIT', 'MAR', 'NATIVITY', 'ESR', 'MIG',
                         'DEAR', 'DEYE', 'DREM', 'RAC1P'],
                    high=[], target=lambda v: np.where(v == 1, 1, -1))
    raise ValueError(f'unknown task {task!r}')


def load_raw(state, year, cfg):
    from folktables import ACSDataSource
    ds = ACSDataSource(survey_year=str(year), horizon='1-Year', survey='person',
                       root_dir=str(DATA_ROOT))
    prob = cfg['prob']
    raw = prob._preprocess(ds.get_data(states=[state], download=True))
    # drop NaN only on the columns we actually use; dropping on unused task features
    # (e.g. FER, which is NaN for all males) would silently bias the sample.
    use = cfg['cont'] + cfg['cat'] + cfg['high'] + [prob.target]
    return raw.dropna(subset=use)[use].reset_index(drop=True)


def build_encoder(df_src, topk, cfg):
    """fix one-hot levels from the SOURCE so all states share identical columns."""
    enc = {'cat': {c: sorted(df_src[c].dropna().unique().tolist()) for c in cfg['cat']}}
    enc['high'] = {c: df_src[c].value_counts().index[:topk].tolist() for c in cfg['high']}
    return enc


def encode(df, enc, cfg):
    """raw ACS df -> (X_orig float matrix, names, y in {-1,1}) under a fixed encoder."""
    cols, names = [], []
    for c in cfg['cont']:
        cols.append(df[c].to_numpy(float)); names.append(c)
    for c in cfg['cat']:
        for lv in enc['cat'][c]:
            cols.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={lv}')
    for c in cfg['high']:
        keep = enc['high'][c]
        for lv in keep:
            cols.append((df[c] == lv).to_numpy(float)); names.append(f'{c}={lv}')
        cols.append((~df[c].isin(keep)).to_numpy(float)); names.append(f'{c}=other')
    y = cfg['target'](df[cfg['prob'].target].to_numpy(float)).astype(int)
    return np.column_stack(cols), names, y


def icp_invariance_q(env_X_y, c_reg=1.0):
    """ICP-flavoured invariance score on the original features: per-feature
    logistic-coefficient sign stability across environments. q_j in [0,1]: 1 = same sign
    in every environment (relationship invariant, a genuine cause), ~0 = the sign flips
    across environments (an association that is policy- or state-dependent and does not
    transport). X is standardised per environment so signs are comparable. This is the
    invariance-based source the cross-environment selection q cannot provide: it sees a
    relationship flip, not just a selection change."""
    from sklearn.linear_model import LogisticRegression
    signs = []
    for X, y in env_X_y:
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            clf = LogisticRegression(C=c_reg, max_iter=2000).fit(Xs, (y > 0).astype(int))
        signs.append(np.sign(clf.coef_[0]))
    return np.abs(np.vstack(signs).mean(0))


def _auc(fr, cols, X, y01):
    if fr is None:
        return float('nan')
    return float(roc_auc_score(y01, np.clip(fr.predict_proba(X[:, cols]), 1e-7, 1 - 1e-7)))


def main():
    args = parse_args()
    cfg = _task_cfg(args.task)
    targets = args.targets.split(',')
    print(f'task={args.task}, source={args.source}, targets={targets} (year {args.year})...', flush=True)
    src_raw = load_raw(args.source, args.year, cfg)
    enc = build_encoder(src_raw, args.topk, cfg)
    Xsrc, names, ysrc = encode(src_raw, enc, cfg)
    names = np.asarray(names)
    tgt = {}
    for t in targets:
        Xt, _, yt = encode(load_raw(t, args.year, cfg), enc, cfg)
        tgt[t] = (Xt, yt)

    idx_Z = None
    if args.inject_spurious:
        # source: Z agrees with the label with prob rho (strong predictor).
        # target: Z is independent noise (its label-correlation is destroyed by the shift).
        agree = np.random.default_rng((args.seed, 777)).random(len(ysrc)) < args.spurious_rho
        z = np.where(agree, ysrc > 0, ysrc <= 0).astype(float)
        Xsrc = np.column_stack([Xsrc, z]); names = np.append(names, 'Z_spurious')
        idx_Z = len(names) - 1
        for t, (Xt, yt) in list(tgt.items()):
            zt = (np.random.default_rng((args.seed, 778, hash(t) % 97)).random(len(yt)) < 0.5).astype(float)
            tgt[t] = (np.column_stack([Xt, zt]), yt)
        print(f'injected Z_spurious: source corr(z,y)={np.corrcoef(z, ysrc > 0)[0, 1]:+.2f}, '
              f'target corr ~ 0', flush=True)

    print(f'features={len(names)} (>=2 cont + indicators); source n={len(ysrc)}; ' +
          '; '.join(f'{t} n={len(v[1])}' for t, v in tgt.items()), flush=True)

    FR = import_fasterrisk()
    rng0 = np.random.default_rng(args.seed)
    perm = rng0.permutation(len(ysrc))
    disc = perm[:args.n_disc]                      # held-out selection split
    pool = perm[args.n_disc:]                      # seeds resample train/test from here

    # selection once on the held-out source discovery split
    print(f'selection on held-out source discovery (n={len(disc)})...', flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q = discover_q('ges_cg', Xsrc[disc], ysrc[disc].astype(float), args.b, args.seed)
        mbs = {'cfs_iamb': cfs_fisherz('iamb', Xsrc[disc], ysrc[disc], args.alpha),
               'cfs_hiton_mb': cfs_fisherz('hiton_mb', Xsrc[disc], ysrc[disc], args.alpha),
               'cfs_cg': bnlearn_mb(Xsrc[disc], ysrc[disc], method='iamb', test='mi-cg',
                                    alpha=args.alpha)}
    for a, mb in mbs.items():
        print(f'  {a}: |MB|={len(mb)}', flush=True)

    # cross-environment (invariance) q: discover on each extra source state under the
    # same encoder and aggregate, so state-specific correlates (POBP) wash out while
    # invariantly-selected causes survive. this is the fix for the single-state q that
    # inherits environment-specific correlates.
    cross = [c for c in args.cross_states.split(',') if c]
    q_ce = q_icp = None
    if cross:
        print(f'cross-environment q over {[args.source] + cross} (agg={args.ce_agg})...', flush=True)
        per_state = [q]
        envs = [(Xsrc[disc], ysrc[disc])]              # source env, for the ICP score
        for c in cross:
            Xc, _, yc = encode(load_raw(c, args.year, cfg), enc, cfg)
            dc = np.random.default_rng((args.seed, abs(hash(c)) % 9973)).permutation(len(yc))[:args.n_disc]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                per_state.append(discover_q('ges_cg', Xc[dc], yc[dc].astype(float), args.b, args.seed))
            envs.append((Xc[dc], yc[dc]))
        Q = np.vstack(per_state)
        q_ce = Q.min(0) if args.ce_agg == 'min' else Q.mean(0)
        q_icp = icp_invariance_q(envs)                 # invariance (sign-stability) q
        print(f'  top cross-env (selection) features: {list(names[np.argsort(-q_ce)[:6]])}', flush=True)
        print(f'  top invariance (ICP) features: {list(names[np.argsort(-q_icp)[:6]])}', flush=True)
        print(f'  least invariant (sign flips): {list(names[np.argsort(q_icp)[:4]])}', flush=True)

    if idx_Z is not None:  # correct external causal evidence: the injected feature is not a cause
        print(f'  discovered q[Z_spurious]={q[idx_Z]:.2f} before override -> 0', flush=True)
        q[idx_Z] = 0.0
        if q_ce is not None:
            q_ce[idx_Z] = 0.0
        if q_icp is not None:
            q_icp[idx_Z] = 0.0

    mu_grid = [0.0] + args.mu_rel
    rows = []
    for s in range(args.reps):
        rng = np.random.default_rng((args.seed, s))
        sel = rng.permutation(len(pool))
        tr = pool[sel[:args.n_train]]
        te = pool[sel[args.n_train:args.n_train + args.n_test]]
        spec, _, parent = fit_binarizer(Xsrc[tr], names.tolist(), args.n_thresholds)
        q_bin = q[parent]
        Xtr = apply_binarizer(Xsrc[tr], spec)
        all_cols = np.arange(Xtr.shape[1])
        Xte_src = apply_binarizer(Xsrc[te], spec)
        yte01 = (ysrc[te] > 0).astype(int)
        tgt_bin = {}  # one target eval subsample per seed, binarized with the source spec
        for t, (Xt, yt) in tgt.items():
            idx = rng.choice(len(yt), min(args.n_test, len(yt)), replace=False)
            tgt_bin[t] = (apply_binarizer(Xt[idx], spec), (yt[idx] > 0).astype(int))

        mu_scale = float(np.median(0.5 * np.abs(Xtr.T @ ysrc[tr])))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for mr in mu_grid:  # causal prior swept (mr=0 is vanilla)
                fr = FR(k=args.k, mu=mr * mu_scale, freq=q_bin.astype(float) if mr > 0 else None)
                fr.fit(Xtr, ysrc[tr])
                rec = {'seed': s, 'arm': 'vanilla' if mr == 0 else 'causal', 'mu_rel': mr,
                       'auc_source': _auc(fr, all_cols, Xte_src, yte01)}
                for t, (Xtb, ytb) in tgt_bin.items():
                    rec[f'auc_{t}'] = _auc(fr, all_cols, Xtb, ytb)
                rows.append(rec)
            for arm_name, q_extra in (('causal_ce', q_ce), ('causal_icp', q_icp)):
                if q_extra is None:
                    continue
                q_eb = q_extra[parent]
                for mr in args.mu_rel:
                    fr = FR(k=args.k, mu=mr * mu_scale, freq=q_eb.astype(float))
                    fr.fit(Xtr, ysrc[tr])
                    rec = {'seed': s, 'arm': arm_name, 'mu_rel': mr,
                           'auc_source': _auc(fr, all_cols, Xte_src, yte01)}
                    for t, (Xtb, ytb) in tgt_bin.items():
                        rec[f'auc_{t}'] = _auc(fr, all_cols, Xtb, ytb)
                    rows.append(rec)
            for arm, mb in mbs.items():
                cols = all_cols[np.isin(parent, mb)] if mb else all_cols[:0]
                fr = None if len(cols) == 0 else FR(k=args.k, mu=0.0, freq=None)
                if fr is not None:
                    fr.fit(Xtr[:, cols], ysrc[tr])
                rec = {'seed': s, 'arm': arm, 'mu_rel': np.nan,
                       'auc_source': _auc(fr, cols, Xte_src, yte01)}
                for t, (Xtb, ytb) in tgt_bin.items():
                    rec[f'auc_{t}'] = _auc(fr, cols, Xtb, ytb)
                rows.append(rec)
        print(f'  seed {s + 1}/{args.reps} done', flush=True)

    long = pd.DataFrame(rows)
    env_cols = ['auc_source'] + [f'auc_{t}' for t in targets]
    summ = long.groupby(['arm', 'mu_rel'], dropna=False)[env_cols].mean().round(4)
    for t in targets:
        summ[f'gap_{t}'] = (summ['auc_source'] - summ[f'auc_{t}']).round(4)

    out = new_run_dir(ROOT / 'results' / 'causal_prior' / 'folktables_transport'
                      / f'{args.source}_to_{"-".join(targets)}_{args.year}_musweep',
                      {**vars(args), 'features': names.tolist()})
    long.to_csv(out / 'seeds.csv', index=False)
    summ.to_csv(out / 'summary.csv')
    plot_transport(long, targets, out / 'transport.png')
    print(summ.to_string(), flush=True)
    print(f'Done. Results in {out}/', flush=True)


def plot_transport(long, targets, out_png):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = targets[-1]  # the larger-shift target
    cau = long[long['arm'] == 'causal'].groupby('mu_rel')[['auc_source', f'auc_{t}']].mean()
    van = long[long['arm'] == 'vanilla'][f'auc_{t}'].mean()
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(cau.index, cau[f'auc_{t}'], marker='o', color='C1', label=f'causal -> {t}')
    ax.plot(cau.index, cau['auc_source'], marker='o', color='C0', ls='--',
            label='causal -> source (in-dist)')
    ax.axhline(van, color='0.5', ls=':', label=f'vanilla -> {t} (mu=0)')
    for arm, c in (('cfs_cg', 'C2'), ('cfs_iamb', '0.6')):
        v = long[long['arm'] == arm][f'auc_{t}'].mean()
        ax.axhline(v, color=c, ls=':', lw=1, label=f'{arm} -> {t}')
    ax.set(xlabel='prior strength mu_rel', ylabel='test AUC',
           title=f'transport under shift ({t})')
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_png, dpi=160); plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['income', 'pubcov'], default='income',
                   help='folktables task; pubcov has income (PINCP) as the policy-dependent '
                        'predictor whose relationship flips between expansion/non-expansion states')
    p.add_argument('--source', default='CA')
    p.add_argument('--targets', default='PR,SD')
    p.add_argument('--year', type=int, default=2018)
    p.add_argument('--reps', type=int, default=6)
    p.add_argument('--k', type=int, default=8)
    p.add_argument('--mu-rel', type=float, nargs='+', default=[0.25, 0.5, 1.0, 2.0, 4.0])
    p.add_argument('--topk', type=int, default=8, help='POBP levels kept before "other"')
    p.add_argument('--n-disc', type=int, default=1500)
    p.add_argument('--n-train', type=int, default=2500)
    p.add_argument('--n-test', type=int, default=4000)
    p.add_argument('--b', type=int, default=40)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n_thresholds', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cross-states', default='',
                   help='extra source states for the cross-environment (invariance) q, '
                        'e.g. "TX,NY,FL"; the prior keeps features selected consistently '
                        'across them, down-weighting state-specific correlates. empty = off.')
    p.add_argument('--ce-agg', choices=['mean', 'min'], default='mean',
                   help='aggregate the per-state q into the invariance prior (min = strict)')
    p.add_argument('--inject-spurious', action='store_true',
                   help='semi-synthetic: inject a feature strongly predictive in the source '
                        'but noise in the target (correlation shifts across environments), and '
                        'give the prior the correct knowledge that it is non-causal (q=0). '
                        'isolates the transport mechanism on a real-data substrate.')
    p.add_argument('--spurious-rho', type=float, default=0.85,
                   help='source agreement of the injected feature with the label')
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    if args.smoke:
        args.reps, args.b, args.n_disc, args.n_train, args.n_test, args.targets, args.mu_rel = \
            2, 10, 800, 1200, 1500, 'SD', [1.0, 4.0]
    return args


if __name__ == '__main__':
    main()
