"""Environment-shift experiment: does the causal prior buy out-of-environment
transport that a merely-selective predictive prior does not?

Setup (one shared SCM per (p_edge, seed) cell; src/data/synthetic_envs.py):
  - train env  : gamma = 1.0 reference draw. q-sources and the scorecard are
                 learned here only.
  - test envs  : gamma in TEST_GAMMAS. gamma=1.0 is a fresh in-distribution draw
                 (the transport baseline); gamma=0.0 decouples the correlates into
                 noise; gamma=-1.0 reverses the spurious associations. The causal
                 mechanism P(Y | Pa(Y)) is identical in all of them; only the
                 non-causal correlate<->Y associations move.

We sweep a small mu_rel grid and fit FasterRisk once per (source, mu). At mu=0
every source collapses to the same vanilla model (the prior is inert), so that
baseline is computed once and shared. K = 2 k* by default: the prior only bites
once there are spare slots beyond k* for FR to otherwise fill with predictive
correlates.

Discriminating quantity: the transport gap
    delta_auc(gamma) = auc(in-distribution gamma=1) - auc(gamma).
Causal/selective-on-S* priors (oracle, ges, pc) keep the support causal and
transport (delta ~0); bootstrap_l1, peaked on confounded correlates, matches
in-distribution but loses AUC under shift (delta > 0). The gap is expected to
*track* correlate_inclusion across p_edge: largest at low density (correlates
abundant and predictive) and closing at high density (few non-causal features
left, so even vanilla selects indirect causes, which transport).

Usage:
    python experiments/causal_prior/synthetic/recovery_shift.py
    python experiments/causal_prior/synthetic/recovery_shift.py --smoke
    python experiments/causal_prior/synthetic/recovery_shift.py --p-edge 0.1,0.2 --n-seeds 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.data.synthetic_envs import make_environments              # noqa: E402
from src.causal_prior.q_sources import (                           # noqa: E402
    oracle_q, uniform_q, adversarial_q)
from src.causal_prior.priors import (                              # noqa: E402
    pc_stability_q, ges_stability_q, bootstrap_l1_q, iamb_stability_q)
from src.causal_prior.metrics import support_recovery_metrics, selectivity  # noqa: E402
from src.causal_prior.scorecard import fit_fr, score_auc           # noqa: E402
from experiments._io import new_run_dir                            # noqa: E402

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery_shift'
TEST_GAMMAS = (1.0, 0.0, -1.0)    # in-dist, correlates->noise, correlates reversed
CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'K', 'q_source', 'mu_rel',
    'support', 'k_actual', 'S_recall', 'S_precision', 'causal_precision',
    'correlate_inclusion', 'selectivity', 'auc_indist', 'test_gamma', 'auc', 'delta_auc',
]
SOURCES_ORDER = ('vanilla', 'oracle', 'ges', 'pc', 'iamb', 'gs', 'uniform',
                 'adversarial', 'bootstrap_l1')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--p', type=int, default=30, help='feature count')
    p.add_argument('--n', type=int, default=300, help='sample size')
    p.add_argument('--k-star', type=int, default=5, help='true sparsity')
    p.add_argument('--k', type=int, default=None,
                   help='sparsity budget (default 2 * k_star)')
    p.add_argument('--b-disc', type=int, default=50,
                   help='subsamples for pc/ges/bootstrap_l1')
    p.add_argument('--n-seeds', type=int, default=12)
    p.add_argument('--seed0', type=int, default=0, help='first seed')
    p.add_argument('--n-jobs', type=int, default=-1, help='joblib parallelism over cells')
    p.add_argument('--p-edge', type=str, default='0.1,0.15,0.2,0.3,0.5',
                   help='comma-separated DAG edge probabilities to sweep')
    p.add_argument('--mu-rel', type=str, default='0.5,2.0',
                   help='comma-separated prior strengths (mu / mu_scale)')
    p.add_argument('--smoke', action='store_true',
                   help='shrink the sweep for a quick end-to-end check')
    args = p.parse_args()
    if args.smoke:
        args.n_seeds, args.b_disc, args.p_edge, args.mu_rel = 2, 10, '0.2', '2.0'
    if args.k is None:
        args.k = 2 * args.k_star
    args.p_edge = tuple(float(x) for x in args.p_edge.split(','))
    args.mu_rel = tuple(float(x) for x in args.mu_rel.split(','))
    return args


def _support(fr):
    b = fr.betas_[0]
    return sorted(int(j) for j in np.where(np.abs(b) > 0)[0])


def build_q_sources(bundle, train, B, rng):
    """The q catalog, learned on the train environment only."""
    p, S_star, conf = bundle.p, sorted(bundle.S_star), sorted(bundle.confounded)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_pc = pc_stability_q(train.X, train.y_continuous, B=B,
                              rng=np.random.default_rng(int(rng.integers(2**31))))
        q_ges, _ = ges_stability_q(train.X, train.y_continuous, B=B,
                                   rng=np.random.default_rng(int(rng.integers(2**31))))
        q_l1 = bootstrap_l1_q(train.X, train.y, B=B,
                              rng=np.random.default_rng(int(rng.integers(2**31))))
        # mb-local learners: the best recoverers in 4.1, now also run through transport
        q_iamb = iamb_stability_q(train.X, train.y_continuous, B=B, method='iamb',
                                  rng=np.random.default_rng(int(rng.integers(2**31))))
        q_gs = iamb_stability_q(train.X, train.y_continuous, B=B, method='gs',
                                rng=np.random.default_rng(int(rng.integers(2**31))))
    return {
        'oracle':       oracle_q(p, S_star, sigma=0.0),
        'uniform':      uniform_q(p, 0.5),
        'adversarial':  adversarial_q(p, conf),
        'pc':           q_pc,
        'ges':          q_ges,
        'iamb':         q_iamb,
        'gs':           q_gs,
        'bootstrap_l1': q_l1,
    }


def _rows(args, seed, p_edge, q_source, mu_rel, support, m, sel, aucs):
    auc_indist = aucs.get(1.0, float('nan'))
    base = {'seed': seed, 'p': args.p, 'n': args.n, 'k_star': args.k_star,
            'p_edge': p_edge, 'K': args.k, 'q_source': q_source, 'mu_rel': mu_rel,
            'support': json.dumps(support), 'k_actual': m['k_actual'],
            'S_recall': m['S_recall'], 'S_precision': m['S_precision'],
            'causal_precision': m['causal_precision'],
            'correlate_inclusion': m['correlate_inclusion'], 'selectivity': sel}
    return [{**base, 'auc_indist': auc_indist, 'test_gamma': g,
             'auc': aucs[g], 'delta_auc': auc_indist - aucs[g]} for g in TEST_GAMMAS]


def run_cell(p_edge, seed, args):
    """Self-contained: build the SCM + environments, discover q on train, fit + score."""
    try:
        bundle = make_environments(args.p, args.n, args.k_star, p_edge,
                                   (1.0,) + TEST_GAMMAS, seed=seed)
    except RuntimeError:
        return []   # no confounding found at this (p_edge, seed); skip the cell
    train = bundle.environments[0]
    tests = {g: env for g, env in zip(TEST_GAMMAS, bundle.environments[1:])}
    S_set, C_set = set(bundle.S_star), set(bundle.confounded)
    causes, correlates = bundle.all_causes, bundle.correlates
    mu_scale = float(np.median(0.5 * np.abs(train.X.T @ train.y)))
    rng = np.random.default_rng(seed)
    sources = build_q_sources(bundle, train, args.b_disc, rng)

    def metrics_and_aucs(fr):
        sup = _support(fr)
        m = support_recovery_metrics(sup, S_set, C_set, causes=causes, correlates=correlates)
        aucs = {g: score_auc(fr, env.X, env.y) for g, env in tests.items()}
        return sup, m, aucs

    rows: list[dict] = []
    fr0 = fit_fr(train.X, train.y, args.k, 0.0, None)
    sup0, m0, aucs0 = metrics_and_aucs(fr0)
    rows += _rows(args, seed, p_edge, 'vanilla', 0.0, sup0, m0, float('nan'), aucs0)
    for q_source, q in sources.items():
        sel = selectivity(q, S_set, C_set)
        for mu_rel in args.mu_rel:
            fr = fit_fr(train.X, train.y, args.k, mu_rel * mu_scale, q)
            sup, m, aucs = metrics_and_aucs(fr)
            rows += _rows(args, seed, p_edge, q_source, mu_rel, sup, m, sel, aucs)
    return rows


def _agg(rows, p_edge, src, mu_sel, gamma, col):
    sel = [r for r in rows if r['p_edge'] == p_edge and r['q_source'] == src
           and r['mu_rel'] == mu_sel and r['test_gamma'] == gamma]
    return float(np.nanmean([r[col] for r in sel])) if sel else float('nan')


def main():
    args = parse_args()
    last = args.seed0 + args.n_seeds - 1
    cells = [(pe, s) for pe in args.p_edge for s in range(args.seed0, args.seed0 + args.n_seeds)]
    print(f'env-shift sweep: p={args.p} n={args.n} k*={args.k_star} K={args.k} '
          f'B_disc={args.b_disc} mu_rel={args.mu_rel} p_edge={args.p_edge} '
          f'seeds={args.seed0}..{last} ({len(cells)} cells, n_jobs={args.n_jobs})', flush=True)
    t = time.time()
    results = Parallel(n_jobs=args.n_jobs)(delayed(run_cell)(pe, s, args) for pe, s in cells)
    all_rows = [r for cell in results for r in cell]
    print(f'done in {time.time()-t:.0f}s, {len(all_rows)} rows', flush=True)

    suffix = (f'shift_p{args.p}_n{args.n}_k{args.k_star}_K{args.k}'
              f'_seeds{args.seed0}-{last}' + ('_smoke' if args.smoke else ''))
    output_dir = new_run_dir(OUT_DIR / suffix, vars(args))
    with (output_dir / 'shift.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)

    # headline: transport gap under the reversed-correlate env, at the strongest prior
    top = max(args.mu_rel)
    print(f'\nmean delta_auc @ gamma=-1, mu_rel={top} (transport gap), by p_edge x source:',
          flush=True)
    hdr = '  '.join(f'pe={pe:<4g}' for pe in args.p_edge)
    print(f'{"q_source":>13s} | {hdr}    (corr_incl @ pe_low->high)', flush=True)
    for src in SOURCES_ORDER:
        mu_sel = 0.0 if src == 'vanilla' else top
        gaps = [_agg(all_rows, pe, src, mu_sel, -1.0, 'delta_auc') for pe in args.p_edge]
        cis = [_agg(all_rows, pe, src, mu_sel, 1.0, 'correlate_inclusion') for pe in args.p_edge]
        gstr = '  '.join(f'{g:+.3f}' for g in gaps)
        cstr = '/'.join(f'{c:.2f}' for c in cis)
        print(f'{src:>13s} | {gstr}    ({cstr})', flush=True)
    print(f'Done. Results in {output_dir}/', flush=True)


if __name__ == '__main__':
    main()
