"""Environment-shift experiment: does the causal prior buy out-of-environment
transport that a merely-selective predictive prior does not? (the §6.1 -> §6 gap).

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
baseline is computed once and shared. K = 2 k* (the §6.1 anchor): the K-ablation
there shows the prior only bites once there are spare slots beyond k* for FR to
otherwise fill with predictive correlates.

Discriminating quantity: the transport gap
    delta_auc(gamma) = auc(in-distribution gamma=1) - auc(gamma).
Causal/selective-on-S* priors (oracle, ges, pc) keep the support causal and
transport (delta ~0); bootstrap_l1, peaked on confounded correlates, matches
in-distribution but loses AUC under shift (delta > 0). The gap is expected to
*track* correlate_inclusion across p_edge: largest at low density (correlates
abundant and predictive) and closing at high density (few non-causal features
left, so even vanilla selects indirect causes, which transport).
"""
from __future__ import annotations

import csv
import json
import os
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
    pc_stability_q, ges_stability_q, bootstrap_l1_q)
from src.causal_prior.metrics import support_recovery_metrics, selectivity  # noqa: E402

from sklearn.metrics import roc_auc_score                          # noqa: E402

P = int(os.environ.get('P', '30'))
N = int(os.environ.get('N', '300'))
K_STAR = int(os.environ.get('K_STAR', '5'))
K = int(os.environ.get('K', str(2 * K_STAR)))     # sparsity budget (default 2 k*, the §6.1 anchor)
B_DISC = int(os.environ.get('B_DISC', '50'))      # subsamples for pc/ges/bootstrap_l1
N_SEEDS = int(os.environ.get('N_SEEDS', '12'))
SEED0 = int(os.environ.get('SEED0', '0'))
N_JOBS = int(os.environ.get('N_JOBS', '-1'))
# informed by §6.1: confounder avoidance (correlate_inclusion) is a low-density
# effect that closes by p_edge>=0.3; sweep across the arc to show transport tracks it.
P_EDGE_GRID = tuple(float(x) for x in os.environ.get('P_EDGE', '0.1,0.15,0.2,0.3,0.5').split(','))
MU_REL_GRID = tuple(float(x) for x in os.environ.get('MU_REL', '0.5,2.0').split(','))
TEST_GAMMAS = (1.0, 0.0, -1.0)    # in-dist, correlates->noise, correlates reversed

OUT_DIR = ROOT / 'results' / 'causal_prior' / 'synthetic' / 'recovery_shift'
CSV_FIELDS = [
    'seed', 'p', 'n', 'k_star', 'p_edge', 'K', 'q_source', 'mu_rel',
    'support', 'k_actual', 'S_recall', 'S_precision', 'causal_precision',
    'correlate_inclusion', 'selectivity', 'auc_indist', 'test_gamma', 'auc', 'delta_auc',
]
SOURCES_ORDER = ('vanilla', 'oracle', 'ges', 'pc', 'uniform', 'adversarial', 'bootstrap_l1')


def _fit_fr(X, y, k, mu, q):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        from fasterrisk.wrapper import FasterRisk
        fr = FasterRisk(k=k, mu=float(mu), freq=None if q is None else q.astype(float))
        fr.fit(X, y)
    return fr


def _support(fr):
    b = fr.betas_[0]
    return sorted(int(j) for j in np.where(np.abs(b) > 0)[0])


def _auc(fr, X, y_signed):
    yb = (y_signed > 0).astype(int)
    if yb.sum() in (0, len(yb)):
        return float('nan')
    p = np.clip(fr.predict_proba(X), 1e-7, 1 - 1e-7)
    return float(roc_auc_score(yb, p))


def build_q_sources(bundle, train, B, rng):
    """The §6.1 catalog, learned on the train environment only."""
    p, S_star, conf = bundle.p, sorted(bundle.S_star), sorted(bundle.confounded)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        q_pc = pc_stability_q(train.X, train.y_continuous, B=B,
                              rng=np.random.default_rng(int(rng.integers(2**31))))
        q_ges, _ = ges_stability_q(train.X, train.y_continuous, B=B,
                                   rng=np.random.default_rng(int(rng.integers(2**31))))
        q_l1 = bootstrap_l1_q(train.X, train.y, B=B,
                              rng=np.random.default_rng(int(rng.integers(2**31))))
    return {
        'oracle':       oracle_q(p, S_star, sigma=0.0),
        'uniform':      uniform_q(p, 0.5),
        'adversarial':  adversarial_q(p, conf),
        'pc':           q_pc,
        'ges':          q_ges,
        'bootstrap_l1': q_l1,
    }


def _rows(seed, p_edge, q_source, mu_rel, support, m, sel, aucs):
    auc_indist = aucs.get(1.0, float('nan'))
    base = {'seed': seed, 'p': P, 'n': N, 'k_star': K_STAR, 'p_edge': p_edge, 'K': K,
            'q_source': q_source, 'mu_rel': mu_rel, 'support': json.dumps(support),
            'k_actual': m['k_actual'], 'S_recall': m['S_recall'],
            'S_precision': m['S_precision'], 'causal_precision': m['causal_precision'],
            'correlate_inclusion': m['correlate_inclusion'], 'selectivity': sel}
    return [{**base, 'auc_indist': auc_indist, 'test_gamma': g,
             'auc': aucs[g], 'delta_auc': auc_indist - aucs[g]} for g in TEST_GAMMAS]


def run_cell(p_edge: float, seed: int) -> list[dict]:
    """Self-contained: build the SCM + environments, discover q on train, fit + score."""
    try:
        bundle = make_environments(P, N, K_STAR, p_edge, (1.0,) + TEST_GAMMAS, seed=seed)
    except RuntimeError:
        return []   # no confounding found at this (p_edge, seed); skip the cell
    train = bundle.environments[0]
    tests = {g: env for g, env in zip(TEST_GAMMAS, bundle.environments[1:])}
    S_set, C_set = set(bundle.S_star), set(bundle.confounded)
    causes, correlates = bundle.all_causes, bundle.correlates
    mu_scale = float(np.median(0.5 * np.abs(train.X.T @ train.y)))
    rng = np.random.default_rng(seed)
    sources = build_q_sources(bundle, train, B_DISC, rng)

    def metrics_and_aucs(fr):
        sup = _support(fr)
        m = support_recovery_metrics(sup, S_set, C_set, causes=causes, correlates=correlates)
        aucs = {g: _auc(fr, env.X, env.y) for g, env in tests.items()}
        return sup, m, aucs

    rows: list[dict] = []
    fr0 = _fit_fr(train.X, train.y, K, 0.0, None)         # mu=0 shared vanilla baseline
    sup0, m0, aucs0 = metrics_and_aucs(fr0)
    rows += _rows(seed, p_edge, 'vanilla', 0.0, sup0, m0, float('nan'), aucs0)
    for q_source, q in sources.items():
        sel = selectivity(q, S_set, C_set)
        for mu_rel in MU_REL_GRID:
            fr = _fit_fr(train.X, train.y, K, mu_rel * mu_scale, q)
            sup, m, aucs = metrics_and_aucs(fr)
            rows += _rows(seed, p_edge, q_source, mu_rel, sup, m, sel, aucs)
    return rows


def _agg(rows, p_edge, src, mu_sel, gamma, col):
    sel = [r for r in rows if r['p_edge'] == p_edge and r['q_source'] == src
           and r['mu_rel'] == mu_sel and r['test_gamma'] == gamma]
    return float(np.nanmean([r[col] for r in sel])) if sel else float('nan')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(pe, s) for pe in P_EDGE_GRID for s in range(SEED0, SEED0 + N_SEEDS)]
    print(f'env-shift sweep: p={P} n={N} k*={K_STAR} K={K} B_disc={B_DISC} '
          f'mu_rel={MU_REL_GRID} p_edge={P_EDGE_GRID} seeds={SEED0}..{SEED0+N_SEEDS-1} '
          f'({len(cells)} cells, n_jobs={N_JOBS})', flush=True)
    t = time.time()
    results = Parallel(n_jobs=N_JOBS)(delayed(run_cell)(pe, s) for pe, s in cells)
    all_rows = [r for cell in results for r in cell]
    print(f'done in {time.time()-t:.0f}s, {len(all_rows)} rows', flush=True)

    out_path = OUT_DIR / (f'shift_sweep_p{P}_n{N}_k{K_STAR}_K{K}'
                          f'_seeds{SEED0}-{SEED0+N_SEEDS-1}.csv')
    with out_path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
    print(f'wrote {out_path}', flush=True)

    # headline table: transport gap under the reversed-correlate env, at the strongest prior
    top = max(MU_REL_GRID)
    print(f'\n=== mean delta_auc @ gamma=-1, mu_rel={top} (transport gap), by p_edge x source ===',
          flush=True)
    hdr = '  '.join(f'pe={pe:<4g}' for pe in P_EDGE_GRID)
    print(f'{"q_source":>13s} | {hdr}    (corr_incl @ pe_low->high)', flush=True)
    for src in SOURCES_ORDER:
        mu_sel = 0.0 if src == 'vanilla' else top
        gaps = [_agg(all_rows, pe, src, mu_sel, -1.0, 'delta_auc') for pe in P_EDGE_GRID]
        cis = [_agg(all_rows, pe, src, mu_sel, 1.0, 'correlate_inclusion') for pe in P_EDGE_GRID]
        gstr = '  '.join(f'{g:+.3f}' for g in gaps)
        cstr = '/'.join(f'{c:.2f}' for c in cis)
        print(f'{src:>13s} | {gstr}    ({cstr})', flush=True)


if __name__ == '__main__':
    main()
