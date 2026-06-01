"""
Baseline comparison: CMM-logistic vs binary-adapted causal discovery baselines.
Sweeps the same parameter grid as run_parameter_sweep.py to mirror Fig D.2 of the CMM paper,
but on mixed binary/continuous data.

Methods:
    cmm_logistic   our method (FLXMRglm binomial)
    pc_pillai      pgmpy PC with Pillai trace CI test (mixed-data adapted)
    ges_cg         pgmpy GES with conditional-Gaussian BIC score (mixed-data adapted)
    empty          null floor (no edges)

Metrics per (param, value, seed, method): all GRAPH_METRICS (sd, sc, shd, fpr, tpr, f1, tp,
mcc, shd-nm), plus precision/recall/F1 split by edge class (bin->bin, bin->cont).

Usage:
    python experiments/mixed_cmm/synthetic/run_baselines_sweep.py
    python experiments/mixed_cmm/synthetic/run_baselines_sweep.py --n_seeds 10
    python experiments/mixed_cmm/synthetic/run_baselines_sweep.py --smoke
    python experiments/mixed_cmm/synthetic/run_baselines_sweep.py --methods cmm_logistic pc_pillai
"""
import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import pandas as pd
from pgmpy.estimators import PC, GES, ExpertKnowledge

from experiments.mixed_cmm.synthetic.config import DEFAULTS, SWEEPS, GRAPH_METRICS
from src_tb.data.synthetic import SyntheticData
from src_tb.causal_discovery.cmm_utils import run_cmm
from src_tb.causal_discovery.evaluation import score_recovered, serialize_edges

REPO_ROOT = Path(__file__).resolve().parents[3]
METHODS = ['cmm_logistic', 'pc_pillai', 'ges_cg', 'empty']

CSV_METRIC_KEYS = list(GRAPH_METRICS) + [
    f'{stat}_{kind}'
    for stat in ('f1', 'precision', 'recall') for kind in ('bin_bin', 'bin_cont')
]
NAN_METRICS = {k: float('nan') for k in CSV_METRIC_KEYS}


def to_dataframe(data: SyntheticData) -> pd.DataFrame:
    """Build a typed DataFrame: binary columns as int, Y as float.
    pgmpy's mixed-data tests rely on dtypes to detect variable types."""
    df = pd.DataFrame(data.X.copy(), columns=data.features)
    for f in data.features[:-1]:
        df[f] = df[f].round().astype(int)
    df[data.features[-1]] = df[data.features[-1]].astype(float)
    return df


def named_forbidden(data: SyntheticData) -> set:
    return {(data.features[i], data.features[j]) for i, j in data.forbidden_edges}


def run_cmm_logistic(data: SyntheticData, seed: int) -> set:
    cmm = run_cmm(data.X, data.forbidden_edges, use_logistic=True, noise_seed=seed)
    return {(data.features[i], data.features[j]) for i, j in cmm.dag.edges()}


def run_pc_pillai(data: SyntheticData, seed: int) -> set:
    df = to_dataframe(data)
    # ExpertKnowledge with enforce_expert_knowledge=False (default) applies the forbidden
    # edges post-hoc, after orientation. Same effect as run_ges_cg's set difference.
    ek = ExpertKnowledge(forbidden_edges=list(named_forbidden(data)))
    est = PC(data=df)
    dag = est.estimate(
        variant='stable',
        ci_test='pillai',
        return_type='dag',
        expert_knowledge=ek,
        show_progress=False,
        n_jobs=1,
    )
    return set(dag.edges())


def run_ges_cg(data: SyntheticData, seed: int) -> set:
    df = to_dataframe(data)
    est = GES(data=df)
    pdag = est.estimate(scoring_method='bic-cg')
    dag = pdag.to_dag()
    # GES has no native blacklist, so post-hoc filter the forbidden Y -> mut edges.
    return set(dag.edges()) - named_forbidden(data)


def run_empty(data: SyntheticData, seed: int) -> set:
    return set()


METHOD_FNS = {
    'cmm_logistic': run_cmm_logistic,
    'pc_pillai':    run_pc_pillai,
    'ges_cg':       run_ges_cg,
    'empty':        run_empty,
}


def fit_method(method: str, data: SyntheticData, seed: int) -> set | None:
    """Fit one method and return the recovered edge set, or None on failure."""
    try:
        return METHOD_FNS[method](data, seed)
    except Exception as e:
        print(f"    {method} failed: {type(e).__name__}: {e}", flush=True)
        return None


def metrics_for_csv(recovered: set | None, data: SyntheticData) -> dict:
    if recovered is None:
        return dict(NAN_METRICS)
    full = score_recovered(recovered, data)
    return {k: full.get(k, float('nan')) for k in CSV_METRIC_KEYS}


def sweep_param(param: str, values: list, n_seeds: int, methods: list, output_path: str) -> None:
    for val in values:
        kwargs = {**DEFAULTS, param: val}
        for seed in range(n_seeds):
            data = SyntheticData(**kwargs, seed=seed)
            for method in methods:
                print(f"  {param}={val}  seed {seed + 1}/{n_seeds}  method={method}", flush=True)
                recovered = fit_method(method, data, seed)
                metrics = metrics_for_csv(recovered, data)
                record = {
                    'param': param, 'value': val, 'seed': seed, 'method': method,
                    'edges': serialize_edges(recovered) if recovered is not None else '',
                    **metrics,
                }
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                pd.DataFrame([record]).to_csv(
                    output_path, mode='a',
                    header=not os.path.exists(output_path), index=False,
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    parser.add_argument('--smoke', action='store_true', help='quick sanity check (n_seeds=2, single param)')
    args = parser.parse_args()

    if args.smoke:
        n_seeds = 2
        sweeps = {'n_obs': [4, 6]}
        print("Smoke test (n_seeds=2, n_obs=[4,6])", flush=True)
    else:
        n_seeds = args.n_seeds
        sweeps = SWEEPS

    output_dir = REPO_ROOT / 'results' / 'mixed_cmm' / f'baselines_{datetime.now().strftime("%Y%m%d_%H%M")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / 'results.csv')

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'n_seeds': n_seeds, 'methods': args.methods,
            'defaults': DEFAULTS, 'sweeps': sweeps,
        }, f, indent=2)

    for param, values in sweeps.items():
        print(f"\nSweeping {param}", flush=True)
        sweep_param(param, values, n_seeds, args.methods, output_path)

    print(f"\nDone. Results in {output_path}", flush=True)


if __name__ == '__main__':
    main()
