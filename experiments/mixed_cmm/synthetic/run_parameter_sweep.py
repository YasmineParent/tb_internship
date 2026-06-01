"""
Parameter sweep validating CMM with logistic extension on mixed binary/continuous data.
Replicates Figure D.2 from the CMM paper, extended to binary observed variables.

Varies 5 parameters (NX, pG, S, NZ, pZ) while fixing others at default.
Compares Gaussian CMM (old) vs Logistic CMM (new).
Results saved incrementally to CSV. Plot in notebook from saved data.

Usage:
    python experiments/mixed_cmm/synthetic/run_parameter_sweep.py
    python experiments/mixed_cmm/synthetic/run_parameter_sweep.py --n_seeds 3
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
from rpy2.rinterface_lib.embedded import RRuntimeError
from experiments.mixed_cmm.synthetic.config import DEFAULTS, SWEEPS, GRAPH_METRICS
from src_tb.data.synthetic import SyntheticData
from src_tb.causal_discovery.cmm_utils import run_cmm
from src_tb.causal_discovery.evaluation import score_recovered, serialize_edges

REPO_ROOT = Path(__file__).resolve().parents[3]

CSV_METRIC_KEYS = list(GRAPH_METRICS) + [
    f'{stat}_{kind}'
    for stat in ('f1', 'precision', 'recall') for kind in ('bin_bin', 'bin_cont')
]
NAN_METRICS = {k: float('nan') for k in CSV_METRIC_KEYS}


def fit_one(data: SyntheticData, use_logistic: bool, seed: int = 0) -> set | None:
    """Fit CMM and return the recovered edge set, or None on R failure."""
    label = 'logistic' if use_logistic else 'gaussian'
    try:
        cmm = run_cmm(data.X, data.forbidden_edges, use_logistic=use_logistic, noise_seed=seed)
    except RRuntimeError as e:
        print(f"    R error ({label}): {e}", flush=True)
        return None
    return {(data.features[i], data.features[j]) for i, j in cmm.dag.edges()}


def metrics_for_csv(recovered: set | None, data: SyntheticData) -> dict:
    if recovered is None:
        return dict(NAN_METRICS)
    full = score_recovered(recovered, data)
    return {k: full.get(k, float('nan')) for k in CSV_METRIC_KEYS}


def sweep_param(param: str, values: list, n_seeds: int, output_path: str) -> None:
    for val in values:
        kwargs = {**DEFAULTS, param: val}
        for seed in range(n_seeds):
            print(f"  {param}={val}  seed {seed + 1}/{n_seeds}", flush=True)
            data = SyntheticData(**kwargs, seed=seed)
            for use_logistic in [False, True]:
                label = 'logistic' if use_logistic else 'gaussian'
                recovered = fit_one(data, use_logistic, seed=seed)
                metrics = metrics_for_csv(recovered, data)
                record = {
                    'param': param, 'value': val, 'seed': seed, 'model': label,
                    'edges': serialize_edges(recovered) if recovered is not None else '',
                    **metrics,
                }
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                pd.DataFrame([record]).to_csv(output_path, mode='a',
                    header=not os.path.exists(output_path), index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    args = parser.parse_args()

    output_dir = REPO_ROOT / 'results' / f'sweep_{datetime.now().strftime("%Y%m%d_%H%M")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / 'results.csv')

    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'n_seeds': args.n_seeds, 'defaults': DEFAULTS, 'sweeps': SWEEPS}, f, indent=2)

    for param, values in SWEEPS.items():
        print(f"\nSweeping {param}", flush=True)
        sweep_param(param, values, args.n_seeds, output_path)

    print(f"\nDone. Results in {output_dir}/results.csv", flush=True)


if __name__ == '__main__':
    main()
