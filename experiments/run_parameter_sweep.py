"""
Parameter sweep validating CMM with logistic extension on mixed binary/continuous data.
Replicates Figure D.2 from the CMM paper, extended to binary observed variables.

Varies 5 parameters (NX, pG, S, NZ, pZ) while fixing others at default.
Compares Gaussian CMM (old) vs Logistic CMM (new).
Results saved incrementally to CSV — plot in notebook from saved data.

Usage:
    python experiments/run_parameter_sweep.py
    python experiments/run_parameter_sweep.py --n_seeds 3
"""
import sys
import os
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings('ignore')

import pandas as pd
from src_tb.config import DEFAULTS, SWEEPS, GRAPH_METRICS
from src_tb.data.synthetic import SyntheticData
from src_tb.causal_recovery.cmm_utils import run_cmm
from src_tb.causal_recovery.evaluation import compute_graph_metrics, eval_recovery

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_one(data: SyntheticData, use_logistic: bool) -> dict:
    cmm = run_cmm(data.X, data.forbidden_edges, use_logistic=use_logistic)
    recovered = {(data.features[i], data.features[j]) for i, j in cmm.dag.edges()}
    graph = compute_graph_metrics(recovered, data.true_edges, data.features)
    pr_bb, re_bb, f1_bb = eval_recovery(recovered, data.true_bin_to_bin)
    pr_bc, re_bc, f1_bc = eval_recovery(recovered, data.true_bin_to_cont)
    return {m: graph.get(m, float('nan')) for m in GRAPH_METRICS} | {
        'f1_bin_bin':        f1_bb, 'precision_bin_bin':  pr_bb, 'recall_bin_bin':  re_bb,
        'f1_bin_cont':       f1_bc, 'precision_bin_cont': pr_bc, 'recall_bin_cont': re_bc,
    }


def sweep_param(param: str, values: list, n_seeds: int, output_path: str) -> list:
    records = []
    for val in values:
        kwargs = {**DEFAULTS, param: val}
        for seed in range(n_seeds):
            print(f"  {param}={val}  seed {seed + 1}/{n_seeds}", flush=True)
            data = SyntheticData(**kwargs, seed=seed)
            for use_logistic in [False, True]:
                label = 'logistic' if use_logistic else 'gaussian'
                metrics = run_one(data, use_logistic)
                record = {'param': param, 'value': val, 'seed': seed, 'model': label, **metrics}
                records.append(record)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                pd.DataFrame([record]).to_csv(output_path, mode='a',
                    header=not os.path.exists(output_path), index=False)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=3)
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
