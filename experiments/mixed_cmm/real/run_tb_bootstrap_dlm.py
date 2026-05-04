"""
Bootstrap CMM-logistic on real TB data (delamanid only).
Builds [dlm_mic, ...mutations] (prevalence-filtered), runs bootstrap_cmm with logistic
driver, and saves edge stability + stable graph + bifxml.

Usage:
    python experiments/mixed_cmm/real/run_tb_bootstrap_dlm.py
    python experiments/mixed_cmm/real/run_tb_bootstrap_dlm.py --n_runs 50 --threshold 0.6
"""
import sys
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import pyagrum as gum
from src_tb.data.load_tb import load_tb_data, prevalence_filter
from src_tb.causal_recovery.cmm_utils import (
    bootstrap_cmm, edge_stability, get_stable_edges, build_stable_bn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
MIC_COL = 'dlm_mic'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=30, help='bootstrap iterations')
    parser.add_argument('--threshold', type=float, default=0.5, help='edge stability threshold')
    parser.add_argument('--min_prev', type=float, default=0.05)
    parser.add_argument('--max_prev', type=float, default=0.98)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    df, mutation_cols, _, _, _ = load_tb_data(str(DATA_PATH))
    keep = prevalence_filter(df, mutation_cols, min_prev=args.min_prev, max_prev=args.max_prev)
    features = [MIC_COL] + keep
    X = df[features].values
    forbidden = {(0, j) for j in range(1, len(features))}
    print(f"mutations after prevalence filter: {len(keep)}, X shape: {X.shape}", flush=True)

    output_dir = REPO_ROOT / 'results' / f'tb_bootstrap_dlm_{datetime.now().strftime("%Y%m%d_%H%M")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'mic_col': MIC_COL, 'features': features}, f, indent=2)

    cmm_list = bootstrap_cmm(X, forbidden, n_runs=args.n_runs, use_logistic=True, seed=args.seed)

    df_stability = edge_stability(cmm_list, features)
    df_stability.sort_values('frequency', ascending=False).to_csv(output_dir / 'edge_stability.csv', index=False)

    stable = get_stable_edges(cmm_list, features, threshold=args.threshold)
    stable.to_csv(output_dir / 'stable_edges.csv', index=False)

    bn = build_stable_bn(cmm_list, features, threshold=args.threshold, continuous_features=[MIC_COL])
    gum.saveBN(bn, str(output_dir / 'stable_graph.bifxml'))

    print(f"{len(stable)} stable edges at threshold {args.threshold}", flush=True)
    print(f"Done. Results in {output_dir}/", flush=True)


if __name__ == '__main__':
    main()
