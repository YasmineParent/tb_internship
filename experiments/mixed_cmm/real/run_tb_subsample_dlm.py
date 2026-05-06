"""
Stability selection (subsampling) for CMM-logistic on real TB data (delamanid only).
Builds [dlm_mic, ...mutations] (prevalence-filtered), runs subsample_cmm with logistic
driver, and saves per-edge selection frequencies. Threshold and stable-graph extraction
are downstream concerns.

Usage:
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py
    python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --n_runs 200 --min_prev 0.1
"""
import sys
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import numpy as np
from src_tb.data.load_tb import load_tb_data, prevalence_filter
from src_tb.causal_recovery.cmm_utils import subsample_cmm, edge_stability

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / 'data' / 'real' / 'processed' / 'tb_pheno_geno_clean.csv'
MIC_COL = 'dlm_mic'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_runs', type=int, default=100, help='stability-selection iterations')
    parser.add_argument('--subsample_frac', type=float, default=0.8, help='fraction of rows per run')
    parser.add_argument('--max_parents', type=int, default=4, help='cap on in-degree per node')
    parser.add_argument('--k_max', type=int, default=5, help='max mixture components per node')
    parser.add_argument('--min_cluster_count', type=int, default=5, help='min positives per cluster for binary cols')
    parser.add_argument('--min_prev', type=float, default=0.05)
    parser.add_argument('--max_prev', type=float, default=0.98)
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    df, mutation_cols, _, _, _ = load_tb_data(str(DATA_PATH))
    n_before = len(df)
    df = df.dropna(subset=[MIC_COL]).reset_index(drop=True)
    print(f"isolates with {MIC_COL}: {len(df)} / {n_before}", flush=True)
    # MIC is measured on a 2-fold dilution series; log2 puts dilution steps on a uniform scale.
    df[MIC_COL] = np.log2(df[MIC_COL])
    keep = prevalence_filter(df, mutation_cols, min_prev=args.min_prev, max_prev=args.max_prev)
    features = [MIC_COL] + keep
    X = df[features].values
    # forbid MIC -> mutation: mutations cause MIC, never the reverse
    forbidden = {(0, j) for j in range(1, len(features))}
    col_threshold = args.min_cluster_count * args.k_max
    print(f"mutations after prevalence filter: {len(keep)}, X shape: {X.shape}", flush=True)
    print(f"k_max={args.k_max}, min_cluster_count={args.min_cluster_count}, col_threshold={col_threshold}", flush=True)
    sparse = [f for f in keep if df[f].sum() < col_threshold]
    print(f"mutations below col_threshold ({col_threshold}): {sparse}", flush=True)

    output_dir = REPO_ROOT / 'results' / f'tb_subsample_dlm_{datetime.now().strftime("%Y%m%d_%H%M")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({**vars(args), 'mic_col': MIC_COL, 'features': features}, f, indent=2)

    cmm_list, features_per_run = subsample_cmm(X, forbidden, n_runs=args.n_runs, use_logistic=True,
                                               max_parents=args.max_parents, k_max=args.k_max,
                                               min_cluster_count=args.min_cluster_count,
                                               subsample_frac=args.subsample_frac,
                                               features=features, seed=args.seed)
    df_stability = edge_stability(cmm_list, features_per_run).sort_values('frequency', ascending=False)
    df_stability.to_csv(output_dir / 'edge_stability.csv', index=False)

    print(f"Done. Results in {output_dir}/", flush=True)


if __name__ == '__main__':
    main()
