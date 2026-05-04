"""
A/B comparison: Gaussian noise (old) vs FLXMRglm binomial (new) on synthetic binary data.
Results saved to results/synthetic_validation_<timestamp>/.

Usage:
    python experiments/mixed_cmm/synthetic/run_synthetic_validation.py
    python experiments/mixed_cmm/synthetic/run_synthetic_validation.py --n_obs 8 --n_seeds 20 --n_bootstrap 30
    python experiments/mixed_cmm/synthetic/run_synthetic_validation.py --smoke       # quick bootstrap sanity check
    python experiments/mixed_cmm/synthetic/run_synthetic_validation.py --single_run  # no bootstrap, checks if edges findable at all
"""
import sys
import os
import argparse
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

warnings.filterwarnings('ignore')

import pandas as pd
from src_tb.data.synthetic import SyntheticData
from src_tb.causal_recovery.evaluation import eval_recovery
from src_tb.causal_recovery.cmm_utils import run_cmm, bootstrap_cmm, get_stable_edges, edge_stability

REPO_ROOT = Path(__file__).resolve().parents[3]


def run_experiment(n_obs: int, n_seeds: int, n_bootstrap: int, threshold: float, use_logistic: bool, output_path: str = None) -> list[dict]:
    label = 'logistic' if use_logistic else 'gaussian'
    records = []
    for seed in range(n_seeds):
        print(f"  [{label}] seed {seed + 1}/{n_seeds}", flush=True)
        data = SyntheticData(n_obs, seed=seed)
        cmm_list = bootstrap_cmm(data.X, data.forbidden_edges, n_runs=n_bootstrap,
                                use_logistic=use_logistic, seed=seed)
        df_freq = edge_stability(cmm_list, data.features)
        freq_map = {(r['source'], r['target']): r['frequency'] for _, r in df_freq.iterrows()}

        stable = get_stable_edges(cmm_list, data.features, threshold=threshold)
        stable_set = set(zip(stable['source'], stable['target']))
        stable_bb = {(s, t) for s, t in stable_set if s != 'Y' and t != 'Y'}
        stable_bc = {(s, t) for s, t in stable_set if t == 'Y'}
        pr_d,  re_d,  f1_d  = eval_recovery(stable_bc, data.true_bin_to_cont)
        pr_bb, re_bb, f1_bb = eval_recovery(stable_bb, data.true_bin_to_bin)

        bb_freq = sum(freq_map.get(e, 0) for e in data.true_bin_to_bin)  / max(len(data.true_bin_to_bin), 1)
        d_freq  = sum(freq_map.get(e, 0) for e in data.true_bin_to_cont) / max(len(data.true_bin_to_cont), 1)

        metrics = {
            'seed': seed,
            'bin_cont_F1': f1_d,  'bin_cont_P': pr_d,  'bin_cont_R': re_d,  'bin_cont_freq': d_freq,
            'bin_bin_F1':  f1_bb, 'bin_bin_P':  pr_bb, 'bin_bin_R':  re_bb, 'bin_bin_freq':  bb_freq,
        }
        records.append(metrics)
        print(f"    bin_cont_F1={f1_d:.3f}  bin_bin_F1={f1_bb:.3f}  |  bin_bin_freq={bb_freq:.2f}", flush=True)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame([metrics]).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    return records


def run_single(n_obs: int, n_seeds: int, use_logistic: bool) -> list[dict]:
    """Single run per seed. No bootstrap, no stability threshold. Checks if edges are findable at all."""
    label = 'logistic' if use_logistic else 'gaussian'
    records = []
    for seed in range(n_seeds):
        print(f"  [{label}] seed {seed + 1}/{n_seeds}", flush=True)
        data = SyntheticData(n_obs, seed=seed)
        cmm = run_cmm(data.X, data.forbidden_edges, use_logistic=use_logistic, noise_seed=seed)
        recovered = {(data.features[i], data.features[j]) for i, j in cmm.dag.edges()}
        rec_bb = {(s, t) for s, t in recovered if s != 'Y' and t != 'Y'}
        rec_bc = {(s, t) for s, t in recovered if t == 'Y'}
        pr_d,  re_d,  f1_d  = eval_recovery(rec_bc, data.true_bin_to_cont)
        pr_bb, re_bb, f1_bb = eval_recovery(rec_bb, data.true_bin_to_bin)
        metrics = {
            'seed': seed,
            'bin_cont_F1': f1_d, 'bin_bin_F1': f1_bb,
        }
        records.append(metrics)
        print(f"    bin_cont_F1={f1_d:.3f}  bin_bin_F1={f1_bb:.3f}", flush=True)
    return records


def print_summary(records_old: list[dict], records_new: list[dict]):
    df_old = pd.DataFrame(records_old)
    df_new = pd.DataFrame(records_new)
    f1_cols   = ['bin_cont_F1', 'bin_bin_F1']
    freq_cols = ['bin_cont_freq', 'bin_bin_freq']
    print("\nSummary (F1 on stable edges)")
    print("           " + "  ".join(f"{c:>12}" for c in f1_cols))
    print("Gaussian:  " + "  ".join(f"{df_old[c].mean():>12.3f}" for c in f1_cols))
    print("Logistic:  " + "  ".join(f"{df_new[c].mean():>12.3f}" for c in f1_cols))
    if all(c in df_old.columns for c in freq_cols):
        print("\nMean bootstrap frequency of true edges")
        print("           " + "  ".join(f"{c:>12}" for c in freq_cols))
        print("Gaussian:  " + "  ".join(f"{df_old[c].mean():>12.3f}" for c in freq_cols))
        print("Logistic:  " + "  ".join(f"{df_new[c].mean():>12.3f}" for c in freq_cols))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_obs', type=int, default=8)
    parser.add_argument('--n_seeds',     type=int, default=20)
    parser.add_argument('--n_bootstrap', type=int, default=30)
    parser.add_argument('--threshold',   type=float, default=0.5)
    parser.add_argument('--smoke',       action='store_true', help='quick bootstrap sanity check (overrides defaults)')
    parser.add_argument('--single_run',  action='store_true', help='no bootstrap, single CMM fit per seed')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke:
        args.n_obs, args.n_seeds, args.n_bootstrap = 4, 2, 3
        print("Smoke test (n_obs=4, n_seeds=2, n_bootstrap=3)", flush=True)

    if args.single_run:
        print(f"Single run (n_obs={args.n_obs}, n_seeds={args.n_seeds}, no bootstrap)", flush=True)
        records_old = run_single(args.n_obs, args.n_seeds, use_logistic=False)
        records_new = run_single(args.n_obs, args.n_seeds, use_logistic=True)
        print_summary(records_old, records_new)
        return

    output_dir = REPO_ROOT / 'results' / f'synthetic_validation_{datetime.now().strftime("%Y%m%d_%H%M")}'
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'config.txt', 'w') as f:
        f.write(f"n_obs={args.n_obs}\nn_seeds={args.n_seeds}\nn_bootstrap={args.n_bootstrap}\nthreshold={args.threshold}\n")

    print("Gaussian noise (old model)", flush=True)
    records_old = run_experiment(args.n_obs, args.n_seeds, args.n_bootstrap, args.threshold, use_logistic=False,
                                output_path=str(output_dir / 'results_gaussian.csv'))

    print("Logistic FLXMRglm (new model)", flush=True)
    records_new = run_experiment(args.n_obs, args.n_seeds, args.n_bootstrap, args.threshold, use_logistic=True,
                                output_path=str(output_dir / 'results_logistic.csv'))

    print(f"\nResults saved to {output_dir}/", flush=True)
    print_summary(records_old, records_new)


if __name__ == '__main__':
    main()
