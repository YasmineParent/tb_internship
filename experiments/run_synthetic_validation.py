"""
A/B comparison: Gaussian noise (old) vs FLXMRglm binomial (new) on synthetic binary data.
Results saved to results/synthetic_validation_<timestamp>/.

Usage:
    python experiments/run_synthetic_validation.py
    python experiments/run_synthetic_validation.py --n_mutations 8 --n_seeds 20 --n_bootstrap 30
    python experiments/run_synthetic_validation.py --smoke       # quick bootstrap sanity check
    python experiments/run_synthetic_validation.py --single_run  # no bootstrap, checks if edges findable at all
"""
import sys
import os
import argparse
import warnings
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'external', 'cmm'))

warnings.filterwarnings('ignore')

import pandas as pd
from src_tb.data.synthetic import SyntheticData
from src_tb.causal_recovery.evaluation import eval_recovery
from src_tb.causal_recovery.cmm_utils import run_cmm, bootstrap_cmm, get_stable_edges, edge_stability


def run_experiment(n_mutations: int, n_seeds: int, n_bootstrap: int, threshold: float, use_logistic: bool, output_path: str = None) -> list[dict]:
    label = 'logistic' if use_logistic else 'gaussian'
    records = []
    for seed in range(n_seeds):
        print(f"  [{label}] seed {seed + 1}/{n_seeds}", flush=True)
        data = SyntheticData(n_mutations, seed=seed)
        cmm_list = bootstrap_cmm(data.X, set(), n_runs=n_bootstrap, use_logistic=use_logistic)
        df_freq = edge_stability(cmm_list, data.features)
        freq_map = {(r['source'], r['target']): r['frequency'] for _, r in df_freq.iterrows()}

        stable = get_stable_edges(cmm_list, data.features, threshold=threshold)
        stable_set = set(zip(stable['source'], stable['target']))
        pr_d,  re_d,  f1_d  = eval_recovery(stable_set, data.true_direct)
        pr_bb, re_bb, f1_bb = eval_recovery(stable_set, data.true_bin_to_bin)
        pr_cc, re_cc, f1_cc = eval_recovery(stable_set, data.true_chain_cont)

        bb_freq = sum(freq_map.get(e, 0) for e in data.true_bin_to_bin) / max(len(data.true_bin_to_bin), 1)
        d_freq  = sum(freq_map.get(e, 0) for e in data.true_direct)     / max(len(data.true_direct), 1)
        cc_freq = sum(freq_map.get(e, 0) for e in data.true_chain_cont) / max(len(data.true_chain_cont), 1)

        metrics = {
            'seed': seed,
            'direct_F1': f1_d,   'bin_bin_F1': f1_bb,  'chain_F1': f1_cc,
            'direct_P': pr_d,    'bin_bin_P': pr_bb,    'chain_P': pr_cc,
            'direct_R': re_d,    'bin_bin_R': re_bb,    'chain_R': re_cc,
            'direct_freq': d_freq, 'bin_bin_freq': bb_freq, 'chain_freq': cc_freq,
        }
        records.append(metrics)
        print(f"    direct_F1={f1_d}  bin_bin_F1={f1_bb}  chain_F1={f1_cc}  |  bin_bin_freq={bb_freq:.2f}", flush=True)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pd.DataFrame([metrics]).to_csv(output_path, mode='a', header=not os.path.exists(output_path), index=False)
    return records


def run_single(n_mutations: int, n_seeds: int, use_logistic: bool) -> list[dict]:
    """Single run per seed — no bootstrap, no stability threshold. Checks if edges are findable at all."""
    label = 'logistic' if use_logistic else 'gaussian'
    records = []
    for seed in range(n_seeds):
        print(f"  [{label}] seed {seed + 1}/{n_seeds}", flush=True)
        data = SyntheticData(n_mutations, seed=seed)
        cmm = run_cmm(data.X, set(), use_logistic=use_logistic)
        recovered = {(data.features[i], data.features[j]) for i, j in cmm.dag.edges()}
        pr_d,  re_d,  f1_d  = eval_recovery(recovered, data.true_direct)
        pr_bb, re_bb, f1_bb = eval_recovery(recovered, data.true_bin_to_bin)
        pr_cc, re_cc, f1_cc = eval_recovery(recovered, data.true_chain_cont)
        metrics = {
            'seed': seed,
            'direct_F1': f1_d, 'bin_bin_F1': f1_bb, 'chain_F1': f1_cc,
        }
        records.append(metrics)
        print(f"    direct_F1={f1_d}  bin_bin_F1={f1_bb}  chain_F1={f1_cc}", flush=True)
    return records


def print_summary(records_old: list[dict], records_new: list[dict]):
    df_old = pd.DataFrame(records_old)
    df_new = pd.DataFrame(records_new)
    f1_cols   = ['direct_F1', 'bin_bin_F1', 'chain_F1']
    freq_cols = ['direct_freq', 'bin_bin_freq', 'chain_freq']
    print("\nSummary (F1 on stable edges)")
    print("           " + "  ".join(f"{c:>12}" for c in f1_cols))
    print("Gaussian:  " + "  ".join(f"{df_old[c].mean():>12.3f}" for c in f1_cols))
    print("Logistic:  " + "  ".join(f"{df_new[c].mean():>12.3f}" for c in f1_cols))
    if all(c in df_old.columns for c in freq_cols):
        print("\nMean bootstrap frequency of true edges")
        print("           " + "  ".join(f"{c:>12}" for c in freq_cols))
        print("Gaussian:  " + "  ".join(f"{df_old[c].mean():>12.3f}" for c in freq_cols))
        print("Logistic:  " + "  ".join(f"{df_new[c].mean():>12.3f}" for c in freq_cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mutations', type=int, default=8)
    parser.add_argument('--n_seeds',     type=int, default=20)
    parser.add_argument('--n_bootstrap', type=int, default=30)
    parser.add_argument('--threshold',   type=float, default=0.5)
    args = parser.parse_args()

    output_dir = os.path.abspath(f'results/synthetic_validation_{datetime.now().strftime("%Y%m%d_%H%M")}')

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.txt'), 'w') as f:
        f.write(f"n_mutations={args.n_mutations}\nn_seeds={args.n_seeds}\nn_bootstrap={args.n_bootstrap}\nthreshold={args.threshold}\n")

    print("Gaussian noise (old model)", flush=True)
    records_old = run_experiment(args.n_mutations, args.n_seeds, args.n_bootstrap, args.threshold, use_logistic=False,
                                 output_path=os.path.join(output_dir, 'results_gaussian.csv'))

    print("Logistic FLXMRglm (new model)", flush=True)
    records_new = run_experiment(args.n_mutations, args.n_seeds, args.n_bootstrap, args.threshold, use_logistic=True,
                                 output_path=os.path.join(output_dir, 'results_logistic.csv'))

    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults saved to {output_dir}/", flush=True)
    print_summary(records_old, records_new)


if __name__ == '__main__':
    if '--smoke' in sys.argv:
        print("Smoke test (n_mutations=4, n_seeds=2, n_bootstrap=3)", flush=True)
        records_old = run_experiment(4, n_seeds=2, n_bootstrap=3, threshold=0.5, use_logistic=False)
        records_new = run_experiment(4, n_seeds=2, n_bootstrap=3, threshold=0.5, use_logistic=True)
        print_summary(records_old, records_new)
    elif '--single_run' in sys.argv:
        print("Single run (n_mutations=4, n_seeds=1, no bootstrap)", flush=True)
        records_old = run_single(4, n_seeds=1, use_logistic=False)
        records_new = run_single(4, n_seeds=1, use_logistic=True)
        print_summary(records_old, records_new)
    else:
        main()
