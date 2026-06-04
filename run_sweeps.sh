#!/bin/bash
set -e
cd ~/work/tb_internship

uv run python -u experiments/causal_prior/synthetic/recovery_sweep.py \
    --cache-dir results/causal_prior/synthetic/cache_p30_headline \
    --out-dir results/causal_prior/synthetic/recovery_p30_headline

uv run python -u experiments/causal_prior/synthetic/recovery_sweep_cv.py \
    --cache-dir results/causal_prior/synthetic/cache_p30_headline \
    --out-dir results/causal_prior/synthetic/recovery_p30_headline_cv

uv run python -u experiments/causal_prior/synthetic/recovery_sweep_cv.py \
    --cache-dir results/causal_prior/synthetic/cache_p30_headline \
    --out-dir results/causal_prior/synthetic/recovery_p30_K_ablation \
    --K-multipliers 1.0,1.5,2.0,3.0 \
    --cell-filter "seed*_p30_n300_k5_pedge0.2.npz"
