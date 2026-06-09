# tb_internship

Causally-informed sparse risk scoring for drug-resistant tuberculosis. A modified FasterRisk classifier takes a per-feature causal-evidence vector q and biases an integer-coefficient scorecard toward causal features, with strength set by a single parameter mu. The full method writeup is in [notes/pipeline.ipynb](notes/pipeline.ipynb).

## Setup

Install [uv](https://astral.sh/uv), then:

    git clone --recurse-submodules <this repo>
    uv sync

Three submodules under `external/` are required: `cmm` (the causal-discovery prior source), `fasterrisk` (the modified classifier), and `riskslim`. On import, `src/__init__.py` extends `src.__path__` to include `external/cmm/src/`, so CMM modules are importable as `src.exp.*`, `src.mixtures.*`, etc. Our own subpackages take precedence.

## Layout

    src/                             reusable package
      causal_prior/                  the causal-prior method: q-sources (PC/GES/L1 stability),
                                       CV mu selection, recovery metrics, plotting
      causal_discovery/              CMM stability-selection utils, evaluation, visualization
      data/                          synthetic generators (lingauss, envs) and TB loaders
    experiments/
      causal_prior/
        synthetic/                   recovery_sweep[_cv], recovery_shift, exact_radii,
                                       two_radii, beam_gap, stability_selection, config
        real/                        tb_dlm_firstpass, tb_dlm_p211, tb_dlm_robustness
      mixed_cmm/
        synthetic/                   run_baselines_sweep, run_parameter_sweep, recompute_metrics
        real/                        run_tb_subsample_dlm (CMM-logistic stability on dlm_mic)
    notebooks/
      causal_prior/synthetic/        recovery_plots (6.1), recovery_shift_plots (6.4)
      mixed_cmm/real/                data_exploration, delamanid_stable_graphs,
                                       delamanid_ablation_comparison
      mixed_cmm/synthetic/           sweep_plots, baselines_sweep_plots
    notes/                           pipeline.ipynb (method writeup), perturbation_theorem.md
    data/
      real/{raw,processed}/          TB data (gitignored)
      synthetic/                     generated (gitignored)
    external/{cmm,fasterrisk,riskslim}/   git submodules
    results/                         run outputs (gitignored)

## Running

    # causal-prior recovery sweep (6.1) and CV mu selection
    uv run python experiments/causal_prior/synthetic/recovery_sweep.py \
        --cache-dir results/causal_prior/synthetic/cache_p30_headline \
        --out-dir results/causal_prior/synthetic/recovery_p30_headline
    uv run python experiments/causal_prior/synthetic/recovery_sweep_cv.py \
        --cache-dir results/causal_prior/synthetic/cache_p30_headline \
        --out-dir results/causal_prior/synthetic/recovery_p30_headline_cv

    # environment-shift transport experiment (6.4)
    uv run python experiments/causal_prior/synthetic/recovery_shift.py

    # TB application
    uv run python experiments/causal_prior/real/tb_dlm_firstpass.py
    uv run python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --n_runs 100

    # CMM source-validation sweeps on TB-shaped synthetic
    uv run python experiments/mixed_cmm/synthetic/run_baselines_sweep.py --n_seeds 3

    uv run jupyter lab
