# tb_internship

Causal inference for drug-resistant tuberculosis biomarker identification.

## Setup

Install [uv](https://astral.sh/uv), then:

    git clone --recurse-submodules <this repo>
    uv sync

The `external/cmm` submodule is required and is auto-injected onto `sys.path` by `src_tb/__init__.py`.

## Layout

    src_tb/                          reusable package
      causal_recovery/               CMM subsampling/stability selection, evaluation, plotting
      data/                          synthetic generators and TB data loaders
    experiments/
      mixed_cmm/
        synthetic/                   synthetic-data validation
          config.py                  shared defaults / sweeps
          run_synthetic_validation.py  A/B Gaussian vs FLXMRglm
          run_parameter_sweep.py     Figure-D.2-style sweep (gaussian vs logistic)
          run_baselines_sweep.py     same sweep, vs PC / GES / empty baselines
          recompute_metrics.py       re-score a results.csv from saved edges
        real/                        TB cohort experiments
          run_tb_subsample_dlm.py    stability selection (CMM-logistic) on dlm_mic + mutations
    notebooks/
      mixed_cmm/
        synthetic/                   sweep_plots.ipynb, baselines_sweep_plots.ipynb
        real/                        delamanid_stable_graphs.ipynb, data_exploration.ipynb
    data/
      real/{raw,processed}/          TB data (gitignored)
      synthetic/                     generated (gitignored)
    external/{cmm,fasterrisk}/       git submodules
    results/                         timestamped run outputs (gitignored)

## Running

    uv run python experiments/mixed_cmm/synthetic/run_synthetic_validation.py --smoke
    uv run python experiments/mixed_cmm/synthetic/run_parameter_sweep.py --n_seeds 3
    uv run python experiments/mixed_cmm/synthetic/run_baselines_sweep.py --n_seeds 3
    uv run python experiments/mixed_cmm/real/run_tb_subsample_dlm.py --n_runs 100
    uv run jupyter lab
