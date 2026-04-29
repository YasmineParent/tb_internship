# tb_internship

Causal inference for drug-resistant tuberculosis biomarker identification.

## Setup

Install [uv](https://astral.sh/uv), then:

    git clone --recurse-submodules <this repo>
    uv sync

The `external/cmm` submodule is required and is auto-injected onto `sys.path` by `src_tb/__init__.py`.

## Layout

    src_tb/                       reusable package
      causal_recovery/            CMM bootstrap, evaluation, plotting
      data/                       synthetic generators and TB data loaders
      config.py                   shared experiment defaults / sweeps
    experiments/
      run_synthetic_validation.py A/B Gaussian vs FLXMRglm on synthetic data
      run_parameter_sweep.py      Figure-D.2-style sweep over NX, pG, S, NZ, pZ
    notebooks/                    Jupyter notebooks
    data/
      real/{raw,processed}/       TB data (gitignored)
      synthetic/                  generated (gitignored)
    external/{cmm,fasterrisk}/    git submodules
    results/                      timestamped run outputs (gitignored)

## Running

    uv run python experiments/run_synthetic_validation.py --smoke
    uv run python experiments/run_parameter_sweep.py --n_seeds 3
    uv run jupyter lab
