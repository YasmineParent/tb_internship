# tb_internship

Internship project.

## Setup

Install uv (https://astral.sh/uv), then:

    uv sync
    uv run jupyter lab

## Structure

    notebooks/   # Jupyter notebooks
    src/         # Reusable Python modules
    data/
      raw/       # Original data (not versioned)
      processed/ # Cleaned/transformed data (not versioned)
    reports/     # Figures, outputs
