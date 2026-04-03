# tb_internship

Internship project.

## Setup

Install [uv](https://astral.sh/uv), then:
```bash
uv sync
uv run jupyter lab
```

## Structurenotebooks/   # Jupyter notebooks
src/         # Reusable Python modules
data/
raw/       # Original data (not versioned)
processed/ # Cleaned/transformed data (not versioned)
reports/     # Figures, outputs


git add .
git commit -m "init: project structure, uv env, README"
uv run python -m ipykernel install --user --name tb_internship --display-name "tb_internship"
code .
code
cat > README.md << 'EOF'
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
