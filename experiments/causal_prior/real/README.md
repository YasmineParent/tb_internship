# Real-data experiments

Two axes, decoupled: **type** lives in the filename, **benchmark** in a `--dataset`
flag, the **loader** in `datasets/`. Method and metric code lives in `src/causal_prior/`
(`binarize`, `stability`, `scorecard`, `baselines`, `priors`, `cv_mu`); these scripts
are pure experiments that import from there.

## Datasets (`datasets/`)

Each exposes `load(args) -> (X_orig, names, y)` with `y in {-1,1}`; categoricals are
one-hot to `{0,1}` so the conditional-gaussian discovery treats them as factors.

- `fico.py` — FICO HELOC (n=10459, near-continuous; `--sentinel-nan` for the -7/-8/-9 codes)
- `heart.py` — Heart Disease Statlog (n=270, mixed clinical, via openml)
- `tb.py` — TB-DLM delamanid resistance (n~152, binary mutations + lineage). Note: the
  generic runners discover q themselves here; the §6.3 *domain-q* result uses the bespoke
  `tb_dlm_*.py` scripts (precomputed CMM prior), not this loader.

## Runners (each takes `--dataset`)

| script | type | paper section |
|---|---|---|
| `parity.py` | causal vs vanilla, AUC parity at matched k | §6.2 |
| `ksweep.py` | parity *curve*: AUC vs model size k | §6.2 |
| `cfs.py` | causal vs CFS baselines + soft-vs-hard ablation; `--n-grid` for the scarcity sweep | §6.2, §6.2b |
| `rashomon.py` | prior steers the diverse pool toward causal supports | §6.2 |
| `cpdag.py` | consensus CPDAG plot (orientation face-validity) | §6.2 diagnostic |

`cfs.py` q-mode is auto: with `--n-grid` q is discovered once on a held-out set
(leakage-free scarcity sweep, FICO); without it q is re-discovered per resample
(small data, heart). Override with `--q-mode`.

## Bespoke (not on the `--dataset` axis, by design)

- `transport.py` — folktables out-of-state transport. Multi-environment (source +
  target states), so it does not fit the single-`(X, y)` loader interface.
- `tb_dlm_{firstpass,p211,robustness}.py` — the TB-DLM domain case study (§6.3),
  application-specific feature sets and three p-regimes.

## Examples

```
python experiments/causal_prior/real/cfs.py --dataset heart --reps 25
python experiments/causal_prior/real/cfs.py --dataset fico --sentinel-nan --n-grid 150,300,600,1200
python experiments/causal_prior/real/parity.py --dataset heart
python experiments/causal_prior/real/ksweep.py --dataset fico --sentinel-nan --qsrc ges_cg
```

## Setup note

The `cfs_*` baselines use pyCausalFS, which is not on PyPI. It is vendored (and
gitignored) at `external/pyCausalFS/`; if missing, `cfs_fisherz` raises with the
clone command. The folktables data (`transport.py`) auto-downloads on first run.
