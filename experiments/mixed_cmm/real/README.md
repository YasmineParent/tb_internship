# Real TB CMM experiments

CMM stability-selection on the real TB data. Mutations are binary nodes, MIC is a continuous sink
(log2 of the 2-fold dilution series); the forbidden-edge mask encodes the causal assumptions
(corrected lineage orientation, MIC sinks, MIC<->MIC forbidden for cross-resistance).

## Scripts (one per experiment)

- **`build_freq_dataset.py`** data prep: binarizes the allele-frequency matrix at a threshold and
  merges the continuous MICs into it, producing `data/real/processed/tb_freq_clean.csv`.
- **`run_tb_subsample_dlm.py`** single-drug (delamanid) stability graph. `build_forbidden` defines
  the mask (corrected lineage default; `--lineage_exogenous` for the legacy orientation). Flags
  select the adjustment condition (`--include_lineage`, `--include_type`, `--forbid_*_to_mic`).
- **`run_tb_multidrug.py`** joint multi-MIC cross-resistance graph. `--drugs`, `--genes` (focus on
  a gene set), `--data` (e.g. the frequency set), rare-variant pathway burdens via `collapse_burden`;
  MIC<->MIC forbidden so a feature parenting >=2 MICs is the cross-resistance read-out.
- `smoke_refit_dlm.py` standalone refit smoke test (pre-existing).

## Running

Each script does one run into `--out_dir`. Pass `--n_shards N` to self-shard across N cores
(distinct seeds, aggregated; equivalent to one long run). The sharding helper lives in
`experiments/_io.py` (`run_sharded` / `self_shard`). Different conditions/configs are just different
invocations, e.g.:

    # delamanid, corrected ablation (run once per condition)
    python -m experiments.mixed_cmm.real.run_tb_subsample_dlm --include_lineage --lineage_merge_below 5 \
        --max_parents 4 --k_max 6 --min_cluster_count 4 --n_shards 10 \
        --out_dir results/mixed_cmm/subsampling/tb_subsampling_dlm_mp4_k6_mcc4_linfix/with_lineage

    # cross-resistance, focused efflux axis on the frequency data
    python -m experiments.mixed_cmm.real.run_tb_multidrug --drugs bdq,cfz \
        --genes rv0678,mmpl5,mmps5,pepq,rv1979c --data data/real/processed/tb_freq_clean.csv \
        --n_shards 10 --out_dir results/mixed_cmm/subsampling/tb_multidrug_freq_focused_efflux_bdq_cfz

## Outputs

`results/mixed_cmm/subsampling/`: `tb_subsampling_dlm_*` (delamanid), `tb_multidrug_*`
(cross-resistance), `*_linfix` (corrected lineage orientation), `*_freq_*` (frequency data).
Notebooks in `notebooks/mixed_cmm/real/` read these and render via `cmm_utils.visualize_stable_bn`.
