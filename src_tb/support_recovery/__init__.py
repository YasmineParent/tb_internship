"""Support-recovery analysis for FasterRisk fits against ground-truth causes S*.

metrics.py     per-fit set-recovery metrics (S_recall, S_precision, C_inclusion)
               and selectivity sel(q) = mean(q on C) / mean(q on S*).
loading.py     load recovery_sweep.py output CSVs and Phase A npz cells into
               tidy DataFrames; compute per-cell selectivity table.
visualization.py
               plot functions for the §6.1 figures: recovery vs mu (single
               panel and faceted by p_edge), recovery vs selectivity scatter,
               soft-vs-hard prior comparison. Imports nothing from loading;
               accepts plain DataFrames.
"""
