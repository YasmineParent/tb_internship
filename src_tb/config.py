"""Shared experiment configuration: defaults and parameter sweeps for synthetic validation.

Defaults are TB-realistic (n_samples=164 matches the TB cohort size). Each sweep
includes the default value so it acts as a reference point in cross-sweep plots.
"""

DEFAULTS = dict(n_obs=10, p_graph=0.4, p_mix=0.5, n_mix=2, k_components=2, n_samples=164)

SWEEPS = {
    'n_mix':     [2, 3, 4, 5],
    'p_mix':     [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    'n_samples': [100, 164, 300, 500, 750, 1000],
    'n_obs':     [4, 6, 8, 10],
    'p_graph':   [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

PARAM_LABELS = {
    'n_mix':     'Mix. Nodes (NZ)',
    'p_mix':     'Mix. Edge Density (pZ)',
    'n_samples': 'Sample Size (S)',
    'n_obs':     'Obs. Nodes (NX)',
    'p_graph':   'Obs. Edge Density (pG)',
}

GRAPH_METRICS = ['sd', 'sc', 'shd', 'fpr', 'tpr', 'f1', 'tp', 'mcc', 'shd-nm']
