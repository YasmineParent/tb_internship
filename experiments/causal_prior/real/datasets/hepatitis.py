"""Hepatitis loader (UCI via openml): predict death from clinical signs.

n=155, mixed: continuous labs (bilirubin, albumin, protime, ...) plus many binary
clinical signs (fatigue, ascites, varices, ...). positive class = DIE (the risk
event). small and categorical-rich, a clinical Fisher-Z stress test.
"""
from experiments.causal_prior.real.datasets._openml import load_mixed


def load(args=None):
    return load_mixed('hepatitis', 1, 'DIE')
