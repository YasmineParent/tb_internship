"""Adult / Census Income (OpenML 'adult'), the canonical large tabular benchmark and
one of FasterRisk's Figure-3 datasets. This is the big-data do-no-harm panel: at this
n the data alone pins the support, so the prior reduces toward vanilla (mu -> 0) and
the point is parity at scale, not a stability gain. target: income > 50K.
"""
from ._openml import load_mixed


def load(args=None):
    return load_mixed('adult', 2, '>50K')
