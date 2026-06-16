"""quantile binarization for the two-stage scorecard pipeline.

fit thresholds on train rows only (leakage-free) and apply to held-out rows
separately. an already-binary column passes through; a continuous column becomes
`x_j <= t` indicators at interior quantiles. parent_idx maps each output column
back to the original feature it came from, so a downstream prior or support can be
read at the original-feature level.
"""
from __future__ import annotations

import numpy as np


def fit_binarizer(X_orig, names, n_thresholds):
    """fit binarization thresholds on the given rows. returns (spec, col_names,
    parent_idx): spec is a list of (j, t) per output column, t=None for a
    passed-through binary column or a threshold for an `x_j <= t` indicator;
    parent_idx[c] is the index into `names` of the feature column c came from."""
    qlevels = np.linspace(0.0, 1.0, n_thresholds + 2)[1:-1]
    spec, col_names, parent = [], [], []
    for j, name in enumerate(names):
        x = X_orig[:, j]
        if set(np.unique(x).tolist()) <= {0.0, 1.0}:
            spec.append((j, None))
            col_names.append(name)
            parent.append(j)
            continue
        for t in np.unique(np.quantile(x, qlevels)):
            spec.append((j, float(t)))
            col_names.append(f'{name}_le_{t:g}')
            parent.append(j)
    return spec, col_names, np.asarray(parent, dtype=int)


def apply_binarizer(X_orig, spec):
    """transform original features into the binarized matrix using a fitted spec."""
    cols = [X_orig[:, j] if t is None else (X_orig[:, j] <= t).astype(float)
            for j, t in spec]
    return np.column_stack(cols)


def binarize(X_orig, names, n_thresholds):
    """convenience wrapper: fit thresholds and transform on the same rows. for a
    leakage-free protocol fit on train rows and apply to test rows separately."""
    spec, col_names, parent = fit_binarizer(X_orig, names, n_thresholds)
    return apply_binarizer(X_orig, spec), col_names, parent
