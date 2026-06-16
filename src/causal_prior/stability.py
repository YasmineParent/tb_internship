"""selection-stability metrics over a list of selected-feature supports.

each support is a set (frozenset) of selected feature identifiers; the list is the
selections across resamples. `mean_pairwise_jaccard` is the mean pairwise Jaccard (the metric
used throughout §6.2 and §6.3); `nogueira` is the chance-corrected index that
stays comparable across methods selecting different numbers of features.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def mean_pairwise_jaccard(supports):
    pairs = [jaccard(a, b) for a, b in combinations(supports, 2)]
    return float(np.mean(pairs)) if pairs else float('nan')


def nogueira(supports, d):
    """nogueira & brown (jmlr 2017) chance-corrected selection stability.

    1 = identical selections across resamples, ~0 = no better than random, can go
    negative. unlike raw jaccard it corrects for chance and stays comparable across
    methods that select different numbers of features (cfs restricts to a small
    blanket, the causal arm roams), which is exactly the asymmetry here. d is the
    size of the full feature universe; features never selected contribute zero
    variance but still count toward d.
    """
    m = len(supports)
    if m < 2 or d == 0:
        return float('nan')
    sizes = np.array([len(s) for s in supports], dtype=float)
    kbar = sizes.mean()
    if kbar == 0 or kbar == d:
        return float('nan')  # denominator undefined (every resample selects all / none)
    cnt = Counter()
    for s in supports:
        cnt.update(s)
    phat = np.array(list(cnt.values()), dtype=float) / m
    var = (m / (m - 1)) * phat * (1 - phat)  # unobserved features add 0 to the sum
    num = var.sum() / d
    den = (kbar / d) * (1 - kbar / d)
    return float(1 - num / den)
