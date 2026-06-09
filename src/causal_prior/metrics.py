"""Per-fit support-recovery metrics for FasterRisk against ground-truth S*."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def support_recovery_metrics(
    support: Iterable[int],
    S_star: Iterable[int],
    confounded: Iterable[int],
    causes: Iterable[int] | None = None,
    correlates: Iterable[int] | None = None,
) -> dict[str, float | int]:
    """Set-recovery metrics of a fitted FR support against ground-truth causes.

    Always returns, against the *direct parents* S* of Y:
        S_recall    = |sup & S*| / |S*|      (NaN if S* empty)
        S_precision = |sup & S*| / |sup|     (proximal-cause precision)
        C_inclusion = |sup & C| / |sup|      (C = confounded set, includes
                                              indirect causes; kept for continuity)
        k_actual    = |sup|

    If `causes` (= S* u indirect causes = Anc(Y)) and `correlates` (= the
    genuinely non-causal subset of C) are supplied, also returns the
    cause-aware metrics that don't penalise selecting a true *upstream* cause:
        causal_precision     = |sup & causes| / |sup|
        correlate_inclusion  = |sup & correlates| / |sup|

    All ratios are 0.0 on empty support.
    """
    sup = set(int(j) for j in support)
    S = set(int(j) for j in S_star)
    C = set(int(j) for j in confounded)
    k_actual = len(sup)
    out: dict[str, float | int]
    if not sup:
        out = {'S_recall': 0.0, 'S_precision': 0.0, 'C_inclusion': 0.0, 'k_actual': 0}
    else:
        out = {
            'S_recall':    len(sup & S) / len(S) if S else float('nan'),
            'S_precision': len(sup & S) / k_actual,
            'C_inclusion': len(sup & C) / k_actual,
            'k_actual':    k_actual,
        }
    if causes is not None:
        A = set(int(j) for j in causes)
        out['causal_precision'] = (len(sup & A) / k_actual) if sup else 0.0
    if correlates is not None:
        R = set(int(j) for j in correlates)
        out['correlate_inclusion'] = (len(sup & R) / k_actual) if sup else 0.0
    return out


def selectivity(q: np.ndarray, S_star: Iterable[int], confounded: Iterable[int]) -> float:
    """sel(q) = mean(q on C) / mean(q on S*); lower means more causally selective.

    Returns inf if mean(q on S*) is 0 (the prior provides no S*-signal); NaN if
    either set is empty.
    """
    q = np.asarray(q)
    S_idx = list(int(j) for j in S_star)
    C_idx = list(int(j) for j in confounded)
    if not S_idx or not C_idx:
        return float('nan')
    bar_S = float(q[S_idx].mean())
    bar_C = float(q[C_idx].mean())
    if bar_S == 0.0:
        return float('inf')
    return bar_C / bar_S
