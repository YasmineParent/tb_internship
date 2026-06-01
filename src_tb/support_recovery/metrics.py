"""Per-fit support-recovery metrics for FasterRisk against ground-truth S*."""
from __future__ import annotations

from typing import Iterable

import numpy as np


def support_recovery_metrics(
    support: Iterable[int],
    S_star: Iterable[int],
    confounded: Iterable[int],
) -> dict[str, float]:
    """Set-recovery metrics of a fitted FR support against ground-truth causes.

    Returns S_recall (= |sup & S*| / |S*|), S_precision (= |sup & S*| / |sup|),
    C_inclusion (= |sup & C| / |sup|), and k_actual. All zero on empty support;
    S_recall NaN if S_star is empty.
    """
    sup = set(int(j) for j in support)
    S = set(int(j) for j in S_star)
    C = set(int(j) for j in confounded)
    k_actual = len(sup)
    if not sup:
        return {'S_recall': 0.0, 'S_precision': 0.0, 'C_inclusion': 0.0, 'k_actual': 0}
    s_hit = len(sup & S)
    c_hit = len(sup & C)
    return {
        'S_recall':    s_hit / len(S) if S else float('nan'),
        'S_precision': s_hit / k_actual,
        'C_inclusion': c_hit / k_actual,
        'k_actual':    k_actual,
    }


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
