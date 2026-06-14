"""CPDAG comparison metrics: skeleton, orientation, and SHD.

All functions take amats in the pcalg coding: nonzero amat[i,j] is a mark from i
into j; amat[i,j]=amat[j,i]=1 is the undirected edge i--j; amat[i,j]=1 with
amat[j,i]=0 is the directed edge i->j (arrowhead at j). Compare an estimated
CPDAG against the *true CPDAG* (dag_to_cpdag_amat), so genuinely unorientable
edges are not charged as orientation errors.

Three views, each answering a different question Nataliya might ask:
  - adjacency (skeleton): did we recover the right edges at all?
  - arrowhead: among the edges, did we orient them the right way?
  - SHD: one number summarising both (lower is better).
"""

from __future__ import annotations

import numpy as np


def skeleton_pairs(amat: np.ndarray) -> set[tuple[int, int]]:
    """Unordered pairs {i<j} carrying any edge (directed or undirected)."""
    A = np.asarray(amat) != 0
    p = A.shape[0]
    return {(i, j) for i in range(p) for j in range(i + 1, p)
            if A[i, j] or A[j, i]}


def arrow_set(amat: np.ndarray) -> set[tuple[int, int]]:
    """Directed marks: ordered (i, j) with i->j (arrowhead at j, none at i)."""
    A = np.asarray(amat) != 0
    p = A.shape[0]
    return {(i, j) for i in range(p) for j in range(p)
            if i != j and A[i, j] and not A[j, i]}


def _prf(true_set: set, est_set: set) -> dict[str, float]:
    """Precision / recall / F1 of est against true. Empty denominators -> nan."""
    tp = len(true_set & est_set)
    prec = tp / len(est_set) if est_set else float('nan')
    rec = tp / len(true_set) if true_set else float('nan')
    # f1 is nan whenever precision or recall is undefined, so cells where the
    # quantity is not applicable (e.g. a fully unoriented true cpdag has no
    # arrowheads to score) are skipped from downstream means, not counted as 0.
    if np.isnan(prec) or np.isnan(rec):
        f1 = float('nan')
    elif (prec + rec) == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {'precision': prec, 'recall': rec, 'f1': f1}


def _ptype(A: np.ndarray, i: int, j: int) -> tuple[bool, bool]:
    return bool(A[i, j]), bool(A[j, i])


def shd_cpdag(true_amat: np.ndarray, est_amat: np.ndarray) -> int:
    """Structural Hamming distance between two CPDAGs: number of unordered pairs
    whose edge type (none / i->j / j->i / i--j) differs. A missing edge, an extra
    edge, and a reversed/under-determined orientation each count once."""
    T = np.asarray(true_amat) != 0
    E = np.asarray(est_amat) != 0
    p = T.shape[0]
    return sum(_ptype(T, i, j) != _ptype(E, i, j)
               for i in range(p) for j in range(i + 1, p))


def all_scores(true_amat: np.ndarray, est_amat: np.ndarray) -> dict[str, float]:
    """SHD plus adjacency_{p,r,f1} and arrowhead_{p,r,f1} in one flat dict."""
    adj = _prf(skeleton_pairs(true_amat), skeleton_pairs(est_amat))
    arr = _prf(arrow_set(true_amat), arrow_set(est_amat))
    out: dict[str, float] = {'shd': float(shd_cpdag(true_amat, est_amat))}
    out.update({f'adjacency_{k}': v for k, v in adj.items()})
    out.update({f'arrowhead_{k}': v for k, v in arr.items()})
    return out
