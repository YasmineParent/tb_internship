"""Multi-environment version of the linear-Gaussian DAG (ICP-style shift).

Builds one structural causal model with LinGaussSyntheticData, then samples it
under several *environments* that differ only in how strongly the non-causal
correlates couple to the rest of the system. The causal mechanism of Y is held
fixed across environments, so this is the canonical invariance setup: P(Y | Pa(Y))
is invariant, P(X) shifts.

Why the correlate edges are the right (and only safe) knob
----------------------------------------------------------
In LinGaussSyntheticData, `correlates` = features that marginally correlate with
Y but are *not* ancestors of Y (descendants of causes / common-effect structure).
Two facts make their incoming edges a clean environment knob:

  - a correlate is never an ancestor of Y, so no edge *into* a correlate lies on
    any directed path to Y;
  - a cause (ancestor of Y) never has a correlate parent (else that correlate
    would itself be an ancestor of Y, i.e. a cause).

Together: rescaling every incoming edge to a correlate node leaves the entire
causal cone (S*, their ancestors, Y, and P(Y | Pa(Y))) distributionally
identical, and changes only the spurious correlate<->Y associations. A causal q
(peaked on S*) transports across environments; a predictive q peaked on
correlates matches in-distribution but degrades out-of-environment. That is the
experiment that distinguishes causal from merely-selective.

gamma is the per-environment multiplier on those incoming-correlate edges:
  gamma = 1.0  -> reference environment (identical law to the base generator)
  gamma = 0.0  -> correlates decouple into pure noise (lose predictive value)
  gamma = -1.0 -> every spurious correlation reverses sign
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.data.synthetic_lingauss import LinGaussSyntheticData


@dataclass
class Environment:
    X: np.ndarray
    y: np.ndarray            # signed binary, thresholded at the shared boundary
    y_continuous: np.ndarray  # Y_lat
    gamma: float             # correlate-edge multiplier that defines this environment


@dataclass
class EnvBundle:
    """One shared SCM sampled under several environments + the structural sets."""
    environments: list[Environment]
    p: int
    S_star: set[int]
    confounded: set[int]
    all_causes: set[int]
    indirect_causes: set[int]
    correlates: set[int]
    dag: nx.DiGraph
    y_node: int
    threshold: float


def _sample_scm(dag: nx.DiGraph, y_node: int, A: np.ndarray, w_star: np.ndarray,
                w_0_star: float, noise_scale: float, n_samples: int,
                threshold: float, rng: np.random.Generator
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One forward pass of the linear-Gaussian SCM with a fixed label boundary.

    Identical to LinGaussSyntheticData's topological pass, but driven by an
    explicit Generator (so environments draw independent noise) and with the
    threshold supplied rather than recomputed (so the label rule is shared).
    """
    p = A.shape[0]
    X = np.zeros((n_samples, p))
    y_continuous = np.zeros(n_samples)
    for j in nx.topological_sort(dag):
        if j == y_node:
            y_continuous = X @ w_star + w_0_star + rng.normal(0.0, noise_scale, n_samples)
        else:
            parents = list(dag.predecessors(j))
            mean = X[:, parents] @ A[parents, j] if parents else 0.0
            X[:, j] = mean + rng.normal(0.0, noise_scale, n_samples)
    y_signed = (2 * (y_continuous > threshold).astype(int) - 1).astype(int)
    return X, y_signed, y_continuous


def make_environments(p: int, n_samples: int, k_star: int, p_edge: float,
                      gammas, noise_scale: float = 1.0, seed: int = 0) -> EnvBundle:
    """Build one SCM and sample it under each gamma in `gammas`.

    The structure (DAG, S*, edge weights A, w_star, label threshold) is built
    once via LinGaussSyntheticData at the requested n so the threshold is a
    well-estimated median of Y_lat. Each environment then rescales the incoming
    edges to correlate nodes by its gamma and draws fresh noise.
    """
    base = LinGaussSyntheticData(p=p, n_samples=n_samples, k_star=k_star,
                                 p_edge=p_edge, noise_scale=noise_scale, seed=seed)
    correlates = sorted(base.correlates)
    rng = np.random.default_rng(seed + 10_000)

    environments: list[Environment] = []
    for gamma in gammas:
        A_e = base.A.copy()
        if correlates:
            A_e[:, correlates] *= float(gamma)   # incoming edges to every correlate
        X, y, y_cont = _sample_scm(base.dag, base.y_node, A_e, base.w_star,
                                   base.w_0_star, noise_scale, n_samples,
                                   base.threshold, rng)
        environments.append(Environment(X=X, y=y, y_continuous=y_cont, gamma=float(gamma)))

    return EnvBundle(
        environments=environments, p=p,
        S_star=set(base.S_star), confounded=set(base.confounded),
        all_causes=set(base.all_causes), indirect_causes=set(base.indirect_causes),
        correlates=set(correlates), dag=base.dag, y_node=base.y_node,
        threshold=base.threshold,
    )
