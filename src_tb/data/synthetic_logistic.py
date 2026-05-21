import random

import numpy as np
import networkx as nx


class LogisticSyntheticData:
    """Continuous-feature linear-Gaussian DAG with a logistic-SCM binary sink Y.

    Generic causal DAG for the §6.1 method-paper validation. Tests the prior-penalty
    mechanism on a setup where causal sufficiency holds, so PC / GES / IAMB are all
    sound prior sources. No latent variables; confounding is mediated only by
    observed common ancestors.

    The DAG has p + 1 nodes: 0..p-1 are continuous features, node p is Y (a sink
    by construction). Pa(Y) is the planted sparse set S_star. Mechanisms:
        - features: x_j = sum_{i in Pa(j)} A[i, j] x_i + N(0, noise_scale^2)
        - sink Y:   P(Y=+1 | x) = sigma(w_star @ x + w_0_star)

    NAMING CONVENTION: features are named 'x_0', ..., 'x_{p-1}', 'y'. The last
    column is Y (signed binary in self.y, FasterRisk convention).

    No CMM dependency; the Erdős-Rényi DAG generator is inlined.
    """

    def __init__(self, p: int = 30, n_samples: int = 500,
                 p_edge: float = 0.2, k_star: int = 5,
                 w_min: float = 1.0, w_max: float = 3.0,
                 a_min: float = 0.5, a_max: float = 2.0,
                 noise_scale: float = 1.0,
                 intercept: float | None = None,
                 seed: int = 0, max_resample: int = 100):
        # Hermetic seeding: snapshot caller's RNG state for both np.random and stdlib
        # random (the inlined ER generator uses np.random; mirror synthetic.py).
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)

            y_node = p
            dag = None
            S_star: list[int] = []
            confounded: list[int] = []
            # Resample if the DAG happens to leave Y with no confounded correlates;
            # degenerate setup (bootstrap-L1 ~= PC), very rare at p_edge >= 0.15.
            for _ in range(max_resample):
                feat_dag = _gen_erdos_dag(p, p_edge)
                S_star = sorted(np.random.choice(p, size=k_star, replace=False).tolist())
                dag = feat_dag.copy()
                dag.add_node(y_node)
                for j in S_star:
                    dag.add_edge(j, y_node)
                confounded = _confounded_set(dag, y_node, S_star)
                if confounded:
                    break
            else:
                raise RuntimeError(
                    f"No observed-common-ancestor confounding after {max_resample} "
                    f"resamples; try increasing p_edge or k_star."
                )

            # Edge weights for feature edges only; edges into Y are encoded by w_star.
            A = np.zeros((p, p))
            for u, v in dag.edges():
                if v == y_node:
                    continue
                A[u, v] = np.random.choice([-1.0, 1.0]) * np.random.uniform(a_min, a_max)

            w_star = np.zeros(p)
            for j in S_star:
                w_star[j] = np.random.choice([-1.0, 1.0]) * np.random.uniform(w_min, w_max)

            # Single topological pass: linear-Gaussian on features, logistic at the sink.
            X = np.zeros((n_samples, p))
            y_signed = np.zeros(n_samples, dtype=int)
            w_0_star = 0.0
            for j in nx.topological_sort(dag):
                if j == y_node:
                    linear = X @ w_star
                    # Default intercept centers the logit so class balance is ~50/50.
                    w_0_star = (
                        float(intercept) if intercept is not None
                        else -float(linear.mean())
                    )
                    probs = _sigmoid(linear + w_0_star)
                    y_signed = 2 * np.random.binomial(1, probs) - 1
                else:
                    parents = list(dag.predecessors(j))
                    mean = X[:, parents] @ A[parents, j] if parents else 0.0
                    X[:, j] = mean + np.random.normal(0.0, noise_scale, size=n_samples)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

        self.X = X
        self.y = y_signed
        self.dag = dag
        self.y_node = y_node
        self.S_star = set(S_star)
        # Non-S_star features d-connected to Y; the bootstrap-L1 trap and the
        # adversarial-q arm both key off this set.
        self.confounded = set(confounded)
        self.w_star = w_star
        self.w_0_star = w_0_star
        self.A = A
        self.features = [f'x_{i}' for i in range(p)] + ['y']


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _gen_erdos_dag(p_nodes: int, p_edge: float) -> nx.DiGraph:
    """Erdős-Rényi DAG: parents sampled from strictly later positions in a random
    causal order, so the result is acyclic by construction (no cycle check needed)."""
    causal_order = np.random.permutation(p_nodes)
    adj = np.zeros((p_nodes, p_nodes))
    for i in range(p_nodes - 1):
        node = causal_order[i]
        possible_parents = causal_order[i + 1:]
        num_parents = int(np.random.binomial(p_nodes - i - 1, p_edge))
        parents = np.random.choice(possible_parents, size=num_parents, replace=False)
        adj[parents, node] = 1
    return nx.DiGraph(adj)


def _confounded_set(dag: nx.DiGraph, y_node: int, S_star: list[int]) -> list[int]:
    """Non-S_star features d-connected to Y in the marginal (no conditioning).
    Equals Desc(Anc(Y)) \\ S_star \\ {Y}: ancestors of Y, descendants of Pa(Y),
    and features sharing a common ancestor with Y, all in one set."""
    ancestors_of_Y = nx.ancestors(dag, y_node)
    confounded = set(ancestors_of_Y)
    for k in ancestors_of_Y:
        confounded |= nx.descendants(dag, k)
    confounded -= set(S_star)
    confounded.discard(y_node)
    return sorted(confounded)
