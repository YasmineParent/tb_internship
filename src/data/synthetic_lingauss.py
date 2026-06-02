import random

import numpy as np
import networkx as nx


class LinGaussSyntheticData:
    """Random linear-Gaussian DAG with a continuous latent sink Y_lat, threshold-binarized.

    The DAG has p + 1 nodes: 0..p-1 are continuous features, node p is the latent
    Y_lat (a sink, no outgoing edges). Pa(Y_lat) is the planted set S_star of size
    k_star, chosen uniformly at random. No latent variables in the causal sense;
    any non-causal correlation between a feature and Y is mediated by an observed
    common ancestor.

    Mechanisms (fully linear-Gaussian):
        - features:  x_j   = sum_{i in Pa(j)} A[i, j] x_i + N(0, noise_scale^2)
        - latent Y:  Y_lat = w_star @ x + w_0_star + N(0, noise_scale^2)
        - observed:  y     = 2 * 1[Y_lat > threshold] - 1

    Two views of Y are exposed:
        - self.y_continuous: Y_lat, used as the target for the causal-discovery stage
          (PC, GES) so standard continuous CI tests apply throughout the DAG.
        - self.y: signed binary threshold of Y_lat, used as the target for FasterRisk.

    This mirrors the TB pipeline (continuous MIC for the causal step, binarized for
    the classifier) and lets PC use Fisher Z on continuous Y_lat instead of a
    weaker mixed-data CI test.

    Naming convention: features are 'x_0', ..., 'x_{p-1}', 'y'. Default threshold
    is the median of Y_lat (~50/50 class balance).
    """

    def __init__(self, p: int = 30, n_samples: int = 500,
                 p_edge: float = 0.2, k_star: int = 5,
                 w_min: float = 1.0, w_max: float = 3.0,
                 a_min: float = 0.5, a_max: float = 2.0,
                 noise_scale: float = 1.0,
                 intercept: float = 0.0,
                 threshold: float | None = None,
                 seed: int = 0, max_resample: int = 100):
        # snapshot the caller's rngs so seeding here doesn't leak globally
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)

            y_node = p
            dag = None
            S_star: list[int] = []
            confounded: list[int] = []
            # resample if the dag happens to leave y with no non-causal correlates;
            # rare when p_edge >= 0.15
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

            # edge weights for feature -> feature edges only; w_star handles edges into Y
            A = np.zeros((p, p))
            for u, v in dag.edges():
                if v == y_node:
                    continue
                A[u, v] = np.random.choice([-1.0, 1.0]) * np.random.uniform(a_min, a_max)

            w_star = np.zeros(p)
            for j in S_star:
                w_star[j] = np.random.choice([-1.0, 1.0]) * np.random.uniform(w_min, w_max)

            # single topological pass: linear-gaussian everywhere, Y_lat at the sink
            w_0_star = float(intercept)
            X = np.zeros((n_samples, p))
            y_continuous = np.zeros(n_samples)
            for j in nx.topological_sort(dag):
                if j == y_node:
                    y_continuous = (
                        X @ w_star + w_0_star
                        + np.random.normal(0.0, noise_scale, size=n_samples)
                    )
                else:
                    parents = list(dag.predecessors(j))
                    mean = X[:, parents] @ A[parents, j] if parents else 0.0
                    X[:, j] = mean + np.random.normal(0.0, noise_scale, size=n_samples)

            # default threshold = median(Y_lat) gives ~50/50 class balance
            tau = float(threshold) if threshold is not None else float(np.median(y_continuous))
            y_signed = (2 * (y_continuous > tau).astype(int) - 1).astype(int)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

        self.X = X
        self.y = y_signed
        self.y_continuous = y_continuous
        self.dag = dag
        self.y_node = y_node
        self.S_star = set(S_star)
        self.confounded = set(confounded)
        self.w_star = w_star
        self.w_0_star = w_0_star
        self.A = A
        self.threshold = tau
        self.features = [f'x_{i}' for i in range(p)] + ['y']


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
    """Non-S_star features that correlate with Y in the marginal (no conditioning).
    Equals Desc(Anc(Y)) \\ S_star \\ {Y}: ancestors of Y, descendants of Pa(Y),
    and features sharing a common ancestor with Y, all in one set."""
    ancestors_of_Y = nx.ancestors(dag, y_node)
    confounded = set(ancestors_of_Y)
    for k in ancestors_of_Y:
        confounded |= nx.descendants(dag, k)
    confounded -= set(S_star)
    confounded.discard(y_node)
    return sorted(confounded)
