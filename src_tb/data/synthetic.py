import random
import tempfile
from pathlib import Path

import numpy as np
import networkx as nx
import pyagrum as gum
from sklearn import preprocessing

import src_tb  # ensures external/cmm is on sys.path
from src.exp.gen.synthetic.data_gen_mixing import DataGen
from src.exp.gen.generate import (
    gen_erdos_graph, gen_random_intervention_targets,
    FunType, NoiseType, IvType, IvMode, DagType
)


class BinaryDataGen(DataGen):
    """Extends DataGen for mixed binary/continuous data.
    Binary nodes use logistic link; continuous nodes use linear (iSCM applied only to continuous)."""

    def __init__(self, params: dict, graph: nx.DiGraph, binary_nodes: set, seed: int = 0, vb: int = 0):
        # Reseed before gen_X so binomial draws are reproducible regardless of how much
        # global state was consumed by upstream graph/intervention generation.
        np.random.seed(seed)
        self.binary_nodes = set(binary_nodes)
        super().__init__(params, graph, seed, vb)

    def _gen_functional_deps(self, X: np.ndarray) -> np.ndarray:
        fun = self.gen_dict[self.gen]
        for i in nx.topological_sort(self.G):
            X = self._gen_functional_dep_i(X, i, fun)
            if self.scale_during and i not in self.binary_nodes:
                X[:, [i]] = preprocessing.StandardScaler().fit(X[:, [i]]).transform(X[:, [i]])
        return X

    def _gen_functional_dep_i(self, X: np.ndarray, i: int, fun) -> np.ndarray:
        if i not in self.binary_nodes:
            return super()._gen_functional_dep_i(X, i, fun)

        par = list(self.G.predecessors(i))
        is_cf = any(i in conf_ind for conf_ind in self.conf_ind_sets)

        # Source binary nodes: replace base-class gaussian noise with Bernoulli draws.
        # For confounded sources, vary p per mixture cluster via sigmoid of the cluster bias.
        if not par:
            if not is_cf:
                X[:, i] = np.random.binomial(1, 0.5, size=X.shape[0])
            else:
                for iz, conf_ind in enumerate(self.conf_ind_sets):
                    if i not in conf_ind:
                        continue
                    for k in np.unique(self.Zs[iz]):
                        mask = self.Zs[iz] == k
                        p = _sigmoid(self.bs[iz][k][i])
                        X[mask, i] = np.random.binomial(1, p, size=int(mask.sum()))
            return X

        if not is_cf:
            linear = X[:, par].dot(self.W[par, i])
            X[:, i] = np.random.binomial(1, _sigmoid(linear))
        else:
            for iz, conf_ind in enumerate(self.conf_ind_sets):
                if i not in conf_ind:
                    continue
                for k in np.unique(self.Zs[iz]):
                    mask = self.Zs[iz] == k
                    linear = X[mask][:, par].dot(self.Ws[iz][k][par, i])
                    X[mask, i] = np.random.binomial(1, _sigmoid(linear))
        return X


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def _default_params(n_obs: int, n_samples: int) -> dict:
    return {
        'N': n_obs, 'S': n_samples, 'P': 0.4,
        'F': FunType.LIN, 'NS': NoiseType.GAUSS,
        'DG': DagType.ERDOS, 'IVT': IvType.SHIFT,
        'IVM': IvMode.MIXING, 'K': 2, 'NZ': 1, 'PZ': 0.5,
        'C': 5, 'IMAX': 3,
    }


class SyntheticData:
    """Binary mutation data with known causal structure for CMM validation.

    Follows the CMM paper's experimental setup (Figure D.2) extended to binary observed variables.
    Random Erdos-Renyi DAG, last node is continuous (Y/MIC), all others are binary (mutations).
    Latent Z creates mixture components with different causal mechanisms.

    Parameters match the paper: n_obs=NX, p_graph=pG, p_mix=pZ, n_mix=NZ, k_components=K, n_samples=S.

    NAMING CONVENTION (load-bearing): features are named 'mut_0', 'mut_1', ..., 'mut_{N-2}', 'Y'.
    The last column is always the continuous target named 'Y'; all others are binary mutations.
    src_tb.causal_recovery.evaluation.score_recovered relies on these exact names to split
    metrics by edge class. If you rename, update both files.
    """

    def __init__(self, n_obs: int = 10, p_graph: float = 0.4, p_mix: float = 0.5,
                 n_mix: int = 1, k_components: int = 2, n_samples: int = 164, seed: int = 0):
        # Hermetic seeding: snapshot caller's RNG state for both np.random and stdlib random
        # (upstream cmm uses both as globals), seed deterministically, restore on exit.
        np_state = np.random.get_state()
        py_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            rng = np.random.default_rng(seed)

            params = _default_params(n_obs, n_samples)
            params.update({'P': p_graph, 'PZ': p_mix, 'NZ': n_mix, 'K': k_components})

            dag = gen_erdos_graph(params)
            y_node = n_obs - 1
            dag.remove_edges_from(list(dag.out_edges(y_node)))
            binary_nodes = set(range(n_obs - 1))

            mixing_sets = gen_random_intervention_targets(params, dag, rng)
            # gen_random_intervention_targets distributes floor(p_mix * n_obs) nodes across
            # n_mix sets, leaving empty sets when there aren't enough nodes to go around.
            # When p_mix > 0, pad empties with unused nodes so the realized mixture count
            # matches n_mix. p_mix=0 keeps everything empty by design.
            if p_mix > 0:
                used = {n for s in mixing_sets for n in s}
                available = [n for n in dag.nodes() if n not in used]
                for i, s in enumerate(mixing_sets):
                    if not s and available:
                        picked = int(rng.choice(available))
                        available.remove(picked)
                        mixing_sets[i] = [picked]
            mixing_sets = [s for s in mixing_sets if s]

            Zs = []
            for _ in mixing_sets:
                p = rng.dirichlet(np.ones(k_components))
                Zs.append(rng.choice(k_components, size=n_samples, p=p))

            gen = BinaryDataGen(params, dag, binary_nodes, seed=seed)
            X = gen.gen_X(mixing_sets, Zs)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

        self.X = X
        self.dag = dag
        self.features = [f'mut_{i}' for i in range(n_obs - 1)] + ['Y']
        self.binary_indices = list(binary_nodes)
        y_idx = n_obs - 1
        self.forbidden_edges = {(y_idx, j) for j in range(n_obs - 1)}

        self.true_edges = {
            (self.features[u], self.features[v]) for u, v in dag.edges()
        }
        self.true_bin_to_bin = {
            (s, t) for s, t in self.true_edges if s != 'Y' and t != 'Y'
        }
        self.true_bin_to_cont = {
            (s, t) for s, t in self.true_edges if t == 'Y'
        }

    def visualize_true_dag(self, size: str = "20") -> None:
        """Visualize the true causal DAG. Rendered as PNG so the output embeds
        in .ipynb in a format that renders reliably on GitHub."""
        import pyagrum.lib.image as gimg
        from IPython.display import display, Image
        bn = gum.BayesNet()
        for i, f in enumerate(self.features):
            if i in self.binary_indices:
                bn.add(gum.LabelizedVariable(f, f, 2))
            else:
                bn.add(gum.RangeVariable(f, f, 0, 1))
        for src, tgt in self.true_edges:
            bn.addArc(src, tgt)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            gimg.export(bn, path, size=size)
            with open(path, 'rb') as fp:
                data = fp.read()
        finally:
            Path(path).unlink(missing_ok=True)
        display(Image(data=data, format='png'))
