import numpy as np
import pyagrum as gum
import pyagrum.lib.notebook as gnb


class SyntheticData:
    """Binary mutation data with known causal structure for CMM validation.

    Structure (n_mutations rounded down to nearest multiple of 4, minimum 4):
      - Quarter A: direct roots, each causes Y directly (binary to continuous).
      - Quarter B: chain roots, each causes one intermediate mutation (binary to binary),
        which then causes Y (binary to continuous).
      - Quarter C: intermediate mutations caused by quarter B roots.
      - Quarter D: independent roots with no path to Y (tests false positive rate).
      Latent Z has n_components subpopulations with different mechanisms for all edges.

    For n_mutations=4: mut_0 -> Y, mut_1 -> mut_2 -> Y, mut_3 independent.
    """

    def __init__(self, n_mutations: int, n_samples: int = 164, n_components: int = 2, seed: int = 0):
        assert n_components >= 2, "n_components must be >= 2"
        rng = np.random.default_rng(seed)
        self.n_quarter = max(1, n_mutations // 4)
        n_mutations = self.n_quarter * 4

        Z = rng.choice(n_components, size=n_samples, p=rng.dirichlet(np.ones(n_components)))
        self.features = [f'mut_{i}' for i in range(n_mutations)] + ['Y']
        self.true_edges = set()

        #quarter A: direct roots (indices 0..n_quarter-1)
        direct_roots = [rng.binomial(1, 0.3, n_samples) for _ in range(self.n_quarter)]

        #quarter B: chain roots (indices n_quarter..2*n_quarter-1)
        chain_roots = [rng.binomial(1, 0.3, n_samples) for _ in range(self.n_quarter)]

        #quarter C: intermediates caused by chain roots (indices 2*n_quarter..3*n_quarter-1)
        intermediates = []
        for j in range(self.n_quarter):
            coeffs = np.where(np.arange(n_components) == 0, 0.7, 0.1)
            intercepts = np.where(np.arange(n_components) == 0, 0.1, 0.4)
            p = np.clip(intercepts[Z] + coeffs[Z] * chain_roots[j], 0, 1)
            intermediates.append(rng.binomial(1, p, n_samples))
            self.true_edges.add((self.features[self.n_quarter + j], self.features[2 * self.n_quarter + j]))

        #quarter D: independent roots (indices 3*n_quarter..n_mutations-1)
        independent = [rng.binomial(1, 0.3, n_samples) for _ in range(self.n_quarter)]

        #Y: caused by direct roots and intermediates, coefficients vary across components
        y_sources = direct_roots + intermediates
        n_y = len(y_sources)
        coeffs_y = rng.dirichlet(np.ones(n_y), size=n_components)
        y = sum(coeffs_y[Z, k] * y_sources[k] for k in range(n_y))
        y += rng.normal(0, 0.2, n_samples)
        for k in range(self.n_quarter):
            self.true_edges.add((self.features[k], 'Y'))
        for k in range(self.n_quarter):
            self.true_edges.add((self.features[2 * self.n_quarter + k], 'Y'))

        self.binary_indices = list(range(n_mutations))
        self.X = np.column_stack(direct_roots + chain_roots + intermediates + independent + [y])

        direct_names       = set(self.features[:self.n_quarter])
        intermediate_names = set(self.features[2 * self.n_quarter:3 * self.n_quarter])
        self.true_direct     = {(s, t) for s, t in self.true_edges if s in direct_names}
        self.true_bin_to_bin = {(s, t) for s, t in self.true_edges if t != 'Y'}
        self.true_chain_cont = {(s, t) for s, t in self.true_edges if s in intermediate_names}
        self.independent_features = self.features[3 * self.n_quarter:n_mutations]

    def visualize_true_dag(self, size: str = "20") -> None:
        """Visualize the true causal DAG."""
        bn = gum.BayesNet()
        for i, f in enumerate(self.features):
            if i in self.binary_indices:
                bn.add(gum.LabelizedVariable(f, f, 2))
            else:
                bn.add(gum.RangeVariable(f, f, 0, 1))
        for src, tgt in self.true_edges:
            bn.addArc(src, tgt)
        gnb.showBN(bn, size=size)
