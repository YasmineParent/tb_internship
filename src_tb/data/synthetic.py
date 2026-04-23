import numpy as np
import pyagrum as gum
import pyagrum.lib.notebook as gnb


def generate_synthetic(n_mutations: int, n_samples: int = 164, n_components: int = 2, seed: int = 0) -> tuple[np.ndarray, list[str], set[tuple[str, str]], list[int]]:
    """Generate binary mutation data with known causal structure.

    Structure (n_mutations is rounded down to the nearest multiple of 4, minimum 4):
      - Quarter A: direct roots, each causes Y directly (binary to continuous).
      - Quarter B: chain roots, each causes one intermediate mutation (binary to binary),
        which then causes Y (binary to continuous).
      - Quarter C: intermediate mutations caused by quarter B roots.
      - Quarter D: independent roots with no path to Y (tests false positive rate).
      Latent Z has n_components subpopulations with different mechanisms for all edges.

    For n_mutations=4: mut_0 -> Y, mut_1 -> mut_2 -> Y, mut_3 independent.
    Returns X, features, true_edges, binary_indices.
    """
    assert n_components >= 2, "n_components must be >= 2"
    rng = np.random.default_rng(seed)
    n_quarter = max(1, n_mutations // 4)
    n_mutations = n_quarter * 4

    Z = rng.choice(n_components, size=n_samples, p=rng.dirichlet(np.ones(n_components)))
    features = [f'mut_{i}' for i in range(n_mutations)] + ['Y']
    true_edges = set()

    #quarter A: direct roots (indices 0..n_quarter-1)
    direct_roots = [rng.binomial(1, 0.3, n_samples) for _ in range(n_quarter)]

    #quarter B: chain roots (indices n_quarter..2*n_quarter-1)
    chain_roots = [rng.binomial(1, 0.3, n_samples) for _ in range(n_quarter)]

    #quarter C: intermediates caused by chain roots (indices 2*n_quarter..3*n_quarter-1)
    #coefficients per component: strong in component 0, weak in others
    intermediates = []
    for j in range(n_quarter):
        coeffs = np.where(np.arange(n_components) == 0, 0.7, 0.1)
        intercepts = np.where(np.arange(n_components) == 0, 0.1, 0.4)
        p = np.clip(intercepts[Z] + coeffs[Z] * chain_roots[j], 0, 1)
        intermediates.append(rng.binomial(1, p, n_samples))
        true_edges.add((features[n_quarter + j], features[2 * n_quarter + j]))

    #quarter D: independent roots (indices 3*n_quarter..n_mutations-1)
    independent = [rng.binomial(1, 0.3, n_samples) for _ in range(n_quarter)]

    #Y: caused by direct roots and intermediates, coefficients vary across components
    y_sources = direct_roots + intermediates
    n_y = len(y_sources)
    coeffs_y = rng.dirichlet(np.ones(n_y), size=n_components)
    y = sum(coeffs_y[Z, k] * y_sources[k] for k in range(n_y))
    y += rng.normal(0, 0.2, n_samples)
    for k in range(n_quarter):
        true_edges.add((features[k], 'Y'))
    for k in range(n_quarter):
        true_edges.add((features[2 * n_quarter + k], 'Y'))

    cols = direct_roots + chain_roots + intermediates + independent + [y]
    binary_indices = list(range(n_mutations))
    return np.column_stack(cols), features, true_edges, binary_indices


def visualize_synthetic_dag(true_edges: set, features: list, binary_indices: list[int], size: str = "20") -> None:
    """Visualize the true causal DAG."""
    bn = gum.BayesNet()
    for i, f in enumerate(features):
        if i in binary_indices:
            bn.add(gum.LabelizedVariable(f, f, 2))
        else:
            bn.add(gum.RangeVariable(f, f, 0, 1))
    for src, tgt in true_edges:
        bn.addArc(src, tgt)
    gnb.showBN(bn, size=size)


