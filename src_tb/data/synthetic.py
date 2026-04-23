import numpy as np
import pyagrum as gum
import pyagrum.lib.notebook as gnb


def generate_synthetic(n_mutations: int, n_samples: int = 164, seed: int = 0):
    """Generate binary mutation data with known causal structure.

    Structure: first half of mutations are roots, second half each caused by one root
    (binary to binary edges). Y (continuous, last column) caused by all roots (binary to
    continuous edges). Latent Z creates two subpopulations with different mechanisms.

    Returns X, features, true_edges, binary_indices.
    """
    assert n_mutations >= 2 and n_mutations % 2 == 0, "n_mutations must be even and >= 2"
    rng = np.random.default_rng(seed)
    n_roots = n_mutations // 2

    Z = rng.binomial(1, 0.4, n_samples)
    cols = []
    true_edges = set()
    features = [f'mut_{i}' for i in range(n_mutations)] + ['Y']

    # roots: binary, no parents
    for i in range(n_roots):
        cols.append(rng.binomial(1, 0.3, n_samples))

    # caused: each caused by one root, same mixture pattern as binary to continuous edges
    for j in range(n_roots):
        x = np.where(Z == 0,
            rng.binomial(1, np.clip(0.1 + 0.7 * cols[j], 0, 1), n_samples),
            rng.binomial(1, np.clip(0.4 + 0.1 * cols[j], 0, 1), n_samples))
        cols.append(x)
        true_edges.add((features[j], features[n_roots + j]))

    # Y: continuous, caused by all roots, mechanism varies by Z
    coeff_z0 = rng.dirichlet(np.ones(n_roots))
    coeff_z1 = rng.dirichlet(np.ones(n_roots))
    y = sum(np.where(Z == 0, coeff_z0[k] * cols[k], coeff_z1[k] * cols[k]) for k in range(n_roots))
    y += rng.normal(0, 0.2, n_samples)
    cols.append(y)
    for k in range(n_roots):
        true_edges.add((features[k], 'Y'))

    binary_indices = list(range(n_mutations))
    return np.column_stack(cols), features, true_edges, binary_indices


def visualize_synthetic_dag(true_edges: set, features: list, size: str = "20"):
    """Visualize the true causal DAG."""
    bn = gum.BayesNet()
    for f in features:
        bn.add(gum.LabelizedVariable(f, f, 2))
    for src, tgt in true_edges:
        bn.addArc(src, tgt)
    gnb.showBN(bn, size=size)


def eval_recovery(stable_set: set, true_edges: set):
    """Compute precision, recall, F1 of recovered stable edges vs true edges."""
    tp = len(stable_set & true_edges)
    fp = len(stable_set - true_edges)
    fn = len(true_edges - stable_set)
    pr = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    re = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * pr * re / (pr + re) if (pr + re) > 0 else 0.0
    return round(pr, 3), round(re, 3), round(f1, 3)
