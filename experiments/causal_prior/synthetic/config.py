"""Shared experiment configuration for the §6.1 mechanism test.

DEFAULTS is the canonical operating point; each sweep in SWEEPS varies one
axis at a time, holding the others at DEFAULTS. The default value is included
in every sweep so it acts as a reference point in cross-sweep plots.

The 'p_edge' sweep is the headline (causal-vs-predictive gap as a function of
confounding); the other three are robustness slices.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULTS = dict(p=30, n=500, k_star=5, p_edge=0.2)

SWEEPS: dict[str, list] = {
    'p_edge':  [0.1, 0.2, 0.4],   # headline: confounding axis
    'n':       [300, 500, 1000],  # sample-size robustness
    'p':       [10, 30, 50],      # feature-count robustness
    'k_star':  [3, 5, 7],         # true-sparsity robustness
}

PARAM_LABELS = {
    'p_edge':  'DAG edge density',
    'n':       'sample size',
    'p':       '#features',
    'k_star':  '|S*| (true causes)',
}

DEFAULT_N_SEEDS = 5


@dataclass(frozen=True, order=True)
class Cell:
    p: int
    n: int
    k_star: int
    p_edge: float
    seed: int

    @property
    def filename(self) -> str:
        return (f'seed{self.seed}_p{self.p}_n{self.n}_k{self.k_star}'
                f'_pedge{self.p_edge}.npz')


def build_cells(scope: str, n_seeds: int = DEFAULT_N_SEEDS) -> list[Cell]:
    """Return the (deduplicated, sorted) cells for one sweep or all sweeps.

    scope: a sweep name from SWEEPS, or 'all' to take the union across sweeps.
    """
    if scope == 'all':
        sweep_names = list(SWEEPS.keys())
    elif scope in SWEEPS:
        sweep_names = [scope]
    else:
        raise ValueError(f'unknown scope {scope!r}; must be one of '
                         f'{list(SWEEPS.keys()) + ["all"]}')

    cells: set[Cell] = set()
    for sweep in sweep_names:
        for val in SWEEPS[sweep]:
            params = {**DEFAULTS, sweep: val}
            for seed in range(n_seeds):
                cells.add(Cell(seed=seed, **params))
    return sorted(cells)
