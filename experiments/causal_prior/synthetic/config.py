"""Shared experiment configuration for the §6.1 mechanism test.

DEFAULTS is the canonical operating point (the p=30 n=300 headline shape);
each sweep in SWEEPS varies one axis at a time holding the others at
DEFAULTS. The default value is included in every sweep so it acts as a
reference point in cross-sweep plots.

The 'p_edge' sweep produces the headline figures (selectivity-driven
recovery as a function of confounding); the other three are robustness
slices defending the anchor cell against op-point cherry-pick concerns.
CLI overrides (--p, --n, --k-star in stability_selection.py) allow
shooting at alternate op-points without touching DEFAULTS.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULTS = dict(p=30, n=300, k_star=5, p_edge=0.2)

SWEEPS: dict[str, list] = {
    'p_edge':  [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],  # headline at p=30, n=300 op-point; robustness at p=15
    'n':       [150, 200, 300, 500, 1000],  # sample-size robustness; 150 keeps a TB-adjacent point
    'p':       [10, 15, 20, 30],         # feature-count robustness; p>=20 probes GES degradation
    'k_star':  [3, 5, 7],                # true-sparsity robustness
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


def build_cells(scope: str, n_seeds: int = DEFAULT_N_SEEDS,
                overrides: dict | None = None) -> list[Cell]:
    """Return the (deduplicated, sorted) cells for one sweep or all sweeps.

    scope: a sweep name from SWEEPS, or 'all' to take the union across sweeps.
    overrides: optional dict of {p, n, k_star, p_edge} values to override
        DEFAULTS for the base shape; the sweep axis still varies on top.
    """
    if scope == 'all':
        sweep_names = list(SWEEPS.keys())
    elif scope in SWEEPS:
        sweep_names = [scope]
    else:
        raise ValueError(f'unknown scope {scope!r}; must be one of '
                         f'{list(SWEEPS.keys()) + ["all"]}')

    base = {**DEFAULTS, **(overrides or {})}
    cells: set[Cell] = set()
    for sweep in sweep_names:
        for val in SWEEPS[sweep]:
            params = {**base, sweep: val}
            for seed in range(n_seeds):
                cells.add(Cell(seed=seed, **params))
    return sorted(cells)
