"""Shared experiment configuration for the mechanism test.

DEFAULTS is the canonical operating point (the p=30 n=300 headline shape);
each sweep in SWEEPS varies one axis at a time holding the others at
DEFAULTS. The default value is included in every sweep so it acts as a
reference point in cross-sweep plots.

The 'p_edge' sweep produces the headline figures (selectivity-driven
recovery as a function of confounding); 'n' is the sample-efficiency
panel; 'p', 'k_star', and 'noise_scale' are robustness slices defending
the anchor cell against op-point cherry-pick concerns. CLI overrides
(--p, --n, --k-star in stability_selection.py) allow shooting at
alternate op-points without touching DEFAULTS.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULTS = dict(p=30, n=300, k_star=5, p_edge=0.2, noise_scale=1.0)

SWEEPS: dict[str, list] = {
    'p_edge':      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],  # headline; 0.7 added for extreme-confounding tail
    'n':           [75, 100, 150, 200, 300, 500, 1000],  # sample-efficiency; 75/100 probe the data-starved regime
    'p':           [10, 15, 20, 30, 50],                 # feature-count robustness; 50 stresses GES
    'k_star':      [3, 5, 7],                            # true-sparsity robustness
    'noise_scale': [0.5, 1.0, 2.0],                       # SNR sweep; prior should help more at high noise
}

# per-sweep seed budget; p_edge is the headline figure so it gets the most.
# n is the sample-efficiency panel (also load-bearing); p, k_star, and
# noise_scale are robustness slices, less central but still need readable bands.
SEED_COUNTS: dict[str, int] = {
    'p_edge':      20,
    'n':           10,
    'p':           10,
    'k_star':      10,
    'noise_scale': 10,
}

PARAM_LABELS = {
    'p_edge':      'DAG edge density',
    'n':           'sample size',
    'p':           '#features',
    'k_star':      '|S*| (true causes)',
    'noise_scale': r'noise scale $\sigma$',
}

DEFAULT_N_SEEDS = 5  # used only when CLI --n-seeds overrides SEED_COUNTS uniformly


@dataclass(frozen=True, order=True)
class Cell:
    p: int
    n: int
    k_star: int
    p_edge: float
    noise_scale: float
    seed: int

    @property
    def filename(self) -> str:
        # only suffix noise when it differs from the default, so cells generated
        # before the noise_scale sweep keep their original names (existing cache
        # and recovery CSVs stay valid; no orphan/duplicate files in the dir).
        base = (f'seed{self.seed}_p{self.p}_n{self.n}_k{self.k_star}'
                f'_pedge{self.p_edge}')
        if self.noise_scale != DEFAULTS['noise_scale']:
            base += f'_noise{self.noise_scale}'
        return f'{base}.npz'


def build_cells(scope: str, n_seeds: int | None = None,
                overrides: dict | None = None) -> list[Cell]:
    """Return the (deduplicated, sorted) cells for one sweep or all sweeps.

    scope: a sweep name from SWEEPS, or 'all' to take the union across sweeps.
    n_seeds: if None, use per-sweep SEED_COUNTS; if set, override uniformly
        (CLI --n-seeds path).
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
        seeds_for_sweep = (n_seeds if n_seeds is not None
                           else SEED_COUNTS.get(sweep, DEFAULT_N_SEEDS))
        for val in SWEEPS[sweep]:
            params = {**base, sweep: val}
            for seed in range(seeds_for_sweep):
                cells.add(Cell(seed=seed, **params))
    return sorted(cells)
