"""build paper figures: python -m src.causal_prior.figures [list|all|<name> ...]

the figures package owns both the builders and this CLI; figures/ holds only the
rendered pdfs.
"""
from __future__ import annotations

import argparse

from . import specs
from .. import figstyle


def _list() -> None:
    ss = specs()
    if not ss:
        print('no figures registered yet.')
        return
    for s in ss:
        print(f'  {s.kind:8s}  {s.name}')


def _build(names: list[str]) -> None:
    ss = {s.name: s for s in specs()}
    todo = list(ss.values()) if names == ['all'] else [ss[n] for n in names]
    figstyle.use_paper_style()
    for s in todo:
        result = s.fn()
        save = figstyle.save_tex if isinstance(result, str) else figstyle.save
        print(f'wrote {save(result, s.name, s.kind).relative_to(figstyle.ROOT)}')


def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m src.causal_prior.figures',
                                 description='build paper figures')
    ap.add_argument('target', nargs='*', help="'list', 'all', or figure names")
    target = ap.parse_args().target or ['list']

    if target == ['list']:
        _list()
        return
    known = {s.name for s in specs()}
    unknown = [n for n in target if n != 'all' and n not in known]
    if unknown:
        ap.error(f'unknown figure(s): {", ".join(unknown)}')
    _build(target)


if __name__ == '__main__':
    main()
