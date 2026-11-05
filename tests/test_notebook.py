"""The notebook's code reads well and its sections cannot overwrite each other's names.

All sections share one namespace. A name that one section assigns and a later section assigns again changes what the
earlier name means by the time it is read (it happened once: a later section overwrote section 6's `same`, and the
closing checklist quoted the wrong number). So no two sections may bind the same top-level name, except a figure and
its headline (`fig`, `finding`) and a few loop variables, where every binding is a `for` loop.
"""

import ast
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook.ipynb"
ALLOWED = {"fig", "finding"}
LOOP_VARIABLES = {"c", "p", "n", "line", "i", "j", "k"}
SECTION = re.compile(r"^## (?:(\d+)\.|(The playbook)|(Appendix))", re.MULTILINE)


def code_cells():
    """(section, source) of every code cell; a section starts at its heading, the setup before section 1."""
    if not NOTEBOOK.exists():
        pytest.skip("no notebook.ipynb beside the package")
    section = "setup"
    for cell in json.loads(NOTEBOOK.read_text())["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and (m := SECTION.search(source)):
            section = next(g for g in m.groups() if g)
        elif cell["cell_type"] == "code":
            yield section, source


def bindings(body, kinds):
    """Fill {name: {"loop", "assign"}} with the names these statements bind in the notebook's namespace."""
    for s in body:
        if isinstance(s, ast.FunctionDef | ast.ClassDef):
            kinds.setdefault(s.name, set()).add("assign")
            continue
        if isinstance(s, ast.Import | ast.ImportFrom):
            for a in s.names:
                kinds.setdefault((a.asname or a.name).split(".")[0], set()).add("assign")
            continue
        targets = {ast.Assign: lambda s: s.targets, ast.AugAssign: lambda s: [s.target]}
        targets |= {ast.AnnAssign: lambda s: [s.target], ast.For: lambda s: [s.target]}
        targets |= {ast.With: lambda s: [i.optional_vars for i in s.items if i.optional_vars]}
        for t in targets.get(type(s), lambda s: [])(s):
            for n in ast.walk(t):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    kinds.setdefault(n.id, set()).add("loop" if isinstance(s, ast.For) else "assign")
        for field in ("body", "orelse", "finalbody"):
            bindings(getattr(s, field, None) or [], kinds)
        for handler in getattr(s, "handlers", None) or []:
            bindings(handler.body, kinds)


def test_no_two_sections_bind_the_same_name():
    sections, kinds = {}, {}
    for section, source in code_cells():
        bound = {}
        bindings(ast.parse(source).body, bound)
        for name, how in bound.items():
            sections.setdefault(name, set()).add(section)
            kinds.setdefault(name, set()).update(how)
    shared = {name: secs for name, secs in sections.items() if len(secs) > 1}
    bad = {
        name: sorted(secs)
        for name, secs in shared.items()
        if name not in ALLOWED and not (name in LOOP_VARIABLES and kinds[name] == {"loop"})
    }
    assert not bad, f"names bound in more than one section: {bad}"


def test_code_lines_fit_in_100_columns():
    long = [(section, line) for section, source in code_cells() for line in source.splitlines() if len(line) > 100]
    assert not long, long[:5]
