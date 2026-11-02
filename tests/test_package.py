"""The package's shape: what `from vccpipe import *` gives the notebook, and how the modules import each other."""

import ast
import importlib
from pathlib import Path

import vccpipe

PKG = Path(vccpipe.__file__).parent
MODULES = sorted(p.stem for p in PKG.glob("*.py") if p.stem != "__init__")


def test_every_module_lists_its_public_names():
    for name in MODULES:
        module = importlib.import_module(f"vccpipe.{name}")
        tree = ast.parse((PKG / f"{name}.py").read_text())
        defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    names = target.elts if isinstance(target, ast.Tuple) else [target]
                    defined |= {t.id for t in names if isinstance(t, ast.Name)}
        public = {n for n in defined if not n.startswith("_")} - {"__all__"}
        assert set(module.__all__) == public, name


def test_the_package_exports_each_module_once():
    exported = vccpipe.__all__
    assert len(exported) == len(set(exported))
    for name in exported:
        assert hasattr(vccpipe, name), name


def test_no_star_imports_between_modules():
    for name in MODULES:
        tree = ast.parse((PKG / f"{name}.py").read_text())
        stars = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names)]
        assert not stars, f"{name} star-imports {[s.module for s in stars]}"


def test_every_function_has_a_docstring():
    for path in PKG.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                assert ast.get_docstring(node), f"{path.name}: {node.name}"
