#!/usr/bin/env python3
"""Prove the test suite would notice broken logic: break each operator, expect a red test."""
import argparse
import ast
import copy
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = {
    "metrics.py": "tests/test_metrics.py",
    "stats.py": "tests/test_stats.py",
    "gate.py": "tests/test_gate.py",
    "retrieval.py": "tests/test_retrieval.py",
    "faithfulness.py": "tests/test_faithfulness.py",
}
COMPARE = {
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
ARITHMETIC = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult}
# Mutations that cannot change behaviour, keyed by (module, source fragment, mutation).
EQUIVALENT = {
    ("stats.py", "extreme = np.count_nonzero", "LtE -> Lt"): "the tie tolerance already absorbs exact equality",
    ("retrieval.py", "fused[int(index)] +=", "1 -> 2"): "scaling every fused score keeps the ranking",
}


class Mutator(ast.NodeTransformer):
    """Apply the target-th possible mutation; with target -1, just count the sites."""

    def __init__(self, target):
        self.target = target
        self.sites = 0
        self.label = None
        self.line = None

    def hit(self, node, label):
        index, self.sites = self.sites, self.sites + 1
        if index == self.target:
            self.label = label
            self.line = node.lineno
            return True
        return False

    def visit_FunctionDef(self, node):
        if node.name == "main":
            return node
        node.body = [self.visit(statement) for statement in node.body]
        return node

    def visit_ClassDef(self, node):
        node.body = [self.visit(s) if isinstance(s, ast.FunctionDef) else s for s in node.body]
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        for i, op in enumerate(node.ops):
            swap = COMPARE.get(type(op))
            if swap and self.hit(node, f"{type(op).__name__} -> {swap.__name__}"):
                node.ops[i] = swap()
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        flipped = ast.Or if isinstance(node.op, ast.And) else ast.And
        if self.hit(node, f"{type(node.op).__name__} -> {flipped.__name__}"):
            node.op = flipped()
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        swap = ARITHMETIC.get(type(node.op))
        if swap and self.hit(node, f"{type(node.op).__name__} -> {swap.__name__}"):
            node.op = swap()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Not) and self.hit(node, "removed not"):
            return node.operand
        if isinstance(node.op, ast.USub) and self.hit(node, "removed unary minus"):
            return node.operand
        return node

    def visit_Constant(self, node):
        if type(node.value) in (int, float) and self.hit(node, f"{node.value} -> {node.value + 1}"):
            return ast.Constant(node.value + 1)
        return node


def _body(tree):
    tree.body = [node for node in tree.body if not isinstance(node, ast.If)]  # the __main__ guard
    return tree


def test_mutant(job):
    module, target, tree, test_file = job
    mutator = Mutator(target)
    mutant = mutator.visit(copy.deepcopy(tree))
    with tempfile.TemporaryDirectory(prefix="ragate-mutant-") as directory:
        workspace = Path(directory)
        shutil.copytree(ROOT / "ragate", workspace / "ragate", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "tests", workspace / "tests", ignore=shutil.ignore_patterns("__pycache__"))
        (workspace / "ragate" / module).write_text(ast.unparse(mutant) + "\n", encoding="utf-8")
        command = [sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", test_file]
        try:
            passed = subprocess.run(command, cwd=workspace, capture_output=True, timeout=240).returncode == 0
        except subprocess.TimeoutExpired:
            passed = False
    return (module, mutator.line, mutator.label) if passed else None


def is_equivalent(survivor):
    module, line, label = survivor
    source = (ROOT / "ragate" / module).read_text(encoding="utf-8").splitlines()[line - 1]
    return any(m == module and fragment in source and l == label for m, fragment, l in EQUIVALENT)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    jobs = []
    for module, test_file in MODULES.items():
        tree = _body(ast.parse((ROOT / "ragate" / module).read_text(encoding="utf-8")))
        counter = Mutator(-1)
        counter.visit(copy.deepcopy(tree))
        jobs.extend((module, target, tree, test_file) for target in range(counter.sites))
    with ThreadPoolExecutor(args.jobs) as pool:
        survivors = [survivor for survivor in pool.map(test_mutant, jobs) if survivor]
    equivalent = [survivor for survivor in survivors if is_equivalent(survivor)]
    live = [survivor for survivor in survivors if survivor not in equivalent]
    for module, line, label in live:
        print(f"SURVIVED {module} line {line}: {label}")
    judged = len(jobs) - len(equivalent)
    print(f"{judged - len(live)}/{judged} mutants killed, {len(equivalent)} equivalent excluded")
    return 1 if live else 0


if __name__ == "__main__":
    sys.exit(main())
