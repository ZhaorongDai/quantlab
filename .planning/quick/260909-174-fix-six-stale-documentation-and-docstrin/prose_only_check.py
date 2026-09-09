#!/usr/bin/env python3
"""Prove that a set of `.py` files changed in PROSE ONLY, relative to a base SHA.

Usage:

    prose_only_check.py BASE_SHA FILE [FILE ...]

For each file the base version (`git show BASE_SHA:FILE`) and the working-tree
version are parsed with `ast` and normalised so that only PROSE differences
disappear:

  * every `str`-valued `Constant` becomes one placeholder constant — so
    docstrings, assertion messages and any other string literal are free to
    change;
  * a `JoinedStr` (f-string) keeps only its `FormattedValue` children — the
    literal chunks are dropped, but every interpolated EXPRESSION survives, so
    changing what an f-string computes still shows up;
  * comments never enter the AST at all, so comment edits pass by construction.

Anything else — a condition, a call, an argument, control flow, an operator —
survives normalisation and makes the two dumps differ.

Exit 0 when every file is prose-only; exit 1 naming the offending files
otherwise.  Comparing against a base SHA rather than the working tree is
deliberate: GSD commits per task, so a working-tree-only comparison would be a
proof that cannot fail (the 03.1 D-06 precedent).
"""

from __future__ import annotations

import ast
import subprocess
import sys

_PLACEHOLDER = "<gsd-prose-placeholder>"


class _Normalise(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant(value=_PLACEHOLDER), node)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        self.generic_visit(node)
        kept = [v for v in node.values if isinstance(v, ast.FormattedValue)]
        return ast.copy_location(ast.JoinedStr(values=kept), node)


def _normalised_dump(source: str, filename: str) -> str:
    tree = ast.parse(source, filename=filename)
    tree = _Normalise().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree)


def _base_source(base_sha: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{base_sha}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print("usage: prose_only_check.py BASE_SHA FILE [FILE ...]", file=sys.stderr)
        return 2

    base_sha, paths = argv[1], argv[2:]
    offenders: list[str] = []

    for path in paths:
        before = _normalised_dump(_base_source(base_sha, path), f"{base_sha}:{path}")
        with open(path, encoding="utf-8") as handle:
            after = _normalised_dump(handle.read(), path)
        if before != after:
            offenders.append(path)

    if offenders:
        print(f"NOT prose-only (executable structure changed vs {base_sha}):")
        for path in offenders:
            print(f"  {path}")
        return 1

    print(f"prose-only confirmed vs {base_sha} for {len(paths)} file(s):")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
