"""StockFit stands alone: no StockFit module imports another vendor's dataset code.

StockFit's panels are keyed by its own identifiers (CIK, CUSIP, fund series)
and read only its own raw tier, so neither the dataset nor the acquisition
package may import ``quantlab.dataset.sharadar`` or ``quantlab.dataset.massive``,
at module level or inside a function.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "quantlab"

#: The StockFit packages.
PACKAGES = (ROOT / "dataset" / "stockfit", ROOT / "acquisition" / "stockfit")

#: The other vendors' dataset packages StockFit must not reach.
FORBIDDEN = ("quantlab.dataset.sharadar", "quantlab.dataset.massive")


def _imports(path: Path) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(path.read_text(), str(path))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found += [node.module] + [f"{node.module}.{alias.name}" for alias in node.names]
    return found


@pytest.mark.parametrize("path", sorted(p for package in PACKAGES for p in package.rglob("*.py")), ids=str)
def test_no_stockfit_module_imports_sharadar_or_massive(path):
    reached = [name for name in _imports(path) if name.startswith(FORBIDDEN)]
    assert not reached, f"{path.relative_to(ROOT.parent)} imports {reached}"
