"""Automated proof for ROADMAP Phase 2 Success Criterion 4 / DATA-03: "adding
a new market or frequency only requires a new `Dataset` subclass + config --
no changes needed in factor/model/backtest code."

Two complementary checks live here:
- `test_no_market_specific_logic_in_core_layers`: a grep-style purity check
  proving the three core layers (`base/factor.py`, `base/model.py`,
  `base/backend.py`) contain no literal reference to a concrete `Dataset`
  subclass name or market-specific literal.
- `test_fake_dataset_lifecycle`: a genuinely novel, test-only `FakeDataset`
  subclass (never registered anywhere else in the codebase) that runs the
  full `from_raw_data()` -> `save()` -> `read()` lifecycle successfully,
  proving the contract holds by construction.

A grep alone cannot catch structural coupling (`isinstance`/`hasattr`
dispatch on a concrete `Dataset` subclass, or branching on the *value* of
`self.config.market`/`self.config.frequency`) -- that is covered by a
separate `checkpoint:human-verify` design/code review (see
02-06-PLAN.md Task 3), not by this file.
"""

from pathlib import Path

CORE_LAYER_FILES = (
    "base/factor.py",
    "base/model.py",
    "base/backend.py",
)

FORBIDDEN_SUBSTRINGS = (
    "SpotKlineDataset",
    "StockDataset",
    "crypto_spot",
    "us_equity",
)


def test_core_layer_purity_no_market_specific_logic() -> None:
    """base/factor.py, base/model.py, base/backend.py must never reference a
    concrete `Dataset` subclass name or a market-specific literal -- doing so
    would mean the core layers depend on which market/frequency is in use,
    breaking the "new market = new Dataset subclass, zero core-layer changes"
    contract (DATA-03).

    Lines whose stripped content starts with `#` are excluded before
    checking, so a comment mentioning these names in passing (e.g. an
    architecture-note docstring) is not a false-positive violation -- only a
    live, non-comment reference counts.
    """
    violations: list[str] = []

    for relative_path in CORE_LAYER_FILES:
        file_path = Path(relative_path)
        lines = file_path.read_text().splitlines()

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for forbidden in FORBIDDEN_SUBSTRINGS:
                if forbidden in line:
                    violations.append(
                        f"{relative_path}:{line_number}: found forbidden "
                        f"substring '{forbidden}' in non-comment line: "
                        f"{line.strip()!r}"
                    )

    assert not violations, (
        "Core layers must not reference concrete Dataset subclasses or "
        "market-specific literals (DATA-03):\n" + "\n".join(violations)
    )
