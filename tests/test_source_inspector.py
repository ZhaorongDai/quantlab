"""Home for the read-only source-inspector proofs: ROADMAP success criteria
SC-3 and SC-4, requirements D-08..D-11.

SC-3 — the read surface answers inventory, coverage, failures and browsing with
NO credentials present and zero vendor requests.
SC-4 — the coverage judgement it reports is computed by the same code the real
acquisition run uses, not a second implementation that can drift.

Scaffolded by plan 03.4-01 (Wave 0). `quantlab/acquisition/inspector.py` does
not exist yet; plan 03.4-04 builds it and fills this file in.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents recorded in
`.planning/STATE.md`:

1. EVERY test here must be a real assertion. A pytest file with zero tests
   exits **5** ("no tests ran"), which a per-file command reads as green, so a
   placeholder, a body that is only a no-op statement, or a skip/xfail marker
   is indistinguishable from a passing file.

   On pytest 9.1.1 exit 5 is ALSO what a `-k` selector matching nothing
   produces ("no tests collected (N deselected)"). Which of the two is a bug
   depends on which was expected: for this scaffold file, exit 5 is the
   failure it exists to prevent; for the rule-2 selectors below it is the
   required result. The three tests below each pin a
   piece of infrastructure the plan-04 work depends on, against the code as it
   stands today.

2. A `-k` selector name must not be attached to a test that does not honestly
   cover that selector's behaviour -- in 03.2 a deleted mechanism left
   `-k fingerprint` green because its only covering test was named outside the
   selector. So no test here is named for a selector `03.4-VALIDATION.md`
   assigns to a later plan (`without_credentials`, `zero_vendor_requests`,
   `inspector_binds_no_client`, `coverage_is_the_same_code`,
   `browse_requires_symbols_and_window`, `browse_prunes`). Those must match
   ZERO tests until the behaviour exists.
"""

import re
from pathlib import Path

import xarray as xr

# ---------------------------------------------------------------------------
# Copied VERBATIM from `tests/test_raw_hive_layout.py:161-169`.
#
# A copy rather than a cross-test import on purpose: `tests/` is not a package
# (there is no `tests/__init__.py`; `pyproject.toml` sets only
# `pythonpath = ["."]`), so importing one test module from another would make
# this file's collection depend on the other file's import-time state. The
# duplication is bounded to nine lines and is kept honest by
# `test_scan_source_count_reads_both_polars_plan_renderings` below, which is
# itself a copy of the origin file's own negative control.
# ---------------------------------------------------------------------------

#: polars ABBREVIATES a long scan source list rather than printing every path:
#:
#:     Parquet SCAN [a/part.pqt, ... 4 other sources]
#:
#: so a naive `explain().count(".pqt")` returns 1 for an UNPRUNED five-file
#: scan and 2 for a pruned two-file one -- exactly backwards, and a pruning
#: test built on it would pass while asserting the opposite of the truth. The
#: abbreviation threshold is polars' business and may change, so both forms are
#: parsed here.
_OTHER_SOURCES = re.compile(r"\.\.\.\s*(\d+)\s*other sources?")


def _scan_source_count(plan: str) -> int:
    """How many parquet files the query plan will actually open."""
    listed = len(re.findall(r"\.pqt", plan))
    hidden = sum(int(match) for match in _OTHER_SOURCES.findall(plan))
    return listed + hidden


def test_scan_source_count_reads_both_polars_plan_renderings() -> None:
    """Self-test for the counter D-10/D-11's pruning assertion will depend on.

    If this helper miscounts, plan 04's `browse_prunes` test asserts nothing
    useful -- and its failure mode is to PASS, because the naive count is
    wrong in the direction that makes an unpruned scan look pruned. Both
    renderings are pinned by example in ONE test, so the helper cannot rot into
    a tautology by having only the form it happens to handle exercised.
    """
    abbreviated = "Parquet SCAN [d/month=2024-01/part-a.pqt, ... 4 other sources]"
    explicit = "Parquet SCAN [d/month=2024-04/part-a.pqt, d/month=2024-05/part-a.pqt]"

    assert _scan_source_count(abbreviated) == 5
    assert _scan_source_count(explicit) == 2


def test_the_hive_raw_tree_fixture_writes_vendor_namespaced_shards(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
) -> None:
    """Two vendors under one parent stay in separate subtrees.

    `browse_raw` (D-10) scans a vendor-terminated raw root. That is only safe
    because the shards of one vendor are unreachable from the other vendor's
    root -- if they were not, a browse rooted at `tiingo/` would silently
    return Alpaca rows and the LazyFrame would carry a cross-vendor union that
    nothing downstream could detect.

    Asserted here, on the fixture, rather than assumed by plan 04's tests: the
    separation is a property of how `hive_raw_tree` lays out its directories,
    and a change to that layout would break the later isolation proofs while
    leaving them looking green.
    """
    root = tmp_path / "downloads"
    rows = [stock_pqt_row("2024-01-15", "AAPL"), stock_pqt_row("2024-02-15", "AAPL")]

    tiingo_root = hive_raw_tree(root, "tiingo", rows)
    alpaca_root = hive_raw_tree(root, "alpaca", rows)

    assert tiingo_root == root / "tiingo"
    assert alpaca_root == root / "alpaca"

    tiingo_shards = sorted(tiingo_root.rglob("*.pqt"))
    alpaca_shards = sorted(alpaca_root.rglob("*.pqt"))

    assert tiingo_shards, "fixture wrote no tiingo shards"
    assert alpaca_shards, "fixture wrote no alpaca shards"

    # Every shard sits under `{root}/{vendor}/{hive_key}=.../`.
    for vendor_root, shards in (
        (tiingo_root, tiingo_shards),
        (alpaca_root, alpaca_shards),
    ):
        for shard in shards:
            assert shard.parent.parent == vendor_root
            assert shard.parent.name.startswith("month=")

    # Neither vendor's shards are reachable from the other's root.
    assert not set(tiingo_shards) & set(alpaca_shards)
    for shard in alpaca_shards:
        assert tiingo_root not in shard.parents
    for shard in tiingo_shards:
        assert alpaca_root not in shard.parents


def test_the_stock_zarr_fixture_opens_and_carries_a_symbol_index(
    stock_zarr,
) -> None:
    """The Zarr-tier contract `browse_zarr` (D-10) is built on.

    `browse_zarr` must return a selection over `symbol` and `timestamp`
    without materialising the store, which presupposes that both are real
    INDEXES rather than plain coordinates -- `.sel(symbol=[...])` raises if
    `symbol` is not indexed. Pinning it on the fixture means plan 04 can assert
    the selection behaviour without first re-proving that its own test store is
    shaped the way the production store is.
    """
    config = stock_zarr(symbols=["AAPL", "MSFT"], periods=10)

    dataset = xr.open_zarr(config.zarr_file_path)
    try:
        assert "symbol" in dataset.indexes
        assert "timestamp" in dataset.indexes

        selected = dataset.sel(symbol=["AAPL"])
        assert list(selected.symbol.values) == ["AAPL"]
        assert selected.sizes["timestamp"] == 10
    finally:
        dataset.close()
