"""Vendor-isolated, hive-partitioned raw tier (03.2 SC-7, D-08/D-11).

Measured against polars 1.44.1 during phase research: two vendors writing the
SAME schema under one scan root do not raise, do not warn, and do not record
which vendor each row came from -- `pl.scan_parquet` simply returns their union.
`dedup_raw_frame(keep="last")` then collapses the duplicate `(timestamp, symbol)`
pairs to one, arbitrarily. The result is an untraceable blended price series
that looks exactly like clean data.

A `.../{vendor}/...` path segment alone stops none of that: it carries no `=`,
so it is not a hive key, and a scan rooted one level up walks straight into
both. Three mutually reinforcing measures make SC-7 structural, and this module
is where they are pinned:

1. `raw_data_dir_path` TERMINATES at the vendor segment, and `_scan_raw`
   asserts `Path(raw_data_dir_path).name == config.vendor`;
2. every raw shard carries a literal `vendor` column, so provenance survives
   even a merged read and the condition is DETECTABLE, not merely prevented --
   `_scan_raw` asserts `n_unique(vendor) == 1` and then drops the column;
3. the hive key is frequency-dependent (`month=` for `1d`, `date=` for `1m`,
   `date=/symbol=` for `tick`) and declared once so writer and reader cannot
   drift. Filtering on the hive key prunes at plan time; filtering on the
   `timestamp` data column does not. Both filters are required.

`extra_columns` / `missing_columns` must stay at their polars defaults. The
`SchemaError` a mixed-schema scan raises is a FEATURE here; setting
`extra_columns='ignore'` "to make the scan work" reopens the silent merge while
looking like a bug fix.

Every test here is offline: parquet under `tmp_path` only. No network call, no
credential, no real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-tests; 03.2-02
Task 4 fills in the `.explain()` pruning assertion and the `_scan_raw`
behavioural tests. It is deliberately NOT an empty placeholder: a pytest file
with zero collected tests exits 5 ("no tests ran"), which a later task's
automated command reads as green.
"""

from datetime import datetime
from pathlib import Path

import polars as pl


def _rows(stock_pqt_row, symbol: str, close: float) -> list[dict]:
    """Two rows in two different months, so every vendor tree has more than one
    hive partition and a pruning assertion has something to prune."""
    return [
        stock_pqt_row("2024-01-02", symbol, close),
        stock_pqt_row("2024-02-05", symbol, close),
    ]


def test_hive_raw_tree_builds_the_two_vendor_tree_vendor_isolation_must_prevent(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Fixture self-test: two vendors, one shared parent, sibling roots.

    Asserts the exact tree shape the SC-7 assertions are written against --
    two sibling vendor directories, each holding at least one `month=`
    partition directory, and each row carrying a literal `vendor` column whose
    single distinct value equals that directory's name.

    If any half of this drifts, the isolation tests that follow would be
    asserting against a tree that cannot merge in the first place, and would
    pass while proving nothing.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"

    tiingo_root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    alpaca_root = hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    assert tiingo_root == parent / "tiingo"
    assert alpaca_root == parent / "alpaca"
    assert tiingo_root.parent == alpaca_root.parent == parent

    vendor_dirs = sorted(p.name for p in parent.iterdir() if p.is_dir())
    assert vendor_dirs == ["alpaca", "tiingo"], (
        f"expected exactly two sibling vendor directories under {parent}, got "
        f"{vendor_dirs}"
    )

    for vendor_root in (tiingo_root, alpaca_root):
        partitions = sorted(
            p.name for p in vendor_root.iterdir() if p.name.startswith("month=")
        )
        assert partitions, (
            f"{vendor_root} holds no `month=` partition directory; the hive key "
            f"is what prunes at plan time"
        )
        assert list(vendor_root.rglob("*.pqt")), f"{vendor_root} holds no shard"

        frame = pl.scan_parquet(vendor_root).collect()
        distinct = frame["vendor"].unique().to_list()
        assert distinct == [vendor_root.name], (
            f"every row under {vendor_root} must carry vendor="
            f"{vendor_root.name!r}; found {distinct}"
        )


def test_a_scan_rooted_above_both_vendors_merges_them_silently(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Fixture self-test: the failure mode is REAL, reproduced here offline.

    This test asserts the bug, not the fix. Same schema, same
    `(timestamp, symbol)` keys, different `close` values -- and a scan rooted
    above both returns the union with no error and no warning. That is what
    makes the `Path(raw_data_dir_path).name == vendor` assertion load-bearing
    rather than decorative.

    The literal `vendor` column is what keeps the merged frame diagnosable:
    without it these rows would be indistinguishable and `dedup_raw_frame`
    would pick one arbitrarily.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    merged = pl.scan_parquet(parent).collect()

    assert sorted(merged["vendor"].unique().to_list()) == ["alpaca", "tiingo"], (
        "a scan rooted above both vendors is expected to merge them; if this "
        "ever raises instead, re-derive SC-7's assertions against the new "
        "polars behaviour rather than deleting this test"
    )
    same_key = merged.filter(
        (pl.col("symbol") == "AAPL")
        & (pl.col("timestamp") == pl.lit(stock_pqt_row("2024-01-02", "AAPL")["timestamp"]))
    )
    assert same_key.height == 2, (
        "the merge must produce duplicate (timestamp, symbol) rows -- that "
        "collision is what dedup_raw_frame would silently collapse"
    )
    assert sorted(same_key["close"].to_list()) == [1.0, 9.9]


# ---------------------------------------------------------------------------
# 03.2-02 Task 4 -- hive directory pruning and structural vendor isolation.
#
# The `-k prun` and `-k vendor_isolation` selectors reach the tests below.
# ---------------------------------------------------------------------------

import re

import pytest

from base.config import DatasetConfig
from dataset.stock import StockDataset

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


def _make_config(root: Path, vendor: str = "tiingo", **overrides) -> DatasetConfig:
    kwargs = dict(
        raw_data_dir_path=str(root),
        zarr_file_path=str(root.parent / "out.zarr"),
        catalog_path=str(root.parent / "catalog"),
        market="us_equity",
        frequency="1d",
        vendor=vendor,
        start_date="2024-01-01",
        end_date="2024-05-31",
    )
    kwargs.update(overrides)
    return DatasetConfig(**kwargs)  # type: ignore[arg-type]


def _five_month_rows(stock_pqt_row) -> list[dict]:
    """One row in each of five consecutive months -> five hive partitions."""
    return [
        stock_pqt_row(f"2024-0{month}-15", "AAPL", close=float(month))
        for month in range(1, 6)
    ]


def test_the_source_count_helper_reads_polars_abbreviated_plan_correctly():
    """Self-test for the counter the pruning assertion depends on.

    If this helper miscounts, the pruning test below asserts nothing useful --
    and its failure mode is to pass. Both plan renderings are pinned by
    example.
    """
    abbreviated = "Parquet SCAN [d/month=2024-01/part-a.pqt, ... 4 other sources]"
    explicit = "Parquet SCAN [d/month=2024-04/part-a.pqt, d/month=2024-05/part-a.pqt]"

    assert _scan_source_count(abbreviated) == 5
    assert _scan_source_count(explicit) == 2


def test_scan_raw_prunes_directories_on_the_hive_key_not_on_timestamp(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """RESEARCH Pitfall 2, asserted against a query PLAN.

    The hive key and the `timestamp` data column are different columns: polars
    prunes on the former at plan time and CANNOT prune on the latter at all. A
    mechanical port of `_scan_raw` that kept only the timestamp filter would
    deliver the new directory tree with none of its benefit, and nothing would
    fail -- every query would simply keep opening every file.

    The negative control is in this same test on purpose. Without it the
    assertion could rot into a tautology the day something else starts
    narrowing the plan, and the test would keep passing for the wrong reason.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _five_month_rows(stock_pqt_row))

    assert len(list(root.rglob("*.pqt"))) == 5, "five partitions, five shards"

    dataset = StockDataset(_make_config(root))

    whole = _scan_source_count(dataset._scan_raw().explain())
    narrowed = _scan_source_count(
        dataset._scan_raw("2024-04-01", "2024-05-31").explain()
    )

    assert whole == 5, f"the unnarrowed scan should open every shard, got {whole}"
    assert narrowed < whole, (
        f"narrowing the window to the last two months must open strictly "
        f"fewer files; opened {narrowed} of {whole}. A plan that still lists "
        f"every source means the hive predicate was dropped and only the "
        f"timestamp filter survives -- which prunes nothing."
    )
    assert narrowed == 2, narrowed

    # NEGATIVE CONTROL: the same window expressed ONLY as a timestamp predicate
    # prunes nothing. This is the measured behaviour that makes the hive
    # predicate load-bearing rather than redundant.
    import polars as pl

    raw = pl.scan_parquet(
        root, hive_partitioning=True, hive_schema={"month": pl.String}
    )
    timestamp_only = raw.filter(
        pl.col("timestamp") >= pl.lit(datetime.fromisoformat("2024-04-01"))
    )
    assert _scan_source_count(timestamp_only.explain()) == whole, (
        "a timestamp-only predicate is expected to prune NOTHING. If this "
        "ever starts pruning, re-derive the two-predicate design against the "
        "new polars behaviour rather than deleting the hive filter."
    )


def test_vendor_isolation_a_root_above_two_vendors_raises_before_scanning(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """SC-7's first measure: the basename assertion.

    Built on the exact tree RESEARCH measured -- two vendor directories with
    IDENTICAL schemas under one parent. Identical schemas are the dangerous
    case; differing ones already raise a SchemaError on their own.

    The error must name BOTH the basename it found and the vendor it was
    configured for, because "does not match" without the two values is a
    message that sends the reader back to the code.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    hive_raw_tree(parent, "alpaca", _rows(stock_pqt_row, "AAPL", 9.9))

    # The parent holds both vendors; a scan rooted here merges them silently
    # (proved by the fixture self-test above).
    dataset = StockDataset(_make_config(parent, vendor="tiingo"))

    with pytest.raises(ValueError) as excinfo:
        dataset._scan_raw()

    message = str(excinfo.value)
    assert "nasdaq_data" in message, "the basename actually found"
    assert "tiingo" in message, "the vendor configured"
    assert "TERMINATE" in message or "terminate" in message


def test_vendor_isolation_two_vendors_under_one_root_raise_on_the_column(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """SC-7's third measure: the written `vendor` column.

    The basename check cannot see a shard hand-copied into the wrong root --
    the path is still correct. The literal `vendor` column is what makes that
    condition DETECTABLE rather than merely unlikely, and the assertion must
    run BEFORE `dedup_raw_frame`, which would otherwise collapse the two
    vendors' overlapping `(timestamp, symbol)` rows to one arbitrarily and
    destroy the evidence.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    # A shard carrying another vendor's rows, copied into this vendor's tree.
    # The PATH is beyond reproach; only the column betrays it.
    import polars as pl

    foreign = pl.DataFrame(_rows(stock_pqt_row, "AAPL", 9.9)).with_columns(
        pl.lit("alpaca").alias("vendor")
    )
    for month, group in foreign.group_by(
        pl.col("timestamp").dt.strftime("%Y-%m")
    ):
        target = root / f"month={month[0]}"
        target.mkdir(parents=True, exist_ok=True)
        group.write_parquet(target / "part-foreign-00000.pqt")

    dataset = StockDataset(_make_config(root, vendor="tiingo"))
    with pytest.raises(ValueError) as excinfo:
        dataset._scan_raw().collect()

    message = str(excinfo.value)
    assert "alpaca" in message and "tiingo" in message
    assert "vendor" in message.lower()


def test_vendor_isolation_output_columns_are_exactly_the_pre_refactor_set(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """The frame handed downstream must be byte-compatible with its old self.

    `_raw_data_to_xr_window` sets a `[timestamp, symbol]` index on this frame
    and calls `.to_xarray()`. A leaked `vendor` or `month` column would become
    a spurious data variable in every Zarr store this project writes -- and
    would do it silently, because an extra variable breaks nothing until
    something downstream enumerates `data_vars`.
    """
    from tests.conftest import _STOCK_PQT_COLUMNS

    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    frame = StockDataset(_make_config(root)).\
        _scan_raw().collect()

    assert "vendor" not in frame.columns, "provenance column must be dropped"
    assert "month" not in frame.columns, "hive key must be dropped"
    assert set(frame.columns) == set(_STOCK_PQT_COLUMNS), (
        f"the column set handed to _raw_data_to_xr_window changed; extra="
        f"{sorted(set(frame.columns) - set(_STOCK_PQT_COLUMNS))} missing="
        f"{sorted(set(_STOCK_PQT_COLUMNS) - set(frame.columns))}"
    )


def test_a_shard_outside_raw_columns_makes_the_scan_raise_rather_than_merge(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Pitfall 6, and the direction its fix must point.

    A directory scan derives ONE schema from the first file and enforces it
    across all of them, so a single shard carrying an extra column makes the
    whole vendor root unreadable. The cure is pinning the projection at WRITE
    time (`frame.select(RAW_COLUMNS)`), never relaxing `extra_columns` at read
    time -- that would silence this error and reopen the silent cross-vendor
    merge in the same move, while looking like a bug fix.
    """
    import polars as pl

    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    rogue = pl.DataFrame([stock_pqt_row("2024-01-03", "AAPL", 5.0)]).with_columns(
        pl.lit("tiingo").alias("vendor"),
        pl.lit(1.0).alias("an_extra_column"),
    )
    rogue.write_parquet(root / "month=2024-01" / "part-rogue-00000.pqt")

    dataset = StockDataset(_make_config(root))
    with pytest.raises(Exception) as excinfo:
        dataset._scan_raw().collect()

    # polars raises a SchemaError (or a ComputeError wrapping one); either way
    # the scan REFUSES rather than quietly returning a ragged union.
    assert "schema" in str(excinfo.value).lower() or "column" in str(
        excinfo.value
    ).lower()

    # The write-side control that prevents this: both vendors pin the
    # projection, so a shard written through `_write_shard` cannot drift.
    from acquisition.tiingo import TiingoAcquisition

    assert TiingoAcquisition.RAW_COLUMNS[:3] == ("timestamp", "symbol", "vendor")


def test_an_absent_raw_root_raises_naming_vendor_frequency_and_path(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """Pitfall 7: distinguish "never fetched" from "window pruned to nothing".

    polars infers a scan's schema from the first file it finds, so an absent
    root raises `ComputeError: ... expanded paths were empty` -- a message
    naming none of the things a user needs in order to act. An empty WINDOW,
    by contrast, is not an error at all and must return an empty frame.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"

    missing = StockDataset(_make_config(parent / "alpaca", vendor="alpaca"))
    with pytest.raises(ValueError) as excinfo:
        missing._scan_raw()

    message = str(excinfo.value)
    assert "alpaca" in message, "names the vendor"
    assert "1d" in message, "names the frequency"
    assert "alpaca" in message and str(parent) in message, "names the path"

    # A root that EXISTS but whose window prunes to zero rows is a legitimate
    # question with a legitimate empty answer -- not an error.
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))
    pruned = StockDataset(
        _make_config(root, start_date="2023-01-01", end_date="2023-06-30")
    )
    frame = pruned._scan_raw().collect()
    assert frame.height == 0
    assert "vendor" not in frame.columns


def test_vendor_isolation_an_unset_vendor_raises_rather_than_scanning(
    tmp_path: Path, hive_raw_tree, stock_pqt_row
):
    """D-11: without a recorded vendor there is nothing to check the path
    against, so the scan refuses rather than guessing.

    Guessing would mean either trusting the basename (which is what is being
    verified) or scanning anyway (which is the silent merge). Refusing is the
    only option that does not quietly produce a wrong answer.
    """
    parent = tmp_path / "downloads" / "us_equity" / "1d" / "nasdaq_data"
    root = hive_raw_tree(parent, "tiingo", _rows(stock_pqt_row, "AAPL", 1.0))

    config = _make_config(root)
    config.vendor = None
    with pytest.raises(ValueError) as excinfo:
        StockDataset(config)._scan_raw()

    message = str(excinfo.value)
    assert "vendor" in message
    assert "D-11" in message
