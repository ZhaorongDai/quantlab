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
