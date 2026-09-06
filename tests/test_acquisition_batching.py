"""Base-class batching, abort ordering and no-data marking (03.2 SC-1/SC-2/SC-4).

Phase 03.2 promotes the unit of acquisition work from a *symbol* to a
`(vendor, batch, page)` triple: `_fetch_page(symbols, start, end, page_token)`
becomes the single abstract fetch primitive and a single-symbol vendor is the
degenerate case (`DEFAULT_BATCH_SIZE = 1`), with no placeholder anywhere in the
hierarchy (SC-1, CONTEXT.md D-01). Two orderings in the shared base are
load-bearing and invisible at runtime if they regress:

- the global-abort check must be `_attempt`'s FIRST statement -- joblib cannot
  cancel already-queued work, so an input-generator check is an optimisation,
  not the guarantee (260906-26o D-05);
- the D-04 "queried, no data" marker must be evaluated only AFTER a batch's
  last page, never per page. Alpaca is symbol-major, so page 1 of a 100-symbol
  batch legitimately holds one symbol; per-page evaluation would stamp 99
  symbols "no data", advance their watermarks and skip them forever -- a silent
  99% loss that looks like a successful run (RESEARCH Pitfall 4).

Every test here is offline. Nothing sleeps for real, makes a network call,
requires a credential, or touches any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-test; 03.2-02 and
03.2-03 fill in the behavioural tests above. It is deliberately NOT an empty
placeholder: a pytest file with zero collected tests exits 5 ("no tests ran"),
which a later task's automated command reads as green.
"""

from pathlib import Path


def test_acquisition_config_fixture_places_watermarks_beside_the_vendor_raw_root(
    acquisition_config,
):
    """Fixture self-test for the two SC-7 path invariants, asserted in BOTH
    directions because only one of them is loud when it breaks.

    Direction 1 -- `raw_data_dir_path` terminates AT the vendor segment. This
    is the equality `StockDataset._scan_raw` asserts; a root pointing one level
    up silently unions two vendors (measured in RESEARCH Pattern 5).

    Direction 2 -- `watermark_path` is NOT a descendant of `raw_data_dir_path`.
    A refactor that tucks watermarks back under the raw root (the pre-03.2
    layout: `{subdir}/_watermarks`) breaks nothing visibly at write time, and
    then a polars directory scan of the raw root walks into the `.json`
    sidecars and fails far away from the cause.
    """
    for vendor in ("tiingo", "alpaca"):
        config = acquisition_config(vendor=vendor)
        raw_root = Path(config.raw_data_dir_path)
        watermark_root = Path(config.watermark_path)

        assert raw_root.name == vendor, (
            f"raw_data_dir_path must terminate at the vendor segment; got "
            f"{raw_root} whose basename is {raw_root.name!r}, not {vendor!r}"
        )

        assert not watermark_root.is_relative_to(raw_root), (
            f"watermark_path {watermark_root} must be a SIBLING of the raw "
            f"root {raw_root}, never a descendant -- a scan of the raw root "
            f"walks every file beneath it, including .json sidecars"
        )
        assert watermark_root.parent.name == "_watermarks"
        assert watermark_root.name == vendor


def test_acquisition_config_fixture_isolates_vendors_from_each_other(
    acquisition_config,
):
    """Two vendors built from the same fixture root share a parent but never a
    raw root -- the precondition every SC-7 assertion is written against."""
    tiingo = acquisition_config(vendor="tiingo")
    alpaca = acquisition_config(vendor="alpaca")

    tiingo_raw = Path(tiingo.raw_data_dir_path)
    alpaca_raw = Path(alpaca.raw_data_dir_path)

    assert tiingo_raw != alpaca_raw
    assert tiingo_raw.parent == alpaca_raw.parent
    assert not tiingo_raw.is_relative_to(alpaca_raw)
    assert not alpaca_raw.is_relative_to(tiingo_raw)
