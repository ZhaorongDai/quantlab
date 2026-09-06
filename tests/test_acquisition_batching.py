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


# ---------------------------------------------------------------------------
# 03.2-03 Task 1 -- batch construction on the hoisted base class.
#
# `_batches` chunks a roster; `_refresh_batches` first GROUPS by recorded
# watermark, because one request carries exactly one `start`. Both are pure
# functions of the roster and the sidecars on disk: no vendor call, no
# credential, no network.
# ---------------------------------------------------------------------------

#: Far larger than any batch size under test, and deliberately not a round
#: multiple of every one of them. Borrowed from `tests/test_tiingo_quota.py`'s
#: `_MANY` idiom: "chunked correctly" and "silently collapsed to one batch"
#: must not be able to produce the same count.
_MANY = tuple(f"SYM{i:04d}" for i in range(250))


def _acquisition(acquisition_config, **config_kwargs):
    """A minimal CONCRETE `Acquisition` whose `_fetch_page` is never called.

    Deliberately not `TiingoAcquisition` or `AlpacaAcquisition`: the batching
    rules under test belong to the base class, and instantiating a vendor here
    would let a vendor override silently satisfy the assertion.
    """
    from base.acquisition import Acquisition

    class _Batching(Acquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            raise AssertionError(
                "batch construction must not issue a vendor request"
            )

    return _Batching(acquisition_config(**config_kwargs))


def test_batches_chunk_the_roster_by_batch_size(acquisition_config):
    """250 symbols is 3 batches at 100 and 250 batches at 1.

    Both directions matter. The 100 case proves chunking happens at all; the 1
    case proves the DEGENERATE vendor (Tiingo) still gets one symbol per
    request, which is what makes its quota-abort check run once per symbol
    rather than once per hundred.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    batches = list(acq._batches(_MANY))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    # Every symbol appears exactly once, in order -- chunking, not sampling.
    assert [symbol for batch in batches for symbol in batch] == list(_MANY)

    degenerate = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 1}
    )
    single = list(degenerate._batches(_MANY))
    assert len(single) == len(_MANY) == 250
    assert all(len(batch) == 1 for batch in single)


def test_refresh_batches_group_symbols_sharing_a_watermark(acquisition_config):
    """Two symbols at the same recorded `last_date` travel together; a third
    at a different one does not.

    One request carries exactly one `start`, so a mixed batch would have to
    either re-fetch history for some symbols or under-fetch for others.
    Grouping is REQUEST PACKING only -- D-06's window rule is untouched, which
    is why the derived starts are asserted here too.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT", "GOOG"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("MSFT", "2024-01-15", start_date="2020-01-01")
    acq._write_watermark("GOOG", "2024-01-20", start_date="2024-01-01")

    batches = list(acq._refresh_batches(["AAPL", "MSFT", "GOOG"]))

    assert sorted(sorted(batch) for batch in batches) == [
        ["AAPL", "MSFT"],
        ["GOOG"],
    ], batches

    # The grouping key is the WATERMARK, not the covered start -- AAPL and
    # MSFT share a `last_date` while recording different `start_date`s, and
    # they still batch together because only `last_date` decides the request.
    by_symbol = {tuple(sorted(batch)): batch for batch in batches}
    assert ("AAPL", "MSFT") in by_symbol


def test_refresh_batches_bucket_un_watermarked_symbols_under_the_config_start(
    acquisition_config,
):
    """A symbol with no sidecar has no watermark to group on, so it buckets
    under `config.start_date` -- which is exactly the start `_attempt_batch`
    would derive for it. Grouping it with a watermarked symbol would request
    the wrong window for one of the two.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    # MSFT has no sidecar at all.

    batches = list(acq._refresh_batches(["AAPL", "MSFT"]))

    assert sorted(sorted(batch) for batch in batches) == [["AAPL"], ["MSFT"]]


def test_refresh_batches_still_chunk_a_large_shared_watermark_group(
    acquisition_config,
):
    """The common case: a routine refresh where every symbol shares one
    watermark. Grouping must not defeat chunking and hand the vendor a
    250-symbol request.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    for symbol in _MANY:
        acq._write_watermark(symbol, "2024-01-15", start_date="2024-01-01")

    batches = list(acq._refresh_batches(list(_MANY)))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    assert sorted(symbol for batch in batches for symbol in batch) == sorted(
        _MANY
    )
