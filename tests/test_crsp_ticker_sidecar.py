"""`{zarr}.crsp_tickers.json`: the write side, and `CrspTickerLookup` reading it.

The panel's `symbol` axis is the int64 PERMNO (D-01). A PERMNO is correct for a
machine and illegible for a human, and the fix D-03 chose is a SIDECAR rather
than a column: a 2-D `ticker(timestamp, symbol)` string variable is refused by
the backend's symbol-dim dtype guards, and a 1-D `ticker(symbol)` coord can only
hold ONE name per PERMNO -- which silently discards FB -> META, the exact case
this file pins twice.

So the sidecar is an INTERVAL TABLE, not a `{PERMNO: ticker}` map. That
distinction is the whole design: `_permno_breakdown` in `crsp.py` aggregates
with `pl.col("symbol").last()` and is deliberately NOT the shape copied here.

**Every quantlab import is INSIDE a test or helper body**, following
`tests/test_crsp_dataset.py`: these tests are written before the names they
assert on exist, and a module-scope import would turn the RED run into a
collection error -- zero tests discovered, which proves nothing (TDD gate
#3770).

**Provenance.** The FB -> META rows are VERBATIM `03.10-LIVE-CHECK.json` key
`C5_ticker_hist_crsp_a_stock.stksecurityinfohist`, reached through
`tests/crsp_fixtures.py:SECINFO_ROWS`; the same source
`tests/test_crsp_symbology.py` uses. Daily rows invented for a scenario carry a
`# SYNTHETIC` comment.
"""

from __future__ import annotations

from datetime import date

import pytest

from tests.test_crsp_dataset import (
    SYNTHETIC_PERMNO,
    _dataset_config,
    _pull,
    _synthetic_secinfo,
)

#: The three PERMNOs the converted panel below carries. 13407 is the rename
#: case (two intervals), 14593 is the multi-interval single-name case, and
#: 10107 is the synthetic one-interval case.
META_PERMNO = "13407"
AAPL_PERMNO = "14593"

PANEL_PERMNOS = (META_PERMNO, AAPL_PERMNO, SYNTHETIC_PERMNO)

#: A window straddling the FB -> META boundary (2022-06-09) so the panel's own
#: dates cross the rename. The sidecar's intervals come from the reference
#: tier and span the whole security history regardless.
WINDOW_START = "2022-06-01"
WINDOW_END = "2022-06-30"

#: Every trading day in that window, for every panel PERMNO. # SYNTHETIC --
#: the live check never sampled daily rows for these three securities in 2022.
_TRADING_DAYS = (
    "2022-06-01", "2022-06-02", "2022-06-03", "2022-06-06", "2022-06-07",
    "2022-06-08", "2022-06-09", "2022-06-10", "2022-06-13", "2022-06-14",
)


def _daily_rows():
    """Flat, priced, unsplit days -- everything `dsf_row` does not default.

    `dlyprc` is what makes the adjustment anchor usable (a PERMNO with no
    strictly-positive close inside the window is refused outright), and this
    file is about NAMES, so the prices are deliberately the least interesting
    ones available.
    """
    from tests.crsp_fixtures import dsf_row

    return [
        dsf_row(
            int(permno),
            day,
            dlyprc="100.000000",  # SYNTHETIC
            dlyret="0.001000",  # SYNTHETIC
            dlyretx="0.001000",  # SYNTHETIC
        )
        for permno in PANEL_PERMNOS
        for day in _TRADING_DAYS
    ]


@pytest.fixture
def converted(mock_crsp_session, tmp_path):
    """A converted CRSP store carrying exactly `PANEL_PERMNOS`.

    Returns the `CrspDatasetConfig`; the store, the filter report and the
    ticker sidecar are all on disk beside it.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _daily_rows(),
        list(PANEL_PERMNOS),
        start=WINDOW_START,
        end=WINDOW_END,
        extra_secinfo=_synthetic_secinfo(),
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start=WINDOW_START, end=WINDOW_END
    )
    CrspStockDataset(dataset_config).from_raw_data().save()
    return dataset_config


def _payload(dataset_config) -> dict:
    import json

    from quantlab.dataset.crsp import CrspStockDataset

    path = CrspStockDataset(dataset_config).ticker_sidecar_path()
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 1 -- the write side
# ---------------------------------------------------------------------------


def test_the_suffix_and_the_path_follow_the_filter_report_s_shape():
    """`.crsp_tickers.json`, a SIBLING of the store -- the same one-line
    `Path(str(zarr_file_path) + SUFFIX)` the filter report uses."""
    from quantlab.base.config import CrspDatasetConfig
    from quantlab.dataset.crsp import TICKER_SIDECAR_SUFFIX, CrspStockDataset

    assert TICKER_SIDECAR_SUFFIX == ".crsp_tickers.json"

    config = CrspDatasetConfig(
        zarr_file_path="/tmp/does-not-exist/crsp.zarr",
        raw_data_dir_path="/tmp/does-not-exist",
        catalog_path="/tmp/does-not-exist/catalog",
        reference_dir="/tmp/does-not-exist/reference",
        start_date="2022-06-01",
        end_date="2022-06-30",
    )
    path = CrspStockDataset(config).ticker_sidecar_path()
    assert str(path) == "/tmp/does-not-exist/crsp.zarr.crsp_tickers.json"


def test_a_conversion_writes_the_three_top_level_keys(converted):
    """`generated_from` / `vintage_product_end` / `intervals`.

    The vintage rides along for the same reason the adjustment anchor records
    it: the SAME PERMNO read against a newer CRSP vintage can carry a later
    interval, so "which names" is only answerable together with "as of which
    vintage".
    """
    payload = _payload(converted)

    assert sorted(payload) == [
        "generated_from",
        "intervals",
        "vintage_product_end",
    ]
    assert payload["generated_from"] == "stksecurityinfohist"
    assert payload["vintage_product_end"] == "2025-12-31"


def test_permno_13407_is_two_intervals_fb_then_meta(converted):
    """VERBATIM C5, and the reason this sidecar is not a `{PERMNO: ticker}` map.

    One PERMNO, two names, a hard boundary. A shape that kept only the LAST
    ticker -- `_permno_breakdown`'s `pl.col("symbol").last()` -- would answer
    "META" for 2012, which is the same defect D-03 rejected a 1-D
    `ticker(symbol)` coord for.
    """
    intervals = _payload(converted)["intervals"][META_PERMNO]

    assert len(intervals) == 2
    assert intervals[0] == {
        "ticker": "FB",
        "start": "2012-05-18",
        "end": "2022-06-08",
    }
    assert intervals[1] == {
        "ticker": "META",
        "start": "2022-06-09",
        "end": "2025-12-31",
    }


def test_intervals_are_keyed_by_permno_string_and_sorted_by_start(converted):
    """JSON object keys can only be strings, so `str(permno)` -- the same
    spelling `_permno_breakdown` uses. Within a PERMNO, `start` ascends."""
    intervals = _payload(converted)["intervals"]

    assert all(isinstance(key, str) for key in intervals)
    assert set(intervals) == {str(int(p)) for p in PANEL_PERMNOS}
    for spans in intervals.values():
        starts = [span["start"] for span in spans]
        assert starts == sorted(starts)
        assert all(span["start"] <= span["end"] for span in spans)


def test_only_the_panel_s_permnos_are_written(converted):
    """Not the whole reference tier.

    `SECINFO_ROWS` carries BRK (83443), GOOGL (90319), Lehman (80599) and
    WestRock (21186) as well, and the live table has 40,518 PERMNOs. A sidecar
    that named every security CRSP has ever issued would be megabytes of names
    for a panel holding three.
    """
    intervals = _payload(converted)["intervals"]

    assert len(intervals) == 3
    assert "83443" not in intervals
    assert "90319" not in intervals


def test_the_sidecar_is_written_indented_and_key_sorted(converted):
    """`indent=2, sort_keys=True`, passed EXPLICITLY.

    `quantlab/utils/atomic.py` forwards `**json_kwargs` verbatim so every
    caller keeps its own formatting; relying on a default here would make the
    format a property of the writer rather than of this sidecar.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    text = CrspStockDataset(converted).ticker_sidecar_path().read_text(
        encoding="utf-8"
    )

    assert text.startswith('{\n  "generated_from"')
    assert '\n  "intervals": {' in text
    assert '\n  "vintage_product_end"' in text


def test_an_existing_store_blocks_the_write(tmp_path):
    """WR-03: `_write_identity_reports` returns before writing anything when
    the store already exists.

    The cost is an append no longer refreshing the sidecar, and it is the
    lesser harm -- without the guard a REFUSED re-conversion would overwrite
    the surviving store's audit files with numbers for a panel that was never
    written. A rebuild deletes the store and its `.crsp_*.json` siblings first;
    `quantlab/dataset/crsp_rebuild.py:CrspStoreRebuilder` is what does that.
    """
    from quantlab.base.config import CrspDatasetConfig
    from quantlab.dataset.crsp import CrspStockDataset

    store = tmp_path / "crsp.zarr"
    store.mkdir()
    config = CrspDatasetConfig(
        zarr_file_path=str(store),
        raw_data_dir_path=str(tmp_path),
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(tmp_path / "reference"),
        start_date="2022-06-01",
        end_date="2022-06-30",
    )
    dataset = CrspStockDataset(config)
    dataset._ticker_intervals = {
        "generated_from": "stksecurityinfohist",
        "vintage_product_end": "2025-12-31",
        "intervals": {"13407": [{"ticker": "META", "start": "2022-06-09",
                                 "end": "2025-12-31"}]},
    }

    dataset._write_identity_reports()

    assert not dataset.ticker_sidecar_path().exists()
