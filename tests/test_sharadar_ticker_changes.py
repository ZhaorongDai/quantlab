"""Mapping Sharadar raw files' tickers to permatickers as of each file's pull (#234).

Sharadar renames a security's whole history when its ticker changes, so each
raw file names securities by the tickers of its own pull, while TICKERS holds
only the current ones. What is locked here:

- a file pulled before a ticker change maps through the ACTIONS
  ``tickerchangefrom`` chain, a reused ticker by the file's pull date;
- a file's own TICKERS snapshot wins over the later TICKERS;
- the store's ticker sidecar maps a ticker the vendor dropped without an
  ACTIONS row, and a rewritten sidecar keeps what the old one knew;
- an updated pull (SF1) replaces the row of the same security and key even
  when its ticker changed;
- a bulk TICKERS pull keeps a snapshot, and unused snapshots are pruned.

Raw files are written directly (``_write``) so each pull time is chosen by the
test; every value is SYNTHETIC.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from tests.sharadar_fixtures import (
    TICKERS_COLUMNS,
    FakeTransport,
    action_row,
    bulk_routes,
    csv_text,
    sep_row,
    sf1_row,
    tickers_row,
)


def _frame(code: str, rows: list[dict]) -> pl.DataFrame:
    """Rows of one raw table with its schema; ``YYYY-MM-DD`` text becomes a date."""
    from quantlab.dataset.sharadar.tables import TABLES

    schema = TABLES[code].schema
    data = {
        name: [
            date.fromisoformat(r[name]) if dtype == pl.Date and r.get(name) else r.get(name)
            for r in rows
        ]
        for name, dtype in schema.items()
    }
    return pl.DataFrame(data, schema=schema)


def _write(root: Path, code: str, rows: list[dict], pulled: datetime, *, window=None, updated=None):
    """Write a bulk file (or a window / updated pull) of ``code`` pulled at ``pulled``."""
    from quantlab.dataset.sharadar.tables import (
        bulk_file,
        updated_file,
        window_file,
        write_bulk_pull,
    )

    if window is not None:
        path = window_file(root, code, pulled, *window)
    elif updated is not None:
        path = updated_file(root, code, pulled, updated)
    else:
        path = bulk_file(root, code)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_bulk_pull(root, code, pulled)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(code, rows).write_parquet(path)
    return path


def _snapshot(root: Path, rows: list[dict], pulled: datetime) -> Path:
    from quantlab.dataset.sharadar.tables import tickers_snapshot_file

    path = tickers_snapshot_file(root, pulled)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame("tickers", rows).write_parquet(path)
    return path


def _change(day: str, new: str, old: str) -> dict:
    """A ``tickerchangefrom`` row: on ``day`` the security trading as ``old`` became ``new``."""
    row = action_row(day, "tickerchangefrom", new, None)
    row.update(contraticker=old, contraname=f"{old} CORP")  # SYNTHETIC
    return row


def _at(day: str) -> datetime:
    return datetime.fromisoformat(day).replace(hour=20, tzinfo=UTC)


BULK_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
WINDOW_DAYS = ["2024-01-10", "2024-01-11"]


def _prices(root, *, old="OLD", new="NEW"):
    """A bulk file pulled on 01-05 naming permaticker 707 ``old``; a window pulled 01-12 naming it ``new``."""
    _write(root, "sep", [sep_row(old, d, 10.0 + i) for i, d in enumerate(BULK_DAYS)], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "sep",
        [sep_row(new, d, 20.0 + i) for i, d in enumerate(WINDOW_DAYS)],  # SYNTHETIC
        _at("2024-01-12"),
        window=(date(2024, 1, 10), date(2024, 1, 12)),
    )


def _mapped(root, code="sep", sidecar=None):
    from quantlab.dataset.sharadar.permatickers import PermatickerResolver
    from quantlab.dataset.sharadar.tables import scan_raw_table

    resolver = PermatickerResolver(root, code, sidecar_path=sidecar)
    frame = scan_raw_table(root, code, annotate=resolver.annotate).collect().sort("date")
    return frame, resolver


def test_a_file_pulled_before_a_ticker_change_maps_through_the_actions_chain(tmp_path):
    root = tmp_path / "sharadar"
    _prices(root)
    _write(root, "tickers", [tickers_row("SEP", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, resolver = _mapped(root)
    assert frame["permaticker"].to_list() == [707] * 6
    assert resolver.unresolved == {}


def test_a_reused_ticker_maps_by_its_files_pull_date(tmp_path):
    # 707 traded as OLD until 01-09, then NEW; 808 listed as OLD on 01-10.
    root = tmp_path / "sharadar"
    _prices(root)
    _write(
        root,
        "sep",
        [sep_row("OLD", d, 5.0) for d in WINDOW_DAYS]  # SYNTHETIC
        + [sep_row("NEW", d, 20.0) for d in WINDOW_DAYS],  # SYNTHETIC
        _at("2024-01-12"),
        window=(date(2024, 1, 10), date(2024, 1, 12)),
    )
    _write(
        root,
        "tickers",
        [
            tickers_row("SEP", 707, "NEW", firstpricedate="2020-01-02"),  # SYNTHETIC
            tickers_row("SEP", 808, "OLD", firstpricedate="2024-01-10"),  # SYNTHETIC
        ],
        _at("2024-01-12"),
    )
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, _ = _mapped(root)
    bulk = frame.filter(pl.col("date") <= date(2024, 1, 5))
    assert set(bulk["permaticker"]) == {707}
    window = frame.filter(pl.col("date") >= date(2024, 1, 10))
    assert window.filter(pl.col("ticker") == "OLD")["permaticker"].unique().to_list() == [808]
    assert window.filter(pl.col("ticker") == "NEW")["permaticker"].unique().to_list() == [707]


def test_a_files_own_tickers_snapshot_wins_over_the_later_tickers(tmp_path):
    # No ACTIONS row: only the snapshot of the bulk's run knows OLD was 707's.
    root = tmp_path / "sharadar"
    _prices(root)
    _snapshot(root, [tickers_row("SEP", 707, "OLD")], _at("2024-01-05").replace(hour=19))  # SYNTHETIC
    current = [tickers_row("SEP", 707, "NEW"), tickers_row("SEP", 808, "OLD")]  # SYNTHETIC
    _snapshot(root, current, _at("2024-01-12").replace(hour=19))
    _write(root, "tickers", current, _at("2024-01-12").replace(hour=19))
    _write(root, "actions", [], _at("2024-01-12"))

    frame, _ = _mapped(root)
    assert frame["permaticker"].to_list() == [707] * 6


def test_a_snapshot_mapping_a_ticker_twice_is_refused(tmp_path):
    root = tmp_path / "sharadar"
    _prices(root)
    _snapshot(
        root,
        [tickers_row("SEP", 707, "OLD"), tickers_row("SEP", 909, "OLD")],  # SYNTHETIC
        _at("2024-01-05").replace(hour=19),
    )
    _write(root, "tickers", [tickers_row("SEP", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    with pytest.raises(ValueError, match="OLD"):
        _mapped(root)


def test_the_stores_ticker_sidecar_maps_a_ticker_tickers_and_actions_forgot(tmp_path):
    # The vendor renamed 707 to OLD1 (a reused ticker) with no ACTIONS row.
    root = tmp_path / "sharadar"
    _prices(root, new="OLD1")
    _write(root, "tickers", [tickers_row("SEP", 707, "OLD1")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    sidecar = tmp_path / "store.zarr.sharadar_tickers.json"
    sidecar.write_text(
        json.dumps({"table": "sep", "intervals": {"707": [{"start": None, "ticker": "OLD", "company": None}]}})
    )

    frame, _ = _mapped(root, sidecar=sidecar)
    assert frame["permaticker"].to_list() == [707] * 6
    frame, resolver = _mapped(root)
    assert frame["permaticker"].null_count() == 4
    assert resolver.unresolved[root / "sep" / "sep.parquet"] == {"OLD": "no permaticker"}


def test_a_rewritten_sidecar_keeps_what_the_old_one_knew(tmp_path):
    from quantlab.dataset.sharadar.tickers import ticker_sidecar_payload

    root = tmp_path / "sharadar"
    _write(root, "tickers", [tickers_row("SEP", 707, "OLD1")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [], _at("2024-01-12"))
    previous = {
        "table": "sep",
        "intervals": {
            "707": [{"start": None, "ticker": "OLD", "company": "OLD CORP"}],
            "909": [{"start": "2023-05-01", "ticker": "UNIT", "company": None}],
        },
    }
    payload = ticker_sidecar_payload(root, "sep", [707, 909], previous=previous)
    assert [s["ticker"] for s in payload["intervals"]["707"]] == ["OLD1"]
    assert payload["former"]["707"] == [{"start": None, "ticker": "OLD", "company": "OLD CORP"}]
    # TICKERS no longer lists 909: it keeps its earlier spans.
    assert payload["intervals"]["909"] == previous["intervals"]["909"]


def test_an_updated_pull_replaces_the_row_of_a_renamed_security(tmp_path):
    root = tmp_path / "sharadar"
    _write(root, "sf1", [sf1_row("OLD", "ARQ", "2024-01-03", "2023-12-31", revenue=1)], _at("2024-01-05"))  # SYNTHETIC
    _write(
        root,
        "sf1",
        [sf1_row("NEW", "ARQ", "2024-01-03", "2023-12-31", revenue=2, lastupdated="2024-01-10")],  # SYNTHETIC
        _at("2024-01-12"),
        updated=date(2024, 1, 10),
    )
    _write(root, "tickers", [tickers_row("SF1", 707, "NEW")], _at("2024-01-12"))  # SYNTHETIC
    _write(root, "actions", [_change("2024-01-09", "NEW", "OLD")], _at("2024-01-12"))  # SYNTHETIC

    frame, _ = _mapped(root, "sf1")
    assert frame.select("ticker", "permaticker", "revenue").rows() == [("NEW", 707, 2)]


def test_a_bulk_tickers_pull_keeps_a_snapshot_and_prunes_unused_ones(tmp_path, monkeypatch):
    from quantlab.acquisition.sharadar.client import SharadarClient
    from quantlab.dataset.sharadar.tables import pull_time, tickers_snapshots

    monkeypatch.setenv("SHARADAR_API_KEY", "synthetic-key")  # SYNTHETIC
    transport = FakeTransport(
        bulk_routes({"tickers": csv_text(TICKERS_COLUMNS, [tickers_row("SEP", 707, "OLD")])})  # SYNTHETIC
    )
    client = SharadarClient(transport=transport, sleep=lambda seconds: None)
    root = tmp_path / "sharadar"
    stale = _snapshot(root, [tickers_row("SEP", 707, "OLD")], datetime(2020, 1, 2, tzinfo=UTC))  # SYNTHETIC
    client.bulk_table("tickers", tmp_path)
    snapshots = tickers_snapshots(root)
    assert stale not in snapshots  # no raw file pairs with it
    assert len(snapshots) == 1
    assert pl.read_parquet(snapshots[0])["ticker"].to_list() == ["OLD"]
    assert abs((pull_time(root / "tickers" / "tickers.parquet") - pull_time(snapshots[0])).total_seconds()) < 1
