"""Proof that `ingest_tiingo.py --universe`/`--as-of-date` wiring resolves a
symbol list purely through `UniverseCatalog.get_symbols_as_of()`, never
touching `TiingoAcquisition`/`StockDataset` inside `_build_configs()` itself.
"""

import argparse
from unittest.mock import patch

import ingest_tiingo


class _FakeCatalog:
    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        return ["AAPL", "MSFT"]


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        universe=None,
        as_of_date=None,
        symbols=None,
        start_date=None,
        end_date=None,
        refresh=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_configs_resolves_symbols_from_universe(monkeypatch):
    monkeypatch.setattr(
        ingest_tiingo.UniverseCatalog, "load", classmethod(lambda cls, config: _FakeCatalog())
    )

    args = _make_args(universe="sp500", as_of_date="2020-01-01")
    acq_config, ds_config = ingest_tiingo._build_configs(args)

    assert tuple(acq_config.symbols) == ("AAPL", "MSFT")
    assert list(ds_config.symbols) == ["AAPL", "MSFT"]


def test_build_configs_never_instantiates_tiingo_acquisition_or_stock_dataset(
    monkeypatch,
):
    monkeypatch.setattr(
        ingest_tiingo.UniverseCatalog, "load", classmethod(lambda cls, config: _FakeCatalog())
    )

    args = _make_args(universe="sp500", as_of_date="2020-01-01")
    with patch("ingest_tiingo.TiingoAcquisition") as mock_acquisition, patch(
        "ingest_tiingo.StockDataset"
    ) as mock_dataset:
        ingest_tiingo._build_configs(args)

        mock_acquisition.assert_not_called()
        mock_dataset.assert_not_called()


def test_build_configs_keeps_explicit_symbols_path_unchanged():
    args = _make_args(symbols="AAPL,MSFT")
    acq_config, ds_config = ingest_tiingo._build_configs(args)

    assert acq_config.symbols == ("AAPL", "MSFT")
    assert ds_config.symbols == ["AAPL", "MSFT"]
