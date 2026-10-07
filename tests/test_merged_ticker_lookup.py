"""A merged dataset names its symbols through its inputs' ticker lookups (#224).

``MergedDataset`` (and ``BadPrintMaskedDataset``, a merge with bad prints
masked) holds no store, so it has no sidecar of its own. Each symbol is named
by the first input whose lookup knows it, and falls back to its id when none
does, so a backtest over a merged CRSP index and ETF, or over a bad-print
masked view, labels its records with tickers as one over the plain store.
"""

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset.bad_prints import BadPrintMaskedDataset
from quantlab.dataset.base import SymbolName, TickerLookup
from quantlab.dataset.memory import FrameDataset
from quantlab.dataset.merged import MergedDataset

DAYS = pd.bdate_range("2024-01-01", periods=5)
DAY = date(2024, 1, 3)


class _Fixed(TickerLookup):
    def __init__(self, known):
        self.known = known

    def names(self, symbols, day):
        return [self.known.get(s, SymbolName(str(s))) for s in symbols]


class _Named(FrameDataset):
    """A frame dataset that names its symbols through a fixed lookup."""

    lookup = None

    def ticker_lookup(self):
        return self.lookup


def _dataset(symbols, known=None):
    frame = xr.Dataset(
        {"close": (("timestamp", "symbol"), np.ones((len(DAYS), len(symbols))))},
        coords={"timestamp": DAYS, "symbol": symbols},
    )
    dataset = _Named(frame)
    dataset.lookup = None if known is None else _Fixed(known)
    return dataset


def test_a_merge_of_inputs_without_lookups_names_none():
    merged = MergedDataset([_dataset([1]), _dataset([2])])
    assert merged.ticker_lookup() is None


def test_each_symbol_is_named_by_the_input_whose_lookup_knows_it():
    index = _dataset([1, 2], {1: SymbolName("AAA", "Aaa Inc")})
    etf = _dataset([3], {3: SymbolName("SPY", "SPDR S&P 500")})
    lookup = MergedDataset([index, etf]).ticker_lookup()
    assert lookup.names([3, 1, 2], DAY) == [
        SymbolName("SPY", "SPDR S&P 500"),
        SymbolName("AAA", "Aaa Inc"),
        SymbolName("2", None),
    ]
    assert lookup.label([1, 3, 9], DAY) == ["AAA", "SPY", "9"]


def test_an_input_without_a_lookup_is_skipped():
    prices = _dataset([1], {1: SymbolName("AAA")})
    lookup = MergedDataset([_dataset([1]), prices]).ticker_lookup()
    assert lookup.label([1], DAY) == ["AAA"]


def test_a_bad_print_masked_view_names_symbols_as_its_inputs_do():
    prices = _dataset([1, 2], {1: SymbolName("AAA", "Aaa Inc")})
    lookup = BadPrintMaskedDataset([prices], price_variable="close", volume_variable="close").ticker_lookup()
    assert lookup.names([1, 2], DAY) == [SymbolName("AAA", "Aaa Inc"), SymbolName("2", None)]
