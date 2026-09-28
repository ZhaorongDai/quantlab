"""A bar with no data yields NaN in every US-equity alpha, never a stand-in value.

``Alpha101Stock`` and ``Alpha158Stock`` z-score every output across symbols,
and ``CrossSectionalZScore`` skips NaN. A symbol with no bar that day (not yet
listed, delisted, or an all-NaN padding column) must therefore come out NaN,
or it enters the mean and the standard deviation of that bar and shifts every
real symbol's value. KunQuant's predefined alphas replace NaN on purpose in a
few operators (``SetInfOrNanToValue`` gives 0, ``Clip`` gives its bound, a
``Select`` between constants or an elementwise ``Max`` / ``Min`` gives a
finite value), which is what these tests lock out.

Every output of both classes is computed, since any one leaking operator is
enough to shift the cross-section. Each class is compiled once per store.

Where no bar is missing, both classes keep KunQuant's values, including the
0 its formulas give a value undefined on real data (a correlation over a
constant window).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from KunQuant.Op import Builder, Input, Output
from KunQuant.predefined import Alpha101, Alpha158
from KunQuant.Stage import Function

from quantlab.base.config import DatasetConfig, FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.dataset.stock import StockDataset
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.alpha158 import Alpha158Stock
from quantlab.my_ops.preprocess import CrossSectionalZScore

N_BARS = 120
#: Symbol 1 lists at this bar: every input is NaN before it.
LISTING_BAR = 40
REAL = list(range(1, 17))
#: All-NaN columns, as the ones macOS pads the symbol axis with.
EMPTY = list(range(101, 109))
COLUMNS = ["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
WINDOW = ("2024-01-01", "2024-12-31")


def _write_store(path: Path, symbols: list[int], listing_bar: int = LISTING_BAR) -> DatasetConfig:
    rng = np.random.default_rng(0)
    close = 50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (N_BARS, len(REAL))), axis=0))
    volume = rng.uniform(1e5, 1e6, (N_BARS, len(REAL)))
    spread = 1.0 + rng.uniform(0.0, 0.03, (N_BARS, len(REAL)))
    bars = {
        "adjOpen": close * rng.uniform(0.98, 1.02, (N_BARS, len(REAL))),
        "adjHigh": close * spread,
        "adjLow": close / spread,
        "adjClose": close,
        "adjVolume": volume,
    }
    for values in bars.values():
        values[:listing_bar, 0] = np.nan
    panel = xr.Dataset(
        {name: (["timestamp", "symbol"], values) for name, values in bars.items()},
        coords={"timestamp": pd.bdate_range("2024-01-01", periods=N_BARS), "symbol": REAL},
    )
    panel = panel.reindex(symbol=symbols)
    for raw, adjusted in zip(("open", "high", "low", "close", "volume"), COLUMNS):
        panel[raw] = panel[adjusted]
    panel.to_zarr(path, mode="w")
    return DatasetConfig(
        raw_data_dir_path=str(path.parent / "raw"),
        zarr_file_path=str(path),
        market="us_equity",
        frequency="1d",
    )


@pytest.fixture(scope="module", params=[Alpha101Stock, Alpha158Stock], ids=lambda cls: cls.__name__)
def panels(request, tmp_path_factory) -> tuple[type, xr.Dataset, xr.Dataset]:
    """The class and its outputs over the real symbols alone and with empty columns added."""
    cls = request.param
    root = tmp_path_factory.mktemp(cls.__name__)
    out = []
    for name, symbols in (("real", REAL), ("with_empty", REAL + EMPTY)):
        factor = cls(
            FactorConfig(
                warmup_bars=0,
                dataset=StockDataset(_write_store(root / f"{name}.zarr", symbols)),
                mode="batch",
                data_columns=COLUMNS,
                file_path=str(root / f"{name}_factors.zarr"),
                njobs=4,
            )
        )
        out.append(factor.compute(*WINDOW))
    return cls, out[0], out[1]


def _leaking(panel: xr.Dataset, symbols, bars=slice(None)) -> list[str]:
    """Names of the variables with any finite value on ``symbols`` over ``bars``."""
    part = panel.sel(symbol=symbols).isel(timestamp=bars)
    return [name for name in part.data_vars if np.isfinite(part[name].values).any()]


def test_empty_columns_come_out_nan(panels):
    _, _, with_empty = panels

    assert _leaking(with_empty, EMPTY) == []


def test_bars_before_listing_come_out_nan(panels):
    _, real, _ = panels

    assert _leaking(real, [1], slice(0, LISTING_BAR)) == []


def test_empty_columns_leave_the_real_symbols_unchanged(panels):
    _, real, with_empty = panels
    moved = [
        name
        for name in real.data_vars
        if not np.allclose(
            real[name].values,
            with_empty[name].sel(symbol=REAL).values,
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        )
    ]

    assert moved == []


def _stock_inputs():
    close, low, high = Input("adjClose"), Input("adjLow"), Input("adjHigh")
    return dict(low=low, high=high, close=close, open=Input("adjOpen"),
                volume=Input("adjVolume"), vwap=(high + low + close) / 3.0)


class _KunQuantAlpha101(FactorKunQuant):
    """KunQuant's own Alpha101 graphs, z-scored across symbols: the reference."""

    def _get_factor_names(self):
        return tuple(alpha.__name__ for alpha in Alpha101.all_alpha)

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            data = Alpha101.AllData(**_stock_inputs())
            for alpha in Alpha101.all_alpha:
                Output(CrossSectionalZScore(alpha(data)), alpha.__name__)
        return Function(builder.ops)


class _KunQuantAlpha158(FactorKunQuant):
    """KunQuant's own Alpha158 graphs with Alpha158Stock's feature set, z-scored."""

    def _ops(self):
        inputs = _stock_inputs()
        data = Alpha158.AllData(**inputs)
        data.vwap = inputs["vwap"]
        return data.build({
            "kbar": {},
            "price": {"windows": [0, 1, 2, 3, 4], "feature": [
                ("OPEN", data.open), ("HIGH", data.high), ("LOW", data.low),
                ("CLOSE", data.close), ("VWAP", data.vwap)]},
            "volume": {"windows": [0, 1, 2, 3, 4]},
            "rolling": {"windows": [5, 10, 20, 30, 60], "exclude": ["BETA", "RSQR", "RESI"]},
        })

    def _get_factor_names(self):
        return tuple(self._ops()[1])

    def _get_factor_func(self):
        builder = Builder()
        with builder:
            for op, name in zip(*self._ops()):
                Output(CrossSectionalZScore(op), name)
        return Function(builder.ops)


#: alpha015 ranks a 3-bar correlation of ranks, which is exactly tied for
#: many symbols; which of the tied values ranks first depends on the last bit
#: of float arithmetic, so any change to the compiled graph can reorder them
#: (KunQuant's own source notes the rank differs from pandas's for this
#: reason). Its missing-bar behaviour is still checked by the tests above.
TIE_ORDER_DEPENDENT = {"alpha015"}


@pytest.mark.parametrize(
    "cls, reference",
    [(Alpha101Stock, _KunQuantAlpha101), (Alpha158Stock, _KunQuantAlpha158)],
    ids=["Alpha101Stock", "Alpha158Stock"],
)
def test_kunquant_values_are_kept_where_no_bar_is_missing(tmp_path, cls, reference):
    dataset = StockDataset(_write_store(tmp_path / "full.zarr", REAL, listing_bar=0))

    def compute(factor_cls):
        return factor_cls(
            FactorConfig(
                warmup_bars=0, dataset=dataset, mode="batch", data_columns=COLUMNS,
                file_path=str(tmp_path / f"{factor_cls.__name__}.zarr"), njobs=4,
            )
        ).compute(*WINDOW)

    ours, expected = compute(cls), compute(reference)
    moved = [
        name
        for name in ours.data_vars
        if name not in TIE_ORDER_DEPENDENT
        and not np.allclose(ours[name].values, expected[name].values, rtol=1e-5, atol=1e-6, equal_nan=True)
    ]

    assert sorted(ours.data_vars) == sorted(expected.data_vars)
    assert moved == []
