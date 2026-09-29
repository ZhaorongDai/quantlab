"""`quantlab.api.analyze_factors`: a caller's factor frame in, the library's factor report out.

Tested through the public function (ADR 0011): the analysis, its ``summary()`` and the errors a
caller sees. Numerical correctness is parity with the library path, ``Factor.analyze`` of the
same factor and a ``Return`` label on the equivalent Zarr-backed dataset. ``FactorAnalysis
.summary()`` is a library addition, so it is also checked on the library result directly.
"""

import warnings

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xarray as xr

from conftest import WHOLE_STORE, compute_all
from quantlab.analysis.factor_report import FactorAnalysis
from quantlab.base.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.polars import FactorPolars
from quantlab.label.predefined.fret import Return
from tests.backtest_fixtures import write_price_store

import quantlab.api as qa

ADJUSTED = ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume")
CANONICAL = ("open", "high", "low", "close", "volume")
SYMBOLS = [f"S{i:02d}" for i in range(12)]
SUMMARY_COLUMNS = [
    "factor", "fret", "ic", "rank_ic", "icir", "rank_icir", "long_short_return", "turnover"
]


class TwoMoves(FactorPolars):
    """The one-bar and three-bar ``adjClose`` change per symbol, two factor variables."""

    def _get_factor_lazyframe(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        close = pl.col("adjClose")
        return (
            lf.sort(["symbol", "timestamp"])
            .with_columns(
                (close / close.shift(1).over("symbol") - 1.0).alias("move_1"),
                (close / close.shift(3).over("symbol") - 1.0).alias("move_3"),
            )
            .select(["timestamp", "symbol", "move_1", "move_3"])
        )


def _setup(tmp_path):
    """The library factor and dataset, and the same factor values and bars as frames."""
    dataset = StockDataset(write_price_store(tmp_path, symbols=SYMBOLS, n_bars=80))
    factor = TwoMoves(PolarsFactorConfig(warmup_bars=0, dataset=dataset))
    store = xr.open_zarr(dataset.config.zarr_file_path).load()
    prices = (
        store[list(ADJUSTED)].rename(dict(zip(ADJUSTED, CANONICAL))).to_dataframe().reset_index()
    )
    factors = compute_all(factor).to_dataframe().reset_index()
    return factor, dataset, factors, prices


def _library_analysis(factor, dataset, span: int, quantiles: int) -> FactorAnalysis:
    label = Return(
        FactorConfig(
            warmup_bars=0,
            dataset=dataset,
            mode="batch",
            data_columns=("adjOpen",),
            kwargs={"n_forward_periods": span},
            njobs=4,
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*warm-up bar\(s\) are needed")
        return factor.analyze(*WHOLE_STORE, frets=[label], quantiles=quantiles)


def _assert_same_metrics(result: FactorAnalysis, expected: FactorAnalysis) -> None:
    assert list(result.pairs) == list(expected.pairs)
    for table in (
        "summary_table", "ic_table", "quantile_returns_table", "turnover_table",
        "monthly_ic_table",
    ):
        pd.testing.assert_frame_equal(getattr(result, table)(), getattr(expected, table)())
    pd.testing.assert_frame_equal(result.correlation.mean, expected.correlation.mean)


# --------------------------------------------------------------------------- parity


@pytest.mark.parametrize("span, quantiles", [(1, 5), (3, 3)])
def test_prices_match_factor_analyze_on_a_zarr_dataset(tmp_path, span, quantiles):
    factor, dataset, factors, prices = _setup(tmp_path)

    result = qa.analyze_factors(factors, prices=prices, span=span, quantiles=quantiles)

    expected = _library_analysis(factor, dataset, span, quantiles)
    assert isinstance(result, FactorAnalysis)
    _assert_same_metrics(result, expected)
    assert list(result.pairs) == [f"move_1__ret_{span}", f"move_3__ret_{span}"]
    assert {pair.quantiles for pair in result.pairs.values()} == {quantiles}
    assert list(result.pairs[f"move_1__ret_{span}"].quantile_returns.columns) == list(
        range(1, quantiles + 1)
    )


def test_own_returns_long_or_wide_match_the_prices_path(tmp_path):
    _, _, factors, prices = _setup(tmp_path)
    returns = qa.forward_returns(prices, span=2)
    wide = returns.pivot(index="timestamp", columns="symbol", values="ret_2")

    by_prices = qa.analyze_factors(factors, prices=prices, span=2)
    by_long = qa.analyze_factors(factors, returns, span=2)
    by_wide = qa.analyze_factors(factors, wide, span=2)

    _assert_same_metrics(by_long, by_prices)
    assert list(by_wide.pairs) == ["move_1__returns", "move_3__returns"]
    summary = by_wide.summary_table().assign(fret="ret_2")
    pd.testing.assert_frame_equal(summary, by_prices.summary_table())


def test_span_is_the_horizon_of_own_returns(tmp_path):
    _, _, factors, prices = _setup(tmp_path)
    returns = qa.forward_returns(prices, span=4)

    result = qa.analyze_factors(factors, returns, span=4)

    assert {pair.horizon for pair in result.pairs.values()} == {4}
    assert result.summary_table()["ic_nw_lags"].min() >= 3


def test_polars_inputs_match_pandas(tmp_path):
    _, _, factors, prices = _setup(tmp_path)

    result = qa.analyze_factors(pl.from_pandas(factors), prices=pl.from_pandas(prices))

    _assert_same_metrics(result, qa.analyze_factors(factors, prices=prices))


def test_columns_renames_every_input(tmp_path):
    _, _, factors, prices = _setup(tmp_path)
    renames = {"timestamp": "date", "symbol": "ticker", "open": "Open"}

    result = qa.analyze_factors(
        factors.rename(columns=renames),
        prices=prices.rename(columns=renames),
        columns={name: canonical for canonical, name in renames.items()},
    )

    _assert_same_metrics(result, qa.analyze_factors(factors, prices=prices))


def test_price_delay_are_those_of_forward_returns(tmp_path):
    _, _, factors, prices = _setup(tmp_path)
    returns = qa.forward_returns(prices, price="close", span=2, delay=0)

    result = qa.analyze_factors(factors, prices=prices, price="close", span=2, delay=0)

    _assert_same_metrics(result, qa.analyze_factors(factors, returns, span=2))


# --------------------------------------------------------------------------- summary


def _expected_summary(analysis: FactorAnalysis) -> pd.DataFrame:
    rows = []
    for pair in analysis.pairs.values():
        s = pair.summary
        rows.append({
            "factor": pair.factor_name,
            "fret": pair.fret_name,
            "ic": s["pearson_ic_mean"],
            "rank_ic": s["ic_mean"],
            "icir": s["pearson_ir"],
            "rank_icir": s["ir"],
            "long_short_return": s["mean_spread"],
            "turnover": (s["mean_turnover_top"] + s["mean_turnover_bottom"]) / 2,
        })
    return pd.DataFrame(rows)


def test_summary_on_the_library_result_is_pandas_matching_the_metrics(tmp_path):
    factor, dataset, _, _ = _setup(tmp_path)
    analysis = _library_analysis(factor, dataset, span=1, quantiles=5)

    summary = analysis.summary()

    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == SUMMARY_COLUMNS
    pd.testing.assert_frame_equal(summary, _expected_summary(analysis))
    pair = analysis.pairs["move_1__ret_1"]
    assert summary.loc[0, "rank_ic"] == pytest.approx(pair.ic.mean())
    assert summary.loc[0, "long_short_return"] == pytest.approx(pair.spread.mean())


@pytest.mark.parametrize("library", ["pandas", "polars"])
def test_summary_through_the_api_is_in_the_callers_library(tmp_path, library):
    _, _, factors, prices = _setup(tmp_path)
    if library == "polars":
        factors, prices = pl.from_pandas(factors), pl.from_pandas(prices)

    analysis = qa.analyze_factors(factors, prices=prices)
    summary = analysis.summary()

    frame_type = pd.DataFrame if library == "pandas" else pl.DataFrame
    assert isinstance(summary, frame_type)
    assert list(summary.columns) == SUMMARY_COLUMNS
    if library == "polars":
        summary = summary.to_pandas()
    pd.testing.assert_frame_equal(summary, _expected_summary(analysis))


def test_figures_and_save_stay_as_they_are(tmp_path):
    _, _, factors, prices = _setup(tmp_path)

    analysis = qa.analyze_factors(factors, prices=prices)
    out = analysis.save(tmp_path / "report")

    assert sorted(analysis.figures) == ["move_1__ret_1", "move_3__ret_1"]
    names = {p.name for p in out.iterdir()}
    assert {"summary.csv", "summary.json", "move_1__ret_1.png", "config.json"} <= names
    pd.testing.assert_frame_equal(
        pd.read_csv(out / "summary.csv")[["factor", "ic_mean"]],
        analysis.summary_table()[["factor", "ic_mean"]],
    )


# --------------------------------------------------------------------------- errors


def _small() -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamps = pd.bdate_range("2024-01-01", periods=12)
    symbols = ["A", "B", "C", "D", "E", "F"]
    rng = np.random.default_rng(0)
    base = pd.DataFrame({
        "timestamp": np.repeat(timestamps, len(symbols)),
        "symbol": symbols * len(timestamps),
    })
    factors = base.assign(signal=rng.normal(size=len(base)))
    prices = base.assign(open=100 + rng.normal(size=len(base)).cumsum())
    return factors, prices


def test_both_or_neither_of_returns_and_prices_raises():
    factors, prices = _small()
    returns = qa.forward_returns(prices)

    with pytest.raises(ValueError, match="exactly one of returns= and prices="):
        qa.analyze_factors(factors)
    with pytest.raises(ValueError, match="exactly one of returns= and prices="):
        qa.analyze_factors(factors, returns, prices=prices)


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"quantiles": 1}, ValueError, "quantiles must be at least 2"),
        ({"quantiles": 2.5}, TypeError, "quantiles must be an int"),
        ({"span": 0}, ValueError, "span must be at least 1"),
        ({"span": "1"}, TypeError, "span must be an int"),
    ],
)
def test_bad_arguments_raise_before_any_work(kwargs, error, match):
    factors, prices = _small()

    with pytest.raises(error, match=match):
        qa.analyze_factors(factors, qa.forward_returns(prices), **kwargs)


def test_price_or_delay_with_own_returns_raises():
    factors, prices = _small()
    returns = qa.forward_returns(prices)

    with pytest.raises(ValueError, match=r"price='close'.*only with prices="):
        qa.analyze_factors(factors, returns, price="close")
    with pytest.raises(ValueError, match=r"delay=0.*only with prices="):
        qa.analyze_factors(factors, returns, delay=0)


def test_a_missing_price_column_raises_naming_it():
    factors, prices = _small()

    with pytest.raises(ValueError, match="price column 'close' is not in the frame"):
        qa.analyze_factors(factors, prices=prices, price="close")


def test_factors_without_a_factor_column_raise():
    factors, prices = _small()

    with pytest.raises(ValueError, match="no factor column"):
        qa.analyze_factors(factors[["timestamp", "symbol"]], prices=prices)


def test_a_non_numeric_factor_column_raises_naming_it():
    factors, prices = _small()

    with pytest.raises(ValueError, match="'sector'.*not numeric"):
        qa.analyze_factors(factors.assign(sector="tech"), prices=prices)


def test_returns_with_two_value_columns_raise():
    factors, prices = _small()
    returns = qa.forward_returns(prices).assign(other=0.0)

    with pytest.raises(ValueError, match="one value column"):
        qa.analyze_factors(factors, returns)


def test_returns_sharing_no_bar_with_the_factors_raise():
    factors, prices = _small()
    returns = qa.forward_returns(prices)
    returns["timestamp"] = returns["timestamp"] + pd.Timedelta(days=365)

    with pytest.raises(ValueError, match="share no"):
        qa.analyze_factors(factors, returns)


def test_a_columns_entry_naming_nothing_raises():
    factors, prices = _small()

    with pytest.raises(ValueError, match="'Date'"):
        qa.analyze_factors(factors, prices=prices, columns={"Date": "timestamp"})


def test_duplicate_factor_rows_raise():
    factors, prices = _small()
    factors = pd.concat([factors, factors.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        qa.analyze_factors(factors, prices=prices)
