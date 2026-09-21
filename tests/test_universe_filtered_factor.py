"""Tests for `quantlab/factor/universe_filter.py:UniverseFilteredFactor` (260915-p91).

The wrapper makes a point-in-time, rule-based stock universe a property of the
FACTOR layer, so neither the model layer nor the backtester is edited. What is
locked here:

- the tracer: a wrapped KunQuant factor and a wrapped `Return` label train an
  `XGBoostRegressor` and rebuild from the checkpoint's `config.json`;
- LS-2: the cross-sectional `Rank` sees ONLY in-universe symbols, matching a
  pandas mask-then-rank reference, while the time-series `ma_close` is
  untouched on in-universe cells;
- LS-3: a label is masked at its OWN timestamp, never at `t + horizon`.

Cost control: KunQuant compiles a module per `cal()`, so the wrapped and
unwrapped panels are computed once each in module-scoped fixtures (the
`tests/test_cross_sectional_zscore.py` precedent).

Everything is synthetic, CPU-only and offline.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import (
    CrossSectionBacktestConfig,
    DLConfig,
    FactorConfig,
    PolarsFactorConfig,
)
from quantlab.dataset.stock import StockDataset
from quantlab.dl_model.mlp import MLPRegressor
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.module import (
    load_backtester_from_config,
    load_model_from_config,
)

from tests.backtest_fixtures import (
    FirstFeatureHead,
    PastReturnFactor,
    make_stock_dataset,
    train_checkpoint,
)
from tests.universe_fixtures import (
    DROPOUT,
    ILLIQUID,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    NEVER_IN_UNIVERSE,
    PENNY,
    RANK_CLOSE_FACTOR_NAMES,
    REENTRY,
    REENTRY_OUT,
    SYMBOLS16,
    RankCloseFactor,
    day,
    hand_panel,
    make_rank_close_factor,
    make_return_label,
    make_wrapped_model,
    pandas_universe_mask,
    wrap,
    write_universe_store,
)

N_BARS = 80
DROP_BAR = 40
WINDOW = 5

TRAIN_START, TRAIN_END = day(0, N_BARS), day(59, N_BARS)
TEST_START, TEST_END = day(60, N_BARS), day(N_BARS - 1, N_BARS)


@pytest.fixture(autouse=True)
def _offline_wandb(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_SILENT", "true")


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    """One synthetic store for the whole module."""
    root = tmp_path_factory.mktemp("universe")
    config = write_universe_store(root, n_bars=N_BARS, drop_bar=DROP_BAR)
    return root, config


@pytest.fixture(scope="module")
def wrapped_features(store):
    """`UniverseFilteredFactor(RankCloseFactor).cal().get_features()`, computed once."""
    _, config = store
    factor = wrap(make_rank_close_factor(config), window=WINDOW)
    return factor.cal().get_features().load()


@pytest.fixture(scope="module")
def unwrapped_features(store):
    """The SAME factor without the wrapper, computed once."""
    _, config = store
    factor = make_rank_close_factor(config)
    return factor.cal().get_features().load()


def _normalized(config: dict) -> dict:
    """JSON round-trip, so tuples and lists compare equal on both sides."""
    return json.loads(json.dumps(config, default=str))


def _only_checkpoint(root: Path) -> Path:
    found = sorted(Path(root).rglob("*.joblib"))
    assert len(found) == 1, found
    return found[0]


# ---------------------------------------------------------------------------
# Tracer: wrapped factor + wrapped label -> XGBoostRegressor -> config rebuild
# ---------------------------------------------------------------------------


def test_tracer_xgb_trains_on_wrapped_factor_and_label_and_config_rebuilds(
    store, tmp_path
):
    """The whole slice, end to end, with ZERO edits to the model layer.

    A wrapped factor and a wrapped label drop into an `MLConfig`, `collect()`
    and `train()` run unchanged, and the checkpoint's `config.json` rebuilds
    into an equivalent wrapper. The symbol axis is the 2026-09-21 state: NO
    symbol ever leaves it, the threshold-failing symbols STAY as all-NaN
    columns.
    """
    _, config = store
    model = make_wrapped_model(
        tmp_path,
        config,
        model_cls=XGBoostRegressor,
        window=WINDOW,
        hyperparameters={"num_boost_round": 5},
        start_date=TRAIN_START,
        end_date=TEST_END,
        train_start=TRAIN_START,
        train_end=TRAIN_END,
        test_start=TEST_START,
        test_end=TEST_END,
    )

    model.collect()
    model.train()

    checkpoint = _only_checkpoint(model.config.model_save_dir)
    saved = json.loads((checkpoint.parent / "config.json").read_text())
    assert saved["trained_on"]["factor_names"] == list(RANK_CLOSE_FACTOR_NAMES)

    panel = model.data_backend.get_xarray_dataset(["timestamp", "symbol"])
    symbols = [str(symbol) for symbol in panel["symbol"].values]

    # Nothing leaves the axis; threshold failures are KEPT as all-NaN columns.
    assert set(symbols) == set(SYMBOLS16)
    assert PENNY in symbols and ILLIQUID in symbols
    for symbol in (PENNY, ILLIQUID):
        for variable in ("rank_close", "ma_close", "ret_1"):
            values = panel[variable].sel(symbol=symbol).values
            assert np.isnan(values).all(), f"{variable}/{symbol} should be all NaN"
    # Guard against a vacuous pass: a plain common must carry real values.
    assert np.isfinite(panel["rank_close"].sel(symbol="AAPL").values).any()

    rebuilt = load_model_from_config(model.get_config())

    for rebuilt_item, original in (
        (rebuilt.config.factors[0], model.config.factors[0]),
        (rebuilt.config.labels[0], model.config.labels[0]),
    ):
        assert type(rebuilt_item) is UniverseFilteredFactor
        assert type(rebuilt_item.factor) is type(original.factor)
        assert rebuilt_item.min_price == original.min_price
        assert rebuilt_item.min_dollar_volume == original.min_dollar_volume
        assert rebuilt_item.window == original.window

    original_config = _normalized(model.get_config())
    rebuilt_config = _normalized(rebuilt.get_config())
    # `resolved_hyperparameters` is a RECORD of what xgboost trained with, not
    # a config field; the loader drops it and an untrained rebuild writes none.
    original_config.pop("resolved_hyperparameters", None)
    rebuilt_config.pop("resolved_hyperparameters", None)
    assert rebuilt_config == original_config


# ---------------------------------------------------------------------------
# LS-2: the cross-sectional op sees only in-universe symbols
# ---------------------------------------------------------------------------


def test_cross_sectional_rank_sees_only_in_universe_symbols(
    store, wrapped_features, unwrapped_features
):
    """`rank_close` matches a pandas mask-then-rank reference within 1e-5 with
    an identical NaN pattern, and the time-series `ma_close` is unchanged by
    the rewrite on in-universe cells.

    The reference recomputes the mask from the RAW columns in pandas, so it
    shares no code with the implementation under test.
    """
    _, config = store
    mask = pandas_universe_mask(config, window=WINDOW)

    panel = xr.open_dataset(Path(config.zarr_file_path)).load()
    panel.close()
    adjusted = panel["adjClose"].to_pandas()
    reference = adjusted.where(mask).rank(axis=1, pct=True)

    symbols = [str(symbol) for symbol in wrapped_features["symbol"].values]
    timestamps = wrapped_features["timestamp"].values
    expected = reference.loc[timestamps, symbols].to_numpy()
    got = (
        wrapped_features["rank_close"]
        .transpose("timestamp", "symbol")
        .to_numpy()
    )

    np.testing.assert_array_equal(np.isnan(got), np.isnan(expected))
    finite = ~np.isnan(expected)
    assert finite.any(), "the reference is entirely NaN -- vacuous comparison"
    np.testing.assert_allclose(got[finite], expected[finite], atol=1e-5, rtol=0)

    # The time-series output is untouched by the rewrite wherever it survives
    # the output mask.
    wrapped_ma = wrapped_features["ma_close"].sel(symbol=symbols)
    unwrapped_ma = unwrapped_features["ma_close"].sel(
        symbol=symbols, timestamp=timestamps
    )
    in_universe = ~np.isnan(wrapped_ma.to_numpy())
    assert in_universe.any()
    np.testing.assert_allclose(
        wrapped_ma.to_numpy()[in_universe],
        unwrapped_ma.to_numpy()[in_universe],
        atol=1e-6,
        rtol=0,
    )


# ---------------------------------------------------------------------------
# LS-3: a label is masked at its own timestamp only
# ---------------------------------------------------------------------------


def test_label_is_masked_at_its_own_timestamp_only(store):
    """The drop-out symbol keeps a FINITE label at the last bar it is in the
    universe, even though the return it measures is realised after it leaves.

    A label at `t` must depend on the mask at `t` alone. Masking at `t + h`
    instead would blank this cell -- which is the look-ahead-shaped mistake
    that would quietly delete every profitable exit from the training set.
    """
    _, config = store
    label = wrap(make_return_label(config, n_forward_periods=1), window=WINDOW)

    labels = label.cal().get_labels().load()
    series = labels["ret_1"].sel(symbol=DROPOUT)
    timestamps = [str(value)[:10] for value in series["timestamp"].values]

    last_in_universe = day(DROP_BAR - 1, N_BARS)
    assert np.isfinite(
        float(series.sel(timestamp=last_in_universe).values)
    ), "the last in-universe bar must keep its label"

    for index, stamp in enumerate(timestamps):
        if stamp >= day(DROP_BAR, N_BARS):
            assert np.isnan(float(series.isel(timestamp=index).values)), (
                f"{DROPOUT} left the universe at {day(DROP_BAR, N_BARS)} but "
                f"still carries a label at {stamp}"
            )


# ---------------------------------------------------------------------------
# LS-1: the mask rule matrix, straight against hand-built panels
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mask_factor(store):
    """A wrapper used ONLY for `compute_universe_mask`, which is a pure
    function of the panel it is handed -- no KunQuant, no store read."""
    _, config = store
    return wrap(make_rank_close_factor(config), window=WINDOW)


def test_mask_price_threshold_is_inclusive(mask_factor):
    """RAW close exactly at `min_price` is IN; a hair below is OUT."""
    close = np.array([[MIN_PRICE, MIN_PRICE - 0.001]] * 6)
    volume = np.full((6, 2), 1_000_000.0)

    mask = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=["AAAA", "BBBB"])
    )

    assert float(mask[5, 0]) == 1.0
    assert np.isnan(float(mask[5, 1]))


def test_mask_dollar_volume_threshold_is_inclusive(mask_factor):
    """A trailing mean exactly at `min_dollar_volume` is IN; below is OUT."""
    close = np.full((6, 2), 10.0)
    # 10.0 * 100_000 == 1_000_000 exactly; the second symbol is one share shy.
    volume = np.array([[100_000.0, 99_999.0]] * 6)

    mask = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=["AAAA", "BBBB"])
    )

    assert float(mask[5, 0]) == 1.0
    assert np.isnan(float(mask[5, 1]))


def test_mask_incomplete_window_is_out(mask_factor):
    """The first `window - 1` bars have no full dollar-volume window."""
    close = np.full((8, 1), 50.0)
    volume = np.full((8, 1), 1_000_000.0)

    mask = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=["AAAA"])
    )

    assert np.isnan(mask.values[: WINDOW - 1, 0]).all()
    assert (mask.values[WINDOW - 1 :, 0] == 1.0).all()


def test_mask_nan_volume_poisons_its_whole_window(mask_factor):
    """One NaN volume at bar j puts the symbol out for bars j .. j+window-1.

    `min_periods=window` counts VALID observations, so a window containing the
    hole is short and yields NaN. "We do not know how liquid it was" must read
    as out, never as in.
    """
    close = np.full((14, 1), 50.0)
    volume = np.full((14, 1), 1_000_000.0)
    hole = 6
    volume[hole, 0] = np.nan

    mask = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=["AAAA"])
    )

    assert float(mask[hole - 1, 0]) == 1.0
    assert np.isnan(mask.values[hole : hole + WINDOW, 0]).all()
    assert float(mask[hole + WINDOW, 0]) == 1.0


def test_mask_ignores_adjusted_columns(mask_factor):
    """Adjusted values that would flip every decision change nothing.

    The mask reads RAW close/volume alone (LS-1). Adjusted history is
    depressed by splits and dividends, so judging "was this a penny stock
    then?" on adjusted prices is simply the wrong question.
    """
    close = np.full((8, 2), 50.0)
    volume = np.full((8, 2), 1_000_000.0)

    baseline = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=["AAAA", "BBBB"])
    )
    flipped = mask_factor.compute_universe_mask(
        hand_panel(
            close,
            volume,
            symbols=["AAAA", "BBBB"],
            adj_close=np.full((8, 2), 0.001),
        )
    )

    xr.testing.assert_identical(baseline, flipped)


@pytest.mark.parametrize("cut", [6, 10, 14])
def test_mask_is_point_in_time(mask_factor, cut):
    """Changing bars AFTER `cut` never changes the mask at or before `cut`.

    This is the property that makes the filter honest: a universe that peeked
    at the future would select exactly the names that went on to do well.
    """
    rng = np.random.default_rng(3)
    close = rng.uniform(4.0, 60.0, size=(20, 3))
    volume = rng.uniform(5_000.0, 2_000_000.0, size=(20, 3))
    symbols = ["AAAA", "BBBB", "CCCC"]

    baseline = mask_factor.compute_universe_mask(
        hand_panel(close, volume, symbols=symbols)
    )

    for scale in (1e-3, 1e3):
        future = close.copy()
        future_volume = volume.copy()
        future[cut + 1 :] *= scale
        future_volume[cut + 1 :] *= scale
        perturbed = mask_factor.compute_universe_mask(
            hand_panel(future, future_volume, symbols=symbols)
        )
        np.testing.assert_array_equal(
            baseline.values[: cut + 1], perturbed.values[: cut + 1]
        )


# ---------------------------------------------------------------------------
# LS-2: invariance to data that is never in the universe
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def perturbed(tmp_path_factory):
    """The same store with the never-in-universe symbols' adjusted close x1000."""
    root = tmp_path_factory.mktemp("universe_perturbed")
    config = write_universe_store(
        root,
        n_bars=N_BARS,
        drop_bar=DROP_BAR,
        perturb_never_in_universe=1000.0,
    )
    wrapped = wrap(make_rank_close_factor(config), window=WINDOW)
    unwrapped = make_rank_close_factor(config)
    return (
        wrapped.cal().get_features().load(),
        unwrapped.cal().get_features().load(),
    )


def test_perturbing_never_in_universe_symbols_leaves_wrapped_outputs_unchanged(
    wrapped_features, unwrapped_features, perturbed
):
    """Multiplying junk symbols' prices by 1000 moves NO in-universe output.

    The second half is what gives this teeth: the UNWRAPPED factor's ranks DO
    move, so the invariance above is the rewrite working rather than the
    perturbation failing to reach the graph.
    """
    perturbed_wrapped, perturbed_unwrapped = perturbed

    for name in RANK_CLOSE_FACTOR_NAMES:
        base = wrapped_features[name].transpose("timestamp", "symbol").values
        after = (
            perturbed_wrapped[name].transpose("timestamp", "symbol").values
        )
        np.testing.assert_array_equal(np.isnan(base), np.isnan(after))
        finite = ~np.isnan(base)
        assert finite.any()
        np.testing.assert_allclose(
            base[finite], after[finite], atol=1e-5, rtol=0
        )

    # Teeth: without the rewrite, the junk data reaches the cross-section.
    in_universe = ~np.isnan(
        wrapped_features["rank_close"].transpose("timestamp", "symbol").values
    )
    symbols = [str(s) for s in wrapped_features["symbol"].values]
    plain_base = (
        unwrapped_features["rank_close"]
        .sel(symbol=symbols)
        .transpose("timestamp", "symbol")
        .values
    )
    plain_after = (
        perturbed_unwrapped["rank_close"]
        .sel(symbol=symbols)
        .transpose("timestamp", "symbol")
        .values
    )
    changed = ~np.isclose(
        plain_base, plain_after, atol=1e-5, rtol=0, equal_nan=True
    )
    assert (changed & in_universe).any(), (
        "the unwrapped factor's in-universe ranks did NOT move, so the "
        "perturbation never reached the cross-section and the invariance "
        "assertion above proves nothing"
    )


def test_time_series_over_cross_sectional_is_nan_after_reentry(
    store, wrapped_features
):
    """LS-5, the accepted cost: `ma_rank` (a 3-bar mean OVER a rank) is NaN
    for 2 bars after a symbol re-enters the universe.

    The rank was NaN while the symbol was out, so the trailing window is still
    short. This is consistent with live trading -- you genuinely do not have
    those observations -- and it is documented rather than worked around.
    """
    _, config = store
    reference = pandas_universe_mask(config, window=WINDOW)
    back = REENTRY_OUT[1]

    # Teeth: the symbol really is back IN the universe on these bars, so the
    # NaNs below are the TS-over-CS cost and not simple exclusion.
    for offset in (0, 1, 2):
        assert bool(reference[REENTRY].iloc[back + offset]), (
            f"{REENTRY} should be in the universe at bar {back + offset}"
        )

    series = wrapped_features["ma_rank"].sel(symbol=REENTRY).values
    assert np.isnan(series[back]), "first bar after re-entry must be NaN"
    assert np.isnan(series[back + 1]), "second bar after re-entry must be NaN"
    assert np.isfinite(series[back + 2]), (
        "the third bar completes the 3-bar window and must be finite"
    )

    # The purely time-series sibling has no such cost: it never consumed a
    # masked cross-section.
    assert np.isfinite(
        wrapped_features["ma_close"].sel(symbol=REENTRY).values[back]
    )


# ---------------------------------------------------------------------------
# Stream parity and the read path
# ---------------------------------------------------------------------------


def test_stream_outputs_match_batch_and_the_mask_row_matches(
    store, wrapped_features
):
    """A stream-mode wrapper fed bar by bar reproduces the batch outputs.

    Both layouts must see the SAME mask, so the stream mask row is also
    checked against the independent pandas reference. A divergence here would
    mean live trading and research silently compute different factors.
    """
    _, config = store
    reference_mask = pandas_universe_mask(config, window=WINDOW)

    stream_config = dataclasses.replace(config, symbols=tuple(SYMBOLS16))
    factor = wrap(
        make_rank_close_factor(
            config,
            mode="stream",
            dataset=StockDataset(stream_config),
        ),
        window=WINDOW,
    )

    columns = ("adjClose", "close", "volume")
    input_dict, symbols, timestamps = factor.config.dataset.to_kunquant(columns)
    symbol_list = [str(symbol) for symbol in symbols]

    survivors = [str(s) for s in wrapped_features["symbol"].values]
    got = {
        name: np.full((len(timestamps), len(survivors)), np.nan)
        for name in RANK_CLOSE_FACTOR_NAMES
    }

    for step in range(len(timestamps)):
        bar = {
            column: np.ascontiguousarray(
                input_dict[column][step].astype(np.float32)
            )
            for column in columns
        }
        factor.cal_stream(bar, int(step), symbol_list)

        # The stream mask row must equal the batch reference at this bar.
        row = factor._universe_mask.isel(timestamp=0)
        expected_row = reference_mask.iloc[step]
        for symbol in symbol_list:
            in_universe = not np.isnan(
                float(row.sel(symbol=symbol).values)
            )
            assert in_universe == bool(expected_row[symbol]), (
                f"stream mask disagrees with the reference at bar {step} "
                f"for {symbol}"
            )

        features = factor.get_features()
        for name in RANK_CLOSE_FACTOR_NAMES:
            got[name][step] = (
                features[name].isel(timestamp=0).sel(symbol=survivors).values
            )

    for name in RANK_CLOSE_FACTOR_NAMES:
        batch = (
            wrapped_features[name]
            .sel(symbol=survivors)
            .transpose("timestamp", "symbol")
            .values
        )
        finite = ~np.isnan(batch)
        assert finite.any()
        np.testing.assert_allclose(
            got[name][finite], batch[finite], atol=1e-5, rtol=0
        )


def test_read_path_reproduces_the_cal_path(store, tmp_path):
    """`cal().save()` then a FRESH wrapper's `read()` gives the same panel.

    The store holds UNMASKED values -- masking is applied by `_get_features`
    on the way out -- so this also pins that a `read`-strategy store must be
    written THROUGH the wrapper.
    """
    _, config = store
    path = str(tmp_path / "factors" / "rank_close.zarr")

    factor = wrap(
        make_rank_close_factor(config, file_path=path), window=WINDOW
    )
    computed = factor.cal().get_features().load()
    factor.save(mode="w")

    fresh = wrap(make_rank_close_factor(config, file_path=path), window=WINDOW)
    read_back = fresh.read().get_features().load()

    xr.testing.assert_allclose(read_back, computed)
    symbols = [str(s) for s in read_back["symbol"].values]
    assert PENNY in symbols
    assert np.isnan(read_back["rank_close"].sel(symbol=PENNY).values).all()


# ---------------------------------------------------------------------------
# LS-3 symbol axis: nothing ever leaves it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["cal", "read"])
def test_the_symbol_axis_is_the_same_in_two_disjoint_date_windows(
    store, tmp_path, strategy
):
    """The symbol axis must NOT depend on the date window.

    Window B lies entirely after the drop-out leaves the universe, so under the
    superseded "drop a whole-window-NaN column" rule its axis would be narrower
    than window A's -- and `DLModel._align_prediction_symbols` raises when a
    trained symbol is missing from a later window. Since 2026-09-21 NO rule
    removes a symbol at all (condition (a) was the last one that could), so the
    two axes are identical and carry the whole fixture roster.
    """
    _, config = store
    window_a = (day(0, N_BARS), day(35, N_BARS))
    window_b = (day(50, N_BARS), day(N_BARS - 1, N_BARS))

    axes = []
    for start, end in (window_a, window_b):
        path = str(tmp_path / f"factors_{strategy}_{start}" / "f.zarr")
        factor = wrap(
            make_rank_close_factor(
                config, start_date=start, end_date=end, file_path=path
            ),
            window=WINDOW,
        )
        factor.cal()
        if strategy == "read":
            factor.save(mode="w")
            factor = wrap(
                make_rank_close_factor(
                    config, start_date=start, end_date=end, file_path=path
                ),
                window=WINDOW,
            )
            factor.read()

        features = factor.get_features().load()
        symbols = [str(s) for s in features["symbol"].values]
        axes.append(symbols)

        assert set(symbols) == set(SYMBOLS16), "no symbol may leave the axis"
        assert np.isnan(features["rank_close"].sel(symbol=PENNY).values).all()

    assert axes[0] == axes[1], (
        "the symbol axis changed between two date windows; a DL head trained "
        "on the first would refuse to predict on the second"
    )

    # Window B is entirely after the drop: the drop-out is kept as a NaN column.
    assert DROPOUT in axes[1]


# ---------------------------------------------------------------------------
# Refusals and lookback widening
# ---------------------------------------------------------------------------


def test_a_polars_factor_is_refused(store):
    _, config = store
    polars_factor = PastReturnFactor(
        PolarsFactorConfig(
            window=5, dataset=make_stock_dataset(config), kwargs={"n": 1}
        )
    )

    with pytest.raises(TypeError, match="cross-sectional"):
        UniverseFilteredFactor(polars_factor)


def test_double_wrapping_is_refused(store):
    _, config = store
    wrapped = wrap(make_rank_close_factor(config), window=WINDOW)

    with pytest.raises(TypeError, match="UniverseFilteredFactor"):
        UniverseFilteredFactor(wrapped)


def test_a_zero_window_is_refused(store):
    _, config = store

    with pytest.raises(ValueError, match="window"):
        wrap(make_rank_close_factor(config), window=0)


def test_from_config_refuses_a_missing_or_unknown_parameter(store):
    _, config = store
    saved = wrap(make_rank_close_factor(config), window=WINDOW).get_config()

    incomplete = {k: v for k, v in saved.items() if k != "min_price"}
    with pytest.raises(ValueError, match="min_price"):
        UniverseFilteredFactor.from_config(incomplete)

    extra = dict(saved, min_market_cap=1e9)
    with pytest.raises(ValueError, match="min_market_cap"):
        UniverseFilteredFactor.from_config(extra)


def test_get_features_before_cal_raises(store):
    _, config = store
    factor = wrap(make_rank_close_factor(config), window=WINDOW)

    with pytest.raises(RuntimeError, match="cal"):
        factor.get_features()


def test_lookback_is_widened_but_never_narrowed(store):
    """The wrapper pulls the dataset start EARLIER so the dollar-volume window
    is full at the first bar of the window -- the backtester's warm-up counts
    only `config.window` bars and knows nothing about this one."""
    _, config = store
    start = "2024-03-01"

    factor = wrap(
        make_rank_close_factor(config, start_date=start, end_date="2024-06-01"),
        window=WINDOW,
    )
    required = (
        pd.Timestamp(start)
        - pd.DateOffset(
            days=UniverseFilteredFactor.LOOKBACK_DAYS_PER_BAR * WINDOW
            + UniverseFilteredFactor.LOOKBACK_PAD_DAYS
        )
    ).strftime("%Y-%m-%d")
    assert factor.config.dataset.config.start_date <= required

    # A large INNER window already reaches further back; the wrapper must not
    # narrow it.
    big = wrap(
        make_rank_close_factor(
            config, window=200, start_date=start, end_date="2024-06-01"
        ),
        window=WINDOW,
    )
    inner_required = (
        pd.Timestamp(start) - pd.DateOffset(days=200)
    ).strftime("%Y-%m-%d")
    assert big.config.dataset.config.start_date == inner_required


# ---------------------------------------------------------------------------
# LS-4: the backtester runs UNCHANGED with wrapped factors
# ---------------------------------------------------------------------------

#: The model trains entirely before the drop bar; the backtest window opens
#: after it, so a rebalance falls on either side of the drop.
MODEL_DATES = dict(
    start_date=day(0, N_BARS),
    end_date=day(35, N_BARS),
    train_start=day(0, N_BARS),
    train_end=day(29, N_BARS),
    test_start=day(30, N_BARS),
    test_end=day(35, N_BARS),
)
BT_START_BAR, BT_END_BAR = 38, 60
REBALANCE_PERIODS = 5
TOP_N = 2

#: Window rows that rebalance: every 5 bars from the window start. Bar 38 is
#: row 0 and bar 43 is row 5, so the drop at bar 40 falls strictly between two
#: rebalances -- which is what makes the "sold late" assertion meaningful.
R1_ROW, R2_ROW = 0, 5


def _backtester(root: Path, config, checkpoint):
    return USEquityCrossectionSelectStockVectorBt(
        CrossSectionBacktestConfig(
            price_dataset=make_stock_dataset(config),
            model=make_wrapped_model(
                root / "backtest",
                config,
                model_cls=FirstFeatureHead,
                window=WINDOW,
                **MODEL_DATES,
            ),
            model_mode="load",
            checkpoint=str(checkpoint),
            start_date=day(BT_START_BAR, N_BARS),
            end_date=day(BT_END_BAR, N_BARS),
            output_dir=str(root / "runs"),
            rebalance_periods=REBALANCE_PERIODS,
            direction="long_only",
            top_n=TOP_N,
            fees=0.0,
            slippage=0.0,
        )
    )


@pytest.fixture(scope="module")
def backtest_run(tmp_path_factory, store):
    """One backtest over the wrapped factors, reused by three tests."""
    _, config = store
    root = tmp_path_factory.mktemp("universe_backtest")
    checkpoint = train_checkpoint(
        make_wrapped_model(
            root / "train",
            config,
            model_cls=FirstFeatureHead,
            window=WINDOW,
            **MODEL_DATES,
        )
    )
    backtester = _backtester(root, config, checkpoint)
    return root, config, checkpoint, backtester.run()


def test_backtester_runs_with_wrapped_factors_and_never_selects_junk(
    backtest_run,
):
    """The whole backtester runs unchanged, and junk never reaches the book.

    The penny name carries the HIGHEST adjusted close in the panel, so an
    unfiltered run buys it first -- that is the +886,077,331% failure this
    task exists to remove. Here its features are all NaN, so `predict_panel`
    scores it NaN and the selector cannot pick it.
    """
    _, _, _, result = backtest_run

    assert sorted(p.name for p in result.run_dir.iterdir()) == [
        "config.json",
        "equity.zarr",
        "fingerprint.json",
        "liquidations.json",
        "metrics.json",
        "report.html",
        "weights.zarr",
    ]

    weights = result.weights["weight"]
    symbols = [str(s) for s in weights["symbol"].values]

    values = weights.transpose("timestamp", "symbol").values
    rebalance_rows = np.flatnonzero(np.isfinite(values).all(axis=1))
    assert rebalance_rows.size >= 2

    for junk in (PENNY, ILLIQUID):
        column = values[rebalance_rows, symbols.index(junk)]
        np.testing.assert_array_equal(
            column, np.zeros_like(column), err_msg=f"{junk} was selected"
        )

    # Teeth: something IS being bought, so the zeros above are a filter
    # working rather than an empty book.
    assert (values[rebalance_rows] > 0).any()

    config = json.loads((result.run_dir / "config.json").read_text())
    for kind in ("factors", "labels"):
        entry = config["model"][kind][0]
        assert (
            entry["name"]
            == "quantlab.factor.universe_filter.UniverseFilteredFactor"
        )
        assert "factor" in entry and "dataset" in entry["factor"]


def test_holding_that_leaves_the_universe_is_sold_at_the_next_rebalance(
    backtest_run,
):
    """LS-4: a drop-out is sold at the OPEN after the next rebalance bar.

    It is held through the drop -- up to `rebalance_periods - 1` bars late --
    because the backtester is deliberately not edited: the universe filter
    acts through NaN features, and eligibility is only re-evaluated on a
    rebalance bar.
    """
    _, _, _, result = backtest_run

    weights = result.weights["weight"].transpose("timestamp", "symbol")
    symbols = [str(s) for s in weights["symbol"].values]
    bars = weights["timestamp"].values
    column = symbols.index(DROPOUT)

    assert float(weights.values[R1_ROW, column]) > 0, (
        "the drop-out must be selectable at the rebalance BEFORE it drops"
    )
    assert float(weights.values[R2_ROW, column]) == 0.0, (
        "it must be ineligible at the first rebalance after it drops"
    )

    orders = result.simulation.orders
    order_symbol = orders["symbol"].values.astype(str)
    order_side = orders["side"].values.astype(str)
    order_time = orders["timestamp"].values.astype("datetime64[ns]")
    sold = order_time[(order_symbol == DROPOUT) & (order_side == "Sell")]

    # The sale fills at the open of the bar AFTER the rebalance (D-05).
    #
    # Compared as datetime64 throughout: `.tolist()` on a datetime64[ns] array
    # yields integer NANOSECONDS, so a `np.datetime64 in set(...)` membership
    # test is always False and would fail on correct behaviour.
    expected_fill = bars[R2_ROW + 1].astype("datetime64[ns]")
    assert bool((sold == expected_fill).any()), (
        f"expected a {DROPOUT} sell at {expected_fill}, got {sold}"
    )

    # ...and NOT before: it really was held across the drop.
    early = sold[
        (sold > bars[R1_ROW + 1].astype("datetime64[ns]"))
        & (sold <= bars[R2_ROW].astype("datetime64[ns]"))
    ]
    assert early.size == 0, f"{DROPOUT} was sold early at {early}"


def _strip_dates(node):
    """Recursively drop every `start_date`/`end_date` from a config subtree."""
    if isinstance(node, dict):
        return {
            key: _strip_dates(value)
            for key, value in node.items()
            if key not in ("start_date", "end_date")
        }
    if isinstance(node, list):
        return [_strip_dates(value) for value in node]
    return node


def _comparable(config: dict) -> dict:
    """A run config with the dates `run()` re-derives normalised away.

    `BaseBacktester.run()` re-dates every factor to "warm-up start .. window
    end" and widens its dataset start, and `config.json` is written AFTER that
    -- so it records the re-dated values. A rebuild constructs the model
    afresh, and `BaseModel`'s config setter calls `_reset_factors_config`,
    which re-derives every factor's dates from the MODEL's dates. The two
    therefore disagree on exactly those fields once a run has happened.

    MEASURED 2026-09-15 on the UNWRAPPED `tests.backtest_fixtures` backtester:
    saved factor 2024-02-05..2024-03-11 vs rebuilt 2024-01-01..2024-02-09, and
    saved dataset start 2024-01-31 vs rebuilt 2023-12-27 -- the same shape,
    with no universe wrapper anywhere in it. This is pre-existing model-layer
    bookkeeping, which is why `tests/test_backtest_rebuild.py` asserts config
    equality only on a backtester that has NOT been run.

    Everything else still compares exactly: every scalar, the price dataset,
    the wrapper's four parameters and the inner factor's identity. The claim
    that actually matters -- the re-run reproduces the run -- is asserted
    separately, on weights and equity.
    """
    config = {k: v for k, v in config.items() if k != "data_fingerprint"}
    model = dict(config["model"])
    for kind in ("factors", "labels"):
        model[kind] = _strip_dates(model[kind])
    config["model"] = model
    return config


def test_rebuilt_backtester_from_run_config_reproduces_the_run(backtest_run):
    """A run directory rebuilds into the same wrapped objects and re-runs
    to byte-identical weights (D-25)."""
    root, _, _, result = backtest_run
    saved = json.loads((result.run_dir / "config.json").read_text())

    rebuilt = load_backtester_from_config(saved)

    for kind in ("factors", "labels"):
        item = getattr(rebuilt.config.model.config, kind)[0]
        assert type(item) is UniverseFilteredFactor
        assert item.min_price == MIN_PRICE
        assert item.min_dollar_volume == MIN_DOLLAR_VOLUME
        assert item.window == WINDOW

    rebuilt_config = json.loads(json.dumps(to_jsonable(rebuilt.get_config())))
    assert _comparable(rebuilt_config) == _comparable(saved)
    # The inner factor's identity survives the round trip un-normalised.
    assert (
        rebuilt_config["model"]["factors"][0]["factor"]["name"]
        == "tests.universe_fixtures.RankCloseFactor"
    )

    rerun = rebuilt.run()
    assert rerun.run_dir != result.run_dir
    xr.testing.assert_identical(
        xr.open_zarr(result.run_dir / "weights.zarr").load(),
        xr.open_zarr(rerun.run_dir / "weights.zarr").load(),
    )
    np.testing.assert_array_equal(
        xr.open_zarr(result.run_dir / "equity.zarr")["value"].values,
        xr.open_zarr(rerun.run_dir / "equity.zarr")["value"].values,
    )


# ---------------------------------------------------------------------------
# DL heads work across windows -- BY DESIGN, not as a documented limitation
# ---------------------------------------------------------------------------


def test_dl_head_predicts_across_windows_when_a_trained_symbol_leaves_the_universe(
    store, tmp_path
):
    """An `MLPRegressor` trained on window A predicts on window B, in which a
    TRAINED symbol is out of the universe the whole time.

    This is the concrete payoff of the 2026-09-15 symbol-axis decision.
    `MLPRegressor` encodes symbol POSITION (it flattens `[S*F]`), and
    `DLModel._align_prediction_symbols` REFUSES a panel missing a trained
    symbol. Under the superseded "drop whole-window-NaN columns" rule the
    drop-out would be absent from window B and this call would raise. Because
    only the date-independent ticker rule removes symbols, it is present as an
    all-NaN column instead, and its predictions are simply NaN.
    """
    _, config = store
    model = make_wrapped_model(
        tmp_path,
        config,
        model_cls=MLPRegressor,
        window=WINDOW,
        config_cls=DLConfig,
        hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
        epochs=1,
        batch_size=16,
        num_workers=0,
        **MODEL_DATES,
    )
    model.collect()
    model.train()
    assert DROPOUT in model._trained_symbols

    # Re-date to window B exactly the way `BaseBacktester._redate_factors` does.
    window_b = (day(50, N_BARS), day(N_BARS - 1, N_BARS))
    for factor in model.config.factors:
        factor.config.start_date, factor.config.end_date = window_b
        factor._reset_dataset_config()
        factor.config.dataset.read(overwrite=True)

    features = model._collect_all_features()
    symbols = [str(s) for s in features["symbol"].values]
    assert DROPOUT in symbols, "the trained symbol must still be on the axis"
    assert np.isnan(features["rank_close"].sel(symbol=DROPOUT).values).all()

    # The whole point: this does not raise.
    predictions = model.predict_panel(features)

    label = list(predictions.data_vars)[0]
    assert np.isnan(predictions[label].sel(symbol=DROPOUT).values).all()
    assert np.isfinite(predictions[label].values).any(), (
        "every prediction is NaN -- the panel carried no in-universe symbol, "
        "so the assertion above proves nothing"
    )
