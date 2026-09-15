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

from quantlab.base.config import FactorConfig, PolarsFactorConfig
from quantlab.dataset.stock import StockDataset
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.utils.module import load_model_from_config

from tests.backtest_fixtures import PastReturnFactor, make_stock_dataset
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
    WARRANT,
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

#: Measured against `data/data/reference/universe.parquet` on 2026-09-15; see
#: the comment block above `UniverseFilteredFactor.NON_COMMON_TICKER_PATTERNS`.
SURVIVE_LOCKS = [
    "BRK-A", "BRK-B", "BF-A", "BF-B", "HEI-A", "LEN-B", "MOG-A", "UA-C",
    "MKC-V", "CWEN-A", "PBR-A", "GOOGL", "AAPL", "MSFT", "ACIW", "AAWW",
    "ACHR", "AMKR", "ALTR",
    # 5-letter and dotted commons that are INDEX CONSTITUENTS, so a rule that
    # excluded any of them would be provably wrong (falsifier 1).
    "BATRA", "BATRK", "CMCSA", "CMCSK", "DISCA", "DISCK", "LBTYA", "LBTYK",
    "LILAK", "QRTEA", "RYAAY", "STRZA", "BRK.B", "BF.B",
]

EXCLUDE_LOCKS = [
    "AACIW", "AAC-WS", "AACBR", "AACBU", "AAC-U", "ACP-R", "ACP-R-W",
    "HYZNW", "GNS-R", "SST-WS", "FINS-R-W", "EMISU", "BARK-WS", "AACTWS",
    "ZWZZT", "ZVZZT", "ZXZZT", "NTEST-A", "CTEST", "JNJ-WD", "AAM-P-A",
]

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
    into an equivalent wrapper. The symbol-axis assertions are the 2026-09-15
    user decision: the ticker-rule symbol leaves the axis, the two
    threshold-failing symbols STAY as all-NaN columns.
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

    # Ticker rule: gone from the axis entirely.
    assert WARRANT not in symbols
    # Threshold failures: KEPT, as all-NaN columns.
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
        assert rebuilt_item.exclude_non_common == original.exclude_non_common

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


@pytest.fixture(scope="module")
def unfiltered_mask_factor(store):
    """The same, with the static ticker rule switched off."""
    _, config = store
    return wrap(
        make_rank_close_factor(config), window=WINDOW, exclude_non_common=False
    )


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


def test_mask_excludes_a_warrant_ticker_unless_the_rule_is_disabled(
    mask_factor, unfiltered_mask_factor
):
    """A rich, liquid warrant is still out -- the ticker rule is static and
    independent of the data. With `exclude_non_common=False` it is in."""
    close = np.full((8, 2), 50.0)
    volume = np.full((8, 2), 1_000_000.0)
    panel = hand_panel(close, volume, symbols=["AAPL", WARRANT])

    mask = mask_factor.compute_universe_mask(panel)
    assert np.isnan(mask.sel(symbol=WARRANT).values).all()
    assert (mask.sel(symbol="AAPL").values[WINDOW - 1 :] == 1.0).all()

    unfiltered = unfiltered_mask_factor.compute_universe_mask(panel)
    assert (unfiltered.sel(symbol=WARRANT).values[WINDOW - 1 :] == 1.0).all()


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
# The measured ticker rule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", SURVIVE_LOCKS)
def test_common_tickers_survive_the_rule(ticker):
    """Class shares, 5-letter commons and dotted commons are COMMON STOCK.

    A rule that merely looked for "has a hyphen" or "is five letters" would
    delete Berkshire Hathaway and Comcast from the roster, silently.
    """
    assert UniverseFilteredFactor.is_common_ticker(ticker), ticker


@pytest.mark.parametrize("ticker", EXCLUDE_LOCKS)
def test_non_common_tickers_are_excluded(ticker):
    """Warrants, rights, units, when-issued lines, preferreds and the
    exchange test symbols are all removed."""
    assert not UniverseFilteredFactor.is_common_ticker(ticker), ticker


def test_is_common_ticker_is_case_insensitive():
    assert not UniverseFilteredFactor.is_common_ticker("aaciw")
    assert not UniverseFilteredFactor.is_common_ticker("aac-ws")
    assert UniverseFilteredFactor.is_common_ticker("brk-b")


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
    assert WARRANT not in symbols
    assert PENNY in symbols
    assert np.isnan(read_back["rank_close"].sel(symbol=PENNY).values).all()


# ---------------------------------------------------------------------------
# LS-3 symbol axis: ticker-rule symbols leave, threshold failures stay
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["cal", "read"])
def test_ticker_rule_symbols_absent_in_every_window_threshold_failures_kept(
    store, tmp_path, strategy
):
    """The symbol axis must NOT depend on the date window (2026-09-15 decision).

    Window B lies entirely after the drop-out leaves the universe, so under the
    superseded "drop a whole-window-NaN column" rule its axis would be narrower
    than window A's -- and `DLModel._align_prediction_symbols` raises when a
    trained symbol is missing from a later window. Only the STATIC ticker rule
    removes symbols, so the two axes are identical.
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

        assert WARRANT not in symbols, "the ticker rule symbol must be gone"
        assert PENNY in symbols and ILLIQUID in symbols
        assert np.isnan(features["rank_close"].sel(symbol=PENNY).values).all()

    assert axes[0] == axes[1], (
        "the symbol axis changed between two date windows; a DL head trained "
        "on the first would refuse to predict on the second"
    )

    # Window B is entirely after the drop: the drop-out is kept as a NaN column.
    assert DROPOUT in axes[1]


def test_exclude_non_common_false_drops_no_symbol(store):
    _, config = store
    factor = wrap(
        make_rank_close_factor(config), window=WINDOW, exclude_non_common=False
    )

    features = factor.cal().get_features().load()

    assert set(str(s) for s in features["symbol"].values) == set(SYMBOLS16)


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
