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

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.ml_model.xgb import XGBoostRegressor
from quantlab.utils.module import load_model_from_config

from tests.universe_fixtures import (
    DROPOUT,
    ILLIQUID,
    MIN_DOLLAR_VOLUME,
    MIN_PRICE,
    PENNY,
    RANK_CLOSE_FACTOR_NAMES,
    WARRANT,
    day,
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
