"""Probe (verify-work 03.11, incidental finding): when something inside
`_backtest_window` raises, the D-27 fingerprint comparison is skipped -- even
though the factor fingerprints were already recorded and already differ.
So the operator gets the downstream error and NO indication the data changed.
"""
import json

import pandas as pd
import pytest
from loguru import logger

import quantlab.utils.module as module_utils
from quantlab.backtest.us_equity import USEquityCrossectionSelectStockVectorBt
from quantlab.base.config import CrossSectionBacktestConfig
from tests.backtest_fixtures import (
    SYMBOLS, make_model, make_stock_dataset, train_checkpoint, write_price_store,
)

N_BARS = 60
BARS = pd.bdate_range("2024-01-01", periods=N_BARS)
TRAIN_END_BAR, WINDOW_START_BAR, WINDOW_END_BAR = 24, 30, 50
FINGERPRINT_WARNING = "data fingerprint mismatch"


def _day(ts): return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _model_dates():
    return dict(start_date=_day(BARS[0]), end_date=_day(BARS[29]),
                train_start=_day(BARS[0]), train_end=_day(BARS[TRAIN_END_BAR]),
                test_start=_day(BARS[TRAIN_END_BAR + 1]), test_end=_day(BARS[29]))


def _backtester(root, dataset_config, *, checkpoint=None):
    return USEquityCrossectionSelectStockVectorBt(CrossSectionBacktestConfig(
        price_dataset=make_stock_dataset(dataset_config),
        model=make_model(root / "backtest", dataset_config, **_model_dates()),
        model_mode="load", checkpoint=None if checkpoint is None else str(checkpoint),
        start_date=_day(BARS[WINDOW_START_BAR]), end_date=_day(BARS[WINDOW_END_BAR]),
        output_dir=str(root / "runs"), rebalance_periods=2, direction="long_only",
        top_n=2, fees=0.0, slippage=0.0, init_cash=1_000_000.0,
    ))


def test_raise_inside_backtest_window_skips_the_fingerprint_comparison(tmp_path, monkeypatch):
    messages = []
    hid = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        store = tmp_path / "store"
        dataset_config = write_price_store(store, n_bars=N_BARS)
        checkpoint = train_checkpoint(
            make_model(tmp_path / "train", dataset_config, **_model_dates()))
        first = _backtester(tmp_path, dataset_config, checkpoint=checkpoint).run()
        saved = json.loads((first.run_dir / "config.json").read_text())

        # A REAL data change at the same path: one symbol gone -> the factor and
        # price fingerprints both differ (digest + n_symbols).
        write_price_store(store, symbols=SYMBOLS[:-1], n_bars=N_BARS)

        # --- control: without any raise, the mismatch IS reported --------------
        messages.clear()
        module_utils.load_backtester_from_config(saved).run()
        control = [m for m in messages if FINGERPRINT_WARNING in m]
        print(f"\n=== CONTROL (no raise): {len(control)} fingerprint warning(s) ===")
        for m in control:
            print("   ", m[:150])
        assert control, "control must show the mismatch is detectable"

        # --- probe: a raise inside _backtest_window, after fingerprints exist --
        messages.clear()
        rebuilt = module_utils.load_backtester_from_config(saved)
        real_predict = rebuilt.config.model.predict_panel

        def boom(features):
            # Stands in for any real post-fingerprint failure, e.g. DLModel's
            # _assert_symbol_types_match on a ticker-era checkpoint vs a PERMNO
            # panel, or "the feature panel lacks N of the symbols".
            raise ValueError("predict_panel: representative downstream failure")

        monkeypatch.setattr(rebuilt.config.model, "predict_panel", boom)
        with pytest.raises(ValueError, match="representative downstream failure"):
            rebuilt.run()

        recorded = sorted(rebuilt._fingerprints)
        probe = [m for m in messages if FINGERPRINT_WARNING in m]
        print(f"\n=== PROBE (raise inside _backtest_window) ===")
        print(f"    fingerprints ALREADY recorded at raise time: {recorded}")
        print(f"    fingerprint warnings emitted: {len(probe)}")
        assert recorded, "factor fingerprints are recorded before predict_panel"
        assert probe == [], (
            "THE GAP: fingerprints were recorded and differ, but the comparison "
            f"never ran, so the operator sees none of it. got: {probe}")
        print("    -> GAP CONFIRMED: recorded, differing, never compared.")
    finally:
        logger.remove(hid)
