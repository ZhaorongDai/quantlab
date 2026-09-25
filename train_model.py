"""Ad hoc example: load an RNN classifier on BTCUSDT and backtest its signals.

Builds Alpha101 and Alpha158 factors plus three forward-return labels for
BTCUSDT through the config factories, loads an ``RNNClassifier`` checkpoint
(``QUANTLAB_CHECKPOINT_PATH`` or a hardcoded trial path), predicts a
two-month window, turns the predicted classes into long/short signals
resampled to 30 minutes and runs a ``vectorbt`` signal backtest, writing
``portfolio_plot.html``. Runs at import with machine-specific paths and a
checkpoint that must already exist. Not part of the library.
"""

import json
import os

import pandas as pd
import plotly.io as pio
import torch
import vectorbt as vbt

from quantlab.base.config import DLConfig
from quantlab.config import alpha101_config, alpha158_config, spot_label_config
from quantlab.dl_model.rnn_classification import RNNClassifier
from quantlab.factor.alpha101 import Alpha101SpotKline
from quantlab.factor.alpha158 import Alpha158SpotKline
from quantlab.label.spot import SpotReturn
from quantlab.utils.module import load_model_from_config

label3 = SpotReturn(
    spot_label_config("ret_1m", n_forward_periods=120, symbols=["BTCUSDT"])
)
label2 = SpotReturn(
    spot_label_config("ret_1m", n_forward_periods=60, symbols=["BTCUSDT"])
)
label1 = SpotReturn(
    spot_label_config("ret_1m", n_forward_periods=30, symbols=["BTCUSDT"])
)
alpha101 = Alpha101SpotKline(alpha101_config(symbols=["BTCUSDT"]))
alpha158 = Alpha158SpotKline(alpha158_config(symbols=["BTCUSDT"]))

mc = DLConfig(
    start_date="2020-01-01",
    end_date="2025-01-01",
    train_start="2022-01-01",
    train_end="2022-08-01",
    test_start="2022-08-02",
    test_end="2022-10-01",
    factors=[alpha158, alpha101],
    labels=[label1, label2, label3],
    model_save_dir="./model_ckpt",
    factor_data_strategy="read",
    label_data_strategy="cal",
    batch_size=30000,
    epochs=50,
    num_workers=4,
    lr=1e-3,
    early_stopping=True,
    early_stopping_patience=5,
    hyperparameters={
        "hidden_sizes": [1024, 512, 256, 128, 64],
        "dropout_rates": [0.5, 0.3, 0.3, 0.3, 0.3],
        "hidden_sizes_linear": [64, 32, 16],
        "dropout_rates_linear": [0.3, 0.3, 0.3],
        "model_type": "gru",
    },
)
model = RNNClassifier(mc)
model.collect()
# model.train()
model.load(
    os.environ.get(
        "QUANTLAB_CHECKPOINT_PATH",
        "./model_ckpt/RNNClassifier_trial_20250909_183651/RNNClassifier_total/RNNClassifier_total.pth",
    )
)
data = model.data_backend.get_xarray_dataset()

data = data.sel(timestamp=slice("2024-01-01", "2024-03-01"))
factors = model.get_factor_names()
# ``DLModel.to_tensor`` orders the last axis exactly as ``factors`` declares
# it, with the same code training used. Sorting the columns by hand here
# would silently misalign them.
data = model.to_tensor(data[factors].fillna(0), factors)
predicts, _ = model.predict(data)
pred_probs = torch.softmax(predicts, dim=-1)
# pred_probs = predicts
pred_class = torch.argmax(pred_probs, dim=-1).detach().cpu()
confidence = torch.max(pred_probs, dim=-1)[0].detach().cpu()


price = model.config.factors[0].config.dataset.data_backend.get_xarray_dataset()
price = price.sel(timestamp=slice("2024-01-01", "2024-03-01"))
timestamp = price.timestamp.values

price = price["Close"].values.reshape(-1)
signals = pred_class.numpy().reshape(-1)

price = pd.Series(price, index=timestamp)
signals = pd.Series(signals, index=timestamp)
confidence = pd.Series(confidence.numpy().reshape(-1), index=timestamp)
data = pd.DataFrame(
    {"price": price, "signals": signals, "confidence": confidence}
)
# data["signals"] = data["signals"].shift(1)
# data["confidence"] = data["confidence"].shift(1)
data = data.dropna()

data.loc[data["signals"] == 0, "signals"] = -1
data["w_signal"] = data["signals"] * data["confidence"]
data = data.resample("30min").agg(
    price=pd.NamedAgg(column="price", aggfunc="last"),
    signals=pd.NamedAgg(column="signals", aggfunc="last"),
    confidence=pd.NamedAgg(column="confidence", aggfunc="last"),
    w_signal=pd.NamedAgg(column="w_signal", aggfunc="mean"),
)
data.loc[data["signals"] < 0, "short"] = 1
data.loc[data["signals"] > 0, "long"] = 1
data = data.fillna(0)
print(data)

# signals = np.roll(signals, 1)
# confidence = np.roll(confidence.numpy().reshape(-1), 1)

# entries = np.where(signals == 1, True, False)
# exits = np.where(signals == 0, True, False)
# short_entries = np.where(signals == 0, True, False)
# short_exits = np.where(signals == 1, True, False)

# Set signals with confidence below the threshold to -1.
# signals[confidence < 0.8] = -1
# signals[signals == 1] = -1


p = vbt.Portfolio.from_signals(
    data["price"],
    # init_cash=1000000,
    entries=data["long"] == 1,
    exits=data["short"] == 1,
    short_entries=data["short"] == 1,
    short_exits=data["long"] == 1,
    fees=0.0,
)
print(p.stats())
fig = p.plot()
pio.write_html(fig, "portfolio_plot.html")
# print(p.asset_flow())
# optimized_portfolio = optimize_crypto_backtest(
#     close=price,
#     signals=signals,
#     confidence=confidence,
#     init_cash=100.0,  # same as the original
#     fees=0.001,
#     confidence_threshold=0.7,    # raise the confidence threshold
#     min_hold_periods=10,         # hold at least 10 minutes
#     rebalance_threshold=0.15,    # rebalance only on a 15% confidence change
#     max_position_pct=1.0,        # at most 100% position
#     use_stops=False              # no stop-loss for now
# )

# print("Optimised backtest results:")
# print(optimized_portfolio.stats())
