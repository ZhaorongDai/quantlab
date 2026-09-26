"""XGBoost pipeline on WRDS CRSP daily data.

Data -> Alpha101 + Alpha158 factors -> forward-return label ->
``XGBoostRegressor`` (``xgb.train`` with native early stopping) -> TopN
cross-sectional backtest against a buy-and-hold ETF benchmark. Edit
``Settings`` below and run ``uv run python examples/wrds_us_equity/xgb.py``,
or step through the ``# %%`` cells. The data root is ``common.DATA_ROOT``.
"""

# %% Settings
from dataclasses import dataclass, field

from common import (
    BacktestSettings,
    DataSettings,
    TrainSettings,
    run_model_pipeline,
)
from quantlab.ml_model.xgb import XGBoostRegressor


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    #: ``xgb.train`` parameters; trained on the pooled loss, early-stopped
    #: on the validation RMSE.
    hyperparameters: dict = field(default_factory=lambda: {
        "num_boost_round": 1000, "eta": 0.05, "max_depth": 6, "nthread": 8,
    })
    #: Weights & Biases: ``"online"`` (needs ``wandb login``), ``"offline"``
    #: (``wandb sync`` later) or ``"disabled"``.
    wandb_mode: str = "online"


SETTINGS = Settings()


# %% Run everything
def main(s: Settings = SETTINGS):
    return run_model_pipeline(
        XGBoostRegressor, "xgb", s.hyperparameters,
        s.data, s.train, s.backtest, s.wandb_mode,
    )


if __name__ == "__main__":
    main()
