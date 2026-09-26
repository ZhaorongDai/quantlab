"""XGB-TD (pytabkit) pipeline on WRDS CRSP daily data.

Data -> Alpha101 + Alpha158 factors -> forward-return label ->
``XGBTDRegressor`` (pytabkit's tuned-default XGBoost; missing features are
filled with 0) -> TopN cross-sectional backtest against a buy-and-hold ETF
benchmark. Edit ``Settings`` below and run
``uv run python examples/wrds_us_equity/xgb_td.py``, or step through the
``# %%`` cells. The data root is ``common.DATA_ROOT``.
"""

# %% Settings
from dataclasses import dataclass, field

from common import (
    BacktestSettings,
    DataSettings,
    TrainSettings,
    run_model_pipeline,
)
from quantlab.ml_model.xgb_td import XGBTDRegressor


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    #: pytabkit ``XGB_TD_Regressor`` constructor arguments.
    hyperparameters: dict = field(default_factory=lambda: {
        "n_estimators": 1000, "n_threads": 8,
    })
    #: Weights & Biases: ``"online"`` (needs ``wandb login``), ``"offline"``
    #: (``wandb sync`` later) or ``"disabled"``.
    wandb_mode: str = "online"


SETTINGS = Settings()


# %% Run everything
def main(s: Settings = SETTINGS):
    return run_model_pipeline(
        XGBTDRegressor, "xgb_td", s.hyperparameters,
        s.data, s.train, s.backtest, s.wandb_mode,
    )


if __name__ == "__main__":
    main()
