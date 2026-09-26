"""RealMLP (pytabkit) pipeline on WRDS CRSP daily data.

Data -> Alpha101 + Alpha158 factors -> forward-return label ->
``RealMLPRegressor`` (pytabkit's tuned-default MLP; it robust-scales its
inputs itself and fills missing features with 0) -> TopN cross-sectional
backtest against a buy-and-hold ETF benchmark. Edit ``Settings`` below and
run ``uv run python examples/wrds_us_equity/realmlp.py``, or step through
the ``# %%`` cells. The data root is ``common.DATA_ROOT``.
"""

# %% Settings
from dataclasses import dataclass, field

from common import (
    BacktestSettings,
    DataSettings,
    TrainSettings,
    run_model_pipeline,
)
from quantlab.ml_model.realmlp import RealMLPRegressor


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    data: DataSettings = field(default_factory=DataSettings)
    train: TrainSettings = field(default_factory=TrainSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)
    #: pytabkit ``RealMLP_TD_Regressor`` constructor arguments; early
    #: stopping patience counts epochs here.
    hyperparameters: dict = field(default_factory=lambda: {
        "n_epochs": 256, "device": "cpu", "n_threads": 8,
    })
    #: Weights & Biases: ``"online"`` (needs ``wandb login``), ``"offline"``
    #: (``wandb sync`` later) or ``"disabled"``.
    wandb_mode: str = "online"


SETTINGS = Settings()


# %% Run everything
def main(s: Settings = SETTINGS):
    return run_model_pipeline(
        RealMLPRegressor, "realmlp", s.hyperparameters,
        s.data, s.train, s.backtest, s.wandb_mode,
    )


if __name__ == "__main__":
    main()
