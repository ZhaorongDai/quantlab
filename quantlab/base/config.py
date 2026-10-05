"""Dataclass configurations threaded through every layer of the pipeline.

Each domain object (dataset, factor, model, backtester, acquisition, universe
catalog) is constructed from one of the config classes here and exposes it as
``self.config``. Dataset, factor and model configs are frozen: a field
cannot be assigned after creation, and a changed config is a new one made
with ``dataclasses.replace``. The object's config setter normalises the
config it is given into a new config, filling in derived values such as
open-ended dates and the ``name`` field, which records the owning class's
dotted import path so the object can be rebuilt from the serialised dict (see
``quantlab.core.component``); the caller's config is never edited. ``to_dict()``
on each config produces that dict, each field declared with ``component()``
written as its component's own config. Acquisition, universe and backtest configs
are not frozen.

Throughout, a *panel* is an ``xarray.Dataset`` indexed by ``timestamp`` and
``symbol``. Several configs describe US-equity data from WRDS (Wharton
Research Data Services, a university data platform). CRSP (the Center for
Research in Security Prices) supplies daily stock data keyed by PERMNO, a
permanent integer id that stays with a security when its ticker changes. TAQ
(Trade and Quote) supplies intraday quotes, and the NBBO (National Best Bid
and Offer) is the best bid and ask across all US exchanges at each moment.

Fields are documented with ``#:`` comments so the meaning of each one sits
beside its definition.
"""

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component, config_to_dict
from quantlab.core.config import FrozenConfig
from quantlab.tracking.base import NullTracker, Tracker
from quantlab.execution.rules import ExecutionSettings
from quantlab.enums.data import Market

if TYPE_CHECKING:
    from quantlab.dataset.base import MarketDataset
    from quantlab.factor.base import Factor
    from .model import BaseModel
    from .portfolio import PortfolioConstructor, RiskModel


@dataclass(frozen=True)
class TopNConfig(FrozenConfig):
    """Config of ``TopNConstructor``: equal-weight top-n selection.

    Examples
    --------
    >>> cfg = TopNConfig(direction="long_short", top_n=20)
    >>> cfg.score_label is None
    True
    """

    #: ``"long_only"`` holds the top ``top_n`` names; ``"long_short"`` also
    #: shorts the bottom ``top_n``, with the two books disjoint.
    direction: Literal["long_only", "long_short"]
    #: Number of names selected per book at every rebalance.
    top_n: int
    #: The label whose prediction ranks the symbols. ``None`` selects the
    #: predictor's first label.
    score_label: str | None = None


@dataclass(frozen=True)
class LedoitWolfConfig(FrozenConfig):
    """Config of ``LedoitWolfRiskModel``: a shrunk sample covariance of trailing returns.

    Examples
    --------
    >>> cfg = LedoitWolfConfig(lookback_bars=252)
    >>> cfg.lookback_bars, cfg.max_stale_bars
    (252, 5)
    """

    #: Bars of trailing one-bar returns the covariance is estimated from;
    #: at least 2. A symbol needs a finite return on every one of them.
    lookback_bars: int
    #: Largest staleness (bars since the symbol's last real price) a symbol
    #: may have at the bar and still be covered; a halt no longer than this
    #: stays in the estimate, flat returns and then its gap.
    max_stale_bars: int = 5


@dataclass(frozen=True, kw_only=True)
class MeanVarianceConfig(FrozenConfig):
    """Config of ``MeanVarianceOptimizer``: Markowitz weights with a turnover penalty.

    The optimiser maximises ``w @ mu - risk_aversion / 2 * w @ Sigma @ w -
    turnover_penalty * |w - w_current|_1``, with ``mu`` and ``Sigma`` on the
    span of ``expected_return_label``.

    Examples
    --------
    >>> cfg = MeanVarianceConfig(
    ...     expected_return_label="ret_5",
    ...     risk_model=LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=252)),
    ...     ic=0.05, risk_aversion=10.0, weight_cap=0.05,
    ... )
    >>> cfg.direction, cfg.calibration, cfg.turnover_penalty, cfg.candidate_top_k
    ('long_only', 'grinold', 0.0, None)
    >>> cfg.volatility_label is None
    True
    """

    #: The label whose prediction gives the expected return; its span sets
    #: the horizon of the expected return and the covariance.
    expected_return_label: str
    #: The risk model estimating the covariance of one-bar returns.
    risk_model: "RiskModel" = component()
    #: Risk aversion ``lambda`` of the variance penalty.
    risk_aversion: float
    #: How the prediction becomes the expected return ``mu``.
    #: ``"grinold"``: ``mu = ic * sigma * z``, ``z`` the prediction's
    #: cross-sectional z-score, so any score will do. ``"raw"``: ``mu`` is
    #: the prediction itself, which must be in the label's own units (the
    #: predictor reports the label as ``"raw"`` in ``label_scales``).
    calibration: Literal["grinold", "raw"] = "grinold"
    #: Information coefficient of the Grinold calibration, for example a CV
    #: run's mean IC; required by ``"grinold"``, unused by ``"raw"``.
    ic: float | None = None
    #: Penalty ``kappa`` per unit of one-way turnover against the current
    #: weights; 0 trades freely.
    turnover_penalty: float = 0.0
    #: Largest absolute weight of one symbol.
    weight_cap: float = 1.0
    #: ``"long_only"``: fully invested, non-negative weights (``sum(w) =
    #: 1``). ``"long_short"``: dollar-neutral (``sum(w) = 0``) with gross
    #: exposure ``|w|_1 <= 1``, a ceiling, not an equality: the optimiser
    #: may leave part of the book uninvested.
    direction: Literal["long_only", "long_short"] = "long_only"
    #: Optimise only over the ``candidate_top_k`` symbols with the largest
    #: expected return (largest absolute one for ``"long_short"``) plus
    #: every symbol currently held; the rest get 0.0. ``None`` optimises
    #: over every tradable symbol.
    candidate_top_k: int | None = None
    #: The label whose prediction gives each symbol's volatility over the
    #: span, such as ``Volatility``; it must have the expected-return
    #: label's span and a ``"raw"`` scale. The covariance is then these
    #: volatilities around the risk model's correlations, and the Grinold
    #: ``sigma`` is the prediction. ``None`` keeps the risk model's own
    #: (historical) volatilities.
    volatility_label: str | None = None


@dataclass(frozen=True)
class ModelConfig(FrozenConfig):
    """Config of every model head, torch or library.

    It holds only what both variants read. Everything a variant or a head
    reads for training goes in ``hyperparameters``, one flat dict. The base
    classes and the shipped heads read these reserved keys from it:

    ``epochs``
        Epoch cap of a ``TorchModel``; default 100, a positive integer.
    ``lr``
        Learning rate of the default ``TorchModel._init_optim``; default
        ``1e-3``.
    ``early_stopping``, ``early_stopping_patience``
        The library's native early stopping in the shipped library heads;
        default off and 5 rounds (or the library's own unit).
    ``training_target``
        A ``LibraryModel``'s training target: ``"cs_rank"`` or
        ``"cs_zscore"``, applied per bar; unset trains on the raw label.
        Set, ``label_scales`` is ``"standardized"``; metrics still read the
        raw label.
    ``batch_size``, ``num_workers``, ``panel_device``, ``panel_dtype``
        Reserved for the torch data loader and training panel.

    A ``LibraryModel``'s ``_init_model`` receives the dict without the
    library keys above. A ``TorchModel``'s receives the whole dict, reserved
    keys included, so a torch head never splats it into a network; it reads
    its own keys by name, or drops the reserved ones with
    ``BaseModel.head_hyperparameters``.

    ``train_start``, ``train_end``, ``test_start`` and ``test_end`` bound the
    training and test windows; rolling cross-validation overwrites them fold
    by fold. ``start_date`` and ``end_date`` bound all the data the model
    collects; they are passed to every factor and label per request.

    Examples
    --------
    With ``factors`` and ``labels`` lists of factor objects built earlier:

    >>> cfg = ModelConfig(
    ...     factors=factors,
    ...     labels=labels,
    ...     model_save_dir="/data/models/xgb",
    ...     factor_data_strategy="read",
    ...     label_data_strategy="read",
    ...     train_start="2018-01-01",
    ...     train_end="2022-12-31",
    ...     test_start="2023-01-01",
    ...     test_end="2023-12-31",
    ...     hyperparameters={"max_depth": 6, "early_stopping": True},
    ... )
    >>> cfg.val_size, cfg.hyperparameters["early_stopping"]
    (0.2, True)
    """

    #: The factors whose values form the model's input features.
    factors: list["Factor"] = component(many=True)
    #: The factors (labels) whose values form the prediction targets.
    labels: list["Factor"] = component(many=True)
    #: Root directory checkpoints and their ``config.json`` are written under.
    model_save_dir: str
    #: ``"read"`` loads factor values from their stores; ``"cal"`` computes
    #: them first.
    factor_data_strategy: Literal["read", "cal"]
    #: ``"read"`` loads label values from their stores; ``"cal"`` computes
    #: them first.
    label_data_strategy: Literal["read", "cal"]
    #: First date of data to collect, inclusive. ``None`` means no lower
    #: bound.
    start_date: str | None = None
    #: Last date of data to collect, inclusive. ``None`` means no upper bound.
    end_date: str | None = None

    #: Training and architecture settings, one flat dict; see the reserved
    #: keys above.
    hyperparameters: dict = field(default_factory=dict)
    #: Fraction of the training window held out, at its end, for validation.
    val_size: float = 0.2
    #: Seed applied to Python, numpy and, for torch heads, torch before
    #: training.
    random_seed: int = 42
    #: First date of the training window, inclusive.
    train_start: str | None = None
    #: Last date of the training window, inclusive.
    train_end: str | None = None
    #: First date of the test window, inclusive.
    test_start: str | None = None
    #: Last date of the test window, inclusive.
    test_end: str | None = None
    #: Where training runs are tracked (ADR 0015); the default
    #: ``NullTracker`` sends nothing anywhere.
    tracker: Tracker = component(default=NullTracker())

    #: Dotted import path of the model class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True)
class BacktestConfig:
    """Fields every backtester shares.

    Selection parameters are not here but on subclasses such as
    ``CrossSectionBacktestConfig``, so a time-series backtester does not carry
    cross-sectional fields. See ``docs/backtest.md``.

    Examples
    --------
    With ``price_dataset`` a market dataset and ``model`` a model built
    earlier:

    >>> cfg = BacktestConfig(
    ...     price_dataset=price_dataset,
    ...     model=model,
    ...     model_mode="load",
    ...     checkpoint="/data/models/xgb/best.joblib",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir="/data/backtests",
    ...     rebalance_periods=5,
    ... )
    >>> cfg.fees, cfg.slippage, cfg.init_cash
    (0.0005, 0.0005, 1000000.0)

    A config for ``run_weights()`` needs no model, and with
    ``output_dir=None`` the run stays in memory:

    >>> weights_cfg = BacktestConfig(
    ...     price_dataset=price_dataset,
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir=None,
    ...     rebalance_periods=5,
    ... )
    >>> weights_cfg.model, weights_cfg.model_mode, weights_cfg.output_dir
    (None, None, None)
    """

    #: The dataset whose prices the simulation trades on.
    price_dataset: "MarketDataset" = component()
    #: First date of the backtest window, inclusive.
    start_date: str
    #: Last date of the backtest window, inclusive.
    end_date: str

    #: Rebalance every this many bars.
    rebalance_periods: int

    #: Directory each run writes its own run directory under. ``None`` keeps
    #: the run in memory: the backtest writes no run directory and the
    #: result's ``run_dir`` is ``None``. It covers the backtest's own files
    #: only: with ``model_mode="train"`` the model still writes its checkpoint
    #: where its own config points.
    output_dir: str | None

    #: The model that produces the scores the target weights are built from:
    #: any object with the members of ``quantlab.base.backtest.Predictor``,
    #: such as a ``BaseModel`` (the annotation names the usual case, because
    #: this module does not import the backtest layer). ``run()`` and
    #: ``run_cv()`` require it; ``run_weights()`` backtests precomputed
    #: weights and ignores it, so it may be ``None`` there. ``model`` and
    #: ``model_mode`` are both set or both ``None``.
    model: "BaseModel | None" = component(default=None)
    #: ``"train"`` trains ``model`` on its own dates first; ``"load"`` restores
    #: a checkpoint (``checkpoint`` for ``run()``, ``cv_project_dir`` for
    #: ``run_cv()``). Required by ``run()`` and ``run_cv()``, like ``model``.
    model_mode: Literal["train", "load"] | None = None

    #: Checkpoint to restore in ``"load"`` mode for ``run()``: the file the
    #: model's ``train()`` returned (a model's checkpoint, an ensemble's
    #: ``run.json``).
    checkpoint: str | None = None
    #: The walk-forward unit a ``train_cv`` run wrote, read by ``run_cv()``
    #: to replay each fold with its own checkpoint.
    cv_project_dir: str | None = None

    #: Proportional fee per trade.
    fees: float = 0.0005
    #: Proportional slippage per trade.
    slippage: float = 0.0005
    #: Starting cash of the simulated portfolio.
    init_cash: float = 1_000_000.0
    #: The price a target weight is sized against. ``"fill"`` (the default)
    #: sizes at the fill price of the bar the order executes on, against the
    #: portfolio valued at those prices: vectorbt's own default. ``"valuation"``
    #: sizes at the valuation price of the signal bar t (its close), against
    #: the portfolio valued at t's close, as a broker order placed after the
    #: close must be sized; the order still fills at t+1's fill price. The
    #: vectorbt engine and portfolio construction's replay both read it,
    #: through ``execution``.
    sizing_basis: Literal["fill", "valuation"] = "fill"

    #: A market dataset holding exactly one symbol, for example the QQQ store
    #: built by ``CrspDatasetConfig.qqq_benchmark``. It is a
    #: ``(timestamp, symbol)`` panel like ``price_dataset`` and needs the same
    #: fill and valuation columns. When set, every run also simulates buying
    #: and holding it on the strategy's bars and reports the portfolio
    #: against it (``benchmark`` and ``relative`` metric blocks, the
    #: benchmark NAV and the excess-return and excess-drawdown charts).
    benchmark_dataset: "MarketDataset | None" = component(default=None)

    #: Where each run's metrics and report are tracked (ADR 0015); the
    #: default ``NullTracker`` sends nothing anywhere.
    tracker: Tracker = component(default=NullTracker())

    #: Dotted import path of the backtester class; filled by the config setter.
    name: str | None = None

    @property
    def execution(self) -> ExecutionSettings:
        """The run's execution settings: its sizing basis, fees and slippage.

        Examples
        --------
        >>> cfg.execution == ExecutionSettings(cfg.sizing_basis, cfg.fees, cfg.slippage)
        True
        """
        return ExecutionSettings(self.sizing_basis, self.fees, self.slippage)

    def to_dict(self):
        """Return the config as a plain dict, each component as its own config.

        The dataset, model, rule and tracker fields are declared with
        ``quantlab.core.component.component`` and written as their
        ``get_config()``; nothing is deep-copied through ``dataclasses.asdict``,
        so no panel or trained model is copied.

        Examples
        --------
        >>> cfg.to_dict()["model"]["name"] == model.import_path
        True
        >>> cfg.to_dict()["rebalance_periods"]
        5
        """
        return config_to_dict(self)


@dataclass(kw_only=True)
class CrossSectionBacktestConfig(BacktestConfig):
    """Config of a cross-sectional backtest: a predictor's scores turned into weights.

    On every rebalance bar ``constructor``, a portfolio construction rule,
    turns the predictions of that bar into the weights to hold after it.

    Examples
    --------
    >>> cfg = CrossSectionBacktestConfig(
    ...     price_dataset=price_dataset,
    ...     model=model,
    ...     model_mode="load",
    ...     checkpoint="/data/models/xgb/best.joblib",
    ...     start_date="2023-01-01",
    ...     end_date="2023-12-31",
    ...     output_dir="/data/backtests",
    ...     rebalance_periods=5,
    ...     constructor=TopNConstructor(TopNConfig(direction="long_short", top_n=20)),
    ... )
    >>> cfg.constructor
    TopNConstructor(direction='long_short', top_n=20, score_label=None)
    """

    #: The portfolio construction rule (a ``PortfolioConstructor``, such as
    #: ``quantlab.portfolio.predefined.top_n.TopNConstructor``) that turns
    #: each rebalance bar's predictions into target weights.
    constructor: "PortfolioConstructor" = component()


@dataclass(kw_only=True)
class WeightsBacktestConfig(BacktestConfig):
    """Config of a backtest of given target weights, market conventions included.

    The backtester reading it (``WeightsVectorBt``) has no fixed market: the fill and
    valuation columns and the annualization are fields here instead of a class
    constant, so one class serves any frame a caller brings. It takes no model; its
    weights come to ``run_weights()``. ``direction`` and ``top_n`` record the
    top-N selection that built the weights from scores, when one did; they are shown
    in the report and never used to select.

    Examples
    --------
    >>> import pandas as pd
    >>> from quantlab.base.config import WeightsBacktestConfig
    >>> from quantlab.dataset.memory import FrameDataset
    >>> price_dataset = FrameDataset(pd.DataFrame({
    ...     "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
    ...     "symbol": ["AAA", "AAA"], "open": [10.0, 11.0], "close": [10.5, 11.5],
    ... }))
    >>> cfg = WeightsBacktestConfig(
    ...     price_dataset=price_dataset,
    ...     start_date="2024-01-01",
    ...     end_date="2024-06-28",
    ...     output_dir=None,
    ...     rebalance_periods=1,
    ...     fill_price_column="open",
    ...     valuation_price_column="close",
    ...     trading_days_per_year=252,
    ...     session_minutes_per_day=390,
    ... )
    >>> cfg.model, cfg.top_n
    (None, None)
    """

    #: Price variable orders fill at.
    fill_price_column: str
    #: Price variable the portfolio is valued at after each bar.
    valuation_price_column: str
    #: Trading days in a year, to annualize daily and longer bars.
    trading_days_per_year: int
    #: Minutes of one trading session, to annualize intraday bars.
    session_minutes_per_day: int
    #: The selection side the weights were built with, or ``None`` for weights
    #: given as such. A record only.
    direction: Literal["long_only", "long_short"] | None = None
    #: The names per side the weights were built with, or ``None``. A record only.
    top_n: int | None = None
