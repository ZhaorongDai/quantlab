"""The configs of the risk layer.

A factor risk model is constructed from one of these and exposes it as
``self.config``. They are frozen; the model's config setter normalises the config
it is given into a new one (the ``name`` field the model is rebuilt from, a list
of datasets merged into one). The exposures factor and the price dataset are
declared with ``component()`` and written as their own configs.

``FactorRiskConfig`` describes any factor risk model: which exposures it
regresses on (continuous style exposures, an optional industry code, an
optional country factor) and how. ``Use4RiskConfig`` is the same with USE4's
factor set on the ``BarraStyle`` exposures as defaults.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.dataset.base import MarketDataset
    from quantlab.factor.base import Factor

#: The regression weights a factor risk model may use, by config name.
REGRESSION_WEIGHTINGS = ("sqrt_cap", "cap", "equal")


@dataclass(kw_only=True, frozen=True)
class FactorRiskConfig(FrozenConfig):
    """Config of ``quantlab.risk.base.FactorRiskModel``, any factor risk model.

    The exposures come from ``exposures``, any factor whose outputs include
    ``style_names`` (and ``industry_name`` and ``estu_name`` when set); the
    returns, market caps and risk-free rate come from ``dataset``. For
    exposures that use a risk-free rate themselves, it should be the dataset
    they read, so exposures and returns agree on what "excess" means.

    Examples
    --------
    A model of two style factors and a market (country) factor over every
    symbol, with ``signals`` a factor outputting ``value`` and ``quality``
    and ``prices`` a dataset with ``adjClose``, ``marketcap`` and
    ``risk_free``:

    >>> cfg = FactorRiskConfig(
    ...     exposures=signals, dataset=prices, exposure_data_strategy="cal",
    ...     style_names=("value", "quality"),
    ...     regression_path="risk/two_style_regression.zarr",
    ... )
    >>> cfg.country, cfg.industry_name, cfg.weighting
    (True, None, 'sqrt_cap')
    """

    #: The factor whose outputs are the exposures.
    exposures: "Factor" = component()
    #: The dataset holding ``price_column``, ``market_cap_column`` and
    #: ``risk_free_column``, or a list of datasets, merged into one.
    dataset: "MarketDataset | tuple" = component()
    #: ``"read"`` reads the exposures from their store, ``"cal"`` computes them.
    exposure_data_strategy: Literal["read", "cal"]
    #: The continuous exposures regressed on, one factor each.
    style_names: tuple[str, ...]
    #: Path of the Zarr store of the regression (factor returns, specific
    #: returns and diagnostics).
    regression_path: str | None = None
    #: Whether the model has a country (market) factor, an exposure of 1 for
    #: every symbol.
    country: bool = True
    #: The exposures variable holding each symbol's industry code; ``None``
    #: for a model without industry factors.
    industry_name: str | None = None
    #: The industry codes, one factor each; a symbol with another code has no
    #: industry exposure and is left out. Used only with ``industry_name``.
    industries: tuple[int, ...] = ()
    #: The exposures variable that is 1 inside the estimation universe, the
    #: symbols the regression is fitted on; ``None`` fits on every symbol.
    estu_name: str | None = None
    #: Regression weights: the square root of the previous bar's market cap,
    #: the market cap, or equal.
    weighting: Literal["sqrt_cap", "cap", "equal"] = "sqrt_cap"
    #: Dataset variable holding the split- and dividend-adjusted close.
    price_column: str = "adjClose"
    #: Dataset variable holding the market capitalization.
    market_cap_column: str = "marketcap"
    #: Dataset variable holding the risk-free rate as a decimal return per bar.
    risk_free_column: str = "risk_free"
    #: When set, the risk-free rate is ``risk_free_column`` of this one symbol,
    #: broadcast across the others, as ``BarraStyleParameters.risk_free_symbol``.
    risk_free_symbol: str | None = None
    #: Fewest fitted members an industry needs on a bar to be in that bar's
    #: regression. Our choice.
    min_industry_members: int = 5
    #: Robust distance from the cross-sectional median, in standard deviations
    #: estimated as 1.4826 times the median absolute deviation, beyond which a
    #: fitted return is trimmed to that bound for the fit. Our choice.
    return_outlier_sigma: float = 5.0
    #: Path of the Zarr store of the estimates (factor covariance and specific
    #: risk), computed from the regression store.
    estimate_path: str | None = None
    #: Half-life, in bars, of the exponential weights of the factor
    #: volatilities (USE4S: 84 trading days; USE4L: 252).
    volatility_half_life: float = 84.0
    #: Bars of factor returns the factor volatilities are estimated from, the
    #: exponential weights truncated there. Our choice: three half-lives.
    volatility_window: int = 252
    #: Half-life, in bars, of the exponential weights of the factor
    #: correlations (USE4S and USE4L: 504 trading days).
    correlation_half_life: float = 504.0
    #: Bars of factor returns the factor correlations are estimated from. Our
    #: choice: three half-lives.
    correlation_window: int = 1512
    #: Half-life, in bars, of the exponential weights of the specific
    #: volatilities (USE4S: 84 trading days; USE4L: 252).
    specific_half_life: float = 84.0
    #: Bars of specific returns the specific volatilities are estimated from.
    #: Our choice: three half-lives.
    specific_window: int = 252
    #: Newey-West lags of the factor volatilities (USE4S and USE4L: 5); 0 for
    #: none.
    volatility_lags: int = 5
    #: Newey-West lags of the factor correlations (USE4S and USE4L: 2); 0 for
    #: none.
    correlation_lags: int = 2
    #: Newey-West lags of the specific volatilities (USE4S and USE4L: 5); 0
    #: for none.
    specific_lags: int = 5
    #: Half-life, in bars, of the exponential weights of the specific returns'
    #: autocorrelations behind the specific Newey-West adjustment (USE4S and
    #: USE4L: 252 trading days).
    specific_autocorrelation_half_life: float = 252.0
    #: Bars of specific returns the autocorrelations are estimated from. Our
    #: choice: three half-lives.
    specific_autocorrelation_window: int = 756
    #: Fewest observations in its window for a factor variance, a pair's
    #: correlation, a symbol's specific volatility or a lagged autocorrelation
    #: to be estimated; NaN with fewer (a lag term: none). Our choice.
    min_observations: int = 21

    #: Dotted import path of the risk model class; filled by the config setter.
    name: str | None = None


#: The style exposures ``BarraStyle`` outputs, in its order.
USE4_STYLES = (
    "style_size",
    "style_beta",
    "style_momentum",
    "style_residual_volatility",
    "style_nonlinear_size",
    "style_nonlinear_beta",
    "style_liquidity",
    "style_dividend_yield",
    "style_book_to_price",
    "style_earnings_yield",
    "style_leverage",
    "style_growth",
)

#: The Fama-French 48 industry codes ``SharadarIndustryDataset`` stores.
FF48_INDUSTRIES = tuple(range(1, 49))


@dataclass(kw_only=True, frozen=True)
class Use4RiskConfig(FactorRiskConfig):
    """Config of ``quantlab.risk.predefined.use4.Use4RiskModel``.

    ``FactorRiskConfig`` with USE4's factor set on the ``BarraStyle``
    outputs as defaults: a country factor, the Fama-French 48 industries of
    ``industry``, the 12 styles, fitted on ``estu`` with square-root-of-cap
    weights. The column fields default to ``BarraStyleParameters``' and must
    match the ones the exposures were computed with. The half-lives default
    to USE4S's (Table 4.1, Table 5.1), and so do the Newey-West lags (5 for
    factor volatilities, 2 for correlations, 5 for specific volatilities with
    an autocorrelation half-life of 252); USE4L is
    ``volatility_half_life=252``, ``specific_half_life=252`` (and windows of
    three half-lives, 756), with the same lags.

    Examples
    --------
    With ``style`` a ``BarraStyle`` and ``prices`` the dataset it reads:

    >>> cfg = Use4RiskConfig(
    ...     exposures=style, dataset=prices, exposure_data_strategy="read",
    ...     regression_path="risk/use4_regression.zarr",
    ... )
    >>> len(cfg.style_names), len(cfg.industries), cfg.estu_name
    (12, 48, 'estu')
    """

    style_names: tuple[str, ...] = USE4_STYLES
    industry_name: str | None = "industry"
    industries: tuple[int, ...] = FF48_INDUSTRIES
    estu_name: str | None = "estu"
