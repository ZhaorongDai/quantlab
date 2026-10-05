"""Barra-style risk-factor exposures in the manner of MSCI's USE4 model.

A *style factor* describes one characteristic that explains part of how a
stock's return moves with others: its size, its sensitivity to the market
(beta), its momentum and so on. A stock's *exposure* to a style is a number
saying how much of that characteristic it has, standardized across stocks
so that exposures of different styles are comparable. Each style is built
from one or more *descriptors*, raw measurements such as the log of market
capitalization.

Exposures are standardized over an *estimation universe* (ESTU), here the
``estimation_universe_size`` largest companies by the previous bar's market
cap, re-ranked every bar inside the graph: on every bar the cap-weighted
mean exposure of the universe is 0 and the equally weighted standard
deviation is 1. Symbols outside the universe are shifted and scaled by the
same numbers, so every symbol with data gets an exposure.

This version outputs the price-based styles: Size (descriptor LNCAP),
Beta (BETA), Momentum (RSTR), Residual Volatility (DASTD, CMRA, HSIGMA),
Non-linear Size and Non-linear Beta, as described in Menchero, Orr and
Wang, *The Barra US Equity Model (USE4), Methodology Notes* (MSCI, 2011),
and its Empirical Notes, Appendix A. The fundamentals-based styles are
added in later versions.

A style with several descriptors is their fixed-weight sum over the
descriptors a symbol has, the weights renormalized over those present, so
a symbol missing one descriptor still gets the style. Residual Volatility,
Non-linear Size and Non-linear Beta are orthogonalized: each is replaced by
its residual from a per-bar weighted least-squares regression, with an
intercept, on Beta and Size, on Size, and on Beta respectively, fitted in
the estimation universe and applied to every symbol, so each has zero
weighted correlation with its regressors there.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output
from KunQuant.ops import BackRef, Log, Select, SetInfOrNanToValue, Sqrt, WindowedSum
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_ops import (
    CMRA,
    CapWeightedStandardize,
    CrossSectionalTopN,
    CrossSectionalWeightedMean,
    CrossSectionalWLSResidual,
    CrossSectionalWLSResidual2,
    EWBeta,
    EWResidualStd,
    EWSum,
    EWVar,
    RenormalizedCombine,
    SigmaClip,
)

#: The regression weights an orthogonalization may use, by config name.
ORTHOGONALIZATION_WEIGHTINGS = ("sqrt_cap", "cap", "equal")


@dataclass(frozen=True)
class BarraStyleParameters:
    """Formula parameters for ``BarraStyle``, read from ``config.kwargs``.

    Every field can be set through ``FactorConfig.kwargs``; unknown keys are
    refused. Windows are counted in bars of the panel the factor runs on, so
    the defaults assume daily bars. Each field says whether its default is
    USE4's published value or a choice of ours where MSCI publishes none.

    Parameters
    ----------
    price_column : str, default "adjClose"
        Panel variable holding the split- and dividend-adjusted close; the
        return of bar ``t`` is ``price[t] / price[t-1] - 1``.
    market_cap_column : str, default "marketcap"
        Panel variable holding the market capitalization (Sharadar DAILY's
        ``marketcap``, in USD).
    risk_free_column : str, default "risk_free"
        Panel variable holding the risk-free rate as a decimal return per
        bar, the same on every symbol. It is lagged one bar before use,
        because the rate of a day is published the next business day.
    estimation_universe_size : int, default 3000
        Number of largest companies, by the previous bar's market cap, in
        the estimation universe. Our choice: USE4 uses the MSCI USA IMI,
        which this approximates; there is no buffer band.
    beta_window : int, default 252
        Bars in the BETA regression window (USE4).
    beta_half_life : float, default 63.0
        Half-life, in bars, of the BETA regression weights (USE4).
    beta_min_observations : int, default 63
        Fewest bars with both a stock and a market return in the BETA window
        for BETA and HSIGMA to be computed; with fewer they are NaN. Our
        choice.
    momentum_window : int, default 504
        Bars in the RSTR window (USE4).
    momentum_half_life : float, default 126.0
        Half-life, in bars, of the RSTR weights (USE4).
    momentum_lag : int, default 21
        Bars RSTR skips before its window, so the latest month is left out
        (USE4).
    momentum_min_observations : int, default 252
        Fewest returns in the RSTR window for RSTR to be computed. Our
        choice.
    dastd_window : int, default 252
        Bars in the DASTD window (USE4).
    dastd_half_life : float, default 42.0
        Half-life, in bars, of the DASTD weights (USE4).
    cmra_months : int, default 12
        Months in CMRA's cumulative range (USE4).
    cmra_month_length : int, default 21
        Bars in one CMRA month (USE4).
    volatility_min_observations : int, default 63
        Fewest returns in the DASTD or CMRA window for that descriptor to be
        computed. Our choice.
    dastd_weight, cmra_weight, hsigma_weight : float, default 0.75, 0.15, 0.10
        Weights of DASTD, CMRA and HSIGMA in Residual Volatility (USE4).
    orthogonalization_weighting : str, default "sqrt_cap"
        Regression weights of the orthogonalizations: ``"sqrt_cap"`` (the
        square root of the previous bar's market cap), ``"cap"`` or
        ``"equal"``. Our choice: USE4 orthogonalizes "on a
        regression-weighted basis" and its factor regression weights by the
        square root of cap, but MSCI does not publish the weights of this
        step.
    data_error_sigma : float, default 10.0
        Standardized descriptor magnitude beyond which a value is treated as
        a data error and dropped. Our choice: USE4 drops data errors but does
        not publish the threshold.
    clip_sigma : float, default 3.0
        Standardized descriptor magnitude descriptors are clipped to (USE4).

    Examples
    --------
    >>> params = BarraStyleParameters(estimation_universe_size=500)
    >>> params.panel_columns
    ('adjClose', 'marketcap', 'risk_free')
    >>> params.warmup_bars
    526
    """

    price_column: str = "adjClose"
    market_cap_column: str = "marketcap"
    risk_free_column: str = "risk_free"
    estimation_universe_size: int = 3000
    beta_window: int = 252
    beta_half_life: float = 63.0
    beta_min_observations: int = 63
    momentum_window: int = 504
    momentum_half_life: float = 126.0
    momentum_lag: int = 21
    momentum_min_observations: int = 252
    dastd_window: int = 252
    dastd_half_life: float = 42.0
    cmra_months: int = 12
    cmra_month_length: int = 21
    volatility_min_observations: int = 63
    dastd_weight: float = 0.75
    cmra_weight: float = 0.15
    hsigma_weight: float = 0.10
    orthogonalization_weighting: str = "sqrt_cap"
    data_error_sigma: float = 10.0
    clip_sigma: float = 3.0

    @classmethod
    def from_config(cls, config: FactorConfig) -> BarraStyleParameters:
        """Build and validate parameters from ``config.kwargs``.

        Parameters
        ----------
        config : FactorConfig
            The factor config whose ``kwargs`` holds the overrides.

        Returns
        -------
        BarraStyleParameters
            The validated parameters.

        Raises
        ------
        ValueError
            If ``config.kwargs`` holds an unknown key or a value that fails
            :meth:`validate`.

        Examples
        --------
        >>> from dataclasses import replace
        >>> config = replace(config, kwargs={"beta_window": 126})
        >>> BarraStyleParameters.from_config(config).beta_window
        126
        """
        kwargs = config.kwargs or {}
        unknown = sorted(set(kwargs) - {field.name for field in fields(cls)})
        if unknown:
            raise ValueError(f"BarraStyle received unknown config.kwargs: {unknown}")
        params = cls(**kwargs)
        params.validate()
        return params

    @property
    def panel_columns(self) -> tuple[str, ...]:
        """Panel variables the factor reads, what ``config.data_columns`` must name.

        Examples
        --------
        >>> BarraStyleParameters().panel_columns
        ('adjClose', 'marketcap', 'risk_free')
        """
        return (self.price_column, self.market_cap_column, self.risk_free_column)

    @property
    def warmup_bars(self) -> int:
        """Bars of history the first exposure needs: the longest window plus one return.

        The longest is RSTR's, its window plus its lag. Set
        ``FactorConfig.warmup_bars`` to at least this.

        Examples
        --------
        >>> BarraStyleParameters().warmup_bars
        526
        """
        return 1 + max(
            self.beta_window,
            self.momentum_window + self.momentum_lag,
            self.dastd_window,
            self.cmra_months * self.cmra_month_length,
        )

    def validate(self) -> None:
        """Raise if a window, a threshold or a column name cannot work.

        Raises
        ------
        ValueError
            If a window or count is too small, the thresholds are out of
            order, or the column names are empty or repeated.

        Examples
        --------
        >>> BarraStyleParameters(beta_min_observations=1).validate()
        Traceback (most recent call last):
            ...
        ValueError: beta_min_observations must be between 2 and beta_window (252), got 1
        """
        if self.estimation_universe_size < 2:
            raise ValueError("estimation_universe_size must be at least 2")
        if self.beta_window < 2:
            raise ValueError("beta_window must be at least 2")
        if not self.beta_half_life > 0:
            raise ValueError("beta_half_life must be positive")
        if not 2 <= self.beta_min_observations <= self.beta_window:
            raise ValueError(
                f"beta_min_observations must be between 2 and beta_window "
                f"({self.beta_window}), got {self.beta_min_observations}"
            )
        for name in ("momentum", "dastd"):
            window, half_life = getattr(self, f"{name}_window"), getattr(self, f"{name}_half_life")
            if window < 2 or not half_life > 0:
                raise ValueError(f"{name}_window must be at least 2 and {name}_half_life positive")
        if self.momentum_lag < 0:
            raise ValueError("momentum_lag cannot be negative")
        if not 1 <= self.momentum_min_observations <= self.momentum_window:
            raise ValueError(
                f"momentum_min_observations must be between 1 and momentum_window "
                f"({self.momentum_window}), got {self.momentum_min_observations}"
            )
        if self.cmra_months < 1 or self.cmra_month_length < 1:
            raise ValueError("cmra_months and cmra_month_length must be at least 1")
        shortest = min(self.dastd_window, self.cmra_months * self.cmra_month_length)
        if not 2 <= self.volatility_min_observations <= shortest:
            raise ValueError(
                f"volatility_min_observations must be between 2 and the shorter of the "
                f"DASTD and CMRA windows ({shortest}), got {self.volatility_min_observations}"
            )
        if not min(self.dastd_weight, self.cmra_weight, self.hsigma_weight) > 0:
            raise ValueError("dastd_weight, cmra_weight and hsigma_weight must be positive")
        if self.orthogonalization_weighting not in ORTHOGONALIZATION_WEIGHTINGS:
            raise ValueError(
                f"orthogonalization_weighting must be one of {ORTHOGONALIZATION_WEIGHTINGS}, "
                f"got {self.orthogonalization_weighting!r}"
            )
        if not 0 < self.clip_sigma <= self.data_error_sigma:
            raise ValueError("need 0 < clip_sigma <= data_error_sigma")
        if any(not name for name in self.panel_columns):
            raise ValueError("BarraStyle input column names cannot be empty")
        if len(set(self.panel_columns)) != len(self.panel_columns):
            raise ValueError("BarraStyle input column names must be unique")


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite and 0 elsewhere."""
    return SetInfOrNanToValue(value * 0.0 + 1.0, 0.0)


class BarraStyle(FactorKunQuant):
    """USE4-style exposures: standardized descriptors and style factors.

    Reads three panel variables (see ``BarraStyleParameters``): the adjusted
    close, the market cap and the risk-free rate, usually from a merge of a
    Sharadar price dataset, the Sharadar DAILY dataset and a risk-free
    series broadcast across symbols. On every bar ``t``:

    - the estimation universe is the ``estimation_universe_size`` symbols
      with the largest market cap at ``t-1``;
    - the market return is the universe's return, weighted by the market
      cap at ``t-1``;
    - returns are in excess of the risk-free rate of the bar before (the
      rate of a day is published the next business day); ``excess`` is
      ``r - r_f`` and ``log_excess`` is ``log(1 + r) - log(1 + r_f)``;
    - LNCAP is ``log(marketcap[t])``;
    - BETA is the slope of an exponentially weighted least-squares fit of
      ``excess`` on the market's excess return over the last
      ``beta_window`` bars (half-life ``beta_half_life``), and HSIGMA the
      weighted standard deviation of that fit's residual;
    - RSTR is the exponentially weighted sum (weights not normalized, as in
      USE4) of ``log_excess`` over ``momentum_window`` bars ending
      ``momentum_lag`` bars ago (half-life ``momentum_half_life``);
    - DASTD is the exponentially weighted standard deviation of ``excess``
      over ``dastd_window`` bars (half-life ``dastd_half_life``);
    - CMRA is ``log(1 + max Z) - log(1 + min Z)``, ``Z(T)`` the sum of
      ``log_excess`` over the last ``T`` months of ``cmra_month_length``
      bars, ``T = 1 .. cmra_months``; NaN when ``min Z <= -1``;
    - a windowed descriptor is NaN with fewer valid returns in its window
      than its ``*_min_observations``;
    - each descriptor is standardized over the universe (market cap at
      ``t-1`` weighted mean 0, equally weighted standard deviation 1), a
      value beyond ``data_error_sigma`` becomes NaN, and the rest are
      clipped to ``clip_sigma``;
    - Size, Beta and Momentum are their one descriptor standardized again;
    - Residual Volatility is ``0.75 DASTD + 0.15 CMRA + 0.10 HSIGMA``
      (``*_weight``) over the descriptors present, standardized,
      orthogonalized against Beta and Size and standardized again;
    - the NLSIZE descriptor is the cube of the Size exposure, standardized
      and clipped like any descriptor; Non-linear Size is it orthogonalized
      against Size and standardized again. NLBETA and Non-linear Beta are
      the same with Beta. The clip comes before the orthogonalization (our
      choice), so the final styles stay exactly orthogonal to their
      regressors;
    - an orthogonalization is a weighted least-squares regression with an
      intercept, weighted by ``orthogonalization_weighting`` and fitted in
      the universe; every symbol gets its residual, a symbol missing a
      regressor being taken at that regressor's weighted mean (our
      choice). Standardizing afterwards keeps the weighted correlation
      with the regressors at zero.

    Outputs, all on ``(timestamp, symbol)``:

    - ``desc_lncap``, ``desc_beta``, ``desc_rstr``, ``desc_dastd``,
      ``desc_cmra``, ``desc_hsigma``, ``desc_nlsize``, ``desc_nlbeta``: the
      standardized, clipped descriptors;
    - ``style_size``, ``style_beta``, ``style_momentum``,
      ``style_residual_volatility``, ``style_nonlinear_size``,
      ``style_nonlinear_beta``: the style exposures;
    - ``estu``: 1 inside the estimation universe of the bar, 0 outside.

    The graph runs in double precision. On 1000 bars of 3008 symbols, with
    a 252-bar BETA window of half-life 63, float32 runs 2x faster but leaves
    errors of 4e-7 in BETA and 6.6e-5 in standardized LNCAP against a
    float64 reference, where double leaves 4e-15 and 3e-10. The factor is batch only: ``mode="stream"`` is
    refused.

    Parameters
    ----------
    config : FactorConfig
        ``mode`` must be ``"batch"``, ``data_columns`` must name exactly
        ``BarraStyleParameters.panel_columns``, ``factor_names`` may pick any
        subset of the outputs, and ``warmup_bars`` should be at least
        ``BarraStyleParameters.warmup_bars`` (526 by default).

    Raises
    ------
    ValueError
        If ``mode`` is not ``"batch"``, ``data_columns`` does not match the
        parameters, ``factor_names`` names an unknown output, or ``kwargs``
        holds an unknown or invalid parameter.

    Examples
    --------
    ``prices`` is a ``SharadarStockDataset``, ``daily`` a
    ``SharadarDailyDataset`` and ``risk_free`` a dataset holding the
    ``risk_free`` variable on the same symbols.

    >>> factor = BarraStyle(FactorConfig(
    ...     warmup_bars=526,
    ...     dataset=[prices, daily, risk_free],
    ...     mode="batch",
    ...     data_columns=("adjClose", "marketcap", "risk_free"),
    ...     file_path="barra_style.zarr",
    ... ))
    >>> factor.get_factor_names()[:4]
    ('desc_lncap', 'desc_beta', 'desc_rstr', 'desc_dastd')
    """

    _OUTPUTS = (
        "desc_lncap",
        "desc_beta",
        "desc_rstr",
        "desc_dastd",
        "desc_cmra",
        "desc_hsigma",
        "desc_nlsize",
        "desc_nlbeta",
        "style_size",
        "style_beta",
        "style_momentum",
        "style_residual_volatility",
        "style_nonlinear_size",
        "style_nonlinear_beta",
        "estu",
    )

    def _parameters(self) -> BarraStyleParameters:
        """Return the validated parameters for the current config."""
        return BarraStyleParameters.from_config(self.config)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return every output."""
        return self._OUTPUTS

    def _validate_config(self) -> None:
        """Refuse stream mode, and columns or outputs the parameters do not produce."""
        if self.config.mode != "batch":
            raise ValueError(
                f"{self.class_name} is batch only: its exposures are standardized "
                f"over a re-ranked estimation universe on every bar, and no "
                f"bar-by-bar graph is built; got mode={self.config.mode!r}"
            )
        params = self._parameters()
        columns = tuple(self.config.data_columns)
        if len(columns) != len(set(columns)) or set(columns) != set(params.panel_columns):
            raise ValueError(
                f"{self.class_name} config.data_columns must exactly match "
                f"{params.panel_columns}; got {columns}"
            )
        unknown = sorted(set(self.get_factor_names()) - set(self._OUTPUTS))
        if unknown:
            raise ValueError(f"{self.class_name} received unknown factor_names: {unknown}")

    def init_stream(self):
        """Refuse: ``BarraStyle`` has no streaming graph.

        Raises
        ------
        ValueError
            Always.

        Examples
        --------
        >>> factor.init_stream()
        Traceback (most recent call last):
            ...
        ValueError: BarraStyle is batch only and has no streaming graph
        """
        raise ValueError(f"{self.class_name} is batch only and has no streaming graph")

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph, emitting only the outputs in ``factor_names``."""
        params = self._parameters()
        wanted = set(self.get_factor_names())
        builder = Builder()
        with builder:
            price = Input(params.price_column)
            cap = Input(params.market_cap_column)
            risk_free = BackRef(Input(params.risk_free_column), 1)

            cap_before = BackRef(cap, 1)
            estu = CrossSectionalTopN(cap_before, params.estimation_universe_size)
            stock_return = price / BackRef(price, 1) - 1.0
            market_return = CrossSectionalWeightedMean(stock_return, cap_before * estu)
            excess = stock_return - risk_free
            market_excess = market_return - risk_free
            log_excess = Log(stock_return + 1.0) - Log(risk_free + 1.0)
            regression_weight = {
                "sqrt_cap": Sqrt(cap_before),
                "cap": cap_before,
                "equal": cap_before * 0.0 + 1.0,
            }[params.orthogonalization_weighting]

            def counted(raw: OpBase, values: OpBase, window: int, least: int) -> OpBase:
                """``raw`` where ``values`` has at least ``least`` valid bars in ``window``."""
                observed = WindowedSum(_present(values), window)
                return Select(observed >= float(least), raw, ConstantOp("nan"))

            def standardize(value: OpBase) -> OpBase:
                return CapWeightedStandardize(value, cap_before, estu)

            def descriptor(raw: OpBase) -> OpBase:
                return SigmaClip(standardize(raw), params.data_error_sigma, params.clip_sigma)

            beta_window, beta_half_life = params.beta_window, params.beta_half_life
            joint = excess + market_excess
            beta = counted(
                EWBeta(excess, market_excess, beta_window, beta_half_life),
                joint, beta_window, params.beta_min_observations,
            )
            hsigma = counted(
                EWResidualStd(excess, market_excess, beta_window, beta_half_life),
                joint, beta_window, params.beta_min_observations,
            )
            lagged = BackRef(log_excess, params.momentum_lag) if params.momentum_lag else log_excess
            rstr = counted(
                EWSum(lagged, params.momentum_window, params.momentum_half_life),
                lagged, params.momentum_window, params.momentum_min_observations,
            )
            dastd = counted(
                Sqrt(EWVar(excess, params.dastd_window, params.dastd_half_life)),
                excess, params.dastd_window, params.volatility_min_observations,
            )
            cmra_window = params.cmra_months * params.cmra_month_length
            cmra = counted(
                CMRA(log_excess, params.cmra_months, params.cmra_month_length),
                log_excess, cmra_window, params.volatility_min_observations,
            )

            desc = {
                "lncap": descriptor(Log(cap)),
                "beta": descriptor(beta),
                "rstr": descriptor(rstr),
                "dastd": descriptor(dastd),
                "cmra": descriptor(cmra),
                "hsigma": descriptor(hsigma),
            }
            style_size = standardize(desc["lncap"])
            style_beta = standardize(desc["beta"])
            desc["nlsize"] = descriptor(style_size * style_size * style_size)
            desc["nlbeta"] = descriptor(style_beta * style_beta * style_beta)
            residual_volatility = standardize(RenormalizedCombine(
                [desc["dastd"], desc["cmra"], desc["hsigma"]],
                [params.dastd_weight, params.cmra_weight, params.hsigma_weight],
            ))
            outputs = {f"desc_{name}": value for name, value in desc.items()}
            outputs.update({
                "style_size": style_size,
                "style_beta": style_beta,
                "style_momentum": standardize(desc["rstr"]),
                "style_residual_volatility": standardize(CrossSectionalWLSResidual2(
                    residual_volatility, style_beta, style_size, regression_weight, estu
                )),
                "style_nonlinear_size": standardize(CrossSectionalWLSResidual(
                    desc["nlsize"], style_size, regression_weight, estu
                )),
                "style_nonlinear_beta": standardize(CrossSectionalWLSResidual(
                    desc["nlbeta"], style_beta, regression_weight, estu
                )),
                "estu": estu,
            })
            for name, value in outputs.items():
                if name in wanted:
                    Output(value, name)
        return Function(builder.ops, name="barra_style")

    def _kunquant_inputs(
        self, inputs: xr.Dataset
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """Export ``data_columns`` as float64 arrays for the double-precision graph.

        ``to_kunquant`` exports float32, which would round a USD market cap
        to seven digits before the graph sees it, so the panel is renamed to
        the shared names as ``to_kunquant`` does and exported in float64.
        """
        panel = self.config.dataset.to_shared_names(inputs).sortby(["timestamp", "symbol"])
        arrays = {
            column: np.ascontiguousarray(panel[column].to_numpy().astype(np.float64))
            for column in self.config.data_columns
        }
        return arrays, panel["symbol"].values, panel["timestamp"].values

    def _make(self):
        """Compile the graph for batch runs in double precision.

        ``no_fast_stat`` recomputes the rolling count exactly instead of
        updating it incrementally. ``allow_unaligned`` lets the symbol count
        be any number on x86; on ARM ``compute`` pads the symbol axis
        instead (see ``FactorKunQuant._pad_symbols``).
        """
        module_name = self.__class__.__name__
        compiler = KunCompilerConfig(
            dtype="double",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=platform.machine().lower() not in {"arm64", "aarch64"},
            options={"no_fast_stat": True},
        )
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )


__all__ = ["BarraStyle", "BarraStyleParameters"]
