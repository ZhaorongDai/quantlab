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

This first version outputs two styles, Size (descriptor LNCAP) and Beta
(descriptor BETA), as described in Menchero, Orr and Wang, *The Barra US
Equity Model (USE4), Methodology Notes* (MSCI, 2011), and its Empirical
Notes, Appendix A. The other USE4 styles are added in later versions.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields

import numpy as np
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output
from KunQuant.ops import BackRef, Log, Select, SetInfOrNanToValue, WindowedSum
from KunQuant.Stage import Function

from quantlab.factor.config import FactorConfig
from quantlab.factor.kunquant import FactorKunQuant
from quantlab.factor.kunquant_ops import (
    CapWeightedStandardize,
    CrossSectionalTopN,
    CrossSectionalWeightedMean,
    EWBeta,
    SigmaClip,
)


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
        for BETA to be computed; with fewer it is NaN. Our choice.
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
    253
    """

    price_column: str = "adjClose"
    market_cap_column: str = "marketcap"
    risk_free_column: str = "risk_free"
    estimation_universe_size: int = 3000
    beta_window: int = 252
    beta_half_life: float = 63.0
    beta_min_observations: int = 63
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
        """Bars of history the first exposure needs: the BETA window plus one return.

        Set ``FactorConfig.warmup_bars`` to at least this.

        Examples
        --------
        >>> BarraStyleParameters().warmup_bars
        253
        """
        return self.beta_window + 1

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
    - LNCAP is ``log(marketcap[t])``;
    - BETA is the slope of an exponentially weighted least-squares fit of
      the stock's excess return on the market's excess return over the last
      ``beta_window`` bars (half-life ``beta_half_life``), both in excess of
      the risk-free rate of the bar before;
    - each descriptor is standardized over the universe (market cap at
      ``t-1`` weighted mean 0, equally weighted standard deviation 1), a
      value beyond ``data_error_sigma`` becomes NaN, and the rest are
      clipped to ``clip_sigma``;
    - each style is its descriptor standardized again, so that the
      universe's styles have mean 0 and standard deviation 1 exactly.

    Outputs, all on ``(timestamp, symbol)``:

    - ``desc_lncap``, ``desc_beta``: the standardized, clipped descriptors;
    - ``style_size``, ``style_beta``: the style exposures;
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
        ``BarraStyleParameters.warmup_bars`` (253 by default).

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
    ...     warmup_bars=253,
    ...     dataset=[prices, daily, risk_free],
    ...     mode="batch",
    ...     data_columns=("adjClose", "marketcap", "risk_free"),
    ...     file_path="barra_style.zarr",
    ... ))
    >>> factor.get_factor_names()
    ('desc_lncap', 'desc_beta', 'style_size', 'style_beta', 'estu')
    """

    _OUTPUTS = ("desc_lncap", "desc_beta", "style_size", "style_beta", "estu")

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

            beta = EWBeta(excess, market_excess, params.beta_window, params.beta_half_life)
            observed = WindowedSum(_present(excess + market_excess), params.beta_window)
            beta = Select(
                observed >= float(params.beta_min_observations), beta, ConstantOp("nan")
            )

            def descriptor(raw: OpBase) -> OpBase:
                standardized = CapWeightedStandardize(raw, cap_before, estu)
                return SigmaClip(standardized, params.data_error_sigma, params.clip_sigma)

            lncap = descriptor(Log(cap))
            beta = descriptor(beta)
            # Size and Beta have one descriptor each, so each style is its
            # descriptor standardized again.
            outputs = {
                "desc_lncap": lncap,
                "desc_beta": beta,
                "style_size": CapWeightedStandardize(lncap, cap_before, estu),
                "style_beta": CapWeightedStandardize(beta, cap_before, estu),
                "estu": estu,
            }
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
