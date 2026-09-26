"""Residual momentum from a rolling Fama-French three-factor regression.

Plain momentum ranks stocks by their past return. Residual momentum first
removes the part of each stock's return that common risk factors explain,
and ranks stocks by what is left (the *residual*). The factors used are the
three Fama-French factors: the market's excess return over the risk-free
rate (``mkt_rf``), small-minus-big size (``smb``) and high-minus-low value
(``hml``). For each stock and month the factor regresses the stock's excess
return on these three series over a rolling window, then sums the
residuals over a recent formation period and divides by their volatility.

The computation runs in KunQuant, a library that compiles a formula,
written as a graph of operators, to native code over a whole
``(timestamp, symbol)`` panel. The input panel must have one row per month.
``mkt_rf``, ``smb``, ``hml`` and ``risk_free`` are the same for every stock,
but KunQuant only reads panel variables, so they must be broadcast across
the ``symbol`` dimension before they reach this factor. Returns are
arithmetic decimal returns (0.01 means 1%), not percentages.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, fields
from typing import NoReturn, Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Driver import KunCompilerConfig
from KunQuant.jit import cfake
from KunQuant.Op import Builder, ConstantOp, Input, OpBase, Output, Rank
from KunQuant.ops import (
    Abs,
    BackRef,
    Select,
    Sqrt,
    WindowedAvg,
    WindowedCovariance,
    WindowedSum,
    WindowedVar,
)
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.utils.timer import Timer


@dataclass(frozen=True)
class ResidualMomentumParameters:
    """Formula parameters for ``ResidualMomentumFF3``, read from ``config.kwargs``.

    Every field can be set through ``FactorConfig.kwargs``; unknown keys are
    refused by ``from_config``.

    Parameters
    ----------
    regression_window : int, default 36
        Months in the rolling factor regression.
    formation_lookback : int, default 12
        Months back from the signal month where the formation period starts.
    skip_recent : int, default 1
        Most recent months left out of the formation period. Skipping the
        last month avoids the short-term reversal effect.
    ridge : float, default 1e-8
        Small value added to each factor variance to keep the regression
        solvable when factors are nearly collinear.
    determinant_floor : float, default 1e-18
        Smallest absolute value the factor covariance determinant may take
        before it is replaced, to avoid dividing by zero.
    variance_floor : float, default 1e-12
        Smallest residual variance used when computing volatility.
    emit_diagnostics : bool, default True
        Whether the factor also exposes the regression intermediates
        (alpha, betas, residual sum and volatility, determinant).
    return_column : str, default "stock_return"
        Panel variable holding each stock's monthly return.
    risk_free_column : str, default "risk_free"
        Panel variable holding the risk-free rate.
    market_column : str, default "mkt_rf"
        Panel variable holding the market excess return.
    smb_column : str, default "smb"
        Panel variable holding the size factor.
    hml_column : str, default "hml"
        Panel variable holding the value factor.

    Examples
    --------
    >>> params = ResidualMomentumParameters(regression_window=24)
    >>> params.regression_window, params.formation_lookback
    (24, 12)
    """

    regression_window: int = 36
    formation_lookback: int = 12
    skip_recent: int = 1
    ridge: float = 1.0e-8
    determinant_floor: float = 1.0e-18
    variance_floor: float = 1.0e-12
    emit_diagnostics: bool = True
    return_column: str = "stock_return"
    risk_free_column: str = "risk_free"
    market_column: str = "mkt_rf"
    smb_column: str = "smb"
    hml_column: str = "hml"

    @classmethod
    def from_config(cls, config: FactorConfig) -> "ResidualMomentumParameters":
        """Build and validate parameters from ``config.kwargs``.

        Parameters
        ----------
        config : FactorConfig
            The factor config whose ``kwargs`` holds the overrides.

        Returns
        -------
        ResidualMomentumParameters
            The validated parameters.

        Raises
        ------
        ValueError
            If ``config.kwargs`` holds an unknown key or a value
            that fails :meth:`validate`.

        Examples
        --------
        >>> config.kwargs = {"regression_window": 24}
        >>> ResidualMomentumParameters.from_config(config).regression_window
        24
        """
        kwargs = config.kwargs or {}
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(kwargs) - known)
        if unknown:
            raise ValueError(
                "ResidualMomentumFF3 received unknown config.kwargs: "
                f"{unknown}"
            )
        params = cls(**kwargs)
        params.validate()
        return params

    @property
    def formation_window(self) -> int:
        """Months in the formation period, ``formation_lookback - skip_recent``.

        Examples
        --------
        >>> ResidualMomentumParameters().formation_window
        11
        """
        return self.formation_lookback - self.skip_recent

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Panel variables the graph reads, in the order KunQuant receives them.

        Examples
        --------
        >>> ResidualMomentumParameters().input_columns
        ('stock_return', 'risk_free', 'mkt_rf', 'smb', 'hml')
        """
        return (
            self.return_column,
            self.risk_free_column,
            self.market_column,
            self.smb_column,
            self.hml_column,
        )

    def validate(self) -> None:
        """Raise if a window, a numerical floor or a column name cannot work.

        Raises
        ------
        ValueError
            If a window is too short, a floor is not positive, or
            the input column names are empty or repeated.

        Examples
        --------
        >>> ResidualMomentumParameters(regression_window=3).validate()
        Traceback (most recent call last):
            ...
        ValueError: regression_window must be at least 4 for FF3 + intercept
        """
        if self.regression_window < 4:
            raise ValueError(
                "regression_window must be at least 4 for FF3 + intercept"
            )
        if self.skip_recent < 0:
            raise ValueError("skip_recent must be non-negative")
        if self.formation_window < 2:
            raise ValueError(
                "formation_lookback - skip_recent must be at least 2"
            )
        if self.ridge < 0.0:
            raise ValueError("ridge must be non-negative")
        if self.determinant_floor <= 0.0 or self.variance_floor <= 0.0:
            raise ValueError("numerical floors must be positive")
        if any(not name for name in self.input_columns):
            raise ValueError("residual-momentum input column names cannot be empty")
        if len(set(self.input_columns)) != len(self.input_columns):
            raise ValueError("residual-momentum input column names must be unique")


def _cov(x: OpBase, y: OpBase, window: int) -> OpBase:
    """Return the rolling covariance of ``x`` and ``y`` over ``window`` bars.

    KunQuant's ``WindowedCovariance`` takes its arguments in the unusual order
    ``(x, window, y)``; this wrapper hides that.
    """
    return WindowedCovariance(x, window, y)


def _signed_floor(value: OpBase, floor: float) -> OpBase:
    """Return ``value``, or ``+-floor`` with the same sign when it is too close to 0.

    Used on divisors so a near-zero value cannot blow up the division.
    """
    fallback = Select(
        value >= 0.0,
        ConstantOp(float(floor)),
        ConstantOp(float(-floor)),
    )
    return Select(Abs(value) > float(floor), value, fallback)


def _ff3_coefficients(
    excess_return: OpBase,
    mkt_rf: OpBase,
    smb: OpBase,
    hml: OpBase,
    params: ResidualMomentumParameters,
) -> tuple[OpBase, OpBase, OpBase, OpBase, OpBase]:
    """Return rolling alpha, the three factor betas and the covariance determinant.

    Solves the three-factor least-squares regression in closed form. With
    ``a`` to ``f`` the entries of the 3x3 factor covariance matrix, the betas
    are the matrix inverse (written with cofactors) applied to the
    covariances between each factor and the excess return. The intercept
    ``alpha`` then follows from the window means.

    Parameters
    ----------
    excess_return : OpBase
        Stock return minus the risk-free rate.
    mkt_rf, smb, hml : OpBase
        The three factor series.
    params : ResidualMomentumParameters
        Supplies the window, the ridge and the determinant floor.

    Returns
    -------
    tuple[OpBase, OpBase, OpBase, OpBase, OpBase]
        ``(alpha, beta_mkt, beta_smb, beta_hml, determinant)``.
    """
    window = params.regression_window
    a = WindowedVar(mkt_rf, window) + params.ridge
    b = _cov(mkt_rf, smb, window)
    c = _cov(mkt_rf, hml, window)
    d = WindowedVar(smb, window) + params.ridge
    e = _cov(smb, hml, window)
    f = WindowedVar(hml, window) + params.ridge

    g0 = _cov(mkt_rf, excess_return, window)
    g1 = _cov(smb, excess_return, window)
    g2 = _cov(hml, excess_return, window)

    cof00 = d * f - e * e
    cof01 = c * e - b * f
    cof02 = b * e - c * d
    cof11 = a * f - c * c
    cof12 = b * c - a * e
    cof22 = a * d - b * b

    determinant = a * cof00 + b * cof01 + c * cof02
    safe_determinant = _signed_floor(determinant, params.determinant_floor)
    beta_mkt = (cof00 * g0 + cof01 * g1 + cof02 * g2) / safe_determinant
    beta_smb = (cof01 * g0 + cof11 * g1 + cof12 * g2) / safe_determinant
    beta_hml = (cof02 * g0 + cof12 * g1 + cof22 * g2) / safe_determinant
    alpha = (
        WindowedAvg(excess_return, window)
        - beta_mkt * WindowedAvg(mkt_rf, window)
        - beta_smb * WindowedAvg(smb, window)
        - beta_hml * WindowedAvg(hml, window)
    )
    return alpha, beta_mkt, beta_smb, beta_hml, determinant


def _formation_statistics(
    excess_return: OpBase,
    mkt_rf: OpBase,
    smb: OpBase,
    hml: OpBase,
    alpha: OpBase,
    beta_mkt: OpBase,
    beta_smb: OpBase,
    beta_hml: OpBase,
    params: ResidualMomentumParameters,
) -> tuple[OpBase, OpBase, OpBase]:
    """Return the formation-period residual sum, its volatility, and their ratio.

    The formation period is the ``formation_window`` months ending
    ``skip_recent`` months before the signal month. Residuals use the alpha
    and betas estimated at the signal month. The residual variance is
    expanded algebraically into factor variances and covariances, so no
    per-month residual series is ever materialized.

    Parameters
    ----------
    excess_return : OpBase
        Stock return minus the risk-free rate.
    mkt_rf, smb, hml : OpBase
        The three factor series.
    alpha, beta_mkt, beta_smb, beta_hml : OpBase
        Regression coefficients from ``_ff3_coefficients``.
    params : ResidualMomentumParameters
        Supplies the formation window, the skip and the variance floor.

    Returns
    -------
    tuple[OpBase, OpBase, OpBase]
        ``(residual_sum, residual_volatility, score)``, where ``score`` is
        ``residual_sum / residual_volatility``.
    """
    skip = params.skip_recent
    window = params.formation_window
    y = BackRef(excess_return, skip) if skip else excess_return
    x0 = BackRef(mkt_rf, skip) if skip else mkt_rf
    x1 = BackRef(smb, skip) if skip else smb
    x2 = BackRef(hml, skip) if skip else hml

    residual_sum = (
        WindowedSum(y, window)
        - alpha * float(window)
        - beta_mkt * WindowedSum(x0, window)
        - beta_smb * WindowedSum(x1, window)
        - beta_hml * WindowedSum(x2, window)
    )
    residual_variance = (
        WindowedVar(y, window)
        + beta_mkt * beta_mkt * WindowedVar(x0, window)
        + beta_smb * beta_smb * WindowedVar(x1, window)
        + beta_hml * beta_hml * WindowedVar(x2, window)
        + 2.0 * beta_mkt * beta_smb * _cov(x0, x1, window)
        + 2.0 * beta_mkt * beta_hml * _cov(x0, x2, window)
        + 2.0 * beta_smb * beta_hml * _cov(x1, x2, window)
        - 2.0 * beta_mkt * _cov(x0, y, window)
        - 2.0 * beta_smb * _cov(x1, y, window)
        - 2.0 * beta_hml * _cov(x2, y, window)
    )
    safe_variance = Select(
        residual_variance > params.variance_floor,
        residual_variance,
        ConstantOp(float(params.variance_floor)),
    )
    residual_volatility = Sqrt(safe_variance)
    return residual_sum, residual_volatility, residual_sum / residual_volatility


class ResidualMomentumFF3(FactorKunQuant):
    """Residual momentum estimated from monthly Fama-French three-factor data.

    Formula settings and optional column renames live in ``config.kwargs``
    (see ``ResidualMomentumParameters``). For a monthly panel built from
    CRSP (the University of Chicago's US stock database), whose return
    variable is ``ret``, pass ``{"return_column": "ret"}``. The four
    Fama-French series must already be variables on the same panel,
    broadcast across symbols; this class does not download or resample
    data.

    With the defaults, the signal at month ``d`` fits the regression on
    months ``d-35`` to ``d`` and sums residuals over months ``d-11`` to
    ``d-1``. It uses data up to month ``d`` only, so it can be traded from
    month ``d+1``.

    Outputs are ``resmom_raw`` (the score) and ``resmom_rank`` (its
    cross-sectional rank in ``[0, 1]`` per month), plus the diagnostics
    listed in ``_DIAGNOSTIC_FACTOR_NAMES`` when ``emit_diagnostics`` is on.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. ``data_columns`` must name exactly the
        five input variables, and ``factor_names`` may pick any subset of
        the outputs.

    Raises
    ------
    ValueError
        If ``data_columns`` does not match the five input variables, if
        ``factor_names`` names an unknown output, or if ``kwargs`` holds an
        unknown or invalid parameter.

    Examples
    --------
    ``dataset`` is a monthly panel that already carries the five input
    variables (``stock_return``, ``risk_free``, ``mkt_rf``, ``smb``,
    ``hml``) on ``(timestamp, symbol)``.

    >>> config = FactorConfig(
    ...     window=0,
    ...     dataset=dataset,
    ...     start_date=dataset.config.start_date,
    ...     end_date=dataset.config.end_date,
    ...     mode="batch",
    ...     data_columns=("stock_return", "risk_free", "mkt_rf", "smb", "hml"),
    ...     factor_names=("resmom_raw", "resmom_rank"),
    ...     file_path="resmom.zarr",
    ...     kwargs={"regression_window": 24},
    ... )
    >>> factor = ResidualMomentumFF3(config)
    >>> factor.get_factor_names()
    ('resmom_raw', 'resmom_rank')
    """

    _CORE_FACTOR_NAMES = ("resmom_raw", "resmom_rank")
    _DIAGNOSTIC_FACTOR_NAMES = (
        "residual_sum",
        "residual_volatility",
        "alpha",
        "beta_mkt",
        "beta_smb",
        "beta_hml",
        "factor_cov_determinant",
    )

    def __init__(self, factor_config: FactorConfig):
        """Initialize the factor and validate its config; see the class docstring."""
        super().__init__(factor_config)
        params = self._parameters()
        configured_columns = tuple(self.config.data_columns)
        if len(configured_columns) != len(set(configured_columns)) or set(
            configured_columns
        ) != set(params.input_columns):
            raise ValueError(
                "ResidualMomentumFF3 config.data_columns must exactly match "
                f"{params.input_columns}; got {configured_columns}"
            )
        unknown_outputs = sorted(
            set(self.get_factor_names()) - set(self._get_factor_names())
        )
        if unknown_outputs:
            raise ValueError(
                "ResidualMomentumFF3 received unknown factor_names: "
                f"{unknown_outputs}"
            )

    def _parameters(self) -> ResidualMomentumParameters:
        """Return the validated parameters for the current config."""
        return ResidualMomentumParameters.from_config(self.config)

    def _reset_dataset_config(self) -> None:
        """Move the dataset start early enough for the regression and formation windows.

        The base class moves the dataset start ``config.window`` calendar
        days before the factor's start date. This method also computes a
        start that many months back, the larger of ``regression_window`` and
        ``formation_lookback``, and keeps whichever start is earlier.
        """
        super()._reset_dataset_config()
        params = self._parameters()
        requested_start = pd.to_datetime(self.config.start_date)
        monthly_start = requested_start - pd.DateOffset(
            months=max(params.regression_window, params.formation_lookback)
        )
        current_start = pd.to_datetime(self.config.dataset.config.start_date)
        self.config.dataset.config.start_date = min(
            monthly_start, current_start
        ).strftime("%Y-%m-%d")

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the signal outputs and, when enabled, the regression diagnostics."""
        names = self._CORE_FACTOR_NAMES
        if self._parameters().emit_diagnostics:
            names += self._DIAGNOSTIC_FACTOR_NAMES
        return names

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph, emitting only the outputs in ``factor_names``."""
        params = self._parameters()
        wanted = set(self.get_factor_names())
        builder = Builder()
        with builder:
            stock_return = Input(params.return_column)
            risk_free = Input(params.risk_free_column)
            mkt_rf = Input(params.market_column)
            smb = Input(params.smb_column)
            hml = Input(params.hml_column)
            excess_return = stock_return - risk_free
            alpha, beta_mkt, beta_smb, beta_hml, determinant = _ff3_coefficients(
                excess_return, mkt_rf, smb, hml, params
            )
            residual_sum, residual_volatility, score = _formation_statistics(
                excess_return,
                mkt_rf,
                smb,
                hml,
                alpha,
                beta_mkt,
                beta_smb,
                beta_hml,
                params,
            )
            outputs = {
                "resmom_raw": score,
                "resmom_rank": Rank(score),
                "residual_sum": residual_sum,
                "residual_volatility": residual_volatility,
                "alpha": alpha,
                "beta_mkt": beta_mkt,
                "beta_smb": beta_smb,
                "beta_hml": beta_hml,
                "factor_cov_determinant": determinant,
            }
            for name, value in outputs.items():
                if name in wanted:
                    Output(value, name)
        return Function(builder.ops, name="residual_momentum_ff3")

    def _make(self):
        """Compile the graph for batch runs.

        ``no_fast_stat`` makes KunQuant recompute rolling statistics
        exactly rather than update them incrementally, which keeps the
        many variance and covariance terms numerically stable.
        ``allow_unaligned`` lets the symbol count be any number on x86; it is
        not supported on ARM, where ``cal`` pads the symbol axis instead on
        macOS (see ``FactorKunQuant._pad_symbols``).
        """
        module_name = self.__class__.__name__
        allow_unaligned = platform.machine().lower() not in {"arm64", "aarch64"}
        compiler = KunCompilerConfig(
            dtype="float",
            input_layout="TS",
            output_layout="TS",
            allow_unaligned=allow_unaligned,
            options={"no_fast_stat": True},
        )
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )

    def cal(self) -> Self:
        """Compute the factor in batch mode and store it on the data backend.

        On macOS the symbol axis is padded with all-NaN dummy symbols to a
        multiple of the SIMD block width and cut back afterwards, as every
        ``FactorKunQuant`` batch run does (see ``_pad_symbols``).

        Returns
        -------
        Self
            ``self``, for chaining.

        Examples
        --------
        On a 60-month panel of 7 symbols, the last row of ``resmom_rank``
        is the cross-sectional rank of each symbol in ``[0, 1]``.

        >>> factor.cal()  # doctest: +SKIP
        >>> out = factor.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        >>> list(out.data_vars), dict(out.sizes)  # doctest: +SKIP
        (['resmom_raw', 'resmom_rank'], {'timestamp': 60, 'symbol': 7})
        >>> out["resmom_rank"].isel(timestamp=-1).round(3).values  # doctest: +SKIP
        array([0.286, 0.143, 1.   , 0.429, 0.857, 0.571, 0.714])
        """
        input_dict, symbols, timestamps = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        num_time = next(iter(input_dict.values())).shape[0]
        num_symbols = len(symbols)
        input_dict = self._pad_symbols(input_dict, num_symbols)

        if self._lib is None:
            self._lib = self._make()
        module = self._lib.getModule(self.__class__.__name__)
        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            outputs = kr.runGraph(executor, module, input_dict, 0, num_time)
        self._lib = None
        self._to_xarray_dataset(
            self._cut_symbols(outputs, num_symbols), timestamps, symbols
        )
        return self

    def _make_stream(self):
        """Compile the graph for bar-by-bar stream runs, with exact rolling stats."""
        module_name = f"{self.__class__.__name__}_stream"
        compiler = KunCompilerConfig(
            dtype="float",
            partition_factor=8,
            input_layout="STREAM",
            output_layout="STREAM",
            options={
                "no_fast_stat": True,
                "opt_reduce": False,
                "fast_log": True,
            },
        )
        return cfake.compileit(
            [(module_name, self._get_factor_func(), compiler)],
            module_name,
            cfake.CppCompilerConfig(),
        )

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        """Return the computed panel unchanged; no post-processing is needed."""
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise ``RuntimeError``: residual momentum is a feature, not a label."""
        raise RuntimeError(f"{self.__class__.__name__} does not support get_labels()")


__all__ = ["ResidualMomentumFF3", "ResidualMomentumParameters"]
