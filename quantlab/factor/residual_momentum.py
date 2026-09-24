"""Fama--French three-factor residual momentum for KunQuant.

The input panel is expected to contain one row per month.  ``mkt_rf``,
``smb``, ``hml`` and ``risk_free`` are common time series, but KunQuant needs
them broadcast over the panel's ``symbol`` dimension before they reach this
factor.  Returns are arithmetic decimal returns, not percentages.
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
    """Formula parameters read from ``FactorConfig.kwargs``."""

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
        """Build parameters from the recognized entries in ``config.kwargs``."""
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
        """Number of observations left after skipping the latest months."""
        return self.formation_lookback - self.skip_recent

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Dataset variables consumed by the operator graph."""
        return (
            self.return_column,
            self.risk_free_column,
            self.market_column,
            self.smb_column,
            self.hml_column,
        )

    def validate(self) -> None:
        """Reject windows, safeguards and input mappings that cannot work."""
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
    """Call KunQuant covariance using its ``(x, window, y)`` signature."""
    return WindowedCovariance(x, window, y)


def _signed_floor(value: OpBase, floor: float) -> OpBase:
    """Keep a divisor away from zero without changing a non-zero sign."""
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
    """Return rolling alpha, three slopes and the factor covariance determinant."""
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
    """Return formation-period residual sum, volatility and standardized score."""
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
    """Residual momentum estimated from monthly Fama--French three-factor data.

    Formula settings and optional column aliases live in ``config.kwargs``.
    For a CRSP-derived monthly panel, use ``{"return_column": "ret"}``.
    The four Fama--French series must already be variables on that same panel,
    broadcast across symbols; this class does not download or resample data.

    The signal at month ``d`` estimates FF3 on ``d-35:d`` by default and uses
    residuals from ``d-11:d-1``.  It is therefore tradable from month ``d+1``.
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
        """Validate the formula contract, then initialize ``FactorKunQuant``."""
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
        """Return validated parameters represented by the current config."""
        return ResidualMomentumParameters.from_config(self.config)

    def _reset_dataset_config(self) -> None:
        """Warm up by formula months as well as ``config.window`` calendar days."""
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
        """Return signal outputs and, when enabled, regression diagnostics."""
        names = self._CORE_FACTOR_NAMES
        if self._parameters().emit_diagnostics:
            names += self._DIAGNOSTIC_FACTOR_NAMES
        return names

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph, pruning outputs not requested in config."""
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
        """Compile batch mode with stable rolling statistics and ragged symbols."""
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
        """Calculate in batch, padding the symbol axis when ARM SIMD requires it.

        KunQuant cannot compile ``allow_unaligned=True`` on ARM.  Its float
        kernels use four-symbol blocks there, so a temporary all-NaN tail is
        added for panels whose width is not divisible by four.  NaN dummy
        symbols do not enter the cross-sectional rank, and outputs are sliced
        back to the real symbol axis before they reach the factor backend.
        """
        input_dict, symbols, timestamps = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        num_time = next(iter(input_dict.values())).shape[0]
        num_symbols = len(symbols)
        if platform.machine().lower() in {"arm64", "aarch64"}:
            padding = (-num_symbols) % 4
            if padding:
                input_dict = {
                    name: np.pad(
                        values,
                        ((0, 0), (0, padding)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                    for name, values in input_dict.items()
                }

        if self._lib is None:
            self._lib = self._make()
        module = self._lib.getModule(self.__class__.__name__)
        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            outputs = kr.runGraph(executor, module, input_dict, 0, num_time)
        self._lib = None
        outputs = {
            name: values[:, :num_symbols] for name, values in outputs.items()
        }
        self._to_xarray_dataset(outputs, timestamps, symbols)
        return self

    def _make_stream(self):
        """Compile stream mode with the same stable-statistics requirement."""
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
        """Return the computed factor panel unchanged."""
        return data

    def _get_labels(self, data: xr.Dataset) -> NoReturn:
        """Raise because residual momentum is a feature, not a label."""
        raise RuntimeError(f"{self.__class__.__name__} does not support get_labels()")


__all__ = ["ResidualMomentumFF3", "ResidualMomentumParameters"]
