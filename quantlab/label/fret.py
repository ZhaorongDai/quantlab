import KunQuant.ops as op
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant


class Return(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        builder = Builder()
        factor_name = self._get_factor_names()[0]
        with builder:
            close = Input("close")
            return_ = op.SubConst(
                op.Div(
                    close,
                    op.BackRef(close, self.config.kwargs["n_forward_periods"]),
                ),
                1.0,
            )
            Output(return_, factor_name)
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        return (f"ret_{self.config.kwargs['n_forward_periods']}",)

    def _get_labels(self, data: xr.Dataset):
        data = data.shift(timestamp=-self.config.kwargs["n_forward_periods"])
        return data

    def _get_features(self, data: xr.Dataset):
        return data


class BinaryReturn(FactorKunQuant):
    def __init__(self, factor_config: FactorConfig):
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        builder = Builder()
        factor_name = self._get_factor_names()[0]
        with builder:
            close = Input("close")
            return_ = op.SubConst(
                op.Div(
                    close,
                    op.BackRef(close, self.config.kwargs["n_forward_periods"]),
                ),
                1.0,
            )
            binary = op.Select(
                return_ > 0, op.ConstantOp(1.0), op.ConstantOp(0.0)
            )
            Output(binary, factor_name)
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        return (f"ret_binary_{self.config.kwargs['n_forward_periods']}",)

    def _get_labels(self, data: xr.Dataset):
        data = data.shift(timestamp=-self.config.kwargs["n_forward_periods"])
        return data

    def _get_features(self, data: xr.Dataset):
        return data
