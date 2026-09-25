"""Forward-return labels computed with the KunQuant backend.

``Return`` is the regression target (the return over the next ``n`` bars)
and ``BinaryReturn`` the classification target (1.0 when that return is
positive, else 0.0). Both read ``adjClose`` and take the horizon from
``config.kwargs["n_forward_periods"]``. The op graph computes a trailing
return, since KunQuant can only look backwards; ``get_labels()`` shifts it
forward so the value at bar ``t`` is the return from ``t`` to ``t + n``.
``get_features()`` returns the unshifted trailing return and must not be
used as a label.
"""

import KunQuant.ops as op
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant


class Return(FactorKunQuant):
    """Forward ``n``-bar return label, ``adjClose_{t+n} / adjClose_t - 1``.

    The output column is ``ret_{n}``, where ``n`` is
    ``config.kwargs["n_forward_periods"]``. Set ``data_columns`` to
    ``["adjClose"]``.

    Example:
        >>> label = Return(FactorConfig(
        ...     window=5, dataset=dataset, mode="batch",
        ...     data_columns=["adjClose"], kwargs={"n_forward_periods": 5},
        ...     file_path="ret.zarr",
        ... ))
        >>> label.cal().get_labels()   # forward 5-bar return at each t
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the label from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the graph for the trailing ``n``-bar return of ``adjClose``."""
        builder = Builder()
        factor_name = self._get_factor_names()[0]
        with builder:
            open_ = Input("adjOpen")
            return_ = op.SubConst(
                op.Div(
                    open_,
                    op.BackRef(open_, self.config.kwargs["n_forward_periods"]),
                ),
                1.0,
            )
            Output(return_, factor_name)
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("ret_{n}",)``."""
        return (f"ret_{self.config.kwargs['n_forward_periods']}",)

    def _get_labels(self, data: xr.Dataset):
        """Shift the trailing return ``n`` bars earlier so it is forward-looking.

        The last ``n`` bars become NaN.
        """
        data = data.shift(
            timestamp=-(self.config.kwargs["n_forward_periods"] + 1)
        )
        return data

    def _get_features(self, data: xr.Dataset):
        """Return the unshifted trailing return; not a label."""
        return data


class BinaryReturn(FactorKunQuant):
    """Forward ``n``-bar direction label, 1.0 for a positive return else 0.0.

    The output column is ``ret_binary_{n}``, where ``n`` is
    ``config.kwargs["n_forward_periods"]``. Set ``data_columns`` to
    ``["adjClose"]``.

    Example:
        >>> label = BinaryReturn(FactorConfig(
        ...     window=5, dataset=dataset, mode="batch",
        ...     data_columns=["adjClose"], kwargs={"n_forward_periods": 5},
        ...     file_path="ret_binary.zarr",
        ... ))
        >>> label.cal().get_labels()   # 1.0 where the next 5 bars are up
    """

    def __init__(self, factor_config: FactorConfig):
        """Create the label from a KunQuant factor config."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the graph: trailing ``n``-bar return, thresholded at zero."""
        builder = Builder()
        factor_name = self._get_factor_names()[0]
        with builder:
            open_ = Input("adjOpen")
            return_ = op.SubConst(
                op.Div(
                    open_,
                    op.BackRef(open_, self.config.kwargs["n_forward_periods"]),
                ),
                1.0,
            )
            binary = op.Select(
                return_ > 0, op.ConstantOp(1.0), op.ConstantOp(0.0)
            )
            Output(binary, factor_name)
        return Function(builder.ops)

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("ret_binary_{n}",)``."""
        return (f"ret_binary_{self.config.kwargs['n_forward_periods']}",)

    def _get_labels(self, data: xr.Dataset):
        """Shift the trailing indicator ``n`` bars earlier so it is forward-looking.

        The last ``n`` bars become NaN.
        """
        data = data.shift(timestamp=-(self.config.kwargs["n_forward_periods"] + 1))
        return data

    def _get_features(self, data: xr.Dataset):
        """Return the unshifted trailing indicator; not a label."""
        return data
