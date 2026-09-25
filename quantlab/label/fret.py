"""Forward-return labels computed with the KunQuant backend.

A *label* is the value a model learns to predict. Here it is the return a
position opened after bar ``t`` would earn over the next ``n`` bars.
``Return`` is the regression target (that return itself) and
``BinaryReturn`` the classification target (1.0 when the return is
positive, else 0.0). Both read ``adjOpen``, the split- and dividend-adjusted
open price, and take the horizon ``n`` from
``config.kwargs["n_forward_periods"]``.

KunQuant, the library that computes the graph, compiles formulas to native
code and can only look backwards in time. The graph therefore computes the
trailing return ``adjOpen[t] / adjOpen[t - n] - 1``, and ``get_labels()``
shifts it ``n + 1`` bars earlier. The label at ``t`` is then
``adjOpen[t + n + 1] / adjOpen[t + 1] - 1``: a position entered at the next
bar's open, because a signal formed at the close of bar ``t`` cannot trade
before then. ``get_features()`` returns the unshifted trailing return, which
is not a label and must not be used as one.
"""

import KunQuant.ops as op
import xarray as xr
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig
from quantlab.base.factor import FactorKunQuant


class Return(FactorKunQuant):
    """Forward n-bar open-to-open return label.

    At signal timestamp ``t`` the label is
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1``: the position is entered at
    the next bar's adjusted open and exited ``n`` bars later at the adjusted
    open. The last ``n + 1`` timestamps have no future prices, so their
    labels are NaN.

    The output column is ``ret_{n}``.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. Set ``data_columns`` to ``["adjOpen"]``
        and ``kwargs["n_forward_periods"]`` to the horizon ``n``.

    Examples
    --------
    >>> label = Return(FactorConfig(
    ...     window=5, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen"], kwargs={"n_forward_periods": 5},
    ...     file_path="ret.zarr",
    ... ))
    >>> labels = label.cal().get_labels()  # forward 5-bar return at each t
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the label; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph for the trailing ``n``-bar return of ``adjOpen``."""
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
        """Shift the trailing return ``n + 1`` bars earlier so it looks forward.

        The last ``n + 1`` timestamps become NaN.
        """
        data = data.shift(
            timestamp=-(self.config.kwargs["n_forward_periods"] + 1)
        )
        return data

    def _get_features(self, data: xr.Dataset):
        """Return the unshifted trailing return, which is not a valid label."""
        return data


class BinaryReturn(FactorKunQuant):
    """Forward n-bar open-to-open direction label.

    At signal timestamp ``t`` the label is 1.0 when
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1 > 0`` and 0.0 otherwise: the
    position is entered at the next bar's adjusted open and judged ``n`` bars
    later at the adjusted open. The last ``n + 1`` timestamps have no future
    prices, so their labels are NaN.

    The output column is ``ret_binary_{n}``.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config. Set ``data_columns`` to ``["adjOpen"]``
        and ``kwargs["n_forward_periods"]`` to the horizon ``n``.

    Examples
    --------
    >>> label = BinaryReturn(FactorConfig(
    ...     window=5,
    ...     dataset=dataset,
    ...     mode="batch",
    ...     data_columns=["adjOpen"],
    ...     kwargs={"n_forward_periods": 5},
    ...     file_path="ret_binary_open.zarr",
    ... ))
    >>> labels = label.cal().get_labels()  # 1.0 where the 5-bar return > 0
    """

    def __init__(self, factor_config: FactorConfig):
        """Initialize the label; see the class docstring for parameters."""
        super().__init__(factor_config)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph: 1.0 where the trailing return is positive."""
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
        """Shift the trailing indicator ``n + 1`` bars earlier so it looks forward.

        The last ``n + 1`` timestamps become NaN.
        """
        data = data.shift(timestamp=-(self.config.kwargs["n_forward_periods"] + 1))
        return data

    def _get_features(self, data: xr.Dataset):
        """Return the unshifted trailing indicator, which is not a valid label."""
        return data
