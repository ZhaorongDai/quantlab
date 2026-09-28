"""Forward-return labels over the adjusted open, computed with KunQuant.

A *label* is the value a model learns to predict. Here it is the return a
position opened after bar ``t`` would earn over the next ``n`` bars.
``Return`` is the regression target (that return itself) and
``BinaryReturn`` the classification target (1.0 when the return is
positive, else 0.0). Both read ``adjOpen``, the split- and dividend-adjusted
open price, and take the horizon ``n`` from
``config.kwargs["n_forward_periods"]``.

KunQuant, the library that computes the graph, compiles formulas to native
code and can only look backwards in time. Each label therefore wraps a
private factor computing the trailing return ``adjOpen[t] / adjOpen[t - n] -
1`` in a ``Forward`` label with ``span = n`` and ``delay = 1``, which shifts
it ``n + 1`` bars earlier. The label at ``t`` is then ``adjOpen[t + n + 1] /
adjOpen[t + 1] - 1``: a position entered at the next bar's open, because a
signal formed at bar ``t`` cannot trade before then.
"""

import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig, ForwardConfig
from quantlab.base.factor import FactorKunQuant
from quantlab.label.forward import Forward


def _trailing_return(factor: FactorKunQuant):
    """Return the KunQuant op of the trailing ``n``-bar return of ``adjOpen``."""
    open_ = Input("adjOpen")
    return op.SubConst(
        op.Div(open_, op.BackRef(open_, factor.config.kwargs["n_forward_periods"])),
        1.0,
    )


class _TrailingOpenReturn(FactorKunQuant):
    """The trailing ``n``-bar open-to-open return, ``Return`` shifts forward."""

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("ret_{n}",)``."""
        return (f"ret_{self.config.kwargs['n_forward_periods']}",)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph of the trailing return."""
        builder = Builder()
        with builder:
            Output(_trailing_return(self), self._get_factor_names()[0])
        return Function(builder.ops)


class _TrailingOpenDirection(FactorKunQuant):
    """1.0 where the trailing ``n``-bar open-to-open return is positive, else 0.0."""

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("ret_binary_{n}",)``."""
        return (f"ret_binary_{self.config.kwargs['n_forward_periods']}",)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph of the trailing return's sign."""
        builder = Builder()
        with builder:
            binary = op.Select(
                _trailing_return(self) > 0, op.ConstantOp(1.0), op.ConstantOp(0.0)
            )
            Output(binary, self._get_factor_names()[0])
        return Function(builder.ops)


class _OpenToOpenLabel(Forward):
    """A ``Forward`` label over a trailing open-to-open factor built from a ``FactorConfig``.

    The ``FactorConfig`` is the whole description: ``get_config()`` returns it
    with this class's import path as ``name``, so
    ``quantlab.utils.module.load_factor_from_config`` rebuilds the label from
    it like any factor.
    """

    #: Config class the label is built from and rebuilt with.
    config_cls = FactorConfig
    #: The private trailing factor this label shifts forward.
    _trailing: type[FactorKunQuant]

    def __init__(self, factor_config: FactorConfig):
        """Initialize the label; see the class docstring for parameters."""
        factor = self._trailing(factor_config)
        super().__init__(
            ForwardConfig(
                factor=factor,
                span=factor.config.kwargs["n_forward_periods"],
                delay=1,
            )
        )

    def get_config(self) -> dict:
        """Return the ``FactorConfig`` dict the label was built from.

        Examples
        --------
        >>> cfg = label.get_config()
        >>> cfg["name"], cfg["kwargs"]
        ('quantlab.label.fret.Return', {'n_forward_periods': 5})
        """
        return {**self.config.factor.get_config(), "name": self.import_path}


class Return(_OpenToOpenLabel):
    """Forward n-bar open-to-open return label.

    At signal timestamp ``t`` the label is
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1``: the position is entered at
    the next bar's adjusted open and exited ``n`` bars later at the adjusted
    open. It is a ``Forward`` label with ``span = n`` and ``delay = 1``, so
    ``lookahead_bars()`` is ``n + 1``. Only timestamps without ``n + 1``
    later bars in the dataset are NaN.

    The output column is ``ret_{n}``.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config of the trailing return. Set
        ``data_columns`` to ``["adjOpen"]`` and
        ``kwargs["n_forward_periods"]`` to the horizon ``n``; ``file_path``
        is where ``build`` writes the trailing return.

    Examples
    --------
    >>> label = Return(FactorConfig(
    ...     warmup_bars=5, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen"], kwargs={"n_forward_periods": 5},
    ...     file_path="ret.zarr",
    ... ))
    >>> label.lookahead_bars(), label.span_bars(), label.get_factor_names()
    (6, 5, ('ret_5',))
    """

    _trailing = _TrailingOpenReturn


class BinaryReturn(_OpenToOpenLabel):
    """Forward n-bar open-to-open direction label.

    At signal timestamp ``t`` the label is 1.0 when
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1 > 0`` and 0.0 otherwise: the
    position is entered at the next bar's adjusted open and judged ``n`` bars
    later at the adjusted open. It is a ``Forward`` label with ``span = n``
    and ``delay = 1``. Only timestamps without ``n + 1`` later bars in the
    dataset are NaN.

    The output column is ``ret_binary_{n}``.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config of the trailing indicator. Set
        ``data_columns`` to ``["adjOpen"]`` and
        ``kwargs["n_forward_periods"]`` to the horizon ``n``.

    Examples
    --------
    >>> label = BinaryReturn(FactorConfig(
    ...     warmup_bars=5, dataset=dataset, mode="batch",
    ...     data_columns=["adjOpen"], kwargs={"n_forward_periods": 5},
    ...     file_path="ret_binary_open.zarr",
    ... ))
    >>> label.get_factor_names()
    ('ret_binary_5',)
    """

    _trailing = _TrailingOpenDirection
