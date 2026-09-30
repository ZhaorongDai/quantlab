"""Forward-return labels over the adjusted open, computed with KunQuant.

A *label* is the value a model learns to predict. Here it is the return a
position opened after bar ``t`` would earn over the next ``n`` bars.
``Return`` is the regression target (that return itself) and
``BinaryReturn`` the classification target (1.0 when the return is
positive, 0.0 when it is not, NaN when it is missing). Both read
``adjOpen``, the split- and dividend-adjusted open price, and take the
horizon ``n`` from ``config.kwargs["n_forward_periods"]``.

KunQuant, the library that computes the graph, compiles formulas to native
code and can only look backwards in time. Each label therefore wraps a
private factor computing the trailing return ``adjOpen[t] / adjOpen[t - n] -
1`` in a ``Forward`` label with ``span = n`` and ``delay = 1``, which shifts
it ``n + 1`` bars earlier. The label at ``t`` is then ``adjOpen[t + n + 1] /
adjOpen[t + 1] - 1``: a position entered at the next bar's open, because a
signal formed at bar ``t`` cannot trade before then.

``Volatility`` follows the same conventions over the same opens: the
sample standard deviation of the one-bar open-to-open returns inside the
``n``-bar window, times ``sqrt(n)``, so a ``Return`` and a ``Volatility``
of the same ``n`` describe the same holding period.
"""

import math

import KunQuant.ops as op
from KunQuant.Op import Builder, Input, Output
from KunQuant.Stage import Function

from quantlab.base.config import FactorConfig, ForwardConfig
from quantlab.factor.kunquant import FactorKunQuant
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
    """The sign of the trailing ``n``-bar open-to-open return.

    1.0 where the return is positive, 0.0 where it is zero or negative, and
    NaN where it is NaN. ``Equals(ret, ret)`` is KunQuant's not-NaN test.
    """

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("ret_binary_{n}",)``."""
        return (f"ret_binary_{self.config.kwargs['n_forward_periods']}",)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph of the trailing return's sign."""
        builder = Builder()
        with builder:
            ret = _trailing_return(self)
            binary = op.Select(
                op.Equals(ret, ret),
                op.Select(ret > 0, op.ConstantOp(1.0), op.ConstantOp(0.0)),
                op.ConstantOp("nan"),
            )
            Output(binary, self._get_factor_names()[0])
        return Function(builder.ops)


class _TrailingOpenVolatility(FactorKunQuant):
    """The trailing ``n``-bar open-to-open volatility, ``Volatility`` shifts forward.

    The sample standard deviation (``ddof=1``) of the last ``n`` one-bar
    returns ``adjOpen[k] / adjOpen[k - 1] - 1``, times ``sqrt(n)``. NaN when
    any of the ``n + 1`` opens is missing.
    """

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return ``("vol_{n}",)``."""
        return (f"vol_{self.config.kwargs['n_forward_periods']}",)

    def _get_factor_func(self) -> Function:
        """Build the KunQuant graph of the trailing volatility."""
        n = self.config.kwargs["n_forward_periods"]
        builder = Builder()
        with builder:
            open_ = Input("adjOpen")
            one_bar = op.SubConst(op.Div(open_, op.BackRef(open_, 1)), 1.0)
            volatility = op.MulConst(op.WindowedStddev(one_bar, n), math.sqrt(n))
            Output(volatility, self._get_factor_names()[0])
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
        ('quantlab.label.predefined.fret.Return', {'n_forward_periods': 5})
        """
        return {**self.config.factor.get_config(), "name": self.import_path}


class Return(_OpenToOpenLabel):
    """Forward n-bar open-to-open return label.

    At signal timestamp ``t`` the label is
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1``: the position is entered at
    the next bar's adjusted open and exited ``n`` bars later at the adjusted
    open. It is a ``Forward`` label with ``span = n`` and ``delay = 1``, so
    ``lookahead_bars()`` is ``n + 1``. The label is NaN where either open is
    missing and at timestamps without ``n + 1`` later bars in the dataset.

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
    ``adjOpen[t + n + 1] / adjOpen[t + 1] - 1 > 0``, 0.0 when that return is
    zero or negative, and NaN when it is NaN: the position is entered at the
    next bar's adjusted open and judged ``n`` bars later at the adjusted
    open. The label is therefore NaN exactly where ``Return`` of the same
    horizon is NaN: where either open is missing (a gap in the prices, a
    symbol not yet listed or already delisted) and at timestamps without
    ``n + 1`` later bars in the dataset. It is a ``Forward`` label with
    ``span = n`` and ``delay = 1``.

    The output column is ``ret_binary_{n}``.

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config of the trailing indicator. Set
        ``data_columns`` to ``["adjOpen"]`` and
        ``kwargs["n_forward_periods"]`` to the horizon ``n``.

    Examples
    --------
    One symbol whose open on 3 January is missing. The labels on 1 and 2
    January read that open and are NaN, like the last two, which have no
    two later bars.

    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.backend import XrBackend
    >>> from quantlab.base.config import DatasetConfig, FactorConfig
    >>> from quantlab.dataset.stock import StockDataset
    >>> from quantlab.label.predefined.fret import BinaryReturn, Return
    >>> opens = np.array([10.0, 11.0, np.nan, 12.0, 11.0, 13.0, 12.0, 14.0])
    >>> XrBackend().to_internal(xr.Dataset(
    ...     {"adjOpen": (["timestamp", "symbol"], opens[:, None])},
    ...     coords={"timestamp": pd.date_range("2024-01-01", periods=8), "symbol": ["A"]},
    ... )).write("data/stock.zarr")
    XrBackend()
    >>> dataset = StockDataset(DatasetConfig(
    ...     raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
    ...     market="us_equity", frequency="1d",
    ... ))
    >>> config = dict(
    ...     warmup_bars=1, dataset=dataset, mode="batch", data_columns=["adjOpen"],
    ...     kwargs={"n_forward_periods": 1}, njobs=1,
    ... )
    >>> ret = Return(FactorConfig(file_path="data/ret.zarr", **config))
    >>> up = BinaryReturn(FactorConfig(file_path="data/up.zarr", **config))
    >>> up.get_factor_names()
    ('ret_binary_1',)
    >>> ret.compute("2024-01-01", "2024-01-08")["ret_1"].values[:, 0].round(3)
    array([   nan,    nan, -0.083,  0.182, -0.077,  0.167,    nan,    nan],
          dtype=float32)
    >>> up.compute("2024-01-01", "2024-01-08")["ret_binary_1"].values[:, 0]
    array([nan, nan,  0.,  1.,  0.,  1., nan, nan], dtype=float32)
    """

    _trailing = _TrailingOpenDirection


class Volatility(_OpenToOpenLabel):
    """Forward n-bar open-to-open volatility label.

    At signal timestamp ``t`` the label is the sample standard deviation
    (``ddof=1``) of the one-bar returns ``adjOpen[k] / adjOpen[k - 1] - 1``
    for ``k`` in ``t + 2 .. t + n + 1``, times ``sqrt(n)``: the volatility
    over the ``n`` bars a position entered at the next bar's adjusted open
    is held, on the same span scale as ``Return`` of the same ``n``. Like
    ``Return`` it is a ``Forward`` label with ``span = n`` and ``delay = 1``,
    so ``lookahead_bars()`` is ``n + 1``. It is NaN where any open in the
    window is missing and at timestamps without ``n + 1`` later bars.

    The output column is ``vol_{n}``. Its ``kind`` is ``"volatility"``, so
    a model predicting it on the label's own scale also reports the level
    metrics ``qlike`` and ``variance_ratio`` (see
    ``quantlab.utils.metrics.volatility_level_metrics``).

    Parameters
    ----------
    factor_config : FactorConfig
        The KunQuant factor config of the trailing volatility. Set
        ``data_columns`` to ``["adjOpen"]`` and
        ``kwargs["n_forward_periods"]`` to the horizon ``n``, at least 2
        (one return has no sample standard deviation).

    Raises
    ------
    ValueError
        If ``n_forward_periods`` is below 2.

    Examples
    --------
    One symbol whose opens alternate between two one-bar returns. Every
    window of two returns then holds one of each.

    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.backend import XrBackend
    >>> from quantlab.base.config import DatasetConfig, FactorConfig
    >>> from quantlab.dataset.stock import StockDataset
    >>> from quantlab.label.predefined.fret import Volatility
    >>> opens = 10.0 * np.cumprod([1.0, 1.1, 0.9, 1.1, 0.9, 1.1, 0.9, 1.1])
    >>> XrBackend().to_internal(xr.Dataset(
    ...     {"adjOpen": (["timestamp", "symbol"], opens[:, None])},
    ...     coords={"timestamp": pd.date_range("2024-01-01", periods=8), "symbol": ["A"]},
    ... )).write("data/stock.zarr")
    XrBackend()
    >>> dataset = StockDataset(DatasetConfig(
    ...     raw_data_dir_path="data/raw", zarr_file_path="data/stock.zarr",
    ...     market="us_equity", frequency="1d",
    ... ))
    >>> vol = Volatility(FactorConfig(
    ...     warmup_bars=3, dataset=dataset, mode="batch", data_columns=["adjOpen"],
    ...     kwargs={"n_forward_periods": 2}, njobs=1, file_path="data/vol.zarr",
    ... ))
    >>> vol.lookahead_bars(), vol.span_bars(), vol.get_factor_names()
    (3, 2, ('vol_2',))
    >>> vol.compute("2024-01-01", "2024-01-08")["vol_2"].values[:, 0].round(4)
    array([0.2, 0.2, 0.2, 0.2, 0.2, nan, nan, nan], dtype=float32)
    """

    _trailing = _TrailingOpenVolatility
    kind = "volatility"

    def __init__(self, factor_config: FactorConfig):
        """Initialize the label; see the class docstring for parameters."""
        if factor_config.kwargs.get("n_forward_periods", 0) < 2:
            raise ValueError(
                "Volatility needs kwargs['n_forward_periods'] >= 2, got "
                f"{factor_config.kwargs.get('n_forward_periods')!r}"
            )
        super().__init__(factor_config)
