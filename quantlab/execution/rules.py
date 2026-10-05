"""The Execution rules: what the market does with a bar's orders, as a public module.

A cross-sectional backtest turns target weights into orders and executes them
bar by bar. This module states how. A weight decided at bar t fills at bar
t + ``FILL_DELAY_BARS`` at that bar's fill price, the last known one when the
bar has none. An order whose raw fill price, or whose sizing price, is missing
is rejected and its holding kept, and a NaN weight sends no order at all. A
holding delisted on bar b is settled on b + 1 at its last valuation whatever
its weight asks, with no fee or slippage, and for nothing at a last valuation
of 0.

Each weight is sized as ``weight * book / price - shares``, the book valued at
the prices of the sizing basis: the order prices (``"fill"``) or the signal
bar's valuation prices (``"valuation"``). A bar's orders run in ascending
order of value, so sells come before buys, and a buy whose cost and fee exceed
the cash left is cut to what the cash pays for. Slippage moves the fill price
against the trade and the fee is a fraction of the traded value.

These are vectorbt 1.1's ``Portfolio.from_orders`` with target-percent sizes,
``direction="both"``, shared cash and ``call_seq="auto"``, down to its
closeness tolerances, so ``replay`` holds what the vectorbt engine executes to
floating-point rounding. The engine plans its orders with ``plan_orders``.
``ExecutionBook`` keeps the same book one bar at a time, for a driver that
learns each rebalance's weights only once it has decided them.

The module imports only numpy: no quantlab layer and no simulation engine.

Examples
--------
Half the book in AAA, decided on bar 0, fills on bar 1 at 10:

>>> import numpy as np
>>> weights = np.array([[0.5], [np.nan], [np.nan]])
>>> prices = np.array([[8.0], [10.0], [11.0]])
>>> result = replay(weights, prices, prices, np.zeros((3, 1), dtype=bool))
>>> result.shares[:, 0].tolist(), result.cash.tolist()
([0.0, 0.05, 0.05], [1.0, 0.5, 0.5])
"""

import math
from dataclasses import dataclass

import numpy as np

#: The bars between the bar a weight is decided on and the bar it fills on.
FILL_DELAY_BARS = 1

_SIZING_BASES = ("fill", "valuation")

#: vectorbt's tolerances for "the same quantity" (``vectorbt.utils.math_``).
_REL_TOL, _ABS_TOL = 1e-9, 1e-12


@dataclass(frozen=True)
class ExecutionSettings:
    """How orders are sized and charged.

    Parameters
    ----------
    sizing_basis : {"fill", "valuation"}
        The prices a weight is sized against: the fill bar's order prices,
        or the signal bar's valuation prices.
    fees : float
        The fee, a fraction of each order's traded value.
    slippage : float
        The fraction a fill price moves against the trade.

    Raises
    ------
    ValueError
        If ``sizing_basis`` is neither ``"fill"`` nor ``"valuation"``.

    Examples
    --------
    >>> ExecutionSettings()
    ExecutionSettings(sizing_basis='fill', fees=0.0, slippage=0.0)
    >>> ExecutionSettings(sizing_basis="close")
    Traceback (most recent call last):
    ...
    ValueError: sizing_basis must be one of ('fill', 'valuation'), got 'close'
    """

    sizing_basis: str = "fill"
    fees: float = 0.0
    slippage: float = 0.0

    def __post_init__(self):
        if self.sizing_basis not in _SIZING_BASES:
            raise ValueError(
                f"sizing_basis must be one of {_SIZING_BASES}, got {self.sizing_basis!r}"
            )


@dataclass(frozen=True)
class OrderPlan:
    """The orders the market accepts, one row per fill bar.

    Every attribute has the shape of the weights it was planned from:
    ``[T, S]`` for a panel, ``[S]`` for one bar.

    Attributes
    ----------
    target : np.ndarray
        The weight each bar fills to, as asked; NaN where none is asked.
    size : np.ndarray
        The target the market accepts: NaN where an order is rejected or
        none is sent, 0.0 where a holding is settled.
    price : np.ndarray
        The order price: the fill price, last known one included, or the
        last valuation where a holding is settled.
    sizing : np.ndarray
        The price a target is sized against under the sizing basis.
    fees, slippage : np.ndarray
        The fee and slippage fractions of each order: the settings', or 0.0
        on a settlement.
    rejected : np.ndarray
        Booleans: a weight asked without a raw fill price or a sizing price.
    settle : np.ndarray
        Booleans: the bar after a symbol's delisting.
    """

    target: np.ndarray
    size: np.ndarray
    price: np.ndarray
    sizing: np.ndarray
    fees: np.ndarray
    slippage: np.ndarray
    rejected: np.ndarray
    settle: np.ndarray


@dataclass(frozen=True)
class BarExecution:
    """What happened to one bar's orders, as booleans on ``[S]``.

    Attributes
    ----------
    rejected : np.ndarray
        An order was rejected where it would have traded: its target is not
        0 or the symbol was held.
    settled : np.ndarray
        A held symbol was settled.
    """

    rejected: np.ndarray
    settled: np.ndarray


@dataclass(frozen=True)
class ExecutionReplay:
    """The holdings after each bar's orders, and what happened to them.

    Attributes
    ----------
    shares : np.ndarray
        Signed shares of each symbol after each bar, on ``[T, S]``.
    cash : np.ndarray
        Cash after each bar, on ``[T]``.
    rejected, settled : np.ndarray
        Booleans on ``[T, S]``, as ``BarExecution`` reports them per bar.
    """

    shares: np.ndarray
    cash: np.ndarray
    rejected: np.ndarray
    settled: np.ndarray


@dataclass(frozen=True)
class _Prices:
    """The prices of a panel as the rules read them, every array on ``[T, S]``.

    ``raw_fill`` is the fill price as given; ``fill`` and ``valuation`` are
    forward-filled; ``sizing`` is the price of the sizing basis on each fill
    bar; ``settle`` marks the bar after each delisting.
    """

    raw_fill: np.ndarray
    fill: np.ndarray
    valuation: np.ndarray
    sizing: np.ndarray
    settle: np.ndarray

    @classmethod
    def build(cls, fill, valuation, delisted, sizing_basis: str) -> "_Prices":
        """Return the prices of a panel from its fill and valuation prices and delistings."""
        raw_fill = np.asarray(fill, dtype=np.float64)
        filled = _ffill(raw_fill)
        valued = _ffill(np.asarray(valuation, dtype=np.float64))
        # The valuation basis sizes at the signal bar's valuation price:
        # vectorbt's previous `close`.
        sizing = _shift(valued) if sizing_basis == "valuation" else filled
        settle = _shift(np.asarray(delisted, dtype=bool), fill_value=False)
        return cls(raw_fill, filled, valued, sizing, settle)

    def plan(self, target: np.ndarray, settings: ExecutionSettings, rows=slice(None)) -> OrderPlan:
        """Return the plan of ``target`` on the fill bars ``rows`` (all, or one bar)."""
        settle = self.settle[rows]
        sizing = self.sizing[rows]
        rejected = np.isfinite(target) & (np.isnan(self.raw_fill[rows]) | np.isnan(sizing)) & ~settle
        return OrderPlan(
            target=target,
            size=np.where(settle, 0.0, np.where(rejected, np.nan, target)),
            price=np.where(settle, self.valuation[rows], self.fill[rows]),
            sizing=sizing,
            fees=np.where(settle, 0.0, settings.fees),
            slippage=np.where(settle, 0.0, settings.slippage),
            rejected=rejected,
            settle=settle,
        )


def plan_orders(
    weights, fill, valuation, delisted, settings: ExecutionSettings = ExecutionSettings()
) -> OrderPlan:
    """Return the orders the market accepts for ``weights`` decided on each bar.

    Parameters
    ----------
    weights : array-like
        Target weights on ``[T, S]``, row t decided at bar t; NaN keeps a
        holding.
    fill, valuation : array-like
        The fill and valuation prices on ``[T, S]`` as given, NaN where a bar
        has none; both are forward-filled here.
    delisted : array-like
        Booleans on ``[T, S]``: a symbol's last bar before it delists.
    settings : ExecutionSettings
        The sizing basis, fees and slippage.

    Returns
    -------
    OrderPlan
        The plan of every fill bar.

    Examples
    --------
    AAA has no fill price on bar 2, so bar 1's exit is rejected there:

    >>> import numpy as np
    >>> plan = plan_orders(
    ...     np.array([[1.0], [0.0], [np.nan]]),
    ...     np.array([[10.0], [10.0], [np.nan]]),
    ...     np.array([[10.0], [10.0], [10.0]]),
    ...     np.zeros((3, 1), dtype=bool),
    ... )
    >>> plan.target[:, 0].tolist(), plan.size[:, 0].tolist(), plan.rejected[:, 0].tolist()
    ([nan, 1.0, 0.0], [nan, 1.0, nan], [False, False, True])
    """
    prices = _Prices.build(fill, valuation, delisted, settings.sizing_basis)
    return prices.plan(_shift(np.asarray(weights, dtype=np.float64)), settings)


class ExecutionBook:
    """Shares and cash, traded one bar at a time under the Execution rules.

    A driver ``submit``s the weights it decides on a bar and ``trade``s the
    bars in time order, reading ``shares`` and ``cash`` as it goes. Stepping
    every bar holds exactly what ``replay`` returns for the same weights.

    Parameters
    ----------
    fill, valuation, delisted : array-like
        As for ``plan_orders``, on ``[T, S]``.
    settings : ExecutionSettings
        The sizing basis, fees and slippage.
    init_cash : float
        The starting cash; the book starts with no shares.

    Attributes
    ----------
    settings : ExecutionSettings
        As given.
    shares : np.ndarray
        Signed shares of each symbol after the last bar traded.
    cash : float
        Cash after the last bar traded.

    Examples
    --------
    >>> import numpy as np
    >>> prices = np.array([[10.0, 20.0], [10.0, 20.0], [12.0, 18.0]])
    >>> book = ExecutionBook(prices, prices, np.zeros((3, 2), dtype=bool))
    >>> book.submit(0, [0.5, -0.5])
    >>> book.trade(1).rejected.tolist()
    [False, False]
    >>> book.shares.tolist(), book.cash
    ([0.05, -0.025], 1.0)
    """

    def __init__(
        self,
        fill,
        valuation,
        delisted,
        settings: ExecutionSettings = ExecutionSettings(),
        init_cash: float = 1.0,
    ):
        self.settings = settings
        self._prices = _Prices.build(fill, valuation, delisted, settings.sizing_basis)
        self.shares = np.zeros(self._prices.fill.shape[1])
        self.cash = float(init_cash)
        self._queued: dict[int, np.ndarray] = {}

    def submit(self, row: int, weights) -> None:
        """Queue the weights decided at ``row``; they fill ``FILL_DELAY_BARS`` later.

        Weights submitted again for the same row replace the earlier ones.
        Weights whose fill bar is past the panel never fill.

        Parameters
        ----------
        row : int
            The bar the weights are decided on.
        weights : array-like
            Target weights on ``[S]``; NaN keeps a holding.

        Examples
        --------
        >>> import numpy as np
        >>> prices = np.full((2, 1), 10.0)
        >>> book = ExecutionBook(prices, prices, np.zeros((2, 1), dtype=bool))
        >>> book.submit(0, [1.0])
        >>> book.trade(0).settled.tolist(), book.shares.tolist()
        ([False], [0.0])
        >>> book.trade(1).settled.tolist(), book.shares.tolist()
        ([False], [0.1])
        """
        self._queued[row + FILL_DELAY_BARS] = np.asarray(weights, dtype=np.float64)

    def trade(self, bar: int) -> BarExecution:
        """Run ``bar``'s orders: the weights that fill there and the settlements.

        Bars must be traded in time order, each once.

        Parameters
        ----------
        bar : int
            The bar traded.

        Returns
        -------
        BarExecution
            The orders rejected and the holdings settled on the bar.

        Raises
        ------
        ValueError
            If an order meets a price of 0 or less on a symbol that is not
            being settled, as vectorbt raises.

        Examples
        --------
        Bar 1 fills 0.1 shares of AAA at 10. AAA delists on bar 1 at a
        valuation of 8, so bar 2 settles them at 8:

        >>> import numpy as np
        >>> fill = np.array([[10.0], [10.0], [np.nan]])
        >>> valuation = np.array([[10.0], [8.0], [np.nan]])
        >>> book = ExecutionBook(fill, valuation, np.array([[False], [True], [False]]))
        >>> book.submit(0, [1.0])
        >>> _ = book.trade(1)
        >>> book.trade(2).settled.tolist(), book.shares.tolist(), book.cash
        ([True], [0.0], 0.8)
        """
        prices = self._prices
        target = self._queued.pop(bar, None)
        n = self.shares.size
        if target is None:
            if not prices.settle[bar].any():
                return BarExecution(np.zeros(n, dtype=bool), np.zeros(n, dtype=bool))
            target = np.full(n, np.nan)
        plan = prices.plan(target, self.settings, bar)
        held_before = self.shares != 0
        report = BarExecution(
            rejected=plan.rejected & ((plan.target != 0.0) | held_before),
            settled=plan.settle & held_before,
        )
        # The prices the book is valued and the targets sized at: the order
        # prices on the fill basis, the signal bar's valuations otherwise.
        val = plan.sizing if self.settings.sizing_basis == "valuation" else plan.price
        value = _snap(self.cash + float(np.sum(self.shares[held_before] * val[held_before])))
        sent = np.isfinite(plan.size) & ~np.isnan(plan.price)
        worthless = sent & plan.settle & (plan.price == 0.0)
        # A settlement at a last valuation of 0 closes the position for nothing.
        self.shares[worthless] = 0.0
        sent &= ~worthless
        if np.any(plan.price[sent] <= 0.0) or np.any(val[sent] <= 0.0):
            raise ValueError(
                f"bar {bar}: an order meets a price of 0 or less on a symbol that "
                f"is not being settled"
            )
        # vectorbt ignores an order without a valuation price or a book value,
        # and rejects one against a book worth nothing.
        sent &= ~np.isnan(val)
        if not value > 0:
            return report
        with np.errstate(invalid="ignore", divide="ignore"):
            amount = np.where(sent, plan.size * value / val - self.shares, 0.0)
        # Only orders that change a holding are walked (vectorbt ignores the
        # rest), in ascending order of value, so sells come first (vectorbt's
        # call_seq="auto", a stable sort).
        trades = np.flatnonzero(np.abs(amount) > _ABS_TOL)
        order_value = plan.size[trades] * value - self.shares[trades] * val[trades]
        trades = trades[np.argsort(order_value, kind="stable")]
        shares = self.shares[trades].tolist()
        cash = self.cash
        for k, (j, size, price, fees, slippage) in enumerate(zip(
            trades.tolist(), amount[trades].tolist(), plan.price[trades].tolist(),
            plan.fees[trades].tolist(), plan.slippage[trades].tolist(),
        )):
            cash = _snap(cash)
            if size > 0:
                bought, paid = _buy(size, price, cash, fees, slippage)
                cash = add(cash, -paid)
                shares[k] = add(_snap(shares[k]), bought)
            else:
                cash = cash + _sell(-size, price, fees, slippage)
                shares[k] = add(_snap(shares[k]), size)
        self.shares[trades] = shares
        self.cash = float(cash)
        return report


def replay(
    weights,
    fill,
    valuation,
    delisted,
    settings: ExecutionSettings = ExecutionSettings(),
    init_cash: float = 1.0,
) -> ExecutionReplay:
    """Replay ``weights`` decided on each bar and return the holdings after every bar.

    Parameters
    ----------
    weights : array-like
        Target weights on ``[T, S]``, row t decided at bar t; NaN keeps a
        holding.
    fill, valuation, delisted : array-like
        As for ``plan_orders``, on ``[T, S]``.
    settings : ExecutionSettings
        The sizing basis, fees and slippage.
    init_cash : float
        The starting cash; the book starts with no shares.

    Returns
    -------
    ExecutionReplay
        Shares and cash after each bar's orders, and the rejections and
        settlements.

    Raises
    ------
    ValueError
        As ``ExecutionBook.trade`` raises.

    Examples
    --------
    A 1% fee on a buy of the whole book: vectorbt cuts it so cost and fee
    are the cash.

    >>> import numpy as np
    >>> prices = np.array([[10.0], [10.0]])
    >>> result = replay(
    ...     np.array([[1.0], [np.nan]]), prices, prices, np.zeros((2, 1), dtype=bool),
    ...     ExecutionSettings(fees=0.01),
    ... )
    >>> round(float(result.shares[1, 0]), 10), float(result.cash[1])
    (0.099009901, 0.0)
    """
    weights = np.asarray(weights, dtype=np.float64)
    book = ExecutionBook(fill, valuation, delisted, settings, init_cash)
    shares = np.zeros(weights.shape)
    cash = np.zeros(weights.shape[0])
    rejected = np.zeros(weights.shape, dtype=bool)
    settled = np.zeros(weights.shape, dtype=bool)
    for row in range(weights.shape[0]):
        report = book.trade(row)
        book.submit(row, weights[row])
        shares[row], cash[row] = book.shares, book.cash
        rejected[row], settled[row] = report.rejected, report.settled
    return ExecutionReplay(shares=shares, cash=cash, rejected=rejected, settled=settled)


def _shift(values: np.ndarray, fill_value=np.nan) -> np.ndarray:
    """Move ``[T, S]`` values ``FILL_DELAY_BARS`` rows later, padding the first rows."""
    shifted = np.full(values.shape, fill_value, dtype=values.dtype)
    shifted[FILL_DELAY_BARS:] = values[:-FILL_DELAY_BARS]
    return shifted


def _ffill(values: np.ndarray) -> np.ndarray:
    """Forward-fill ``[T, S]`` values down the time axis; leading NaNs stay."""
    filled = np.array(values, dtype=np.float64, copy=True)
    for row in range(1, filled.shape[0]):
        gap = np.isnan(filled[row])
        filled[row, gap] = filled[row - 1, gap]
    return filled


def is_close(a: float, b: float) -> bool:
    """vectorbt's ``is_close_nb``: equal up to its relative and absolute tolerance.

    One of the vectorbt numeric rules this module owns; the backtest metrics that
    rebuild vectorbt's accounting use it too, so fills and metrics cannot drift.

    Examples
    --------
    >>> is_close(1.0, 1.0 + 1e-12), is_close(1.0, 1.001), is_close(float("nan"), 0.0)
    (True, False, False)
    """
    if not (math.isfinite(a) and math.isfinite(b)):
        return False
    if a == b:
        return True
    return abs(a - b) <= max(_REL_TOL * max(abs(a), abs(b)), _ABS_TOL)


def add(a: float, b: float) -> float:
    """vectorbt's ``add_nb``: ``a + b``, exactly 0 when the two cancel up to tolerance.

    Of opposite signs (or one of them 0) they cancel when their magnitudes
    are ``is_close``; of the same sign when the sum is ``is_close`` to 0.
    Written out rather than through ``is_close``, since it runs per order.
    Takes numpy scalars as well as floats, and returns a float.

    Examples
    --------
    >>> add(0.3, -0.3 + 1e-13), add(1.0, 2.0)
    (0.0, 3.0)
    """
    a, b = float(a), float(b)
    total = a + b
    if ((a > 0) - (a < 0)) != ((b > 0) - (b < 0)):
        tol = max(_REL_TOL * max(abs(a), abs(b)), _ABS_TOL)
    else:
        tol = _ABS_TOL
    return 0.0 if abs(total) <= tol else total


def _snap(value: float) -> float:
    """Return 0.0 for a value ``is_close`` to 0, as vectorbt reads its state.

    ``is_close(v, 0)`` holds exactly when ``|v| <= 1e-12``, since the
    relative tolerance of ``|v|`` is below ``|v|`` itself.
    """
    return 0.0 if abs(value) <= _ABS_TOL else value


def _buy(amount: float, price: float, cash: float, fees: float, slippage: float) -> tuple[float, float]:
    """Return the shares bought and the cash paid, fee included, as vectorbt's ``buy_nb``.

    The price moves up by the slippage; a buy whose cost and fee exceed the
    cash is cut so they equal it, and none is made without cash.
    """
    if cash == 0:
        return 0.0, 0.0
    paid_price = price * (1 + slippage)
    cost = amount * paid_price
    total = cost + cost * fees
    if total <= cash or is_close(total, cash):
        return amount, total
    affordable = cash / (1 + fees)
    if affordable <= 0:
        return 0.0, 0.0
    return affordable / paid_price, cash


def _sell(amount: float, price: float, fees: float, slippage: float) -> float:
    """Return the cash a sale of ``amount`` shares brings in, as vectorbt's ``sell_nb``."""
    received = amount * price * (1 - slippage)
    return add(received, -received * fees)
