"""Custom KunQuant time-series operators: each symbol against its own past.

KunQuant is the library this project uses to compute most factors. A factor
formula is written as a graph of operators (``WindowedAvg``, ``Rank``, ...),
and KunQuant compiles that graph to native C++ code that runs over a whole
``(timestamp, symbol)`` array at once. This module adds operators that look
along time, one symbol at a time; ``quantlab.factor.kunquant_cs`` has the
ones that look across symbols on one bar.

``WindowedZScore`` is a *time-series* normalization: each symbol is compared
with its own recent past. Its cross-sectional counterpart,
``kunquant_cs.CrossSectionalZScore``, compares each symbol with all the other
symbols on the same bar. Which one is right depends on the strategy that
consumes the factor. A strategy that trades one asset over time wants the
first; a strategy that ranks many assets against each other wants the
second. They are not two implementations of the same thing.

The exponentially weighted window statistics (``EWSum``, ``EWMean``,
``EWVar``, ``EWCov``, ``EWBeta``, ``EWAlpha``, ``EWResidualStd``) weight the
trailing ``window`` bars by ``0.5 ** (age / half_life)``, where ``age`` is 0
for the current bar, and skip missing values. ``CMRA`` is USE4's cumulative
range of monthly returns. ``BarraStyle``
(``quantlab.factor.predefined.barra``) uses them; any factor can.
"""

from KunQuant.Op import (
    Builder,
    ForeachBackWindow,
    IterValue,
    WindowedTempOutput,
    WindowLoopIndex,
)
from KunQuant.ops import *


class WindowedZScore(WindowedCompositiveOp):
    """Rolling z-score along time, ``(x - rolling_mean) / rolling_std``.

    Each symbol is standardized against its own trailing ``window`` bars;
    nothing is computed across symbols. NaN inputs propagate, and the first
    ``window - 1`` bars are NaN until the window is full, as with every
    KunQuant rolling operator. No fill is applied, so fill missing values in
    the caller if you need them.

    The operator is a *composite* op: KunQuant replaces it with simpler
    built-in operators (see ``decompose``) while compiling.

    Parameters
    ----------
    v : OpBase
        The input series, usually a factor expression.
    window : int
        Number of trailing bars the mean and standard deviation use.

    Examples
    --------
    >>> Output(WindowedZScore(alpha(all_data), 20), "alpha001")
    """

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into ``WindowedAvg``, ``WindowedStddev``, ``Sub`` and ``Div``.

        KunQuant calls this while compiling the graph.

        Parameters
        ----------
        options : dict
            Decomposition options passed by KunQuant. Unused, but required
            by KunQuant's composite-operator interface.

        Returns
        -------
        list[OpBase]
            The replacement operators, in dependency order.

        Examples
        --------
        It can also be called directly on an op built inside a ``Builder``:

        >>> with Builder():
        ...     z = WindowedZScore(Input("close"), 20)
        >>> [type(op).__name__ for op in z.decompose({})]
        ['WindowedAvg', 'WindowedStddev', 'Sub', 'Div']
        """
        window: int = self.attrs["window"]  # type: ignore
        b = Builder(self.get_parent())
        with b:
            rolling_mean = WindowedAvg(self.inputs[0], window)
            rolling_std = WindowedStddev(self.inputs[0], window)

            diff = Sub(self.inputs[0], rolling_mean)
            z_score = Div(diff, rolling_std)

        return b.ops


def _check_ew_window(window: int, half_life: float) -> tuple[int, float]:
    """Return ``(window, half_life)`` as ``(int, float)``, or raise if unusable."""
    window, half_life = int(window), float(half_life)
    if window < 1:
        raise ValueError(f"an EW window needs window >= 1, got {window}")
    if not half_life > 0.0:
        raise ValueError(f"an EW window needs half_life > 0, got {half_life}")
    return window, half_life


def _ew_weight(loop: ForeachBackWindow, window: int, half_life: float) -> OpBase:
    """Return ``0.5 ** (age / half_life)`` for the loop's current bar, inside ``loop``.

    ``age = window - 1 - WindowLoopIndex`` is 0 for the newest bar. KunQuant's
    ``Exp`` is a short polynomial accurate to about 1e-7 even in double, so the
    weight is built exactly instead: ``age`` is split into powers of two and
    the matching constants ``decay ** (2 ** k)``, computed in Python float64,
    are multiplied together.
    """
    decay = 0.5 ** (1.0 / half_life)
    rest = (WindowLoopIndex(loop) - float(window - 1)) * -1.0
    weight: OpBase = ConstantOp(1.0)
    step = 1 << max(window - 1, 1).bit_length() - 1
    while step >= 1 and window > 1:
        has_bit = rest >= float(step)
        weight = weight * Select(has_bit, ConstantOp(decay**step), ConstantOp(1.0))
        rest = rest - Select(has_bit, ConstantOp(float(step)), ConstantOp(0.0))
        step //= 2
    return weight


def _zero_where_missing(value: OpBase) -> OpBase:
    """Return ``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite and 0 where it is NaN or infinite."""
    return _zero_where_missing(value * 0.0 + 1.0)


def _jointly(value: OpBase, other: OpBase) -> OpBase:
    """Return ``value``, NaN wherever ``other`` is NaN or infinite."""
    return value + other * 0.0


def _window_filled(value: OpBase, window: int) -> OpBase:
    """Return 0 once ``window`` bars of history exist and NaN before.

    Added to a windowed result so that it stays NaN until the window has
    filled; a missing value inside the window does not count against it.
    """
    zero = _zero_where_missing(value * 0.0)
    return BackRef(zero, window - 1) if window > 1 else zero


class _EWWindowOp(CompositiveOp, WindowedTrait):
    """Shared machinery of the exponentially weighted window statistics.

    The weights are ``0.5 ** (age / half_life)`` over the trailing ``window``
    bars, ``age`` 0 being the current bar. A NaN or infinite value is skipped:
    it adds nothing to a weighted sum and its weight is left out of the sum
    of weights. The result is NaN until ``window`` bars of history exist, and
    a statistic with no valid value in its window is NaN. Every op here is a
    composite op: KunQuant expands it into loops over the window
    (``ForeachBackWindow``) while compiling, so the parameters can differ
    between two uses in one graph.
    """

    def __init__(self, inputs: list[OpBase], window: int, half_life: float) -> None:
        """Initialize the operator; see the subclass docstring for parameters."""
        window, half_life = _check_ew_window(window, half_life)
        super().__init__(inputs, [("window", window), ("half_life", half_life)])

    @property
    def _window(self) -> int:
        return self.attrs["window"]  # type: ignore[return-value]

    @property
    def _half_life(self) -> float:
        return self.attrs["half_life"]  # type: ignore[return-value]

    def _weighted_sums(self, b: Builder, sources: list[OpBase], terms) -> list[OpBase]:
        """Return one EW sum per term, all computed in a single window loop.

        ``terms(values, weight)`` receives the loop's current value of each
        source and that bar's weight, and returns the per-bar terms to sum.
        A term that is NaN or infinite adds 0.
        """
        window = self._window
        windowed = [WindowedTempOutput(source, window) for source in sources]
        loop = ForeachBackWindow(windowed[0], window, *windowed[1:])
        b.set_loop(loop)
        values = [IterValue(loop, source) for source in windowed]
        per_bar = [_zero_where_missing(term) for term in terms(values, _ew_weight(loop, window, self._half_life))]
        b.set_loop(self.get_parent())
        return [ReduceAdd(term) for term in per_bar]

    def _means(self, b: Builder, x: OpBase, y: OpBase | None = None) -> list[OpBase]:
        """Return the EW means of ``x`` (and ``y``) and the sum of weights, in one loop.

        Pass ``x`` and ``y`` already masked to their joint presence.
        """
        sources = [x] if y is None else [x, y]
        sums = self._weighted_sums(
            b,
            sources,
            lambda values, weight: [weight * _present(values[0])]
            + [weight * value for value in values],
        )
        total = sums[0]
        return [weighted / total for weighted in sums[1:]] + [total]

    def _beta_parts(self, b: Builder) -> tuple[OpBase, OpBase, OpBase, OpBase]:
        """Return ``(mean_y, mean_x, cov_xy, var_x)`` over the bars where both are valid.

        Shared by ``EWCov``, ``EWBeta`` and ``EWAlpha``, whose inputs are
        ``(y, x)``. The means come first, then the centred moments in a
        second loop (a two-pass estimate, which keeps its precision when the
        mean is large against the spread).
        """
        return self._regression_moments(b, ["xy", "xx"])  # type: ignore[return-value]

    def _regression_moments(self, b: Builder, products: list[str]) -> list[OpBase]:
        """Return ``[mean_y, mean_x]`` and the named centred moments, over the bars where both are valid.

        ``products`` names each moment by its two factors, ``"xy"``,
        ``"xx"`` or ``"yy"``: ``"xy"`` is ``sum(w (x - mean_x)(y - mean_y)) /
        sum(w)``. All of them come from one second loop. See ``_beta_parts``.
        """
        y = _jointly(self.inputs[0], self.inputs[1])
        x = _jointly(self.inputs[1], self.inputs[0])
        mean_x, mean_y, total = self._means(b, x, y)

        def terms(values, weight):
            centred = {"x": values[0] - mean_x, "y": values[1] - mean_y}
            return [weight * centred[a] * centred[c] for a, c in products]

        sums = self._weighted_sums(b, [x, y], terms)
        filled = _window_filled(self.inputs[0] + self.inputs[1], self._window)
        return [mean_y + filled, mean_x + filled] + [value / total + filled for value in sums]


class EWSum(_EWWindowOp):
    """Exponentially weighted window sum, ``sum(w_age * x)`` over valid bars.

    The weights are ``w_age = 0.5 ** (age / half_life)`` over the trailing
    ``window`` bars, ``age`` 0 being the current bar; they are not
    normalized. A NaN or infinite value adds nothing. The result is NaN for
    the first ``window - 1`` bars and 0 for a window with no valid value.

    Parameters
    ----------
    v : OpBase
        The input series.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWSum(Input("ret"), 252, 63), "ret_ew_sum")
    """

    def __init__(self, v: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into one window loop; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            (total,) = self._weighted_sums(
                b, [self.inputs[0]], lambda values, weight: [weight * values[0]]
            )
            total + _window_filled(self.inputs[0], self._window)
        return b.ops


class EWMean(_EWWindowOp):
    """Exponentially weighted window mean, ``sum(w * x) / sum(w)`` over valid bars.

    The weights are those of ``EWSum``; dividing by the sum of the weights
    of the valid bars normalizes them. The result is NaN for the first
    ``window - 1`` bars and for a window with no valid value.

    Parameters
    ----------
    v : OpBase
        The input series.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWMean(Input("ret"), 252, 63), "ret_ew_mean")
    """

    def __init__(self, v: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into one window loop; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            mean, _ = self._means(b, self.inputs[0])
            mean + _window_filled(self.inputs[0], self._window)
        return b.ops


class EWVar(_EWWindowOp):
    """Exponentially weighted window variance, ``sum(w * (x - m)**2) / sum(w)``.

    ``m`` is the ``EWMean`` of the same window and the sums run over the
    valid bars. The variance is the weighted population variance: no
    small-sample correction is applied, as in USE4's DASTD. NaN for the first
    ``window - 1`` bars and for a window with no valid value.

    Parameters
    ----------
    v : OpBase
        The input series.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(Sqrt(EWVar(Input("ret"), 252, 42)), "dastd")
    """

    def __init__(self, v: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into two window loops; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            x = self.inputs[0]
            mean, total = self._means(b, x)
            (square,) = self._weighted_sums(
                b, [x], lambda values, weight: [weight * (values[0] - mean) * (values[0] - mean)]
            )
            square / total + _window_filled(x, self._window)
        return b.ops


class EWCov(_EWWindowOp):
    """Exponentially weighted window covariance of ``y`` and ``x``.

    ``sum(w * (x - m_x) * (y - m_y)) / sum(w)`` over the bars where both are
    valid, the means taken over the same bars. A weighted population
    covariance, symmetric in its two inputs. NaN for the first ``window - 1``
    bars and for a window with no bar where both are valid.

    Parameters
    ----------
    y, x : OpBase
        The two input series.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWCov(Input("ret"), Input("market"), 252, 63), "cov")
    """

    def __init__(self, y: OpBase, x: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into two window loops; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            _, _, cov, _ = self._beta_parts(b)
            # KunQuant takes the last op built as the result; ``cov`` is not.
            cov * 1.0
        return b.ops


class EWBeta(_EWWindowOp):
    """Slope of the exponentially weighted least-squares fit of ``y`` on ``x``.

    The fit ``y = alpha + beta * x`` minimizes ``sum(w * residual**2)`` over
    the bars where both are valid, with an intercept. In closed form
    ``beta = cov_w(x, y) / var_w(x)`` (see ``EWCov`` and ``EWVar``). NaN for
    the first ``window - 1`` bars, and NaN or infinite when ``x`` does not
    vary over the valid bars.

    Parameters
    ----------
    y : OpBase
        The regressand, such as a stock's excess return.
    x : OpBase
        The regressor, such as the market's excess return.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWBeta(Input("excess"), Input("market_excess"), 252, 63), "beta")
    """

    def __init__(self, y: OpBase, x: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into two window loops; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            _, _, cov, var = self._beta_parts(b)
            cov / var
        return b.ops


class EWAlpha(_EWWindowOp):
    """Intercept of the exponentially weighted least-squares fit of ``y`` on ``x``.

    ``alpha = m_y - beta * m_x``, with ``beta`` as in ``EWBeta`` and the EW
    means taken over the bars where both are valid.

    Parameters
    ----------
    y : OpBase
        The regressand.
    x : OpBase
        The regressor.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWAlpha(Input("excess"), Input("market_excess"), 252, 63), "alpha")
    """

    def __init__(self, y: OpBase, x: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into two window loops; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            mean_y, mean_x, cov, var = self._beta_parts(b)
            mean_y - cov / var * mean_x
        return b.ops


class EWResidualStd(_EWWindowOp):
    """Standard deviation of the residual of ``EWBeta``'s fit of ``y`` on ``x``.

    With ``alpha`` and ``beta`` the exponentially weighted least-squares
    intercept and slope (see ``EWBeta``), the result is
    ``sqrt(sum(w * (y - alpha - beta * x)**2) / sum(w))`` over the bars
    where both are valid, computed in closed form as
    ``sqrt(var_w(y) - cov_w(x, y)**2 / var_w(x))`` and floored at 0 before
    the root. A weighted population figure, like ``EWVar``; USE4's HSIGMA
    with ``(y, x)`` the stock's and the market's excess returns. NaN for the
    first ``window - 1`` bars and when ``x`` does not vary over the valid
    bars.

    Parameters
    ----------
    y : OpBase
        The regressand, such as a stock's excess return.
    x : OpBase
        The regressor, such as the market's excess return.
    window : int
        Number of trailing bars, the current one included.
    half_life : float
        Age in bars at which the weight is one half.

    Raises
    ------
    ValueError
        If ``window < 1`` or ``half_life <= 0``.

    Examples
    --------
    >>> Output(EWResidualStd(Input("excess"), Input("market_excess"), 252, 63), "hsigma")
    """

    def __init__(self, y: OpBase, x: OpBase, window: int, half_life: float) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x], window, half_life)

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into two window loops; KunQuant calls this while compiling."""
        b = Builder(self.get_parent())
        with b:
            _, _, cov, var_x, var_y = self._regression_moments(b, ["xy", "xx", "yy"])
            residual = var_y - cov * cov / var_x
            Sqrt(Select(residual > 0.0, residual, residual * 0.0))
        return b.ops


class CMRA(CompositiveOp):
    """USE4's cumulative range of monthly returns, from a series of daily log returns.

    With ``Z(T)`` the sum of ``v`` over the last ``T * month_length`` bars,
    ``T = 1 .. months``::

        CMRA = log(1 + max_T Z(T)) - log(1 + min_T Z(T))

    as in USE4's Residual Volatility (``v`` is the daily log excess return
    ``log(1 + r) - log(1 + r_f)``, 12 months of 21 days). A missing value
    adds nothing to a sum. NaN for the first ``months * month_length - 1``
    bars, and where ``min_T Z(T) <= -1``, whose log is undefined. KunQuant's
    ``Log`` is accurate to about 4e-10 absolute in double, so is the result.

    Parameters
    ----------
    v : OpBase
        Daily log (excess) returns.
    months : int, default 12
        Number of trailing months.
    month_length : int, default 21
        Bars in one month.

    Raises
    ------
    ValueError
        If ``months`` or ``month_length`` is below 1.

    Examples
    --------
    >>> log_excess = Log(1.0 + Input("ret")) - Log(1.0 + Input("rf"))
    >>> Output(CMRA(log_excess, 12, 21), "cmra")
    """

    def __init__(self, v: OpBase, months: int = 12, month_length: int = 21) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        months, month_length = int(months), int(month_length)
        if months < 1 or month_length < 1:
            raise ValueError(
                f"CMRA: need months >= 1 and month_length >= 1, got {months}, {month_length}"
            )
        super().__init__([v], [("months", months), ("month_length", month_length)])

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into one window sum per month, a running max and min, and two logs."""
        months: int = self.attrs["months"]  # type: ignore[assignment]
        month_length: int = self.attrs["month_length"]  # type: ignore[assignment]
        b = Builder(self.get_parent())
        with b:
            daily = _zero_where_missing(self.inputs[0])
            sums = [WindowedSum(daily, month * month_length) for month in range(1, months + 1)]
            highest, lowest = sums[0], sums[0]
            for total in sums[1:]:
                highest, lowest = Max(highest, total), Min(lowest, total)
            # The longest sum is NaN until its window fills; adding it times
            # zero keeps the result NaN until then, whatever Max and Min do with NaN.
            cmra = Log(highest + 1.0) - Log(lowest + 1.0) + sums[-1] * 0.0
            Select(lowest > -1.0, cmra, ConstantOp("nan"))
        return b.ops
