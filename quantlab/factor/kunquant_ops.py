"""Custom KunQuant operators: normalizations, outlier handling and weighted statistics.

KunQuant is the library this project uses to compute most factors. A factor
formula is written as a graph of operators (``WindowedAvg``, ``Rank``, ...),
and KunQuant compiles that graph to native C++ code that runs over a whole
``(timestamp, symbol)`` array at once. This module adds operators to that
vocabulary.

``WindowedZScore`` is a *time-series* normalization: each symbol is compared
with its own recent past. ``CrossSectionalZScore`` is a *cross-sectional*
normalization: at each timestamp, each symbol is compared with all the other
symbols on the same bar. Which one is right depends on the strategy that
consumes the factor. A strategy that trades one asset over time wants the
first; a strategy that ranks many assets against each other wants the
second. They are not two implementations of the same thing.

``CrossSectionalWinsorize`` (winsorizing, 缩尾) and ``CrossSectionalTrim``
(trimming, 截尾) handle outliers on each bar: the first clips values to that
bar's lower and upper quantiles across symbols, the second replaces values
outside them with NaN. They are typically applied before a cross-sectional
z-score so that a few extreme symbols do not dominate its mean and spread.

The exponentially weighted window statistics (``EWSum``, ``EWMean``,
``EWVar``, ``EWCov``, ``EWBeta``, ``EWAlpha``) weight the trailing ``window``
bars by ``0.5 ** (age / half_life)``, where ``age`` is 0 for the current bar,
and skip missing values. The cross-sectional ``CrossSectionalWeightedMean``,
``CrossSectionalTopN`` and ``CapWeightedStandardize`` and the elementwise
``SigmaClip`` are the estimation-universe tools of a Barra-style risk factor
(see ``quantlab.factor.predefined.barra``), usable by any factor.
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


class CrossSectionalZScore(GenericCrossSectionalOp):
    """Cross-sectional z-score, ``(x - mean_t) / std_t`` over symbols per bar.

    At every timestamp the mean and the sample standard deviation
    (``ddof=1``, matching pandas ``.std()`` and KunQuant's
    ``WindowedStddev``) are taken over all symbols, ignoring NaN. NaN inputs
    stay NaN. A bar with fewer than two valid values, or with zero standard
    deviation, is NaN for every symbol. No fill is applied.

    KunQuant's composite ops can only be built from time-series operators,
    so this op is a ``GenericCrossSectionalOp`` whose loop body is
    hand-written C++ (see ``generate_body``). That choice brings three
    constraints. First, the C++ body must not read any parameter of the op:
    KunQuant reuses generated C++ functions by class name and data layout
    alone, so a parameterized variant needs a class of its own. Second, batch
    runs must start at bar 0, because in KunQuant 0.1.11 a non-zero start
    index gives wrong results for every ``GenericCrossSectionalOp``. Third,
    the number of symbols must be a multiple of the host's SIMD block width
    (the number of values the CPU's vector instructions process at once), in
    both the time-major ``TS`` layout used for batch runs and the ``STREAM``
    layout used for bar-by-bar runs.

    ``Alpha101Stock`` and ``Alpha158Stock`` apply this op to every output;
    the spot-kline classes apply ``WindowedZScore`` instead. Choosing between
    the two is a strategy decision.

    Parameters
    ----------
    v : OpBase
        The input series, usually a factor expression.

    Examples
    --------
    >>> Output(CrossSectionalZScore(alpha(all_data)), "alpha001_cs")
    """

    def __init__(self, v: OpBase) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], None)

    def generate_head(self) -> str:
        """Return the C++ preamble for the generated function, which is empty.

        Examples
        --------
        >>> CrossSectionalZScore(Input("close")).generate_head()
        ''
        """
        return ""

    def generate_body(self) -> str:
        """Return the C++ loop that z-scores ``input_0`` into ``output_0``.

        KunQuant calls this when it emits the C++ for the graph. The loop
        runs once per timestamp over ``num_stocks`` symbols: one pass for
        the mean, one for the variance, and one to write the scores.

        Examples
        --------
        >>> body = CrossSectionalZScore(Input("close")).generate_body()
        >>> body.strip().splitlines()[0]
        'T sum = 0;'
        """
        return """
        T sum = 0;
        size_t n = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            if (!std::isnan(v)) { sum += v; n++; }
        }
        T mean = n > 0 ? sum / n : NAN;
        T ss = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            if (!std::isnan(v)) { T d = v - mean; ss += d * d; }
        }
        T sd = n > 1 ? std::sqrt(ss / (n - 1)) : NAN;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            output_0[i] = (std::isnan(v) || !(sd > 0)) ? NAN : (v - mean) / sd;
        }
        """


def _quantile_tag(q: float) -> str:
    """Spell a quantile as a C++-identifier-safe fragment, e.g. ``0.01 -> 0p01``."""
    return repr(float(q)).replace(".", "p").replace("-", "m").replace("+", "")


class _CrossSectionalQuantileBounds(GenericCrossSectionalOp):
    """Shared machinery for the quantile-bounded cross-sectional ops.

    At every timestamp the valid (non-NaN) values across symbols are sorted
    and the ``lower`` and ``upper`` quantiles are taken with linear
    interpolation, as numpy's ``np.nanquantile`` and pandas' ``quantile``
    do by default. Subclasses decide what happens to a value outside those
    bounds by setting ``_TRIM``.

    KunQuant names the generated C++ function after the op's class and data
    layout only, so two instances with different quantiles would silently
    share one function. To keep a parameterized API anyway, constructing
    ``CrossSectionalWinsorize(v, 0.05, 0.95)`` returns an instance of a
    subclass created once per ``(lower, upper)`` pair, whose name carries
    the quantiles (``CrossSectionalWinsorize_0p05_0p95``) and whose C++ body
    has them baked in as literals. ``isinstance(op, CrossSectionalWinsorize)``
    still holds.
    """

    _TRIM: bool = False
    _LOWER: float = 0.01
    _UPPER: float = 0.99
    _variants: dict = {}

    def __new__(cls, v: OpBase, lower: float = 0.01, upper: float = 0.99):
        """Return an instance of the subclass specialized to ``(lower, upper)``."""
        lower, upper = float(lower), float(upper)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(
                f"{cls.__name__}: need 0 <= lower < upper <= 1, "
                f"got lower={lower}, upper={upper}"
            )
        base = cls._public_base()
        key = (base, lower, upper)
        variant = _CrossSectionalQuantileBounds._variants.get(key)
        if variant is None:
            name = f"{base.__name__}_{_quantile_tag(lower)}_{_quantile_tag(upper)}"
            variant = type(name, (base,), {"_LOWER": lower, "_UPPER": upper})
            variant.__module__ = base.__module__
            _CrossSectionalQuantileBounds._variants[key] = variant
        return super().__new__(variant)

    @classmethod
    def _public_base(cls) -> type:
        """The user-facing class (``CrossSectionalWinsorize`` or ``CrossSectionalTrim``)."""
        for klass in cls.__mro__:
            if _CrossSectionalQuantileBounds in klass.__bases__:
                return klass
        raise TypeError(f"{cls.__name__} is not a quantile-bounds op")

    def __init__(self, v: OpBase, lower: float = 0.01, upper: float = 0.99) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], [("lower", float(lower)), ("upper", float(upper))])

    def generate_head(self) -> str:
        """Return the C++ set-up that runs once per generated function call.

        It declares the scratch buffer the valid values are sorted in and a
        ``quantile`` helper using linear interpolation between order
        statistics.
        """
        return """
        std::vector<T> sorted_buf;
        sorted_buf.reserve(num_stocks);
        auto quantile = [&](double q) -> T {
            double pos = q * (double)(sorted_buf.size() - 1);
            size_t lo = (size_t)std::floor(pos);
            size_t hi = std::min(lo + 1, sorted_buf.size() - 1);
            double frac = pos - (double)lo;
            return (T)((double)sorted_buf[lo]
                       + frac * ((double)sorted_buf[hi] - (double)sorted_buf[lo]));
        };
        """

    def generate_body(self) -> str:
        """Return the C++ loop that bounds ``input_0`` into ``output_0`` for one bar.

        The quantiles and the clip-or-trim choice come from class constants,
        not from the op's attributes, and are written into the code as
        literals (see the class docstring for why).
        """
        outside = "NAN" if self._TRIM else "(v < lo ? lo : hi)"
        return f"""
        sorted_buf.clear();
        for (size_t i = 0; i < num_stocks; i++) {{
            T v = input_0[i];
            if (!std::isnan(v)) sorted_buf.push_back(v);
        }}
        T lo = NAN, hi = NAN;
        if (!sorted_buf.empty()) {{
            std::sort(sorted_buf.begin(), sorted_buf.end());
            lo = quantile({self._LOWER!r});
            hi = quantile({self._UPPER!r});
        }}
        for (size_t i = 0; i < num_stocks; i++) {{
            T v = input_0[i];
            output_0[i] = (std::isnan(v) || (v >= lo && v <= hi)) ? v : {outside};
        }}
        """


class CrossSectionalWinsorize(_CrossSectionalQuantileBounds):
    """Cross-sectional winsorization (缩尾): clip each bar to its quantile bounds.

    At every timestamp the ``lower`` and ``upper`` quantiles are computed
    over all symbols, ignoring NaN, with linear interpolation (numpy's and
    pandas' default). A value below the lower bound is replaced by the lower
    bound, a value above the upper bound by the upper bound, and every other
    value passes through unchanged. Winsorizing keeps every observation but
    limits how far an outlier can pull a later cross-sectional statistic,
    such as ``CrossSectionalZScore``.

    NaN inputs stay NaN, and an all-NaN bar stays all NaN. A bar with a
    single valid value returns that value. No fill is applied.

    The op shares ``CrossSectionalZScore``'s constraints: batch runs must
    start at bar 0, and the number of symbols must be a multiple of the
    host's SIMD block width. Each distinct ``(lower, upper)`` pair compiles
    its own C++ function (see ``_CrossSectionalQuantileBounds``), so the
    constructed object's class is a subclass such as
    ``CrossSectionalWinsorize_0p01_0p99``.

    Parameters
    ----------
    v : OpBase
        The input series, usually a factor expression.
    lower : float, default 0.01
        Lower quantile, in ``[0, 1)``.
    upper : float, default 0.99
        Upper quantile, in ``(lower, 1]``.

    Raises
    ------
    ValueError
        If ``0 <= lower < upper <= 1`` does not hold.

    Examples
    --------
    >>> Output(CrossSectionalZScore(CrossSectionalWinsorize(alpha(all_data))), "alpha001_cs")
    """


class CrossSectionalTrim(_CrossSectionalQuantileBounds):
    """Cross-sectional trimming (截尾): drop values outside each bar's quantile bounds.

    The bounds are computed exactly as in ``CrossSectionalWinsorize``. A
    value strictly below the lower bound or strictly above the upper bound
    becomes NaN; a value equal to a bound is kept. Unlike winsorizing,
    trimming removes the outliers from the cross-section altogether, so
    downstream consumers see fewer valid symbols on each bar.

    NaN inputs stay NaN, and an all-NaN bar stays all NaN. A bar with a
    single valid value returns that value. No fill is applied. The same
    batch-start, SIMD-width and one-class-per-quantile-pair notes as for
    ``CrossSectionalWinsorize`` apply.

    Parameters
    ----------
    v : OpBase
        The input series, usually a factor expression.
    lower : float, default 0.01
        Lower quantile, in ``[0, 1)``.
    upper : float, default 0.99
        Upper quantile, in ``(lower, 1]``.

    Raises
    ------
    ValueError
        If ``0 <= lower < upper <= 1`` does not hold.

    Examples
    --------
    >>> Output(CrossSectionalTrim(alpha(all_data), 0.05, 0.95), "alpha001_trim")
    """

    _TRIM = True


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
        mean_y, mean_x, cov, var_x, _ = self._regression_moments(b, with_var_y=False)
        return mean_y, mean_x, cov, var_x

    def _regression_moments(
        self, b: Builder, with_var_y: bool = True
    ) -> tuple[OpBase, OpBase, OpBase, OpBase, OpBase | None]:
        """Return ``(mean_y, mean_x, cov_xy, var_x, var_y)`` over the bars where both are valid.

        ``var_y`` is ``None`` unless ``with_var_y``. See ``_beta_parts``.
        """
        y = _jointly(self.inputs[0], self.inputs[1])
        x = _jointly(self.inputs[1], self.inputs[0])
        mean_x, mean_y, total = self._means(b, x, y)

        def terms(values, weight):
            dx, dy = values[0] - mean_x, values[1] - mean_y
            return [weight * dx * dy, weight * dx * dx] + ([weight * dy * dy] if with_var_y else [])

        sums = self._weighted_sums(b, [x, y], terms)
        filled = _window_filled(self.inputs[0] + self.inputs[1], self._window)
        moments = [mean_y + filled, mean_x + filled] + [value / total + filled for value in sums]
        return (*moments, None) if not with_var_y else tuple(moments)  # type: ignore[return-value]


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
            _, _, cov, var_x, var_y = self._regression_moments(b)
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


class RenormalizedCombine(CompositiveOp):
    """Fixed-weight sum of several values, renormalized over the ones present.

    ``sum(w_k * v_k) / sum(w_k)`` over the ``k`` whose ``v_k`` is finite,
    so a symbol missing one descriptor still gets the combination of the
    others, as USE4 combines descriptors into a style. NaN when none is
    present.

    Parameters
    ----------
    values : list of OpBase
        The values to combine, such as standardized descriptors.
    weights : list of float
        One positive weight per value.

    Raises
    ------
    ValueError
        If the lengths differ, no value is given, or a weight is not positive.

    Examples
    --------
    >>> Output(RenormalizedCombine([dastd, cmra, hsigma], [0.75, 0.15, 0.10]), "resvol")
    """

    def __init__(self, values: list[OpBase], weights: list[float]) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        weights = [float(weight) for weight in weights]
        if not values or len(values) != len(weights):
            raise ValueError(
                f"RenormalizedCombine: need one weight per value, got {len(values)} "
                f"value(s) and {len(weights)} weight(s)"
            )
        if any(not weight > 0.0 for weight in weights):
            raise ValueError(f"RenormalizedCombine: weights must be positive, got {weights}")
        super().__init__(list(values), [("weights", tuple(weights))])

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into masked sums and one division."""
        weights: tuple = self.attrs["weights"]  # type: ignore[assignment]
        b = Builder(self.get_parent())
        with b:
            total: OpBase = ConstantOp(0.0)
            present: OpBase = ConstantOp(0.0)
            for value, weight in zip(self.inputs, weights):
                total = total + _zero_where_missing(value) * weight
                present = present + _present(value) * weight
            Select(present > 0.0, total / present, ConstantOp("nan"))
        return b.ops


class CrossSectionalWeightedMean(GenericCrossSectionalOp):
    """Weighted mean of ``v`` across symbols per bar, broadcast to every symbol.

    ``sum(w * v) / sum(w)`` over the symbols whose ``v`` is finite and whose
    weight is finite and positive; any other symbol is left out.
    A bar with no such symbol is NaN. Give the weight 0 outside a universe
    to average over the universe only, as the cap-weighted market return of
    an estimation universe does.

    Shares ``CrossSectionalZScore``'s constraints: batch runs start at bar
    0 and the symbol count is a multiple of the SIMD block width.

    Parameters
    ----------
    v : OpBase
        The values to average.
    w : OpBase
        The weights, such as market capitalizations.

    Examples
    --------
    >>> Output(CrossSectionalWeightedMean(Input("ret"), Input("cap")), "market")
    """

    def __init__(self, v: OpBase, w: OpBase) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v, w], None)

    def generate_head(self) -> str:
        """Return the C++ preamble for the generated function, which is empty."""
        return ""

    def generate_body(self) -> str:
        """Return the C++ loop that writes the weighted mean of one bar."""
        return """
        T sw = 0, swv = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i], w = input_1[i];
            if (std::isfinite(v) && std::isfinite(w) && w > 0) { sw += w; swv += w * v; }
        }
        T mean = sw > 0 ? swv / sw : NAN;
        for (size_t i = 0; i < num_stocks; i++) output_0[i] = mean;
        """


class CrossSectionalTopN(GenericCrossSectionalOp):
    """1 for the ``n`` symbols with the largest ``v`` on each bar, 0 elsewhere.

    NaN and infinite values are never selected and come out 0; when fewer than ``n``
    values are valid, all of them are selected. A tie at the boundary is
    broken by symbol order: the earlier symbol on the axis wins. The output
    is a 0/1 mask, such as an estimation universe of the ``n`` largest
    companies by the previous bar's market cap.

    The C++ body cannot read an op parameter (see
    ``_CrossSectionalQuantileBounds``), so constructing
    ``CrossSectionalTopN(v, 3000)`` returns an instance of a subclass created
    once per ``n`` (``CrossSectionalTopN_3000``) with ``n`` written into its
    code. ``isinstance(op, CrossSectionalTopN)`` still holds. The batch-start
    and SIMD-width notes of ``CrossSectionalZScore`` apply.

    Parameters
    ----------
    v : OpBase
        The values to rank, larger is better.
    n : int
        How many symbols to select per bar, at least 1.

    Raises
    ------
    ValueError
        If ``n < 1``.

    Examples
    --------
    >>> Output(CrossSectionalTopN(BackRef(Input("marketcap"), 1), 3000), "estu")
    """

    _N: int = 1
    _variants: dict = {}

    def __new__(cls, v: OpBase, n: int):
        """Return an instance of the subclass specialized to ``n``."""
        if int(n) != n or n < 1:
            raise ValueError(f"CrossSectionalTopN: need an integer n >= 1, got {n}")
        n = int(n)
        variant = CrossSectionalTopN._variants.get(n)
        if variant is None:
            variant = type(f"CrossSectionalTopN_{n}", (CrossSectionalTopN,), {"_N": n})
            variant.__module__ = CrossSectionalTopN.__module__
            CrossSectionalTopN._variants[n] = variant
        return super().__new__(variant)

    def __init__(self, v: OpBase, n: int) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v], [("n", int(n))])

    def generate_head(self) -> str:
        """Return the C++ set-up: a buffer of the valid symbols' indices."""
        return """
        std::vector<size_t> valid;
        valid.reserve(num_stocks);
        """

    def generate_body(self) -> str:
        """Return the C++ loop that marks the top ``n`` symbols of one bar."""
        return f"""
        valid.clear();
        for (size_t i = 0; i < num_stocks; i++) {{
            output_0[i] = 0;
            if (std::isfinite(input_0[i])) valid.push_back(i);
        }}
        size_t keep = std::min(valid.size(), (size_t){self._N});
        auto larger = [&](size_t a, size_t b) {{
            T va = input_0[a], vb = input_0[b];
            return va > vb || (va == vb && a < b);
        }};
        if (keep < valid.size()) {{
            std::nth_element(valid.begin(), valid.begin() + keep, valid.end(), larger);
        }}
        for (size_t k = 0; k < keep; k++) output_0[valid[k]] = 1;
        """


class CapWeightedStandardize(GenericCrossSectionalOp):
    """Standardize to a weighted mean of 0 and an equal-weighted std of 1 in a universe.

    On each bar, with ``U`` the symbols where ``universe > 0`` and ``v`` is
    finite::

        mean = sum(w * v) / sum(w)       over U where w is finite and > 0
        std  = sample std of v (ddof=1)  over U, equally weighted
        out  = (v - mean) / std          for EVERY symbol with a valid v

    This is USE4's standardization with ``w`` the market cap: the
    cap-weighted universe then has zero exposure, and the universe's
    exposures have unit spread. Symbols outside the universe are shifted and
    scaled by the same numbers, so they get exposures too. NaN and infinite
    inputs come out NaN; a bar whose mean or std is undefined, or whose std is 0, is NaN
    for every symbol.

    The batch-start and SIMD-width notes of ``CrossSectionalZScore`` apply.

    Parameters
    ----------
    v : OpBase
        The raw values, such as a descriptor.
    w : OpBase
        The weights of the mean, such as the previous bar's market cap.
    universe : OpBase
        A mask, positive inside the universe (``CrossSectionalTopN``'s 1).

    Examples
    --------
    >>> cap = BackRef(Input("marketcap"), 1)
    >>> estu = CrossSectionalTopN(cap, 3000)
    >>> Output(CapWeightedStandardize(Log(Input("marketcap")), cap, estu), "size")
    """

    def __init__(self, v: OpBase, w: OpBase, universe: OpBase) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([v, w, universe], None)

    def generate_head(self) -> str:
        """Return the C++ preamble for the generated function, which is empty."""
        return ""

    def generate_body(self) -> str:
        """Return the C++ loops that standardize one bar."""
        return """
        T sw = 0, swv = 0, sum = 0;
        size_t n = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i], w = input_1[i];
            if (input_2[i] > 0 && std::isfinite(v)) {
                sum += v; n++;
                if (std::isfinite(w) && w > 0) { sw += w; swv += w * v; }
            }
        }
        T mean_eq = n > 0 ? sum / n : NAN;
        T ss = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            if (input_2[i] > 0 && std::isfinite(v)) { T d = v - mean_eq; ss += d * d; }
        }
        T sd = n > 1 ? std::sqrt(ss / (n - 1)) : NAN;
        T mean = sw > 0 ? swv / sw : NAN;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            output_0[i] = (!std::isfinite(v) || std::isnan(mean) || !(sd > 0)) ? NAN : (v - mean) / sd;
        }
        """


class SigmaClip(CompositiveOp):
    """Drop data errors and clip outliers of an already standardized value.

    A value whose magnitude exceeds ``data_error`` is treated as a data
    error and becomes NaN; any other value is clipped to
    ``[-clip, clip]``. NaN stays NaN. Apply it to a standardized exposure,
    where both thresholds are in standard deviations (USE4 clips at 3).

    Parameters
    ----------
    v : OpBase
        A standardized series, such as ``CapWeightedStandardize``'s output.
    data_error : float, default 10.0
        Magnitude beyond which a value is dropped.
    clip : float, default 3.0
        Magnitude values are clipped to, at most ``data_error``.

    Raises
    ------
    ValueError
        If ``0 < clip <= data_error`` does not hold.

    Examples
    --------
    >>> Output(SigmaClip(CapWeightedStandardize(raw, cap, estu)), "beta")
    """

    def __init__(self, v: OpBase, data_error: float = 10.0, clip: float = 3.0) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        data_error, clip = float(data_error), float(clip)
        if not 0.0 < clip <= data_error:
            raise ValueError(
                f"SigmaClip: need 0 < clip <= data_error, got clip={clip}, "
                f"data_error={data_error}"
            )
        super().__init__([v], [("data_error", data_error), ("clip", clip)])

    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into comparisons and ``Select``; NaN compares false and passes through."""
        data_error: float = self.attrs["data_error"]  # type: ignore[assignment]
        clip: float = self.attrs["clip"]  # type: ignore[assignment]
        b = Builder(self.get_parent())
        with b:
            v = self.inputs[0]
            clipped = Select(
                v > clip,
                ConstantOp(clip),
                Select(v < -clip, ConstantOp(-clip), v),
            )
            Select(Abs(v) > data_error, ConstantOp("nan"), clipped)
        return b.ops


#: C++ of the residual ops' shared step: the weighted means over the fit sample.
_WLS_MEANS = """
        T sw = 0, swy = 0, swx1 = 0, swx2 = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            if (!fit(i)) continue;
            T w = weight(i);
            sw += w; swy += w * input_0[i]; swx1 += w * x1(i); swx2 += w * x2(i);
        }
        T my = sw > 0 ? swy / sw : NAN, m1 = sw > 0 ? swx1 / sw : NAN, m2 = sw > 0 ? swx2 / sw : NAN;
        T s11 = 0, s12 = 0, s22 = 0, s1y = 0, s2y = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            if (!fit(i)) continue;
            T w = weight(i), d1 = x1(i) - m1, d2 = x2(i) - m2, dy = input_0[i] - my;
            s11 += w * d1 * d1; s12 += w * d1 * d2; s22 += w * d2 * d2;
            s1y += w * d1 * dy; s2y += w * d2 * dy;
        }
"""

#: C++ writing the residual of every symbol with a finite ``y``; a missing
#: regressor is taken at its weighted mean, so it adjusts nothing.
_WLS_WRITE = """
        for (size_t i = 0; i < num_stocks; i++) {
            T y = input_0[i];
            if (!std::isfinite(y) || std::isnan(b1) || std::isnan(b2)) { output_0[i] = NAN; continue; }
            T a1 = std::isfinite(x1(i)) ? x1(i) - m1 : 0, a2 = std::isfinite(x2(i)) ? x2(i) - m2 : 0;
            output_0[i] = y - my - b1 * a1 - b2 * a2;
        }
"""


class CrossSectionalWLSResidual(GenericCrossSectionalOp):
    """Residual of a per-bar weighted least-squares fit of ``y`` on ``x``, with an intercept.

    On each bar the fit uses the symbols where ``universe > 0``, ``y`` and
    ``x`` are finite and ``w`` is finite and positive::

        b   = sum(w (x - mx)(y - my)) / sum(w (x - mx)**2)
        out = y - my - b (x - mx)          for EVERY symbol with a finite y

    ``mx`` and ``my`` being the weighted means of the fit sample. The
    residual has zero weighted covariance with ``x`` over the fit sample.
    This orthogonalizes one style against another, as USE4 orthogonalizes
    Non-linear Size against Size. A symbol whose ``x`` is missing is taken
    at ``mx`` and so keeps ``y - my``. A bar with no fit sample, or where
    ``x`` does not vary over it, is NaN for every symbol.

    The batch-start and SIMD-width notes of ``CrossSectionalZScore`` apply.

    Parameters
    ----------
    y : OpBase
        The values to orthogonalize.
    x : OpBase
        The regressor.
    w : OpBase
        The regression weights, such as the square root of market cap.
    universe : OpBase
        A mask, positive for the symbols the fit uses.

    Examples
    --------
    >>> Output(CrossSectionalWLSResidual(size_cubed, size, Sqrt(cap), estu), "nlsize")
    """

    def __init__(self, y: OpBase, x: OpBase, w: OpBase, universe: OpBase) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x, w, universe], None)

    def generate_head(self) -> str:
        """Return the C++ preamble for the generated function, which is empty."""
        return ""

    def _accessors(self) -> str:
        """Return the C++ accessors of the fit sample, declared in each bar's body."""
        return """
        auto x1 = [&](size_t i) -> T { return input_1[i]; };
        auto x2 = [&](size_t i) -> T { return (T)0; };
        auto weight = [&](size_t i) -> T { return input_2[i]; };
        auto fit = [&](size_t i) -> bool {
            T w = input_2[i];
            return input_3[i] > 0 && std::isfinite(input_0[i]) && std::isfinite(input_1[i])
                && std::isfinite(w) && w > 0;
        };
        """

    def generate_body(self) -> str:
        """Return the C++ that fits one bar and writes its residuals."""
        return self._accessors() + _WLS_MEANS + """
        T b1 = s11 > 0 ? s1y / s11 : NAN, b2 = 0;
""" + _WLS_WRITE


class CrossSectionalWLSResidual2(GenericCrossSectionalOp):
    """Residual of a per-bar weighted least-squares fit of ``y`` on ``x1`` and ``x2``, with an intercept.

    ``CrossSectionalWLSResidual`` with two regressors: the slopes solve the
    weighted, mean-centred 2x2 normal equations over the fit sample (the
    symbols where ``universe > 0``, ``y``, ``x1`` and ``x2`` are finite and
    ``w`` is finite and positive), and every symbol with a finite ``y`` gets
    ``y - my - b1 (x1 - m1) - b2 (x2 - m2)``, a missing regressor taken at
    its mean. USE4 orthogonalizes Residual Volatility against Beta and Size
    this way. A bar whose regressors are collinear over the fit sample is
    NaN for every symbol.

    The batch-start and SIMD-width notes of ``CrossSectionalZScore`` apply.

    Parameters
    ----------
    y : OpBase
        The values to orthogonalize.
    x1, x2 : OpBase
        The regressors.
    w : OpBase
        The regression weights.
    universe : OpBase
        A mask, positive for the symbols the fit uses.

    Examples
    --------
    >>> Output(CrossSectionalWLSResidual2(resvol, beta, size, Sqrt(cap), estu), "resvol")
    """

    def __init__(self, y: OpBase, x1: OpBase, x2: OpBase, w: OpBase, universe: OpBase) -> None:
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__([y, x1, x2, w, universe], None)

    def generate_head(self) -> str:
        """Return the C++ preamble for the generated function, which is empty."""
        return ""

    def _accessors(self) -> str:
        """Return the C++ accessors of the fit sample, declared in each bar's body."""
        return """
        auto x1 = [&](size_t i) -> T { return input_1[i]; };
        auto x2 = [&](size_t i) -> T { return input_2[i]; };
        auto weight = [&](size_t i) -> T { return input_3[i]; };
        auto fit = [&](size_t i) -> bool {
            T w = input_3[i];
            return input_4[i] > 0 && std::isfinite(input_0[i]) && std::isfinite(input_1[i])
                && std::isfinite(input_2[i]) && std::isfinite(w) && w > 0;
        };
        """

    def generate_body(self) -> str:
        """Return the C++ that fits one bar and writes its residuals."""
        return self._accessors() + _WLS_MEANS + """
        T det = s11 * s22 - s12 * s12;
        T b1 = det > 0 ? (s22 * s1y - s12 * s2y) / det : NAN;
        T b2 = det > 0 ? (s11 * s2y - s12 * s1y) / det : NAN;
""" + _WLS_WRITE
