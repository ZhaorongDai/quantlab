"""Custom KunQuant operators that normalize a factor or tame its outliers.

KunQuant is the library this project uses to compute most factors. A factor
formula is written as a graph of operators (``WindowedAvg``, ``Rank``, ...),
and KunQuant compiles that graph to native C++ code that runs over a whole
``(timestamp, symbol)`` array at once. This module adds four operators to
that vocabulary.

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
"""

from KunQuant.Op import Builder
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

    No factor class applies this op by default; choosing it over
    ``WindowedZScore`` is a strategy decision.

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

