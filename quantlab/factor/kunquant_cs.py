"""Custom KunQuant cross-sectional operators: each symbol against the others on one bar.

KunQuant is the library this project uses to compute most factors. A factor
formula is written as a graph of operators (``WindowedAvg``, ``Rank``, ...),
and KunQuant compiles that graph to native C++ code that runs over a whole
``(timestamp, symbol)`` array at once. This module adds operators that work
on one bar at a time; ``quantlab.factor.kunquant_ts`` has the ones that look
along time.

``CrossSectionalZScore`` is a *cross-sectional* normalization: at each
timestamp, each symbol is compared with all the other symbols on the same
bar. Its time-series counterpart, ``kunquant_ts.WindowedZScore``, compares
each symbol with its own recent past; which one is right depends on the
strategy that consumes the factor.

``CrossSectionalWinsorize`` (winsorizing, 缩尾) and ``CrossSectionalTrim``
(trimming, 截尾) handle outliers on each bar: the first clips values to that
bar's lower and upper quantiles across symbols, the second replaces values
outside them with NaN. They are typically applied before a cross-sectional
z-score so that a few extreme symbols do not dominate its mean and spread.

``CrossSectionalWeightedMean``, ``CrossSectionalTopN``,
``CapWeightedStandardize``, the weighted least-squares residuals
(``CrossSectionalWLSResidual``, ``CrossSectionalWLSResidual2``) and
``CrossSectionalIndustrySizeFill`` are the estimation-universe tools of a
Barra-style risk factor (``quantlab.factor.predefined.barra``), usable by
any factor. ``SigmaClip`` and ``RenormalizedCombine`` are elementwise, with
no time or symbol window; they are here because they are the per-bar steps
of that same pipeline (clipping a standardized value, combining
descriptors).
"""

from KunQuant.Op import Builder
from KunQuant.ops import *


def _zero_where_missing(value: OpBase) -> OpBase:
    """Return ``value`` with NaN and infinities replaced by 0."""
    return SetInfOrNanToValue(value, 0.0)


def _present(value: OpBase) -> OpBase:
    """Return 1 where ``value`` is finite and 0 where it is NaN or infinite."""
    return _zero_where_missing(value * 0.0 + 1.0)


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


#: C++ of the residual ops' shared step: the weighted means and centred
#: moments over the fit sample. ``WLS_TOLERANCE`` is the relative size below
#: which a regressor's centred spread (or the two regressors' determinant) is
#: taken as zero, so a constant or collinear regressor gives NaN, not a slope
#: made of rounding.
_WLS_MEANS = """
        const T WLS_TOLERANCE = (T)1e-10;
        T sw = 0, swy = 0, swx1 = 0, swx2 = 0, q1 = 0, q2 = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            if (!fit(i)) continue;
            T w = weight(i);
            sw += w; swy += w * input_0[i]; swx1 += w * x1(i); swx2 += w * x2(i);
            q1 += w * x1(i) * x1(i); q2 += w * x2(i) * x2(i);
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
    ``x`` does not vary over it (its centred spread below 1e-10 of its
    uncentred one), is NaN for every symbol.

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
        T b1 = s11 > WLS_TOLERANCE * q1 && s11 > 0 ? s1y / s11 : NAN, b2 = 0;
""" + _WLS_WRITE


class CrossSectionalWLSResidual2(GenericCrossSectionalOp):
    """Residual of a per-bar weighted least-squares fit of ``y`` on ``x1`` and ``x2``, with an intercept.

    ``CrossSectionalWLSResidual`` with two regressors: the slopes solve the
    weighted, mean-centred 2x2 normal equations over the fit sample (the
    symbols where ``universe > 0``, ``y``, ``x1`` and ``x2`` are finite and
    ``w`` is finite and positive), and every symbol with a finite ``y`` gets
    ``y - my - b1 (x1 - m1) - b2 (x2 - m2)``, a missing regressor taken at
    its mean. USE4 orthogonalizes Residual Volatility against Beta and Size
    this way. A bar where a regressor does not vary, or the two are
    collinear (the determinant below 1e-10 of the product of the spreads),
    over the fit sample is NaN for every symbol.

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
        bool solvable = s11 > WLS_TOLERANCE * q1 && s22 > WLS_TOLERANCE * q2
            && det > WLS_TOLERANCE * s11 * s22 && det > 0;
        T b1 = solvable ? (s22 * s1y - s12 * s2y) / det : NAN;
        T b2 = solvable ? (s11 * s2y - s12 * s1y) / det : NAN;
""" + _WLS_WRITE


#: Largest industry code ``CrossSectionalIndustrySizeFill`` accepts; it sizes
#: the per-industry accumulators.
_MAX_INDUSTRY_CODE = 4096


class CrossSectionalIndustrySizeFill(GenericCrossSectionalOp):
    """Fill a missing value from a per-bar weighted regression on industry and size.

    On each bar the fit uses the symbols where ``universe > 0``, ``y`` is
    finite, ``w`` is finite and positive, and the regressors used are
    finite (``industry`` an integer code from 0 to 4096; any other code
    counts as missing). An infinite ``y`` counts as missing. The model is one intercept
    per industry plus a common slope on ``size``, fitted by weighted least
    squares without a matrix solve (Frisch-Waugh-Lovell): ``y`` and ``size``
    are demeaned within each industry with the weights, the slope is
    ``b = sum(w dy ds) / sum(w ds**2)``, and an industry's intercept is
    ``mean_y - b mean_size`` over its fit members. Then::

        out = y                                    where y is finite
        out = intercept[industry] + b * size       where y is missing and eligible > 0

    a missing value whose industry has no fit member, or whose regressors
    are missing, stays NaN. USE4 imputes a style a stock lacks this way,
    from a regression on industry and size. Constructing it with
    ``use_industry=False`` fits one common intercept; with
    ``use_size=False`` it fits the industry means alone (the slope is 0,
    and also when ``size`` does not vary within the industries, below
    1e-10 of its spread). Each pair of flags gets a class of its own
    (``CrossSectionalIndustrySizeFill_industry_size``), whose C++ has the
    flags written in; ``isinstance`` still holds. The batch-start and
    SIMD-width notes of ``CrossSectionalZScore`` apply.

    Parameters
    ----------
    y : OpBase
        The values to fill, such as a style exposure.
    size : OpBase
        The size regressor, such as the Size exposure.
    industry : OpBase
        The industry code, a non-negative integer stored as a float.
    w : OpBase
        The regression weights, such as the square root of market cap.
    universe : OpBase
        A mask, positive for the symbols the fit uses.
    eligible : OpBase
        A mask, positive for the symbols a missing value may be filled for,
        such as the symbols with a market cap on the bar.
    use_industry, use_size : bool, default True
        Which regressors the model has; at least one.

    Raises
    ------
    ValueError
        If both flags are false.

    Examples
    --------
    >>> Output(CrossSectionalIndustrySizeFill(beta, size, Input("industry"), Sqrt(cap), estu, live), "beta")
    """

    _USE_INDUSTRY: bool = True
    _USE_SIZE: bool = True
    _variants: dict = {}

    def __new__(cls, y, size, industry, w, universe, eligible, use_industry=True, use_size=True):
        """Return an instance of the subclass specialized to the two flags."""
        if not (use_industry or use_size):
            raise ValueError("CrossSectionalIndustrySizeFill: needs industry, size or both")
        key = (bool(use_industry), bool(use_size))
        variant = CrossSectionalIndustrySizeFill._variants.get(key)
        if variant is None:
            tag = "_".join(name for name, used in zip(("industry", "size"), key) if used)
            variant = type(
                f"CrossSectionalIndustrySizeFill_{tag}",
                (CrossSectionalIndustrySizeFill,),
                {"_USE_INDUSTRY": key[0], "_USE_SIZE": key[1]},
            )
            variant.__module__ = CrossSectionalIndustrySizeFill.__module__
            CrossSectionalIndustrySizeFill._variants[key] = variant
        return super().__new__(variant)

    def __init__(self, y, size, industry, w, universe, eligible, use_industry=True, use_size=True):
        """Initialize the operator; see the class docstring for parameters."""
        super().__init__(
            [y, size, industry, w, universe, eligible],
            [("use_industry", bool(use_industry)), ("use_size", bool(use_size))],
        )

    def generate_head(self) -> str:
        """Return the C++ set-up: per-industry accumulators, sized on each bar."""
        return """
        std::vector<T> group_w, group_y, group_s;
        """

    def generate_body(self) -> str:
        """Return the C++ that fits one bar and fills its missing values."""
        industry = "(size_t)input_2[i]" if self._USE_INDUSTRY else "(size_t)0"
        industry_ok = (
            "std::isfinite(input_2[i]) && input_2[i] >= 0 && input_2[i] <= MAX_INDUSTRY"
            " && input_2[i] == std::floor(input_2[i])"
            if self._USE_INDUSTRY
            else "true"
        )
        size_ok = "std::isfinite(input_1[i])" if self._USE_SIZE else "true"
        size = "input_1[i]" if self._USE_SIZE else "(T)0"
        return f"""
        const T MAX_INDUSTRY = {_MAX_INDUSTRY_CODE};
        auto regressors_ok = [&](size_t i) -> bool {{ return ({industry_ok}) && ({size_ok}); }};
        auto fit = [&](size_t i) -> bool {{
            T w = input_3[i];
            return input_4[i] > 0 && std::isfinite(input_0[i]) && std::isfinite(w) && w > 0
                && regressors_ok(i);
        }};
        size_t groups = 1;
        for (size_t i = 0; i < num_stocks; i++) {{
            if (regressors_ok(i)) groups = std::max(groups, {industry} + 1);
        }}
        group_w.assign(groups, 0); group_y.assign(groups, 0); group_s.assign(groups, 0);
        for (size_t i = 0; i < num_stocks; i++) {{
            if (!fit(i)) continue;
            size_t g = {industry};
            T w = input_3[i];
            group_w[g] += w; group_y[g] += w * input_0[i]; group_s[g] += w * {size};
        }}
        for (size_t g = 0; g < groups; g++) {{
            if (group_w[g] > 0) {{ group_y[g] /= group_w[g]; group_s[g] /= group_w[g]; }}
        }}
        T sss = 0, ssy = 0, scale = 0;
        for (size_t i = 0; i < num_stocks; i++) {{
            if (!fit(i)) continue;
            size_t g = {industry};
            T w = input_3[i], ds = {size} - group_s[g], dy = input_0[i] - group_y[g];
            sss += w * ds * ds; ssy += w * ds * dy; scale += w * {size} * {size};
        }}
        T slope = sss > (T)1e-10 * scale && sss > 0 ? ssy / sss : 0;
        for (size_t i = 0; i < num_stocks; i++) {{
            T y = input_0[i];
            if (std::isfinite(y)) {{ output_0[i] = y; continue; }}
            output_0[i] = NAN;
            if (!(input_5[i] > 0) || !regressors_ok(i)) continue;
            size_t g = {industry};
            if (g >= groups || !(group_w[g] > 0)) continue;
            output_0[i] = group_y[g] + slope * ({size} - group_s[g]);
        }}
        """
