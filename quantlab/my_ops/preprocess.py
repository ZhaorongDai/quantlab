"""Custom KunQuant operators that normalize a factor.

KunQuant is the library this project uses to compute most factors. A factor
formula is written as a graph of operators (``WindowedAvg``, ``Rank``, ...),
and KunQuant compiles that graph to native C++ code that runs over a whole
``(timestamp, symbol)`` array at once. This module adds two operators to
that vocabulary.

``WindowedZScore`` is a *time-series* normalization: each symbol is compared
with its own recent past. ``CrossSectionalZScore`` is a *cross-sectional*
normalization: at each timestamp, each symbol is compared with all the other
symbols on the same bar. Which one is right depends on the strategy that
consumes the factor. A strategy that trades one asset over time wants the
first; a strategy that ranks many assets against each other wants the
second. They are not two implementations of the same thing.
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
