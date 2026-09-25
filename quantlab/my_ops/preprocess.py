"""Custom KunQuant operators for factor normalization.

``WindowedZScore`` standardizes each symbol against its own trailing window
(time-series); ``CrossSectionalZScore`` standardizes each timestamp across
all symbols (cross-sectional). Which axis to normalize on is a property of
the strategy consuming the factor, so the two are alternatives, not
interchangeable implementations.
"""

from KunQuant.Op import Builder
from KunQuant.ops import *


class WindowedZScore(WindowedCompositiveOp):
    """Rolling z-score along time, ``(x - rolling_mean) / rolling_std``.

    Each symbol is standardized against its own trailing ``window`` bars,
    which is a time-series normalization; nothing is computed across
    symbols. No missing-value handling is applied: NaN inputs propagate, and
    the first ``window - 1`` bars are NaN until the window is full, as with
    every KunQuant rolling operator. Fill values in the caller if you need
    them.

    Example:
        >>> Output(WindowedZScore(alpha(all_data), 20), "alpha001")
    """

    # `options` is required by KunQuant's CompositiveOp interface.
    def decompose(self, options: dict) -> list[OpBase]:
        """Expand into ``WindowedAvg``, ``WindowedStddev``, ``Sub`` and ``Div``.

        Args:
            options: Decomposition options passed by KunQuant; unused.

        Example:
            KunQuant calls this while compiling; it can also be called
            directly on an op built inside a ``Builder``:

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

    At every timestamp the NaN-aware mean and sample standard deviation
    (``ddof=1``, matching pandas ``.std()`` and KunQuant's
    ``WindowedStddev``) are taken over all symbols. NaN inputs stay NaN, and
    a row with fewer than two valid values or zero standard deviation is NaN
    throughout. No fill is applied.

    It is a ``GenericCrossSectionalOp`` with a hand-written C++ body because a
    ``CompositiveOp`` can only decompose into time-series operators. Three
    KunQuant constraints follow. The body must not depend on the op's
    attributes, since KunQuant deduplicates generated C++ functions by class
    name and layout only; a parameterized variant needs a class of its own.
    Batch runs must start at bar 0: in KunQuant 0.1.11 a non-zero ``start``
    gives wrong results for every ``GenericCrossSectionalOp``. And the number
    of symbols must be a multiple of the SIMD block width on the host, in
    both the ``TS`` and ``STREAM`` layouts.

    This op and ``WindowedZScore`` normalize along different axes; which one
    a factor uses is a strategy decision, and no factor class applies this
    one by default.

    Example:
        >>> Output(CrossSectionalZScore(alpha(all_data)), "alpha001_cs")
    """

    def __init__(self, v: OpBase) -> None:
        """Wrap a single input op."""
        super().__init__([v], None)

    def generate_head(self) -> str:
        """Return no per-function preamble.

        Example:
            >>> CrossSectionalZScore(Input("close")).generate_head()
            ''
        """
        return ""

    def generate_body(self) -> str:
        """Return the C++ loop that z-scores ``input_0`` into ``output_0``.

        KunQuant calls this when it emits the C++ for the graph.

        Example:
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
