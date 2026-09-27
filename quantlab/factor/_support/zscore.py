"""The z-score window of factor sets normalized along time.

``Alpha101SpotKline`` and ``Alpha158SpotKline`` wrap every output in
``WindowedZScore``. Its window is the factor's own parameter,
``kwargs["zscore_window"]``, independent of ``warmup_bars``: a rolling
window nested inside another needs the sum of both lengths before its first
full value, so the warm-up must cover the alpha's lookback plus the z-score
window.
"""

from quantlab.base.factor import FactorKunQuant


class TimeSeriesZScoredFactor(FactorKunQuant):
    """A KunQuant factor whose outputs are z-scored over ``zscore_window`` bars.

    Subclasses wrap each output in ``WindowedZScore(..., self.zscore_window)``.
    The window is ``config.kwargs["zscore_window"]``, or
    ``DEFAULT_ZSCORE_WINDOW`` when unset.

    Examples
    --------
    >>> factor = Alpha158SpotKline(FactorConfig(
    ...     warmup_bars=24, dataset=dataset, mode="batch",
    ...     data_columns=["close"], factor_names=["STD5"],
    ...     kwargs={"zscore_window": 20}, file_path="std5.zarr",
    ... ))
    >>> factor.zscore_window
    20
    """

    #: Bars the z-score uses when ``kwargs["zscore_window"]`` is unset.
    DEFAULT_ZSCORE_WINDOW = 20

    @property
    def zscore_window(self) -> int:
        """Bars each symbol's outputs are z-scored against.

        Examples
        --------
        >>> factor.zscore_window     # kwargs without "zscore_window"
        20
        """
        return (self.config.kwargs or {}).get(
            "zscore_window", self.DEFAULT_ZSCORE_WINDOW
        )

    def _validate_config(self) -> None:
        """Refuse a ``zscore_window`` that is not a positive integer."""
        window = self.zscore_window
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(
                f"{self.class_name}: kwargs['zscore_window'] must be a positive "
                f"integer number of bars, got {window!r}."
            )
