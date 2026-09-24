"""Pre-flight volume guard for SQL-backed sources such as WRDS.

The REST-oriented guard in ``quantlab.universe`` prices a fetch in vendor
requests per minute. A PostgreSQL pull has no request quota; the real risk
is disk. ``SqlVolumeGuard`` therefore prices a fetch from real row counts
per bucket, as returned by a ``count(*)`` probe, and refuses above a byte
ceiling and a row ceiling unless the caller forces it. A refusal names the
longest date prefix that would fit, so a large backfill can be run as several
guarded segments.

This module imports only the standard library: no acquisition class and no
database client, so a refusal costs nothing beyond the counting probe that
produced its input. The counts come from the WRDS probes
(``WrdsNbboVolumeProbe.count_rows_by_day`` for TAQ, where a bucket is a
trading day, and ``CrspVolumeProbe.count_rows_by_year`` for CRSP, where a
bucket is a calendar year); the ``unit`` keyword only changes the word a
refusal uses for a bucket.
"""

from collections.abc import Mapping

_GIB = 1024**3


class SqlVolumeGuard:
    """Row and byte ceilings over per-bucket ``count(*)`` row counts.

    The three knobs (``max_raw_bytes``, ``max_raw_rows``, ``bytes_per_row``)
    are resolved once at construction from a kwargs mapping, normally a
    config's ``kwargs``, over the class defaults. ``force=True`` on
    ``assert_acquisition_volume_fits`` skips the refusal but never the
    arithmetic, and it is a per-call argument; no environment variable or
    config key disables the guard wholesale.

    Example:
        >>> from quantlab.acquisition._support.sql_volume import SqlVolumeGuard
        >>> guard = SqlVolumeGuard()
        >>> guard.max_raw_rows, guard.bytes_per_row
        (700000000, 30)
        >>> SqlVolumeGuard({"max_raw_rows": 1_000_000}).max_raw_rows
        1000000
    """

    #: Raw-byte ceiling for one fetch: 20 GiB, kept equal to the universe
    #: guard's ``MAX_RAW_BYTES`` so both acquisition paths share one disk
    #: budget. The equality is checked by a test rather than an import, since
    #: this module must not import the acquisition layer.
    MAX_RAW_BYTES = 20 * 1024**3

    #: Bytes one raw NBBO row is assumed to cost on disk. An assumption, not a
    #: measurement: synthetic NBBO parquet measured about 13 bytes per row,
    #: but real shards carry 16 columns, so 30 is a conservative working
    #: value.
    DEFAULT_BYTES_PER_ROW = 30

    #: Row ceiling for one fetch, about 20 GiB at the assumed bytes per row.
    #: Independent of the byte ceiling, so lowering ``bytes_per_row`` alone
    #: cannot admit a pull the row count says is too large.
    MAX_RAW_ROWS = 700_000_000

    #: ``(label, estimate key, class constant, kwargs key)`` per ceiling, in
    #: the order a refusal reports them, so the constant a reader is told to
    #: edit and the kwargs key they are told to pass cannot drift from the
    #: value being checked.
    CEILINGS: tuple[tuple[str, str, str, str], ...] = (
        ("raw-bytes", "raw_bytes", "MAX_RAW_BYTES", "max_raw_bytes"),
        ("raw-rows", "rows", "MAX_RAW_ROWS", "max_raw_rows"),
    )

    def __init__(self, kwargs: Mapping | None = None):
        """Resolve the three knobs from ``kwargs`` over the class defaults."""
        kwargs = {} if kwargs is None else kwargs
        self.max_raw_bytes = self._positive(
            kwargs, "max_raw_bytes", self.MAX_RAW_BYTES
        )
        self.max_raw_rows = self._positive(kwargs, "max_raw_rows", self.MAX_RAW_ROWS)
        self.bytes_per_row = self._positive(
            kwargs, "bytes_per_row", self.DEFAULT_BYTES_PER_ROW
        )

    @staticmethod
    def _positive(kwargs: Mapping, key: str, default):
        """Read knob ``key`` from ``kwargs``, requiring a positive number.

        Zero would refuse everything (or divide by zero), a negative value is
        meaningless, and a bool or str is a config typo that must not be
        coerced into a ceiling.
        """
        value = kwargs.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not value > 0
        ):
            raise ValueError(
                f"SqlVolumeGuard knob {key!r} must be a positive number, got "
                f"{value!r}."
            )
        return value

    def _limits(self) -> dict[str, float]:
        """Return the two ceilings keyed by their kwargs names."""
        return {"max_raw_bytes": self.max_raw_bytes, "max_raw_rows": self.max_raw_rows}

    def _crossed(self, rows: int, raw_bytes: float) -> list[str]:
        """Return the labels of every ceiling that ``rows``/``raw_bytes`` exceed."""
        actual = {"rows": rows, "raw_bytes": raw_bytes}
        limits = self._limits()
        return [
            label
            for label, key, _constant, keyword in self.CEILINGS
            if actual[key] > limits[keyword]
        ]

    #: The word a message uses for one key of ``rows_by_day``, placed in front
    #: of ``(s)``. The default is TAQ's, where a bucket is a trading day.
    DEFAULT_UNIT = "trading day"

    def estimate(
        self,
        rows_by_day: Mapping[str, int],
        *,
        symbols: int,
        start_date: str,
        end_date: str,
        unit: str = DEFAULT_UNIT,
    ) -> dict:
        """Price a pull from its per-bucket row counts, without refusing.

        Pure arithmetic. ``fitting_end_date`` is the last bucket of the
        longest ascending prefix of ``rows_by_day`` whose cumulative rows and
        bytes stay within both ceilings, or ``None`` when not even the first
        bucket fits. Keys are ISO dates, so a string sort is a date sort.

        Args:
            rows_by_day: ``{iso_date: row_count}``, one entry per bucket.
            symbols: How many symbols the counts cover; echoed into the result.
            start_date: Start of the requested window; echoed into the result.
            end_date: End of the requested window; echoed into the result.
            unit: The word a message uses for one bucket (``"trading day"``
                for TAQ, ``"calendar year"`` for CRSP). It changes wording
                only; the returned key is always ``trading_days``.

        Returns:
            A dict with the inputs echoed, ``trading_days`` (bucket count),
            ``rows``, ``bytes_per_row``, ``raw_bytes``, both ceilings,
            ``crossed`` (labels of exceeded ceilings), ``fitting_end_date``
            and ``forced`` (always ``False`` here).

        Example:
            >>> guard = SqlVolumeGuard({"max_raw_rows": 2_000_000})
            >>> estimate = guard.estimate(
            ...     {"2024-01-02": 1_200_000, "2024-01-03": 1_300_000},
            ...     symbols=1, start_date="2024-01-02", end_date="2024-01-03",
            ... )
            >>> estimate["rows"], estimate["crossed"], estimate["fitting_end_date"]
            (2500000, ['raw-rows'], '2024-01-02')
        """
        days = sorted(rows_by_day)
        rows = 0
        fitting_end_date = None
        for day in days:
            rows += int(rows_by_day[day])
            if not self._crossed(rows, rows * self.bytes_per_row):
                fitting_end_date = day
            else:
                break
        rows = sum(int(rows_by_day[day]) for day in days)
        raw_bytes = rows * self.bytes_per_row
        return {
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
            "trading_days": len(days),
            "unit": unit,
            "rows": rows,
            "bytes_per_row": self.bytes_per_row,
            "raw_bytes": raw_bytes,
            "max_raw_bytes": self.max_raw_bytes,
            "max_raw_rows": self.max_raw_rows,
            "crossed": self._crossed(rows, raw_bytes),
            "fitting_end_date": fitting_end_date,
            "forced": False,
        }

    def assert_acquisition_volume_fits(
        self,
        rows_by_day: Mapping[str, int],
        *,
        symbols: int,
        start_date: str,
        end_date: str,
        force: bool = False,
        unit: str = DEFAULT_UNIT,
    ) -> dict:
        """Return the estimate, or raise if the pull crosses either ceiling.

        With ``force=True`` an over-ceiling estimate is returned with
        ``forced=True`` and ``crossed`` still naming every crossed ceiling.
        Never touches the network.

        Args:
            rows_by_day: ``{iso_date: row_count}``, one entry per bucket.
            symbols: How many symbols the counts cover.
            start_date: Start of the requested window.
            end_date: End of the requested window.
            force: Return the estimate instead of raising when over a ceiling.
            unit: The word a message uses for one bucket; see ``estimate``.

        Returns:
            The dict ``estimate`` returns, with ``forced`` set when ``force``
            overrode a refusal.

        Raises:
            ValueError: If any ceiling is crossed and ``force`` is false. The
                message names every crossed ceiling with the constant and
                kwargs key that raise it, a date segment that would fit (or
                that none does), and the ``--force-volume`` override.

        Example:
            >>> guard = SqlVolumeGuard({"max_raw_rows": 2_000_000})
            >>> guard.assert_acquisition_volume_fits(
            ...     {"2024-01-02": 1_200_000}, symbols=1,
            ...     start_date="2024-01-02", end_date="2024-01-02",
            ... )["crossed"]
            []
            >>> guard.assert_acquisition_volume_fits(
            ...     {"2024-01-02": 1_200_000, "2024-01-03": 1_300_000},
            ...     symbols=1, start_date="2024-01-02", end_date="2024-01-03",
            ... )
            Traceback (most recent call last):
            ...
            ValueError: Refusing to pull 1 symbol(s) over 2024-01-02..2024-01-03: ...
        """
        estimate = self.estimate(
            rows_by_day,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            unit=unit,
        )
        crossed = estimate["crossed"]
        if not crossed:
            return estimate
        if force:
            estimate["forced"] = True
            return estimate

        rendered = {
            "raw-bytes": (
                f"{estimate['raw_bytes'] / _GIB:.2f} GiB > "
                f"{self.max_raw_bytes / _GIB:.2f} GiB"
            ),
            "raw-rows": f"{estimate['rows']:,} > {self.max_raw_rows:,} rows",
        }
        by_label = {label: (constant, keyword) for label, _k, constant, keyword in self.CEILINGS}
        reasons = "; ".join(
            f"over the {label} ceiling ({rendered[label]}; raise "
            f"{by_label[label][0]} or the {by_label[label][1]!r} kwargs key)"
            for label in crossed
        )
        fitting = estimate["fitting_end_date"]
        if fitting is None:
            cure = (
                "No date segment fits: no single day fits under the ceilings, "
                "so narrow --symbols/--universe instead."
            )
        else:
            cure = (
                f"A date segment that fits: --start-date {start_date} "
                f"--end-date {fitting}; run the rest as further guarded "
                f"segments (D-24)."
            )
        raise ValueError(
            f"Refusing to pull {estimate['symbols']:,} symbol(s) over "
            f"{start_date}..{end_date}: {estimate['trading_days']:,} "
            f"{unit}(s), {estimate['rows']:,} row(s) (counted with count(*)) x "
            f"{estimate['bytes_per_row']} B/row = "
            f"{estimate['raw_bytes'] / _GIB:.2f} GiB. {reasons}. {cure} Or "
            f"pass --force-volume to proceed anyway."
        )
