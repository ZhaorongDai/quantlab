"""Check, before downloading, that a SQL-based data pull will fit on disk.

WRDS (Wharton Research Data Services) is a research data provider accessed
through a PostgreSQL database rather than a web API. The volume check in
``quantlab.universe``, written for web APIs, estimates a download in requests
per minute. A SQL pull has no request quota; the real risk is filling the
disk. ``SqlVolumeGuard`` therefore estimates a pull from actual row counts
per *bucket* (a trading day or a calendar year), as returned by a
``count(*)`` query, and refuses when the pull would exceed a byte limit or a
row limit, unless the caller forces it. A refusal names the longest leading
date range that would fit, so a large backfill can be run as several smaller
checked pieces.

This module imports only the standard library, with no acquisition class
and no database client, so a refusal costs nothing beyond the counting query
that produced its input. The counts come from the WRDS counting helpers:
``WrdsNbboVolumeProbe.count_rows_by_day`` for TAQ (the NYSE Trade and Quote
database; NBBO is the national best bid and offer, the best quoted prices
across all US exchanges), where a bucket is a trading day, and
``CrspVolumeProbe.count_rows_by_year`` for CRSP (the Center for Research in
Security Prices daily stock file), where a bucket is a calendar year. The
``unit`` keyword only changes the word a refusal uses for a bucket.
"""

from collections.abc import Mapping

_GIB = 1024**3


class SqlVolumeGuard:
    """Enforce row and byte limits on a pull, given its per-bucket row counts.

    The three settings (``max_raw_bytes``, ``max_raw_rows`` and
    ``bytes_per_row``) are read once at construction from a mapping,
    normally a config's ``kwargs``, falling back to the class defaults.
    ``force=True`` on ``assert_acquisition_volume_fits`` skips the refusal
    but still computes the estimate. It is a per-call argument; no
    environment variable or config key turns the check off entirely.

    Parameters
    ----------
    kwargs : Mapping or None, default None
        Optional overrides for ``max_raw_bytes``, ``max_raw_rows`` and
        ``bytes_per_row``. Each must be a positive number.

    Attributes
    ----------
    max_raw_bytes : int or float
        Byte limit for one pull.
    max_raw_rows : int or float
        Row limit for one pull.
    bytes_per_row : int or float
        Assumed on-disk size of one row.

    Raises
    ------
    ValueError
        If a setting is not a positive number.

    Examples
    --------
    >>> from quantlab.acquisition._support.sql_volume import SqlVolumeGuard
    >>> guard = SqlVolumeGuard()
    >>> guard.max_raw_rows, guard.bytes_per_row
    (700000000, 30)
    >>> SqlVolumeGuard({"max_raw_rows": 1_000_000}).max_raw_rows
    1000000
    """

    #: Raw-byte limit for one pull: 20 GiB, kept equal to
    #: ``UniverseCatalog.MAX_RAW_BYTES`` so both download paths share one disk
    #: budget. A test checks the two are equal, because this module must not
    #: import the acquisition layer.
    MAX_RAW_BYTES = 20 * 1024**3

    #: Bytes one raw NBBO row is assumed to take on disk. An assumption, not a
    #: measurement: synthetic NBBO parquet measured about 13 bytes per row,
    #: but real files carry 16 columns, so 30 is a cautious estimate.
    DEFAULT_BYTES_PER_ROW = 30

    #: Row limit for one pull, about 20 GiB at the assumed bytes per row. It
    #: is checked separately from the byte limit, so lowering
    #: ``bytes_per_row`` alone cannot let through a pull with too many rows.
    MAX_RAW_ROWS = 700_000_000

    #: ``(label, estimate key, class constant, kwargs key)`` per limit, in the
    #: order a refusal reports them. Keeping them together ensures the
    #: constant and kwargs key named in a refusal are the ones being checked.
    CEILINGS: tuple[tuple[str, str, str, str], ...] = (
        ("raw-bytes", "raw_bytes", "MAX_RAW_BYTES", "max_raw_bytes"),
        ("raw-rows", "rows", "MAX_RAW_ROWS", "max_raw_rows"),
    )

    def __init__(self, kwargs: Mapping | None = None):
        """Initialize the guard; see the class docstring for parameters."""
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
        """Read setting ``key`` from ``kwargs``, requiring a positive number.

        Zero would refuse everything (or divide by zero), a negative value is
        meaningless, and a bool or string is a config typo that must not be
        turned into a limit.
        """
        value = kwargs.get(key, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not value > 0
        ):
            raise ValueError(
                f"SqlVolumeGuard setting {key!r} must be a positive number, got "
                f"{value!r}."
            )
        return value

    def _limits(self) -> dict[str, float]:
        """Return the two limits keyed by their kwargs names."""
        return {"max_raw_bytes": self.max_raw_bytes, "max_raw_rows": self.max_raw_rows}

    def _crossed(self, rows: int, raw_bytes: float) -> list[str]:
        """Return the labels of every limit that ``rows`` or ``raw_bytes`` exceeds."""
        actual = {"rows": rows, "raw_bytes": raw_bytes}
        limits = self._limits()
        return [
            label
            for label, key, _constant, keyword in self.CEILINGS
            if actual[key] > limits[keyword]
        ]

    #: The word a message uses for one key of ``rows_by_day``, followed by
    #: ``(s)``. The default suits TAQ, where a bucket is a trading day.
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
        """Estimate a pull's size from its per-bucket row counts, without refusing.

        Only arithmetic; nothing is read or fetched. ``fitting_end_date`` is
        the last bucket of the longest run of buckets, starting from the
        earliest, whose running totals of rows and bytes stay within both
        limits. It is ``None`` when not even the first bucket fits. Keys are
        ISO dates, so sorting them as text sorts them by date.

        Parameters
        ----------
        rows_by_day : Mapping[str, int]
            ``{iso_date: row_count}``, one entry per bucket.
        symbols : int
            How many symbols the counts cover; echoed into the result.
        start_date : str
            Start of the requested window; echoed into the result.
        end_date : str
            End of the requested window; echoed into the result.
        unit : str, default "trading day"
            The word a message uses for one bucket (``"trading day"``
            for TAQ, ``"calendar year"`` for CRSP). It changes wording
            only; the returned key is always ``trading_days``.

        Returns
        -------
        dict
            A dict with the inputs echoed, ``trading_days`` (bucket count),
            ``rows``, ``bytes_per_row``, ``raw_bytes``, both limits,
            ``crossed`` (labels of exceeded limits), ``fitting_end_date``
            and ``forced`` (always ``False`` here).

        Examples
        --------
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
        """Return the estimate, or raise if the pull exceeds either limit.

        With ``force=True`` an over-limit estimate is returned with
        ``forced=True``, and ``crossed`` still names every exceeded limit.
        Never touches the network.

        Parameters
        ----------
        rows_by_day : Mapping[str, int]
            ``{iso_date: row_count}``, one entry per bucket.
        symbols : int
            How many symbols the counts cover.
        start_date : str
            Start of the requested window.
        end_date : str
            End of the requested window.
        force : bool, default False
            Return the estimate instead of raising when over a limit.
        unit : str, default "trading day"
            The word a message uses for one bucket; see ``estimate``.

        Returns
        -------
        dict
            The dict ``estimate`` returns, with ``forced`` set when ``force``
            overrode a refusal.

        Raises
        ------
        ValueError
            If any limit is exceeded and ``force`` is false. The message
            names every exceeded limit with the constant and kwargs key
            that raise it, a date range that would fit (or says none
            does), and the ``--force-volume`` override.

        Examples
        --------
        >>> guard = SqlVolumeGuard({"max_raw_rows": 2_000_000})
        >>> guard.assert_acquisition_volume_fits(
        ...     {"2024-01-02": 1_200_000}, symbols=1,
        ...     start_date="2024-01-02", end_date="2024-01-02",
        ... )["crossed"]
        []
        >>> guard.assert_acquisition_volume_fits(
        ...     {"2024-01-02": 1_200_000, "2024-01-03": 1_300_000},
        ...     symbols=1, start_date="2024-01-02", end_date="2024-01-03",
        ... )  # doctest: +ELLIPSIS
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
                f"segments, each checked the same way."
            )
        raise ValueError(
            f"Refusing to pull {estimate['symbols']:,} symbol(s) over "
            f"{start_date}..{end_date}: {estimate['trading_days']:,} "
            f"{unit}(s), {estimate['rows']:,} row(s) (counted with count(*)) x "
            f"{estimate['bytes_per_row']} B/row = "
            f"{estimate['raw_bytes'] / _GIB:.2f} GiB. {reasons}. {cure} Or "
            f"pass --force-volume to proceed anyway."
        )
