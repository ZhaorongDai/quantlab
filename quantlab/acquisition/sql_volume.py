"""The SQL-shaped pre-flight volume guard for WRDS pulls (D-16, D-24).

`UniverseCatalog.assert_acquisition_volume_fits` prices a REST fetch in vendor
requests per minute. That model does not apply to a PostgreSQL pull: there is
no request quota, and the real risk is disk. So this guard prices a fetch from
REAL row counts per trading day, the output of a `count(*)` probe, and refuses
above a byte ceiling and a row ceiling unless the caller forces it.

**A leaf by construction.** This module imports only the standard library. It
imports no acquisition class and no database client, so a refusal costs zero
WRDS queries beyond the counting probe that produced its input: there is
nothing here that could open a connection, whatever the call order. The
counts come from `WrdsNbboVolumeProbe` (plan 04), which counts per symbol batch
and never over a whole daily table (D-24).

**Segmented backfills (D-24).** At the calibrated 2024 volumes (AAPL ~1.24M
NBBO rows/day, full market ~313M rows/day) the default 20 GiB ceiling admits
only a few days of an S&P 500 pull. Large backfills therefore run as several
guarded date segments, and a refusal names the longest prefix of the
requested trading days that WOULD fit, so the next command is concrete.
"""

from collections.abc import Mapping

_GIB = 1024**3


class SqlVolumeGuard:
    """Row and byte ceilings over per-day `count(*)` row counts.

    Knobs are resolved once, at construction, from a kwargs mapping (normally
    a config's `kwargs`) over the class defaults. `force=True` on
    `assert_acquisition_volume_fits` is the blunt override: it skips the
    RAISE and never the arithmetic, and it is an explicit per-run parameter --
    there is no environment variable and no config key that disables the
    guard wholesale (T-03.9-08).
    """

    #: Raw-byte ceiling for one fetch (D-24): 20 GiB, deliberately EQUAL to
    #: `UniverseCatalog.MAX_RAW_BYTES` so both acquisition paths share one disk
    #: budget. The equality is pinned by a test rather than by an import,
    #: because this module must not import the acquisition layer.
    MAX_RAW_BYTES = 20 * 1024**3

    #: Bytes a raw NBBO row costs on disk. An ASSUMPTION, not a measurement:
    #: synthetic NBBO parquet measured 13.1 B/row, but real shards carry 16
    #: columns, so 30 is a conservative working value until the plan-08 live
    #: smoke measures real shard bytes/row.
    DEFAULT_BYTES_PER_ROW = 30

    #: Row ceiling for one fetch: about 20 GiB at the assumed bytes/row, and
    #: INDEPENDENT of the byte ceiling so lowering `bytes_per_row` alone cannot
    #: admit a pull the row count says is too large.
    MAX_RAW_ROWS = 700_000_000

    #: `(label, estimate key, class constant, keyword)` per ceiling, in the
    #: order a refusal reports them -- the universe guard's table style, so the
    #: constant a reader is told to edit and the kwargs key they are told to
    #: pass cannot drift apart from the value being checked.
    CEILINGS: tuple[tuple[str, str, str, str], ...] = (
        ("raw-bytes", "raw_bytes", "MAX_RAW_BYTES", "max_raw_bytes"),
        ("raw-rows", "rows", "MAX_RAW_ROWS", "max_raw_rows"),
    )

    def __init__(self, kwargs: Mapping | None = None):
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
        """A knob must be a positive number: zero would silently refuse
        everything (or divide by zero), a negative one is meaningless, and a
        bool/str is a config typo that must not be coerced into a ceiling."""
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
        return {"max_raw_bytes": self.max_raw_bytes, "max_raw_rows": self.max_raw_rows}

    def _crossed(self, rows: int, raw_bytes: float) -> list[str]:
        actual = {"rows": rows, "raw_bytes": raw_bytes}
        limits = self._limits()
        return [
            label
            for label, key, _constant, keyword in self.CEILINGS
            if actual[key] > limits[keyword]
        ]

    def estimate(
        self,
        rows_by_day: Mapping[str, int],
        *,
        symbols: int,
        start_date: str,
        end_date: str,
    ) -> dict:
        """Price a pull from its per-trading-day row counts. Pure arithmetic.

        `fitting_end_date` is the last day of the longest ASCENDING prefix of
        `rows_by_day` whose cumulative rows and bytes stay within both
        ceilings, or `None` when not even the first day fits. Day keys are
        ISO dates, so a string sort is a date sort.
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
    ) -> dict:
        """Raise if the pull is over either ceiling; otherwise return the
        estimate.

        With `force=True` an over-ceiling estimate is returned with
        `forced=True` and `crossed` still naming every crossed ceiling. A
        refusal names every crossed ceiling (not only the first), the
        constant AND the kwargs key that raise it, a date segment that fits,
        and `--force-volume`. Never touches the network.
        """
        estimate = self.estimate(
            rows_by_day, symbols=symbols, start_date=start_date, end_date=end_date
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
            f"{start_date}..{end_date}: {estimate['trading_days']:,} trading "
            f"day(s), {estimate['rows']:,} row(s) (counted with count(*)) x "
            f"{estimate['bytes_per_row']} B/row = "
            f"{estimate['raw_bytes'] / _GIB:.2f} GiB. {reasons}. {cure} Or "
            f"pass --force-volume to proceed anyway."
        )
