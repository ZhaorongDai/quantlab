"""Shared argparse helpers for the ingest entry points.

The scripts under ``scripts/`` (``ingest_tiingo.py``, ``ingest_us_equity.py``,
``ingest_alpaca.py`` and the WRDS shells) share most of their command line:
``--symbols`` / ``--universe`` / ``--as-of-date``, the date window, ``--chunk``,
``--to-zarr``, ``--data-dir`` and the pre-flight volume guard flags. This module
defines each of those groups once, resolves the symbol roster the flags select,
and renders the value objects (volume estimates, conversion results) the scripts
print. Each script adds only the flags that are its own.

The module is deliberately light at import time. Its only module-scope project
imports are ``quantlab.base.chunking`` and ``quantlab.base.data``, and both are
there so that a ``choices`` list is derived from the constant that defines it
(``TimeChunkPlanner.GRANULARITIES`` for ``--chunk``,
``BaseDataset.NEW_LISTING_STRATEGIES`` for ``--on-new-listing``). Everything
heavier (``quantlab.config``, ``quantlab.universe``, the SQL volume guard) is
imported inside the function that needs it, so importing this module does not
pull in the dataset layer. Nothing here constructs an acquisition client or
issues a request.

One thing is intentionally not unified: ``resolve_symbols`` takes a keyword-only
``mode`` with no default, because point-in-time membership on one day
(``"as_of"``) and interval overlap across a window (``"in_range"``) are
different rosters, and a silent default would quietly pick one of them.
"""

import argparse
from typing import Literal

from quantlab.base.chunking import TimeChunkPlanner
from quantlab.base.data import BaseDataset

#: Maps each ``--universe`` choice to a universe category name. The ``choices``
#: of ``--universe`` are derived from this map, so a category added here is
#: selectable from every script that offers the flag.
UNIVERSE_CATEGORY_MAP = {
    "sp500": "sp500_constituent",
    "nasdaq100": "nasdaq100_constituent",
    "nasdaq_all": "nasdaq_all",
    "us_all": "us_all",
}

#: The two meanings a ``--start-date`` / ``--end-date`` window can have, and
#: the help text each one carries. ``"request-range"`` is a per-symbol request
#: range handed to the vendor. ``"interval-overlap"`` additionally filters the
#: roster to every symbol that traded at any point inside the window.
WindowSemantics = Literal["request-range", "interval-overlap"]

_WINDOW_HELP = {
    "request-range": {
        "start": "Start date (inclusive), e.g. 2024-01-01.",
        "end": "End date (inclusive), e.g. 2024-12-31.",
    },
    "interval-overlap": {
        "start": (
            "Window start (inclusive), default {default}. Applied "
            "as interval overlap: every symbol that traded at any point in "
            "the window is kept, including those that delisted inside it. "
            "Only symbols whose listing ended before this date are dropped."
        ),
        "end": "Window end (inclusive), default today.",
    },
}

#: The two roster-resolution modes ``resolve_symbols`` accepts. Neither is a
#: default; see that function.
RosterMode = Literal["as_of", "in_range"]
ROSTER_MODES: tuple[str, ...] = ("as_of", "in_range")


def add_window_args(
    parser: argparse.ArgumentParser,
    *,
    default_start_date: str | None = None,
    semantics: WindowSemantics = "request-range",
) -> argparse.ArgumentParser:
    """Add ``--start-date`` and ``--end-date`` to ``parser``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.
    default_start_date : str | None
        Default for ``--start-date``; ``None`` leaves it
        unset.
    semantics : WindowSemantics
        Which help text the two flags carry. ``"request-range"``
        describes a per-symbol request range; ``"interval-overlap"``
        describes a window that also filters the roster, which is the
        less obvious of the two and gets the longer text.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> import argparse
    >>> parser = add_window_args(
    ...     argparse.ArgumentParser(), default_start_date="2016-01-01"
    ... )
    >>> parser.parse_args(["--end-date", "2024-12-31"])
    Namespace(start_date='2016-01-01', end_date='2024-12-31')
    """
    help_text = _WINDOW_HELP[semantics]
    parser.add_argument(
        "--start-date",
        type=str,
        default=default_start_date,
        help=help_text["start"].format(default=default_start_date),
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help=help_text["end"],
    )
    return parser


def add_universe_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--symbols``, ``--universe`` and ``--as-of-date`` to ``parser``.

    The ``--universe`` choices are derived from ``UNIVERSE_CATEGORY_MAP``. The
    help text spells out the domain distinctions users trip over, such as the
    Nasdaq-100 index versus the full NASDAQ roster.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_universe_args(argparse.ArgumentParser())
    >>> parser.parse_args(["--universe", "sp500", "--as-of-date", "2024-06-28"])
    Namespace(symbols=None, universe='sp500', as_of_date='2024-06-28')
    """
    parser.add_argument(
        "--symbols",
        type=str,
        required=False,
        default=None,
        help=(
            "Comma-separated tickers (e.g. AAPL,MSFT). Mutually exclusive "
            "with --universe. This is the ticker-side entry point: a CRSP "
            "panel's symbol axis is the int64 PERMNO, so a CRSP conversion "
            "takes --permnos instead, and the ticker a PERMNO wore on a "
            "given day is read from the '.crsp_tickers.json' sidecar beside "
            "the store."
        ),
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=sorted(UNIVERSE_CATEGORY_MAP),
        default=None,
        help=(
            "Resolve a symbol list from the persisted universe table "
            "instead of --symbols. 'sp500' resolves point-in-time S&P 500 "
            "constituent membership; 'nasdaq100' resolves point-in-time "
            "Nasdaq-100 (NDX) index membership; 'nasdaq_all' resolves the "
            "full NASDAQ-listed Common Stock roster (current + delisted); "
            "'us_all' resolves the full US listed-equity roster -- NYSE + "
            "NASDAQ + AMEX common stock, delisted included (~15.4k tickers). "
            "Note that 'nasdaq100' and 'nasdaq_all' are different universes "
            "that merely share the word Nasdaq: the former is the ~100-name "
            "index, the latter every symbol ever listed on the exchange. "
            "'us_all' is a strict superset of 'nasdaq_all'; both are kept "
            "deliberately. Requires --as-of-date. For a full-window backfill "
            "of every symbol that traded at any point in a date range "
            "(rather than membership on one day), use ingest_us_equity.py, "
            "which queries by interval overlap instead."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help=(
            "Required with --universe; point-in-time date (YYYY-MM-DD) to "
            "resolve membership as of."
        ),
    )
    return parser


def add_to_zarr_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add ``--to-zarr``, the opt-in flag that gates raw-to-Zarr conversion.

    The flag is off by default on every ingest script, and a run without it
    stops at the raw shards and says so. The conversion it enables is the
    chunked, resumable one that ``--chunk`` governs.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_to_zarr_arg(argparse.ArgumentParser())
    >>> parser.parse_args(["--to-zarr"])
    Namespace(to_zarr=True)
    """
    parser.add_argument(
        "--to-zarr",
        action="store_true",
        help=(
            "After acquisition, convert the raw parquet into the Zarr store. "
            "The conversion densifies and appends one --chunk window at a "
            "time, so peak RAM scales with the window rather than the whole "
            "range, and an interrupted run resumes at the first unwritten "
            "window. Off by default because it is slow, not because it is "
            "impossible; without it the run stops at the raw shards and says "
            "so."
        ),
    )
    return parser


def refuse_conversion_without_raw_data(dataset, result) -> None:
    """Exit when this run fetched nothing and the raw tree holds nothing.

    A run whose every symbol failed (an expired key, an exhausted quota, a
    roster of typos) would otherwise walk into the conversion and fail deep
    inside the dataset layer with an error that reads like a conversion bug.
    Both halves of the condition matter: a run in which every symbol was
    skipped because its watermark already covered the window also reports
    zero successes, yet it has raw data on disk that must still be converted.
    The disk probe is ``dataset.has_raw_data()``, the same predicate the
    dataset's own raw scan uses, so the two cannot disagree.

    The message names paths and counts only; it never includes a credential
    or a vendor response body.

    Parameters
    ----------
    dataset
        A constructed dataset whose ``config.raw_data_dir_path`` is
        the raw tree the conversion would read.
    result
        An acquisition result with ``succeeded`` and ``failures``
        collections.

    Raises
    ------
    SystemExit
        With an explanatory message when there is nothing to
        convert.

    Examples
    --------
    Called by an ingest script between acquisition and conversion:

    >>> refuse_conversion_without_raw_data(dataset, result)

    The call returns ``None`` when ``result.succeeded`` is non-empty or
    ``dataset.has_raw_data()`` is true. Otherwise it raises
    ``SystemExit`` with a message beginning ``Refusing to convert: this
    run fetched 0 symbol(s) successfully``.
    """
    if result.succeeded or dataset.has_raw_data():
        return
    raise SystemExit(
        f"Refusing to convert: this run fetched {len(result.succeeded)} "
        f"symbol(s) successfully and {len(result.failures)} failed, and "
        f"there is no raw data under "
        f"{str(dataset.config.raw_data_dir_path)!r} to convert. No Zarr "
        f"store was written and none was modified. This is NOT a fault in "
        f"the conversion layer -- the acquisition step returned nothing to "
        f"convert. Read the per-symbol reasons through "
        f"SourceInspector.failures() (the _failures.json manifest beside the "
        f"watermarks); the usual causes are an unset or rejected credential, "
        f"an exhausted quota, and a roster whose symbols the vendor does not "
        f"serve."
    )


def add_chunk_args(
    parser: argparse.ArgumentParser,
    *,
    default: str = "year",
) -> argparse.ArgumentParser:
    """Add ``--chunk`` and ``--on-new-listing`` to ``parser``.

    Both ``choices`` lists are derived from the constants that define them
    (``TimeChunkPlanner.GRANULARITIES`` and
    ``BaseDataset.NEW_LISTING_STRATEGIES``), so a value added at the dataset
    layer is selectable here without a second edit. ``--on-new-listing`` sits
    in this group because it is a knob on the same chunked ``--to-zarr``
    conversion that ``--chunk`` governs.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.
    default : str
        Default granularity for ``--chunk``.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_chunk_args(argparse.ArgumentParser())
    >>> parser.parse_args([])
    Namespace(chunk='year', on_new_listing='refuse')
    >>> parser.parse_args(["--chunk", "month", "--on-new-listing", "widen"])
    Namespace(chunk='month', on_new_listing='widen')
    """
    parser.add_argument(
        "--chunk",
        type=str,
        choices=list(TimeChunkPlanner.GRANULARITIES),
        default=default,
        help=(
            "Time granularity of one --to-zarr conversion window (default "
            "year). Finer windows use less peak RAM and resume at a finer "
            "granularity, at the cost of more append round trips. 'hour' is "
            "available so an intraday window has a rung of its own."
        ),
    )
    parser.add_argument(
        "--on-new-listing",
        type=str,
        choices=list(BaseDataset.NEW_LISTING_STRATEGIES),
        default="refuse",
        help=(
            "What to do when the raw roster has grown since the Zarr store "
            "was built, the routine consequence of a new listing between two "
            "refreshes (default refuse). 'refuse' halts with the roster error "
            "and leaves the store untouched. 'rebuild' re-densifies every "
            "--chunk window from raw onto the new symbol union, recovering "
            "the new listing's real history at the cost of a full "
            "re-densify. 'widen' keeps the store and widens its symbol axis "
            "in place, which is fast but leaves the new listing's entire "
            "historical block NaN because raw is not re-read."
        ),
    )
    return parser


def add_concurrency_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_workers: int,
) -> argparse.ArgumentParser:
    """Add ``--limit`` and ``--max-workers`` to ``parser``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.
    default_max_workers : int
        Default for ``--max-workers``. It is a parameter
        rather than an import so this module does not depend on any one
        vendor's acquisition class.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_concurrency_args(
    ...     argparse.ArgumentParser(), default_max_workers=8
    ... )
    >>> parser.parse_args(["--limit", "50"])
    Namespace(limit=50, max_workers=8)
    """
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N resolved symbols, in ascending symbol "
            "order. The roster queries return sorted lists on purpose, so the "
            "same --universe/--limit pair truncates to the same N symbols on "
            "every run and a second run meets the watermarks the first one "
            "wrote. Useful for smoke-testing the pipeline end to end before "
            "committing to the full roster."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_max_workers,
        help=(
            "Concurrent in-flight symbol fetches (default "
            f"{default_max_workers}). Passed "
            "through config.kwargs, so it stays config-driven."
        ),
    )
    return parser


def validate_roster_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Enforce that exactly one of ``--symbols`` and ``--universe`` is set.

    ``--as-of-date`` is additionally required with ``--universe``. Misuse goes
    through ``parser.error``, so it exits with status 2 and the usage block
    like every other argparse error.

    Examples
    --------
    >>> parser = add_universe_args(argparse.ArgumentParser(prog="ingest"))
    >>> args = parser.parse_args(["--symbols", "AAPL,MSFT"])
    >>> validate_roster_args(parser, args)

    With ``--universe sp500`` and no ``--as-of-date`` the same call
    prints the usage block followed by ``ingest: error: --as-of-date is
    required when --universe is set.`` and exits with status 2.
    """
    if bool(args.symbols) == bool(args.universe):
        parser.error("Exactly one of --symbols or --universe must be set.")
    if args.universe and not args.as_of_date:
        parser.error("--as-of-date is required when --universe is set.")


def roster_category(args: argparse.Namespace) -> str | None:
    """Return the universe category ``args`` selects, or ``None``.

    Two spellings reach here: ``--universe``, whose token is mapped through
    ``UNIVERSE_CATEGORY_MAP``, and ``ingest_us_equity.py``'s ``--category``,
    which names the category directly. ``None`` means an explicit
    ``--symbols`` list, so a caller can size it differently instead of
    treating it as a roster.

    Examples
    --------
    >>> roster_category(argparse.Namespace(universe="nasdaq100"))
    nasdaq100_constituent
    >>> roster_category(argparse.Namespace(category="us_all"))
    us_all
    >>> print(roster_category(argparse.Namespace(symbols="AAPL", universe=None)))
    None
    """
    universe = getattr(args, "universe", None)
    if universe:
        return UNIVERSE_CATEGORY_MAP[universe]
    return getattr(args, "category", None) or None


def resolve_symbols(
    args: argparse.Namespace,
    catalog,
    *,
    mode: RosterMode,
) -> tuple[str, ...]:
    """Resolve the symbol roster ``args`` selects.

    An explicit ``--symbols`` list is returned verbatim, and ``catalog`` may
    be ``None`` on that path. Otherwise the category is resolved through
    ``catalog`` in one of two modes, and ``--limit`` truncates the result.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments carrying ``symbols``, ``as_of_date``,
        ``start_date``, ``end_date`` and optionally ``limit``,
        ``universe`` or ``category``.
    catalog
        A ``UniverseCatalog`` (or compatible object) used to resolve
        a category; ignored for an explicit list.
    mode : RosterMode
        ``"as_of"`` resolves point-in-time membership on
        ``args.as_of_date``. ``"in_range"`` resolves interval overlap
        across ``args.start_date .. args.end_date``, keeping every symbol
        that traded at any point in the window, delisted names included.
        Keyword-only with no default, because the two rosters differ and
        a silent default would hide a survivorship bias that nothing
        notices unless it looks for delisted tickers.

    Returns
    -------
    tuple[str, ...]
        The resolved symbols as a tuple, in the order the catalog returned
        them.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``ROSTER_MODES``.

    Examples
    --------
    An explicit list needs no catalog; ``--limit`` truncates it:

    >>> args = argparse.Namespace(symbols="AAPL,MSFT,NVDA", universe=None, limit=2)
    >>> resolve_symbols(args, None, mode="as_of")
    ('AAPL', 'MSFT')

    A category resolves through ``catalog``, a ``UniverseCatalog`` built
    from the universe table:

    >>> args = parser.parse_args(["--universe", "sp500", "--as-of-date", "2024-06-28"])
    >>> symbols = resolve_symbols(args, catalog, mode="as_of")
    """
    if mode not in ROSTER_MODES:
        raise ValueError(
            f"mode must be one of {list(ROSTER_MODES)}, got {mode!r}. "
            f"'as_of' is point-in-time membership on one day; 'in_range' is "
            f"interval overlap across the window. They are not "
            f"interchangeable and neither is a default."
        )

    category = roster_category(args)
    if category is None:
        symbols = tuple(
            token.strip() for token in args.symbols.split(",") if token.strip()
        )
    elif mode == "as_of":
        symbols = tuple(catalog.get_symbols_as_of(category, args.as_of_date))
    else:
        symbols = tuple(
            catalog.get_symbols_in_range(category, args.start_date, args.end_date)
        )

    limit = getattr(args, "limit", None)
    if limit is not None:
        symbols = symbols[:limit]
    return symbols


def add_data_dir_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--data-dir``, the per-run storage root override, to ``parser``.

    Defined once here so the flag name, help text and precedence story are
    identical on every entry point that offers it. See ``apply_data_dir``.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_data_dir_arg(argparse.ArgumentParser())
    >>> parser.parse_args(["--data-dir", "/mnt/quant"])
    Namespace(data_dir='/mnt/quant')
    """
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Storage root for this run: everything the run reads and writes "
            "(raw downloads, watermarks, Zarr stores, the universe table) is "
            "derived from it. Precedence is --data-dir > QUANTLAB_DATA_DIR > "
            "the repo-root data/ directory. The directory does not need to "
            "exist; the run creates what it needs. It relocates the whole "
            "root; ingest_binance_spot.py's --raw-data-dir is a different "
            "knob that points at one pre-existing raw CSV directory, and the "
            "two compose."
        ),
    )
    return parser


def apply_data_dir(args: argparse.Namespace) -> "object | None":
    """Apply ``--data-dir`` to the process-level storage root, if given.

    Each script calls this explicitly from its ``__main__``, directly after
    ``parse_args()`` and before anything that builds a config: the config
    factories snapshot their paths as strings at construction time, so an
    override applied later silently does nothing. It is a plain call rather
    than an argparse action so the root relocation is visible at the call
    site. The ``quantlab.config`` import is deferred to call time to keep
    this module light at import.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments. An object without a ``data_dir`` attribute is
        treated as if the flag were absent.

    Returns
    -------
    object | None
        The stored root ``Path``, or ``None`` when the flag was absent, in
        which case ``QUANTLAB_DATA_DIR`` or the repository default still
        applies.

    Examples
    --------
    >>> apply_data_dir(parser.parse_args(["--data-dir", "/mnt/quant"]))
    PosixPath('/mnt/quant')
    >>> apply_data_dir(parser.parse_args([])) is None
    True
    """
    value = getattr(args, "data_dir", None)
    if value is None:
        return None

    from quantlab.config import set_data_root

    return set_data_root(value)


def add_volume_guard_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--force-volume`` and ``--rows-per-symbol-day`` to ``parser``.

    ``--force-volume`` is an explicit, per-run opt-out of the pre-flight
    volume guard: it skips the refusal, never the arithmetic, so a forced run
    still prints its estimate. There is deliberately no environment variable
    or config key that disables the guard for a whole machine.
    ``--rows-per-symbol-day`` has no default because tick volume cannot be
    derived from a calendar; the guard refuses to size a tick fetch without a
    measured figure rather than inventing one.

    Returns
    -------
    argparse.ArgumentParser
        ``parser``, for chaining.

    Examples
    --------
    >>> parser = add_volume_guard_args(argparse.ArgumentParser())
    >>> parser.parse_args(["--force-volume", "--rows-per-symbol-day", "250000"])
    Namespace(force_volume=True, rows_per_symbol_day=250000)
    """
    parser.add_argument(
        "--force-volume",
        action="store_true",
        help=(
            "Proceed even when the pre-flight volume estimate is over a "
            "ceiling. The arithmetic still runs and is still printed; only "
            "the refusal is skipped. Explicit and per-run on purpose: there is "
            "no environment variable and no config key that disables the guard "
            "wholesale."
        ),
    )
    parser.add_argument(
        "--rows-per-symbol-day",
        type=int,
        default=None,
        help=(
            "Measured rows per symbol per session, required to size a "
            "--frequency tick fetch and ignored otherwise. No default: sample "
            "one symbol-day and count. A guessed figure produces a guessed "
            "budget, and the guard's whole value is that its number is "
            "defensible."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Call-site helpers for the pre-flight volume guard.
#
# The guard itself is `UniverseCatalog.assert_acquisition_volume_fits`, which
# lives in a module that binds no acquisition class. That is what makes
# "refuses before the client is constructed" a structural property, and the
# helpers below preserve it: they are imported by the ingest scripts, never
# the reverse, and they construct no client.
# ---------------------------------------------------------------------------

#: Category name a refusal reports when the roster came from an explicit
#: ``--symbols`` list. It is not a real category and is never validated
#: against one; see ``_explicit_symbol_catalog``.
EXPLICIT_SYMBOLS_CATEGORY = "(explicit --symbols list)"

#: Window start assumed, for sizing only, when a script is run with no
#: ``--start-date``. An unbounded Tiingo request returns a symbol's whole
#: history, which cannot be priced without a start. The assumption is printed
#: beside the estimate and is never written back onto ``args`` or a config.
#: The value is the project's documented backfill floor, which errs toward a
#: longer window than most runs fetch, the safe direction for a guard.
UNBOUNDED_WINDOW_START = "2016-01-01"

#: Bytes in one GiB.
_GIB = 1024**3

#: Cache for the class ``_explicit_symbol_catalog`` builds on first use.
_EXPLICIT_CATALOG_CLASS = None


def _explicit_symbol_catalog(symbol_count: int):
    """Return a pricing view that sizes an explicitly named symbol list.

    An explicit 15,000-symbol list costs exactly what the same roster
    resolved from a category costs, so the guard must price it rather than
    skip it. The catalog derives symbol counts and listing spans from its
    interval table, which an explicit list does not have, so this returns an
    instance of a ``UniverseCatalog`` subclass that replaces the roster
    profile step with one driven by ``symbol_count``. Every ceiling, report
    and refusal message stays the catalog's own, so the explicit path cannot
    drift from the category path.

    The subclass is built on first use and cached, and ``quantlab.universe``
    is imported here rather than at module scope to keep this module light
    at import.

    Parameters
    ----------
    symbol_count : int
        Number of symbols in the explicit list.
    """
    global _EXPLICIT_CATALOG_CLASS
    if _EXPLICIT_CATALOG_CLASS is None:
        import datetime

        from quantlab.universe import UniverseCatalog

        class _ExplicitSymbolCatalog(UniverseCatalog):
            """Catalog view whose roster is a bare symbol count."""

            def __init__(self, symbols: int):  # noqa: D107
                """Store the symbol count without touching the reference table.

                ``super().__init__`` is skipped on purpose: this view never
                reads ``universe.parquet``, so ``--symbols AAPL`` works on a
                machine that has never built it.
                """
                self._explicit_symbols = symbols

            def _validate_category(self, category: str) -> None:
                """Accept ``EXPLICIT_SYMBOLS_CATEGORY``; delegate the rest.

                This view resolves no roster from the reference table, so
                the explicit sentinel has nothing to be validated against.
                The base validator still applies to every other token, so a
                ``--limit``-truncated real category keeps its typo check.
                Nothing on the pricing path currently calls this override,
                because ``_roster_window_profile`` below is replaced
                wholesale; it is kept so the sentinel stays safe if that
                override ever delegates to ``super()``.
                """
                if category != EXPLICIT_SYMBOLS_CATEGORY:
                    super()._validate_category(category)

            def _roster_window_profile(
                self,
                category: str,
                start_date: str,
                end_date: str,
                bars_per_day: int = 1,
            ) -> dict:
                """Profile the window for a fixed symbol count at density 1.0.

                Mirrors the base signature, ``bars_per_day`` included, since
                this is what the volume guard calls on the explicit path.
                Dates are normalised exactly as the base method does before
                any arithmetic. Density is 1.0 rather than the catalog's
                survivorship-adjusted figure: a hand-named list carries no
                delisting structure, and assuming it did would understate
                the fetch by roughly 2.7x, the wrong direction for a guard.
                """
                start_date = self._normalize_iso_date(start_date, "start_date")
                end_date = self._normalize_iso_date(end_date, "end_date")
                if bars_per_day < 1:
                    raise ValueError(
                        f"bars_per_day must be >= 1, got {bars_per_day!r}."
                    )
                window_days = (
                    datetime.date.fromisoformat(end_date)
                    - datetime.date.fromisoformat(start_date)
                ).days + 1
                trading_days = max(
                    round(
                        window_days
                        * self.TRADING_DAYS_PER_YEAR
                        / self.CALENDAR_DAYS_PER_YEAR
                    ),
                    1,
                )
                symbols = self._explicit_symbols
                timestamps = trading_days * bars_per_day
                dense_cells = symbols * timestamps
                return {
                    "symbols": symbols,
                    "trading_days": trading_days,
                    "bars_per_day": bars_per_day,
                    "timestamps": timestamps,
                    "dense_cells": dense_cells,
                    "observed_cells": dense_cells,
                    "density": 1.0,
                }

        _EXPLICIT_CATALOG_CLASS = _ExplicitSymbolCatalog
    return _EXPLICIT_CATALOG_CLASS(symbol_count)


def _sizing_window(args: argparse.Namespace) -> tuple[str, str, bool]:
    """Return ``(start, end, assumed)`` for the volume estimate.

    A missing ``--start-date`` falls back to ``UNBOUNDED_WINDOW_START`` and a
    missing ``--end-date`` to today; ``assumed`` reports whether either
    fallback applied. Nothing is written back onto ``args``: an assumption
    made to size a fetch must not become the window the fetch requests.
    """
    import datetime

    start = getattr(args, "start_date", None)
    end = getattr(args, "end_date", None)
    assumed = start is None or end is None
    return (
        start or UNBOUNDED_WINDOW_START,
        end or datetime.date.today().isoformat(),
        assumed,
    )


def volume_pricing(
    args: argparse.Namespace,
    catalog,
    *,
    symbols,
) -> tuple[object, str, str, str, bool]:
    """Return what the caller needs to run the volume guard.

    The guard itself, ``assert_acquisition_volume_fits``, is not called here:
    each ingest script names it at its own call site so the decision to fetch
    is visible where it is made. What is shared is which object prices the
    fetch and over which window. An explicit ``--symbols`` list, or a
    category truncated by ``--limit``, is priced from its real symbol count
    through ``_explicit_symbol_catalog``; only a whole, untruncated category
    is priced through the catalog's density-adjusted estimate.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    catalog
        The ``UniverseCatalog`` that resolved ``symbols``; may be
        ``None`` for an explicit list.
    symbols
        The resolved roster, as returned by ``resolve_symbols``.

    Returns
    -------
    tuple[object, str, str, str, bool]
        ``(pricing, category, start_date, end_date, window_assumed)``, where
        ``pricing`` is the object to call the guard on, ``category`` is the
        name to report, and ``window_assumed`` says whether the window came
        from the fallbacks in ``_sizing_window``.

    Examples
    --------
    >>> args = argparse.Namespace(
    ...     symbols="AAPL,MSFT", start_date="2024-01-01", end_date="2024-12-31"
    ... )
    >>> symbols = resolve_symbols(args, None, mode="as_of")
    >>> pricing, category, start, end, assumed = volume_pricing(
    ...     args, None, symbols=symbols
    ... )
    >>> category, start, end, assumed
    ('(explicit --symbols list)', '2024-01-01', '2024-12-31', False)
    """
    category = roster_category(args)
    truncated = getattr(args, "limit", None) is not None
    if category is None or truncated:
        pricing = _explicit_symbol_catalog(len(symbols))
        reported_category = category or EXPLICIT_SYMBOLS_CATEGORY
    else:
        pricing = catalog
        reported_category = category

    start_date, end_date, assumed = _sizing_window(args)
    return pricing, reported_category, start_date, end_date, assumed


def print_volume_estimate(
    estimate: dict,
    *,
    category: str,
    start_date: str,
    end_date: str,
    window_assumed: bool = False,
    forced: bool = False,
    print_fn=print,
) -> dict:
    """Print an admitted volume estimate and return it.

    Call sites pass the guard's return value straight in, as in
    ``print_volume_estimate(pricing.assert_acquisition_volume_fits(...))``,
    so this runs only when the guard admits the fetch. A refusal never gets
    here; its exception message already carries the same arithmetic plus
    every ceiling crossed and a narrowing that would fit. What this adds is
    the admitted case: the numbers the user proceeded with, and a
    ``--force-volume`` line that distinguishes "under every ceiling" from
    "over one and overridden".

    Parameters
    ----------
    estimate : dict
        The dict returned by ``assert_acquisition_volume_fits``.
    category : str
        Roster name to print.
    start_date : str
        Window start used for sizing.
    end_date : str
        Window end used for sizing.
    window_assumed : bool
        Whether the window came from the sizing fallbacks;
        if so, a line saying so is printed first.
    forced : bool
        Whether ``--force-volume`` was set.
    print_fn
        Output function, injectable for tests.

    Returns
    -------
    dict
        ``estimate``, unchanged, so a call site can compose.

    Examples
    --------
    Continuing from ``volume_pricing``, two symbols over 2024:

    >>> estimate = pricing.assert_acquisition_volume_fits(
    ...     category, start, end, frequency="1d", batch_size=1
    ... )
    >>> _ = print_volume_estimate(
    ...     estimate, category=category, start_date=start, end_date=end
    ... )
    Pre-flight volume estimate (zero vendor requests issued):
      roster:            (explicit --symbols list)
      window:            2024-01-01 .. 2024-12-31
      symbols:           2
      trading days (~):  253
      rows (~):          506 (1/symbol-day, density 1.000)
      raw on disk (~):   0.00 GiB
      requests (~):      2 (batch_size=1, page_limit=10,000)
      wall clock (~):    0.0 h at 200 req/min
    """
    if window_assumed:
        print_fn(
            f"No complete --start-date/--end-date given; this fetch was sized "
            f"against the ASSUMED window {start_date}..{end_date} "
            f"(utils.cli.UNBOUNDED_WINDOW_START). The assumption is used for "
            f"the estimate only and is never sent to the vendor."
        )
    print_fn("Pre-flight volume estimate (zero vendor requests issued):")
    print_fn(f"  roster:            {category}")
    print_fn(f"  window:            {start_date} .. {end_date}")
    print_fn(f"  symbols:           {estimate['symbols']:,}")
    print_fn(f"  trading days (~):  {estimate['trading_days']:,}")
    print_fn(
        f"  rows (~):          {estimate['rows']:,} "
        f"({estimate['bars_per_day']:,}/symbol-day, density "
        f"{estimate['density']:.3f})"
    )
    print_fn(f"  raw on disk (~):   {estimate['raw_bytes'] / _GIB:.2f} GiB")
    print_fn(
        f"  requests (~):      {estimate['requests']:,} "
        f"(batch_size={estimate['batch_size']:,}, "
        f"page_limit={estimate['page_limit']:,})"
    )
    print_fn(
        f"  wall clock (~):    {estimate['wall_clock_hours']:.1f} h at "
        f"{estimate['rate_limit_per_min']:,} req/min"
    )
    if forced:
        print_fn(
            "  --force-volume:    ON -- any ceiling crossed above was NOT "
            "enforced. The arithmetic still ran; only the refusal was skipped."
        )
    return estimate


def print_sql_volume_estimate(
    estimate: dict, *, forced: bool = False, print_fn=print
) -> dict:
    """Print an admitted ``SqlVolumeGuard`` estimate and return it.

    The WRDS counterpart of ``print_volume_estimate``, under the same rule: it
    is reached only when the guard admitted the pull, since a refusal carries
    its own numbers in its exception. The bucket line is labelled from
    ``estimate["unit"]``: ``trading days:`` for a TAQ pull, whose pages are
    days, and ``year buckets:`` for a CRSP pull, whose pages are calendar
    years. An estimate without a ``unit`` key is treated as a TAQ estimate.
    Only counts, dates and ceilings are printed, never a credential.

    Parameters
    ----------
    estimate : dict
        The dict returned by the SQL volume guard.
    forced : bool
        Whether ``--force-volume`` was set.
    print_fn
        Output function, injectable for tests.

    Returns
    -------
    dict
        ``estimate``, unchanged.

    Examples
    --------
    With ``estimate`` from ``SqlVolumeGuard.estimate`` for two symbols
    over two trading days:

    >>> lines = []
    >>> _ = print_sql_volume_estimate(estimate, print_fn=lines.append)
    >>> lines[1]
    '  symbols:           2'
    >>> lines[4]
    '  rows:              400,000'
    >>> lines[5]
    '  bytes/row:         30 (ASSUMPTION: default, not a measured shard size)'
    """
    # Imported at call time so this module stays light at import.
    from quantlab.acquisition._support.sql_volume import SqlVolumeGuard

    bytes_per_row = estimate["bytes_per_row"]
    assumed = (
        " (ASSUMPTION: default, not a measured shard size)"
        if bytes_per_row == SqlVolumeGuard.DEFAULT_BYTES_PER_ROW
        else ""
    )
    print_fn(
        "Pre-flight WRDS volume estimate (counted with count(*) per symbol "
        "batch, no data pulled):"
    )
    print_fn(f"  symbols:           {estimate['symbols']:,}")
    print_fn(f"  window:            {estimate['start_date']} .. {estimate['end_date']}")
    bucket_label = f"{estimate.get('unit', 'trading day')}s:"
    print_fn(f"  {bucket_label:<19}{estimate['trading_days']:,}")
    print_fn(f"  rows:              {estimate['rows']:,}")
    print_fn(f"  bytes/row:         {bytes_per_row}{assumed}")
    print_fn(f"  raw on disk (~):   {estimate['raw_bytes'] / _GIB:.2f} GiB")
    print_fn(
        f"  ceilings:          raw-bytes {estimate['max_raw_bytes'] / _GIB:.2f} "
        f"GiB, raw-rows {estimate['max_raw_rows']:,}"
    )
    if forced:
        crossed = ", ".join(estimate.get("crossed") or []) or "none"
        print_fn(
            f"  --force-volume:    ON -- crossed ceiling(s) [{crossed}] were NOT "
            f"enforced. The arithmetic still ran; only the refusal was skipped."
        )
    return estimate


def print_conversion_result(result, *, print_fn=print):
    """Print the ``ConversionResult`` returned by ``quantlab.registry.convert``.

    Every line comes off the returned object, not the config and not a
    read-back of the store, so the printed path is the one actually written
    and ``windows_skipped`` / ``resumed`` describe this run rather than
    whatever the ledger holds. Only paths, integer counts and booleans are
    printed. The argument is left unannotated so this module does not import
    the result class.

    Parameters
    ----------
    result
        The conversion result to render.
    print_fn
        Output function, injectable for tests.

    Returns
    -------
    ConversionResult
        ``result``, unchanged, so a call site can compose.

    Examples
    --------
    With ``result`` a ``ConversionResult`` whose run wrote five yearly
    windows:

    >>> _ = print_conversion_result(result)
    Zarr store written at: /mnt/quant/data/us_equity/1d/stock.zarr
      windows:           5 written, 0 skipped of 5 planned (year)
      symbols pinned:    2
      rows appended:     2,516
      peak window:       0.00 GiB
      chunk ledger:      /mnt/quant/data/us_equity/1d/stock.zarr.chunks.json
    """
    print_fn(f"Zarr store written at: {result.zarr_path}")
    print_fn(
        f"  windows:           {result.windows_written} written, "
        f"{result.windows_skipped} skipped of {result.windows_planned} "
        f"planned ({result.granularity})"
    )
    print_fn(f"  symbols pinned:    {result.pinned_symbols}")
    print_fn(f"  rows appended:     {result.rows_written:,}")
    if result.resumed:
        # Said explicitly: "0 windows written" and "nothing left to do" read
        # identically in a log otherwise, and one of them is a bug report.
        print_fn(
            "  resumed:           yes -- windows the chunk ledger already "
            "recorded were skipped, not rewritten"
        )
    if result.peak_window_bytes is not None:
        # A fully resumed run materialises no window and has no observed peak;
        # printing 0.00 GiB would claim a measurement nobody took.
        predicted = (
            ""
            if result.predicted_peak_bytes is None
            else (
                f" (predicted "
                f"{result.predicted_peak_bytes / _GIB:.2f} GiB)"
            )
        )
        print_fn(
            f"  peak window:       "
            f"{result.peak_window_bytes / _GIB:.2f} GiB{predicted}"
        )
    print_fn(f"  chunk ledger:      {result.ledger_path}")
    return result
