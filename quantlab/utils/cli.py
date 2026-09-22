"""Argument groups and roster resolution shared by the ingest entry points.

D-14 keeps one top-level script per source -- `ingest_tiingo.py`,
`ingest_us_equity.py`, `ingest_alpaca.py` -- and forbids a third copy of the
`--symbols` / `--universe` / `--as-of-date` / `--start-date` / `--end-date` /
`--chunk` parsing those scripts share. The definitions live HERE once; each
script adds only the flags that are genuinely its own.

This module is deliberately dependency-light in the `utils/` tradition
(`utils/file.py`, `utils/timer.py`): it builds `argparse` groups and resolves a
roster through a catalog handed to it. It constructs no acquisition client and
issues no request, and it opens no file -- `refuse_conversion_without_raw_data`
does reach the filesystem, but only by asking a Dataset handed to it whether
its raw root holds a shard (a directory stat, read-only, no parquet opened).
Every object it works on arrives as an argument. Its two module-scope project imports are
both there for the same reason -- a `choices` list must be DERIVED from the
literal that defines it rather than restated here, or a value added at the
Dataset layer stays unreachable from the command line:
`base.chunking.TimeChunkPlanner.GRANULARITIES` for `--chunk`, and
`base.data.BaseDataset.NEW_LISTING_STRATEGIES` for `--on-new-listing`.
`apply_data_dir` additionally reaches the `config` layer, but imports it at
CALL time for the same reason `_explicit_symbol_catalog` defers
`acquisition.universe`: a module-scope `from config import set_data_root`
would drag `dataset.backend`, `dataset.spot`, `dataset.stock` and
`base.config` into every import of this module.

**The one thing this module must not unify.** `ingest_tiingo.py` resolves
point-in-time membership on a single day; `ingest_us_equity.py` resolves
interval OVERLAP across a window. That is a deliberate semantic difference, not
duplication -- see `resolve_symbols`, whose `mode` is keyword-only and has no
default for exactly that reason.
"""

import argparse
from typing import Literal

from quantlab.base.chunking import TimeChunkPlanner
from quantlab.base.data import BaseDataset

#: Maps the CLI-facing --universe choice to enums.data.UniverseCategory.
#:
#: The --universe `choices` are DERIVED from this map rather than repeated as a
#: second hardcoded list: when they were two separate literals, adding the
#: nasdaq100_constituent category produced it into universe.parquet while
#: leaving it unselectable from the only CLI that consumes the table.
#:
#: It lives here rather than in any one script because it is now read by every
#: script that offers `--universe`; a per-script copy would reintroduce the
#: same drift one level up.
UNIVERSE_CATEGORY_MAP = {
    "sp500": "sp500_constituent",
    "nasdaq100": "nasdaq100_constituent",
    "nasdaq_all": "nasdaq_all",
    "us_all": "us_all",
}

#: The two window semantics the repo's ingest scripts actually have, and the
#: help text each one carries. Kept as data rather than as a caller-supplied
#: string so the distinction is stated once, where the flags are defined.
#:
#: - `"request-range"`: the window is a per-symbol REQUEST range handed to the
#:   vendor (`ingest_tiingo.py`, `ingest_alpaca.py`).
#: - `"interval-overlap"`: the window additionally FILTERS the roster, keeping
#:   every symbol that traded at any point inside it (`ingest_us_equity.py`).
WindowSemantics = Literal["request-range", "interval-overlap"]

_WINDOW_HELP = {
    "request-range": {
        "start": "Start date (inclusive), e.g. 2024-01-01.",
        "end": "End date (inclusive), e.g. 2024-12-31.",
    },
    "interval-overlap": {
        "start": (
            "Window start (inclusive), default {default}. Applied "
            "as interval OVERLAP: every symbol that traded at ANY point in "
            "the window is kept, INCLUDING those that delisted inside it. "
            "Only symbols whose listing ended before this date are dropped."
        ),
        "end": "Window end (inclusive), default today.",
    },
}

#: The two roster-resolution semantics `resolve_symbols` exposes. Neither is a
#: default; see that function's docstring.
RosterMode = Literal["as_of", "in_range"]
ROSTER_MODES: tuple[str, ...] = ("as_of", "in_range")


def add_window_args(
    parser: argparse.ArgumentParser,
    *,
    default_start_date: str | None = None,
    semantics: WindowSemantics = "request-range",
) -> argparse.ArgumentParser:
    """Add `--start-date` / `--end-date`.

    `semantics` selects which help text the two flags carry, and it is NOT
    cosmetic: a request range and a roster-filtering interval overlap are
    different promises to the user, and the longer text exists because the
    second one is the non-obvious of the two.
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
    """Add `--symbols`, `--universe` and `--as-of-date`.

    The `--universe` help text carries real domain distinctions -- the
    Nasdaq-100 index is not the NASDAQ roster, and a full-window backfill wants
    a different script -- which is the reason this extraction is worth doing
    rather than letting each script grow its own shorter, vaguer version.
    """
    parser.add_argument(
        "--symbols",
        type=str,
        required=False,
        default=None,
        help=(
            "Comma-separated TICKERS (e.g. AAPL,MSFT). Mutually exclusive "
            "with --universe. This is the ticker-side entry point: a CRSP "
            "panel's symbol axis is the int64 PERMNO (D-01), so a CRSP "
            "conversion takes --permnos instead, and the ticker a PERMNO wore "
            "on a given day is read from the '.crsp_tickers.json' sidecar "
            "beside the store."
        ),
    )
    parser.add_argument(
        "--universe",
        type=str,
        choices=sorted(UNIVERSE_CATEGORY_MAP),
        default=None,
        help=(
            "Resolve a symbol list from the persisted universe table "
            "(02-08-PLAN.md) instead of --symbols. 'sp500' resolves "
            "point-in-time S&P 500 constituent membership; 'nasdaq100' "
            "resolves point-in-time Nasdaq-100 (NDX) index membership; "
            "'nasdaq_all' resolves the full NASDAQ-listed Common Stock roster "
            "(current + delisted); 'us_all' resolves the full US listed-equity "
            "roster -- NYSE + NASDAQ + AMEX common stock, delisted included "
            "(~15.4k tickers). Note 'nasdaq100' and 'nasdaq_all' are "
            "DIFFERENT universes that merely share the word Nasdaq -- the "
            "former is the ~100-name index, the latter every symbol ever "
            "listed on the exchange. 'us_all' is a strict superset of "
            "'nasdaq_all'; both are kept deliberately. Requires --as-of-date. "
            "For a full-window BACKFILL of every symbol that traded at any "
            "point in a date range (rather than membership on one day), use "
            "ingest_us_equity.py, which queries by interval overlap instead."
        ),
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=None,
        help="Required with --universe; point-in-time date (YYYY-MM-DD) to resolve membership as of.",
    )
    return parser


def add_to_zarr_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add `--to-zarr`, the opt-in that gates every raw-to-Zarr conversion.

    Defined HERE once, for the reason D-14 gives for `--symbols` and friends:
    `ingest_us_equity.py` had the flag and the other two converted
    unconditionally, so the three front doors disagreed about what a run
    without arguments does. A user who learned one script's default learned
    the wrong thing about the other two (G-03.4-1b).

    OFF by default in all three, and the default path SAYS so rather than
    staying quiet: a conversion that silently did not happen is the same class
    of silence this flag exists to end.

    **One conversion path, therefore one help text (D-06).** This helper used
    to take a `mode` selecting between a chunked clause and a whole-window
    one, following `add_concurrency_args(default_max_workers=...)`. There is
    no second mode to select any more -- the chunked, resumable conversion is
    what every shell runs -- so the parameter and the two-armed help dict were
    retired with the mode itself, and the surviving text is the chunked arm's
    word for word. The roadmap's "three modes" was stale arithmetic.
    """
    parser.add_argument(
        "--to-zarr",
        action="store_true",
        help=(
            "After acquisition, convert the raw parquet into the Zarr store. "
            "The full window is no longer refused: the conversion densifies "
            "and appends ONE --chunk window at a time, so peak RAM scales "
            "with the window rather than the range, and an interrupted run "
            "resumes at the first unwritten window."
            " OFF by default because it is slow, not because it is "
            "impossible; without it the run stops at the raw shards and says "
            "so."
        ),
    )
    return parser


def refuse_conversion_without_raw_data(dataset, result) -> None:
    """Refuse a raw-to-Zarr conversion when this run fetched nothing AND the
    raw tree is empty.

    G-03.4-1a. A run whose every symbol failed (an expired key, an exhausted
    quota, a roster of typos) used to walk straight into
    `StockDataset.from_raw_data()` and end on `dataset/stock.py`'s uncaught
    absent-root `ValueError` -- a traceback that reads like a bug in the
    conversion layer when the actual event was "the vendor returned nothing".
    That raise is the correct LOWER-level signal and is unchanged; this is the
    upper-level caller that translates it into a clean, non-zero exit.

    **Both halves of the condition are load-bearing.** A run in which every
    symbol was SKIPPED because its watermark already covers the window also
    reports zero successes -- and it has raw data on disk that must still be
    converted. Counting successes alone would refuse that legitimate run. What
    separates the two cases is the disk probe, so the probe is the test and
    the counts are only reported.

    The probe is `dataset.has_raw_data()` -- the SAME predicate `_scan_raw`
    decides on -- rather than a second `exists() / rglob()` written here. Two
    spellings of one fact is the ancestor shape of this gap.

    Takes CONSTRUCTED objects and imports nothing: this module is
    dependency-light by contract (see the module docstring), and
    `tests/test_data_dir_cli.py::
    test_utils_cli_does_not_import_config_at_module_scope` holds it there.

    The message names paths, counts and where to read the failure list. It
    carries NO credential value and no vendor response body: this repo has
    already leaked one real key, and a refusal path is exactly where a
    "helpful" dump of the vendor's error gets added.
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
    """Add `--chunk` and `--on-new-listing`, both with DERIVED `choices`.

    Derived, never restated: a granularity added to `GRANULARITIES` and its
    `_period_key`, or a strategy added to
    `BaseDataset.NEW_LISTING_STRATEGIES`, must not need a second edit here to
    become selectable.

    `--on-new-listing` belongs in THIS group rather than a new one: it is a
    knob on the same chunked `--to-zarr` conversion `--chunk` governs, and this
    group is already registered in the shared-group wiring test's table.
    """
    parser.add_argument(
        "--chunk",
        type=str,
        choices=list(TimeChunkPlanner.GRANULARITIES),
        default=default,
        help=(
            "Time granularity of one --to-zarr conversion window (default "
            "year). Finer windows use less peak RAM and give a finer resume "
            "granularity, at the cost of more append round trips. The ladder "
            "now reaches 'hour', so an intraday window has a rung of its own."
        ),
    )
    parser.add_argument(
        "--on-new-listing",
        type=str,
        choices=list(BaseDataset.NEW_LISTING_STRATEGIES),
        default="refuse",
        help=(
            "What to do when the raw roster has grown since the Zarr store "
            "was built -- the routine consequence of a new listing between two "
            "refreshes (default refuse). 'refuse' halts with the roster error, "
            "exactly as before this flag existed, leaving the store untouched. "
            "'rebuild' re-densifies every --chunk window from raw onto the new "
            "symbol union, recovering the new listing's REAL history at the "
            "cost of a full re-densify. 'widen' keeps the store and widens its "
            "symbol axis in place, which is fast but leaves the new listing's "
            "entire historical block NaN because raw is not re-read."
        ),
    )
    return parser


def add_concurrency_args(
    parser: argparse.ArgumentParser,
    *,
    default_max_workers: int,
) -> argparse.ArgumentParser:
    """Add `--max-workers` and `--limit`.

    `default_max_workers` is a PARAMETER rather than an import: this module
    stays vendor-agnostic, so reaching into `TiingoAcquisition` for a default
    would give the shared CLI module a dependency on one particular vendor's
    acquisition class.
    """
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Process only the first N resolved symbols, where FIRST means "
            "ASCENDING BY SYMBOL -- the roster queries return sorted lists on "
            "purpose, so the same --universe/--limit pair truncates to the "
            "SAME N symbols on every run and a second run meets the "
            "watermarks the first one wrote. For smoke-testing the pipeline "
            "end to end before committing to the full roster."
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
    """The `--symbols` / `--universe` mutual exclusion, message text unchanged.

    Uses `parser.error(...)` rather than raising, so a misuse exits 2 with the
    usage block the way every other argparse misuse in these scripts does.
    """
    if bool(args.symbols) == bool(args.universe):
        parser.error("Exactly one of --symbols or --universe must be set.")
    if args.universe and not args.as_of_date:
        parser.error("--as-of-date is required when --universe is set.")


def roster_category(args: argparse.Namespace) -> str | None:
    """The universe category these arguments select, or `None` for an explicit
    `--symbols` list.

    Two spellings reach this: `--universe`, whose CLI-facing token is mapped
    through `UNIVERSE_CATEGORY_MAP`, and `ingest_us_equity.py`'s `--category`,
    which already names the category directly. Returning `None` rather than
    raising is what lets a caller size an explicit symbol list differently
    instead of pretending it came from a roster.
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
    """Resolve the symbol roster these arguments select.

    `mode` is KEYWORD-ONLY and has NO DEFAULT, deliberately:

    - `"as_of"` calls `UniverseCatalog.get_symbols_as_of` -- point-in-time
      membership on ONE day. Right for "fetch today's S&P 500".
    - `"in_range"` calls `UniverseCatalog.get_symbols_in_range` -- interval
      OVERLAP across the window. A backfill wants every symbol that traded at
      ANY point in the window, including the ~6.9k that delisted inside it;
      resolving membership on a single day there would reintroduce exactly the
      survivorship bias this roster exists to remove.

    The two are a deliberate semantic difference, not duplication. A default
    would make the wrong choice SILENT -- and wrong in the direction whose
    symptom (a roster missing every delisted name) is invisible to every test
    that does not specifically look for delisted tickers. So there is none, and
    a caller that omits `mode` gets a `TypeError` instead of a quiet bias.

    An explicit `--symbols` list bypasses the catalog entirely and is returned
    verbatim; `catalog` may be `None` on that path.
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
    """Add `--data-dir`, the per-run storage root override.

    Defined here once (D-14) so the flag name, its help text and its
    precedence story are identical on every entry point that offers it.
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
            "exist -- the run creates what it needs. It relocates the whole "
            "ROOT; ingest_binance_spot.py's --raw-data-dir is a different "
            "knob that points at one pre-existing raw CSV directory, and the "
            "two compose."
        ),
    )
    return parser


def apply_data_dir(args: argparse.Namespace) -> "object | None":
    """Apply `--data-dir` to the process-level storage root, if it was given.

    Returns the stored root `Path`, or `None` when the flag was absent (in
    which case the root is left exactly as it was, so `QUANTLAB_DATA_DIR` or
    the repo default still answers).

    This is an EXPLICIT call each script makes from its own `__main__`, not an
    argparse `action=` side effect. A custom Action firing inside
    `parse_args()` would make the ordering structurally unbreakable, which is
    tempting -- but it was rejected for the reason stated at the volume guard's
    call-site helper below: each script names the thing it invokes at its own
    call site, so a reader of the script, and a grep across the entry points,
    sees the decision where it is made rather than one level of indirection
    away. A root-relocating side effect hidden inside argument parsing is
    exactly the kind of invisible action that comment exists to prevent. The
    ordering is enforced instead by the AST guard in
    `tests/test_data_dir_cli.py`.

    The `config` import is deferred to call time so this module keeps the
    module-scope dependency surface its docstring promises -- the same reason
    `_explicit_symbol_catalog` defers `acquisition.universe`.
    """
    value = getattr(args, "data_dir", None)
    if value is None:
        return None

    from quantlab.config import set_data_root

    return set_data_root(value)


def add_volume_guard_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the pre-flight volume guard's two flags: `--force-volume` and
    `--rows-per-symbol-day`.

    Defined HERE rather than in each script so the flag name and the help text
    exist once (D-14). `--force-volume` is an EXPLICIT, visible opt-out: it is
    a flag a user types, never an environment variable and never a config key
    that could turn the guard off for a whole machine without anyone noticing.
    It skips the RAISE and never the arithmetic, so a forced run still prints
    the estimate, with a line saying a ceiling was crossed and overridden. A
    REFUSED run prints no estimate -- its numbers travel in the exception
    message instead (WR-07).

    `--rows-per-symbol-day` has no default on purpose. Tick volume is not
    derivable from a calendar the way a bar count is, so
    `assert_acquisition_volume_fits` REFUSES a tick estimate without a measured
    figure rather than inventing one -- an invented row count would make the
    guard confidently wrong in exactly the regime it exists for.
    """
    parser.add_argument(
        "--force-volume",
        action="store_true",
        help=(
            "Proceed even when the pre-flight volume estimate is over a "
            "ceiling. The arithmetic still runs and is still printed -- only "
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
            "Measured rows per symbol per session, REQUIRED to size a "
            "--frequency tick fetch and ignored otherwise. No default: sample "
            "one symbol-day and count. A guessed figure produces a guessed "
            "budget, and the guard's whole value is that its number is "
            "defensible."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# The pre-flight volume guard's call-site helper (D-09 / SC-6)
#
# `UniverseCatalog.assert_acquisition_volume_fits` is the guard. It lives in a
# module that imports no acquisition module and binds no `Acquisition`
# subclass, which is what makes "refuses before the client is constructed" a
# STRUCTURAL property rather than a matter of call order. Nothing below may
# undo that: this helper is imported BY the ingest scripts, never the reverse,
# and it constructs no client either.
# ---------------------------------------------------------------------------

#: The category name a refusal reports when the roster came from an explicit
#: `--symbols` list rather than a universe category. It is not a real category
#: and is never validated against one -- see `_ExplicitSymbolCatalog`.
EXPLICIT_SYMBOLS_CATEGORY = "(explicit --symbols list)"

#: Window start assumed for SIZING ONLY when a script is invoked with no
#: `--start-date`. `ingest_tiingo.py` has no default window, and an unbounded
#: Tiingo EOD request returns a symbol's whole history -- which cannot be
#: priced without a start.
#:
#: This is a STATED assumption, not a hidden default: `run_volume_guard` prints
#: the assumed window on the line above the estimate whenever it applies, and
#: it is never written back onto `args` or into any config. The value is the
#: project's own documented backfill floor (`ingest_us_equity.DEFAULT_START_DATE`,
#: D-05), and assuming it errs toward a LONGER window than most such runs
#: actually fetch, which is the safe direction for a guard.
UNBOUNDED_WINDOW_START = "2016-01-01"

_GIB = 1024**3

#: Built once, on first use, by `_explicit_symbol_catalog`.
_EXPLICIT_CATALOG_CLASS = None


def _explicit_symbol_catalog(symbol_count: int):
    """A pricing view that sizes an explicitly named symbol list.

    An explicit 15,000-symbol list is exactly as expensive as the same roster
    resolved from a category, so the guard must price it rather than skip it.
    But `_roster_window_profile` derives its symbol count and its listing spans
    from the catalog's interval table, and an explicit list has neither.

    So this SUBCLASSES `UniverseCatalog` and overrides `_roster_window_profile`
    WHOLESALE -- the one step that is ABOUT the roster's provenance. It also
    carries a `_validate_category` override (admitting this view's own sentinel,
    delegating everything else), and that override is CURRENTLY UNREACHABLE
    precisely because the `_roster_window_profile` override is wholesale: the
    base's validator call lives inside base methods this view either replaces
    or never enters. It is retained deliberately -- see the comment on the
    method for why. Every ceiling, every crossed-ceiling report, the refusal
    message and the re-estimated narrowing search stay the catalog's own, which
    is the point: the explicit path cannot drift away from the category path,
    because it is the same code.

    The import is deferred to call time so this module keeps its module-scope
    dependency surface to `base.chunking`; the class is built once and cached.
    """
    global _EXPLICIT_CATALOG_CLASS
    if _EXPLICIT_CATALOG_CLASS is None:
        import datetime

        from quantlab.universe import UniverseCatalog

        class _ExplicitSymbolCatalog(UniverseCatalog):
            def __init__(self, symbols: int):  # noqa: D107 - see factory
                # No `super().__init__`: this view never reads the reference
                # table, so it needs no backend and no config, and requiring
                # one would make `--symbols AAPL` fail on a machine that has
                # never built universe.parquet.
                self._explicit_symbols = symbols

            def _validate_category(self, category: str) -> None:
                # This view resolves NO roster from the reference table, so
                # `EXPLICIT_SYMBOLS_CATEGORY` is a token it can legitimately be
                # asked about and there is nothing to validate it against.
                #
                # CURRENTLY UNREACHABLE, and deliberately kept. The validator
                # call this override was written to intercept belongs to the
                # BASE `_roster_window_profile`, which opens with
                # `self._validate_category(category)` and would reject the
                # sentinel with "Unknown universe category '(explicit
                # --symbols list)'" before a single byte was fetched. But this
                # class overrides `_roster_window_profile` WHOLESALE, and that
                # override makes no validator call -- so on this view nothing
                # reaches here. The base's other two call sites
                # (`get_symbols_in_range`, `get_symbols_as_of`) read the
                # reference table this view does not have and are never entered
                # on the pricing path, whose only entry point is
                # `assert_acquisition_volume_fits` ->
                # `estimate_acquisition_volume` -> the override below.
                #
                # KEEP IT anyway. (i) It is the guard that makes the sentinel
                # safe IF this view's `_roster_window_profile` is ever narrowed
                # to delegate to `super()` -- which is exactly the drift this
                # class exists to prevent, per the factory docstring's "the
                # explicit path cannot drift away from the category path".
                # (ii) Deleting it removes that protection in exchange for
                # nothing measurable. What was wrong here was the REASON given,
                # not the code.
                #
                # Every OTHER token still goes to the base check, so a
                # `--limit`-truncated REAL category (which `volume_pricing`
                # reports under its own name) keeps the typo protection.
                # `known_categories()` reads the two class-level fetcher
                # registries, never the backend this view does not have, so
                # delegating is safe without a config.
                if category != EXPLICIT_SYMBOLS_CATEGORY:
                    super()._validate_category(category)

            def _roster_window_profile(
                self,
                category: str,
                start_date: str,
                end_date: str,
                bars_per_day: int = 1,
            ) -> dict:
                # The signature MIRRORS the base's, `bars_per_day` included.
                # This override is what the acquisition-volume guard reaches
                # on the explicit path, and an override that dropped the
                # keyword would raise TypeError on the one path that matters --
                # an explicit `--symbols` list at `--frequency 1m`.
                # Rebound, exactly as the base method does: the validator
                # NORMALISES (`"20180101"` -> `"2018-01-01"`), and a caller
                # that validates and then uses its own raw string is the CR-01
                # bug. Nothing here compares dates lexicographically today,
                # but this override exists precisely so the explicit path
                # cannot drift from the category path.
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
                # `observed_cells == dense_cells`, density 1.0. The catalog's
                # 0.368 density is a property of a survivorship-bias-free
                # ROSTER over a decade -- most of it delisted for most of the
                # window. A hand-named symbol list carries no such structure,
                # and assuming it does would UNDERSTATE the fetch by ~2.7x,
                # which is the wrong direction for a guard.
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
    """`(start, end, assumed)` for the estimate. Never written back onto
    `args` -- an assumption made to size a fetch must not silently become the
    window that fetch actually requests."""
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
    """`(pricing, category, start_date, end_date, window_assumed)` for the
    guard call the CALLER makes.

    This helper deliberately does NOT call the guard. Each ingest script names
    `assert_acquisition_volume_fits` at its own call site, so a reader of the
    script -- and a grep across the entry points -- sees the guard where the
    decision to fetch is made, rather than one level of indirection away. What
    IS shared is the part that would otherwise be copied three times and drift:
    which object prices the fetch, and over which window.

    Prices the ACTUAL roster. An explicit `--symbols` list, or a category
    truncated by `--limit`, is sized from its real symbol count rather than
    exempted -- an explicit 15,000-symbol list costs exactly what the same
    roster resolved from a category costs. Only a whole, untruncated category
    is priced through the catalog's own density-adjusted estimate.
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
    """Print what the user just committed to.

    **Reached only when the guard ADMITS the fetch.** Every call site is
    structured as `print_volume_estimate(pricing.assert_acquisition_volume_fits(
    ...), ...)`, so the guard is an ARGUMENT: when it raises, this function is
    never invoked. That is deliberate and it is not a gap -- the refusal message
    already carries the same arithmetic (symbols, trading days, rows, requests,
    GiB, hours) plus every ceiling crossed and a concrete narrowing that would
    fit, so a refused user sees MORE than this prints, not less.

    An earlier version of this docstring claimed the estimate was "printed
    whether or not the guard refused". It was not, and could not be, in this
    structure; the claim is corrected rather than the structure changed, and
    `test_a_refusal_prints_no_estimate_and_carries_the_numbers_itself` pins
    which of the two is actually true (WR-07).

    What this adds over the refusal message is the ADMITTED case: a user who
    proceeds sees the numbers they proceeded with, and the `--force-volume`
    line distinguishes "under every ceiling" from "over one and overridden",
    which the estimate alone cannot say.
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
    """Print an admitted `SqlVolumeGuard` estimate (D-16).

    The WRDS twin of `print_volume_estimate`, under the same rule: it is
    reached only when the guard ADMITTED the pull -- a refusal carries its own
    numbers (and a date segment that fits) in the exception message. It takes
    the returned dict alone, so this module imports nothing new and needs no
    connection; it prints counts, dates and ceilings, never a credential.

    The bucket line is labelled from `estimate['unit']` -- `trading days:`
    for a TAQ pull (whose pages ARE days), `year buckets:` for a CRSP one
    (whose pages are calendar years). An estimate with no `unit` key comes
    from a caller older than 03.10-10 and is a TAQ estimate by construction,
    so it reads `trading days:` exactly as it always did.
    """
    # Imported at call time for the reason `apply_data_dir` defers `config`:
    # this module's module-scope project imports stay pinned at quantlab.base.*.
    from quantlab.acquisition.sql_volume import SqlVolumeGuard

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
    """Render the `ConversionResult` `quantlab.registry.convert()`
    returns.

    Beside `print_volume_estimate` and shaped the same way -- `print_fn`
    injected LAST, the input returned so a call site can compose -- because
    the same rule applies: `quantlab/acquisition/` and `quantlab/base/`
    produce VALUE objects, and every print of one lives in this module. Three
    shells render the same outcome, so a third copy of these lines in a third
    shell is the duplication that hoisting into this module (D-13) was about.

    **Every line comes off the RETURNED object, none from the config and none
    from a read-back of the store.** That is half of what `ConversionResult`
    exists for (03.5 D-04): the written path is echoed rather than re-derived,
    so a run that wrote somewhere other than where the caller expected says
    so, and `windows_skipped`/`resumed` describe THIS run rather than what the
    ledger happens to hold.

    Takes the CONSTRUCTED object and imports nothing, exactly as
    `refuse_conversion_without_raw_data` above does: this module's module-scope
    project dependency surface is pinned at `quantlab.base.*` by its own
    module docstring, and naming `ConversionResult` for an annotation would
    widen it for no behaviour.

    Prints paths, integer counts and booleans only (T-03.5-17). There is no
    vendor response body and no credential in a `ConversionResult` to leak,
    and this renderer adds no field of its own.
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
        # Said out loud, because "0 windows written" and "this run had nothing
        # left to do" read identically in a log otherwise -- and one of them
        # is a bug report.
        print_fn(
            "  resumed:           yes -- windows the chunk ledger already "
            "recorded were skipped, not rewritten"
        )
    if result.peak_window_bytes is not None:
        # Prediction beside outcome, and only when there IS an outcome: a
        # fully-resumed run materialises no window, so it has no observed peak
        # and printing `0.00 GiB` would claim a measurement nobody took.
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
