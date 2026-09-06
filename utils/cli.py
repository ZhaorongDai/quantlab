"""Argument groups and roster resolution shared by the ingest entry points.

D-14 keeps one top-level script per source -- `ingest_tiingo.py`,
`ingest_us_equity.py`, `ingest_alpaca.py` -- and forbids a third copy of the
`--symbols` / `--universe` / `--as-of-date` / `--start-date` / `--end-date` /
`--chunk` parsing those scripts share. The definitions live HERE once; each
script adds only the flags that are genuinely its own.

This module is deliberately dependency-light in the `utils/` tradition
(`utils/file.py`, `utils/timer.py`): it builds `argparse` groups and resolves a
roster through a catalog handed to it. It constructs no acquisition client,
opens no file and issues no request. The one project import it takes at module
scope is `base.chunking.TimeChunkPlanner`, itself a leaf, because `--chunk`'s
`choices` must be DERIVED from `GRANULARITIES` rather than restated.

**The one thing this module must not unify.** `ingest_tiingo.py` resolves
point-in-time membership on a single day; `ingest_us_equity.py` resolves
interval OVERLAP across a window. That is a deliberate semantic difference, not
duplication -- see `resolve_symbols`, whose `mode` is keyword-only and has no
default for exactly that reason.
"""

import argparse
from typing import Literal

from base.chunking import TimeChunkPlanner

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
        help="Comma-separated symbols (e.g. AAPL,MSFT). Mutually exclusive with --universe.",
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


def add_chunk_args(
    parser: argparse.ArgumentParser,
    *,
    default: str = "year",
) -> argparse.ArgumentParser:
    """Add `--chunk`, with `choices` derived from `TimeChunkPlanner`.

    Derived, never restated: a granularity added to `GRANULARITIES` and its
    `_period_key` must not need a second edit here to become selectable.
    """
    parser.add_argument(
        "--chunk",
        type=str,
        choices=list(TimeChunkPlanner.GRANULARITIES),
        default=default,
        help=(
            "Time granularity of one --to-zarr conversion window (default "
            "year). Finer windows use less peak RAM and give a finer resume "
            "granularity, at the cost of more append round trips. Pass "
            "'month' for a dense year the per-chunk sizing guard refuses."
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
            "Process only the first N resolved symbols. For smoke-testing the "
            "pipeline end to end before committing to the full roster."
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


def add_volume_guard_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add the pre-flight volume guard's two flags: `--force-volume` and
    `--rows-per-symbol-day`.

    Defined HERE rather than in each script so the flag name and the help text
    exist once (D-14). `--force-volume` is an EXPLICIT, visible opt-out: it is
    a flag a user types, never an environment variable and never a config key
    that could turn the guard off for a whole machine without anyone noticing.

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
