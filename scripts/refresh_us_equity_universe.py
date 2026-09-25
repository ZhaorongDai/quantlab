"""Build or refresh the point-in-time US-equity universe table.

The table is the survivorship-free reference that the ingest scripts resolve
their rosters from. All of the work happens in
``quantlab.universe.UniverseCatalog``; this script builds the catalog, saves
it and prints where it landed. No API key is needed: the NASDAQ roster and
the S&P 500 membership sources are public.

By default the run refuses to persist a table built from a fetcher's cached
snapshot, because ``save()`` overwrites ``universe.parquet`` in place and a
stale reconstruction is indistinguishable from a fresh one. Pass
``--allow-stale`` to accept a knowingly frozen table.

Usage:
    uv run python scripts/refresh_us_equity_universe.py
    uv run python scripts/refresh_us_equity_universe.py --allow-stale
    uv run python scripts/refresh_us_equity_universe.py --data-dir /Volumes/BigDisk
"""

import argparse

from quantlab.universe import UniverseCatalog
from quantlab.config import universe_config
from quantlab.utils.cli import add_data_dir_arg, apply_data_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Build/refresh the point-in-time US-equity universe reference "
            "table. Requires no API key."
        )
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Persist even if a change-log fetch/parse failed and its cached "
            "snapshot was used. Off by default: a stale table written over "
            "universe.parquet looks exactly like a fresh one."
        ),
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Must run before ``universe_config()`` is called: the factories snapshot
    # their paths at construction time, so a later root override is ignored.
    apply_data_dir(args)

    config = universe_config()
    UniverseCatalog(config).build(allow_stale=args.allow_stale).save()
    print(f"Universe table refreshed at: {config.output_path}")
