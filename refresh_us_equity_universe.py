"""Build/refresh the survivorship-bias-free, point-in-time US-equity universe.

Thin CLI entry point -- all logic lives in `quantlab.acquisition.universe.
UniverseCatalog` (02-08-PLAN.md); this script is glue only. Requires no
Tiingo API key: the NASDAQ roster and S&P 500 membership sources are both
public/unauthenticated.

By default this refuses to persist a table built from a fetcher's cached
snapshot: `save()` overwrites `universe.parquet` in place, so a stale
reconstruction written there is indistinguishable from a fresh one and a
permanently-broken source would silently freeze the universe at the cache
date. Pass --allow-stale to accept a knowingly-frozen table.

Usage:
    uv run python refresh_us_equity_universe.py
    uv run python refresh_us_equity_universe.py --allow-stale
    uv run python refresh_us_equity_universe.py --data-dir /Volumes/BigDisk
"""

import argparse

from quantlab.acquisition.universe import UniverseCatalog
from quantlab.config import universe_config
from quantlab.utils.cli import add_data_dir_arg, apply_data_dir


def _build_arg_parser() -> argparse.ArgumentParser:
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

    # Before `universe_config()`, and the position is load-bearing: the config
    # factories snapshot their paths as strings at construction time, so a root
    # override applied afterwards silently does nothing (DDIR-04).
    apply_data_dir(args)

    config = universe_config()
    UniverseCatalog(config).build(allow_stale=args.allow_stale).save()
    print(f"Universe table refreshed at: {config.output_path}")
