"""Build/refresh the survivorship-bias-free, point-in-time US-equity universe.

Thin CLI entry point -- all logic lives in `acquisition.universe.
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
"""

import argparse

from acquisition.universe import UniverseCatalog
from config import universe_config

if __name__ == "__main__":
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
    args = parser.parse_args()

    config = universe_config()
    UniverseCatalog(config).build(allow_stale=args.allow_stale).save()
    print(f"Universe table refreshed at: {config.output_path}")
