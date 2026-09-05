"""Build/refresh the survivorship-bias-free, point-in-time US-equity universe.

Thin CLI entry point -- all logic lives in `acquisition.universe.
UniverseCatalog` (02-08-PLAN.md); this script is glue only. Requires no
Tiingo API key: the NASDAQ roster and S&P 500 membership sources are both
public/unauthenticated.

Usage:
    uv run python refresh_us_equity_universe.py
"""

from acquisition.universe import UniverseCatalog
from config import universe_config

if __name__ == "__main__":
    config = universe_config()
    UniverseCatalog(config).build().save()
    print(f"Universe table refreshed at: {config.output_path}")
