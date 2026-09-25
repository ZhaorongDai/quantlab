"""Build or refresh the point-in-time US-equity universe table.

The universe table records which symbols belonged to each category (the
S&P 500, the Nasdaq-100, the whole NYSE/NASDAQ/AMEX listing and so on) and
over which dates. It is point-in-time: it can answer "who was a member on
2015-06-01", not only "who is a member today". It also keeps delisted
names, which avoids survivorship bias, the error of backtesting only on
companies that survived to the present. The ingest scripts resolve their
symbol rosters from this table.

All of the work happens in ``quantlab.universe.UniverseCatalog``; this
script builds the table, saves it as ``universe.parquet`` under the storage
root and prints where it landed. The storage root is ``--data-dir``, else
``QUANTLAB_DATA_DIR``, else the repository's ``data/`` directory. No API key
is needed, because the NASDAQ listing and the S&P 500 membership sources are
public.

By default the run refuses to save a table rebuilt from a cached copy of a
source that failed to download, because ``save()`` overwrites
``universe.parquet`` in place and a stale table looks exactly like a fresh
one. Pass ``--allow-stale`` to accept such a table knowingly.

Usage::

    uv run python scripts/refresh_us_equity_universe.py --help
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
            "Build or refresh the point-in-time US-equity universe table. "
            "Requires no API key."
        )
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Save the table even if downloading or parsing a membership "
            "change log failed and its cached copy was used instead. Off by "
            "default, because a stale table written over universe.parquet "
            "looks exactly like a fresh one."
        ),
    )
    add_data_dir_arg(parser)
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    # Must run before ``universe_config()`` is called: the factory copies the
    # data root into its paths when called, so a later override is ignored.
    apply_data_dir(args)

    config = universe_config()
    UniverseCatalog(config).build(allow_stale=args.allow_stale).save()
    print(f"Universe table refreshed at: {config.output_path}")
