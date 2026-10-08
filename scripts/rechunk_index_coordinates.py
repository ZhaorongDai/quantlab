"""Rewrite the index coordinates of existing Zarr stores in one chunk, in place.

A store created from a short window and grown by appends before #232 holds
its ``timestamp`` coordinate in many small chunks (one per bar when the first
window was one bar, as on the Sharadar stock stores), and every open of the
store reads each of them. ``XrBackend.append`` now keeps the coordinate in one
chunk; this script repairs a store written before that, through
``quantlab.backend.zarr.rechunk_index_coordinates``. Only the chunk grid of the
1-D index coordinates (``timestamp``, ``symbol``, ...) changes: no value,
attribute or data variable is touched, so the data fingerprints of the stores
are unchanged. A store already in one chunk is not written.

Do not run it while another process reads or writes the same store. A run
killed midway leaves ``.rechunk.tmp`` / ``.replaced.tmp`` sidecars beside the
store; the next run (or the next append to the store) finishes or rolls back
that swap before doing anything else.

Usage::

    uv run python scripts/rechunk_index_coordinates.py \\
        /data/quantlab/zarrs/sharadar_sp500_1d.zarr \\
        /data/quantlab/zarrs/sharadar_sep_1d.zarr

    # Every store directly under a directory:
    uv run python scripts/rechunk_index_coordinates.py /data/quantlab/zarrs/*.zarr

``--dry-run`` only reports the chunk count of each index coordinate.
"""

import argparse
import time
from pathlib import Path

import zarr

from quantlab.backend.zarr import (
    _index_coordinate_names,
    rechunk_index_coordinates,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build this script's argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite the 1-D index coordinates (timestamp, symbol, ...) of "
            "each Zarr store in one chunk, without changing any value."
        )
    )
    parser.add_argument("stores", nargs="+", type=Path, help="Zarr store directories.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report each index coordinate's chunk count and write nothing.",
    )
    return parser


def _coordinate_chunks(store: Path) -> dict[str, int]:
    """Return the number of chunks of each 1-D index coordinate of ``store``."""
    group = zarr.open_group(str(store), mode="r", use_consolidated=False)
    return {
        name: int(group[name].nchunks) for name in _index_coordinate_names(group)
    }


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()
    missing = [str(store) for store in args.stores if not store.is_dir()]
    if missing:
        parser.exit(1, f"no such store: {missing}\n")
    for store in args.stores:
        before = _coordinate_chunks(store)
        if args.dry_run:
            print(f"{store}: chunks {before}")
            continue
        started = time.perf_counter()
        rewritten = rechunk_index_coordinates(store)
        elapsed = time.perf_counter() - started
        print(
            f"{store}: rewrote {rewritten or 'nothing'}; chunks {before} -> "
            f"{_coordinate_chunks(store)} ({elapsed:.2f} s)"
        )
