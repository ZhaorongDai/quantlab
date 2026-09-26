"""Shared argparse helpers for the download scripts under ``scripts/``.

A download script pulls raw market data from a vendor and converts it into a
Zarr store. The scripts share ``--data-dir`` (the per-run storage root) and
the rendering of the conversion result they print; this module defines both
once. Each script adds the flags that are its own.

The module is light at import time: it imports nothing from the project at
module scope, and ``quantlab.config`` is imported inside ``apply_data_dir``
when the flag is used.
"""

import argparse

#: Bytes in one GiB.
_GIB = 1024**3


def add_data_dir_arg(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add ``--data-dir``, the per-run storage root override, to ``parser``.

    Defined once here so the flag name, help text and precedence rules are
    identical on every entry point that offers it. See ``apply_data_dir``.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser to extend.

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
            "(raw downloads, watermarks, Zarr stores) is "
            "derived from it. Precedence is --data-dir > QUANTLAB_DATA_DIR > "
            "the repo-root data/ directory. The directory does not need to "
            "exist; the run creates what it needs."
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


def print_conversion_result(result, *, print_fn=print):
    """Print the ``ConversionResult`` returned by ``quantlab.registry.convert``.

    Every line comes off the returned object, not the config and not a
    read-back of the store, so the printed path is the one actually written
    and ``windows_skipped`` / ``resumed`` describe this run rather than
    whatever the chunk ledger (the resume record of written windows) holds.
    Only paths, integer counts and booleans are
    printed. The argument is left unannotated so this module does not import
    the result class.

    Parameters
    ----------
    result : ConversionResult
        The conversion result to render.
    print_fn : callable, default print
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
        # Without this line, "0 windows written" could mean either "nothing
        # left to do" or a bug.
        print_fn(
            "  resumed:           yes -- windows the chunk ledger already "
            "recorded were skipped, not rewritten"
        )
    if result.peak_window_bytes is not None:
        # A fully resumed run loads no window, so it has no measured peak;
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
