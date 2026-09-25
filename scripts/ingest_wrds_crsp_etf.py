"""Download ETF daily data from WRDS CRSP and save each ETF to its own Zarr store.

Each ETF's ``crsp_a_stock.dsf_v2`` rows are pulled by PERMNO into the shared
CRSP raw tier and converted into ``data/us_equity/1d/wrds_crsp_<name>_1d.zarr``
(``CrspDatasetConfig.etf_benchmark``), the store a backtest reads as
``benchmark_dataset``. ``spy`` and ``qqq`` are known by name; any other ETF is
given as ``name=PERMNO``.

Credentials: ``WRDS_USERNAME`` in the environment, the password in
``~/.pgpass``; neither is an argument.

Usage:
    uv run python scripts/ingest_wrds_crsp_etf.py --etf spy,qqq \\
        --start-date 2000-01-01 --end-date 2025-12-31
    uv run python scripts/ingest_wrds_crsp_etf.py --etf myetf=12345 \\
        --start-date 2010-01-01 --end-date 2025-12-31
"""

import argparse

from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.base.config import QQQ_PERMNO, SPY_PERMNO, CrspDatasetConfig
from quantlab.config import get_data_root
from quantlab.utils.cli import add_data_dir_arg, add_window_args, apply_data_dir

SOURCE = DataSourceRegistry.get("wrds")
CAPABILITY = ("us_equity", "1d", "crsp_daily")
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: ETFs that can be named without a PERMNO.
KNOWN_ETFS = {"spy": SPY_PERMNO, "qqq": QQQ_PERMNO}


def _parse_etfs(parser: argparse.ArgumentParser, value: str) -> dict[str, str]:
    """Return ``{name: PERMNO}`` for ``--etf`` (``spy`` or ``name=PERMNO``)."""
    etfs = {}
    for token in filter(None, (part.strip() for part in value.split(","))):
        name, _, permno = token.partition("=")
        name = name.strip().lower()
        permno = permno.strip() or KNOWN_ETFS.get(name, "")
        if not permno.isdigit():
            parser.error(
                f"--etf {token!r}: give a known ETF ({', '.join(KNOWN_ETFS)}) "
                f"or name=PERMNO."
            )
        etfs[name] = permno
    if not etfs:
        parser.error("--etf names no ETF.")
    return etfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download ETFs from WRDS CRSP by PERMNO, one Zarr store each."
    )
    parser.add_argument(
        "--etf", required=True, help="Comma-separated: spy, qqq or name=PERMNO."
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Continue from each ETF's watermark."
    )
    add_window_args(parser)
    add_data_dir_arg(parser)
    args = parser.parse_args()
    apply_data_dir(args)  # before any path is derived from the data root
    etfs = _parse_etfs(parser, args.etf)
    if not args.start_date or not args.end_date:
        parser.error("--start-date and --end-date are both required.")

    # Imported here so tests can replace the session with an offline stand-in.
    from quantlab.acquisition.wrds.crsp import CrspQueries
    from quantlab.acquisition.wrds.crsp_reference import CrspReferenceTables
    from quantlab.acquisition.wrds.taq import WrdsSession

    session = WrdsSession.shared()
    try:
        # The download clips the window to the CRSP release end itself; the
        # conversion window is clipped the same way.
        product_end = CrspQueries.product_end(session)
        end_date = min(args.end_date, product_end.isoformat())

        # 1. Download the ETFs' daily rows by PERMNO.
        acq_config = SOURCE.config_factory_for(*CAPABILITY)(
            symbols=tuple(sorted(set(etfs.values()), key=int)),
            start_date=args.start_date,
            end_date=end_date,
            kwargs={"clip_to_product_end": True},
        )
        # The conversion reads the ticker history from the reference tables.
        reference_dir = ACQ.reference_dir_for(acq_config)
        CrspReferenceTables(session, reference_dir).pull(
            product_end=product_end, include_sp500=False
        )
        result = run(SOURCE, acq_config, refresh=args.refresh)
        print(f"{len(result.succeeded)} ETF(s) downloaded, {len(result.failures)} failed")

        # 2. Save each ETF to its own store.
        data_dir = get_data_root() / "data" / "us_equity" / "1d"
        for name, permno in etfs.items():
            config = CrspDatasetConfig.etf_benchmark(
                permno=permno,
                zarr_file_path=str(data_dir / f"wrds_crsp_{name}_1d.zarr"),
                raw_data_dir_path=acq_config.raw_data_dir_path,
                reference_dir=str(reference_dir),
                start_date=args.start_date,
                end_date=end_date,
            )
            convert(SOURCE, config, data_type="crsp_daily")
            print(f"{name.upper()} (PERMNO {permno}) -> {config.zarr_file_path}")
    finally:
        WrdsSession.close_shared()
