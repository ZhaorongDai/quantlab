"""Rebuild a CRSP daily store from the raw and reference files already on disk.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A CRSP *store* is the
Zarr directory holding the converted daily price panel. *Sidecars* are the
small JSON files written next to it (``<store>.zarr.<suffix>``).

``CrspStoreRebuilder`` fills in the CRSP-specific parts of
``quantlab.base.rebuild.BaseStoreRebuilder``: it lists the sidecar files a
CRSP store owns, says which converter writes the store, and computes a few
quality counts once the store is written. The order of operations (check
inputs, back up, clear, convert, measure) and every refusal live in
``BaseStoreRebuilder``.

A rebuild is offline. It reads the raw parquet files and the downloaded
reference tables and writes the Zarr store. Nothing here uses a vendor
client, a network endpoint or a credential, so it needs no WRDS account.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from quantlab.base.rebuild import BaseStoreRebuilder
from quantlab.dataset._support.cleaning import REQUIRED_COLUMNS

#: Every sidecar that belongs to a CRSP store, as a suffix on the store path:
#:
#: - ``.chunks.json``: the chunk ledger, recording which time windows were
#:   written, which the next append checks against.
#: - ``.crsp_adjustment.json``: no longer written by the conversion. It is
#:   listed so a rebuild over a store written by an older version still
#:   removes it.
#: - ``.crsp_filter_report.json``: what the security filter dropped and what
#:   an explicit roster overrode.
#: - ``.crsp_symbology_report.json``: no longer written; listed for the same
#:   reason as the adjustment file.
#: - ``.crsp_tickers.json``: the table of PERMNO-to-ticker intervals that
#:   display code reads to print readable names for the integer axis. (A
#:   PERMNO is CRSP's permanent integer id for one security.)
#:
#: All five must be deleted before a rebuild, not only the store directory.
#: The conversion does not write its reports when the store already exists,
#: so an append never overwrites an existing store's audit files. A rebuild
#: that removed only the ``.zarr`` directory would therefore end with audit
#: files describing the old panel next to the new one. Leaving a suffix off
#: this list would likewise leave a stale file next to the new store.
CRSP_SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".chunks.json",
    ".crsp_adjustment.json",
    ".crsp_filter_report.json",
    ".crsp_symbology_report.json",
    ".crsp_tickers.json",
)


class CrspStoreRebuilder(BaseStoreRebuilder):
    """Re-convert one CRSP store from the raw files and measure the result.

    Constructor parameters are those of ``BaseStoreRebuilder``: a
    ``CrspDatasetConfig`` and ``data_root``, the directory that relative
    config paths are resolved against.

    Examples
    --------
    >>> from quantlab.base.config import CrspDatasetConfig
    >>> from quantlab.dataset.crsp.rebuild import CrspStoreRebuilder
    >>> config = CrspDatasetConfig(
    ...     zarr_file_path="data/data/us_equity/1d/crsp.zarr",
    ...     raw_data_dir_path="data/downloads/us_equity/1d/wrds_crsp/wrds",
    ...     reference_dir="data/downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2024-01-01",
    ...     end_date="2024-12-31",
    ...     security_filter="equity_common",
    ...     roster_universe="crsp_sp500",
    ... )
    >>> rebuilder = CrspStoreRebuilder(config, data_root="/path/to/repo")
    >>> [p.name for p in rebuilder.sidecar_paths()][:2]
    ['crsp.zarr.chunks.json', 'crsp.zarr.crsp_adjustment.json']
    >>> measurement = rebuilder.rebuild(backup_dir=Path("/path/to/backup"))

    The last line needs real raw and reference files under ``data_root``;
    without them ``rebuild`` refuses before deleting anything.
    """

    SIDECAR_SUFFIXES = CRSP_SIDECAR_SUFFIXES

    def _required_inputs(self) -> tuple[Path, ...]:
        """Return the raw vendor directory and the reference directory.

        Both are resolved against ``data_root``, and an absolute config path
        is used as it is. Configs with absolute and with repository-relative
        paths therefore point to the same place, and neither depends on the
        current working directory.

        Two paths are easy to get wrong. ``raw_data_dir_path`` must end at
        the ``/wrds`` vendor directory (the dataset refuses a root one level
        higher), and ``reference_dir`` is the ``_reference`` directory inside
        ``wrds_crsp``, next to ``wrds`` rather than inside it.

        Returns
        -------
        tuple of Path
            The raw directory and the reference directory.
        """
        return (
            self.data_root / str(self.config.raw_data_dir_path),
            self.data_root / str(self.config.reference_dir),
        )

    def _convert(self) -> object:
        """Re-run the offline conversion and return its ``ConversionResult``.

        The imports are inside the function so that importing the rebuilder
        does not load ``quantlab.acquisition``. ``on_new_listing="refuse"``
        is deliberate: a rebuild should reproduce a known panel, and a new
        listing appearing means the raw files are not the ones the earlier
        measurements were taken from.

        Returns
        -------
        ConversionResult
            The result returned by ``quantlab.registry.convert``.
        """
        from quantlab.registry import convert
        from quantlab.acquisition.wrds import WRDS_SOURCE

        return convert(
            WRDS_SOURCE,
            self.config,
            data_type="crsp_daily",
            granularity="year",
            on_new_listing="refuse",
        )

    def _measure(self) -> dict[str, int]:
        """Return seven quality counts read from the written store.

        ``structural_gaps`` counts cells with no bar, using the same rule as
        the cleaning step: every column in ``REQUIRED_COLUMNS`` is null
        there. The list is imported from the cleaning module rather than
        repeated. ``anomaly_flag_true`` is 0 for a panel with no
        ``anomaly_flag`` variable; that is a valid measurement, not an error.

        Returns
        -------
        dict of str to int
            The counts ``anomaly_flag_true``, ``adj_close_le_zero``,
            ``close_eq_zero``, ``adj_close_nan``, ``adj_volume_nan``,
            ``structural_gaps`` and ``symbol_count``.
        """
        panel = xr.open_zarr(self.store_path)
        try:
            structural_mask = None
            for column in REQUIRED_COLUMNS:
                is_null = panel[column].isnull()
                structural_mask = (
                    is_null
                    if structural_mask is None
                    else (structural_mask & is_null)
                )

            anomaly_flag_true = (
                int(panel["anomaly_flag"].sum())
                if "anomaly_flag" in panel.data_vars
                else 0
            )

            return {
                "anomaly_flag_true": anomaly_flag_true,
                "adj_close_le_zero": int((panel["adjClose"] <= 0).sum()),
                "close_eq_zero": int((panel["close"] == 0).sum()),
                "adj_close_nan": int(panel["adjClose"].isnull().sum()),
                "adj_volume_nan": int(panel["adjVolume"].isnull().sum()),
                "structural_gaps": (
                    0 if structural_mask is None else int(structural_mask.sum())
                ),
                "symbol_count": int(panel.sizes["symbol"]),
            }
        finally:
            panel.close()

    def _measure_dims(self) -> tuple[dict[str, int], int]:
        """Return the store's dimension sizes and its number of data variables."""
        panel = xr.open_zarr(self.store_path)
        try:
            return dict(panel.sizes), len(panel.data_vars)
        finally:
            panel.close()
