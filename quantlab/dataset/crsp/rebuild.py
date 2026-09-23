"""Rebuild the CRSP Stock v2 daily panel from its raw tier (phase 03.11 W0).

The concrete half of `quantlab/base/rebuild.py`: which sidecars the CRSP store
owns, which converter writes it, and what to count once it is written. The
generic four-step order and every refusal live in the ABC; nothing here
re-states them.

**This path is OFFLINE.** `quantlab/registry.py:convert()` says so
word for word (registry.py:565-571): "`run()` downloads to the raw parquet tier
and stops; this reads that tier and writes the Zarr store... nothing here
touches a vendor client, an endpoint, or a credential." So a rebuild needs no
WRDS account, no Duo push and no network -- only the raw and reference tiers
already on disk. That is what makes W0 a precondition one can actually satisfy
rather than a task that waits on a credential.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from quantlab.base.rebuild import BaseStoreRebuilder
from quantlab.dataset._support.cleaning import REQUIRED_COLUMNS

#: Every sidecar that belongs to a CRSP store, as a suffix on the store path.
#:
#: - `.chunks.json` -- `base/chunking.py:ChunkLedger`'s record of which windows
#:   were written, which the next append reconciles against.
#: - `.crsp_adjustment.json` -- the window quadruple the adjusted series used
#:   to be anchored against. NO LONGER GENERATED: 03.12 moved the anchor to
#:   each PERMNO's first usable row, which does not move when the window is
#:   extended forward, so the machinery that recorded and compared it is gone.
#:   It stays on this list for the same reason `.crsp_symbology_report.json`
#:   does -- a suffix dropped from here would leave a file describing a deleted
#:   mechanism sitting beside a store it never described.
#: - `.crsp_filter_report.json` -- what the security filter dropped and what an
#:   explicit roster overrode (`roster_overrides`). The D-17 audit artefact.
#: - `.crsp_symbology_report.json` -- every ticker identity decision.
#: - `.crsp_tickers.json` -- the PERMNO -> period-correct ticker interval table
#:   a display layer reads to spell the int64 axis for a human (03.11-09,
#:   D-03). It names the PANEL's PERMNOs, so a rebuild that narrowed or widened
#:   the roster leaves the previous panel's names behind unless it is cleared.
#:
#: **All five must be DELETED before a rebuild, not just the store.**
#: `dataset/crsp.py:_write_identity_reports` opens with a store-exists guard
#: (crsp/__init__.py:1414): when the store is already on disk it returns without
#: rewriting the reports. That guard is correct for an append -- it stops a
#: REFUSED re-conversion from replacing a surviving store's audit trail with
#: numbers for a panel that was never written -- but it means a rebuild that
#: removed only the `.zarr` directory finishes with audit files describing the
#: PREVIOUS panel beside the new one.
#:
#: `.crsp_symbology_report.json` stays on this list even though later plans in
#: this phase stop GENERATING it. A suffix dropped from here would leave a file
#: describing a deleted mechanism sitting beside a store it never described --
#: the exact confidently-wrong audit trail the clearing rule exists to prevent.
CRSP_SIDECAR_SUFFIXES: tuple[str, ...] = (
    ".chunks.json",
    ".crsp_adjustment.json",
    ".crsp_filter_report.json",
    ".crsp_symbology_report.json",
    ".crsp_tickers.json",
)


class CrspStoreRebuilder(BaseStoreRebuilder):
    """Re-convert one CRSP store from the raw tier and measure the result."""

    SIDECAR_SUFFIXES = CRSP_SIDECAR_SUFFIXES

    def _required_inputs(self) -> tuple[Path, ...]:
        """The raw vendor directory and the reference directory.

        Both are resolved against `data_root`; an already-absolute config path
        is used as-is (`Path.__truediv__` yields the absolute operand), so a
        config written with absolute paths and one written with
        repository-relative paths both land in the same place -- and neither
        can fall through to the current working directory.

        **Two path traps, each of which fails a first rebuild attempt:**

        1. `raw_data_dir_path` must TERMINATE at `/wrds`. Passing the parent
           `.../wrds_crsp` is refused by `dataset/stock.py:107`'s
           `_assert_vendor_root`: "The raw path must TERMINATE at the vendor
           segment (D-11)." A root one level up would walk into every vendor
           directory beneath it and merge them with no provenance.
        2. `reference_dir` is a SIBLING of `wrds_crsp` named `_reference`, NOT
           a child of `wrds`. The reference tier is pulled by a different step,
           which is exactly why `CrspDatasetConfig` makes it a required field
           rather than deriving it from the raw root.
        """
        return (
            self.data_root / str(self.config.raw_data_dir_path),
            self.data_root / str(self.config.reference_dir),
        )

    def _convert(self) -> object:
        """Re-run the offline conversion; return its `ConversionResult`.

        Imported inside the body, the convention `tests/conftest.py` follows
        for the same reason: keeping `quantlab.acquisition.*` off this module's
        import graph so importing the rebuilder costs nothing.

        `on_new_listing="refuse"` is deliberate. The rebuild is supposed to
        reproduce a known panel; a listing that appeared out of nowhere means
        the raw tier is not the one the measurements were taken against, and
        refusing says so instead of silently widening the axis.

        Nothing in this call reaches the network. See the module docstring and
        `registry.py:565-571` for the verbatim guarantee.
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
        """The seven numbers phase 03.11 reasons about, from the written store.

        `structural_gaps` is computed the way
        `quantlab/dataset/_support/cleaning.py:validate_schema` computes its own structural mask
        -- a logical AND over `isnull()` of every column in `REQUIRED_COLUMNS`,
        imported from that module rather than re-listed here. A second copy of
        the rule would drift, and the whole force of the acceptance criterion
        is that `adj_close_nan == structural_gaps` compares against
        cleaning.py's OWN definition of an empty cell, not against a number
        this file chose.

        `anomaly_flag_true` degrades to 0 on a panel that has no
        `anomaly_flag` variable: a store written without it is a store with no
        anomalies recorded, which is a measurement, not an error.
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
        """`(dict(panel.sizes), len(panel.data_vars))` of the written store."""
        panel = xr.open_zarr(self.store_path)
        try:
            return dict(panel.sizes), len(panel.data_vars)
        finally:
            panel.close()
