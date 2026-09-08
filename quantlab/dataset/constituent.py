"""Concrete index-membership panel datasets, one class per index.

Every index binds itself to the shared densification machinery in
`base/constituent.py` by implementing exactly two hooks. Nothing else lives
here, and nothing index-specific lives in `base/` -- which is what makes
adding an index a `dataset/` change only (DATA-06).
"""

import polars as pl

from quantlab.acquisition.universe import (
    Nasdaq100MembershipFetcher,
    SP500MembershipFetcher,
)
from quantlab.base.config import ConstituentDatasetConfig
from quantlab.base.constituent import IndexConstituentDataset


class SP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel (DATA-05, D-01, D-04).

    The first of D-01's two concrete daily constituent dataset classes,
    carrying date plus ticker as an `is_member` boolean grid over dims
    `(timestamp, symbol)`, persisted to Zarr.

    Source: Wikipedia's "Historical components of the S&P 500" change log,
    replayed forward and anchored against a current-constituents CSV. Coverage
    starts 1976-07-01 -- the verified earliest row of that change log, not the
    page's own prose claim of 1963. Membership before that date cannot be
    answered from the source, so the panel's left edge never precedes it.

    The two source URLs and the coverage constant all live on
    `SP500MembershipFetcher`, deliberately not here: this class is the binding
    between an index and the panel machinery, not a second place a maintainer
    has to remember to update when a source moves.
    """

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        return SP500MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return SP500MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()


class Nasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 (NDX) membership panel (DATA-05, D-01,
    D-04).

    The second of D-01's two concrete daily constituent dataset classes,
    carrying date plus ticker as an `is_member` boolean grid over dims
    `(timestamp, symbol)`, persisted to its own Zarr store.

    Sources: Wikipedia's "Historical components of the Nasdaq-100" change log,
    replayed forward and anchored against a commercial current-constituent
    snapshot (stockanalysis.com; slickcharts.com is the documented fallback).
    Wikipedia's own `Nasdaq-100` page renders its components through a navbox
    template with no parseable constituents table, so unlike the S&P 500 --
    whose anchor is a GitHub-hosted CSV -- this index has no free, structured
    anchor available.

    Coverage starts 2007-02-01, the verified earliest row of that change log
    (`LOGI` added / `CMVT` removed). That left edge is ~31 years LATER than
    `SP500ConstituentDataset`'s 1976-07-01, which is why the two panels get
    two separate stores: unioning them onto one timestamp axis would imply
    1976 Nasdaq-100 coverage that does not exist, and all-False rows read as
    "nobody was a member" rather than "unknown".

    The index carries multiple share classes for some issuers (GOOGL/GOOG,
    FOX/FOXA), so its member count on any given day exceeds one hundred -- an
    `== 100` expectation is wrong against correct data.

    Both source URLs, the coverage constant and the anchor shape guard live on
    `Nasdaq100MembershipFetcher`, deliberately not here: this class is the
    binding between an index and the panel machinery, not a second place a
    maintainer has to remember to update when a source moves.
    """

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        return Nasdaq100MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        return Nasdaq100MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()
