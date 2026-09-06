"""Concrete index-membership panel datasets, one class per index.

Every index binds itself to the shared densification machinery in
`base/constituent.py` by implementing exactly two hooks. Nothing else lives
here, and nothing index-specific lives in `base/` -- which is what makes
adding an index a `dataset/` change only (DATA-06).
"""

import polars as pl

from acquisition.universe import SP500MembershipFetcher
from base.config import ConstituentDatasetConfig
from base.constituent import IndexConstituentDataset


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
