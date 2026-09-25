"""Concrete point-in-time index-membership panels, one class per index.

Each class binds one membership source to the densification in
``quantlab/base/constituent.py`` by implementing its two hooks. Two sources
cover the same two indexes: the Wikipedia-based pair
(``SP500ConstituentDataset``, ``Nasdaq100ConstituentDataset``) replays a
public change log and produces a ticker-keyed ``symbol`` axis, while the
CRSP/Compustat pair (``CrspSP500ConstituentDataset``,
``CompustatNasdaq100ConstituentDataset``) reads the local CRSP reference
tier and produces an int64 PERMNO axis that lines up with the CRSP price
panel column for column. ``CrspMarketConstituentDataset`` is the
whole-market variant: every security CRSP lists, rather than an index.

``config.cache_dir`` is the local directory a panel is answered from: the
scraped-HTML cache for the Wikipedia pair, the CRSP reference directory for
the CRSP classes. The CRSP classes read parquet on disk and need no WRDS
credential.
"""

import polars as pl

from quantlab.universe import (
    Nasdaq100MembershipFetcher,
    SP500MembershipFetcher,
)
from quantlab.base.config import ConstituentDatasetConfig
from quantlab.base.constituent import IndexConstituentDataset
from quantlab.dataset.crsp.market import CrspMarketRoster
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference


def _rename_permno_to_symbol(permno_intervals: pl.DataFrame) -> pl.DataFrame:
    """Rename the ``permno`` column to ``symbol`` for ``_densify``.

    A rename and nothing else: the column is already ``pl.Int64``, which is
    the dtype the CRSP price panel's symbol axis carries, so no cast or
    re-derivation is needed.
    """
    return permno_intervals.rename({"permno": "symbol"})


class SP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel from a public change log.

    Membership intervals come from ``SP500MembershipFetcher``, which replays
    Wikipedia's historical-components change log against a current
    constituents CSV. Coverage starts on 1976-07-01, the earliest row of
    that log, so the panel's left edge never precedes it. The ``symbol``
    axis holds tickers. The source URLs and the coverage constant live on
    the fetcher, not here.

    Examples
    --------
    Building the panel fetches the change log over the network (or reads
    the snapshot cached under ``cache_dir``):

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/sp500_constituent.zarr",
    ...     cache_dir="data/reference/_cache",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     as_of="2024-12-31",
    ... )
    >>> SP500ConstituentDataset(config).from_raw_data().save()
    >>> panel = SP500ConstituentDataset(config).read().get_xarray_dataset()
    """

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        """Create the dataset from a ``ConstituentDatasetConfig``."""
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        """Return the fetcher's coverage start, ``"1976-07-01"``."""
        return SP500MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the ticker-keyed membership intervals from the fetcher."""
        return SP500MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()


class Nasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 membership panel from a public change log.

    Membership intervals come from ``Nasdaq100MembershipFetcher``, which
    replays Wikipedia's historical-components change log against a scraped
    current-constituents snapshot. Coverage starts on 2007-02-01, the
    earliest row of that log, about thirty-one years later than the S&P 500
    panel; the two panels keep separate stores so that neither implies
    coverage the other lacks. The ``symbol`` axis holds tickers.

    The index carries several share classes of some issuers (for example
    GOOGL and GOOG), so the member count on a given day exceeds one hundred.

    Examples
    --------
    Building the panel fetches the change log over the network (or reads
    the snapshot cached under ``cache_dir``):

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/nasdaq100_constituent.zarr",
    ...     cache_dir="data/reference/_cache",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     as_of="2024-12-31",
    ... )
    >>> Nasdaq100ConstituentDataset(config).from_raw_data().save()
    """

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        """Create the dataset from a ``ConstituentDatasetConfig``."""
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        """Return the fetcher's coverage start, ``"2007-02-01"``."""
        return Nasdaq100MembershipFetcher.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the ticker-keyed membership intervals from the fetcher."""
        return Nasdaq100MembershipFetcher(
            cache_dir=self.config.cache_dir
        ).build_intervals()


class CrspSP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership panel from CRSP's own spells.

    Intervals come from ``CrspMembership`` over the CRSP reference tier at
    ``config.cache_dir`` (the ``dsp500list_v2`` table). Coverage starts on
    1925-12-31. The ``symbol`` axis is the int64 PERMNO, the same identifier
    the CRSP price panel keys its columns by, so a mask built here lines up
    with that panel with no ticker rule in between; it is not interchangeable
    with the ticker axis of ``SP500ConstituentDataset``.

    Every interval carries an explicit end no later than the reference
    tier's product end, so the panel's right edge is that product end and
    never today.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/crsp_sp500.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ... )
    >>> panel = CrspSP500ConstituentDataset(config).from_raw_data()
    >>> panel.get_xarray_dataset()["symbol"].dtype
    dtype('int64')
    """

    #: The ``CrspMembership`` universe this class binds to, named once so the
    #: two hooks cannot drift apart.
    INDEX = CrspMembership.SP500

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        """Create the dataset from a ``ConstituentDatasetConfig``."""
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        """Return ``CrspMembership``'s coverage start for the S&P 500."""
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed intervals, scoped to the config window."""
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
                window=(self.config.start_date, self.config.end_date),
            )
        )


class CompustatNasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 membership panel from Compustat via CCM.

    Intervals come from ``CrspMembership`` over the reference tier at
    ``config.cache_dir``: Compustat's index-constituent history, with each
    ``(gvkey, iid)`` spell mapped to a PERMNO through the CRSP/Compustat
    link table. The ``symbol`` axis is the int64 PERMNO. Coverage starts on
    1995-01-01, which is where Compustat's history begins rather than where
    those memberships did; the panel never starts earlier.

    A membership spell with no PERMNO link is refused by default, because
    dropping it would silently shrink the universe. Pass
    ``kwargs={"allow_unlinked": True}`` to keep the linked days and read the
    rest from the membership report; the option lives in the config so a run
    that tolerated the gap says so in its own ``config.json``. Only gaps
    inside this panel's configured window count.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/compustat_nasdaq100.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     kwargs={"allow_unlinked": True},
    ... )
    >>> CompustatNasdaq100ConstituentDataset(config).from_raw_data().save()
    """

    #: The ``CrspMembership`` universe this class binds to.
    INDEX = CrspMembership.NASDAQ100

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        """Create the dataset from a ``ConstituentDatasetConfig``."""
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        """Return ``CrspMembership``'s coverage start for the Nasdaq-100."""
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed intervals, scoped to the config window."""
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
                # The panel's own edges are a subset of this window, so a link
                # gap outside it cannot affect any cell the panel produces.
                window=(self.config.start_date, self.config.end_date),
            )
        )


class CrspMarketConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time whole-market listing panel from CRSP.

    Not an index: ``is_member`` answers "was this security listed, and of
    the requested type, on this day" for every security in the CRSP
    reference tier at ``config.cache_dir``, through ``CrspMarketRoster``. A
    whole-market price panel is thousands of columns wide and mostly NaN on
    any given day; this mask lets a consumer tell "not listed" from "listed,
    no trade". The ``symbol`` axis is the int64 PERMNO, matching the CRSP
    price panel.

    Coverage starts where CRSP's daily prices begin (1925-12-31), and every
    interval ends no later than the reference tier's product end, so the
    right edge is never today. The security filter is part of the panel's
    identity: it is read from ``kwargs["security_filter"]`` (default
    ``"equity_common"``) and therefore recorded in the run's ``config.json``.

    Examples
    --------
    Needs a CRSP reference directory on disk; no WRDS connection is made:

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/reference/crsp_market.zarr",
    ...     cache_dir="downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2020-01-01",
    ...     end_date="2024-12-31",
    ...     kwargs={"security_filter": "equity_common"},
    ... )
    >>> CrspMarketConstituentDataset(config).from_raw_data().save()
    """

    #: Start of CRSP's daily file; a whole-market universe cannot be answered
    #: before the prices exist.
    PIT_COVERAGE_START = "1925-12-31"

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        """Create the dataset from a ``ConstituentDatasetConfig``."""
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        """Return ``PIT_COVERAGE_START``."""
        return self.PIT_COVERAGE_START

    def _build_intervals(self) -> pl.DataFrame:
        """Return the PERMNO-keyed listing spans under the security filter."""
        return _rename_permno_to_symbol(
            CrspMarketRoster(CrspReference(self.config.cache_dir)).permno_intervals(
                security_filter=(self.config.kwargs or {}).get(
                    "security_filter", "equity_common"
                )
            )
        )
