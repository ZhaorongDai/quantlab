"""Concrete index-membership panel datasets, one class per index.

Every index binds itself to the shared densification machinery in
`base/constituent.py` by implementing exactly two hooks. Nothing else lives
here, and nothing index-specific lives in `base/` -- which is what makes
adding an index a `dataset/` change only (DATA-06).

**Four classes, two SOURCES of the same two indexes.** The Wikipedia-based
pair replays a public change log; the CRSP-vendor pair reads the CRSP/
Compustat reference tier. They are separate classes with separate stores
rather than a source switch on one class, because their coverage starts and
their SYMBOL AXES differ: the CRSP pair's `symbol` is the int64 PERMNO, the
same identifier `CrspStockDataset` keys its price columns by since 03.11-03,
so a CRSP mask lines up with a CRSP panel with no derivation in between --
and a Wikipedia mask, whose axis is tickers, does not.

**`cache_dir` means "the local directory this universe is answered from",
and that is two different directories.** For the Wikipedia pair it is where
the fetcher caches scraped HTML; for the CRSP pair it is the reference tier
at `{downloads}/us_equity/1d/wrds_crsp/_reference/`. One field rather than
two, because a second directory field would have to be null for half the
classes.

**The CRSP pair reaches no acquisition module.** `CrspMembership` and
`CrspReference` are dataset-layer leaves over parquet, so a CRSP universe
resolves with no WRDS credential and no database driver. That is also why
`quantlab/universe.py` gained no category: its vocabulary is
Tiingo/Wikipedia-shaped (tickers, exchange filters) and its "imports no
acquisition module" rule is AST-enforced, so a CRSP category would have
coupled two vocabularies for no gain.
"""

import polars as pl

from quantlab.universe import (
    Nasdaq100MembershipFetcher,
    SP500MembershipFetcher,
)
from quantlab.base.config import ConstituentDatasetConfig
from quantlab.base.constituent import IndexConstituentDataset
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference


def _rename_permno_to_symbol(permno_intervals: pl.DataFrame) -> pl.DataFrame:
    """A PERMNO-interval frame under `_densify`'s own column names.

    `IndexConstituentDataset._densify` reads `(symbol, start_date, end_date)`,
    and the DIMENSION is still called `symbol` on both panels (RULING 3) --
    only what it spells changed. So this is a RENAME and nothing else: no cast,
    no re-derivation, no second identity rule to keep in step with the price
    panel's. `CrspMembership.permno_intervals` already types the column
    `pl.Int64`, which is the dtype the CRSP price panel's axis carries since
    03.11-03.
    """
    return permno_intervals.rename({"permno": "symbol"})


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


class CrspSP500ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time S&P 500 membership from CRSP itself (D-05).

    Source: `crsp_a_indexes.dsp500list_v2`, CRSP's OWN membership spells by
    PERMNO, read out of the reference tier at `config.cache_dir`. Coverage
    starts 1925-12-31, the start of index family 1100500 -- fifty years
    earlier than `SP500ConstituentDataset`'s Wikipedia change log, which is
    the whole reason both exist.

    **Its symbols ARE the CRSP PRICE PANEL'S symbols**, because both sides are
    the int64 PERMNO itself rather than something derived from it. A mask built
    here and applied to a `CrspStockDataset` panel therefore lines up column
    for column, and a rename (FB -> META) or a share class (BRK.B) is not an
    event either side has to handle in step with the other -- there is no
    ticker rule left to disagree about. It is NOT interchangeable with the
    Wikipedia panel's symbol axis, which is why this is a separate class and a
    separate store rather than a source switch on that one.

    **The right edge is the CRSP annual product end, never today.** Every
    interval `CrspMembership` produces carries an explicit end bounded by
    `CrspReference.product_end`, so `_densify`'s open-interval branch -- which
    extends to wall-clock today -- is never taken. A universe that ran past
    CRSP's price coverage would be True over a stretch with no prices at all.
    """

    #: The `CrspMembership` universe this class binds to. A class attribute
    #: rather than a literal in two method bodies, so the pair cannot drift.
    INDEX = CrspMembership.SP500

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
            )
        )


class CompustatNasdaq100ConstituentDataset(IndexConstituentDataset):
    """Daily point-in-time Nasdaq-100 membership from Compustat + CCM (D-14).

    Source: `comp.idxcst_his` at `gvkeyx = '000208'`, whose `(gvkey, iid)`
    spells become PERMNOs through `crsp_a_ccm.ccmxpf_lnkhist`. The link join
    is on `gvkey` AND `iid = liid`, never on `linkprim`: both Alphabet classes
    are members under one gvkey (160329, iid 01 -> GOOGL 90319 and iid 03 ->
    GOOG 14542), and the conventional `linkprim IN ('P','C')` filter would
    keep the primary issue only, silently dropping a real index member.

    **Left-censored at 1995-01-01, twelve years earlier than
    `Nasdaq100ConstituentDataset`'s 2007-02-01.** That date is a CENSOR rather
    than a start: a hundred spells begin exactly there because that is where
    Compustat's history begins, not where those memberships did. The panel
    still never starts earlier, for the same reason every coverage clamp
    exists -- all-False and unknown are indistinguishable in a boolean panel.

    **An unlinked spell REFUSES by default.** A membership day with no PERMNO
    has no security and therefore no symbol; dropping it would remove a real
    index member from the universe, which reads downstream as a slightly
    smaller universe rather than as an error. Pass
    `kwargs={"allow_unlinked": True}` to proceed with the linked days and read
    the rest from the membership `report["unlinked"]` -- the opt-out lives in
    the config, so a run that tolerated the gap says so in its own
    `config.json`.
    """

    #: The `CrspMembership` universe this class binds to.
    INDEX = CrspMembership.NASDAQ100

    def __init__(self, dataset_config: ConstituentDatasetConfig):
        super().__init__(dataset_config)

    def _pit_coverage_start(self) -> str:
        return CrspMembership.PIT_COVERAGE_START[self.INDEX]

    def _build_intervals(self) -> pl.DataFrame:
        return _rename_permno_to_symbol(
            CrspMembership(CrspReference(self.config.cache_dir)).permno_intervals(
                self.INDEX,
                allow_unlinked=bool(
                    (self.config.kwargs or {}).get("allow_unlinked", False)
                ),
            )
        )
