"""StockFit vendor package and the registry entry for a StockFit key.

The client lives in ``quantlab.acquisition.stockfit.client``, the endpoint
table in ``quantlab.acquisition.stockfit.endpoints`` and the refresh in
``quantlab.acquisition.stockfit.refresh``; this package module holds only the
descriptor, as ``quantlab.acquisition.massive`` does.

``STOCKFIT_SOURCE`` registers StockFit (SEC-filing data parsed from EDGAR,
``https://api.stockfit.io``) with ``quantlab.acquisition.registry``: one
``Capability`` per kind of data its raw tier holds, the credential variable,
and the raw-tier refresh. StockFit has no bulk files: its raw tier is one
directory per pull of an endpoint for an entity
(``quantlab.dataset.stockfit.raw``), so the descriptor names no symbol-batch
class and no config factory, and ``registry.download_symbols()`` refuses it.

Examples
--------
>>> import quantlab.acquisition.registry
>>> from quantlab.acquisition.base import DataSourceRegistry
>>> source = DataSourceRegistry.get("stockfit")
>>> [c.data_type for c in source.capabilities]
['reference', 'statements', 'as_reported', 'derived', 'scores_trends', 'footnotes', 'fund_profile', 'fund_flows', 'fund_fees', 'fund_ncen', 'fund_holdings', 'corporate_events', 'declared_events', 'congress_trades', 'filings', 'insiders', 'beneficial_owners', 'executives', 'listing', 'earnings_consensus', 'ipo_calendar', 'offerings', 'filing_calendar', 'news_counts', 'holdings']
>>> source.required_env
('STOCKFIT_API_KEY',)
"""

from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.acquisition.stockfit.refresh import refresh_raw_tier
from quantlab.dataset.stockfit.as_reported import StockFitAsReportedDataset
from quantlab.dataset.stockfit.beneficial_owners import StockFitBeneficialOwnersDataset
from quantlab.dataset.stockfit.congress import StockFitCongressTradesDataset
from quantlab.dataset.stockfit.derived import StockFitDerivedDataset
from quantlab.dataset.stockfit.events import (
    StockFitCorporateEventsDataset,
    StockFitDeclaredEventsDataset,
)
from quantlab.dataset.stockfit.executives import StockFitExecutivesDataset
from quantlab.dataset.stockfit.filings import StockFitFilingsDataset
from quantlab.dataset.stockfit.footnotes import StockFitFootnoteDataset
from quantlab.dataset.stockfit.funds import (
    StockFitFundFeesDataset,
    StockFitFundFlowsDataset,
    StockFitFundNcenDataset,
    StockFitFundProfileDataset,
)
from quantlab.dataset.stockfit.holdings import StockFitHoldingsDataset
from quantlab.dataset.stockfit.insiders import StockFitInsidersDataset
from quantlab.dataset.stockfit.scores_trends import StockFitScoresTrendsDataset
from quantlab.dataset.stockfit.snapshots import (
    StockFitEarningsConsensusDataset,
    StockFitFilingCalendarDataset,
    StockFitIpoCalendarDataset,
    StockFitListingDataset,
    StockFitNewsCountsDataset,
    StockFitOfferingsDataset,
)
from quantlab.dataset.stockfit.statements import StockFitStatementsDataset

#: The plan the endpoints need.
_ENTITLEMENT = "StockFit Professional (api.stockfit.io)"

#: The registered descriptor for a StockFit key.
STOCKFIT_SOURCE = register_source(
    SourceDescriptor(
        vendor="stockfit",
        display_name="StockFit (SEC filings parsed from EDGAR; REST)",
        symbol_batch_cls=None,
        config_factory=None,
        capabilities=(
            # The company lists, operating and delisted, keyed by CIK, and the data dictionaries
            # (SFIC sectors and industries, exchange names): raw only.
            Capability(market="us_equity", frequency="1d", data_type="reference", entitlement=_ENTITLEMENT),
            # The income, balance-sheet and cash-flow statements per CIK, annual and quarterly;
            # the quarterly pulls build the quarterly and TTM panels, as known on each day.
            Capability(
                market="us_equity", frequency="1d", data_type="statements",
                dataset_cls=StockFitStatementsDataset, entitlement=_ENTITLEMENT,
            ),
            # As-reported us-gaap facts per CIK, quarterly and annual: a panel of the chosen concepts.
            Capability(
                market="us_equity", frequency="1d", data_type="as_reported",
                dataset_cls=StockFitAsReportedDataset, entitlement=_ENTITLEMENT,
            ),
            # Growth, key metrics, EPS and dividend history, and share classes per CIK, at the statements' filing dates.
            Capability(
                market="us_equity", frequency="1d", data_type="derived",
                dataset_cls=StockFitDerivedDataset, entitlement=_ENTITLEMENT,
            ),
            # Scores and trends per CIK, accumulated from snapshots.
            Capability(
                market="us_equity", frequency="1d", data_type="scores_trends",
                dataset_cls=StockFitScoresTrendsDataset, entitlement=_ENTITLEMENT,
            ),
            # The 17 footnote tables per CIK, annual and quarterly: one panel per footnote and period type.
            Capability(
                market="us_equity", frequency="1d", data_type="footnotes", entitlement=_ENTITLEMENT,
                dataset_cls=StockFitFootnoteDataset,
            ),
            # Fund series (<cik>-<series id>): N-PORT summary and monthly flows, N-1A fees, N-CEN fields.
            Capability(market="us_equity", frequency="1d", data_type="fund_profile",
                       dataset_cls=StockFitFundProfileDataset, entitlement=_ENTITLEMENT),
            Capability(market="us_equity", frequency="1d", data_type="fund_flows",
                       dataset_cls=StockFitFundFlowsDataset, entitlement=_ENTITLEMENT),
            Capability(market="us_equity", frequency="1d", data_type="fund_fees",
                       dataset_cls=StockFitFundFeesDataset, entitlement=_ENTITLEMENT),
            Capability(market="us_equity", frequency="1d", data_type="fund_ncen",
                       dataset_cls=StockFitFundNcenDataset, entitlement=_ENTITLEMENT),
            # N-CSR performance, N-PORT holdings, daily ETF holdings, N-CEN providers, advisers: raw only.
            Capability(market="us_equity", frequency="1d", data_type="fund_holdings", entitlement=_ENTITLEMENT),
            # XBRL splits at their first disclosure and successions at their effective day, per CIK.
            Capability(
                market="us_equity", frequency="1d", data_type="corporate_events",
                dataset_cls=StockFitCorporateEventsDataset, entitlement=_ENTITLEMENT,
            ),
            # The declared splits and dividends of the coming days, accumulated from snapshots, per CIK.
            Capability(
                market="us_equity", frequency="1d", data_type="declared_events",
                dataset_cls=StockFitDeclaredEventsDataset, entitlement=_ENTITLEMENT,
            ),
            # House and Senate members' trade reports, placed on the CIK of the trade-date ticker holder.
            Capability(
                market="us_equity", frequency="1d", data_type="congress_trades",
                dataset_cls=StockFitCongressTradesDataset, entitlement=_ENTITLEMENT,
            ),
            # Each company's SEC filing list, and the filing panel by CIK: form counts, 8-K events, amendments.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="filings",
                dataset_cls=StockFitFilingsDataset,
                earliest_available="1994-03-08",
                entitlement=_ENTITLEMENT,
            ),
            # Officers' and directors' Form 4/5 transactions per CIK, with the Sharadar insider panel's variables.
            Capability(
                market="us_equity", frequency="1d", data_type="insiders",
                dataset_cls=StockFitInsidersDataset, earliest_available="2003-01-01", entitlement=_ENTITLEMENT,
            ),
            # Schedule 13D/13G stakes per CIK, each placed by its filing's acceptance time.
            Capability(
                market="us_equity", frequency="1d", data_type="beneficial_owners",
                dataset_cls=StockFitBeneficialOwnersDataset, earliest_available="2025-01-01",
                entitlement=_ENTITLEMENT,
            ),
            # Proxy pay versus performance, governance flags and company-selected measures per CIK.
            Capability(
                market="us_equity", frequency="1d", data_type="executives",
                dataset_cls=StockFitExecutivesDataset, earliest_available="2022-01-01", entitlement=_ENTITLEMENT,
            ),
            # Panels accumulated from forward snapshots, each placed at its pull time, on the CIK axis.
            *(
                Capability(market="us_equity", frequency="1d", data_type=data_type, dataset_cls=cls, entitlement=_ENTITLEMENT)
                for data_type, cls in (
                    ("listing", StockFitListingDataset),
                    ("earnings_consensus", StockFitEarningsConsensusDataset),
                    ("ipo_calendar", StockFitIpoCalendarDataset),
                    ("offerings", StockFitOfferingsDataset),
                    ("filing_calendar", StockFitFilingCalendarDataset),
                    ("news_counts", StockFitNewsCountsDataset),
                )
            ),
            # 13F: holder rows per company and quarter, managers' filings, managers' portfolios (raw only).
            Capability(
                market="us_equity", frequency="1d", data_type="holdings", entitlement=_ENTITLEMENT,
                dataset_cls=StockFitHoldingsDataset,
            ),
        ),
        # A literal, checked against the client's constant by a test.
        required_env=("STOCKFIT_API_KEY",),
        raw_tier_refresh=refresh_raw_tier,
    )
)
