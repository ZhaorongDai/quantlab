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
['reference', 'statements', 'filings']
>>> source.required_env
('STOCKFIT_API_KEY',)
"""

from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.acquisition.stockfit.refresh import refresh_raw_tier
from quantlab.dataset.stockfit.filings import StockFitFilingsDataset

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
            # The company lists, operating and delisted: raw only, keyed by CIK.
            Capability(market="us_equity", frequency="1d", data_type="reference", entitlement=_ENTITLEMENT),
            # The income, balance-sheet and cash-flow statements per CIK, annual, quarterly and TTM: raw only.
            Capability(market="us_equity", frequency="1d", data_type="statements", entitlement=_ENTITLEMENT),
            # Each company's SEC filing list, and the filing panel by CIK: form counts, 8-K events, amendments.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="filings",
                dataset_cls=StockFitFilingsDataset,
                earliest_available="1994-03-08",
                entitlement=_ENTITLEMENT,
            ),
        ),
        # A literal, checked against the client's constant by a test.
        required_env=("STOCKFIT_API_KEY",),
        raw_tier_refresh=refresh_raw_tier,
    )
)
