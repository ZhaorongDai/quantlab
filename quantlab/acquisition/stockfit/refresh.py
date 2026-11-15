"""Keep StockFit's raw tier current: every endpoint of ``ENDPOINTS``, entity by entity.

Each endpoint (``quantlab.acquisition.stockfit.endpoints.ENDPOINTS``) is
pulled for each of its entities, in table order: the company lists first,
then each company's statements and footnotes. A pull lands under
``<download-dir>/stockfit/<endpoint>/<entity>/<stamp>/`` by one rename, or is
dropped when it equals the entity's newest pull; the entity's watermark moves
either way (``quantlab.dataset.stockfit.raw``).

Failures follow ``quantlab.acquisition.refresh``. An endpoint StockFit
refuses (``StockFitEntitlementError``), or a day whose requests are spent
(``StockFitDailyCapReached``), stops that endpoint and fails its part; an
entity whose request kept failing (``StockFitTransportError``) or came back
unreadable (``StockFitResponseError``) is skipped, and its endpoint's part
fails once every other entity is pulled. Any other error, a bug of ours among
them, is raised. A missing ``STOCKFIT_API_KEY`` raises before any request.

Examples
--------
Needs ``STOCKFIT_API_KEY`` and the network::

    from quantlab.acquisition.registry import refresh_raw_tier
    from quantlab.acquisition.base import DataSourceRegistry

    result = refresh_raw_tier(DataSourceRegistry.get("stockfit"), "/data/quantlab/downloads")
    result.failed  # the endpoints not refreshed
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path

from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart
from quantlab.acquisition.stockfit.client import (
    StockFitClient,
    StockFitDailyCapReached,
    StockFitEntitlementError,
    StockFitResponseError,
    StockFitTransportError,
)
from quantlab.acquisition.stockfit.endpoints import ENDPOINTS, Endpoint, PullContext
from quantlab.dataset.stockfit.raw import VENDOR_DIR

#: The errors that stop one endpoint; the endpoints after it still run.
ENDPOINT_FAILURES = (StockFitEntitlementError, StockFitDailyCapReached)

#: The errors that skip one entity; its endpoint's other entities still run.
ENTITY_FAILURES = (StockFitTransportError, StockFitResponseError)


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    endpoints: Iterable[str | Endpoint] | None = None,
    ciks: Iterable[int] | None = None,
    client: StockFitClient | None = None,
) -> RawTierRefreshResult:
    """Pull StockFit's endpoints into the raw tier under ``<download_dir>/stockfit/``.

    Parameters
    ----------
    download_dir : str or Path
        The download root.
    end : date or None, default None
        Taken for the refresh protocol; none of these endpoints is asked by
        date, so it bounds nothing.
    endpoints : iterable of str or Endpoint, or None, default None
        The endpoints to pull, by name in ``ENDPOINTS`` or as rows; every one
        of ``ENDPOINTS``, in its order, when omitted.
    ciks : iterable of int or None, default None
        The companies a per-company endpoint is pulled for; every company of
        the raw tier's company lists when omitted.
    client : StockFitClient or None, default None
        The client to pull through; one reading ``STOCKFIT_API_KEY`` when
        omitted.

    Returns
    -------
    RawTierRefreshResult
        One part per endpoint: how many entities published a new pull, were
        unchanged or held nothing, or why the endpoint failed.

    Raises
    ------
    ValueError
        If an endpoint name is not in ``ENDPOINTS``; nothing is requested.
    StockFitCredentialError
        If ``client`` is omitted and ``STOCKFIT_API_KEY`` is unset; nothing
        is requested.
    Exception
        Any error not in ``ENDPOINT_FAILURES`` or ``ENTITY_FAILURES`` (a bug
        of ours), from the entity it happened on.

    Examples
    --------
    Needs the key and the network; two companies' statements::

        refresh_raw_tier("/data/quantlab/downloads", endpoints=("income_statement_quarter",), ciks=(320193, 789019))
    """
    del end  # no endpoint here is asked by date
    selected = list(ENDPOINTS.values()) if endpoints is None else [
        endpoint if isinstance(endpoint, Endpoint) else _named(endpoint) for endpoint in endpoints
    ]
    client = client if client is not None else StockFitClient()
    vendor_root = Path(download_dir) / VENDOR_DIR
    context = PullContext(vendor_root, None if ciks is None else tuple(int(cik) for cik in ciks))
    parts = []
    for endpoint in selected:
        started = time.monotonic()
        outcomes: Counter[str] = Counter()
        skipped: list[str] = []
        stopped = None
        for entity in endpoint.entities(context):
            try:
                outcomes[endpoint.pull(client, vendor_root, entity).value] += 1
            except ENDPOINT_FAILURES as error:
                stopped = str(error)
                break
            except ENTITY_FAILURES as error:
                skipped.append(f"{entity}: {error}")
        done = ", ".join(f"{count:,} {outcome}" for outcome, count in sorted(outcomes.items())) or "nothing pulled"
        if stopped is not None:
            parts.append(RefreshedPart.failure(endpoint.name, f"{stopped} ({done} before it)"))
        elif skipped:
            parts.append(RefreshedPart.failure(
                endpoint.name, f"{len(skipped):,} entit(ies) failed, first {skipped[0]} ({done})"
            ))
        else:
            parts.append(RefreshedPart(endpoint.name, f"{done}; {time.monotonic() - started:.0f}s"))
    return RawTierRefreshResult("stockfit", tuple(parts))


def _named(name: str) -> Endpoint:
    if name not in ENDPOINTS:
        raise ValueError(f"unknown StockFit endpoint {name!r}; expected one of {list(ENDPOINTS)}")
    return ENDPOINTS[name]
