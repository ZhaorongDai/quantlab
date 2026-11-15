"""Keep StockFit's raw tier current: a backfill that resumes across days, then only what new filings touched.

One run, in two steps:

1. Every market-wide list of ``quantlab.acquisition.stockfit.endpoints.ENDPOINTS``
   is pulled (a snapshot each run): the company lists, whose CIKs the
   per-company endpoints take, the recent-filings feed
   (``/api/filings/latest``, per form type), and the Congress trade feeds
   (each from its since-watermark).
2. The planner (``quantlab.acquisition.stockfit.planner``) orders the
   endpoint x entity pulls due now within the requests left today: first the
   entities a filing made since their last pull touches (a 10-K re-pulls the
   filer's statements and footnotes), then the entities never pulled, in table order (the
   backfill order), then the periodic re-pulls (each company's statements
   once a week). A share of the day's requests (``snapshot_share``) goes to
   the daily snapshots the vendor keeps no history of (each company's
   predicted filing calendar and latest headlines, each fund's daily
   holdings) right after the filings, so their history starts on the first
   day of a backfill that lasts months. The pulls of a round can add entities (a series' N-PORT
   quarters, the advisers of its service providers), so the run plans again
   until nothing new is due. The rest waits for the next run, which plans it
   again from the raw tier: so a backfill longer than a day's requests
   resumes where it stopped.

A pull lands under ``<download-dir>/stockfit/<endpoint>/<entity>/<stamp>/``
by one rename, or is dropped when it equals the entity's newest pull; the
entity's watermark moves either way (``quantlab.dataset.stockfit.raw``).

Running out of the day's requests is not a failure: the run stops sending
(``StockFitDailyCapReached``) and each endpoint's part says how many pulls it
deferred. An HTTP 429 within the day is waited out and retried by the client.
Failures follow ``quantlab.acquisition.refresh``: an endpoint StockFit refuses
(``StockFitEntitlementError``) fails its part and the others still run; an
entity whose request kept failing (``StockFitTransportError``) or came back
unreadable (``StockFitResponseError``) is skipped, and its endpoint's part
fails once every other entity is pulled. Any other error, a bug of ours among
them, is raised. A missing ``STOCKFIT_API_KEY`` raises before any request.

Examples
--------
Needs ``STOCKFIT_API_KEY`` and the network::

    from quantlab.acquisition.registry import refresh_raw_tier
    from quantlab.acquisition.base import DataSourceRegistry

    result = refresh_raw_tier(DataSourceRegistry.get("stockfit"), "/data/downloads")
    result.summary()  # per endpoint: published, unchanged, deferred to the next run
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart
from quantlab.acquisition.stockfit.client import (
    StockFitClient,
    StockFitDailyCapReached,
    StockFitEntitlementError,
    StockFitResponseError,
    StockFitTransportError,
)
from quantlab.acquisition.stockfit.endpoints import (
    ENDPOINTS,
    FEED_FORMS,
    FILINGS_LATEST,
    Endpoint,
    MarketList,
    PullContext,
    filer_cik,
)
from quantlab.acquisition.stockfit.planner import EndpointState, EntityState, Job, plan_jobs
from quantlab.dataset.stockfit.raw import VENDOR_DIR, last_pulled, pulls, read_pull

#: The errors that skip one entity; its endpoint's other entities still run.
ENTITY_FAILURES = (StockFitTransportError, StockFitResponseError)

#: How far back the recent-filings feed reaches, in days.
FEED_DAYS = 30

#: The share of a run's requests kept for the daily snapshots, ahead of the backfill.
SNAPSHOT_SHARE = 0.2


class _Tally:
    """What one endpoint's pulls did in a run."""

    def __init__(self) -> None:
        self.outcomes: Counter[str] = Counter()
        self.skipped: list[str] = []
        self.refused: str | None = None
        self.deferred = 0
        self.started = time.monotonic()

    def part(self, name: str) -> RefreshedPart:
        done = ", ".join(f"{count:,} {outcome}" for outcome, count in sorted(self.outcomes.items()))
        if self.deferred:
            done = f"{done}, " if done else ""
            done += f"{self.deferred:,} deferred to the next run"
        done = done or "nothing due"
        if self.refused is not None:
            return RefreshedPart.failure(name, f"{self.refused} ({done})")
        if self.skipped:
            return RefreshedPart.failure(name, f"{len(self.skipped):,} entit(ies) failed, first {self.skipped[0]} ({done})")
        return RefreshedPart(name, f"{done}; {time.monotonic() - self.started:.0f}s")


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    endpoints: Iterable[str | Endpoint] | None = None,
    ciks: Iterable[int] | None = None,
    fund_series: Iterable[str] | None = None,
    client: StockFitClient | None = None,
    snapshot_share: float = SNAPSHOT_SHARE,
) -> RawTierRefreshResult:
    """Bring StockFit's raw tier under ``<download_dir>/stockfit/`` current, within today's requests.

    Parameters
    ----------
    download_dir : str or Path
        The download root.
    end : date or None, default None
        The day the refresh runs for: an endpoint asked per quarter (13F)
        pulls the quarters whose filings are due by it. The client's today
        (UTC) when omitted.
    endpoints : iterable of str or Endpoint, or None, default None
        The endpoints to refresh, by name in ``ENDPOINTS`` or as rows; every
        one of ``ENDPOINTS``, in its order, when omitted. The filings that
        drive re-pulls are read from the feed's pulls on disk, so a refresh
        of some endpoints still re-pulls what new filings touched.
    ciks : iterable of int or None, default None
        The companies a per-company endpoint is pulled for; every company of
        the raw tier's company lists when omitted.
    fund_series : iterable of str or None, default None
        The fund series (``<cik>-<series id>``) a per-series endpoint is
        pulled for; every series of the raw tier's fund lists when omitted.
    client : StockFitClient or None, default None
        The client to pull through; one reading ``STOCKFIT_API_KEY`` when
        omitted. Its ``day_reserve`` is left unspent.
    snapshot_share : float, default ``SNAPSHOT_SHARE`` (0.2)
        The share of the requests left after the market-wide lists that the
        due daily snapshots (``Endpoint.snapshot``) may take ahead of the
        backfill; what they leave goes to the backfill. 0 plans them with
        the backfill. No effect without a daily limit.

    Returns
    -------
    RawTierRefreshResult
        One part per endpoint: how many entities published a new pull, were
        unchanged, held nothing, or were deferred to the next run, or why the
        endpoint failed.

    Raises
    ------
    ValueError
        If an endpoint name is not in ``ENDPOINTS``, or ``snapshot_share``
        is outside [0, 1]; nothing is requested.
    StockFitCredentialError
        If ``client`` is omitted and ``STOCKFIT_API_KEY`` is unset; nothing
        is requested.
    Exception
        Any error not in ``ENTITY_FAILURES`` and not a refusal or the daily
        cap (a bug of ours), from the entity it happened on.

    Examples
    --------
    Needs the key and the network; two companies' quarterly income statements::

        refresh_raw_tier("/data/downloads", endpoints=("income_statement_quarter",), ciks=(320193, 789019))
    """
    if not 0 <= snapshot_share <= 1:
        raise ValueError(f"snapshot_share must be between 0 and 1, got {snapshot_share}.")
    selected = list(ENDPOINTS.values()) if endpoints is None else [
        endpoint if isinstance(endpoint, Endpoint) else _named(endpoint) for endpoint in endpoints
    ]
    client = client if client is not None else StockFitClient()
    vendor_root = Path(download_dir) / VENDOR_DIR
    context = PullContext(
        vendor_root,
        None if ciks is None else tuple(int(cik) for cik in ciks),
        None if fund_series is None else tuple(str(series) for series in fund_series),
        end=end if end is not None else client.now().date(),
    )
    tallies = {endpoint.name: _Tally() for endpoint in selected}
    by_name = {endpoint.name: endpoint for endpoint in selected}

    lists = [endpoint for endpoint in selected if isinstance(endpoint, MarketList)]
    jobs = [Job(endpoint.name, entity, "snapshot") for endpoint in lists for entity in endpoint.entities(context)]
    capped = _run(jobs, by_name, tallies, client, vendor_root)
    others = [endpoint for endpoint in selected if not isinstance(endpoint, MarketList)]
    tried: set[tuple[str, str]] = set()
    snapshots: int | None = None  # the requests kept for the daily snapshots, set at the first plan
    while not capped:
        # Planned again after each round: some entities come from earlier pulls of this run
        # (a series' N-PORT quarters, the advisers of its service providers).
        remaining = client.remaining_day
        budget = None if remaining is None else max(remaining - client.day_reserve, 0)
        if snapshots is None:
            snapshots = 0 if budget is None else int(budget * snapshot_share)
        planned = plan_jobs(_states(others, context, client.now()), now=client.now(), budget=budget, snapshots=snapshots)
        fresh = [job for job in planned.jobs if (job.endpoint, job.entity) not in tried]
        snapshots -= sum(by_name[job.endpoint].snapshot for job in fresh)
        if not fresh or planned.deferred:
            for name, count in planned.deferred.items():
                tallies[name].deferred += count
        if not fresh:
            break
        tried.update((job.endpoint, job.entity) for job in fresh)
        capped = _run(fresh, by_name, tallies, client, vendor_root)
        if planned.deferred:
            break
    return RawTierRefreshResult("stockfit", tuple(tally.part(name) for name, tally in tallies.items()))


def _run(jobs: list[Job], endpoints: dict[str, Endpoint], tallies: dict[str, _Tally], client, vendor_root) -> bool:
    """Pull the jobs in order; return whether the day's requests ran out (the rest deferred)."""
    for index, job in enumerate(jobs):
        tally = tallies[job.endpoint]
        if tally.refused is not None:
            continue
        try:
            tally.outcomes[endpoints[job.endpoint].pull(client, vendor_root, job.entity).value] += 1
        except StockFitDailyCapReached:
            for later in jobs[index:]:
                tallies[later.endpoint].deferred += 1
            return True
        except StockFitEntitlementError as error:
            tally.refused = str(error)
        except ENTITY_FAILURES as error:
            tally.skipped.append(f"{job.entity}: {error}")
    return False


def _filed(vendor_root: Path, now: datetime) -> dict[str, dict[int, datetime]]:
    """Return, per form type, each filer's newest filing time in the feed's pulls of the last 30 days.

    A filing with an acceptance time is dated by it; one with only a filing
    date by the end of that day (UTC), so a pull made on the next day covers it.
    """
    since = now - timedelta(days=FEED_DAYS + 1)
    filed: dict[str, dict[int, datetime]] = {}
    for entity, form in FEED_FORMS.items():
        times = pulls(vendor_root, FILINGS_LATEST, entity)
        # An unchanged feed pull is dropped, so the rows of a recent day may sit in an older pull.
        recent = [stamp for stamp in times if stamp >= now - timedelta(days=2 * FEED_DAYS)] or times[-1:]
        for stamp in recent:
            for row in read_pull(vendor_root, FILINGS_LATEST, entity, stamp):
                cik = filer_cik(row)
                at = _filing_time(row)
                if cik is None or at is None or at < since:
                    continue
                newest = filed.setdefault(form, {})
                newest[cik] = max(newest.get(cik, at), at)
    return filed


def _filing_time(row: dict) -> datetime | None:
    if row.get("acceptedAt"):
        return datetime.fromisoformat(str(row["acceptedAt"])).astimezone(UTC)
    if row.get("dateFiled"):
        return datetime.combine(date.fromisoformat(row["dateFiled"]) + timedelta(days=1), datetime.min.time(), UTC)
    return None


def _states(endpoints: list[Endpoint], context: PullContext, now: datetime) -> list[EndpointState]:
    """Read what the planner needs from the raw tier: each entity's last pull and newest driving filing."""
    filed = _filed(context.vendor_root, now)
    states = []
    for endpoint in endpoints:
        touched: dict[str, datetime] = {}
        for form in endpoint.triggers:
            newest = filed.get(form, {})
            for cik, names in endpoint.filer_entities(newest, context).items():
                for name in names:
                    touched[name] = max(touched.get(name, newest[cik]), newest[cik])
        entities = tuple(
            EntityState(entity, last_pulled(context.vendor_root, endpoint.name, entity), touched.get(entity))
            for entity in endpoint.entities(context)
        )
        states.append(EndpointState(endpoint.name, entities, endpoint.repull_days, endpoint.snapshot))
    return states


def _named(name: str) -> Endpoint:
    if name not in ENDPOINTS:
        raise ValueError(f"unknown StockFit endpoint {name!r}; expected one of {list(ENDPOINTS)}")
    return ENDPOINTS[name]
