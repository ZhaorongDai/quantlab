"""The StockFit refresh planner: which endpoint x entity pulls a run makes, in what order, within the day's budget.

A pure function of what the raw tier holds (when each entity was last pulled,
and the newest filing since then that touches it) and of the requests left
today. No request, no file: the refresh (``quantlab.acquisition.stockfit.refresh``)
reads the raw tier into ``EndpointState`` rows and runs the jobs it gets back.

A job is due for one of three reasons, taken in this order, each in table
order (the order of ``ENDPOINTS``, which is the backfill order):

1. ``filed``: the entity was pulled before, and a filing that drives its
   endpoint (a 10-K for the statements, a Form 4 for the insiders) was made
   after that pull;
2. ``backfill``: the entity was never pulled;
3. ``stale``: the endpoint is re-pulled every ``repull_days`` days (the
   weekly re-pull of the statements) and the entity's last pull is that many
   UTC days old, oldest first.

Filings first, so a day's new filings land even while the backfill runs;
the periodic re-pulls last, since they only back the filings up. A snapshot
endpoint (``EndpointState.snapshot``: a daily snapshot the vendor keeps no
history of, such as each company's latest headlines) would wait behind the
whole backfill, so up to ``snapshots`` of its due jobs (never pulled first,
then the oldest) go right after the filings: its history starts on the
first day. Each job is counted as one request; the jobs past ``budget`` are
deferred, and the next run plans them again, since nothing about them has
changed on disk.

Examples
--------
>>> from datetime import UTC, datetime
>>> now = datetime(2026, 11, 20, tzinfo=UTC)
>>> week_ago = datetime(2026, 11, 13, tzinfo=UTC)
>>> state = EndpointState("income_statement_quarter", (
...     EntityState("1093691", last_pulled=week_ago),
...     EntityState("1046179", last_pulled=week_ago, filed=datetime(2026, 11, 19, tzinfo=UTC)),
...     EntityState("1418091"),
... ), repull_days=7)
>>> [(job.entity, job.reason) for job in plan([state], now=now).jobs]
[('1046179', 'filed'), ('1418091', 'backfill'), ('1093691', 'stale')]
>>> plan([state], now=now, budget=1).deferred
{'income_statement_quarter': 2}

A snapshot's due jobs jump the backfill, up to ``snapshots`` of them:

>>> news = EndpointState("news", (EntityState("1652044"),), repull_days=1, snapshot=True)
>>> [job.endpoint for job in plan([state, news], now=now, snapshots=1).jobs]
['income_statement_quarter', 'news', 'income_statement_quarter', 'income_statement_quarter']
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

#: The reasons a job is due, in the order they are planned.
REASONS = ("filed", "backfill", "stale")


@dataclass(frozen=True)
class EntityState:
    """What the raw tier says of one entity of an endpoint.

    Parameters
    ----------
    entity : str
        The entity's directory name.
    last_pulled : datetime or None, default None
        When it was last pulled, published or not; ``None`` if never.
    filed : datetime or None, default None
        When the newest filing that drives the endpoint was made for it;
        ``None`` if none is known.
    """

    entity: str
    last_pulled: datetime | None = None
    filed: datetime | None = None


@dataclass(frozen=True)
class EndpointState:
    """One endpoint's entities, in pull order, and how often it is re-pulled.

    Parameters
    ----------
    name : str
        The endpoint.
    entities : tuple of EntityState
        Its entities, in pull order.
    repull_days : int or None, default None
        Re-pull an entity once its last pull is this many UTC days old;
        ``None`` re-pulls only on a new filing.
    snapshot : bool, default False
        Whether the endpoint is a daily snapshot without history, whose due
        jobs may take the requests ``plan`` reserves for snapshots.
    """

    name: str
    entities: tuple[EntityState, ...]
    repull_days: int | None = None
    snapshot: bool = False


@dataclass(frozen=True)
class Job:
    """One pull to make: an entity of an endpoint, and why it is due."""

    endpoint: str
    entity: str
    reason: str


@dataclass(frozen=True)
class Plan:
    """The jobs a run makes, in order, and per endpoint how many due jobs wait for the next run."""

    jobs: tuple[Job, ...]
    deferred: dict[str, int] = field(default_factory=dict)


def _reason(entity: EntityState, repull_days: int | None, now: datetime) -> str | None:
    if entity.last_pulled is None:
        return "backfill"
    if entity.filed is not None and entity.filed > entity.last_pulled:
        return "filed"
    age = (now.astimezone(UTC).date() - entity.last_pulled.astimezone(UTC).date()).days
    if repull_days is not None and age >= repull_days:
        return "stale"
    return None


def plan(
    endpoints: Sequence[EndpointState], *, now: datetime, budget: int | None = None, snapshots: int = 0
) -> Plan:
    """Return the jobs due now: filings, up to ``snapshots`` snapshot jobs, the backfill, the re-pulls.

    Parameters
    ----------
    endpoints : sequence of EndpointState
        The endpoints, in table (backfill) order.
    now : datetime
        The run's time; the age of a last pull is counted in UTC days to it.
    budget : int or None, default None
        The requests the run may spend, one per job; ``None`` for no limit.
    snapshots : int, default 0
        How many due jobs of the snapshot endpoints go before the backfill,
        never-pulled entities first, then the oldest; the rest keep their
        place.

    Returns
    -------
    Plan
        The jobs within the budget, in order, and the count of due jobs
        deferred per endpoint (endpoints with none deferred left out).

    Examples
    --------
    See the module docstring.
    """
    due: dict[str, list[tuple[datetime, int, Job]]] = {reason: [] for reason in REASONS}
    for order, endpoint in enumerate(endpoints):
        for entity in endpoint.entities:
            reason = _reason(entity, endpoint.repull_days, now)
            if reason is not None:
                oldest = entity.last_pulled if reason == "stale" else datetime.min.replace(tzinfo=UTC)
                due[reason].append((oldest, order, Job(endpoint.name, entity.entity, reason)))
    due["stale"].sort(key=lambda item: (item[0].astimezone(UTC).date(), item[1]))
    ordered = [job for reason in REASONS for _, _, job in due[reason]]
    if snapshots > 0:
        kinds = {endpoint.name for endpoint in endpoints if endpoint.snapshot}
        first = len(due["filed"])
        early = [job for job in ordered[first:] if job.endpoint in kinds][:snapshots]
        taken = set(early)
        ordered = ordered[:first] + early + [job for job in ordered[first:] if job not in taken]
    kept = ordered if budget is None else ordered[: max(budget, 0)]
    deferred: dict[str, int] = {}
    for job in ordered[len(kept):]:
        deferred[job.endpoint] = deferred.get(job.endpoint, 0) + 1
    return Plan(tuple(kept), deferred)
