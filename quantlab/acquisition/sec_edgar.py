"""SEC EDGAR's quarterly filing indexes: the client, the raw-tier refresh and the registry entry.

EDGAR (``https://www.sec.gov``) serves, free and without a key, a master
index of every quarter's filings
(``/Archives/edgar/full-index/<YYYY>/QTR<n>/master.idx``); the layout of the
raw tier and its readers are ``quantlab.dataset.sec_edgar``. The StockFit 13F
panel dates a holding row by its filing's date in these indexes when StockFit
names no date for the filing.

SEC's fair-access rules ask every automated client to declare who it is in
its ``User-Agent`` (a name and a contact address) and to send at most 10
requests a second. The client reads the header from ``SEC_EDGAR_USER_AGENT``
and refuses to send anything without it (``SecEdgarCredentialError``, before
the first request), keeps at least ``MIN_INTERVAL`` seconds between requests,
and retries a 403 (SEC's answer to a client over the rate), a 429, a 5xx or a
broken connection after a doubling back-off, up to ``retries`` times
(``quantlab.acquisition._support.request_policy``). A 404 is a quarter EDGAR
has no index for yet; any other status is a ``SecEdgarResponseError``. The
header's value is kept out of every message.

``refresh_raw_tier`` backfills every quarter from the first quarter of
StockFit's 13F data (``FIRST_13F_QUARTER``) and then, on each run, pulls
again the current quarter (its index is rebuilt every night), a quarter
pulled before its index could hold all of it, and, for ``RECENT_DAYS`` after
a quarter ends, that quarter too. A pull equal to the file on disk is
dropped; a changed one is published with one rename; the quarter's watermark
moves to the pull's UTC day either way. A quarter that fails (the errors in
``QUARTER_FAILURES``) is a failed part and the other quarters still run.
``SEC_EDGAR_SOURCE`` at the bottom registers the vendor.

Examples
--------
Needs ``SEC_EDGAR_USER_AGENT`` and the network::

    from quantlab.acquisition.base import DataSourceRegistry
    from quantlab.acquisition.registry import refresh_raw_tier

    result = refresh_raw_tier(DataSourceRegistry.get("sec_edgar"), "/data/downloads")
    result.summary()  # per quarter pulled: published, unchanged or not on EDGAR yet
"""

from __future__ import annotations

import gzip
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from quantlab.acquisition._support.http import HttpResponse, Transport, http_get
from quantlab.acquisition._support.request_policy import (
    RETRY_STATUSES,
    RequestPolicy,
    TransientError,
    require_env,
    retry_after,
)
from quantlab.acquisition.base import Capability, SourceDescriptor, register_source
from quantlab.acquisition.refresh import RawTierRefreshResult, RefreshedPart
from quantlab.acquisition.stockfit.endpoints import FIRST_13F_QUARTER
from quantlab.dataset.sec_edgar import (
    HEADER,
    INDEX_LAG_DAYS,
    VENDOR_DIR,
    index_file,
    next_quarter,
    quarter_end,
    quarter_of,
    read_index,
    read_watermark,
    watermark_file,
)
from quantlab.utils.atomic import move_durably
from quantlab.utils.watermark import write_watermark

#: The environment variable holding the ``User-Agent`` SEC asks every client to declare.
USER_AGENT_ENV = "SEC_EDGAR_USER_AGENT"

#: The quarterly indexes' base URL.
BASE_URL = "https://www.sec.gov/Archives/edgar/full-index"

#: Seconds to connect, and to wait for each read.
TIMEOUT = (30, 120)

#: Seconds kept between two requests: SEC allows 10 a second.
MIN_INTERVAL = 0.1

#: Statuses retried: SEC's 403 to a client over its rate, and every "try again later".
RETRIED_STATUSES = RETRY_STATUSES | {403}

#: Days after a quarter's end during which its index is pulled again on every run, so the filings of
#: its last days, indexed the night after, are in it whenever the run happens.
RECENT_DAYS = 28


class SecEdgarCredentialError(RuntimeError):
    """``SEC_EDGAR_USER_AGENT`` is not set."""


class SecEdgarTransportError(TransientError):
    """The connection failed or broke, or SEC answered a status worth retrying; retried."""


class SecEdgarResponseError(ValueError):
    """SEC answered a status other than 200, 404 and the retried ones, or a body that is not a master index."""


#: The errors that fail one quarter; the other quarters still run.
QUARTER_FAILURES = (SecEdgarTransportError, SecEdgarResponseError)


def http_transport(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> HttpResponse:
    """Send one GET; the client's default ``http_get``.

    ``quantlab.acquisition._support.http.http_get`` with ``TIMEOUT``, not
    following redirects (the ``User-Agent`` names a contact), and raising
    ``SecEdgarTransportError`` for a connection that fails or breaks.

    Examples
    --------
    Needs the network; without a ``User-Agent`` SEC answers 403::

        http_transport(BASE_URL + "/2024/QTR1/master.idx", params={}, headers={}).status
    """
    return http_get(url, params=params, headers=headers, timeout=TIMEOUT, error=SecEdgarTransportError)


class SecEdgarClient:
    """Fetch EDGAR's quarterly master indexes within SEC's fair-access rules; see the module docstring.

    Parameters
    ----------
    user_agent : str, optional
        The ``User-Agent`` to declare; read from ``SEC_EDGAR_USER_AGENT``
        when omitted.
    http_get : Transport, optional
        ``http_get(url, params=..., headers=...) -> HttpResponse``; the real
        HTTP adapter ``http_transport`` when omitted. Tests pass a fake.
    retries : int, default 5
        Retries of a request answered with a retried status, or whose
        connection broke, before it raises.
    backoff : float, default 2.0
        Seconds before the first retry without ``Retry-After``; doubled on
        each later one.
    sleep : callable, default time.sleep
        How the client waits.
    monotonic : callable, default time.monotonic
        The clock the spacing between requests is measured on.
    now : callable, default the system clock in UTC
        The current time, an aware ``datetime``; it dates the pulls.

    Raises
    ------
    SecEdgarCredentialError
        If ``user_agent`` is omitted and ``SEC_EDGAR_USER_AGENT`` is unset.

    Attributes
    ----------
    requests : int
        Requests sent by this client.

    Examples
    --------
    >>> from quantlab.acquisition._support.http import HttpResponse
    >>> def sec(url, *, params, headers):
    ...     return HttpResponse(404, b"Not Found")
    >>> client = SecEdgarClient(user_agent="Example research example@example.invalid", http_get=sec)
    >>> client.get_index(2031, 1) is None
    True
    >>> client.requests
    1
    """

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        http_get: Transport | None = None,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Initialize; see the class docstring."""
        if user_agent is None:
            (user_agent,) = require_env(
                (USER_AGENT_ENV,),
                hint="set it to a name and contact address, as SEC asks of automated clients. Nothing was downloaded.",
                error=SecEdgarCredentialError,
            )
        self._user_agent = user_agent
        self.http_get = http_get if http_get is not None else http_transport
        self.sleep = sleep
        self.monotonic = monotonic
        self.now = now
        self.requests = 0
        self._last: float | None = None
        self._policy = RequestPolicy(
            {USER_AGENT_ENV: user_agent}, retries=retries, backoff=backoff, sleep=sleep, vendor="SEC EDGAR",
            credential=USER_AGENT_ENV,
        )

    def _space(self) -> None:
        """Wait until ``MIN_INTERVAL`` has passed since the last request."""
        if self._last is not None:
            wait = MIN_INTERVAL - (self.monotonic() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.monotonic()

    def get_index(self, year: int, quarter: int) -> bytes | None:
        """Return one quarter's master index as EDGAR serves it; ``None`` when EDGAR has none (404).

        Parameters
        ----------
        year, quarter : int
            The calendar quarter.

        Returns
        -------
        bytes or None
            The uncompressed index.

        Raises
        ------
        SecEdgarTransportError
            If the request kept failing past ``retries``.
        SecEdgarResponseError
            If SEC answered another status, or a 200 whose body is not a
            master index.

        Examples
        --------
        See the class docstring.
        """
        path = f"/{year:04d}/QTR{quarter}/master.idx"
        headers = {"User-Agent": self._user_agent, "Accept-Encoding": "gzip, deflate"}

        def attempt() -> bytes | None:
            self._space()
            self.requests += 1
            response = self.http_get(BASE_URL + path, params={}, headers=headers)
            body = response.read()
            if response.status == 404:
                return None
            if response.status in RETRIED_STATUSES:
                raise SecEdgarTransportError(
                    self._policy.scrub(f"HTTP {response.status} on {path}"), retry_after=retry_after(response.headers)
                )
            if response.status != 200:
                raise SecEdgarResponseError(self._policy.scrub(f"HTTP {response.status} on {path}"))
            if HEADER.encode() not in body[:8192]:
                raise SecEdgarResponseError(f"{path} answered 200 with a body that is not a master index.")
            return body

        return self._policy.call(attempt, lambda tries, wait, error: f"SEC EDGAR: {error}; retry {tries + 1} in {wait:g}s.")

    def __repr__(self) -> str:
        """Return a repr without the ``User-Agent``."""
        return f"SecEdgarClient(requests={self.requests})"


def due_quarters(vendor_root: str | Path, today: date, start: date = FIRST_13F_QUARTER) -> list[tuple[int, int]]:
    """Return the quarters a refresh on ``today`` pulls, oldest first.

    From the quarter holding ``start`` through the one holding ``today``: a
    quarter never pulled; one whose last pull was too early to hold its
    whole index (less than ``INDEX_LAG_DAYS`` after its end, the current
    quarter always); and one that ended at most ``RECENT_DAYS`` ago.

    Examples
    --------
    >>> due_quarters("/nonexistent/sec_edgar", date(2024, 2, 1), start=date(2023, 10, 1))
    [(2023, 4), (2024, 1)]
    """
    due = []
    current, last = quarter_of(start), quarter_of(today)
    while current <= last:
        end = quarter_end(*current)
        pulled = read_watermark(vendor_root, *current)
        if pulled is None or pulled - timedelta(days=INDEX_LAG_DAYS) < end or today <= end + timedelta(days=RECENT_DAYS):
            due.append(current)
        current = next_quarter(*current)
    return due


def _publish(vendor_root: Path, year: int, quarter: int, data: bytes) -> bool:
    """Write one quarter's index unless it equals the file on disk; return whether it was written."""
    target = index_file(vendor_root, year, quarter)
    if target.exists() and read_index(vendor_root, year, quarter) == data:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, staged = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(gzip.compress(data, mtime=0))
        move_durably(staged, target)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    return True


def refresh_raw_tier(
    download_dir: str | Path,
    *,
    end: date | None = None,
    client: SecEdgarClient | None = None,
    start: date = FIRST_13F_QUARTER,
) -> RawTierRefreshResult:
    """Keep the EDGAR quarterly indexes under ``<download_dir>/sec_edgar`` current.

    Parameters
    ----------
    download_dir : str or Path
        The download root.
    end : date or None, default None
        The day the refresh runs as: its quarter is the last pulled. ``None``
        is the client's UTC day.
    client : SecEdgarClient, optional
        The client; one reading ``SEC_EDGAR_USER_AGENT`` when omitted.
    start : date, default ``FIRST_13F_QUARTER``
        A day of the oldest quarter pulled.

    Returns
    -------
    RawTierRefreshResult
        One part per quarter pulled (``2024QTR1``): ``published``,
        ``unchanged``, or ``not on EDGAR yet``; a quarter that failed with
        its error.

    Raises
    ------
    SecEdgarCredentialError
        If ``SEC_EDGAR_USER_AGENT`` is unset; nothing was sent.

    Examples
    --------
    >>> import tempfile
    >>> from quantlab.acquisition._support.http import HttpResponse
    >>> body = b"CIK|Company Name|Form Type|Date Filed|Filename\\n---\\n"
    >>> client = SecEdgarClient(user_agent="Example research example@example.invalid",
    ...                         http_get=lambda url, *, params, headers: HttpResponse(200, body), sleep=lambda s: None)
    >>> refresh_raw_tier(tempfile.mkdtemp(), end=date(2024, 2, 1), client=client, start=date(2024, 1, 1)).summary()
    '2024QTR1: published'
    """
    client = client if client is not None else SecEdgarClient()
    vendor_root = Path(download_dir) / VENDOR_DIR
    pulled_at = client.now()
    today = end if end is not None else pulled_at.astimezone(UTC).date()
    parts = []
    for year, quarter in due_quarters(vendor_root, today, start):
        name = f"{year:04d}QTR{quarter}"
        try:
            data = client.get_index(year, quarter)
        except QUARTER_FAILURES as error:
            parts.append(RefreshedPart.failure(name, str(error)))
            continue
        if data is None:
            parts.append(RefreshedPart(name, "not on EDGAR yet"))
            continue
        published = _publish(vendor_root, year, quarter, data)
        write_watermark(
            watermark_file(vendor_root, year, quarter),
            pulled_at.astimezone(UTC).date(),
            {"quarter": name, "pulled_at": pulled_at.astimezone(UTC).isoformat()},
        )
        parts.append(RefreshedPart(name, "published" if published else "unchanged"))
    return RawTierRefreshResult(VENDOR_DIR, tuple(parts))


#: The registered descriptor of SEC EDGAR's filing indexes.
SEC_EDGAR_SOURCE = register_source(
    SourceDescriptor(
        vendor="sec_edgar",
        display_name="SEC EDGAR (quarterly filing indexes; free)",
        symbol_batch_cls=None,
        config_factory=None,
        capabilities=(
            # Every quarter's master index of filings: raw only, read by the StockFit 13F panel.
            Capability(market="us_equity", frequency="1d", data_type="filing_index", earliest_available="1993-01-01"),
        ),
        required_env=(USER_AGENT_ENV,),
        raw_tier_refresh=refresh_raw_tier,
    )
)
