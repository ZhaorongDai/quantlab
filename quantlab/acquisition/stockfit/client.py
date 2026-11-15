"""The StockFit client: one GET with a bearer key, kept inside the vendor's rate limits.

StockFit (``https://api.stockfit.io/v1``) takes its key as
``Authorization: Bearer $STOCKFIT_API_KEY`` and limits requests on two levels:
a sliding 60-second window and a hard daily cap reset at 00:00 UTC. Every
response states both in its headers (``X-RateLimit-Limit-Minute``,
``X-RateLimit-Remaining-Minute``, ``X-RateLimit-Limit-Day``,
``X-RateLimit-Remaining-Day``), and the limits differ by key, so the client
reads them from the headers and hard-codes neither:

- before a request it waits until the minute window has room, counting its
  own requests of the last 60 seconds against the stated per-minute limit,
  and waiting for the window to roll when the vendor says none remain;
- once the vendor says no more than ``day_reserve`` requests remain today, it
  refuses to send more (``StockFitDailyCapReached``) until the UTC day
  changes, so a refresh stops cleanly instead of running into HTTP 429;
- an HTTP 429 while requests remain today is waited out (the server's
  ``Retry-After``, or a full minute window without one) and sent again, as
  often as it takes: a throttle never fails a request; a 429 with none left
  today raises ``StockFitDailyCapReached``;
- an HTTP 5xx, or a broken connection, is retried after the server's
  ``Retry-After``, up to ``retries`` times
  (``quantlab.acquisition._support.request_policy``).

A 401 or 403 raises ``StockFitEntitlementError``; a 404 returns ``None``
(the vendor holds nothing for that entity); any other 4xx, or a 200 body that
is not JSON, raises ``StockFitResponseError``. The key is read from the
environment only, kept out of every message, and never sent in a URL.

Examples
--------
Needs ``STOCKFIT_API_KEY`` and the network::

    client = StockFitClient()
    client.get_json("/api/financials/income-statement", {"cik": 1093691, "period": "quarter"})
    client.remaining_day
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from quantlab.acquisition._support.http import HttpResponse, Transport, http_get
from quantlab.acquisition._support.request_policy import (
    REFUSAL_STATUSES,
    RequestPolicy,
    TransientError,
    require_env,
    retry_after,
)

#: The environment variable holding the API key.
API_KEY_ENV = "STOCKFIT_API_KEY"

#: The API's base URL; every endpoint path is appended to it.
BASE_URL = "https://api.stockfit.io/v1"

#: Seconds to connect, and to wait for each read.
TIMEOUT = (30, 120)

#: The length of the per-minute window, in seconds.
WINDOW_SECONDS = 60.0

#: The rate-limit headers StockFit sends with every response.
LIMIT_MINUTE = "X-RateLimit-Limit-Minute"
REMAINING_MINUTE = "X-RateLimit-Remaining-Minute"
LIMIT_DAY = "X-RateLimit-Limit-Day"
REMAINING_DAY = "X-RateLimit-Remaining-Day"


class StockFitCredentialError(RuntimeError):
    """``STOCKFIT_API_KEY`` is not set."""


class StockFitEntitlementError(RuntimeError):
    """StockFit refused the key (HTTP 401) or the endpoint for the key's plan (HTTP 403)."""


class StockFitTransportError(TransientError):
    """The connection failed or broke, or StockFit answered a server error; retried."""


class StockFitResponseError(ValueError):
    """StockFit answered a request error (a 4xx other than 401, 403, 404 and 429) or an unreadable body."""


class StockFitDailyCapReached(RuntimeError):
    """The day's requests are spent down to the reserve; nothing more is sent until 00:00 UTC."""


@dataclass(frozen=True)
class StockFitCredentials:
    """The API key, read from the environment; ``repr`` hides it.

    Examples
    --------
    >>> StockFitCredentials("k")
    StockFitCredentials(<hidden>)
    """

    api_key: str = field(repr=False)

    def __repr__(self) -> str:
        """Return a repr without the key."""
        return "StockFitCredentials(<hidden>)"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> StockFitCredentials:
        """Read the key from ``STOCKFIT_API_KEY``.

        Raises
        ------
        StockFitCredentialError
            If the variable is unset or empty.

        Examples
        --------
        >>> StockFitCredentials.from_env({})
        Traceback (most recent call last):
        ...
        quantlab.acquisition.stockfit.client.StockFitCredentialError: STOCKFIT_API_KEY not set; ...
        """
        (api_key,) = require_env(
            (API_KEY_ENV,),
            hint="set it in the environment. Nothing was downloaded.",
            error=StockFitCredentialError,
            environ=environ,
        )
        return cls(api_key)


def http_transport(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> HttpResponse:
    """Send one GET; the client's default ``http_get``.

    ``quantlab.acquisition._support.http.http_get`` with ``TIMEOUT``, not
    following redirects (the key travels in a header), and raising
    ``StockFitTransportError`` for a connection that fails or breaks.

    Examples
    --------
    Needs the network; without a key StockFit answers 401::

        http_transport(BASE_URL + "/api/symbols", params={}, headers={}).status
    """
    return http_get(url, params=params, headers=headers, timeout=TIMEOUT, error=StockFitTransportError)


def _header_int(headers: Mapping[str, str], name: str) -> int | None:
    lowered = {key.lower(): value for key, value in headers.items()}
    try:
        return int(lowered[name.lower()])
    except (KeyError, TypeError, ValueError):
        return None


class StockFitClient:
    """Send GETs to StockFit within the key's rate limits; see the module docstring.

    Parameters
    ----------
    credentials : StockFitCredentials, optional
        The key; read from ``STOCKFIT_API_KEY`` when omitted.
    http_get : Transport, optional
        ``http_get(url, params=..., headers=...) -> HttpResponse``; the real
        HTTP adapter ``http_transport`` when omitted. Tests pass a fake vendor.
    retries : int, default 5
        Retries of a failed request (a 5xx or a broken connection) before it raises.
    backoff : float, default 2.0
        Seconds before the first retry of a failure without ``Retry-After``;
        doubled on each later one.
    day_reserve : int, default 0
        Requests of the daily cap left unspent (for other work on the same
        key); the client stops once the vendor says no more remain.
    sleep : callable, default time.sleep
        How the client waits.
    now : callable, default the system clock in UTC
        The current time, an aware ``datetime``; it dates pulls too.

    Attributes
    ----------
    remaining_day : int or None
        Requests left today as the last response said; ``None`` before the
        first response of the UTC day.
    requests : int
        Requests sent by this client.

    Examples
    --------
    >>> from quantlab.acquisition._support.http import HttpResponse
    >>> def vendor(url, *, params, headers):
    ...     return HttpResponse(200, b"[]", {"X-RateLimit-Remaining-Day": "41"})
    >>> client = StockFitClient(credentials=StockFitCredentials("k"), http_get=vendor)
    >>> client.get_json("/api/symbols", {})
    []
    >>> client.remaining_day
    41
    """

    def __init__(
        self,
        *,
        credentials: StockFitCredentials | None = None,
        http_get: Transport | None = None,
        retries: int = 5,
        backoff: float = 2.0,
        day_reserve: int = 0,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        """Initialize; see the class docstring."""
        self.credentials = credentials if credentials is not None else StockFitCredentials.from_env()
        self.http_get = http_get if http_get is not None else http_transport
        self.retries = retries
        self.backoff = backoff
        self.day_reserve = day_reserve
        self.sleep = sleep
        self.now = now
        self.requests = 0
        self._sent: deque[datetime] = deque()
        self._limit_minute: int | None = None
        self._remaining_minute: int | None = None
        self._day: tuple[date, int] | None = None

    @property
    def remaining_day(self) -> int | None:
        """Requests left today as the vendor last said; ``None`` before the day's first response.

        Read from the ``X-RateLimit-Remaining-Day`` header of the latest
        response, so it follows the key's own daily limit; a response of an
        earlier UTC day does not count.

        Examples
        --------
        >>> from quantlab.acquisition._support.http import HttpResponse
        >>> def vendor(url, *, params, headers):
        ...     return HttpResponse(200, b"[]", {"X-RateLimit-Remaining-Day": "24999"})
        >>> client = StockFitClient(credentials=StockFitCredentials("k"), http_get=vendor)
        >>> client.remaining_day is None  # nothing asked yet today
        True
        >>> _ = client.get_json("/api/symbols/exchanges", {})
        >>> client.remaining_day
        24999
        """
        if self._day is None or self._day[0] != self.now().astimezone(UTC).date():
            return None
        return self._day[1]

    @property
    def _policy(self) -> RequestPolicy:
        return RequestPolicy(
            {API_KEY_ENV: self.credentials.api_key},
            retries=self.retries,
            backoff=self.backoff,
            sleep=self._wait,
            vendor="StockFit",
            credential=API_KEY_ENV,
            refusal_error=StockFitEntitlementError,
        )

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self.sleep(seconds)

    def _make_room(self) -> None:
        """Refuse past the daily reserve; wait until the minute window has room."""
        remaining = self.remaining_day
        if remaining is not None and remaining <= self.day_reserve:
            raise StockFitDailyCapReached(
                f"StockFit: {remaining} request(s) left today and {self.day_reserve} kept in reserve; "
                f"nothing more is sent before 00:00 UTC."
            )
        now = self.now()
        while self._sent and now - self._sent[0] >= timedelta(seconds=WINDOW_SECONDS):
            self._sent.popleft()
        full = self._limit_minute is not None and len(self._sent) >= self._limit_minute
        if full or self._remaining_minute == 0:
            oldest = self._sent[0] if self._sent else now
            self._wait(WINDOW_SECONDS - (now - oldest).total_seconds() or WINDOW_SECONDS)
            self._remaining_minute = None

    def _read_limits(self, headers: Mapping[str, str]) -> None:
        self._limit_minute = _header_int(headers, LIMIT_MINUTE) or self._limit_minute
        self._remaining_minute = _header_int(headers, REMAINING_MINUTE)
        remaining_day = _header_int(headers, REMAINING_DAY)
        if remaining_day is not None:
            self._day = (self.now().astimezone(UTC).date(), remaining_day)

    def get_json(self, path: str, params: Mapping[str, object]) -> object | None:
        """GET one endpoint path and return its JSON body; ``None`` for a 404.

        Parameters
        ----------
        path : str
            The endpoint's path, for example ``"/api/symbols"``.
        params : Mapping
            The query parameters; never a credential.

        Returns
        -------
        object or None
            The decoded body, or ``None`` when StockFit holds nothing for the
            request (HTTP 404).

        Raises
        ------
        StockFitDailyCapReached
            If the day's requests are spent down to ``day_reserve``; nothing
            was sent.
        StockFitEntitlementError
            If StockFit refused the key (401) or the endpoint (403).
        StockFitTransportError
            If the request kept failing past ``retries``.
        StockFitResponseError
            If a 200 body is not JSON.

        Examples
        --------
        See the class docstring.
        """
        policy = self._policy
        query = {name: str(value) for name, value in params.items()}
        headers = {"Authorization": f"Bearer {self.credentials.api_key}"}

        def send() -> HttpResponse:
            """Send until StockFit answers other than 429: a throttle is waited out, never counted as a failure."""
            while True:
                self._make_room()
                self._sent.append(self.now())
                self.requests += 1
                response = self.http_get(BASE_URL + path, params=query, headers=headers)
                self._read_limits(response.headers)
                if response.status != 429:
                    return response
                response.read()
                if self.remaining_day == 0:
                    raise StockFitDailyCapReached(f"StockFit answered 429 on {path} with no requests left today.")
                self._remaining_minute = None
                self._wait(retry_after(response.headers) or WINDOW_SECONDS)

        def attempt() -> object | None:
            response = send()
            body = response.read()
            if response.status == 404:
                return None
            if response.status in REFUSAL_STATUSES:
                raise policy.refusal(response.status, where=path, detail=policy.scrub(body[:200].decode(errors="replace")))
            if response.status >= 500:
                raise StockFitTransportError(
                    policy.scrub(f"HTTP {response.status} on {path}: {body[:200].decode(errors='replace')}"),
                    retry_after=retry_after(response.headers),
                )
            if response.status != 200:
                raise StockFitResponseError(
                    policy.scrub(f"HTTP {response.status} on {path}: {body[:200].decode(errors='replace')}")
                )
            try:
                return json.loads(body)
            except ValueError as error:
                raise StockFitResponseError(f"{path} answered 200 with a body that is not JSON: {error}") from error

        return policy.call(attempt, lambda tries, wait, error: f"StockFit: {error}; retry {tries + 1} in {wait:g}s.")

    def __repr__(self) -> str:
        """Return a repr without the key."""
        return f"StockFitClient(requests={self.requests}, remaining_day={self.remaining_day})"

