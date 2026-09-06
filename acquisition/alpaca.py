import os

import polars as pl
import requests

from base.acquisition import Acquisition
from base.config import AcquisitionConfig

#: The two environment variables Alpaca market-data credentials are read from.
#:
#: MODULE-LEVEL on purpose, and read by `AlpacaAcquisition._scrub` from here
#: rather than off `_AlpacaMarketDataClient`. The transport class is a patch
#: target -- `tests/conftest.py:mock_alpaca_client` replaces it wholesale, and
#: so would any future fake -- so a redaction routine that reached its variable
#: names THROUGH that symbol could be silently disabled by substituting a stub
#: that happens not to define them. A security control must not be reachable
#: through an indirection whose whole purpose is to be replaced.
#:
#: `_AlpacaMarketDataClient` re-exposes both as class attributes because they
#: are part of its own published contract (and the acceptance criteria assert
#: them there), but they are DEFINED here.
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"


class _AlpacaMarketDataClient:
    """Thin, single-page Alpaca Market Data transport built on `requests`.

    **Why this exists rather than `alpaca-py`.** The official SDK's market-data
    client (`StockHistoricalDataClient._get_marketdata()`) loops EVERY page into
    one in-memory dict and discards `next_page_token` before returning, and
    `StockBarsRequest` exposes no `page_token` field at all. Page-level resume
    (D-03) is therefore unreachable through its public API, and a whole batch
    materialises in RAM before a volume guard (D-09) could ever see it. Two
    endpoints and two headers do not justify taking a dependency that forecloses
    the phase's central requirement.

    **Credentials are read from `os.environ` HERE and never leave this object.**
    They are not assigned to `self.config`, to any dataclass-facing attribute,
    or to anything `asdict()` can reach -- `AcquisitionConfig.to_dict()` lands
    in persisted configs and in the JSON saved beside model checkpoints, and
    this repo has already leaked one real vendor key exactly that way. Same rule
    and same shape as `acquisition/tiingo.py:TiingoAcquisition.__init__`
    (D-15, T-03.2-02, CLAUDE.md 凭证安全).
    """

    #: PINNED, and never a scheme or host read from config (T-03.2-08). A
    #: config-overridable base URL turns a config file into a credential
    #: exfiltration primitive: the session sends `APCA-API-KEY-ID` and
    #: `APCA-API-SECRET-KEY` on every request, to whatever host it is pointed at.
    BASE_URL = "https://data.alpaca.markets/v2"

    #: Re-exposed from the module-level constants above; see their comment for
    #: why the definition does not live on this (patchable) class.
    KEY_ENV = KEY_ENV
    SECRET_ENV = SECRET_ENV

    #: Per-request timeout in seconds. A request with no timeout can hang a
    #: worker thread indefinitely against an unresponsive vendor, which in a
    #: 15k-symbol backfill silently deadlocks the whole run.
    TIMEOUT_SECONDS = 60

    def __init__(self) -> None:
        key = os.environ.get(self.KEY_ENV)
        secret = os.environ.get(self.SECRET_ENV)
        if not key or not secret:
            raise RuntimeError(
                f"{self.KEY_ENV} and {self.SECRET_ENV} environment variables "
                f"must both be set. Alpaca market-data credentials are read "
                f"from the environment and are never stored on the config "
                f"(D-15) -- export them before running acquisition (see your "
                f"Alpaca dashboard for the key pair)."
            )
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
            }
        )

    def get_page(self, path: str, params: dict) -> dict:
        """Issue exactly ONE request and return the raw envelope.

        The envelope is returned INCLUDING `next_page_token` -- that token is
        the entire reason this class exists instead of the SDK's page-looping
        client. TLS verification is left at the `requests` default and is never
        disabled.
        """
        response = self._session.get(
            f"{self.BASE_URL}{path}",
            params=params,
            timeout=self.TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()


class AlpacaAcquisition(Acquisition):
    """Alpaca Market Data acquisition behind the shared `Acquisition` base.

    Multi-symbol per request and genuinely paginated, so this is the vendor the
    batched primitive was inverted for: `_fetch_page` issues one request and
    hands back `(rows, next_page_token)`, and the base's `_fetch_batch` owns the
    page loop, the shard writes and the ledger.

    **Subscription tier is UNRESOLVED (D-12).** The feed level is a
    `config.kwargs["feed"]` parameter with NO in-code default, and this
    docstring deliberately makes no claim about whether the free (Basic) tier
    reaches historical SIP data. When `feed` is unset the parameter is OMITTED
    from the request entirely and the vendor picks the best feed the account's
    subscription allows -- which is the only honest behaviour until the tier
    question is settled empirically against a real credential (03.2-07).

    **Credentials never touch this object's config.** See
    `_AlpacaMarketDataClient`.
    """

    VENDOR = "alpaca"

    #: Symbols per request. 100 is comfortably under the 200 a single forum post
    #: about an older API version cites, and the real ceiling is NOT documented.
    #: It is deliberately NOT encoded as a `MAX_BATCH_SIZE` constant, because a
    #: constant asserts a vendor fact this project has not verified. The actual
    #: limit is probed once credentials exist (03.2-07); until then this is a
    #: conservative working value, overridable via `config.kwargs["batch_size"]`.
    DEFAULT_BATCH_SIZE = 100

    #: Project frequency token -> Alpaca `timeframe` parameter. A mapping rather
    #: than an inline literal inside `_fetch_page`, so adding a bar size is a
    #: one-line data change rather than an edit to request-building logic.
    #:
    #: `1Min` and `1Day` are both from the vendor's documented grammar
    #: (`[1-59]Min`, `[1-23]Hour`, `1Day`, `1Week`, `[1,2,3,4,6,12]Month`).
    TIMEFRAME_MAP = {"1d": "1Day", "1m": "1Min"}

    #: The trading session whose calendar day the intraday `date=` hive key is
    #: derived from -- RESEARCH Assumption A8, decided at plan time and stated
    #: here rather than only in research.
    #:
    #: Alpaca DAILY bars are date-stamped, but MINUTE bars carry true intraday
    #: UTC instants. Parsing them as UTC and dropping the zone (the convention
    #: `acquisition/tiingo.py` established and `_fetch_page` below follows)
    #: yields naive UTC, which is self-consistent but is NOT US/Eastern market
    #: time: a 09:30 ET bar reads as 14:30.
    #:
    #: That is fine for the timestamp VALUES -- they stay naive UTC, unchanged,
    #: matching every other timestamp in this codebase -- but it is NOT fine for
    #: a day-boundary key. The US regular session runs 14:30-21:00 UTC and
    #: extended hours run past 01:00 UTC of the following calendar day, so a
    #: UTC-derived `date=` key files the last ~4 hours of EVERY session under
    #: the FOLLOWING day. A "give me one trading day" query is then wrong at
    #: both edges, and wrong in the shape that reads as sparse data rather than
    #: as a bug: the close missing, the previous session's tail present.
    #:
    #: So ONLY the derived partition key converts. Do not "simplify" this to a
    #: plain `dt.date()` truncation on the grounds that the timestamps are UTC
    #: anyway -- that is precisely the mistake, and
    #: `test_an_0200_utc_bar_lands_in_the_previous_days_session_partition` is
    #: the test that catches it.
    SESSION_TIME_ZONE = "America/New_York"

    #: Data type -> endpoint path. `quotes` and `trades` land in 03.2-06.
    ENDPOINT_MAP = {"bars": "/stocks/bars"}

    #: The vendor's single-letter bar fields -> this project's column names.
    #: This mapping is the single most likely place for the class to be quietly
    #: wrong: a swapped `o`/`c` produces plausible-looking data forever. It is
    #: pinned here and asserted in tests/test_alpaca_acquisition.py.
    FIELD_MAP = {
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "n": "trade_count",
        "vw": "vwap",
    }

    #: The pinned shard projection AND order -- see `Acquisition.RAW_COLUMNS`.
    RAW_COLUMNS = (
        "timestamp",
        "symbol",
        "vendor",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "vwap",
    )

    #: The dtype-explicit schema an EMPTY page is built with. Without it a
    #: no-rows response produces a schemaless frame whose `symbol` column
    #: cannot be read, so "no data" would raise instead of reporting absence.
    RAW_SCHEMA = {
        "timestamp": pl.Datetime,
        "symbol": pl.String,
        "vendor": pl.String,
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
        "trade_count": pl.Float64,
        "vwap": pl.Float64,
    }

    #: What a credential value is replaced with in any captured message.
    REDACTION = "<APCA CREDENTIAL REDACTED>"

    #: The two credentials the shared `Acquisition._scrub` redacts before any
    #: message reaches a log line or the failure manifest (T-03.2-01).
    #:
    #: Built from the MODULE-LEVEL constants, never from
    #: `_AlpacaMarketDataClient` -- that class is a patch target and a security
    #: control must not be reachable through an indirection whose whole purpose
    #: is to be replaced (03.2-02 deviation #2). Alpaca sends credentials in
    #: HEADERS rather than in the URL, so the Tiingo query-parameter leak shape
    #: does not apply here directly; scrubbed anyway, because a `requests`
    #: exception chain can reach `exc.request.headers` and a manifest is not a
    #: place to rely on a vendor's choice of auth transport staying the same.
    CREDENTIAL_ENV_VARS = (KEY_ENV, SECRET_ENV)

    #: HTTP statuses that mean "slow down", NOT "you are out of allocation".
    #:
    #: 429 here is a per-MINUTE ceiling -- 200 requests/min on the free
    #: (Basic) tier -- that a healthy full-market run is EXPECTED to hit
    #: repeatedly and that clears in under a minute. Alpaca publishes no
    #: request-allocation concept at all, so this class has no quota state and
    #: deliberately declares no `QUOTA_STATUS_CODES`: reading this 429 as
    #: global would abort every Alpaca run within seconds of starting while
    #: logging an allocation message for a vendor that has no allocation
    #: (T-03.2-16, 03.2-RESEARCH.md Pitfall 1).
    RATE_LIMIT_STATUS_CODES = frozenset({429})

    #: Response headers the vendor MAY send alongside a 429. Read defensively
    #: for logging only: a missing header is NO INFORMATION, never a default,
    #: and never a computed reset instant -- Alpaca does not guarantee these on
    #: every response, so the backoff interval stays a configured constant.
    RATE_LIMIT_HEADERS = (
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    )

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)
        self._client = _AlpacaMarketDataClient()

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Whatever `X-RateLimit-*` the vendor happened to send, or `{}`.

        Purely informational. An ABSENT header contributes no entry rather
        than a default one, so a caller can distinguish "the vendor said
        nothing" from "the vendor said zero" -- the two mean opposite things
        and a default would silently merge them.
        """
        response = self._vendor_response(exc)
        headers = getattr(response, "headers", None) or {}
        return {
            name: str(headers[name])
            for name in self.RATE_LIMIT_HEADERS
            if name in headers
        }

    def _classify_error(self, exc: BaseException) -> str:
        """Alpaca reads 429 as TRANSIENT -- the opposite of Tiingo's reading.

        See `RATE_LIMIT_STATUS_CODES`. Everything else is per-unit: this
        vendor has no global condition to report, so nothing here can ever
        return `"quota"`.
        """
        if self._status_of(exc) in self.RATE_LIMIT_STATUS_CODES:
            return "rate_limited"
        return "failed"

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """One `GET /v2/stocks/bars` request, flattened to rows.

        Returns `(frame, next_page_token)`. `frame` is projected and ordered to
        `RAW_COLUMNS`; a `None` token means this was the batch's last page.
        """
        symbols = self._validate_symbols(symbols)

        params = {
            "symbols": ",".join(symbols),
            "timeframe": self.TIMEFRAME_MAP[self.config.frequency],
            "start": start_date,
            "end": end_date,
            # The vendor maximum. Fewer rows per page means more requests for
            # the same data, which is pure rate-limit pressure.
            "limit": self._knob("page_limit", 10_000),
            "adjustment": self._knob("adjustment", "raw"),
            # PINNED to "asc", never read from a knob. Alpaca sorts symbol-major
            # then timestamp; with `asc` that is a total, monotone order, so the
            # ledger's "furthest position reached" is a well-defined resume
            # point. A `desc` request inverts the ordering and makes that
            # recorded position meaningless -- a resumed run would re-fetch what
            # it had and skip what it had not.
            "sort": "asc",
            # Passed EXPLICITLY, including as None, and never left to the
            # vendor's default of "today". That default maps each symbol onto
            # whatever entity holds that ticker NOW, so a delisted ticker
            # silently returns the current occupant's history -- exactly the
            # survivorship bias the point-in-time roster exists to remove
            # (03.2-RESEARCH.md Pitfall 5).
            "asof": self._knob("asof", None),
        }

        # OMITTED entirely when unset (assumption O-1 / D-12): no in-code feed
        # default is written anywhere in this phase, because whether the free
        # tier reaches historical SIP data is unresolved. Sending an unset feed
        # as an empty string would be a claim; omitting it lets the vendor pick
        # what the subscription allows.
        feed = self._knob("feed", None)
        if feed:
            params["feed"] = feed

        if page_token:
            params["page_token"] = page_token

        payload = self._client.get_page(self.ENDPOINT_MAP["bars"], params)

        rows = [
            {
                "symbol": symbol,
                "vendor": self.VENDOR,
                **{
                    self.FIELD_MAP[key]: value
                    for key, value in bar.items()
                    if key in self.FIELD_MAP
                },
            }
            for symbol, bars in (payload.get("bars") or {}).items()
            for bar in (bars or [])
        ]

        if not rows:
            frame = pl.DataFrame(schema=self.RAW_SCHEMA)
            return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")

        frame = pl.DataFrame(rows)
        # Alpaca returns RFC-3339 with a trailing `Z`, the same shape Tiingo
        # does. Parse as UTC then DROP the zone, so the dtype is a naive
        # `pl.Datetime` matching every other timestamp in this codebase --
        # parsing without an explicit time zone raises on tz-aware strings, and
        # a tz-aware column would compare unequal to the naive filter bounds
        # `StockDataset._scan_raw` builds.
        frame = frame.with_columns(
            pl.col("timestamp")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        frame = frame.cast(
            {
                name: dtype
                for name, dtype in self.RAW_SCHEMA.items()
                if name != "timestamp" and name in frame.columns
            }
        )
        return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")
