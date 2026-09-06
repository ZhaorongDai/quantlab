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

#: What is sent as `asof` to mean "do NOT map this symbol onto whatever entity
#: holds the ticker today".
#:
#: **A `None` value is not this, and cannot be made to be.** `requests` DROPS
#: any param whose value is `None` before it builds the query string, so an
#: `"asof": None` entry in the params dict reaches the wire as nothing at all
#: and the vendor's current-day default applies -- which maps a delisted ticker
#: onto its current occupant and reintroduces exactly the survivorship bias the
#: point-in-time roster exists to remove. Measured:
#:
#:     >>> requests.Request("GET", url, params={"symbols": "AAPL",
#:     ...     "asof": None}).prepare().url
#:     'https://data.alpaca.markets/v2/stocks/bars?symbols=AAPL'
#:
#: So a real, encodable value has to be sent, and this is it.
#:
#: **UNVERIFIED against a live request, and deliberately named rather than
#: inlined so it stays that way visibly.** `"-"` is the vendor's documented
#: no-mapping sentinel; this project has no credential with which to prove it,
#: and the same one-request probe that settles the `feed` question (03.2-07
#: human-check A) settles this one. The failure mode if it is wrong is a 4xx on
#: every request -- LOUD, refused per batch, and recorded in the failure
#: manifest -- which is the direction to be wrong in. The failure mode of the
#: `None` it replaces was silent, plausible-looking, biased data.
ASOF_NO_MAPPING = "-"

#: Sentinel distinguishing "`asof` was never set" from "`asof` was explicitly
#: set to `None`". `_knob` cannot tell them apart on its own, and they must
#: mean different things: unset sends `ASOF_NO_MAPPING`, while an explicit
#: `None` is the ONE way a caller deliberately asks for the vendor's
#: current-day mapping and is the only path on which the key is omitted.
_ASOF_UNSET = object()


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

    Data types
    ----------
    Three, selected by ONE token (see `_data_type`): daily and minute `bars`
    from `frequency`, and `quotes` or `trades` from `kwargs["data_type"]` under
    `frequency="tick"`. Tick rows are written at FULL resolution -- there is no
    resampling, bucketing or dedup anywhere between `_fetch_page` and the shard
    (D-16).

    Subscription tier -- the numbers
    -------------------------------
    Quoted from the vendor's equities plan comparison
    (docs.alpaca.markets/us/docs/about-market-data-api), read against THIS
    phase's use case, which is a historical backfill whose `end` is by
    definition days or years in the past:

    - **Historical data timeframe: since 2016 on BOTH tiers.** No difference.
      (It happens to equal `ingest_us_equity.py`'s existing default start date.)
    - **Available fields: identical on both tiers.** No difference.
    - **Recency floor: the latest 15 minutes are unavailable on the free
      (Basic) tier, unrestricted on Algo Trader Plus.** Never binds a backfill,
      because a backfill's `end` is never inside the last 15 minutes.
    - **Historical API rate limit: 200/min free versus 10,000/min paid.** A 50x
      factor, and the ONLY difference that matters here. It is the dominant
      cost driver: at 200/min a full-market DAILY backfill is ~8 minutes and a
      full-market MINUTE backfill is ~50 hours and ~358 GB.

    So the tier question is not "can I get the data" but "how much data am I
    asking for" -- which is exactly the question the pre-flight volume guard
    (D-09, `acquisition/universe.py`) forces the user to answer before a run
    starts.

    Subscription tier -- the SIP question is UNRESOLVED
    --------------------------------------------------
    Whether the free (Basic) tier can request historical SIP data at all is
    **not settled**, and this class takes no position on it. Two vendor pages
    disagree:

    - *Reading 1 -- yes, if older than 15 minutes.* "For historical queries,
      the `end` parameter must be at least 15 minutes old to query SIP data
      without a subscription" (market-data-faq), which frames the restriction
      as recency-only. The plan table's "Historical data limitation: latest 15
      minutes" and the bars reference's `end` default both read the same way.
    - *Reading 2 -- no, IEX only.* The `iex` row of the data-sources table
      states flatly "This is the only feed that can be used without a
      subscription" (historical-stock-data-1).

    Three sources support reading 1 and one contradicts it. **That is a vote
    count, not evidence**, and it is not presented here as one. The difference
    is material: if reading 2 holds, free-tier quotes and trades are IEX-only
    -- roughly 2.5% of consolidated volume -- which would make free-tier tick
    data unusable for research.

    Consequently `feed` is a `config.kwargs` parameter with **NO in-code
    default**, in either direction. When it is unset the parameter is OMITTED
    from the request entirely and the vendor picks the best feed the account's
    subscription allows; sending an unset feed as any concrete value would be a
    claim this project has not earned. The resolution path is a single
    one-request probe against a real credential, planned in 03.2-07 -- not more
    documentation reading, which is what produced the conflict.

    Corporate actions
    -----------------
    The abstraction does not preclude them, and that accommodation required no
    code: **there is no corporate-actions method, no unreachable branch and no
    unused constant here**, and a test asserts the class exposes no such
    surface. D-07 scopes corporate actions out of this phase, and the codebase
    has twice refused the "add a placeholder now" fork.

    When it IS implemented, it must never be described as a replacement for
    the Tiingo `supported_tickers.csv` delisting signal: Alpaca's Corporate
    Actions API explicitly EXCLUDES delistings and reorganizations
    (02-08-RESEARCH.md:11). The two answer different questions, and conflating
    them would silently reintroduce the survivorship bias the point-in-time
    roster exists to remove.

    Credentials
    -----------
    Market-data endpoints ONLY. No trading API, no broker API, and no
    paper/live switch -- the market-data API does not distinguish the two, so a
    switch here would be a knob with no effect pretending to be a safety
    control (D-15).

    `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` are read from `os.environ` in
    `_AlpacaMarketDataClient.__init__` and are never assigned to a config
    dataclass: `AcquisitionConfig.to_dict()` is `asdict(self)` and lands in
    persisted configs and in the JSON saved beside model checkpoints, and this
    repo has already leaked one real vendor key exactly that way.
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

    #: Data type -> endpoint path. The envelope's row key happens to equal the
    #: data type for all three (`{"bars": {...}}`, `{"quotes": {...}}`,
    #: `{"trades": {...}}`), so one token indexes the request AND the response.
    ENDPOINT_MAP = {
        "bars": "/stocks/bars",
        "quotes": "/stocks/quotes",
        "trades": "/stocks/trades",
    }

    #: The data types reachable under `frequency="tick"`, and the ONLY accepted
    #: values of the `data_type` knob. There is deliberately no default: quotes
    #: and trades land under ONE vendor root, distinguished only by the leading
    #: `data_type=` hive key, so a wrong default files one as the other with
    #: the other's projection applied on the way in (T-03.2-26).
    TICK_DATA_TYPES = ("quotes", "trades")

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

    #: The vendor's quote fields -> this project's column names. `c` is
    #: CONDITIONS here and CLOSE on a bar; that collision is the whole reason
    #: the field map is per data type rather than one shared mapping.
    QUOTE_FIELD_MAP = {
        "t": "timestamp",
        "bx": "bid_exchange",
        "bp": "bid_price",
        "bs": "bid_size",
        "ax": "ask_exchange",
        "ap": "ask_price",
        "as": "ask_size",
        "c": "conditions",
        "z": "tape",
    }

    #: The vendor's trade fields -> this project's column names.
    TRADE_FIELD_MAP = {
        "t": "timestamp",
        "x": "exchange",
        "p": "price",
        "s": "size",
        "i": "trade_id",
        "c": "conditions",
        "z": "tape",
    }

    #: Data type -> its field map. `FIELD_MAP` above stays the bars entry and
    #: keeps its name: it is pinned by direct equality in
    #: tests/test_alpaca_acquisition.py and is what a reader looking for "the
    #: bar mapping" will search for.
    FIELD_MAP_BY_DATA_TYPE = {
        "bars": FIELD_MAP,
        "quotes": QUOTE_FIELD_MAP,
        "trades": TRADE_FIELD_MAP,
    }

    #: The pinned shard projection AND order, PER DATA TYPE -- see
    #: `Acquisition.RAW_COLUMNS`. Each entry begins
    #: `("timestamp", "symbol", "vendor")` and then names that endpoint's own
    #: fields, so the raw tier is schema-stable by construction and a directory
    #: scan never has to relax its strictness (03.2-RESEARCH.md Pitfall 6).
    #:
    #: The three sets are deliberately DISJOINT beyond that shared prefix. A
    #: shared projection would force a trade to carry `bid_price`/`ask_price`
    #: as nulls -- columns that have no meaning on a trade at all -- producing
    #: a schema that describes neither endpoint.
    RAW_COLUMNS_BY_DATA_TYPE = {
        "bars": (
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
        ),
        "quotes": (
            "timestamp",
            "symbol",
            "vendor",
            "bid_exchange",
            "bid_price",
            "bid_size",
            "ask_exchange",
            "ask_price",
            "ask_size",
            "conditions",
            "tape",
        ),
        "trades": (
            "timestamp",
            "symbol",
            "vendor",
            "exchange",
            "price",
            "size",
            "trade_id",
            "conditions",
            "tape",
        ),
    }

    #: The dtype-explicit schema an EMPTY page is built with, per data type.
    #: Without it a no-rows response produces a schemaless frame whose `symbol`
    #: column cannot be read, so "no data" would raise instead of reporting
    #: absence. It also types the columns a sparse response omits entirely --
    #: `conditions` is genuinely optional on the wire.
    RAW_SCHEMA_BY_DATA_TYPE = {
        "bars": {
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
        },
        "quotes": {
            "timestamp": pl.Datetime,
            "symbol": pl.String,
            "vendor": pl.String,
            "bid_exchange": pl.String,
            "bid_price": pl.Float64,
            "bid_size": pl.Float64,
            "ask_exchange": pl.String,
            "ask_price": pl.Float64,
            "ask_size": pl.Float64,
            "conditions": pl.List(pl.String),
            "tape": pl.String,
        },
        "trades": {
            "timestamp": pl.Datetime,
            "symbol": pl.String,
            "vendor": pl.String,
            "exchange": pl.String,
            "price": pl.Float64,
            "size": pl.Float64,
            "trade_id": pl.Int64,
            "conditions": pl.List(pl.String),
            "tape": pl.String,
        },
    }

    #: Per data type, the columns the vendor may legitimately omit from EVERY
    #: row of a page. `conditions` is genuinely optional on the wire.
    #:
    #: Any OTHER absent column is a MAPPING FAILURE -- a field map that no
    #: longer matches the envelope -- and `_fetch_page` raises on it rather
    #: than filling it with nulls. That distinction is load-bearing: a mutation
    #: swapping `TRADE_FIELD_MAP` for `QUOTE_FIELD_MAP` maps only `t`, `c` and
    #: `z`, so a blanket null-fill would write an all-null `price` column and
    #: keep doing so forever, looking exactly like a stretch of untraded
    #: symbols. It is the same class of silent-wrongness as a swapped `o`/`c`.
    OPTIONAL_COLUMNS_BY_DATA_TYPE = {
        "bars": (),
        "quotes": ("conditions",),
        "trades": ("conditions",),
    }

    #: Closed literal sets for the request knobs a caller can set (ASVS V5).
    #: An out-of-set value RAISES rather than being forwarded: the vendor would
    #: reject some of them and silently ignore others, and "silently ignored"
    #: means a run that asks for split-adjusted bars and stores raw ones.
    #: Sourced from `alpaca_py-0.44.0/alpaca/data/enums.py`.
    FEED_VALUES = frozenset(
        {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}
    )
    ADJUSTMENT_VALUES = frozenset({"raw", "split", "dividend", "all"})
    SORT_VALUES = frozenset({"asc", "desc"})

    #: PINNED to `asc`, never read from a knob -- see `_fetch_page`.
    SORT = "asc"

    #: Re-exposed from the module-level constant; see its comment for why a
    #: `None` `asof` is not an option and why this value is still unverified.
    ASOF_NO_MAPPING = ASOF_NO_MAPPING

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
        # Resolve and validate EAGERLY, before the transport exists, so a bad
        # `data_type`/`feed`/`adjustment` raises at construction rather than
        # inside a worker thread where `_attempt_batch` would classify it as a
        # per-batch failure and file it in the manifest as if the vendor had
        # rejected it.
        self._assert_knobs_are_in_range()
        self._client = _AlpacaMarketDataClient()

    # -- data type: one knob drives the endpoint AND the projection ---------

    @property
    def _data_type(self) -> str:
        """`bars`, `quotes` or `trades` -- resolved ONCE and used for both the
        endpoint and the written projection, so the two cannot disagree.

        Bar frequencies resolve from `config.frequency`; `tick` resolves from
        `config.kwargs["data_type"]` with NO default. `tick` plus a knob covers
        both tick shapes without touching `enums/data.py`'s locked `Frequency`
        literal set, whose extension would require revisiting 02-RESEARCH.md
        Assumptions Log A2.
        """
        frequency = self.config.frequency
        if frequency in self.TIMEFRAME_MAP:
            return "bars"

        data_type = self._knob("data_type", None)
        if data_type not in self.TICK_DATA_TYPES:
            raise ValueError(
                f"{self.class_name}: frequency {frequency!r} needs "
                f"kwargs['data_type'] set to one of "
                f"{sorted(self.TICK_DATA_TYPES)}; got {data_type!r}. There is "
                f"deliberately NO default -- quotes and trades land under the "
                f"same vendor root, distinguished only by the leading "
                f"`data_type=` hive key, so guessing here would file one as "
                f"the other with the other's column projection applied."
            )
        return data_type

    @property
    def RAW_COLUMNS(self) -> tuple[str, ...]:  # noqa: N802 - base attr name
        """This run's pinned shard projection, resolved from `_data_type`.

        A PROPERTY rather than a class attribute because the projection depends
        on which endpoint the run reads. `RAW_COLUMNS_BY_DATA_TYPE` stays the
        class-level source of truth for anything introspecting the contract.
        """
        return self.RAW_COLUMNS_BY_DATA_TYPE[self._data_type]

    @property
    def RAW_SCHEMA(self) -> dict:  # noqa: N802 - matches RAW_COLUMNS
        """This run's dtype-explicit empty-page schema."""
        return self.RAW_SCHEMA_BY_DATA_TYPE[self._data_type]

    def _assert_knobs_are_in_range(self) -> None:
        """Reject out-of-set request knobs before any request is built.

        `feed` is checked ONLY when set: an unset feed is omitted from the
        request entirely (D-12 / O-1), and validating a `None` into existence
        would be the in-code default this class refuses to have.
        """
        self._data_type  # resolves and validates, or raises

        for name, allowed, default in (
            ("feed", self.FEED_VALUES, None),
            ("adjustment", self.ADJUSTMENT_VALUES, "raw"),
        ):
            value = self._knob(name, default)
            if value is None:
                continue
            if value not in allowed:
                raise ValueError(
                    f"{self.class_name}: kwargs[{name!r}]={value!r} is not one "
                    f"of {sorted(allowed)}. An unrecognised value is refused "
                    f"rather than forwarded: the vendor rejects some and "
                    f"silently ignores others, and 'silently ignored' means a "
                    f"run that asks for one thing and stores another."
                )

        if self.SORT not in self.SORT_VALUES:  # pragma: no cover - constant
            raise ValueError(f"{self.class_name}: SORT={self.SORT!r} is invalid")

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
        """One `GET /v2/stocks/{bars,quotes,trades}` request, flattened to rows.

        Returns `(frame, next_page_token)`. `frame` is projected and ordered to
        `RAW_COLUMNS` for THIS run's data type; a `None` token means this was
        the batch's last page.

        **No aggregation, anywhere (D-16).** Every row the envelope carries
        becomes exactly one row in the frame and therefore exactly one row in
        the shard. There is no resampling, no bucketing and no dedup on this
        path, and there must not be: a "just resample tick to 1s to save space"
        edit destroys the resolution the tick tier exists to capture, and it
        looks like an optimisation while doing it.
        """
        symbols = self._validate_symbols(symbols)
        data_type = self._data_type

        params = {
            "symbols": ",".join(symbols),
            "start": start_date,
            "end": end_date,
            # The vendor maximum. Fewer rows per page means more requests for
            # the same data, which is pure rate-limit pressure.
            "limit": self._knob("page_limit", 10_000),
            # PINNED to "asc", never read from a knob. Alpaca sorts symbol-major
            # then timestamp; with `asc` that is a total, monotone order, so the
            # ledger's "furthest position reached" is a well-defined resume
            # point. A `desc` request inverts the ordering and makes that
            # recorded position meaningless -- a resumed run would re-fetch what
            # it had and skip what it had not.
            "sort": self.SORT,
        }

        # Sent as a real, ENCODABLE value and never as `None`. `requests` drops
        # None-valued params before the query string is built, so the previous
        # `"asof": None` entry reached the wire as nothing and the vendor's
        # current-day default applied -- mapping each symbol onto whatever
        # entity holds that ticker NOW, so a delisted ticker silently returned
        # the current occupant's history. That is exactly the survivorship bias
        # the point-in-time roster exists to remove (03.2-RESEARCH.md
        # Pitfall 5), and it arrived looking like clean data.
        #
        # `test_asof_survives_query_string_encoding_at_the_transport` asserts
        # this at the PREPARED URL rather than on this dict: the dict is not the
        # request, and asserting on it is what let the defect live behind two
        # green tests.
        asof = self._knob("asof", _ASOF_UNSET)
        if asof is _ASOF_UNSET:
            params["asof"] = self.ASOF_NO_MAPPING
        elif asof is not None:
            params["asof"] = asof
        # An EXPLICIT `kwargs={"asof": None}` is the only path that omits the
        # key, and it means "I want the vendor's current-day mapping". It is
        # reachable on purpose (a caller may genuinely want today's entity map)
        # and it is the one setting that reintroduces the bias above, so it has
        # to be typed deliberately rather than inherited from a default.

        if data_type == "bars":
            # BAR-ONLY parameters. The quotes and trades endpoints have no
            # concept of a bar size or of a price adjustment, and sending
            # either is at best ignored and at worst rejected.
            params["timeframe"] = self.TIMEFRAME_MAP[self.config.frequency]
            params["adjustment"] = self._knob("adjustment", "raw")

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

        payload = self._client.get_page(self.ENDPOINT_MAP[data_type], params)

        # The envelope keys its rows by the data type itself -- `{"bars": ...}`,
        # `{"quotes": ...}`, `{"trades": ...}` -- so one resolved token indexes
        # the endpoint, the response and the projection alike.
        field_map = self.FIELD_MAP_BY_DATA_TYPE[data_type]
        rows = [
            {
                "symbol": symbol,
                "vendor": self.VENDOR,
                **{
                    field_map[key]: value
                    for key, value in item.items()
                    if key in field_map
                },
            }
            for symbol, items in (payload.get(data_type) or {}).items()
            for item in (items or [])
        ]

        schema = self.RAW_SCHEMA
        if not rows:
            frame = pl.DataFrame(schema=schema)
            return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")

        # `infer_schema_length=None` means "infer over the WHOLE page", never
        # the default 100 rows. A page carries up to `limit` (10,000) rows in
        # symbol-major order and the vendor OMITS an absent field from a row
        # rather than nulling it, so a field first appearing at row 101 would be
        # dropped from the frame with no error and no warning:
        #
        #     >>> pl.DataFrame([{"a": 1}] * 150 + [{"a": 2, "b": 9}]).columns
        #     ['a']
        #
        # Both outcomes of that are wrong. An OPTIONAL column (`conditions`)
        # would be null-filled by the branch below -- discarding the conditions
        # data rows 101..N actually carried, which is the exact silent data loss
        # `OPTIONAL_COLUMNS_BY_DATA_TYPE` exists to prevent, firing on the wrong
        # side because inference removed the column rather than the vendor. A
        # required one (`vwap`, `price`, `bid_price`) would raise and fail all
        # 100 symbols of the batch every run, with a message accusing the field
        # map of being stale.
        frame = pl.DataFrame(rows, infer_schema_length=None)

        # A field the vendor omitted from EVERY row of this page is filled as a
        # TYPED null column, so the shard's schema is identical either way --
        # which is what keeps a directory scan of the vendor root readable
        # (Pitfall 6) -- but ONLY if that column is declared optional.
        #
        # Anything else absent means the field map no longer matches the
        # envelope, and that MUST raise here. Filling it would write an
        # all-null `price` (or `close`) column that looks like data forever;
        # this is the same silent-wrongness as a swapped `o`/`c`, and it is the
        # one this projection exists to make impossible.
        missing = [name for name in self.RAW_COLUMNS if name not in frame.columns]
        optional = self.OPTIONAL_COLUMNS_BY_DATA_TYPE[data_type]
        unexpected = [name for name in missing if name not in optional]
        if unexpected:
            raise ValueError(
                f"{self.class_name}: the {data_type} envelope produced no "
                f"{unexpected} column(s). Only {list(optional)} may be absent "
                f"from a {data_type} page; anything else means "
                f"FIELD_MAP_BY_DATA_TYPE[{data_type!r}] no longer matches what "
                f"the vendor sends. Refusing rather than writing null columns "
                f"that would read as untraded data forever. Vendor fields seen: "
                f"{sorted(field_map)}."
            )
        if missing:
            frame = frame.with_columns(
                pl.lit(None, dtype=schema[name]).alias(name) for name in missing
            )

        # Alpaca returns RFC-3339 with a trailing `Z`, the same shape Tiingo
        # does. Parse as UTC then DROP the zone, so the dtype is a naive
        # `pl.Datetime` matching every other timestamp in this codebase --
        # parsing without an explicit time zone raises on tz-aware strings, and
        # a tz-aware column would compare unequal to the naive filter bounds
        # `StockDataset._scan_raw` builds.
        #
        # The intraday `date=` hive key is derived from these naive-UTC values
        # by `Acquisition._session_date`, in `SESSION_TIME_ZONE`. The VALUES
        # stay UTC; only that derived key converts.
        frame = frame.with_columns(
            pl.col("timestamp")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        frame = frame.cast(
            {
                name: dtype
                for name, dtype in schema.items()
                if name != "timestamp" and name in frame.columns
            }
        )
        return frame.select(self.RAW_COLUMNS), payload.get("next_page_token")
