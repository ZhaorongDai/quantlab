"""Alpaca Market Data acquisition: daily and minute bars, quotes and trades.

``AlpacaAcquisition`` is the vendor-specific half of the acquisition layer. It
builds one request per page against Alpaca's historical stock endpoints and
hands each page to the shared ``Acquisition`` base, which owns pagination,
resume, concurrency, failure isolation and the raw parquet shards. Credentials
are read from ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` in the environment
and are never stored on a config or written to a log. ``ALPACA_SOURCE`` at the
bottom of the module registers the vendor with ``quantlab.registry``.
"""

import functools
import os

import polars as pl
import requests

from quantlab.registry import (
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset

#: The two environment variables Alpaca credentials are read from.
#:
#: Defined at module level, and read from here by the credential-scrubbing
#: routine, rather than off ``_AlpacaMarketDataClient``. That class is a patch
#: target that tests replace with a fake, and a redaction routine must not
#: depend on a symbol whose whole purpose is to be replaced. The client
#: re-exposes both names as class attributes for its own callers.
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"

#: The ``asof`` value that tells Alpaca not to map a symbol onto whatever
#: entity holds the ticker today.
#:
#: A ``None`` value cannot serve this purpose: ``requests`` drops any parameter
#: whose value is ``None`` before building the query string, so ``"asof": None``
#: reaches the wire as nothing and the vendor's current-day default applies.
#: That default maps a delisted ticker onto its current occupant, which is the
#: survivorship bias the point-in-time roster exists to remove, so a real,
#: encodable value has to be sent. ``"-"`` is the vendor's documented
#: no-mapping value; it has not been verified against a live request. If it is
#: wrong the failure is loud (a 4xx on every request, recorded in the failure
#: manifest), whereas the failure of ``None`` was silent, biased data.
ASOF_NO_MAPPING = "-"

#: Sentinel distinguishing "``asof`` was never set" from "``asof`` was set to
#: ``None``". Unset sends ``ASOF_NO_MAPPING``; an explicit ``None`` is the one
#: way a caller asks for the vendor's current-day mapping, and the only path on
#: which the parameter is omitted.
_ASOF_UNSET = object()


class _AlpacaMarketDataClient:
    """Minimal single-page HTTP transport for Alpaca Market Data.

    The official ``alpaca-py`` client loops over every page internally and
    discards ``next_page_token`` before returning, which makes page-level
    resume impossible and holds a whole batch in memory. This class issues
    exactly one request per call and returns the raw envelope, token included.

    Credentials are read from ``os.environ`` here and live only on the
    ``requests.Session`` headers. They are never copied onto a config or onto
    any attribute a serialiser could reach.
    """

    #: Fixed, never read from config. The session sends both credential
    #: headers to whatever host it targets, so a configurable base URL would
    #: turn a config file into a way to exfiltrate the key.
    BASE_URL = "https://data.alpaca.markets/v2"

    #: Re-exposed from the module-level constants; see their comment for why
    #: the definition does not live on this class.
    KEY_ENV = KEY_ENV
    SECRET_ENV = SECRET_ENV

    #: Per-request timeout in seconds. Without one a worker thread can hang
    #: forever on an unresponsive vendor and stall a whole backfill.
    TIMEOUT_SECONDS = 60

    def __init__(self) -> None:
        """Open a session that carries the credentials from the environment.

        Raises
        ------
        RuntimeError
            If ``APCA_API_KEY_ID`` or ``APCA_API_SECRET_KEY`` is
            unset or empty.
        """
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
        """Issue exactly one request and return the raw JSON envelope.

        The envelope is returned with its ``next_page_token`` intact, which is
        the reason this class exists instead of the SDK's page-looping client.
        TLS verification is left at the ``requests`` default.

        Parameters
        ----------
        path : str
            Endpoint path below ``BASE_URL``, such as ``"/stocks/bars"``.
        params : dict
            Query parameters, sent as given.

        Returns
        -------
        dict
            The decoded response body: the rows keyed by data type (``bars``,
            ``quotes`` or ``trades``) plus ``next_page_token``.

        Raises
        ------
        requests.HTTPError
            On any non-2xx status.

        Examples
        --------
        Needs ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` exported;
        this call reaches the network.

        >>> client = _AlpacaMarketDataClient()
        >>> page = client.get_page(
        ...     "/stocks/bars",
        ...     {"symbols": "AAPL", "timeframe": "1Day",
        ...      "start": "2024-01-02", "end": "2024-01-03"},
        ... )
        >>> page["bars"]["AAPL"], page["next_page_token"]
        """
        response = self._session.get(
            f"{self.BASE_URL}{path}",
            params=params,
            timeout=self.TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()


class AlpacaAcquisition(Acquisition):
    """Alpaca Market Data acquisition behind the shared ``Acquisition`` base.

    Alpaca serves many symbols per request and paginates, so ``_fetch_page``
    issues one request and returns ``(rows, next_page_token)``; the base class
    loops over the pages, writes the shards and keeps the page ledger.

    Data types. One token, resolved by ``_data_type``, selects the endpoint,
    the field map and the shard projection. ``frequency="1d"`` and ``"1m"``
    fetch bars; ``frequency="tick"`` fetches ``quotes`` or ``trades`` and
    requires ``kwargs["data_type"]`` to say which, with no default. Tick rows
    are written at full resolution: nothing between the request and the shard
    resamples, buckets or deduplicates them.

    Subscription tiers. Alpaca's free (Basic) plan and its paid Algo Trader
    Plus plan both serve history since 2016 with the same fields. The free plan
    withholds the latest 15 minutes, which a historical backfill never asks
    for. The difference that matters is the historical request rate: 200
    requests per minute on the free plan against 10,000 on the paid one, a
    factor of 50. At 200 per minute a full-market daily backfill takes about
    8 minutes and a full-market minute backfill about 50 hours and 358 GB, so
    the tier question is really "how much data am I asking for", which the
    pre-flight volume guard in ``quantlab.universe`` makes the user answer
    before a run starts.

    The SIP question is UNRESOLVED. Whether the free plan can request
    historical SIP (consolidated) data at all is not settled by the vendor's
    own documentation, and this class takes no position. One reading is yes,
    for data older than 15 minutes: the FAQ says ``end`` must be at least 15
    minutes old to query SIP data without a subscription, and the plan table
    reads the same way. The other reading is no, IEX only: the data-sources
    table states that IEX is the only feed available without a subscription.
    Three vendor pages support the first reading and one the second; that is
    a vote count, not evidence. The difference is material, because IEX
    carries roughly 2.5% of consolidated volume, and free-tier tick data
    restricted to IEX would be unusable for research. Consequently ``feed``
    is a ``config.kwargs`` option with NO in-code default: when it is unset
    the parameter is omitted from the request and the vendor picks the best
    feed the account allows. Only one request against a real credential can
    settle the question.

    Corporate actions. This class has no corporate-actions method, endpoint
    or constant. If one is ever added it must not be treated as a substitute
    for the Tiingo ``supported_tickers.csv`` delisting signal: Alpaca's
    Corporate Actions API excludes delistings and reorganisations, and
    conflating the two would reintroduce the survivorship bias the
    point-in-time roster removes.

    Credentials. Only the market-data endpoints are used; there is no trading
    or broker API and no paper/live switch, because the market-data API does
    not distinguish the two. ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY``
    are read from the environment inside ``_AlpacaMarketDataClient`` and are
    never assigned to a config dataclass, because ``AcquisitionConfig``
    serialises to JSON beside model checkpoints and a key stored there would
    end up on disk. Their values are redacted from every captured message.

    Examples
    --------
    Needs ``APCA_API_KEY_ID`` and ``APCA_API_SECRET_KEY`` exported.

    >>> from quantlab.base.config import AcquisitionConfig
    >>> cfg = AcquisitionConfig(
    ...     market="us_equity", frequency="1d", vendor="alpaca",
    ...     raw_data_dir_path="downloads/nasdaq_data/alpaca",
    ...     watermark_path="downloads/nasdaq_data/_watermarks/alpaca",
    ...     symbols=("AAPL",), start_date="2024-01-02",
    ...     end_date="2024-01-03",
    ... )
    >>> acq = AlpacaAcquisition(cfg).download()
    >>> acq.coverage_report()
    {'requested': 1, 'pending': 0, 'skipped': 1, 'covered': 1,
     'widened': 0, 'legacy': 0, 'no_data': 0}

    Trades at full resolution use ``frequency="tick"`` together with
    ``kwargs={"data_type": "trades"}``.
    """

    VENDOR = "alpaca"

    #: Symbols per request. Alpaca does not document the ceiling, so this is a
    #: conservative working value rather than a verified limit; override it
    #: through ``config.kwargs["batch_size"]``.
    DEFAULT_BATCH_SIZE = 100

    #: Project frequency token to the Alpaca ``timeframe`` parameter. Both
    #: values come from the vendor's documented grammar. Adding a bar size is
    #: a one-line change here rather than an edit to request building.
    TIMEFRAME_MAP = {"1d": "1Day", "1m": "1Min"}

    #: The time zone the intraday ``date=`` hive key is derived in.
    #:
    #: Minute bars carry true UTC instants, and the timestamp values stay
    #: naive UTC like every other timestamp in the codebase. Only the derived
    #: partition key converts: the US regular session runs 14:30 to 21:00 UTC
    #: and extended hours cross UTC midnight, so a key taken from the UTC date
    #: would file the last hours of every session under the following day and
    #: make a one-day query wrong at both edges.
    SESSION_TIME_ZONE = "America/New_York"

    #: Data type to endpoint path. The envelope keys its rows by the same
    #: token, so one value indexes both the request and the response.
    ENDPOINT_MAP = {
        "bars": "/stocks/bars",
        "quotes": "/stocks/quotes",
        "trades": "/stocks/trades",
    }

    #: The values ``kwargs["data_type"]`` accepts under ``frequency="tick"``.
    #: There is no default: quotes and trades share one vendor root and differ
    #: only by the leading ``data_type=`` hive key, so a guess would file one
    #: as the other with the wrong column projection.
    TICK_DATA_TYPES = ("quotes", "trades")

    #: Alpaca's single-letter bar fields to this project's column names. This
    #: is the most likely place for a quiet error: a swapped ``o``/``c`` would
    #: produce plausible-looking data forever.
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

    #: Alpaca's quote fields to this project's column names. ``c`` means
    #: conditions on a quote and close on a bar, which is why each data type
    #: has its own map.
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

    #: Alpaca's trade fields to this project's column names.
    TRADE_FIELD_MAP = {
        "t": "timestamp",
        "x": "exchange",
        "p": "price",
        "s": "size",
        "i": "trade_id",
        "c": "conditions",
        "z": "tape",
    }

    #: Data type to its field map. ``FIELD_MAP`` stays the bars entry under
    #: its original name.
    FIELD_MAP_BY_DATA_TYPE = {
        "bars": FIELD_MAP,
        "quotes": QUOTE_FIELD_MAP,
        "trades": TRADE_FIELD_MAP,
    }

    #: Shard column projection and order, per data type. Each begins with
    #: ``("timestamp", "symbol", "vendor")`` and then carries only that
    #: endpoint's own fields, so a directory scan of the vendor root always
    #: sees one schema and a trade never carries null bid/ask columns.
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

    #: Explicit dtypes per data type. They build an empty page (so a no-rows
    #: response still has a readable ``symbol`` column), type the columns a
    #: sparse response omits, and cast a populated page so every shard has the
    #: same schema.
    #:
    #: The ``timestamp`` unit is part of the schema. Bars are microseconds;
    #: quotes and trades are nanoseconds because that is the resolution Alpaca
    #: sends. Parsing them at microseconds would truncate sub-microsecond
    #: ordering and manufacture ``(timestamp, symbol)`` ties in the one tier
    #: that never deduplicates.
    RAW_SCHEMA_BY_DATA_TYPE = {
        "bars": {
            "timestamp": pl.Datetime("us"),
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
            "timestamp": pl.Datetime("ns"),  # vendor sends nanoseconds
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
            "timestamp": pl.Datetime("ns"),  # vendor sends nanoseconds
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

    #: Per data type, the columns the vendor may legitimately omit from every
    #: row of a page; ``conditions`` is optional on the wire. Any other
    #: missing column means the field map no longer matches the envelope, and
    #: ``_fetch_page`` raises rather than null-filling it: an all-null
    #: ``price`` column would read as untraded symbols forever.
    OPTIONAL_COLUMNS_BY_DATA_TYPE = {
        "bars": (),
        "quotes": ("conditions",),
        "trades": ("conditions",),
    }

    #: Accepted values for the request options a caller may set. An
    #: unrecognised value raises instead of being forwarded, because the
    #: vendor rejects some and silently ignores others, and "silently ignored"
    #: means a run that asks for split-adjusted bars and stores raw ones.
    #: Taken from ``alpaca-py`` 0.44.0.
    FEED_VALUES = frozenset(
        {"iex", "sip", "delayed_sip", "otc", "boats", "overnight"}
    )
    ADJUSTMENT_VALUES = frozenset({"raw", "split", "dividend", "all"})
    SORT_VALUES = frozenset({"asc", "desc"})

    #: Always ``asc``, never read from an option; see ``_fetch_page``.
    SORT = "asc"

    #: How an intraday window's edges are sent; see ``_window_bounds``.
    #:
    #: Alpaca documents ``start``/``end`` as RFC 3339 instants. A bare
    #: ``YYYY-MM-DD`` is unambiguous for daily bars but not for minute or tick
    #: data, where the most plausible reading of a bare ``end`` is
    #: ``00:00:00Z``, which would exclude the whole final session while the
    #: watermark still recorded that date as covered. Sending explicit bounds
    #: is what makes "queried up to ``end_date``" a true statement. Nine
    #: fractional digits match the nanosecond tick timestamps.
    INTRADAY_START_SUFFIX = "T00:00:00Z"
    INTRADAY_END_SUFFIX = "T23:59:59.999999999Z"

    #: Re-exposed from the module-level constant; see its comment.
    ASOF_NO_MAPPING = ASOF_NO_MAPPING

    #: What a credential value is replaced with in any captured message.
    REDACTION = "<APCA CREDENTIAL REDACTED>"

    #: The two credentials ``Acquisition._scrub`` redacts before a message
    #: reaches a log line or the failure manifest. Built from the module-level
    #: constants, never from the patchable client class. Alpaca sends
    #: credentials in headers rather than in the URL, but a ``requests``
    #: exception chain can still expose ``exc.request.headers``.
    CREDENTIAL_ENV_VARS = (KEY_ENV, SECRET_ENV)

    #: HTTP statuses that mean "slow down", not "out of allocation".
    #:
    #: Alpaca's 429 is a per-minute ceiling (200 requests per minute on the
    #: free plan) that a healthy full-market run hits repeatedly and that
    #: clears within a minute, so it is retried inside the worker. Alpaca has
    #: no request-allocation concept, so this class declares no
    #: ``QUOTA_STATUS_CODES``: reading this 429 as global would abort every
    #: run within seconds of starting.
    RATE_LIMIT_STATUS_CODES = frozenset({429})

    #: Headers Alpaca may send alongside a 429, read for logging only. A
    #: missing header is no information, never a default, and the backoff
    #: interval stays a configured constant rather than a computed reset time.
    RATE_LIMIT_HEADERS = (
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
    )

    def __init__(self, config: AcquisitionConfig):
        """Validate the request options, then open the transport.

        The options are checked before the client exists so that a bad
        ``data_type``, ``feed`` or ``adjustment`` raises at construction
        rather than inside a worker thread, where it would be filed in the
        failure manifest as if the vendor had rejected the batch.

        Raises
        ------
        ValueError
            If an option is outside its accepted set.
        RuntimeError
            If the credentials are missing from the environment.
        """
        super().__init__(config)
        self._assert_knobs_are_in_range()
        self._client = _AlpacaMarketDataClient()

    # -- data type: one token drives the endpoint and the projection --------

    @property
    def _data_type(self) -> str:
        """Return ``"bars"``, ``"quotes"`` or ``"trades"`` for this run.

        Bar frequencies resolve from ``config.frequency``; ``"tick"`` resolves
        from ``config.kwargs["data_type"]``, which has no default. The same
        token selects the endpoint and the shard projection, so the two
        cannot disagree.

        Raises
        ------
        ValueError
            If ``frequency`` is not a bar frequency and
            ``data_type`` is missing or not in ``TICK_DATA_TYPES``.
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
        """Return the shard column projection for this run's data type.

        A property rather than a class attribute because the projection
        depends on which endpoint the run reads. ``RAW_COLUMNS_BY_DATA_TYPE``
        remains the class-level source of truth.

        Examples
        --------
        >>> acq.RAW_COLUMNS
        ('timestamp', 'symbol', 'vendor', 'open', 'high', 'low', 'close',
         'volume', 'trade_count', 'vwap')
        """
        return self.RAW_COLUMNS_BY_DATA_TYPE[self._data_type]

    @property
    def RAW_SCHEMA(self) -> dict:  # noqa: N802 - matches RAW_COLUMNS
        """Return the explicit column dtypes for this run's data type.

        Examples
        --------
        >>> acq.RAW_SCHEMA["timestamp"]
        Datetime(time_unit='us', time_zone=None)
        """
        return self.RAW_SCHEMA_BY_DATA_TYPE[self._data_type]

    def _assert_knobs_are_in_range(self) -> None:
        """Reject out-of-range request options before any request is built.

        ``feed`` is checked only when set. An unset feed is omitted from the
        request rather than defaulted, so there is nothing to validate.
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

    def _window_bounds(self, start_date: str, end_date: str) -> tuple[str, str]:
        """Return ``(start, end)`` as this data type needs them on the wire.

        Daily bars keep bare dates. Minute and tick windows are widened to
        explicit RFC 3339 instants spanning the whole calendar day, because a
        bare ``YYYY-MM-DD`` end most plausibly resolves to ``00:00:00Z`` and
        would drop the entire final US session while the watermark still
        marked the date as covered. A bound that already carries a time
        (a ``T`` is present) is passed through untouched.
        """
        if self._data_type == "bars" and self.config.frequency == "1d":
            return start_date, end_date
        start = (
            start_date
            if "T" in str(start_date)
            else f"{start_date}{self.INTRADAY_START_SUFFIX}"
        )
        end = (
            end_date
            if "T" in str(end_date)
            else f"{end_date}{self.INTRADAY_END_SUFFIX}"
        )
        return start, end

    def _rate_limit_headers(self, exc: BaseException) -> dict[str, str]:
        """Return whichever ``X-RateLimit-*`` headers the vendor sent, or ``{}``.

        The base class calls this on the first backoff of a rate-limited batch
        and logs the result. An absent header contributes no entry, so "the
        vendor said nothing" stays distinct from "the vendor said zero".
        Nothing branches on the result; the backoff interval is a configured
        constant.
        """
        response = self._vendor_response(exc)
        headers = getattr(response, "headers", None) or {}
        return {
            name: str(headers[name])
            for name in self.RATE_LIMIT_HEADERS
            if name in headers
        }

    def _classify_error(self, exc: BaseException) -> str:
        """Return ``"rate_limited"`` for a 429 and ``"failed"`` for anything else.

        Alpaca has no global allocation condition, so this never returns
        ``"quota"``. ``TiingoAcquisition`` reads the same status the opposite
        way; see ``RATE_LIMIT_STATUS_CODES``.
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
        """Issue one request and return its rows as ``(frame, next_page_token)``.

        The frame is projected and ordered to ``RAW_COLUMNS`` for this run's
        data type, and a ``None`` token means this was the batch's last page.
        Every row in the envelope becomes exactly one row in the frame: there
        is no resampling, bucketing or deduplication on this path, and a
        "resample ticks to save space" edit would destroy the resolution the
        tick tier exists to capture.

        Parameters
        ----------
        symbols : list[str]
            The batch's symbols, joined into one ``symbols`` value.
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.
        page_token : str | None
            The token from the previous page, or ``None`` for the
            first page.

        Raises
        ------
        ValueError
            If a required column is missing from the envelope,
            which means the field map no longer matches the vendor.
        """
        symbols = self._validate_symbols(symbols)
        data_type = self._data_type

        # Bare dates for daily bars; explicit instants covering the whole
        # session for minute and tick data. See `_window_bounds`.
        window_start, window_end = self._window_bounds(start_date, end_date)

        params = {
            "symbols": ",".join(symbols),
            "start": window_start,
            "end": window_end,
            # The vendor maximum. Fewer rows per page means more requests for
            # the same data, which is pure rate-limit pressure.
            "limit": self._knob("page_limit", 10_000),
            # Always ascending. Alpaca orders symbol-major then by timestamp,
            # so the ledger's "furthest position reached" is a well-defined
            # resume point; a descending request would make it meaningless.
            "sort": self.SORT,
        }

        # `asof` is sent as a real value, never `None`: `requests` drops
        # None-valued params, and an omitted `asof` makes the vendor map each
        # symbol onto whatever entity holds the ticker today, so a delisted
        # ticker would silently return the current occupant's history. An
        # explicit `kwargs={"asof": None}` is the one way to ask for that
        # current-day mapping, and the only path that omits the key.
        asof = self._knob("asof", _ASOF_UNSET)
        if asof is _ASOF_UNSET:
            params["asof"] = self.ASOF_NO_MAPPING
        elif asof is not None:
            params["asof"] = asof

        if data_type == "bars":
            # Bar-only parameters: the quotes and trades endpoints have no bar
            # size or price adjustment, and sending either is at best ignored.
            params["timeframe"] = self.TIMEFRAME_MAP[self.config.frequency]
            params["adjustment"] = self._knob("adjustment", "raw")

        # Omitted entirely when unset: there is no in-code feed default, and
        # leaving the key out lets the vendor pick what the subscription
        # allows. See the class docstring.
        feed = self._knob("feed", None)
        if feed:
            params["feed"] = feed

        if page_token:
            params["page_token"] = page_token

        payload = self._client.get_page(self.ENDPOINT_MAP[data_type], params)

        # The envelope keys its rows by the data type itself, so the resolved
        # token indexes the endpoint, the response and the projection alike.
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

        # Infer the schema over the whole page, never the default 100 rows.
        # The vendor omits an absent field from a row rather than nulling it,
        # so a field first seen at row 101 would otherwise be dropped from the
        # frame with no error: an optional column would lose the data later
        # rows carried, and a required one would raise on every run.
        frame = pl.DataFrame(rows, infer_schema_length=None)

        # A column absent from every row of the page is filled as a typed
        # null column, so the shard schema is identical either way, but only
        # when it is declared optional. Anything else absent means the field
        # map no longer matches the envelope and must raise: filling it would
        # write an all-null `price` (or `close`) column that reads as data.
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

        # Alpaca returns RFC 3339 with a trailing `Z`. Parse as UTC and drop
        # the zone so the dtype is a naive `pl.Datetime`, matching every other
        # timestamp in the codebase; the intraday `date=` hive key is derived
        # from these naive-UTC values in `SESSION_TIME_ZONE` by the base class.
        # The time unit comes from the schema: quotes and trades are
        # nanoseconds, and parsing at polars' microsecond default would
        # silently truncate them.
        frame = frame.with_columns(
            pl.col("timestamp")
            .str.to_datetime(time_zone="UTC", time_unit=schema["timestamp"].time_unit)
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


#: The registry entry for this vendor, defined beside the class so that adding
#: a vendor touches one file. ``quantlab.registry`` imports this module at its
#: bottom, after every definition, so a cold ``import quantlab.registry`` still
#: enumerates this source. Binance spot data has no descriptor: that path
#: reads locally dropped CSV files and downloads nothing.
ALPACA_SOURCE = register_source(
    SourceDescriptor(
        vendor="alpaca",
        display_name="Alpaca Market Data",
        acquisition_cls=AlpacaAcquisition,
        config_factory=functools.partial(stock_acquisition_config, vendor="alpaca"),
        #: Four capabilities, which a cross-product of markets and frequencies
        #: could not express: ``tick`` carries a data type the bar
        #: frequencies do not, and it splits into two endpoints. The two bar
        #: rows convert through ``StockDataset``, shared with Tiingo's row by
        #: direct class reference. The two tick rows carry no ``dataset_cls``,
        #: so ``registry.convert()`` refuses them because no conversion target
        #: exists, not because it recognises the token: a quotes or trades
        #: stream flattened onto a dense ``[timestamp, symbol]`` grid would be
        #: a plausible-looking panel that is wrong.
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="bars",
                dataset_cls=StockDataset,
            ),
            Capability(
                market="us_equity",
                frequency="1m",
                data_type="bars",
                dataset_cls=StockDataset,
            ),
            Capability(market="us_equity", frequency="tick", data_type="quotes"),
            Capability(market="us_equity", frequency="tick", data_type="trades"),
        ),
        #: Restated as literals rather than derived from ``CREDENTIAL_ENV_VARS``
        #: so the two declarations stay independently checkable.
        required_env=("APCA_API_KEY_ID", "APCA_API_SECRET_KEY"),
        #: Advisory only; the roster comes from ``UniverseCatalog``, never from
        #: the vendor.
        universe_categories=(
            "nasdaq_all",
            "us_all",
            "sp500_constituent",
            "nasdaq100_constituent",
        ),
    )
)
