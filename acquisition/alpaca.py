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
    #: than an inline literal inside `_fetch_page`, so adding `1m` in 03.2-06 is
    #: a one-line data change rather than an edit to request-building logic.
    TIMEFRAME_MAP = {"1d": "1Day"}

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

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)
        self._client = _AlpacaMarketDataClient()

    def _scrub(self, message: str) -> str:
        """Redact both Alpaca credential values before `message` is logged or
        written to the failure manifest (T-03.2-01).

        Alpaca sends credentials in HEADERS rather than in the URL, so the
        Tiingo `?token=` leak shape does not apply here directly. Scrub anyway:
        a `requests` exception chain can reach `exc.request.headers`, and a
        message travelling into a log aggregator or a committed manifest is not
        a place to be relying on a vendor's choice of auth transport staying
        the same.

        This is the single choke point every captured vendor message passes
        through. The base-class `_scrub` seam is hoisted in 03.2-03; the method
        exists on this class from the first commit that can produce an Alpaca
        exception, so the control is never absent while the risk is present.
        """
        for name in (KEY_ENV, SECRET_ENV):
            value = os.environ.get(name)
            if value:
                message = message.replace(value, self.REDACTION)
        return message

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
