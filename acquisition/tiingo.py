import os

import polars as pl
from tiingo import TiingoClient

from base.acquisition import Acquisition
from base.config import AcquisitionConfig
from enums.data import TiingoColumns

#: The environment variable the Tiingo API key is read from.
#:
#: MODULE-LEVEL on purpose, mirroring `acquisition/alpaca.py`: the shared
#: `Acquisition._scrub` reaches this name through `CREDENTIAL_ENV_VARS`, and a
#: security control must not be reachable through an indirection whose whole
#: purpose is to be replaced by a test double (03.2-02 deviation #2).
KEY_ENV = "TIINGO_API_KEY"

# Extend when intraday frequencies are added -- never hardcode "daily" inline
# in _fetch_one.
_FREQUENCY_MAP = {"1d": "daily"}

#: Tiingo's EOD field names, in the order `TiingoColumns.EOD` requests them.
#: Named here so `RAW_COLUMNS` below is derived from one list rather than being
#: a second hand-maintained copy that could drift from what is actually asked
#: for.
_TIINGO_EOD_COLUMNS = tuple(TiingoColumns.EOD.split(","))


class TiingoAcquisition(Acquisition):
    """Config-driven, incrementally-refreshable Tiingo EOD data acquisition.

    `TIINGO_API_KEY` is read directly from `os.environ` in `__init__` and
    passed only into the in-memory `TiingoClient` constructor argument --
    never assigned to `self.config` or any other dataclass-facing attribute.

    Tiingo's EOD endpoint takes ONE symbol per call and returns a whole date
    range in one response, so this class is the batched primitive's DEGENERATE
    case: `DEFAULT_BATCH_SIZE = 1` and `_fetch_page` never returns a token.
    Both are complete implementations of the base contract, not placeholders.

    **Orchestration lives on `Acquisition`, classification lives here** (D-02,
    03.2-03). The concurrent fan-out, the resume/skip partition, the failure
    manifest and the global abort event are the base class's; what this class
    still owns is the reading of a Tiingo exception -- `QUOTA_STATUS_CODES`,
    `QUOTA_MESSAGE_TOKEN` and `_is_quota_error`. That split is not stylistic:
    Tiingo's 429 means the hourly ALLOCATION is gone and the correct response
    is to stop the world for an hour, while Alpaca returns the same status for
    a per-minute ceiling a healthy run is expected to hit. Hoisting the status
    set would abort every Alpaca run within seconds while logging an
    allocation message for a vendor that has no allocation concept
    (03.2-RESEARCH.md Pitfall 1).
    """

    VENDOR = "tiingo"

    #: The one credential this vendor reads, named for the shared
    #: `Acquisition._scrub` choke point. Sourced from the module-level
    #: constant, never from the (substitutable) client class.
    CREDENTIAL_ENV_VARS = (KEY_ENV,)

    #: What the API key is replaced with in any captured message.
    REDACTION = "<TIINGO_API_KEY REDACTED>"

    #: HTTP statuses that mean the account's request allocation is gone, i.e.
    #: a GLOBAL condition (D-05). 429 ONLY, deliberately: Tiingo also returns
    #: 403 for a plan-restricted single ticker, which is a PER-SYMBOL
    #: condition, and treating that as global would let one restricted ticker
    #: abort a 15,000-symbol run. A 403 whose body carries the allocation
    #: wording is still caught by the textual signal below, so the stricter
    #: status set costs nothing.
    #:
    #: Deliberately NOT a base-class attribute -- see the class docstring.
    QUOTA_STATUS_CODES = frozenset({429})

    #: The durable fragment of the observed body: "Error: You have run over
    #: your hourly request allocation. Contact us at support@tiingo.com to
    #: have these lifted." Matching the full sentence would break the moment
    #: the vendor says "daily" instead of "hourly" or edits its support
    #: address. This is still text matching and it is still brittle -- that
    #: brittleness is a conscious choice, confined to this one constant.
    QUOTA_MESSAGE_TOKEN = "request allocation"

    #: One symbol per request, and that number is LOAD-BEARING rather than a
    #: conservative default.
    #:
    #: Tiingo's price endpoint is single-symbol, so any other value would mean
    #: looping inside `_fetch_page` while presenting the batch as atomic. Three
    #: concrete consequences follow from that pretence:
    #:
    #: - ONE failure would fail N symbols instead of one, because the batch is
    #:   the unit of success;
    #: - the quota abort check runs once per batch, so an exhausted allocation
    #:   would let N-1 further requests through before it was noticed -- the
    #:   exact fast-failing burn 260906-26o D-05 exists to stop;
    #: - resume granularity would coarsen from one symbol to N.
    #:
    #: At 1 the batched path is behaviourally identical to the per-symbol path
    #: it replaces, which is what makes this a real implementation of
    #: `_fetch_page` rather than a multi-symbol interface being faked.
    DEFAULT_BATCH_SIZE = 1

    #: The pinned shard projection and order: the identity columns first, then
    #: Tiingo's EOD fields in the order `TiingoColumns.EOD` requests them --
    #: which is also the order `tests/conftest.py:_STOCK_PQT_COLUMNS` records.
    #: `vendor` is new in 03.2 and is what keeps a cross-vendor merge
    #: DETECTABLE as well as prevented (D-11).
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *_TIINGO_EOD_COLUMNS)

    def __init__(self, config: AcquisitionConfig):
        super().__init__(config)

        if not os.environ.get(KEY_ENV):
            raise RuntimeError(
                f"{KEY_ENV} environment variable is not set. Export it "
                f"before running acquisition (see Tiingo dashboard for your "
                f"key)."
            )
        self._client = TiingoClient(
            {"session": True, "api_key": os.environ[KEY_ENV]}
        )

    @staticmethod
    def _vendor_response(exc: BaseException):
        """The vendor `requests.Response` reachable from `exc`, or None.

        Measured, not assumed: `tiingo/restclient.py:_request` catches the
        `requests.exceptions.HTTPError` and re-raises `RestClientError(e)`, so
        `RestClientError` has NO `.response` of its own -- the obvious
        `getattr(exc, "response", None)` one-liner returns None every time.
        The status lives at `exc.args[0].response.status_code`. Hence the walk
        over the exception AND its args.
        """
        for candidate in (exc, *getattr(exc, "args", ())):
            response = getattr(candidate, "response", None)
            if response is not None and getattr(response, "status_code", None):
                return response
        return None

    def _is_quota_error(self, exc: BaseException) -> bool:
        """Whether `exc` means the account's request allocation is exhausted
        -- a GLOBAL, recoverable condition rather than one ticker's fault.

        Two independent signals, either sufficient:

        - **structured:** the reachable status is in `QUOTA_STATUS_CODES`;
        - **textual:** `QUOTA_MESSAGE_TOKEN` appears case-insensitively in the
          reachable response body or in the rendered exception.

        Both are needed. The status alone would miss a vendor that stops
        setting 429; the text alone would miss a 429 with an empty body. Any
        vendor text read here passes through `_scrub` first, because this is
        the text that then travels into log lines (T-26o-01).
        """
        response = self._vendor_response(exc)
        if response is not None and response.status_code in self.QUOTA_STATUS_CODES:
            return True

        body = ""
        if response is not None:
            try:
                body = response.text or ""
            except Exception:  # noqa: BLE001 -- a decode failure is not a quota signal
                body = ""
        haystack = self._scrub(f"{body}\n{exc}").lower()
        return self.QUOTA_MESSAGE_TOKEN in haystack

    def _fetch_one(
        self, symbol: str, start_date: str, end_date: str
    ) -> pl.DataFrame | None:
        """One symbol's whole range in one vendor call, or None if empty.

        The pre-03.2 `_fetch_and_write` body verbatim MINUS the write: the REST
        call, the timestamp normalisation with its reasoning, and the literal
        `symbol` provenance column, plus the new literal `vendor` column that
        makes provenance survive even a merged read.
        """
        frequency = _FREQUENCY_MAP[self.config.frequency]
        response = self._client.get_ticker_price(
            symbol,
            fmt="json",
            startDate=start_date,
            endDate=end_date,
            frequency=frequency,
            columns=TiingoColumns.EOD,
        )
        data = pl.DataFrame(response)
        if data.is_empty():
            return None

        # Tiingo's `date` field is an ISO-8601 string with a trailing `Z`
        # (UTC) offset (e.g. "2024-01-02T00:00:00.000Z"). Parse it as UTC
        # then drop the tz so the resulting dtype is a naive `pl.Datetime`,
        # matching the naive timestamps produced elsewhere in the codebase
        # (e.g. `StockDataset`'s naive `str.to_datetime()` filter bounds) --
        # parsing without an explicit time zone raises on tz-aware strings.
        data = data.with_columns(
            pl.col("date")
            .str.to_datetime(time_zone="UTC")
            .dt.replace_time_zone(None)
        )
        data = data.rename({"date": "timestamp"})
        data = data.with_columns(
            pl.lit(symbol).alias("symbol"),
            pl.lit(self.VENDOR).alias("vendor"),
        )
        return data.select(self.RAW_COLUMNS)

    def _empty_frame(self) -> pl.DataFrame:
        """An empty frame carrying the RAW_COLUMNS schema.

        Returned when a batch produced no rows at all. It must carry the schema
        rather than being a bare `pl.DataFrame()`, because `_fetch_batch` reads
        `symbol` off it and a schemaless empty frame would raise there instead
        of reporting "no rows".
        """
        return pl.DataFrame(
            schema={
                "timestamp": pl.Datetime,
                "symbol": pl.String,
                "vendor": pl.String,
                **{name: pl.Float64 for name in _TIINGO_EOD_COLUMNS},
            }
        ).select(self.RAW_COLUMNS)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """One page for `symbols` -- which for Tiingo is always the WHOLE range.

        Returns `(frame, None)` unconditionally: the EOD endpoint returns the
        full requested range in one response and publishes no pagination
        cursor, so there is never a next page. `page_token` is accepted to
        satisfy the base contract and is ignored, because a token this vendor
        never issues can never be handed back.

        `symbols` is looped rather than joined because the endpoint is
        single-symbol; at `DEFAULT_BATCH_SIZE = 1` that loop has one iteration.
        """
        frames = []
        for symbol in symbols:
            frame = self._fetch_one(symbol, start_date, end_date)
            if frame is not None and not frame.is_empty():
                frames.append(frame)
        if not frames:
            return self._empty_frame(), None
        return pl.concat(frames, how="vertical"), None


class ConcurrentTiingoAcquisition(TiingoAcquisition):
    """TRANSITIONAL: `TiingoAcquisition` under its pre-03.2-03 name.

    Everything this class used to own -- the threaded fan-out, the
    resume/skip partition, the per-unit failure isolation, the failure
    manifest, the global abort event and the bounded wait/resume loop -- was
    hoisted onto `Acquisition` by 03.2-03, so that ONE implementation drives
    every vendor rather than each vendor carrying a copy that could drift
    (D-02). What stayed vendor-specific is the CLASSIFICATION of a Tiingo
    exception, and that lives on `TiingoAcquisition` above.

    Nothing of substance is left here. The name survives for exactly one
    commit so every existing import keeps resolving while the lift is
    verified; it is deleted, along with every reference to it, in the next
    commit of this plan (03.1 D-03 precedent: the retired name is retired,
    never kept as a permanent alias).
    """
