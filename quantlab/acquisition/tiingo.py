"""Tiingo end-of-day acquisition for US equities.

``TiingoAcquisition`` is the vendor-specific half of the acquisition layer for
Tiingo's daily price endpoint. It fetches one symbol's whole date range per
request and hands the rows to the shared ``Acquisition`` base, which owns
resume, concurrency, failure isolation and the raw parquet shards. The API key
is read from ``TIINGO_API_KEY`` in the environment and is never stored on a
config or written to a log. ``TIINGO_SOURCE`` at the bottom of the module
registers the vendor with ``quantlab.registry``.
"""

import functools
import os

import polars as pl
from tiingo import TiingoClient

from quantlab.registry import (
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.base.acquisition import Acquisition
from quantlab.base.config import AcquisitionConfig
from quantlab.config import stock_acquisition_config
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns

#: The environment variable the Tiingo API key is read from.
#:
#: Defined at module level, and read from here by the credential-scrubbing
#: routine, rather than off ``TiingoClient``. That class is a patch target
#: that tests replace with a fake, and a redaction routine must not depend on
#: a symbol whose whole purpose is to be replaced.
KEY_ENV = "TIINGO_API_KEY"

# Project frequency token to Tiingo's ``frequency`` parameter. Add intraday
# entries here rather than spelling a frequency inline in ``_fetch_one``.
_FREQUENCY_MAP = {"1d": "daily"}

#: Tiingo's EOD field names, in the order ``TiingoColumns.EOD`` requests them.
#: ``RAW_COLUMNS`` below is derived from this tuple so the projection cannot
#: drift from what is actually asked for.
_TIINGO_EOD_COLUMNS = tuple(TiingoColumns.EOD.split(","))


class TiingoAcquisition(Acquisition):
    """Tiingo end-of-day acquisition behind the shared ``Acquisition`` base.

    Tiingo's price endpoint takes one symbol per call and returns the whole
    requested date range in one response, so this vendor runs at
    ``DEFAULT_BATCH_SIZE = 1`` and ``_fetch_page`` never returns a page token.
    Both are the complete contract for this vendor, not placeholders.

    Error classification is the one thing this class owns beyond building the
    request. A 429 from Tiingo means the account's hourly request allocation
    is spent, which is a global condition: it is classified as ``"quota"`` and
    the base class stops dispatching work for the whole run instead of letting
    every remaining symbol fail in turn. ``AlpacaAcquisition`` reads the same
    status as a per-minute rate limit to back off from, which is why the
    status set lives on the vendor class rather than on the base.

    Run options such as ``resume``, ``max_workers``, ``progress``,
    ``batch_size``, ``legacy_watermarks``, ``wait_for_quota``,
    ``quota_wait_seconds`` and ``quota_max_waits`` are read from
    ``config.kwargs``, so every run is reproducible from its config file.

    Credentials. ``TIINGO_API_KEY`` is read from the environment in
    ``__init__`` and passed only to the in-memory ``TiingoClient``. It is
    never assigned to the config or to any attribute a serialiser could reach,
    and its value is redacted from every captured message.

    Examples
    --------
    Needs ``TIINGO_API_KEY`` exported; ``download()`` reaches the network.

    >>> from quantlab.base.config import AcquisitionConfig
    >>> cfg = AcquisitionConfig(
    ...     market="us_equity", frequency="1d", vendor="tiingo",
    ...     raw_data_dir_path="downloads/nasdaq_data/tiingo",
    ...     watermark_path="downloads/nasdaq_data/_watermarks/tiingo",
    ...     symbols=("AAPL", "MSFT"), start_date="2024-01-02",
    ...     end_date="2024-01-03",
    ... )
    >>> acq = TiingoAcquisition(cfg).download()
    >>> acq.coverage_report()
    {'requested': 2, 'pending': 0, 'skipped': 2, 'covered': 2,
     'widened': 0, 'legacy': 0, 'no_data': 0}

    Each symbol lands as one parquet shard under ``month=YYYY-MM/``
    beneath ``raw_data_dir_path``, projected to ``RAW_COLUMNS``.
    """

    VENDOR = "tiingo"

    #: The one credential ``Acquisition._scrub`` redacts before a message
    #: reaches a log line or the failure manifest. Sourced from the
    #: module-level constant, never from the patchable client class.
    CREDENTIAL_ENV_VARS = (KEY_ENV,)

    #: What the API key is replaced with in any captured message.
    REDACTION = "<TIINGO_API_KEY REDACTED>"

    #: HTTP statuses that mean the account's request allocation is spent, a
    #: global condition. Only 429: Tiingo also returns 403 for a single
    #: plan-restricted ticker, which is a per-symbol condition, and treating
    #: it as global would let one restricted ticker abort a full-market run.
    #: A 403 whose body carries the allocation wording is still caught by
    #: ``QUOTA_MESSAGE_TOKEN``.
    QUOTA_STATUS_CODES = frozenset({429})

    #: The durable fragment of the vendor's allocation message ("You have run
    #: over your hourly request allocation. Contact us at ..."). Matching the
    #: whole sentence would break the moment the vendor edits it. This is
    #: still text matching, and that brittleness is confined to this constant.
    QUOTA_MESSAGE_TOKEN = "request allocation"

    #: One symbol per request, because the endpoint is single-symbol. A larger
    #: value would loop inside ``_fetch_page`` while presenting the batch as
    #: atomic: one failure would fail N symbols, the quota check would run
    #: once per N requests, and resume would coarsen from one symbol to N.
    DEFAULT_BATCH_SIZE = 1

    #: Shard column projection and order: the identity columns first, then
    #: Tiingo's EOD fields in the order they are requested. ``vendor`` keeps
    #: provenance visible even after a cross-vendor merge.
    RAW_COLUMNS = ("timestamp", "symbol", "vendor", *_TIINGO_EOD_COLUMNS)

    #: Explicit dtype of every shard column. ``pl.DataFrame`` infers per
    #: response, so a symbol whose ``divCash`` is all integer zero or whose
    #: ``volume`` is all null would otherwise land a shard typed differently
    #: from its siblings and make the whole directory scan fail with a
    #: ``SchemaError`` that names a file rather than a cause.
    RAW_SCHEMA = {
        "timestamp": pl.Datetime("us"),
        "symbol": pl.String,
        "vendor": pl.String,
        **{name: pl.Float64 for name in _TIINGO_EOD_COLUMNS},
    }

    def __init__(self, config: AcquisitionConfig):
        """Open the Tiingo client with the API key from the environment.

        Raises
        ------
        RuntimeError
            If ``TIINGO_API_KEY`` is unset or empty.
        """
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

    def _classify_error(self, exc: BaseException) -> str:
        """Return ``"quota"`` for a spent allocation and ``"failed"`` otherwise.

        This vendor declares no ``RATE_LIMIT_STATUS_CODES``, so
        ``"rate_limited"`` is never returned: backing off inside a worker
        against an exhausted hourly allocation would burn thousands of
        fast-failing requests before anything noticed.
        """
        return "quota" if self._is_quota_error(exc) else "failed"

    def _is_quota_error(self, exc: BaseException) -> bool:
        """Return whether ``exc`` means the account's request allocation is spent.

        Two independent signals, either sufficient: the response status is in
        ``QUOTA_STATUS_CODES``, or ``QUOTA_MESSAGE_TOKEN`` appears
        case-insensitively in the response body or the rendered exception.
        The status alone would miss a vendor that stops sending 429; the text
        alone would miss a 429 with an empty body. The text is scrubbed of
        credentials before it is inspected, because it later travels into log
        lines.
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
        """Return one symbol's rows for the whole window, or ``None`` if empty.

        Timestamps are parsed as UTC and stored naive, and literal ``symbol``
        and ``vendor`` columns are added, so the frame comes back already in
        ``RAW_COLUMNS`` order and ``RAW_SCHEMA`` dtypes.
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
        # Infer the schema over the whole response, never the default 100
        # rows. Tiingo backfills `low`/`adjLow` as null for a leading run of
        # some symbols' early history (hundreds of sessions), so a 100-row
        # window would type the column `Null` and the first real value would
        # then fail the frame build itself, before the cast below could
        # repair it. A column null in every row still infers `Null`, and the
        # cast turns it into `Float64`; the two measures are complementary.
        data = pl.DataFrame(response, infer_schema_length=None)
        if data.is_empty():
            return None

        # Tiingo's `date` is ISO-8601 with a trailing `Z`. Parse as UTC and
        # drop the zone so the dtype is a naive `pl.Datetime`, matching every
        # other timestamp in the codebase; parsing without an explicit zone
        # raises on zone-aware strings.
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
        # Cast as well as project: the projection pins names and order, the
        # cast pins dtypes so every shard has the same schema. `timestamp`
        # was just parsed above and is excluded.
        return data.select(self.RAW_COLUMNS).cast(
            {
                name: dtype
                for name, dtype in self.RAW_SCHEMA.items()
                if name != "timestamp"
            }
        )

    def _empty_frame(self) -> pl.DataFrame:
        """Return an empty frame carrying ``RAW_COLUMNS`` and ``RAW_SCHEMA``.

        Returned when a batch produced no rows. It must carry the schema
        because ``_fetch_batch`` reads ``symbol`` off it; a schemaless empty
        frame would raise there instead of reporting "no rows".
        """
        # Built from `RAW_SCHEMA` so an empty page and a populated one land
        # the same schema.
        return pl.DataFrame(schema=dict(self.RAW_SCHEMA)).select(self.RAW_COLUMNS)

    def _fetch_page(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        page_token: str | None = None,
    ) -> tuple[pl.DataFrame, str | None]:
        """Return ``(frame, None)``: for Tiingo one page is the whole range.

        The EOD endpoint returns the full requested range in one response and
        has no pagination cursor, so there is never a next page and
        ``page_token`` is accepted only to satisfy the base contract.
        ``symbols`` is looped because the endpoint is single-symbol; at
        ``DEFAULT_BATCH_SIZE = 1`` the loop has one iteration.

        Parameters
        ----------
        symbols : list[str]
            The batch's symbols.
        start_date : str
            First date of the window, inclusive.
        end_date : str
            Last date of the window, inclusive.
        page_token : str | None
            Ignored; this vendor never issues one.
        """
        frames = []
        for symbol in symbols:
            frame = self._fetch_one(symbol, start_date, end_date)
            if frame is not None and not frame.is_empty():
                frames.append(frame)
        if not frames:
            return self._empty_frame(), None
        return pl.concat(frames, how="vertical"), None


#: The registry entry for this vendor, defined beside the class so that adding
#: a vendor touches one file. ``quantlab.registry`` imports this module at its
#: bottom, after every definition, so a cold ``import quantlab.registry`` still
#: enumerates this source.
TIINGO_SOURCE = register_source(
    SourceDescriptor(
        vendor="tiingo",
        display_name="Tiingo EOD",
        acquisition_cls=TiingoAcquisition,
        #: One factory serves both vendors, differing by this keyword.
        config_factory=functools.partial(stock_acquisition_config, vendor="tiingo"),
        #: Exactly one capability: daily US equities. ``data_type=None``
        #: records that this vendor has no bars/quotes/trades distinction,
        #: not a wildcard, so ``supports("us_equity", "tick")`` is false.
        #: ``dataset_cls`` is what makes ``registry.convert()`` reachable for
        #: this capability; the same class converts Alpaca's bar rows.
        capabilities=(
            Capability(
                market="us_equity",
                frequency="1d",
                data_type=None,
                dataset_cls=StockDataset,
            ),
        ),
        #: Restated as a literal rather than derived from
        #: ``CREDENTIAL_ENV_VARS`` so the two declarations stay independently
        #: checkable.
        required_env=("TIINGO_API_KEY",),
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
