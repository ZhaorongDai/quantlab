"""The Massive REST endpoints ``MassiveClient.pull`` keeps in the raw tier.

An endpoint is one config object of its pull shape
(``quantlab.acquisition.massive.shapes``): a ``DatedEndpoint``, a
``WindowedEndpoint``, a ``SnapshotEndpoint``, an ``AsOfEndpoint`` or an
``IdsEndpoint``, with its name (the raw tier's directory,
``quantlab.dataset.massive.raw.rest_dir``), its URL and the fields of that
shape. A new endpoint is one more entry of ``ENDPOINTS`` (docs/user-guide/massive.md,
"REST pulls"). ``CONDITIONS`` stands apart: a day conversion needs it, so it is
pulled beside the day files (``MassiveClient.condition_table``), not by a refresh.

Examples
--------
>>> ENDPOINTS["splits"].url
'https://api.massive.com/stocks/v1/splits'
>>> ENDPOINTS["dividends"].date_field
'ex_dividend_date'
"""

from __future__ import annotations

from datetime import date

from quantlab.acquisition.massive.shapes import (
    AsOfEndpoint,
    ConditionTableEndpoint,
    DatedEndpoint,
    DatedSort,
    IdsEndpoint,
    RestEndpoint,
    SnapshotEndpoint,
    WindowedEndpoint,
)

#: The host of every REST endpoint.
REST_HOST = "https://api.massive.com"

#: The stock condition table.
CONDITIONS = ConditionTableEndpoint(
    "conditions", f"{REST_HOST}/v3/reference/conditions", params={"asset_class": "stocks"}, limit=1000
)

# Every stock ticker listed as of each session (date=), from the first day of the Ticker
# map's Day aggregates; the inputs of a later FIGI version of the Ticker map.
_TICKERS = AsOfEndpoint(
    "tickers", f"{REST_HOST}/v3/reference/tickers", "date", date(2016, 10, 10),
    params={"market": "stocks", "sort": "ticker", "order": "asc"}, limit=1000,
)

#: Every endpoint the raw tier keeps, by name.
ENDPOINTS: dict[str, RestEndpoint] = {
    endpoint.name: endpoint
    for endpoint in (
        # Splits (and stock dividends) on their execution date, from 1978; the current
        # endpoint, not the deprecated /v3/reference/splits.
        DatedEndpoint(
            "splits", f"{REST_HOST}/stocks/v1/splits", "execution_date", date(1978, 1, 1), overlap_days=7
        ),
        # Cash dividends on their ex-date, from 2000-08; a dividend is declared before its
        # ex-date, so a pull past the watermark (always before today) reaches every new one.
        DatedEndpoint(
            "dividends", f"{REST_HOST}/stocks/v1/dividends", "ex_dividend_date", date(2000, 1, 1), overlap_days=7
        ),
        # Every news article (#357) on its publication time, from 2016-06. The endpoint sorts
        # with sort=<field> and order=asc; a day is read again for articles indexed late.
        DatedEndpoint(
            "news", f"{REST_HOST}/v2/reference/news", "published_utc", date(2016, 6, 1),
            limit=1000, overlap_days=1, sort=DatedSort.ORDER_PARAM, timestamped=True,
        ),
        # FINRA short interest per settlement date, twice a month from 2017-12 (#355). A
        # release is published about 8 trading days after its settlement date, when the
        # watermark has passed it, so each pull asks again for the last 21 days. The
        # records have no id: a record is all its fields, the panel keeps the newest per
        # ticker and settlement date.
        DatedEndpoint(
            "short_interest", f"{REST_HOST}/stocks/v1/short-interest", "settlement_date", date(2017, 12, 1),
            limit=50000, overlap_days=21,
        ),
        # Raw-only tables (#360): filing text and second sources, on the filing date. Their
        # rows have no id of their own (a 13-F has one row per holding, a Form 4 one per
        # line), so a record is identified by all its fields; the sort is on the filing date
        # alone. The 10-K sections refuse that sort (only period_end), so they are pulled a
        # month of filing dates at a time, unsorted.
        WindowedEndpoint("10k_sections", f"{REST_HOST}/stocks/filings/10-K/vX/sections", "filing_date",
                         date(1993, 1, 1), limit=100, overlap_days=7, window_days=31),
        *(
            DatedEndpoint(name, f"{REST_HOST}{path}", "filing_date", date(1993, 1, 1),
                          limit=limit, overlap_days=7, sort=DatedSort.DATE)
            for name, path, limit in (
                ("8k_text", "/stocks/filings/8-K/vX/text", 100),
                ("risk_factors", "/stocks/filings/vX/risk-factors", 50000),
                ("form_3", "/stocks/filings/vX/form-3", 10000),
                ("form_4", "/stocks/filings/vX/form-4", 10000),
                ("13f", "/stocks/filings/vX/13-F", 1000),
            )
        ),
        # The Float table holds only each ticker's latest free float, so a snapshot of the
        # whole table is kept every day and the history accumulates from the first pull.
        SnapshotEndpoint("float", f"{REST_HOST}/stocks/vX/float"),
        _TICKERS,
        # IPOs on their listing date, from 2008-12 (#359). Every status is kept raw
        # (withdrawn, pending); the IPO panel reads the listed ones. The records have no
        # id and a revision (pending -> new -> history) changes their fields, so a record
        # is identified by all of them and the panel keeps each IPO's newest copy.
        DatedEndpoint(
            "ipos", f"{REST_HOST}/vX/reference/ipos", "listing_date", date(2008, 12, 1),
            limit=1000, overlap_days=7, sort=DatedSort.ORDER_PARAM,
        ),
        # Each composite FIGI's ticker changes (experimental vX endpoint, one FIGI a call),
        # asked for every FIGI of the reference tickers once and again when its ticker changes.
        IdsEndpoint(
            "ticker_events", f"{REST_HOST}/vX/reference/tickers/{{id}}/events", _TICKERS, limit=1000,
            key="composite_figi",
        ),
        # Daily short volume per ticker (FINRA's files, totals and per venue), from 2024-02-06
        # (#356). A record has no id (one per ticker and day), so it is identified by all its
        # fields and the short volume panel keeps the newest copy of each (ticker, date).
        DatedEndpoint(
            "short_volume", f"{REST_HOST}/stocks/v1/short-volume", "date", date(2024, 2, 6),
            limit=50000, overlap_days=7,
        ),
        # 8-K disclosures (#358), one record per categorised disclosure of a filing, from
        # 2020-01; no id field (a filing has several), sorted on the filing date alone.
        DatedEndpoint(
            "8k_disclosures", f"{REST_HOST}/stocks/filings/8-K/vX/disclosures", "filing_date",
            date(2020, 1, 1), limit=1000, overlap_days=7, sort=DatedSort.DATE,
        ),
        # The disclosure taxonomy (#358): primary, secondary and tertiary categories with a
        # description; a small table kept as one snapshot a day.
        SnapshotEndpoint("disclosure_taxonomy", f"{REST_HOST}/stocks/taxonomies/vX/disclosures", limit=1000),
    )
}
