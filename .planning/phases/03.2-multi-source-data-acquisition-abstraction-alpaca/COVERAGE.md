# API Coverage — Alpaca Market Data

> Full coverage by default. Opt-outs are explicit, reasoned decisions.
>
> Detector: `api-coverage.cjs --json` over the phase scope (CONTEXT + RESEARCH + ROADMAP entry)
> returned `detected: true` (signals: `api`, `sdk`). Matrix produced at plan time so
> `api-coverage.verify-pre` does not block the seal.
>
> **Re-decided from a full-coverage baseline, not carried over from Tiingo.** Phase 2's Tiingo
> integration opted out of most of this surface; every row below was decided again for Alpaca so a
> first-class/fallback asymmetry cannot accumulate silently (D-10 keeps the two vendors parallel
> alternatives, not primary/fallback).

## Endpoint capabilities

| capability | decision | reason |
|---|---|---|
| `stocks/bars` (historical, daily `1Day`) | INTEGRATE | |
| `stocks/bars` (historical, minute `1Min`) | INTEGRATE | |
| `stocks/quotes` (historical L1) | INTEGRATE | |
| `stocks/trades` (historical) | INTEGRATE | |
| `stocks/bars/latest` | OPT-OUT | Real-time/latest snapshot surface; D-16 fences this phase to historical raw acquisition and D-15 excludes any live/streaming path |
| `stocks/quotes/latest` | OPT-OUT | Same as `stocks/bars/latest` — latest-value endpoints serve a live path this phase does not build |
| `stocks/trades/latest` | OPT-OUT | Same as `stocks/bars/latest` |
| `stocks/snapshots` | OPT-OUT | Composite latest-value endpoint; no historical window parameter, so it cannot feed the resumable backfill this phase delivers |
| `stocks/auctions` | OPT-OUT | Not in D-07's delivered set (daily bars, minute bars, quotes/trades); opening/closing auction prices have no consumer in the current pipeline and no success criterion references them |
| `stocks/meta/conditions` | OPT-OUT | Reference metadata for decoding trade/quote condition codes. Raw tick is landed undownsampled and undecoded per D-16; decoding belongs to the deferred 03.3 tick→Zarr phase |
| `stocks/meta/exchanges` | OPT-OUT | Same as `stocks/meta/conditions` — exchange-code decoding is a 03.3 concern, not an acquisition concern |
| `corporate-actions` | OPT-OUT | Out of scope per the ROADMAP scope fence and D-07. RESEARCH also records it excludes delistings/reorganizations, so it cannot substitute for Tiingo's `supported_tickers.csv` delisting signal |
| `news` | OPT-OUT | Unstructured text, not market data. May be ingested later, but no v1 requirement selects it and it needs a non-OHLCV Dataset shape that D-18 defers |
| `screener/stocks/most-actives` | OPT-OUT | Derived ranking, not raw market data; the pipeline derives rankings in the factor layer from data it already owns |
| `screener/{market}/movers` | OPT-OUT | Same as `most-actives` |
| `crypto/{loc}/bars`, `crypto` quotes/trades/snapshots | OPT-OUT | The `crypto_spot` market is sourced from Binance (Phase 2). No requirement selects Alpaca as a crypto vendor, and this phase's `Market` scope is `us_equity` |
| `options/bars`, options trades/quotes/snapshots/chain | OPT-OUT | No options instrument model exists in the pipeline; `enums/data.py:Market` is a locked literal set with no options token, and adding one requires revisiting 02-RESEARCH.md Assumptions Log A2 |
| `forex/rates`, `forex/latest-rates` | OPT-OUT | No FX market token in `enums/data.py:Market`; no requirement references FX |
| `logos/{symbol}` | OPT-OUT | Presentation asset; CLAUDE.md fences this milestone to backend only, no frontend |
| streaming websocket (stocks / crypto / options / news) | OPT-OUT | D-15/D-16 exclude live/streaming this phase. The `Acquisition` contract this phase generalizes is request-response; a streaming source is a different lifecycle |
| Trading / Broker API (orders, positions, account) | OPT-OUT | Explicit ROADMAP scope fence and D-15 — market-data endpoints only. Overlaps Phase 6's NautilusTrader work |

## Request-parameter capabilities (`stocks/bars`, `stocks/quotes`, `stocks/trades`)

Opting out of a parameter is as consequential as opting out of an endpoint — `asof` in particular
silently reintroduces survivorship bias when left at its vendor default (RESEARCH Pitfall 5).

| capability | decision | reason |
|---|---|---|
| `symbols` (comma-separated, multi-symbol) | INTEGRATE | This is SC-1's whole point |
| `page_token` / `next_page_token` | INTEGRATE | SC-3 |
| `timeframe` | INTEGRATE | |
| `start` / `end` | INTEGRATE | |
| `limit` (page size, vendor max 10,000) | INTEGRATE | `config.kwargs["page_limit"]` |
| `adjustment` (`raw`/`split`/`dividend`/`all`) | INTEGRATE | `config.kwargs["adjustment"]`, no in-code default beyond the vendor's `raw` |
| `asof` (symbol-mapping date) | INTEGRATE | Required-by-convention, never left to the vendor default of "today" |
| `feed` (`iex`/`sip`/`delayed_sip`/`otc`/`boats`/`overnight`) | INTEGRATE | `config.kwargs["feed"]` with **no in-code default** until the SIP-tier human checkpoint resolves (D-12, RESEARCH Open Question 1 / Assumption A7) |
| `sort` | INTEGRATE | Pinned to `asc` rather than exposed: `desc` inverts the page-resume semantics SC-3 depends on |
| `currency` | OPT-OUT | Every symbol in the `us_all` / index rosters is USD-denominated; a non-USD request would return values the `[timestamp, symbol]` panel has no unit column to disambiguate |
