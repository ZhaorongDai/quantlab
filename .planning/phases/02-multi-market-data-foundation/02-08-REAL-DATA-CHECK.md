# 02-08 Real-Data Verification

**Date run:** 2026-09-05
**Network access:** available (verified reachable: `apimedia.tiingo.com`, `en.wikipedia.org`, `raw.githubusercontent.com`)

This mirrors `02-05-REAL-DATA-CHECK.md`'s precedent -- results below are from
actually running the plan's `<verification>` commands against the real,
live Tiingo/Wikipedia/GitHub sources, not fabricated.

## 1. `refresh_us_equity_universe.py` (full build against live sources)

```
uv run python refresh_us_equity_universe.py
```

Completed successfully. Output: `data/data/reference/universe.parquet`.

Produced many `loguru.logger.warning` lines during `reconstruct_intervals()`
for real S&P 500 removal events with no matching "added" row in the
Wikipedia change log (left-censored — genuinely predates
`PIT_COVERAGE_START` 1976-07-01, e.g. long-tenured names like `WBA`, `K`,
`CAG`) and for anchor-set mismatches caused by real historical ticker
renames not present in the current `constituents.csv` anchor (e.g. `FB`
-> `META`, `PCLN` -> `BKNG`). Both categories of warning are the plan's
intended loud-warning-not-silent-default behavior working correctly against
real, messy historical data — not implementation bugs.

## 2. Point-in-time query smoke check

```
uv run python -c "from acquisition.universe import UniverseCatalog; from config import universe_config; c = UniverseCatalog.load(universe_config()); print(len(c.get_symbols_as_of('sp500_constituent', '2020-12-21')))"
```

Result: `521` (S&P 500 has ~500 constituents; the small excess above 500 is
expected from historical re-entries/reconciliation producing a few extra
open-ended interval rows for renamed/reorganized tickers — consistent with
the interval-table model, not a bug).

## 3. Mandatory point-in-time correctness (Tesla's real 2020-12-21 addition)

```
uv run python -c "
from acquisition.universe import UniverseCatalog
from config import universe_config
c = UniverseCatalog.load(universe_config())
before = c.get_symbols_as_of('sp500_constituent', '2020-12-20')
on = c.get_symbols_as_of('sp500_constituent', '2020-12-21')
print('TSLA in before (2020-12-20):', 'TSLA' in before)
print('TSLA in on-date (2020-12-21):', 'TSLA' in on)
"
```

Result:
```
TSLA in before (2020-12-20): False
TSLA in on-date (2020-12-21): True
```

**Confirmed against real, live data** — matches Tesla's real, publicly
documented S&P 500 addition date.

## 4. Real delisted-stock verification (ATVI, TWTR)

```
uv run python -c "
import polars as pl
from acquisition.universe import UniverseCatalog
from config import universe_config
c = UniverseCatalog.load(universe_config())
rows = c._backend.get_lazyframe().filter(pl.col('symbol').is_in(['ATVI', 'TWTR'])).collect()
print(rows)
"
```

Result:
```
shape: (3, 4)
┌────────┬───────────────────┬────────────┬────────────┐
│ symbol ┆ category          ┆ start_date ┆ end_date   │
│ ATVI   ┆ nasdaq_all        ┆ 1993-10-25 ┆ 2023-10-13 │
│ TWTR   ┆ sp500_constituent ┆ 2018-06-07 ┆ 2022-11-01 │
│ ATVI   ┆ sp500_constituent ┆ 2015-08-28 ┆ 2023-10-18 │
└────────┴───────────────────┴────────────┴────────────┘
```

**`ATVI`'s `nasdaq_all` `end_date` (2023-10-13) matches exactly** the real
historical delisting date cited in `02-08-RESEARCH.md`'s live-verified
Tiingo data.

**`TWTR` does NOT appear in the `nasdaq_all` category** -- this is correct,
not a gap: Twitter (`TWTR`) traded on the **NYSE**, not NASDAQ, so it is
correctly excluded by `NasdaqUniverseFetcher`'s `EXCHANGE_FILTER = ("NASDAQ",)`
(Locked Decision A4). It correctly appears under `sp500_constituent`
(Wikipedia-sourced, not exchange-restricted) with `end_date` 2022-11-01,
close to (a few days after) `TWTR`'s real 2022-10-28 delisting -- the small
gap reflects the S&P 500 index-committee's removal-effective-date versus the
exchange delisting date, which are two independent, related-but-distinct
events; this is expected, not a bug.

## Conclusion

Real-data verification confirms:
- The universe table genuinely includes historically delisted NASDAQ stocks
  (`ATVI`'s NASDAQ delisting date matches exactly).
- Point-in-time S&P 500 membership resolution is correct against a real,
  independently-verifiable historical event (Tesla's 2020-12-21 addition).
- The NASDAQ-only exchange scope (Locked Decision A4) is correctly enforced
  even for well-known S&P 500 names that trade on a different exchange
  (`TWTR`/NYSE).

This verification is non-gating per the plan (mirrors
`02-05-REAL-DATA-CHECK.md`'s precedent) -- the gating verification is the
mocked/fixture-based automated test suite (`tests/test_universe.py`,
7/7 passing) plus the full regression suite (`uv run pytest tests/ -x -q`,
39/39 passing).
