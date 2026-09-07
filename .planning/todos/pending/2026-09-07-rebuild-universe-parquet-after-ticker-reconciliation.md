---
status: open
opened: 2026-09-07
severity: major
area: data / acquisition
source: quick task 260907-10t
blocks: a full-market `download()` / `refresh()` run over `us_all` or `nasdaq_all`
---

# Rebuild `universe.parquet` after the ticker-guard reconciliation

## What is still broken

`data/data/reference/universe.parquet` was BUILT BEFORE quick task
260907-10t and still contains **all 13 malformed entries**. The build-time
well-formedness filter added in that task lives in
`acquisition/universe.py:TiingoRosterFetcher.fetch()` and therefore only takes
effect on a **rebuild**. It does not retroactively clean the persisted table.

Measured 2026-09-07 against the live table (25,709 rows) — these symbols are
still in it and are still refused by `Acquisition._validate_symbols`:

```
us_all     (6): CAPTW(EXP20260807), DTV_1, ETP-, NSPR-WSB,
                NXT(EXP20091224), OXY-WSW
nasdaq_all (7): -P-HIZ, ASRV 8.45 06-30-28, CAPTW(EXP20260807), CHNG 6,
                DTV_1, NSPR-WSB, NXT(EXP20091224)
```

So a full-market run against the CURRENT table **still halts at pre-flight**.
What changed is only WHICH symbol it halts on: `NXG-R-W` before the fix,
`ETP-` / `DTV_1` / `CAPTW(EXP20260807)` after it. The 83 → 6 reduction in
`us_all` failures is real, but 6 is not 0, and the pre-flight refuses on the
first one it meets.

## What WAS verified, and what was NOT

**Verified (offline, no network, no credentials):**

- Every symbol both roster builders PRODUCE is accepted by
  `Acquisition._validate_symbols` — asserted on the full builder output of
  both rosters, not a spot-check
  (`tests/test_ticker_pattern_reconciliation.py`).
- Both fetchers drop all 9 measured malformed literals; the retained
  multi-suffix warrants and every `nasdaq_all` preferred survive.
- The chosen pattern rejects exactly and only the 13 symbols listed above,
  measured directly against the persisted table.

**NOT verified, and explicitly not claimed:**

- A live full-market `download()` run. `TIINGO_API_KEY` is not set in this
  environment and `scripts/refresh_us_equity_universe.py` hits the live
  vendor, so the table could not be rebuilt here and no end-to-end run was
  attempted.

## How to close this

```bash
export TIINGO_API_KEY=...        # Tiingo Dashboard -> API -> Token
uv run python scripts/refresh_us_equity_universe.py
```

Then confirm the table is clean:

```bash
uv run python -c "
import polars as pl
from enums.data import TRADEABLE_TICKER_PATTERN as P
df = pl.read_parquet('data/data/reference/universe.parquet')
bad = [s for s in sorted(set(df['symbol'].to_list())) if not P.match(s)]
print('unfetchable symbols remaining:', bad)
"
```

An empty list closes this todo. A non-empty one means the vendor directory
has grown a shape the pattern does not cover — read
`enums/data.py:TRADEABLE_TICKER_PATTERN`'s comment before widening anything,
in particular the recorded `NSPR-WSB` / `OXY-WSW` argument.

## Carried-forward finding (do not silently reverse)

`NSPR-WSB` is dropped and, unlike `OXY-WSW`, that drop **does lose a
security**: the family is `['NSPR', 'NSPR-WS', 'NSPR-WSB']` with no
`NSPR-WS-B`, so one microcap warrant series leaves the roster. Accepted
deliberately rather than widen every suffix segment from `{1,2}` to `{1,3}`
for all 14,485 symbols on the evidence of two outliers, one of which
(`OXY-WSW`) is redundant with the already-present `OXY-WS-W`. Recorded in the
same idiom 260906-eme used for the retained warrants.
