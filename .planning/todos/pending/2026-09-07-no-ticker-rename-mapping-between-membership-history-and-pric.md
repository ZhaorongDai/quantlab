---
created: 2026-09-07T05:26:07.494Z
title: No ticker-rename mapping between membership history and price roster
area: data / acquisition
severity: major
triggers: a backtest (or any panel join) that resolves point-in-time index membership and
  then looks those symbols up in the price panel. Harmless while only recent --as-of-date
  values are used; fires as soon as historical membership meets historical prices.
files:
  - acquisition/universe.py (IndexMembershipFetcher — Wikipedia change logs, period-correct tickers)
  - acquisition/universe.py (TiingoRosterFetcher — Tiingo directory, history rewritten onto the CURRENT ticker)
  - base/constituent.py (UniverseCatalog — where the two vocabularies are joined)
---

## Problem

The two halves of this project speak **different ticker vocabularies across a rename**,
and nothing maps between them. Grepped 2026-09-07 across `acquisition/`, `base/acquisition.py`
and `dataset/`: there is no rename map, no alias table, no permaticker, no predecessor/
successor concept anywhere.

- **Index membership** comes from Wikipedia change logs, which record the ticker **as it
  was on the date of the event**. So the S&P 500 panel legitimately contains `FB`.
- **Prices** come from Tiingo's directory, which **rewrites a renamed security's whole
  history onto the new ticker**. `FB` does not exist there at all; `META` carries
  `start_date = 2012-05-18` — Facebook's IPO date.

Measured against the rebuilt table (2026-09-07):

```
sp500_constituent symbols with no counterpart in the us_all price roster: 89 / 876

renames (new ticker present, old absent):
  FB -> META      CDAY -> DAY      DWDP -> DD      COG -> CTRA      ADS -> BFH
notation (separate todo):  BF.B, BRK.B
exchange filter / long-delisted:  CBOE (listed on CBOE, excluded by EXCHANGE_FILTER),
  EK, BS, CEPH, ABK, CFC, ...
```

**Why this is `major` and not cosmetic.** The failure is silent and produces a wrong
answer rather than an error: a constituent whose period-correct ticker has no price
column reads as "this symbol has no data in the window", which is indistinguishable from
a genuine data gap. A backtest would quietly drop Facebook from every pre-2022 S&P 500
portfolio instead of refusing to run.

**Not yet reachable in practice**, which is why it is filed rather than fixed now: current
usage passes a recent `--as-of-date`, where membership tickers are already the current
ones. It becomes reachable at Phase 4+ when a model or backtest walks membership through
history.

## Solution

TBD — needs a design decision, not just an implementation. Options, none yet evaluated:

1. **Vendor identity key.** Tiingo exposes a `permaTicker` on some endpoints; if the
   directory feed can carry it, join on identity instead of on the display symbol. Removes
   the whole class of problem, but `supported_tickers.csv` (what `TiingoRosterFetcher`
   currently reads) does not include it — this would change the source, not just the code.
2. **Explicit rename map**, maintained as reference data alongside the universe table, and
   applied when membership symbols are resolved against the price roster.
3. **Reconcile at build time**, e.g. detect that a membership symbol is absent from the
   roster while a roster symbol covers the same interval, and record the correspondence.
   Cheap to get wrong — two unrelated securities can share an interval.

Whatever is chosen, the join must **fail loudly** on an unresolvable membership symbol
rather than yielding an empty column. That property is worth more than any particular
mapping mechanism, and it is what is missing today.

## Verification requirement

A test that walks a historical membership date across a known rename (FB/META is the
clearest case: the S&P 500 panel holds both) and asserts the price lookup resolves — plus
a test that an unresolvable symbol RAISES rather than returning an all-null column. This
project has ten recorded instances of tests passing for the wrong reason; asserting only
that "the join returned something" would be exactly such a test.

## Related

`2026-09-07-normalize-ticker-delimiter-between-membership-panel-and-pric.md` covers the
`BF.B`/`BRK.B` notation half of the same 89-symbol gap. It is a separate, much smaller fix
and does not address renames.

## 2026-09-20 — how the CRSP vendor sidesteps this (phase 03.10)

**This todo stays OPEN.** Phase 03.10 did not fix it; it added one vendor for which the
problem cannot arise, which is a different thing and is worth recording so the next reader
does not mistake a green CRSP panel for a closed defect.

The CRSP/WRDS vendor (`example/wrds_crsp.md`) has **one PERMNO-keyed symbology shared by
its prices and its universes**. PERMNO is CRSP's permanent security identifier; a rename
(FB → META) does not change it. The raw tier is keyed by PERMNO and carries no ticker at
all — the period-correct ticker is derived at CONVERSION time from
`crsp_a_stock.stksecurityinfohist` by `quantlab/dataset/crsp/symbology.py:CrspSymbology`.

Both halves that disagree for Tiingo/Wikipedia are then produced by the SAME object:

- the price panel's row labels — `CrspStockDataset._derivation` calls
  `CrspSymbology.label_rows`;
- the membership intervals — `CrspMembership.symbol_intervals` intersects PERMNO
  membership with `CrspSymbology.symbol_intervals`.

So the two vocabularies agree **by construction** rather than by a mapping that could be
incomplete: `FB` is `FB` through 2022-06-08 and `META` from 2022-06-09 on BOTH sides, and
`BRK.B` is spelled one way on both sides. This is proved end to end rather than asserted —
`tests/test_crsp_constituent.py::test_membership_symbols_agree_with_the_crsp_price_panel`
runs a real pull, a real conversion, and compares `(date, symbol)` sets per PERMNO.

Two things this does NOT do:

1. **It does not help the Tiingo/Alpaca + Wikipedia path**, which is where the measured
   89/876 gap lives. Those vendors have no permanent identifier in the feeds this project
   reads, so the "Solution" options above are still the open design decision.
2. **It does not give the failure the loud behaviour this todo asks for.** The
   verification requirement — an unresolvable membership symbol must RAISE rather than
   yield an empty column — remains unmet for the Wikipedia-based universes. (The CRSP side
   does behave that way: a membership day no CRSP ticker covers is dropped into
   `report['unlabelled_members']` with a warning, and an unlinked Nasdaq-100 spell raises
   by default.)

If option 1 (a vendor identity key) is eventually chosen, the CRSP implementation is the
worked example of what it buys: the whole class of problem disappears, and no rename map
has to be maintained.
