---
created: 2026-09-07T05:26:07.494Z
title: Normalize ticker delimiter between membership panel and price roster
area: data / acquisition
severity: minor
files:
  - quantlab/universe.py (IndexMembershipFetcher — Wikipedia-sourced symbols)
  - quantlab/universe.py (TiingoRosterFetcher — vendor-sourced symbols)
  - data/data/reference/universe.parquet (both categories land in one table)
---

## Problem

Class-share tickers are spelled with a **dot** by the Wikipedia change-log source and
with a **hyphen** by Tiingo's directory. Both land in `universe.parquet` under different
categories, so the same security carries two spellings and the two never join.

Measured 2026-09-07 against the rebuilt table:

```
sp500_constituent symbols absent from the us_all price roster, dot-spelled: ['BF.B', 'BRK.B']
both are present in us_all after s.replace('.', '-'):                        ['BF-B', 'BRK-B']
```

`BRK-B` and `BF-B` are not obscure — `quantlab/universe.py`'s own 260906-eme docstring
lists them among the class shares it verified as surviving the preferred-share filter.
So the securities are present on both sides; only the notation differs.

Impact is currently limited: a membership-driven run resolves symbols from the membership
category and looks them up in the price roster, and these two silently miss. Nothing
raises. It is `minor` rather than `major` only because a caller can normalize the
delimiter themselves and because it is two symbols, not a class of them.

## Solution

Normalize at the boundary where the two vocabularies meet, not at every call site.
Candidates, to decide when picking this up:

1. Normalize dot→hyphen when the membership fetchers emit symbols, so `universe.parquet`
   holds one spelling. Simplest, but rewrites what the source said.
2. Keep both spellings and add an explicit alias resolution step in `UniverseCatalog`.
   Preserves provenance; more machinery.

Whichever is chosen, note that `TRADEABLE_TICKER_PATTERN` (`enums/data.py`) already admits
BOTH `.` and `-` as the delimiter, so neither spelling is refused by the fetch guard — this
is purely a join problem, not a validation one.

**Verification requirement.** A test that asserts every membership-category symbol resolves
to a price-roster symbol for the class-share cases specifically. Asserting `BRK.B` alone
would pass for the wrong reason if the fix hardcoded that one pair.

## Related

See `2026-09-07-no-ticker-rename-mapping-between-membership-history-and-pric.md` — the
same join breaks for a different, larger reason (renames). These two together explain
89 of the 876 sp500_constituent symbols. Fixing this one does NOT fix that one.
