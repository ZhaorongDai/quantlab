---
status: accepted
date: 2026-09-25
---

# Downloads run without a volume guard

quantlab used to refuse a download before it started when an estimate said it would exceed a
ceiling (20 GiB of raw data, 50,000 requests or 4 hours for the REST vendors Tiingo and Alpaca;
20 GiB or 700 million rows for WRDS, priced from a `count(*)` probe). Both guards, their CLI flags
(`--force-volume`, `--rows-per-symbol-day`) and their tests are removed as of 2026-09-25, when the
download scripts were reorganised around WRDS.

We decided that a script should be a thin shell over one library call and expose only the
arguments that change between runs. The guards cost every run an extra SQL round trip or a roster
sizing pass, added two flags and about 1,800 lines of tests, and had never blocked a download that
the user wanted: the whole CRSP daily file fits under the ceiling, and TAQ pulls are scoped by
symbol and window before they run. The REST guard also lived in `quantlab.universe` purely so it
could run before a vendor client existed, which forced an import-order rule on three package
`__init__` files. Removing the guard removes that rule.

## Consequences

- A download's size is not estimated or checked. A too-large TAQ window fills the disk instead of
  being refused. Scope such pulls by symbol list and date range.
- `quantlab.universe` no longer needs to stay free of acquisition imports for the guard's sake.
  The volume-guard tests that enforced this were deleted with the guard.
- If a guard is wanted again, put it inside the library ingest function for the vendor that needs
  it, not in the scripts and not in `quantlab.universe`.
