---
phase: quick-260906-eme
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - acquisition/universe.py
  - tests/conftest.py
  - tests/test_universe.py
autonomous: true
requirements: [DATA-05]
user_setup: []

estimate:
  tokens: 70000
  raw_tokens: 70000
  tasks: 2
  confidence: low

must_haves:
  truths:
    - "A Wikipedia change-log ticker cell carrying upstream delimiter residue (the live `ALLE |` / `JCP |` / `ITT |` shape) is normalized to a clean ticker, and the normalization is announced by a logger.warning naming the index, the column and every raw->normalized pair."
    - "A cell that is STILL not a well-formed ticker after normalization raises, naming every offending cell — it does not pass through and it does not get silently corrected."
    - "A cell whose entire content is delimiter residue normalizes to blank and becomes the existing `None` no-change-on-this-side sentinel, not an error."
    - "After the fix `JCP` and `ITT` are real, market-data-matchable symbols in the S&P 500 membership intervals, and `ALLE` appears exactly once instead of twice."
    - "`USEquityUniverseFetcher.fetch()` drops preferred shares and baby bonds: 965 of 16,138 measured rows (940 of 15,425 distinct tickers) go, leaving 15,173 rows."
    - "The exclusion keeps every class-share ticker: BRK-A/BRK-B, BF-A/BF-B, PBR-A, HEI-A, MOG-A, LEN-B, CWEN-A, UA-C, MKC-V, AGM-A, GEF-B, STZ-B, UHAL-B, CRD-A, LGF-A all survive, verified against the live Tiingo directory."
    - "`NasdaqUniverseFetcher.fetch()` is byte-for-byte unaffected: given one CSV containing NASDAQ preferreds, `nasdaq_all` keeps them and `us_all` drops them (Locked Decision 1)."
    - "The post-exclusion row count is checked against `MIN_ROSTER_ROWS`, so the exclusion cannot trip the drift guard it is not meant to trip (15,173 vs a floor of 8,000 = 1.90x headroom)."
  artifacts:
    - "acquisition/universe.py — IndexMembershipFetcher._normalize_ticker_cell() + _WELL_FORMED_TICKER validation inside _parse_changes_table()"
    - "acquisition/universe.py — TiingoRosterFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES (default False) + _PREFERRED_SHARE_PATTERN / _BABY_BOND_PATTERN, opted into by USEquityUniverseFetcher only"
    - "tests/conftest.py — NYSE-only preferred/baby-bond rows appended to the mock_universe_fetchers roster CSV"
    - "tests/test_universe.py — normalization/validation tests and roster-exclusion tests including the nasdaq_all immunity proof"
  key_links:
    - "_normalize_ticker_cell() must run BEFORE the _BLANK_TICKER_CELLS sentinel test in _parse_changes_table's per-column loop — otherwise a residue-only cell is validated as a ticker and raises instead of becoming None."
    - "The exclusion filter must run BEFORE the MIN_ROSTER_ROWS check in fetch() — otherwise the guard validates a count that is not the count that gets persisted."
    - "EXCLUDE_NON_COMMON_SECURITY_TYPES defaults to False on the base — that default is the ONLY thing keeping nasdaq_all's semantics frozen per Locked Decision 1."
---

<objective>
Two independent roster-hygiene fixes in `acquisition/universe.py`, both of which
change what lands in `universe.parquet`.

**(A)** Normalize and validate ticker cells parsed from the Wikipedia
index-change tables, so an upstream editor's wikitable typo cannot enter the
membership table as an unmatchable phantom symbol.

**(B)** Exclude preferred shares and baby bonds from the `us_all` full-market
roster, scoped so `nasdaq_all` is untouched.

Purpose: (A) closes a survivorship-bias hole — `JCP` and `ITT` are today
representable in the S&P 500 panel ONLY as `JCP |` / `ITT |`, which match no
market-data symbol, so two decades-long members are simply absent, while
`ALLE` is double-counted. (B) removes ~940 non-common-stock lines from the
full-market roster so downstream ingestion stops spending Tiingo requests and
panel columns on preferred series and notes.

Output: `acquisition/universe.py` carrying both guards, plus tests pinning the
measured criteria.
</objective>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@acquisition/universe.py
@base/constituent.py
@dataset/constituent.py
@tests/conftest.py
@tests/test_universe.py
</context>

<measured_evidence>
Everything below was MEASURED at planning time. Do not re-derive it; do not
substitute a guess for it.

## (A) Wikipedia change-log ticker vocabulary — live fetch, 2026-09-06

Both change logs were fetched live and pushed through the CURRENT
`IndexMembershipFetcher._parse_changes_table`:

| Index | rows | non-null ticker cells | malformed | character vocabulary |
|---|---|---|---|---|
| S&P 500 | 407 | 772 | **3** — `ALLE \|`, `ITT \|`, `JCP \|` | `A-Z`, space, `\|` |
| Nasdaq-100 | 226 | 418 | **0** | `A-Z` only |

Ticker-cell lengths observed: 1–6 (S&P), 2–5 (NDX). No `.`, no `-`, no digits
appear in either live table today. The malformed shape is a TRAILING ` |` in
every case; there is no interior-pipe case.

After stripping whitespace and leading/trailing pipes, **all 1,190 live cells
across both indexes** match `^[A-Z]{1,7}(?:[.-][A-Z]{1,2})?$`.

Persisted consequence in the user's real
`/Volumes/SSD/data/data/reference/_cache/sp500_changes_snapshot.parquet`:
`ALLE` exists twice (once clean, once as a phantom `ALLE |` with a fabricated
2013-12-02 → 2026-08-18 interval), and `JCP` / `ITT` exist ONLY as phantoms.

## (B) Tiingo `supported_tickers.csv` — downloaded and measured, 2026-09-06

Downloaded from the public URL (no API key). Applying
`USEquityUniverseFetcher`'s existing filter
(`exchange in {NASDAQ, NYSE, AMEX, NYSE MKT}`, `assetType == "Stock"`,
`priceCurrency == "USD"`) yields **16,138 rows / 15,425 distinct tickers**.

Enumerated shapes of the non-common-stock population:

| Shape | count (distinct) | what it is | in scope? |
|---|---|---|---|
| `ROOT-P-<SERIES>` (3 seg) | 823 | preferred | YES |
| `ROOT-P-<SERIES>-<X>` (4 seg) | 84 | preferred | YES |
| `ROOT-P` (2 seg) | 20 | preferred, no series letter | YES |
| `ROOT/P<SERIES>` (slash) | 3 (`BC/PA`, `BC/PB`, `BC/PC`) | preferred, alternate notation | YES |
| `ROOT- PR-<X>` | 1 (`NYCB- PR-U`) | preferred, `PR` spelling + stray space | YES |
| `ROOT--P-<X>` | 3 (`SCE--P-D`, `IMH-P--B`, `IMH-P--C`) | preferred, doubled delimiter | YES |
| `-P-HIZ` | 1 | preferred, leading-hyphen malformation | YES |
| `ROOT <coupon>[ <maturity>]` | 8 | baby bonds / notes | YES |
| `ROOT-WS`, `-U`, `-W`, `-R`, `-WD`, `-WI`, `-CL` | 1,124 | warrants / units / rights / when-issued | **NO — out of scope** |
| `ROOT-<single letter>` | ~100 | **class shares** (BRK-B, BF-B, PBR-A, …) | **NO — MUST survive** |

**The criterion, pinned against that reality:**

```
_PREFERRED_SHARE_PATTERN = r"[-/]\s*P[A-Z]?\s*(?:[-/]|$)"
_BABY_BOND_PATTERN       = r"\s\d"
```

Measured result of `~contains(PREFERRED) & ~contains(BABY_BOND)`:

- **16,138 rows → 15,173 rows** (965 rows / **940 distinct tickers** excluded:
  932 preferred + 8 baby bonds, zero overlap).
- Cross-checked against an independent Python segment-split reference
  implementation (split on `[-/]`, whitespace-strip each segment, any segment
  at index >= 1 matching `^P(?:R|[A-Z])?$`): **the two agree on all 15,425
  tickers, exactly**.
- **Zero legitimate common stocks excluded.** Verified survivors: `AAPL`,
  `MSFT`, `BRK-A`, `BRK-B`, `BF-A`, `BF-B`, `PBR-A`, `HEI-A`, `MOG-A`,
  `MOG-B`, `LEN-B`, `CWEN-A`, `UA-C`, `MKC-V`, `AKO-A`, `AKO-B`, `AGM-A`,
  `NYLD-A`, `TAP-A`, `LGF-A`, `LGF-B`, `GTN-A`, `HVT-A`, `BWL-A`, `BH-A`,
  `BNRE-A`, `CRD-A`, `CRD-B`, `RDS-A`, `RDS-B`, `FCE-A`, `GEF-B`, `STZ-B`,
  `UHAL-B`, `WSO-B`, `BIO-B`, `CIG-C`, `EBR-B`, `TI-A`, `GGO-C`, `BALY-T`,
  `SPWR-V`, `DGAC-UN`, and every warrant/unit/right form (`C-WS-A`,
  `GM-WS-B`, `MIMO-W-A`, `ACP-R-W`).
- Every one of the 34 excluded tickers that is NOT of the canonical
  `ROOT-P-SERIES` shape was individually inspected and is a preferred or a
  baby bond. There are no unclassified drops.
- `MIN_ROSTER_ROWS` headroom: 15,173 vs a floor of 8,000 = **1.90x**. The
  exclusion cannot trip that guard.

**Deliberately out of scope, stated rather than silently omitted:** the
1,124 warrant / unit / right / when-issued lines (`-WS` 391, `-U` 360,
`-W` 164, `-R` 138, `-CL` 57, `-WD` 11, `-WI` 3) remain in `us_all`. The task
scope is preferred shares and baby bonds. This is a named, measured,
carried-forward finding — not an oversight.
</measured_evidence>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Normalize and validate Wikipedia change-log ticker cells</name>
  <files>acquisition/universe.py, tests/test_universe.py</files>
  <behavior>
    - `_normalize_ticker_cell("ALLE |") == "ALLE"`; likewise `"JCP |"` -> `"JCP"`, `"ITT |"` -> `"ITT"` (the three real live values).
    - `_normalize_ticker_cell(" | ") == ""` — a residue-only cell normalizes to blank.
    - A change-log table containing `ALLE |` parses to `added_ticker == "ALLE"` AND emits a `logger.warning` naming the index label, the column and the `ALLE |` -> `ALLE` pair.
    - A change-log table whose ticker cells are already clean emits NO normalization warning.
    - A cell whose whole content is delimiter residue becomes `None` (the existing no-change-on-this-side sentinel), not a validation error.
    - A cell still malformed after normalization (interior residue, e.g. `AL|LE`, or a lowercase/spaced value) raises `ValueError` whose message names every offending cell.
    - The existing `NA`-ticker behaviour is unchanged: `NA` survives parsing as the string `"NA"` (`test_na_tickered_change_rows_survive_parsing` must stay green).
  </behavior>
  <read_first>
    acquisition/universe.py — `_BLANK_TICKER_CELLS`, `IndexMembershipFetcher._parse_changes_table` (the per-column sentinel loop and the `unparseable effective_date` guard immediately after it, which is the precedent for the raise shape).
    tests/conftest.py — `sp500_changes_html_fixture` / `ndx_changes_html_fixture`, and `tests/test_universe.py::test_unparseable_effective_date_names_the_offending_rows` / `test_na_tickered_change_rows_survive_parsing` for the standalone-parse test idiom.
  </read_first>
  <action>
Add a module-level compiled pattern next to `_BLANK_TICKER_CELLS`:

    _WELL_FORMED_TICKER = re.compile(r"^[A-Z]{1,7}(?:[.-][A-Z]{1,2})?$")

Its comment must record the measurement it is pinned to: 1,190 live ticker
cells across both change logs, vocabulary `A-Z` only, lengths 1-6; the
optional `[.-][A-Z]{1,2}` tail is deliberate headroom for the `BRK.B` /
`BRK-B` class-share notations that do not appear in either table today but
are the one shape a future S&P row could legitimately carry. Import `re` at
the top of the module.

Add a `@staticmethod _normalize_ticker_cell(value: str) -> str` on
`IndexMembershipFetcher` that applies, in order: `.strip()`, then strip
leading/trailing pipe characters, then `.strip()` again.

**Place it on `IndexMembershipFetcher`, not on either subclass.** Argue this
in the docstring, and say why: `_parse_changes_table` is already concrete and
shared precisely so no index can ship a parse that skips the base's
validation — that is safety property 1 on the class docstring, and it exists
because the previous per-subclass parse meant nothing looked at the source
header at all. The Nasdaq-100 table has zero malformed cells today, but it is
the same MediaWiki editing surface, and the S&P table's three cells arrived
by editor typo alone. Per-subclass placement would reintroduce exactly the
shape that property was refactored away from.

Rewrite the existing per-column loop in `_parse_changes_table` so that, for
each of `added_ticker` / `removed_ticker`:

  1. Normalization runs FIRST, on the whitespace-stripped raw values.
  2. Every cell the normalization CHANGED is collected, then reported in ONE
     `logger.warning` per column naming `self.INDEX_LABEL`, the column, and
     the full list of raw -> normalized pairs. One aggregated line, not one
     per cell: a systematic parser regression then shows up as a single huge
     list rather than flooding a cron log, while the ongoing 3-cell upstream
     typo case stays a single readable line. The logging is load-bearing —
     silent correction would swallow a future genuine parser regression,
     which is the whole reason this module's other guards are loud.
  3. The `_BLANK_TICKER_CELLS` sentinel test runs on the NORMALIZED value, so
     a residue-only cell becomes the `None` sentinel instead of reaching
     validation. This ordering is the key link; state it in a comment.
  4. Non-blank normalized values are matched against `_WELL_FORMED_TICKER`;
     failures are accumulated across BOTH columns.

After the loop, if any accumulated failures exist, raise `ValueError` naming
`self.INDEX_LABEL`, `self.CHANGES_URL` and every offending `(column, value)`
pair — the same shape as the `unparseable effective_date` guard directly
below, and for the same reason: `fetch_changes()` catches it, falls back to
the cached snapshot without overwriting it, and `build()` then refuses unless
`allow_stale=True`. Raising is therefore loud-but-non-destructive, which is
what makes it safe to raise on a shape we have not seen.

Do NOT make an interior-residue cell pass. An interior pipe is not the
observed upstream shape and would more likely mean two cells were merged by a
parser regression, which must be loud.

Add tests to `tests/test_universe.py`, in a new section, following the
standalone-parse idiom of `test_unparseable_effective_date_names_the_offending_rows`
(construct the fetcher via `__new__`, call `_parse_changes_table` on synthetic
HTML). Use the three REAL observed literals in at least one test. Assert the
warning via `caplog` or the repo's loguru capture idiom already used by
`test_orphaned_open_interval_is_flagged_as_an_inferred_end`.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run pytest tests/test_universe.py -q</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run python -c "from acquisition.universe import IndexMembershipFetcher as F; assert [F._normalize_ticker_cell(v) for v in ('ALLE |','JCP |','ITT |',' | ','AAPL')] == ['ALLE','JCP','ITT','','AAPL']; print('normalization OK')"</automated>
  </verify>
  <done>
    `tests/test_universe.py` passes with the new normalization/validation tests
    green and every pre-existing test in it still green.
    `_normalize_ticker_cell` maps the three real observed values to clean
    tickers and a residue-only cell to the empty string.
    A still-malformed cell raises a `ValueError` naming every offending cell.
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: Exclude preferred shares and baby bonds from the us_all roster only</name>
  <files>acquisition/universe.py, tests/conftest.py, tests/test_universe.py</files>
  <behavior>
    - `USEquityUniverseFetcher.fetch()` drops every measured preferred shape: `AAM-P-A`, `ZB-P-F-CL`, `MTB-P`, `BC/PA`, `SCE--P-D`, `IMH-P--B`, `NYCB- PR-U`, `-P-HIZ`.
    - `USEquityUniverseFetcher.fetch()` drops every measured baby-bond shape: `ASRV 8.45 06-30-28`, `SO 6.75 08-01-22`, `NEE 6.219`, `CHNG 6`.
    - `USEquityUniverseFetcher.fetch()` KEEPS every class-share shape: `BRK-A`, `BRK-B`, `BF-B`, `PBR-A`, `MOG-A`, `LEN-B`, `UA-C`, `MKC-V`, `AGM-A`, `CRD-A`, `LGF-A`, `GEF-B`, `STZ-B`, `UHAL-B`, `CWEN-A`, `HEI-A`.
    - `USEquityUniverseFetcher.fetch()` KEEPS warrants/units/rights (`C-WS-A`, `GM-WS-B`, `MIMO-W-A`, `ACP-R-W`, `DGAC-UN`) — out of scope, and their retention is asserted so a later widening is a deliberate edit.
    - Given ONE CSV carrying NASDAQ preferreds, `NasdaqUniverseFetcher().fetch()` keeps them and `USEquityUniverseFetcher().fetch()` drops them.
    - `NasdaqUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is False` and `USEquityUniverseFetcher.EXCLUDE_NON_COMMON_SECURITY_TYPES is True`.
    - `MIN_ROSTER_ROWS` is evaluated on the POST-exclusion count: a payload whose surviving common-stock rows fall below the floor raises even when the pre-exclusion count would have cleared it.
  </behavior>
  <read_first>
    acquisition/universe.py — `TiingoRosterFetcher` (class docstring, `fetch()`, `MIN_ROSTER_ROWS`), `NasdaqUniverseFetcher` (the D-02/A4 comment block), `USEquityUniverseFetcher` (the measured exchange-token table in its docstring — extend it, do not replace it).
    tests/conftest.py — `mock_universe_fetchers`, in particular the `nasdaq_csv` APPEND-ONLY comment and the two `monkeypatch.setattr(..., "MIN_ROSTER_ROWS", 1)` lines.
    tests/test_universe.py — `test_us_equity_fetcher_filters_nyse_nasdaq_amex_and_excludes_others`, `test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged`, `test_us_equity_roster_guard_rejects_a_drifted_filter`.
  </read_first>
  <action>
Add two module-level compiled patterns near `TiingoRosterFetcher`, carrying
the full measurement from `<measured_evidence>` in their comments (the counts,
the cross-check against the independent segment-split reference, and the
explicit statement that class shares such as `BRK-B` and `BF-B` are the trap
these patterns are shaped to avoid):

    _PREFERRED_SHARE_PATTERN = r"[-/]\s*P[A-Z]?\s*(?:[-/]|$)"
    _BABY_BOND_PATTERN = r"\s\d"

Add a class constant on `TiingoRosterFetcher`:

    EXCLUDE_NON_COMMON_SECURITY_TYPES: bool = False

**The default False is the entire mechanism protecting Locked Decision 1** —
`nasdaq_all`'s semantics are frozen and a shared unconditional filter would
have silently changed them. Say that in the constant's comment. Use a class
constant rather than an overridable method because this module's established
idiom is that a roster is DATA — "adding a roster is a data change, three
class constants, not a code change" (the `TiingoRosterFetcher` docstring) —
so the opt-in belongs in the same place `EXCHANGE_FILTER`, `MIN_ROSTER_ROWS`
and `CATEGORY` already live, and the criterion itself stays in exactly one
place instead of being duplicated per subclass.

In `fetch()`, after the existing exchange/assetType/priceCurrency filter and
BEFORE the rename and BEFORE the `MIN_ROSTER_ROWS` check, apply the exclusion
when the flag is set — filtering on the source's own `ticker` column, keeping
rows that match neither pattern. Log at INFO the number of rows removed and
the pre/post counts, so a future criterion change is visible in a refresh log
rather than only in the resulting parquet.

The BEFORE-`MIN_ROSTER_ROWS` ordering is a key link: the guard must validate
the count that actually gets persisted, not a pre-exclusion count. State it in
a comment. Note the measured headroom there too (15,173 surviving rows against
a floor of 8,000, 1.90x) so a reader can see the exclusion does not endanger
the guard.

Set `EXCLUDE_NON_COMMON_SECURITY_TYPES = True` on `USEquityUniverseFetcher`
and extend its docstring with:
  - the measured shape table and counts from `<measured_evidence>`;
  - the verified class-share survivors;
  - the 1,124 warrant/unit/right lines that deliberately REMAIN, named as a
    carried-forward finding rather than left implicit;
  - **an explicit statement of the resulting asymmetry with `nasdaq_all`**:
    `us_all` is no longer a strict superset of `nasdaq_all` for the preferred
    and baby-bond lines, because `nasdaq_all`'s semantics are frozen by
    Locked Decision A4 / D-02. Point a reader who wants to "fix" that
    inconsistency at this decision instead of at the code.

Extend `NasdaqUniverseFetcher`'s existing D-02 comment block with one
sentence recording that it deliberately does NOT opt in.

In `tests/conftest.py`, append preferred/baby-bond rows to the
`mock_universe_fetchers` roster CSV. **Every appended row must be NYSE (or
AMEX / NYSE MKT), never NASDAQ/Stock/USD** — the fixture's own APPEND-ONLY
comment explains why: `test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged`
pins an exact NASDAQ symbol set and a NASDAQ row would break it. This is the
single easiest thing to get wrong here. Include at least the doubled-delimiter,
slash, `PR`-with-space and leading-hyphen shapes plus two baby bonds, and at
least four class-share controls that must survive.

Add tests to `tests/test_universe.py`:
  - one exclusion test over `mock_universe_fetchers` asserting drops and
    survivals by set membership;
  - a class-share survival test using the real measured literals;
  - a standalone `nasdaq_all`-immunity test that does NOT use
    `mock_universe_fetchers`: build its own zip payload containing NASDAQ
    preferred rows (idiom: `test_us_equity_roster_guard_rejects_a_drifted_filter`),
    `monkeypatch.setattr` both fetchers' `MIN_ROSTER_ROWS` down, and assert
    that ONE payload yields kept-in-`nasdaq_all` / dropped-from-`us_all`, plus
    the two `EXCLUDE_NON_COMMON_SECURITY_TYPES` class-constant assertions;
  - a guard-ordering test: a payload whose post-exclusion count falls below a
    monkeypatched `MIN_ROSTER_ROWS` while the pre-exclusion count clears it
    must raise.
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run pytest tests/test_universe.py -q</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab &amp;&amp; uv run python -c "
import re
from acquisition.universe import _PREFERRED_SHARE_PATTERN as P, _BABY_BOND_PATTERN as B, NasdaqUniverseFetcher as N, USEquityUniverseFetcher as U
drop = ['AAM-P-A','ZB-P-F-CL','MTB-P','BC/PA','SCE--P-D','IMH-P--B','NYCB- PR-U','-P-HIZ','ASRV 8.45 06-30-28','SO 6.75 08-01-22','NEE 6.219','CHNG 6']
keep = ['AAPL','MSFT','BRK-A','BRK-B','BF-A','BF-B','PBR-A','HEI-A','MOG-A','LEN-B','CWEN-A','UA-C','MKC-V','AGM-A','CRD-A','LGF-A','GEF-B','STZ-B','UHAL-B','C-WS-A','GM-WS-B','MIMO-W-A','ACP-R-W','DGAC-UN']
hit = lambda t: bool(re.search(P,t)) or bool(re.search(B,t))
assert all(hit(t) for t in drop), [t for t in drop if not hit(t)]
assert not any(hit(t) for t in keep), [t for t in keep if hit(t)]
assert N.EXCLUDE_NON_COMMON_SECURITY_TYPES is False and U.EXCLUDE_NON_COMMON_SECURITY_TYPES is True
print('criterion OK')"</automated>
  </verify>
  <done>
    `tests/test_universe.py` passes with the new exclusion tests green and every
    pre-existing test in it still green, including
    `test_nasdaq_roster_exchange_filter_and_symbol_set_are_unchanged`.
    The two patterns drop all 12 measured non-common-stock literals and keep all
    24 measured common-stock / warrant literals.
    `USEquityUniverseFetcher` opts in; `NasdaqUniverseFetcher` does not.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| Wikipedia -> `_parse_changes_table` | Publicly editable HTML crosses into the membership reconstruction. No authentication, no schema contract, ongoing editor typos. |
| Tiingo `supported_tickers.zip` -> `TiingoRosterFetcher.fetch()` | Vendor-controlled CSV whose token vocabulary can drift without notice. |
| Both -> `universe.parquet` | `save()` overwrites the reference table in place; a bad build destroys the previous good roster. |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-eme-01 | Tampering | `IndexMembershipFetcher._parse_changes_table` | high | mitigate | Task 1: normalize delimiter residue, log every correction, raise on anything still malformed. A phantom ticker today silently erases a real multi-decade index member from the panel. |
| T-eme-02 | Tampering | `IndexMembershipFetcher._parse_changes_table` | medium | mitigate | Task 1: the normalization is announced by `logger.warning`, so a systematic parser regression is not laundered into "3 tickers were tidied". Silent correction is explicitly rejected (Locked Decision 3). |
| T-eme-03 | Denial of Service | `USEquityUniverseFetcher.fetch()` | high | mitigate | Task 2: an over-broad exclusion regex would silently delete real common stocks and reintroduce survivorship bias. Mitigated by pinning the criterion against the measured live directory and asserting 24 real class-share/mainstream literals survive. |
| T-eme-04 | Denial of Service | `TiingoRosterFetcher.fetch()` `MIN_ROSTER_ROWS` | medium | mitigate | Task 2: the exclusion runs BEFORE the guard so the guard sees the persisted count; measured headroom 15,173 vs 8,000. A guard-ordering test pins it. |
| T-eme-05 | Tampering | `NasdaqUniverseFetcher` | high | mitigate | Task 2: `EXCLUDE_NON_COMMON_SECURITY_TYPES` defaults to `False` on the base, and a dedicated same-payload test proves `nasdaq_all` keeps what `us_all` drops (Locked Decision 1). |

No package-manager installs are introduced by this plan, so no package
legitimacy gate applies.
</threat_model>

<verification>
1. `cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/ -q` — must
   report at least the 216 pre-existing passes plus the new tests, zero
   failures. (Bare `uv run pytest` is unusable: `backtest/test_strategy.py` has
   a PRE-EXISTING collection error, out of scope.)
2. Both per-task inline `uv run python -c` checks pass.
3. `git diff --stat` touches only `acquisition/universe.py`,
   `tests/conftest.py`, `tests/test_universe.py`.
4. Nothing under `/Volumes/SSD/data` is written, moved or deleted (Locked
   Decision 4). All tests are offline; `TIINGO_API_KEY` is not set and is not
   needed.
</verification>

<success_criteria>
- All `must_haves.truths` hold.
- `uv run pytest tests/ -q` is green with no regressions against the 216-test
  baseline.
- The `us_all` exclusion criterion in the source is the measured one, with its
  counts recorded in the code, not a guess.
- The `nasdaq_all` / `us_all` asymmetry is stated in `USEquityUniverseFetcher`'s
  docstring, pointing at Locked Decision A4 / D-02.
</success_criteria>

<post_execution_note>
**This plan changes the CONTENTS of `universe.parquet`. The fix does not reach
the user's data until they rebuild the reference table:**

```
cd /Users/daizhaorong/projects/quantlab && uv run python refresh_us_equity_universe.py
```

Expect the `us_all` category to shrink by roughly 940 symbols (15,425 ->
~14,485 distinct) and `JCP` / `ITT` to appear in `sp500_constituent` as real
symbols, with the duplicate `ALLE` gone.

**A changed `us_all` roster will make `ChunkLedger.assert_consistent` refuse an
incremental append to the existing Zarr store — a full store rebuild is
required.** That rebuild is ALREADY required for an unrelated reason (the
existing store covers 9,945 of 15,424 roster symbols), so this adds no new
burden; it is stated here so it is not discovered mid-run.

The S&P membership panel store must also be rebuilt for the phantom-ticker fix
to reach it.
</post_execution_note>

<out_of_scope>
- Warrants, units, rights and when-issued lines (1,124 measured) remain in
  `us_all`. Named, measured, deliberately carried forward.
- `nasdaq_all` semantics are unchanged (Locked Decision 1 / A4 / D-02).
- The open human-verify checkpoint in quick task 260906-26o (Task 3, legacy
  Tiingo watermark stamping) is untouched.
- `backtest/test_strategy.py`'s pre-existing collection error.
</out_of_scope>

<output>
Create `.planning/quick/260906-eme-roster-ticker-wikipedia-ticker-us-all/260906-eme-SUMMARY.md` when done
</output>
