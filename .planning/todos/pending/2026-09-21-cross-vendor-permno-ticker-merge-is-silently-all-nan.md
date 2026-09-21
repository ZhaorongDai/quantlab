---
created: "2026-09-21T00:00:00.000Z"
title: Merging a PERMNO-axis CRSP panel with a ticker-axis vendor panel yields a silent all-NaN backtest instead of a refusal
area: data / model / backtest
severity: major
status: pending
resolves_phase: null  # deliberately NOT 03.11 — D-21 was answered `backlog` by the operator
triggers: >
  any model or backtest whose `factors` / `labels` mix a CRSP source (panel `symbol` = int64 PERMNO,
  since Phase 03.11) with a ticker-axis source (Tiingo / Alpaca / Binance, which keep the ticker axis
  by D-02). Silent — nothing raises, no NaN guard fires, and the run completes with an empty
  portfolio that reads as "the strategy found nothing to hold".
files:
  - quantlab/base/model.py (:218, :239, :248 — the three `xr.combine_by_coords` calls)
  - quantlab/base/backtest.py (:782-784 — `predictions.reindex(timestamp=..., symbol=prices.symbol.values)`)
  - quantlab/backtest/selection.py (:162-166 — the single `logger.warning` that is the only outward sign)
---

# A PERMNO panel merged with a ticker panel is silently empty

Filed by Phase 03.11 plan 11, Task 2 (`checkpoint:decision` D-21). The operator answered
**`backlog`**: do not implement the assertion inside 03.11, whose scope is the identity-axis
migration itself. This file exists so the next person does not have to rediscover the mechanism.

## The failure chain, end to end

Phase 03.11 moved the CRSP price panel's `symbol` dimension to the **int64 PERMNO**. D-02 deliberately
left every non-CRSP vendor on the **ticker** axis — Binance will never have a PERMNO, and routing
Tiingo/Alpaca through a CRSP mapping would make them downstream dependencies of CRSP and import an
as-of-date correctness risk. That decision is sound, and this todo does not reopen it.

The consequence is a type collision that no layer checks:

1. **`quantlab/base/model.py:218` / `:239` / `:248` — `xr.combine_by_coords(all_ds)`.**
   Given one dataset indexed by `symbol = [10104, 10107, ...]` (int64) and another indexed by
   `symbol = ["AAPL", "MSFT", ...]` (str), `combine_by_coords` performs an **outer** join. It does
   not raise on the dtype difference; it produces a union axis whose two halves never intersect, so
   every feature cell is NaN wherever the label is present and vice versa.

2. **`quantlab/base/backtest.py:782-784` — `predictions.reindex(timestamp=..., symbol=prices.symbol.values)`.**
   The comment there states the intended contract: "预测铺到价格数据集的全部标的上：缺的标的是 NaN,
   也就不可选 (D-06)". That contract is correct for a genuinely missing symbol. Under an axis-type
   mismatch **every** symbol is missing, so the reindex produces an all-NaN prediction panel and the
   backtest simulates an empty portfolio.

3. **`quantlab/backtest/selection.py:162-166`** logs `only {k} eligible symbol(s) per book for
   top_n=...`. That lone warning is the *only* outward sign, and it is indistinguishable from the
   ordinary, legitimate case of a thin universe on an early bar.

Net effect: a run that is structurally meaningless completes successfully and reports a flat equity
curve. There is no exception, no dtype check, and no NaN-density gate anywhere on the path.

> Line numbers re-measured against `1c8203e` on 2026-09-21. `03.11-11-PLAN.md`'s context block cites
> `model.py:207/:228/:237` and `backtest.py:756-757`; those had already drifted. Anchor by content
> (`grep -n 'combine_by_coords' quantlab/base/model.py`, `grep -n 'reindex' quantlab/base/backtest.py`),
> not by line number.

## The original ruling this comes from

`03.11-CONTEXT.md` **D-02** locks "CRSP only, no cross-vendor identity work this round" and records
the all-NaN outcome as a **known and accepted** consequence. Its closing sentence is the reason this
file exists rather than a patch:

> "A few-line dtype/intersection assertion at those two merge points would turn silence into a loud
> refusal. **Optional task — propose it, let the operator decide; do not assume it is wanted.**"

Proposed at the 03.11-11 Task 2 checkpoint; operator chose `backlog`. Phase scope stays pure and no
legitimate workflow is at risk of a false rejection today.

## The unsolved part — why this is not just "add two asserts"

A naive `assert predictions.symbol.dtype == prices.symbol.dtype` (or an intersection-emptiness check)
would also reject **legitimate partial overlap**. A caller may deliberately run on a subset of the
intersection — a sector slice, a liquidity screen, a deliberately narrowed roster — and a strict
guard turns that into a crash. So whoever implements this owes an **escape hatch design**, not just
the assertion:

- What is the refusal keyed on — dtype mismatch (narrow, catches exactly this bug), or
  intersection emptiness (broader, but fires on legitimately disjoint slices too)?
- Where does the opt-out live, and how does it avoid becoming the flag everyone sets by reflex?
- Is an empty intersection *always* wrong, or only wrong when the two sides' dtypes differ? (Likely
  the latter: same-dtype-but-disjoint is a roster question, different-dtype is a category error.)

A dtype-keyed refusal with no opt-out is the narrowest version and may be sufficient — a str axis and
an int64 axis can never be two slices of one universe. That is worth evaluating first.

## Shape to follow when it is implemented

`quantlab/dataset/backend.py:470-505` is the house guard idiom: a message that opens with what is
being refused, carries the **measured** evidence (both dtypes, both axis lengths, the intersection
size), and names the remedy. Estimated ~10-20 lines plus two tests.

## Notes

- Does **not** block Phase 03.11; 03.11 closed with this explicitly deferred.
- Does not block any current workflow either: no shipped code merges a CRSP panel with a ticker-axis
  panel today. The risk is the day someone writes that config for the first time.
