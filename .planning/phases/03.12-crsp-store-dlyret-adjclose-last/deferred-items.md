# Phase 03.12 — Deferred Items

Out-of-scope discoveries logged during execution. Per the executor scope
boundary these are NOT fixed here: they are not caused by this phase's
changes.

---

## `StockDataset.to_kunquant()` never synthesises `amount` (D-02)

**Found during:** 03.12-01, plan-level verification step 3 (the D-04 guardrail
suite).

**Symptom:** three failures in `tests/test_factor_kunquant.py`, all
`KeyError: 'amount'`:

- `test_stock_to_kunquant_synthesizes_amount_as_adjusted_dollar_volume`
- `test_stock_to_kunquant_without_amount_leaves_arrays_unchanged`
- `test_alpha101_stock_bugfix_batch_cal_returns_xarray_dataset`

```
KeyError: "No variable named 'amount'. Variables on the dataset include
['adjClose', 'adjHigh', 'adjLow', 'adjOpen', 'adjVolume', ..., 'low', 'open',
'volume', 'symbol', 'timestamp']"
```

**Why it is out of scope:** `grep -rn amount quantlab/dataset/stock.py
quantlab/base/data.py` returns nothing — the D-02 synthesis the tests demand
does not exist anywhere in the shipping package. This phase's entire
production diff is seven lines inside
`quantlab/dataset/crsp/__init__.py` (`git diff --name-only <phase base> --
quantlab/` prints that one path), none of which is on the
`to_kunquant` code path. The failures predate 03.12.

`tests/test_backtest_run.py` and `tests/test_backtest_run_cv.py` — the other
half of the D-04 guardrail — are green, so the "factor / backtest untouched"
claim this phase makes still holds for everything that was passing before.

**Suggested home:** a `StockDataset.to_kunquant` fix in its own quick task or
phase; it is a factor-layer gap, not a CRSP one.
