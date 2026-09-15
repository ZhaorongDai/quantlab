## Deferred Items

- Codebase maps still describe the Nautilus strategy deleted 2026-09-07 as current
  status: open
  **Found during:** 03.7-12 Task 2 (not caused by this plan; outside its backtest/vecbt scope)
  **What:** `.planning/codebase/STACK.md` lines 31 and 47 still list `backtest/test_strategy.py` as a live Nautilus trading-engine user, and `.planning/codebase/ARCHITECTURE.md` still carries the "Live Prediction Path (Nautilus Strategy)" data-flow section, the Factor/Label and Model layers' "Used by ... backtest/test_strategy.py" clauses, and the Error Handling strategy sentence about its try/except blocks. CLAUDE.md's copy of the stack section already records the deletion. 03.7-12 only annotated the `Test` (Strategy) component row, which sits in the table it rewrote.

- ARCHITECTURE.md error-handling example still claims ML training is unimplemented
  status: open
  **Found during:** 03.7-12 Task 2
  **What:** `.planning/codebase/ARCHITECTURE.md` "Patterns" under Error Handling cites `base/model.py:_auto_train` raising `NotImplementedError("ML training not implemented")`. `MLModel` and `XGBoostRegressor` train today. The D-37 fit guard this plan was asked to replace appears only in CLAUDE.md, so this sibling statement was left as found.

- CLAUDE.md state-management line mentions `self.predictions_history` "in the live strategy"
  status: open
  **Found during:** 03.7-12 Task 2
  **What:** CLAUDE.md Architecture > Data Flow still names an attribute of the deleted Nautilus strategy. It is generated from ARCHITECTURE.md, so fix it there together with the first item.
