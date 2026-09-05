# Phase 3: Factor Computation (KunQuant + Polars) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-09-05
**Phase:** 03-factor-computation-kunquant-polars
**Areas discussed:** Alpha158 market coverage, amount-field approximation for US equities, new Polars factor engine architecture

---

## Alpha158 market coverage

| Option | Description | Selected |
|--------|-------------|----------|
| 同时新增 Alpha158Stock 支持美股 | Mirror the existing Alpha101SpotKline/Alpha101Stock dual-market pattern | ✓ |
| 只沿用现有 Alpha158SpotKline（币安） | No US-equity extension this phase | |

**User's choice (free text):** "alpha158拓展到美股非常简单，调用的步骤完全一致。现有的 alpha158 和 alpha101 在美股和币安都可以使用，区别可能是是否使用一些截面因子"
**Notes:** User considers this a low-risk, mechanical extension. Flagged (not blocking) that some Alpha158 factors may be cross-sectional and might need per-market handling differences — left to planner/researcher discretion.

---

## amount field for Alpha158Stock's VWAP features

| Option | Description | Selected |
|--------|-------------|----------|
| 用 volume × close 近似 amount | Standard dollar-volume proxy | ✓ |
| 跳过需要 amount 的特征 | Drop VWAP-dependent features for the Stock variant | |

**User's choice:** volume × close approximation.

---

## New Polars factor engine architecture

**User's choice (free text, extensive):** "polars 的因子实现一个类，参考FactorKunQuant，我需要这两个因子类可以无缝替换。实现因子类后，在 factor 文件夹中加一个简单的 polars 计算因子。注意，Polars 不会返回Function()和因子名称，所以要做针对性的设计，比如提取 polars dataframe 的属于因子的列名。polars 因子计算要使用 lazyframe，我的想法是用户在类似_get_factor_func()的函数中编写因子，返回一个只有 symbol date 因子列的 polars lazyframe，然后当用户调用 cal()的时候开始计算。"

**Notes:** This substantially deepened the original "add a Polars batch factor backend" scope item into a concrete architecture requirement: extract a shared `Factor` ABC from `FactorKunQuant` so `FactorKunQuant` and a new `FactorPolars` are drop-in interchangeable (captured as CONTEXT.md D-03 through D-08). This is a refactor of existing core architecture, not purely additive — flagged clearly in CONTEXT.md's canonical refs so research/planning treat `base/factor.py`'s refactor with appropriate care (must not change behavior for existing Alpha101/158 classes).

---

## Claude's Discretion

- Exact naming of the new shared `Factor` ABC and `FactorPolars`'s abstract method.
- Cross-sectional-factor per-market handling differences for Alpha158 (if any).
- The exact formula for the one example Polars-computed factor.
- Class-hierarchy placement of KunQuant-streaming-specific methods relative to the new shared base.

## Deferred Ideas

None — discussion stayed within phase scope.
