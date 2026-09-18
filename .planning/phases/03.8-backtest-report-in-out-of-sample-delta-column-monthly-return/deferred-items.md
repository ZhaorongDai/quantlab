# Phase 03.8 — Deferred Items

超出当前计划范围、执行中发现但未修复的问题。

## 03.8-02

- **`_cell` 把非 `float` 子类的 numpy 非有限值渲染成字面 `nan` / `inf`**
  （`quantlab/utils/backtest_report.py::_cell`，早于本阶段即存在）。
  `_cell` 只对 `isinstance(value, float)` 做 `math.isfinite` 判断；`np.float32` /
  `np.float16` 是 `numbers.Real` 但不是 `float`，会落到 `str(value)` 分支，于是
  whole / in_sample / out_of_sample 三列里的 `np.float32("nan")` 显示为 `nan`，
  而不是破折号。同理，有限的 `np.float32` 也绕过了 `.6g` 格式化。
  delta 列不受影响：`_delta` 自己做 `math.isfinite`，且有 `np.float32` 用例锁住。
  目前真实运行的 metrics 都是 float64，所以尚未在报告中出现。修法建议：
  `_cell` 的浮点分支改为 `isinstance(value, numbers.Real)`（先排除 bool 与 int）。
