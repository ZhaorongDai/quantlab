# 交接 TODO（2026-09-15 会话）

新窗口从这里接。所有已完成工作都已合并进 `main` 并推送。

## 一、本会话已完成（已推送）

| 内容 | 提交 | 说明 |
|---|---|---|
| 03.7 代码审查修复 | `a28a8e6` `fec1c0e` `f730dc8` | CR-01（XGBoost 特征重要性/inplace 预测崩溃丢 checkpoint）、WR-01（旧 checkpoint 按集合比对 factor_names）、WR-02（Alpha158Stock vwap 改复权典型价） |
| quick `260915-o5y` timer 日志 | `35a4601` | collect merge / to_array / fit_model / evaluate / align_and_predict / simulate 六处计时 |
| quick `260915-ocw` 截面 Z 值算子 | `6a8f8dd` | `quantlab/my_ops/preprocess.py:CrossSectionalZScore`，`GenericCrossSectionalOp` + C++ 循环体；未接入任何因子类 |
| quick `260915-p91` 股票池过滤 | `0ae8e8d` | `quantlab/factor/universe_filter.py:UniverseFilteredFactor`，verifier 11/11 通过 |

基线：22 个测试文件 391 通过 / 0 失败（3 个已知 `amount` 失败被 deselect）。

## 二、进行中：quick `260915-sxx` 回测报告优化

**计划已写好，未提交、未执行**：`.planning/quick/260915-sxx-improve-the-backtest-html-report-dates-m/260915-sxx-PLAN.md`（3 个任务，tracer 优先）。

要做的事：报告页面加上回测日期、训练窗口、样本内外区间；指标表（通用渲染）；净值图带对数/线性切换；回撤；月度收益柱状图；强平标记。图表 trace 集合为 `equity` / `drawdown` / `monthly_return` / `liquidation`。

planner 的关键发现（已并入计划）：
- 报告写入在 `_persist_run_dir` 的暂存闭包里，**报告抛异常会删掉整个运行目录**，所以指标表必须按实际键通用渲染，绝不能硬编码指标名。
- 运行目录的文件集合在 4 处被断言为**精确集合**（含 `tests/test_universe_filtered_factor.py:932`），报告必须保持单文件，不能产生附属资源。
- `_report_traces` 用 `trace["name"]` 建字典，未命名 trace 会 KeyError，所以指标表必须是 HTML 而非 `go.Table`。
- 两处 `write_backtest_report` 的 monkeypatch 桩已经是 `*args, **kwargs`，改签名不会破坏它们。

判断项（可推翻）：净值 y 轴保留原始金额（倍数放进 hover），默认线性轴。

**执行步骤**（沿用本会话流程）：
1. `git add` 计划文件 → 提交 → `git push`（不推送则 worktree 隔离会自动降级）。
2. `gsd-tools query worktree.base-check --mode harness-worktree --pick shouldDegrade` 确认为 false。
3. `gsd-tools query dispatch-isolation --raw --force-isolation harness-worktree` 写 sentinel（必须是派发前最后一次 dispatch-isolation 调用）。
4. 派 `gsd-executor`（`isolation: worktree`），prompt 内嵌 worktree 分支校验（基准 = 计划提交，备选 = 其父提交）。
5. 完成后：拷出 SUMMARY → 写 worktree 清单 → `worktree.cleanup-wave` 合并清理 → 更新 STATE.md → 提交文档 → push。

## 三、待办（按建议优先级）

### 1. 重跑回测，验证股票池过滤效果（最高优先，在 Linux 上做）
`git pull` 后改 `test.py`，把因子和标签都包上包装器，**两边参数必须一致**：

```python
from quantlab.factor.universe_filter import UniverseFilteredFactor
UNIVERSE = dict(min_price=5.0, min_dollar_volume=1e6, window=20)
factor = UniverseFilteredFactor(alpha101(), **UNIVERSE)
label  = UniverseFilteredFactor(Return(label_config), **UNIVERSE)
# price_dataset 仍传未过滤的原始 dataset
```
参数不一致的危险方向：标签在池内、因子不在时，特征全 NaN 但标签有效，XGBoost 只看标签，垃圾股极端收益会重新进入损失函数。

建议同时把 `rebalance_periods` 调小（掉出股票池的持仓要等到下个调仓 bar 才卖出）。对照上一次结果：+886,077,331%、最大回撤 98.6%、`best_iteration=0`、`val-rmse` 单调上升。

### 2. 回测引擎缺陷修复（审计结论，建议顺序 F1 → F2 → F3）

- **F1（阻塞）`engine_vectorbt.py:265`**：判定退市只看下一个 bar 的成交价是否 NaN，没检查之后还有没有价格。停牌几天再复牌会被当成退市，按停牌前最后价格卖出。复现：停牌 3 个 bar、复牌价从 1.00 掉到 0.10，净值全程不动，90% 亏损没吃到，还会重新入选造成来回进出。修法：强平前加"之后是否还有有限价格"的判断。测试缺口：只有永久退市和前导 NaN 的用例。
- **F2（阻塞）`engine_vectorbt.py:103/108`**：成交价和估值价都做 ffill。离场价系统性偏乐观；估值冻结让冻结期收益恒为 0，压低波动和回撤，抬高 Sharpe/Sortino/Calmar。修法：退市离场价打折或设上限；考虑估值列不做 ffill。
- **F3 → 按用户决定改为"删除"**：换手率是自研指标，vectorbt 没有对应实现，**指标只用 vectorbt 提供的**。删掉 `BaseBacktester._turnover` / `_turnover_summary`，`metrics.json` 不再有 `turnover`、`traded_notional` 等自定义聚合；可改用 vectorbt 的 `gross_exposure`、`net_exposure`、`total_fees_paid`。连带要改：**D-22 作废并改写 `example/backtest.md`**、`tests/test_backtest_metrics.py:546` 等相关断言。
  - 补充事实：观测到的 `turnover=5.37` 是分母用错（用了上一 bar 收盘净值，而 vectorbt 按当前 bar 成交价计的组合净值定仓），引擎实际只交易了 2 倍净值，**不是引擎多交易**。
- **F4（警告）`us_equity.py:53`**：t 日可选条件要求 t+1 有成交价，属前视；文档记为 D-12 有意为之，建议在文档中明确标注为前视而非中性条件。
- **提示级**：`mean_per_rebalance` 只对有成交的 bar 取平均；`run_cv` 拼接在折边界打乱调仓节奏；换手用了含滑点成交价；敞口断言只管目标权重。

审计确认**正确**的部分：D-05 t+1 开盘成交、`targetpercent` 以成交价计净值为基准且无前视、NaN=持有/0=清仓、混合行防护、年化口径（含 CR-02 修正）、D-17 按精确 bar 切分、平局打破、池外标的不可选、最后一个 bar 不调仓；Sharpe 等比率取自 vectorbt。

**根因教训**：这些 bug 逃过测试是因为夹具太干净（恒定价格、无跳空、无停牌洞）。修复时应补脏数据夹具：跳空、停牌后复牌暴跌、真退市、单点缺价。

### 3. 标签处理（解决训练不收敛）
现象：`train-rmse` 单调下降、`val-rmse` 从第 1 轮起单调上升、`best_iteration=0`、`best_score=6.13`（相当于 600% 量级）。原因三条叠加：验证集是训练窗口按时间切的最后 20%（`val_size=0.2`，不同市场阶段）；标签被极端值主导；RMSE 衡量数值而策略只需要排序。

先看那次 run 的 W&B summary 里的 `val_ic` / `val_rank_ic`：明显为正说明只是判据用错，接近 0 或为负说明确实没学到。

建议：股票池过滤（已做）→ 标签换成截面排序或截面 Z 值（`CrossSectionalZScore` 已可用）或截尾 → `hyperparameters={"objective": "reg:pseudohubererror"}` → 最后才考虑改早停判据（需要自定义评估函数，要动模型层）。

### 4. 杂项
- `tests/test_factor_hierarchy.py` 在当前代码上导入失败（`cannot import name 'SpotReturn' from quantlab.label.fret`），本会话之前就存在，已排除出门禁，需单独修。
- `example/universe.md` 补一行"因子与标签的股票池参数必须一致"，并可加便捷入口 `UniverseFilteredFactor.wrap_all([...], **UNIVERSE)` 从源头杜绝写歪。
- 过时描述：`CLAUDE.md` 组件表只列 `WindowedZScore`；`tests/test_factor_kunquant.py` 里 normalization matrix 测试的 docstring 仍说没有截面 Z 算子；`example/factor.md` 常见坑第 1 条只说 batch 模式有 SIMD 对齐限制（stream 也有）。

## 四、已锁定的设计决策（不要重新讨论）

1. **股票池过滤 = Factor 包装器**，不是 Dataset 包装器，也不加配置字段；模型层和回测器不动。
2. **掩码切点**：截面算子的**输入**按 t 日掩码遮盖（KunQuant 图改写，`Div(v, mask)`）；时序算子用完整历史；因子输出和标签按各自时间戳 t 遮盖（标签用未过滤价格算，绝不用 t+h 的池子状态）。
3. **symbol 删除规则**：只删代码规则排除的（权证、单元、配股权、测试代码）；价格或成交额不达标的保留成 NaN 列，保证 symbol 轴与日期窗口无关，DL 模型可用。
4. **已接受的代价**：DL 模型训练时池外格子被 `nan_to_num` 填 0 参与训练（`DLModel._fit` 对标签也做 `_preprocess`），预测不受影响；修需要动模型层，暂不做。
5. **指标只用 vectorbt 提供的**（本会话最后决定，见 F3）。
6. **回测器不改**：持仓掉出股票池后，在下一个调仓 bar 不可选，按次日真实开盘价卖出。

## 五、环境与流程注意事项

- 包管理 `uv`；测试 `uv run pytest ...`。
- **禁止 `git stash`**；红证据用 `git show HEAD:file` + 临时副本。
- worktree 隔离要求 `origin/HEAD` 与本地一致，**每次派发前先 push**，否则自动降级为主工作区顺序执行。
- 派发执行 agent 前，`dispatch-isolation --force-isolation` 必须是最后一次 dispatch-isolation 调用，之后立刻 `Agent()`。
- macOS 同进程混用 torch 与 xgboost 需 `OMP_NUM_THREADS=1`（`tests/conftest.py` 已设）。
- KunQuant 在 aarch64（Mac）上要求标的数为 8 的倍数；`runGraph` 的 start 必须为 0（0.1.11 的截面算子在 start>0 时结果错误）。
- 本会话的原型脚本在 scratchpad（`cszscore/universe_rewrite.py` 等），**会随会话清理**，需要的话从本文件的描述重建。
