---
phase: quick-260914-lno
plan: 01
subsystem: model
status: complete
tags: [model-layer, xgboost, cross-validation, metrics, macos-openmp]
requires: []
provides:
  - "BaseModel / DLModel / MLModel 三层模型类，公开 train/train_cv/load/predict 只实现一份"
  - "BaseModel._cv_folds：全仓唯一的折边界算术；train_cv 返回逐折结果并写 {cls}_cv_summary run"
  - "XGBoostRegressor：原生早停、逐轮 wandb、sklearn 别名归一化、resolved_hyperparameters 记录"
  - "quantlab/utils/metrics.py：向量化截面 IC / RankIC 与误差指标"
affects: [quantlab/base/model.py, quantlab/dl_model, quantlab/ml_model, quantlab/utils/module.py, tests/conftest.py]
tech-stack:
  added: [scipy（直接声明）, xgboost（首个调用点）]
  patterns: [按框架拆变体 + 抽象类属性 config_cls/checkpoint_suffix, 重构前先捕获 golden, macOS 测试进程单线程 OpenMP 守卫]
key-files:
  created:
    - quantlab/utils/metrics.py
    - quantlab/ml_model/xgb.py
    - tests/test_model_cv.py
    - tests/test_model_hierarchy.py
    - tests/test_ml_models.py
    - tests/test_metrics.py
    - tests/test_xgb_model.py
    - tests/test_macos_openmp_guard.py
  modified:
    - quantlab/base/model.py
    - quantlab/base/config.py
    - quantlab/utils/module.py
    - quantlab/ml_model/backend.py
    - quantlab/dl_model/mlp.py
    - quantlab/dl_model/rnn.py
    - quantlab/dl_model/rnn_classification.py
    - pyproject.toml
    - uv.lock
    - tests/conftest.py
    - tests/test_model_layer.py
    - tests/test_ml_backend.py
    - example/model.md
    - example/backend.md
    - example/README.md
    - example/factor.md
    - README.md
    - CLAUDE.md
    - train_model.py
decisions:
  - "R-2：_train_dl / _predict_nn 调用改名为 _fit / _predict，不留别名；_auto_train 直接删除"
  - "R2-2：DL 早停保持内联原样（DLModel._fit 就是原 _train_dl 函数体），DL 语义零改动"
  - "R3-4：CV 均值写进独立的 {cls}_cv_summary run，因为每折 _fit 结束时已 finish 自己的 run"
  - "macOS torch/xgboost OpenMP 冲突：tests/conftest.py 在一切 import 之前于 darwin 上 setdefault OMP_NUM_THREADS=1；ctypes 预加载 Homebrew libomp 方案实测会让 torch 自身段错误，已否决"
  - "sklearn 别名在合并默认参数之前于用户字典上归一化，别名与原生键同时出现抛 ValueError"
  - "实际生效超参写入 config.json 顶层 resolved_hyperparameters 与 wandb config；加载器只丢弃这一个键"
metrics:
  duration: "3h22m（2026-09-14T20:21:32Z → 23:43:43Z，含两次检查点等待）"
  completed: 2026-09-14
  tasks: 3
  files: 27
plan_head_before: e03b75f70cd9d462b89b8911795c865e0cc371f2
actuals:
  tokens: 61708
  tasks: 3
  commits: 6
---

# Phase quick-260914-lno Plan 01: 模型层三层拆分 + XGBoost 回归头 + ML 交叉验证 Summary

模型层拆成框架无关的 `BaseModel` 与 torch 变体 `DLModel`、numpy/树模型变体 `MLModel` 三层，共用一个 CV 折生成器；新增用 xgboost 原生早停预测未来收益的 `XGBoostRegressor`（含 sklearn 别名归一化与实际生效超参记录）、向量化截面 IC/RankIC 指标模块，以及让 macOS 上 torch 与 xgboost 能在同一测试进程共存的 OpenMP 守卫。

## 提交

| 任务 | 提交 | 说明 |
|---|---|---|
| 1（A2 golden） | `5d9a605` | `test(260914-lno): pin DL train_cv fold geometry before extraction`——早于任何生产代码改动，只含 `tests/test_model_cv.py` |
| 1 | `bea3712` | `feat(260914-lno): split BaseModel into DLModel/MLModel with a shared public interface and one CV fold generator` |
| 1 | `d549c06` | `test(260914-lno): lock model hierarchy contracts, MLModel orchestration and panel metrics` |
| 2 | `89a7aec` | `feat(260914-lno): add XGBoostRegressor with native early stopping and CV support` |
| 2 | `d6a0b3e` | `test(260914-lno): cover XGBoostRegressor and guard macOS torch/xgboost OpenMP clash` |
| 3 | `5b01c65` | `docs(260914-lno): document the model hierarchy, native early stopping and ML cross-validation` |

基线 SHA：`e03b75f70cd9d462b89b8911795c865e0cc371f2`（钉在 `.git/quick-260914-lno-base`）。

## 测试结果（前后对比）

全量命令（基线与最终用同一组参数，原因见偏差 1、2）：

```
uv run pytest -q -p no:cacheprovider -rf --continue-on-collection-errors \
  --deselect tests/test_factor_stream.py::test_init_stream_binds_a_buffer_handle_for_every_declared_name
```

| 时点 | 汇总行 |
|---|---|
| 基线（任何改动之前） | `56 failed, 726 passed, 1 skipped, 1 deselected, 521 warnings, 1 error in 139.48s (0:02:19)` |
| Task 1 之后 | `56 failed, 790 passed, 1 skipped, 1 deselected, 542 warnings, 1 error in 114.53s (0:01:54)` |
| Task 2 之后（含 macOS 守卫） | `56 failed, 826 passed, 1 skipped, 1 deselected, 542 warnings, 1 error in 103.25s (0:01:43)`，0 次超时、0 次 Fatal Python error |
| 最终（Task 3 之后） | `56 failed, 826 passed, 1 skipped, 1 deselected, 543 warnings, 1 error in 103.73s (0:01:43)`，0 次超时、0 次 Fatal Python error，NO-NEW-FAILURES |

- 基线失败集合 57 条（56 FAILED + 1 个收集 ERROR），全部在模型层之外：`test_ingest_shells.py` 14、`test_ingest_tiingo_universe_wiring.py` 11、`test_data_dir_cli.py` 10、`test_ingest_conversion_gate.py` 6、`test_volume_guard.py` 5、`test_factor_kunquant.py` 4、`test_chunked_ingest.py` 2、`test_spot_dataset.py` 2、`test_entry_point_contracts.py` 1、`test_universe.py` 1，以及 `tests/test_factor_hierarchy.py` 的收集错误。
- 最终失败集合相对基线**没有新增**（`comm -13` 为空），passed 严格增加。
- 规划时记录的「58 failed / 736 passed」与本次基线不同：规划后你提交了 2e2ea12（标签改名），导致 `tests/test_factor_hierarchy.py` 收集失败，整份文件的 11 个测试从 passed/failed 计数里消失。
- 回归四件套：`tests/test_model_layer.py` 只改 3 行（numstat `3 3`），`tests/test_dl_models.py` 零字节改动，二者全绿。
- 顺序无关性（此前会段错误/死锁的组合）：`uv run pytest tests/test_dl_models.py tests/test_xgb_model.py tests/test_model_layer.py` → `61 passed`；反序 `tests/test_xgb_model.py tests/test_model_layer.py tests/test_dl_models.py` → `61 passed`。
- `tests/test_factor_hierarchy.py`（含对 `base/model.py` 的按行扫描）无法正常收集；用一次性脚本在内存里补 `SpotReturn = Return` 别名后运行：10 passed、1 failed，失败的是测试文件自身的 `NameError: Return`（基线即如此），按行扫描测试通过。
- 各门：DIFF-GATES-OK、DEP-DOC-GATES-OK、STALE-GATES-OK、POSITIVE-GATES-OK、PROSE-ONLY-OK、NO-NEW-FAILURES 均有输出。
- uv.lock：相对基线只多了两行 `{ name = "scipy" }` / `{ name = "scipy", specifier = ">=1.18.1" }`，`^[-+]version = ` 行变化数为 0。

### TDD 红证据

在基线 `e03b75f` 的 `git archive` 副本上运行 Task 1 的四个新测试文件（真实运行，非手写）：`4 errors`，均为 `ImportError: cannot import name 'DLModel' from 'quantlab.base.model'`。golden 两条则在未改动代码上先跑绿（`2 passed`）再单独提交。

## 重新核实的 xgboost 3.4.1 事实（Task 2 第 0 步探针原样输出）

```
xgboost 3.4.1
patience=5 order=es_first: rounds=1 best_iteration=0 best_score=1.0367233764685897 (float) recorder_steps=0..4 n=5 last_keys=['train-rmse', 'val-rmse'] actual_last_iter==best+patience: False
patience=5 order=rec_first: rounds=1 best_iteration=0 best_score=1.0367233764685897 (float) recorder_steps=0..5 n=6 last_keys=['train-rmse', 'val-rmse'] actual_last_iter==best+patience: True
  joblib roundtrip: rounds=1 best_iteration=0 best_score=1.0367233764685897 same_pred=True
patience=10 order=es_first: rounds=1 best_iteration=0 best_score=1.0367233764685897 (float) recorder_steps=0..9 n=10 last_keys=['train-rmse', 'val-rmse'] actual_last_iter==best+patience: False
patience=10 order=rec_first: rounds=1 best_iteration=0 best_score=1.0367233764685897 (float) recorder_steps=0..10 n=11 last_keys=['train-rmse', 'val-rmse'] actual_last_iter==best+patience: True
no early stopping: rounds 30 has best_iteration attr: False
save_config max_depth: '2'
save_config nthread: '1'
save_config nthread when not passed: '0'
two labels predict shape: (500, 2)
(n,1) label predict shape: (500,)
nan feature ok rounds: 3
inf feature raises: XGBoostError
empty inplace_predict shape: (0, 0)
```

补充探针：
- 早停启用但一直跑满轮数时同样截断：`signal, run to budget: rounds 60 best_iteration 59 rounds==best+1 True`。
- Booster 配置里 `eta`、`alpha`、`lambda` 在 `tree_train_param`（`eta` 存成 `'0.300000012'`），`seed`、`nthread` 在 `generic_param`，值都是字符串。
- `WANDB_MODE=disabled` 下 `NoopRun.config.update(..., allow_val_change=True)` 可连续调用。
- `EarlyStopping` 的 `metric_name=None` 解析为 `list(data_log.keys())[-1]`（`xgboost/callback.py` 第 482 行）。

与规划记录的差异：无实质差异。规划写的「触发停止那一轮被漏掉」对应的是回调排在 EarlyStopping 之后；实测确认实际最后一轮 = `best_iteration + patience`，并据此写成断言。

## 关键决定

- **R-2（调用改名，不留别名）**：`_train_dl` → `DLModel._fit`，`_predict_nn` → `DLModel._predict`，`_auto_train` 删除。一个方法两个活名字，正是日后读者会「修」错的歧义。
- **R2-2（DL 早停保持内联）**：`DLModel._fit` 就是原 `_train_dl` 函数体，唯一改动是 backtest 守卫消息点名 `_fit(backtest=True)`；三条 WR-02 测试与 patience 测试一行不改。
- **R3-4（均值写进独立 run）**：各折 `test_*` 均值写入 `{cls}_cv_summary` run 的 summary，因为每折 `_fit` 结束时已 finish 自己的 run。

## 范围追加（用户批准）

1. **sklearn 风格别名归一化**（`quantlab/ml_model/xgb.py:_PARAM_ALIASES`）：`n_estimators→num_boost_round`、`learning_rate→eta`、`random_state→seed`、`n_jobs→nthread`、`reg_alpha→alpha`、`reg_lambda→lambda`。在用户字典副本上、取出 `num_boost_round` 与合并 `DEFAULT_PARAMS` 之前完成，所以别名能覆盖默认值；别名与原生键同时出现抛 `ValueError` 并点名两个键。测试断言在训练出的 Booster 的 `save_config()` / `num_boosted_rounds()` 上，并确认 `config.hyperparameters` 未被修改。
2. **记录实际生效超参**：`MLModel._resolved_hyperparameters()` 钩子（默认 None），`XGBoostRegressor` 返回合并后的参数加 `num_boost_round`。`MLModel._fit` 在解析后写入 wandb run config（`allow_val_change=True`），`MLModel.get_config` 在 `config.json` 顶层写 `resolved_hyperparameters`；`load_model_from_config` 只丢弃这一个键（其他未知键仍 `TypeError`）；DL 的 `config.json` 不含该键（测试锁定）。
3. **macOS OpenMP 守卫**（检查点决定，选项 A）：`tests/conftest.py` 第一段可执行代码在 darwin 上 `os.environ.setdefault("OMP_NUM_THREADS", "1")`，排在一切 import 之前；Linux 不受影响。由 `tests/test_macos_openmp_guard.py` 锁定：守卫位置（AST）、非 macOS 不生效、不覆盖显式值、真实规模混合序列（GRU + 2 万行 cross_entropy → 5 万×30 的 xgb.train → torch → 线程并行 xgb.train → torch）子进程 exit 0。`quantlab/__init__.py` 保持为空。

## 与计划的偏差

### 自动修复/调整

**1. [Rule 3 - 阻塞] 基线命令加 `--continue-on-collection-errors`**
- 发现于：Task 1 步骤 A
- 问题：你在 2e2ea12 改了标签名，`tests/test_factor_hierarchy.py:52` 仍导入不存在的 `SpotReturn`，计划原命令在收集阶段整体中断（`1 error`，0 个测试运行）。
- 处理：基线、中间与最终全量运行统一加该参数；门的比较口径不变。

**2. [Rule 3 - 阻塞] 排除一个挂起的测试**
- 问题：`tests/test_factor_stream.py::test_init_stream_binds_a_buffer_handle_for_every_declared_name` 单独运行也稳定挂住（0% CPU，180 秒不结束），同文件另一个流式测试单跑 2 秒通过。
- 处理：所有全量运行用 `--deselect` 排除；未修改该测试（不在范围内）。

**3. [Rule 1 - Bug] CV 均值排除日期键**
- 问题：逐折结果里的 `test_start` / `test_end` 也以 `test_` 开头。
- 处理：`BaseModel._cv_mean_metrics` 只对非日期、数值型的 `test_*` 求均值；「DL 结果不含 `test_` 键」改为「不含指标键」断言。提交 `bea3712`。

**4. [调整] 提交划分**
- `tests/test_model_layer.py` 与 `tests/test_model_cv.py` 的基类改名行随 feat 提交（`bea3712`），保证每个提交独立为绿。

**5. [Rule 4 → 检查点] macOS 上 torch 与 xgboost 的 OpenMP 冲突**
- 第一次检查点：先 torch 后 xgboost 训练段错误（`OMP: Error #179`）。用户先选 ctypes 预加载 Homebrew libomp。
- 第二次检查点：实测预加载会让 torch 自身的 GRU / cross_entropy 段错误（与 xgboost 无关），且「先 xgboost 后 torch」无论是否预加载都死锁；预加载方案已撤回，未提交。用户改选单线程 OpenMP 守卫（见范围追加 3）。
- 未写「故意段错误」的测试；无守卫时的对照实测（exit -11）记在测试 docstring 里。

**6. [Rule 1 - Bug，测试自身] 守卫顺序测试的节点比较**
- 问题：`test_guard_is_the_first_executable_code_in_conftest` 初版分两次解析 conftest，跨解析树比较 AST 节点必然失败。
- 处理：同一次解析内定位；修复后 7 passed。提交 `d6a0b3e`。

**7. [文档] TinyRegressor 示例重跑**
- 旧输出第一行来自已删除的 `_auto_train`，已不再出现；`train_loss` / `test_loss` 逐位相同，`val_loss` 数值与 2026-09-07 的旧输出不同，原因未追查，文档如实注明。

## 示例的实际运行命令

两个示例脚本都放在仓库外的 scratchpad 目录运行，未提交；`example/model.md` 里的代码块与实际执行的脚本逐字节一致（已校验）：

```
WANDB_MODE=disabled OMP_NUM_THREADS=1 uv run python <scratchpad>/min_model.py
WANDB_MODE=disabled OMP_NUM_THREADS=1 uv run python <scratchpad>/min_xgb_model.py
```

XGBoost 示例实测：`rounds kept: 80 | best_iteration: 79`，测试 IC / RankIC `0.8811 / 0.8663`，8 折 CV 平均 `test_ic` 0.8878。真实数据用法（MLConfig + `Return`）依赖本机不存在的行情数据，文档标注「此例未实际运行」。

## 不在范围内的旧标签名引用（你正在进行的改名，未改动）

2e2ea12 把 `label/spot.py` 改名为 `label/fret.py`（`SpotReturn`→`Return`，`SpotBinaryReturn`→`BinaryReturn`），以下位置仍是旧名字：

- `quantlab/factor/momentum.py:35`（docstring 提到 `label/spot.py:SpotReturn`）
- `tests/test_factor_hierarchy.py:52, 175, 178, 199`（第 52 行的导入导致整份文件收集失败）
- `tests/test_factor_update.py:357`（docstring 提到 `SpotBinaryReturn`）
- `example/factor.md:5, 29, 112, 287, 289, 292`
- `example/model.md:331, 333-335`（「简单用法」一节抄自 `train_model.py` 的示例，本次未重写该节）
- `train_model.py:16, 19, 22, 25`
- `cal.py:16, 40-45`
- `CLAUDE.md:38, 49, 99, 128`
- `README.md:214`

## 已知限制 / 后续

- **Linux 未验证**：守卫只在 darwin 生效。建议在 Linux 工作站跑一次 `uv run pytest tests/test_xgb_model.py tests/test_ml_models.py -q`，留意 aarch64 上的 `cannot allocate memory in static TLS block`。
- **macOS 用户脚本**：同一进程混用 torch 与 xgboost 必须先 `export OMP_NUM_THREADS=1`（或在 import torch 之前设置）。已写入 `example/model.md` 与 `CLAUDE.md`。
- 按 IC 早停（`custom_metric` + `EarlyStopping(metric_name=..., maximize=True)`）只在文档中说明，未实现。

## Self-Check: PASSED

- 27 个创建/修改文件全部存在；6 个提交（`5d9a605`、`bea3712`、`d549c06`、`89a7aec`、`d6a0b3e`、`5b01c65`）均可解析。
- `quantlab/ml_model/xgboost.py` 不存在；`quantlab/__init__.py` 保持为空（预加载方案已撤回）。
- 工作树只剩你未提交的 `quantlab/enums/data.py`（从未 stage）和本 SUMMARY。
- STATE.md / ROADMAP.md 未改动：`state.advance-plan` 等命令针对 phase 03.6 的计划计数，quick 任务调用会弄乱当前阶段位置；按约束 ROADMAP 不动，状态记录交由编排者处理。
