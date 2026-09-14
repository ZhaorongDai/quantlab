---
phase: quick-260914-lno
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - quantlab/base/config.py
  - quantlab/base/model.py
  - quantlab/dl_model/mlp.py
  - quantlab/dl_model/rnn.py
  - quantlab/dl_model/rnn_classification.py
  - quantlab/ml_model/backend.py
  - quantlab/ml_model/xgb.py
  - quantlab/utils/module.py
  - quantlab/utils/metrics.py
  - pyproject.toml
  - uv.lock
  - tests/test_model_layer.py
  - tests/test_model_hierarchy.py
  - tests/test_model_cv.py
  - tests/test_ml_models.py
  - tests/test_metrics.py
  - tests/test_xgb_model.py
  - tests/test_ml_backend.py
  - example/model.md
  - example/backend.md
  - example/README.md
  - example/factor.md
  - README.md
  - CLAUDE.md
  - train_model.py
autonomous: true
requirements: [260914-lno]

estimate:
  tokens: 240000
  raw_tokens: 240000
  tasks: 3
  confidence: low

must_haves:
  truths:
    - "用户可以这样训练：`XGBoostRegressor(MLConfig(factors=..., labels=[SpotReturn(...)], early_stopping=True, early_stopping_patience=50, hyperparameters={\"num_boost_round\": 1000})).collect().train()`。xgboost 用验证集 RMSE 作判据做原生早停，保存下来的 Booster 已截断到 `best_iteration + 1` 棵树。每一轮的 `train-rmse`/`val-rmse` 以 `step=iteration` 记入 wandb；训练结束后，`{split}_loss`、`{split}_{mse,rmse,mae,r2,ic,rank_ic}` 与 `best_iteration`/`best_score` 写入 wandb summary。落盘 `.joblib` + `config.json`。`predict(np.ndarray [T,S,F])` 返回 `[T,S,L]` 的未来收益预测。在标签由因子决定的合成面板上测试集 IC > 0.5；标签与因子无关的对照面板上 |IC| < 0.2。"
    - "ML 模型完整支持交叉验证（R3）。`XGBoostRegressor(...).train_cv(train_periods, gap_periods, parallel, njobs)` 每一折都用该折自己的日期调用 `_fit`，折内训练段的尾部验证段驱动原生早停，写出 `{class}_cv_fold_{i}/{class}_cv_fold_{i}.joblib`。它返回逐折结果 list，每项含 fold、四个日期、experiment_name、checkpoint 与 `test_*` 指标；同时在一个名为 `{class}_cv_summary` 的独立 wandb run 的 summary 里写入 `cv_mean_test_*` 与 `cv_n_folds`。顺序分支与 `parallel=True` 分支消费同一个 `_cv_folds` 生成器，产出相同的折与 checkpoint。`nthread` 原样透传，不被改写。DL 的 `train_cv` 折几何由一条在重构之前就捕获的 golden 测试锁定为不变。"
    - "模型层是三层类结构（R-1）：`BaseModel` 是框架无关的共享生命周期，公开的 `train`/`train_cv`/`load`/`predict` 只在这里实现一份；`DLModel` 走 torch，带五个张量钩子；`MLModel` 走 numpy，带四个钩子（R2-3）。三者的 `__abstractmethods__` 由测试钉成精确集合。MLPRegressor、RNNRegressor、RNNClassifier、XGBoostRegressor 都不覆盖那四个公开方法。`from quantlab.base.model import BaseModel` 继续可用。"
    - "两条路径的早停边界一致，机制各按框架（R2）。DL 仍按 epoch 早停，并回滚到最优 epoch：`DLModel._fit` 原样沿用今天的循环，三条 WR-02 测试与 patience 测试一行不改仍然全绿。ML 交给库的原生早停：`early_stopping=True` 且有验证段时，保存的 Booster 轮数等于 `best_iteration + 1`，纯噪声标签下轮数严格小于 `num_boost_round`；`early_stopping=False` 时恰好 `num_boost_round` 轮。`MLModel` 里没有任何 deepcopy 回滚（AST 锁）。"
    - "config 类型错配时，在任何副作用发生之前抛 `TypeError`（R-1 setter）。`load()` 按 `checkpoint_suffix` 分流，并在构建模型之前拒绝错误后缀；`MLModel.load()` 不调用 `_init_model`（LD-B8）。`utils/module.py:load_model_from_config` 用 `cls.config_cls(**config)` 重建模型（R-5）。"
    - "既有测试零语义改动。`tests/test_model_layer.py` 只改 3 行：import、`RecordingRegressor` 的基类、`_train_dl(` 改名为 `_fit(`。`tests/test_dl_models.py` 不改一个字节。四个回归文件规划时实测 56 passed，改完仍全绿。全量 `uv run pytest` 与 Task 1 步骤 A 记录的基线失败集合相比没有新增失败，且 passed 数严格增加。规划时基线为 58 failed / 736 passed / 1 skipped，失败都在模型层之外，例如 tests/test_volume_guard.py。"
    - "`quantlab/utils/metrics.py` 的 IC/RankIC 是向量化实现：函数体内没有 for/while/推导式（AST 锁）。NaN 处理正确：先做联合掩码再排名；截面有效标的 <2 或零方差的时间戳被跳过；全部被跳过时返回 NaN，且不发 RuntimeWarning。结果与逐行 scipy pearsonr/spearmanr 参考实现数值一致。"
    - "文档描述的是三层类结构、两套钩子契约、ML 原生早停的理由，以及 ML 交叉验证：折几何、gap_periods、每折早停、nthread/njobs 建议、返回结果。README.md、CLAUDE.md、example/*.md 与 MlBackend docstring 里不再有过期陈述，包括「MLConfig 未实现」「MlBackend 没有调用点」「to_tensor 属于 BaseModel」，由 T3 的 verify 逐条 grep 为 0。example/model.md 里标注为真实输出的代码块都重新跑过。"
  artifacts:
    - path: quantlab/base/model.py
      provides: "BaseModel(ABC)（含 `_cv_folds` 生成器、`_train_one_fold`、CV summary）+ DLModel + MLModel"
    - path: quantlab/ml_model/xgb.py
      provides: "XGBoostRegressor(MLModel)：xgb.train + EarlyStopping(save_best=True) + 逐轮 wandb 回调 + NaN 标签行丢弃 + inf 转 NaN + nthread 透传"
    - path: quantlab/utils/metrics.py
      provides: "mse/rmse/mae/r2/cross_sectional_ic/cross_sectional_rank_ic/regression_panel_metrics"
    - path: quantlab/base/config.py
      provides: "MLConfig.early_stopping=False / early_stopping_patience=5（没有 epochs 字段）"
    - path: quantlab/utils/module.py
      provides: "load_model_from_config 按 cls.config_cls 构建配置"
    - path: tests/test_model_hierarchy.py
      provides: "三层类契约锁、配置类型锁、删除锁、加载器配置类锁"
    - path: tests/test_model_cv.py
      provides: "重构前捕获的 DL train_cv golden、`_cv_folds` 几何、两个分支共用生成器、ML stub CV 结果与均值 summary"
    - path: tests/test_ml_models.py
      provides: "MLModel._fit 编排语义（stub 头）"
    - path: tests/test_xgb_model.py
      provides: "XGBoostRegressor：学习、原生早停截断、wandb 逐轮回调、多标签、NaN/inf、持久化、顺序/并行 CV、nthread 透传"
    - path: tests/test_metrics.py
      provides: "指标函数手算用例 + scipy 参考一致性 + 向量化 AST 锁"
  key_links:
    - "`BaseModel.train()` 和 `BaseModel._train_one_fold()`（顺序与并行两个分支共用）都通过 `self.checkpoint_suffix` 拼 checkpoint 名。今天这两处都硬编码 `.pth`；漏改任何一处，ML checkpoint 都会被 `load()` 的后缀检查拒收。"
    - "`BaseModel.train_cv` 的两个分支都消费 `BaseModel._cv_folds(timestamps, train_periods, gap_periods)`，由它产出每折的 fold 编号与四个日期。顺序分支直接调用 `self._train_one_fold(fold, project_name)`；并行分支调用 `_train_fold_with_config`，即先 `copy.deepcopy(self)` 再 `._train_one_fold(...)`。折边界算术只存在一份，ML 与 DL、顺序与并行都不可能各自漂移。"
    - "`MLModel._fit` 的调用链：`_fit_model(train_x, train_y, val_x|None, val_y|None)` 恰好调用一次；然后 `_evaluate(\"train\"|\"val\"|\"test\", ...)` 写入 `self._wandb_recorder.summary` 并返回带前缀的指标 dict；然后 `_save_model` 调 `_write_checkpoint`，最终执行 `MlBackend().to_internal(self.model).write(...)`；`_fit` 返回 test 指标 dict。`_train_one_fold` 把这个 dict 并进该折的结果，`train_cv` 再对各折的 `test_*` 求均值，写进 `{class}_cv_summary` run。"
    - "XGB 的 callbacks 顺序必须是逐轮 wandb 回调在前、`EarlyStopping` 在后。xgboost 的回调容器按短路方式依次调用回调。规划时实测 `[EarlyStopping, 记录器]` 这个顺序，记录器漏掉了触发停止的那一轮。"
    - "`EarlyStopping(rounds=patience, data_name=\"val\", save_best=True)` 让 `xgb.train` 返回的 Booster 已截断到 `best_iteration + 1`；`best_iteration`/`best_score` 在截断和 joblib 往返之后都还在（规划时在 3.4.1 实测）。所以落盘的 `.joblib` 就是最优模型，测试断言读的是磁盘文件。"
    - "`load_model_from_config` 使用 `cls.config_cls(**config)`，BaseModel setter 再做 isinstance 检查。这样 MLConfig 字典不能再静默建出 DLConfig、走 torch 路径。"
---

<objective>
给模型层接入非 torch 的 ML/树模型训练路径（含交叉验证），并按用户 2026-09-14 的三轮设计修订把 `BaseModel` 拆成三层：
框架无关的抽象 `BaseModel`，以及两个各有一套抽象钩子的子类 `DLModel` 与 `MLModel`。二者共享同一套公开训练/预测接口
（`train`/`train_cv`/`load`/`predict`）。然后在 `quantlab/ml_model/xgb.py` 新建 `XGBoostRegressor(MLModel)`，用 xgboost
原生早停预测未来收益率。

Purpose: CLAUDE.md 的架构契约是「收益模型输出未来收益/收益排名预测」，树模型是量化里最常用的基线。今天 `MLConfig` 和 `MlBackend`
都存在，却没有任何一条路把它们接起来。把 ML 分支硬塞进一个 torch 形状的基类，会让每个方法都长出 `isinstance(self.model, nn.Module)`
分支，所以用户选择按框架拆类，每一层只暴露自己需要的钩子。ML 头用各库的原生早停，而不是一个外层 epoch 循环，原因有三：
- xgboost/LightGBM/CatBoost 都按每棵树判定；
- 验证分数靠预测缓存增量计算，成本是线性的，而每个 epoch 用全部树重算整个验证集是二次的；
- 回滚靠切片树，不需要 deepcopy 整个模型。
交叉验证是量化模型评估的标配，ML 路径必须真能跑，而不只是继承了一个方法名。

Output: 三层模型类、统一的 CV 折生成器与 CV 结果、XGBoost 回归头、可复用的截面指标模块、五份新测试文件、同步后的文档。

设计来源与优先级：
- 基线：编排者的 locked_design（LD-A..LD-E）。
- 修订 #1（R-1..R-7）：三层拆分、公开接口、config_cls、checkpoint_suffix、加载器、测试与文档范围。
- 修订 #2（R2-1..R2-6）：
  - MLConfig 不加 `epochs`；
  - DL 早停保持内联原样（R2-2 允许二选一，本计划选内联：DL 语义零改动，既有测试逐字不动）；
  - MLModel 没有 epoch 循环，钩子是 `_fit_model`，取消 `_snapshot_model` 与 deepcopy 回滚；
  - XGB 使用 `num_boost_round` + `xgb.train` + `EarlyStopping(save_best=True)` + 逐轮 wandb 回调。
- 修订 #3（R3-1..R3-6）：
  - `train_cv` 抽出唯一的折生成器；
  - 每折 `_fit` 返回指标，`train_cv` 返回逐折结果并写入均值 summary；
  - 并行 ML 折透传 `nthread` 并给出文档建议；
  - CV 测试与文档。
  - 对修订 #2 的一处覆盖：`_evaluate` 的返回值从 float 改为带前缀的指标 dict，loss 在 dict 的 `{split}_loss` 里。理由是 R3-4 要求 ML 的折结果就是 `_evaluate("test")` 的指标 dict。
- R3-4 的二选一：均值写进一个独立的 `{class}_cv_summary` run，而不是最后一折的 summary。原因是每折的 `_fit` 结束时已经 `finish()` 了自己的 run，均值算出来时最后一折的 run 已经关闭，往已结束的 run 写 summary 不成立。
- 仍然有效：
  - LD-B 的 6/7/8/10；
  - LD-C 的 NaN 标签行丢弃、inf 转 NaN、文件名 `xgb.py`、多标签、指标定义；
  - LD-D 中与上述不冲突的测试；
  - LD-E 的文档范围。
</objective>

<assumption_delta_decision>
- 触发形态：pluralization。模型层从「只有 torch 一种」变成「torch + 非 torch 两种」。
- 现在为主的名词：框架无关的模型生命周期 `BaseModel`。它拥有公开的 `train`/`train_cv`/`load`/`predict`、CV 折几何、checkpoint 目录约定与 `config.json`。
- 决定：**promote**。torch 相关的一切，包括 device、to_tensor、DataLoader epoch 循环、state_dict checkpoint、refit 优化器，都降为 `DLModel` 这个变体的细节。ML 是与之平级的 `MLModel`；训练编排与早停机制各按框架实现，CV 折几何只有 BaseModel 这一份。
- 不变量测试（已纳入 Task 1）：公开四方法只定义在 `BaseModel` 上，且不被任何出厂头覆盖；`train_cv` 的两个分支必须经过同一个 `_cv_folds`。
</assumption_delta_decision>

<execution_context>
@~/.claude/gsd-core/workflows/execute-plan.md
@~/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@quantlab/base/model.py
@quantlab/base/config.py
@quantlab/ml_model/backend.py
@quantlab/utils/module.py
@tests/test_model_layer.py
@tests/test_dl_models.py
@example/model.md

规划时实测（2026-09-14，HEAD = 89b027c）：

测试基线
- 回归基线：`uv run pytest tests/test_model_layer.py tests/test_dl_models.py tests/test_ml_backend.py tests/test_factor_hierarchy.py -q` 为 56 passed。
- 全量基线：`uv run pytest -q` 为 **58 failed, 736 passed, 1 skipped**，耗时约 110 秒。失败都在模型层之外（例如 tests/test_volume_guard.py），可能与用户工作树里未提交的改动有关。本计划不修也不碰它们。验收口径是「失败集合不新增」，不是「0 failed」。

依赖
- xgboost 3.4.1、scipy 1.18.1、scikit-learn 1.9.0 已安装。
- `xgboost` 已在 pyproject 声明。**scipy 没有直接声明**，只以 scikit-learn 依赖的身份锁在 uv.lock。

xgboost 3.4.1 探针
- 早停场景：纯噪声标签，调用 `xgb.train(params, dtrain, 300, evals=[(dtrain,"train"),(dval,"val")], callbacks=[EarlyStopping(rounds=5, data_name="val", save_best=True), rec], verbose_eval=False)`。
  - 返回的 Booster `num_boosted_rounds() == 1`，`best_iteration == 0`，`best_score` 是 float。
  - joblib dump/load 之后，轮数、`best_iteration`、预测都不变。
  - 排在 EarlyStopping **之后**的记录回调只收到第 0-4 轮，漏掉了触发停止的那一轮。
  - `evals_log` 的结构是 `{"train": {"rmse": [...]}, "val": {"rmse": [...]}}`。
- 不带早停时恰好 30 轮，并且没有 `best_iteration` 属性。
- 双标签加早停可以正常工作：`inplace_predict` 返回 `(n, 2)`。标签形状为 `(n,1)` 时返回 `(n,)`。
- 从 `json.loads(booster.save_config())` 能读回两个参数：
  - `["learner"]["gradient_booster"]["tree_train_param"]["max_depth"]`；
  - `["learner"]["generic_param"]["nthread"]`。
  - 两者的值都是字符串。
- 编排者另外验证过：DMatrix 含 inf 会抛 XGBoostError。

wandb 与 scipy 探针
- wandb：`WANDB_MODE=disabled` 时 `wandb.init` 返回 `NoopRun`，`.summary.update({...})`、`.summary[k] = v`、`.log(d, step=i)`、`.finish()` 都可以调用。
- scipy：`rankdata(a, axis=1, nan_policy="omit")` 给出平均秩，NaN 位置保持 NaN，整行 NaN 输出整行 NaN，在 `-W error::RuntimeWarning` 下不报警。

结构性测试陷阱
- `tests/test_factor_hierarchy.py::test_base_model_does_not_dispatch_on_concrete_factor_types` 按**行**扫描 `quantlab/base/model.py`，只豁免以 `#` 开头的行。
- 任何一行（**包括 docstring 行**）同时出现 `isinstance` 和 `Factor`，或者出现 `FactorKunQuant`，都会让它变红。

受影响的调用点（grep）
- `BaseModel` 被四处继承：`quantlab/dl_model/mlp.py`、`rnn.py`、`rnn_classification.py`、`tests/test_model_layer.py`。
- `cal.py` 只 import 不使用。拆分后 `BaseModel` 名字仍在，cal.py 不改。
- `_train_dl` 唯一的代码调用在 `tests/test_model_layer.py:553`。
- `_predict_nn` 和 `_auto_train` 在 `quantlab/base/model.py` 之外没有代码调用。
- `train_cv` 与 `_train_fold_with_config` 在 `quantlab/base/model.py` 之外没有任何代码调用。`quantlab/base/acquisition.py:1329` 只有一句注释，说 train_cv 用 threading 后端，重构后这句话依然成立。
- 仓库里目前**没有任何** `train_cv` 测试。

仓库约束（执行者必须遵守）

提交与工作树
- 用户工作树里有未提交的 `quantlab/enums/data.py`、`test.py`，以及未跟踪的 `quantlab/acquisition/wrds.py`。
  - **绝不** stage、修改或提交这三个文件。
  - 每次提交都用显式路径 `git add <path>...`；禁止 `git add -A` / `git add .` / `git commit -a`。
  - **绝不** `git stash`。需要 A/B 对比时，用 `git show <sha>:<file>` 拷到临时路径。
- 在 Task 1 第一步、任何编辑之前，把基线 SHA 钉进 git 目录：`git rev-parse HEAD > "$(git rev-parse --git-dir)/quick-260914-lno-base"`。所有「只改了 N 行」的 diff 门都对这个 SHA 比较，而不是对工作区比较：GSD 逐任务提交，只看工作区的比较不可能失败。

注释与测试写法
- docstring 和注释的语言跟随所在文件：`quantlab/base/model.py`、`quantlab/ml_model/`、`quantlab/utils/metrics.py` 用中文 docstring；测试文件用英文 docstring，每条测试说明它锁什么、什么改动会让它变红。
- 测试一律用合成数据，CPU，离线。沿用 `tests/test_model_layer.py` 的 `FakePanel` + autouse `_offline_wandb` 模式（`WANDB_MODE=disabled`、`WANDB_SILENT=true`）。
- 伪 recorder 需要提供：记录 `log(data, step=None)`、一个 dict 类型的 `summary` 属性、`finish()` 计数。
- 凡是涉及 `train_cv(parallel=True)` 的 wandb 断言，都要在**类**上 monkeypatch `_init_wandb`，并把创建出的 recorder 登记到测试作用域的 list 里。因为并行分支会 `copy.deepcopy(self)`，实例级 monkeypatch 的闭包会把 recorder 挂到原实例上，而不是挂到折副本上。
- 同一个测试里既跑顺序 CV 又跑并行 CV 时，必须给两次运行不同的 `model_save_dir`：project_name 精确到秒，同一秒内两次运行会撞同一个目录，`_save_model` 会拒绝覆盖。

提交信息
- 格式为 `feat(260914-lno): ...` / `test(260914-lno): ...` / `docs(260914-lno): ...`。
- 结尾两行：`Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` 与 `Claude-Session: https://claude.ai/code/session_01JRsYP7rLLEvU8bsd7mnRCJ`。

TDD 证据
- GSD 的 tdd-red-evidence 检查只解析 node TAP。如果它要求 TAP 证据，必须用一次性脚本把**真实**的 pytest 运行结果投影成 TAP，绝不手写计数。
</context>

<tasks>

<task type="tracer" tdd="true">
  <name>Task 1 (tracer): 先用 golden 钉住 DL train_cv，再拆出 BaseModel / DLModel / MLModel 与统一的 CV 折生成器，用一个纯 numpy 的 stub ML 头端到端打通 train / train_cv -> .joblib -> load -> predict（含 wandb summary 与 CV 均值）</name>
  <files>quantlab/base/config.py, quantlab/base/model.py, quantlab/dl_model/mlp.py, quantlab/dl_model/rnn.py, quantlab/dl_model/rnn_classification.py, quantlab/ml_model/backend.py, quantlab/utils/module.py, quantlab/utils/metrics.py, pyproject.toml, uv.lock, tests/test_model_layer.py, tests/test_model_hierarchy.py, tests/test_model_cv.py, tests/test_ml_models.py, tests/test_metrics.py</files>
  <read_first>
    - quantlab/base/model.py：全文，741 行；其中第 494-660 行是 train_cv 与 _train_fold_with_config。
    - quantlab/base/config.py：DLConfig / MLConfig。
    - quantlab/ml_model/backend.py。
    - quantlab/utils/module.py。
    - tests/test_model_layer.py 第 48-260 行：FakePanel、RecordingRegressor、_make_config 是要镜像的模式。
    - tests/test_dl_models.py 第 64-175 行：随机值 FakePanel。
    - tests/test_factor_hierarchy.py 第 104-148 行：按行扫描 base/model.py 的结构性测试。
  </read_first>
  <precondition>基线 SHA 已钉进 `$(git rev-parse --git-dir)/quick-260914-lno-base`；`grep -q '^name = "scipy"' uv.lock` 成功；上面四个回归文件 56 passed。</precondition>
  <behavior>
    DL train_cv golden（重构**之前**写、在未改动的代码上跑绿，之后不改断言）
    - 130 个时间戳的面板上执行 `train_cv(train_periods=50, gap_periods=3)`：
      - 顺序模式写出恰好 7 个折目录 `{cls}_cv_fold_{i}`（i=0..6），每个目录含 `{cls}_cv_fold_{i}.pth` 和 `config.json`。
      - 每折实际训练用的 `(train_start, train_end, test_start, test_end)` 等于公式给出的值：训练段取下标 `i*10` 到 `i*10+49`，测试段取 `i*10+53` 到 `i*10+62`。日期字符串取自已 collect 面板的 timestamp 坐标，用 `np.datetime_as_string` 生成，与 train_cv 自己的做法一致。
      - 在 stub 的 `_init_optim` 里把 `self.config` 的四个日期登记到模块级 list 中，由此得到每折实际训练用的日期。
    - `parallel=True, njobs=2`（使用另一个 model_save_dir）：得到同一组折目录、同一组日期。
    - golden 只断言 checkpoint 文件和日期，不断言返回值。重构前 train_cv 返回 None。

    `_cv_folds` 与 train_cv 的统一
    - `BaseModel._cv_folds(timestamps, train_periods=50, gap_periods=3)` 在 130 个时间戳上返回 7 个 dict，键恰好是 `fold, train_start, train_end, test_start, test_end`。
    - 同一折内：train_end 与 test_start 之间恰好隔 3 个时间戳，训练段与测试段不相交。
    - 相邻两折的测试窗口首尾相接、不重叠。
    - 数据不足（总长度 < train+gap+test）时返回 `[]`。
    - 用 monkeypatch 把 `BaseModel._cv_folds` 换成一个返回手工构造的 2 折 list 的函数：顺序分支和并行分支都恰好训练这 2 折，由 checkpoint 目录与登记的日期证明。

    CV 结果与均值
    - ML stub 头（随机值面板，`_forward` 返回首个因子作预测，使 IC 有限）调用 `train_cv(train_periods=50)` 时：
      - 返回 8 个结果 dict；每个含 `fold`、四个日期、`experiment_name`、`checkpoint`，以及 `test_loss, test_mse, test_rmse, test_mae, test_r2, test_ic, test_rank_ic`；
      - 日期等于 `_cv_folds` 的输出；
      - 每个 `checkpoint` 路径是一个存在的 `.joblib` 文件；
      - 一个新实例 `load` 每折之后，predict 返回 `[T,S,L]`。
    - wandb（在类上 monkeypatch `_init_wandb`）：
      - 创建了 8 个折 run 加 1 个名为 `{cls}_cv_summary` 的 run；
      - summary run 的 summary 里每个 `cv_mean_test_{k}` 等于各折 `test_{k}` 的有限值均值，并且含 `cv_n_folds == 8`；
      - summary run 恰好 finish 一次。
    - ML stub 头 `parallel=True, njobs=2`（使用另一个 model_save_dir）：得到与顺序模式相同的折 checkpoint 相对路径集合与日期。
    - DL（RecordingRegressor 同构的 stub）调用 train_cv 时：
      - 返回的结果 dict 只含 `fold`、四个日期、`experiment_name`、`checkpoint`，不含任何 `test_` 键；
      - `_init_wandb` 的调用次数等于折数，不创建 summary run。

    契约
    - `BaseModel.__abstractmethods__ == {config_cls, checkpoint_suffix, _fit, _predict, _write_checkpoint, _read_checkpoint}`
    - `DLModel.__abstractmethods__ == {_init_model, _train_one_batch, _val_one_batch, _test_one_batch, _preprocess}`
    - `MLModel.__abstractmethods__ == {_init_model, _preprocess, _fit_model, _forward}`
    - 公开的 `train`、`train_cv`、`load`、`predict` 只出现在 `BaseModel.__dict__`。对 DLModel、MLModel、MLPRegressor、RNNRegressor、RNNClassifier 和测试里的 stub ML 头，其 MRO 中位于 BaseModel 之前的每个类，`__dict__` 都不含这四个名字。
    - `DLModel.config_cls is DLConfig`，`DLModel.checkpoint_suffix == ".pth"`；`MLModel.config_cls is MLConfig`，`MLModel.checkpoint_suffix == ".joblib"`。
    - 用 MLConfig 构造 MLPRegressor、用 DLConfig 构造 stub ML 头，都抛 TypeError；被拒的 FakePanel 的 `config.start_date` 仍为 None。

    删除锁
    - BaseModel 上没有 `_auto_train`；DLModel 上没有 `_train_dl` 和 `_predict_nn`；MLModel 上没有 `_snapshot_model` 和 `_train_one_epoch`。
    - `"epochs" not in MLConfig.__dataclass_fields__`；MLConfig 默认 `early_stopping is False`、`early_stopping_patience == 5`。
    - MLModel 类体的 AST 里，没有任何 `ast.Name`/`ast.Attribute` 引用 `deepcopy`。

    加载器与 DL 推理
    - `load_model_from_config` 选择配置类：
      - DL 字典经真实点分路径 `quantlab.dl_model.mlp.MLPRegressor` 构造，得到 `type(config) is DLConfig`；
      - ML 字典（stub 头，monkeypatch `get_cls_from_path`）得到 `type(config) is MLConfig`；
      - 两种情况都 monkeypatch `load_factor_from_config`，让它返回 FakePanel。
    - DLModel.predict 接受 float64 的 `np.ndarray`，结果与同值 float32 张量完全一致。

    stub ML 头的编排
    - 每次 `_fit` 恰好调用一次 `_fit_model`。
    - 100 个训练时间戳、val_size=0.2 时：train 部分 80 个、val 部分 20 个，合计 100。
    - `val_size=0.0` 时，val_x 和 val_y 都是 None。
    - `val_size=1.0` 时 `_fit` 抛 ValueError，且 `_fit_model` 没有被调用。
    - 非字母序声明的因子（zeta, alpha, mid）和标签（ret_30, ret_60, ret_120）按声明顺序到达 `_fit_model`。
    - `_fit` 返回的 dict 含 `test_loss`。

    stub ML 头的 wandb 与持久化
    - 伪 recorder 的 summary 恰好出现 `{train,val,test}_{loss,mse,rmse,mae,r2,ic,rank_ic}` 这 21 个键；`val_size=0.0` 时没有 `val_` 键；`finish()` 恰好调用一次。
    - `train()` 只写出一个 `.joblib` 加 `config.json`，没有 `.pth`；磁盘上的 `.joblib` 读回来等于 `_fit_model` 留下的对象。
    - 新实例（不 collect）`load()` 之后，`_init_model` 调用次数为 0，predict 与训练实例逐位相同。
    - 对 ML 模型 `load("x.pth")` 抛 ValueError，且不调 `_init_model`。
    - predict 对 ndarray 和 torch.Tensor 输入都返回 `[T,S,L]`；未训练、未加载时 predict 抛 ValueError。
    - `MLModel._loss` 默认值：标签含 NaN 的 (t,s) 行被排除，结果等于手算 MSE；没有有效行时返回 NaN，且不发 RuntimeWarning。

    metrics
    - 基础误差：`mse([1,2,nan],[1,4,5]) == 2`，`mae == 1`，`rmse == sqrt(2)`，`r2 == 1-4/4.5`。
    - IC：
      - pred 两行都是 [1,2,3]、target 为 [1,2,3] 和 [3,2,1] 时，IC 为 0.0。
      - 只剩联合有效对 (1,2)/(2,4) 的行 IC 为 1.0。
      - 有效数 <2 的行和零方差行被跳过；全部被跳过时为 NaN。
    - RankIC：
      - pred [1,10,100] 对 target [1,2,3] 为 1.0。
      - 平局：pred [1,2,3] 对 target [1,1,2] 为 0.8660254。
      - 联合掩码先于排名：pred [1,2,3,4] 对 target [4,3,nan,1] 为 -1.0。
    - 在随机含 NaN 的面板上，与逐行 scipy pearsonr/spearmanr 一致（atol 1e-10）。
    - `cross_sectional_ic` 与 `cross_sectional_rank_ic` 的函数体 AST 里没有 For/While/推导式节点。
    - 整个 metrics 测试模块在 `error::RuntimeWarning` 下运行。
  </behavior>
  <action>
按以下顺序执行。A2 的 golden 必须在任何生产代码改动之前写好并跑绿。本仓库 03.6 的决定是「先捕获 golden 再抽取」：抽取之后再取的基线，只能发现日后的漂移，发现不了抽取动作本身引入的漂移。B-C 做完后先跑既有回归，再写任何 ML 代码。

A. 全量基线
  1. 钉住基线 SHA，令 `GD=$(git rev-parse --git-dir)`。
  2. 运行 `uv run pytest -q -p no:cacheprovider -rf > "$GD/quick-260914-lno-baseline-pytest.txt" 2>&1`。约 2 分钟；因为有既有失败，会以非零状态退出，这是预期的。
  3. 运行 `grep '^FAILED' "$GD/quick-260914-lno-baseline-pytest.txt" | sed 's/ - .*//' | sort -u > "$GD/quick-260914-lno-baseline-failures"`。
  4. 把 baseline-pytest.txt 最后一行的汇总，以及基线失败文件的行数，原样写进 SUMMARY。

A2. DL train_cv golden（R3-5，重构前）
  1. 新建 `tests/test_model_cv.py`，写两条 golden 测试：`test_dl_train_cv_fold_geometry_golden_sequential` 与 `test_dl_train_cv_fold_geometry_golden_parallel`，逐条对应 behavior 块里的 golden 部分。
  2. stub 此时继承**当前的** `BaseModel`：一个在最后一维上作用的 `nn.Linear` 小头，`epochs=1`，并在 `_init_optim` 里登记日期。
  3. 在未改动的生产代码上运行 `uv run pytest tests/test_model_cv.py -q -k golden`，必须为绿。
  4. 立即提交 `test(260914-lno): pin DL train_cv fold geometry before extraction`，只 stage 这一个文件。

B. `quantlab/base/model.py`：BaseModel 与 DLModel 拆分（R-1、R-2、R3-1、R3-4）
  三个类都留在这个文件里。注意 tests/test_factor_hierarchy.py 的按行扫描：任何非 `#` 行（包括 docstring 行）都不能同时出现 `isinstance` 和 `Factor`，也不能出现 `FactorKunQuant`。
  1. 中文类 docstring：`BaseModel(ABC)` 的 docstring 说明三层拆分与理由。
  2. 抽象类属性：`config_cls` 与 `checkpoint_suffix` 在 BaseModel 上声明为 `@property` + `@abstractmethod`，具体子类用普通类属性满足它们，ABCMeta 会因此把它们移出 `__abstractmethods__`。之所以必须是类属性，是因为 `utils/module.py` 要在实例化之前从类上读 `config_cls`。
  3. `__init__` 保持现有顺序：config setter、`_set_random_seed`、`self.model = None`、`XrBackend()`、recorder 置 None。`_set_random_seed` 保持 staticmethod；BaseModel 版本只播种 python `random` 与 numpy。
  4. config setter 的**第一条语句**：`config` 不是 `self.config_cls` 的实例时抛 TypeError，消息点名模型类名、期望的配置类名、实际类型名。这一步先于给 `_config` 赋值，也先于触碰任何因子或标签；setter 其余逻辑不变。
  5. 原样保留在 BaseModel 上的成员：
     - `__repr__`、config property；
     - num_times、class_name、num_symbols、symbols、num_null、import_path、num_factors、num_labels；
     - 三个 `_reset_*_config`、两个 `_collect_all_*`、`collect`；
     - `get_factor_names`、`get_label_names`、`get_config`、`_get_config_with_extra_kv`；
     - `_init_wandb`。
     `_assert_shape_match_x/_y` 的注解放宽为 `np.ndarray | torch.Tensor`。
  6. 新增 `to_array(data, variables) -> np.ndarray`：函数体就是现在 `to_tensor` 里那条 xarray 链，并把「最后一维严格按 variables 顺序」这段说明挪进它的 docstring。
  7. `_save_model(p)`：保留 None 检查、父目录已存在时的 RuntimeError、mkdir、写 config.json；最后改为调用 `self._write_checkpoint(p)`。
  8. 公开 `load(p)`：
     - str 转 Path；文件不存在时抛 FileNotFoundError。
     - 然后检查 `p.suffix != self.checkpoint_suffix`，不符时抛 ValueError（点名实际后缀与期望后缀）。这一步先于任何模型构建。
     - 再调用 `self._read_checkpoint(p)`，返回 self。
  9. 公开 `predict(data)`：model 为 None 时抛 ValueError，沿用现有消息 "Model not initialized, please call load() or train() first"；否则返回 `self._predict(data)`。
  10. 公开 `train()`：名字生成照旧，但 `model_name = f"{experiment_name}{self.checkpoint_suffix}"`；然后调用 `_init_wandb`，再调用 `self._fit(project_name=..., experiment_name=..., model_name=...)`。返回值保持 None。
  11. CV 统一（R3-1、R3-2、R3-4）：
     - 私有 staticmethod `_cv_folds(timestamps, train_periods, gap_periods) -> list[dict]`：
       - 这是折边界算术**唯一**的实现，逐字保留现有几何：`test_periods = train_periods // 5`；`n_splits = max(1, (total - train_periods - gap_periods) // test_periods)`；各下标与今天两个分支的写法完全相同；`test_end_idx > total` 时 `logger.warning(f"Skipping fold {i}: test set exceeds data range")` 并跳过该折。
       - 日期用 `np.datetime_as_string` 生成。
       - 每折产出 `{"fold": i, "train_start": ..., "train_end": ..., "test_start": ..., "test_end": ...}`。
       - `test_periods == 0` 时现有的 ZeroDivisionError 行为保持不变，不在本计划范围内。
     - 实例方法 `_train_one_fold(fold, project_name) -> dict`：
       - 把 fold 的四个日期写进 `self.config`，令 `experiment_name = f"{self.class_name}_cv_fold_{fold['fold']}"`、`model_name = experiment_name + self.checkpoint_suffix`。
       - 执行 `self._init_wandb(project_name, experiment_name)`，然后 `metrics = self._fit(project_name=..., experiment_name=..., model_name=...)`。
       - 返回 `{**fold, "experiment_name": ..., "checkpoint": str(Path(model_save_dir)/project_name/experiment_name/model_name), **(metrics or {})}`。
     - `_train_fold_with_config(fold, project_name) -> dict`：执行 `copy.deepcopy(self)._train_one_fold(fold, project_name)`。签名改为接收 fold dict 和 project_name；它在 model.py 之外没有调用者。
     - 私有 staticmethod `_cv_mean_metrics(results) -> dict`：
       - 对所有结果中出现的每个 `test_` 开头的键，取各折的有限值求均值，键名为 `cv_mean_{key}`；某个键没有任何有限值时为 NaN，用显式计数，不发 RuntimeWarning。
       - 另加 `cv_n_folds`。
       - 如果没有任何 `test_` 键，返回空 dict。
     - `train_cv(train_periods, gap_periods=0, parallel=False, njobs=-1) -> list[dict]`：
       1. 保留日期过滤、"No data found" 的 ValueError，以及原有的 info 日志。
       2. `folds = self._cv_folds(timestamps, train_periods, gap_periods)`，并逐折记录原有的 "Fold {i}: Train [...], Test [...]" info 日志。
       3. 顺序分支：`results = [self._train_one_fold(f, project_name) for f in folds]`。
       4. 并行分支：保留原有的 "Starting parallel training" 日志，然后 `results = Parallel(n_jobs=njobs, backend="threading")(delayed(self._train_fold_with_config)(f, project_name) for f in folds)`。
       5. `means = self._cv_mean_metrics(results)`。means 非空时：执行 `self._init_wandb(project_name, f"{self.class_name}_cv_summary")`；如果 recorder 存在，`summary.update(means)` 后 `finish()`。
       6. 返回 results。
     - DL 的 `_fit` 返回 None，所以 means 为空、不会多开 run，DL 行为不变。
     - 删除 `_auto_train`，不留别名：一个方法两个活名字，正是日后读者会「修」错的歧义，本仓库 03.1/03.2 已两次按此决定。
  12. BaseModel 抽象方法：`_fit(project_name, experiment_name, model_name) -> dict | None`（docstring 说明：返回 test 指标 dict，不产出指标的实现返回 None）、`_predict(data)`、`_write_checkpoint(path: Path)`、`_read_checkpoint(path: Path)`。
  13. `DLModel(BaseModel)`：类属性 `config_cls = DLConfig`、`checkpoint_suffix = ".pth"`。
     - `_set_random_seed`（staticmethod）先调 `BaseModel._set_random_seed(seed)`，再原样执行 torch/cuda 四个种子与 `cudnn.deterministic`。
     - 原样迁入：`device`、`_init_model_and_optim`、`_init_optim`、`_get_refit_optim`。
     - `to_tensor`：改为 `torch.from_numpy(self.to_array(...))`，再做现有的浮点 dtype 统一。这段统一抽成私有 staticmethod（例如 `_to_default_float`），供 `_predict` 复用。
     - `_fit(project_name, experiment_name, model_name, backtest=False)`：就是现在的 `_train_dl` 函数体，原样照搬，包括内联的早停簿记与最优 state_dict 快照（R2-2 内联）。唯一改动是 backtest 守卫消息改为点名 `_fit(backtest=True)`，并保留 "Phase 6"。
     - `_predict(data)`：`np.ndarray` 经 `torch.from_numpy` + `_to_default_float` 转换，`torch.Tensor` 原样使用，其他类型抛 TypeError；之后执行现在 `_predict_nn` 的函数体，去掉已经上提的 None 检查。删除 `_predict_nn`，不留别名。
     - `_write_checkpoint(p)`：`torch.save(self.model.state_dict(), p)`。
     - `_read_checkpoint(p)`：等于今天的 `.pth` 路径，即先 `_init_model(...).to(self.device)`，再 `load_state_dict(torch.load(p))`。
     - 五个抽象钩子的签名保持不变。
  14. 删除已经不再使用的 `import joblib`，保留 `from joblib import Parallel, delayed`；同时 import `copy`。

C. DL 头与测试改基类
  1. `quantlab/dl_model/rnn.py`、`rnn_classification.py`：只把 import 与类定义里的 BaseModel 换成 DLModel，每个文件恰好 2 行。
  2. `quantlab/dl_model/mlp.py`：同样换基类，另外把 docstring 里点名的旧调用点改成新名字（`DLModel._init_model_and_optim`、`DLModel._preprocess`、`DLModel._fit`、`DLModel._predict`），只改散文。
  3. `tests/test_model_layer.py` 恰好改 3 行：import、`class RecordingRegressor(DLModel)`，以及 `test_train_dl_rejects_a_truthy_backtest_flag` 里的 `model._train_dl(` 改为 `model._fit(`。这是 R-2 的二选一：调用改名，不留别名。断言、测试名、fixture、docstring 一律不动。
  4. `tests/test_model_cv.py`：只把 golden stub 的 import 与基类换成 DLModel，golden 的断言不动。
  5. 运行四个回归文件加 `tests/test_model_cv.py -k golden`：必须 56 passed，golden 全绿，之后才能进入 D。

D. `quantlab/base/config.py`（R2-1）
  - 只在 `MLConfig` 的「训练相关」段加两个字段：`early_stopping: bool = False`、`early_stopping_patience: int = 5`。字段旁注释写明：对树模型头，patience 按 boosting 轮数计。
  - **不加** `epochs`，DLConfig 不动。

E. 新建 `quantlab/utils/metrics.py`（中文 docstring）
  - 公共约定：所有函数把输入转成 float64；两个输入形状不同时抛 ValueError；只统计两边都有限的「联合掩码」位置。
  - `mse`、`rmse`、`mae`、`r2`：返回 float，掩码为空时返回 NaN。`r2` 在有效数 <2 或总平方和为 0 时也返回 NaN。
  - `cross_sectional_ic(pred, target)`：
    - 要求 2-D `[T, S]` 输入，否则抛 ValueError。
    - 用 `np.where(mask, ..., 0)` 按行求有效数、均值、去均值交叉积与方差，得到逐行 Pearson。
    - 排除有效数 <2 的行和零方差行。零方差用掩码后 max 等于 min 做精确判定，不用浮点方差阈值。
    - 返回剩余行的均值；没有剩余行时返回 NaN。
    - 在 `np.errstate(invalid="ignore", divide="ignore")` 里计算，并且不得发出任何 RuntimeWarning：不对全 NaN 做 nanmean，改用显式计数。
    - 函数体里不能有 Python 行级循环。
  - `cross_sectional_rank_ic`：
    1. 先把联合掩码之外的格子在**两个**数组上都置为 NaN；
    2. 再调用 `scipy.stats.rankdata(..., axis=1, nan_policy="omit")`；
    3. 对得到的秩调用 `cross_sectional_ic`。
  - `regression_panel_metrics(pred, target) -> dict[str, float]`：键恰好是 `mse, rmse, mae, r2, ic, rank_ic`。

F. 声明 scipy 依赖
  - 运行 `uv add "scipy>=1.18.1"`。离线失败时，改为手工在 pyproject.toml 的 dependencies 里按字母序加这一行，再运行 `uv lock --offline`。
  - uv.lock 的 diff 里不得出现任何 `version = ` 行的增删。

G. `MLModel(BaseModel)`（同文件，R2-3、R3-4）
  类属性：`config_cls = MLConfig`、`checkpoint_suffix = ".joblib"`。
  1. 中文类 docstring 说明两点：ML 头使用所属库的原生早停，所以 MLModel 没有 epoch 循环，也不做任何模型拷贝回滚；以及为什么 DL 仍然按 epoch 走。
  2. 抽象方法（各带中文 docstring）：
     - `_init_model(num_features, num_labels, hyperparameters)`：返回值原样赋给 `self.model`，允许为 None。
     - `_preprocess(data: np.ndarray) -> np.ndarray`。
     - `_fit_model(train_x, train_y, val_x, val_y) -> None`：验证段为空时 val 两个参数为 None；早停与最优模型回滚由它用库原生机制完成，遵循 `config.early_stopping` 与 `config.early_stopping_patience`；返回时 `self.model` 必须已经是要保存的模型。
     - `_forward(x: np.ndarray) -> np.ndarray`：返回 `[T,S,L]`。
  3. 非抽象、可覆盖的方法：
     - `_loss(y, pred) -> float`：只在所有标签都有限的 (t,s) 位置上，对全部标签求 MSE；没有有效位置时显式返回 NaN。
     - `_compute_metrics(y, pred) -> dict`：返回 `regression_panel_metrics(pred[..., 0], y[..., 0])`。
     - `_evaluate(split, x, y) -> dict[str, float]`：
       - 计算 `pred = self._forward(x)`；
       - 组装 `{split}_loss` 与每个指标的 `{split}_{key}`；
       - recorder 存在时执行 `self._wandb_recorder.summary.update(metrics)`，写的是最终值、不带 step，不和逐轮 `log(step=...)` 的曲线争抢 step；
       - 返回该 dict。R3 覆盖了 R2-3 原先返回 float 的约定。
  4. `_fit(project_name, experiment_name, model_name) -> dict`：
     1. 日期校验，沿用 DL 的报错消息。
     2. `self.model = self._init_model(num_features=self.num_factors, num_labels=self.num_labels, hyperparameters=self.config.hyperparameters)`。
     3. 按配置日期从 `get_xarray_dataset(["timestamp","symbol"])` 切出 train 与 test。
     4. 四份数组各做一次 `to_array`，再各做一次 `_preprocess`，然后做形状校验。
     5. 尾部切分 `train_split = int(T * (1 - val_size))`：`[:train_split]` 为训练段，`[train_split:]` 为验证段。训练段 0 个时间戳时抛 ValueError；验证段 0 个时间戳时 val 取 None。
     6. 调用一次 `self._fit_model(train_x, train_y, val_x, val_y)`。
     7. 依次评估：`_evaluate("train", ...)`；val 不为 None 时 `_evaluate("val", ...)`；测试段至少 1 个时间戳时 `test_metrics = _evaluate("test", ...)`，否则 `test_metrics = {}`。空切分一律跳过，因为 xgboost 对空输入给出 `(0,0)`。
     8. 调用 `_save_model(Path(model_save_dir)/project_name/experiment_name/model_name)`；recorder 存在时执行 finish。
     9. 返回 `test_metrics`。
  5. `_predict(data)`：`torch.Tensor` 先 `.detach().cpu().numpy()`；非 ndarray 抛 TypeError；返回 `self._forward(self._preprocess(data))`。
  6. `_write_checkpoint(p)`：`MlBackend().to_internal(self.model).write(str(p))`（LD-B7）。
  7. `_read_checkpoint(p)`：`self.model = MlBackend().read(str(p)).get_model()`，不调 `_init_model`（LD-B8）。
  8. 从 `quantlab.ml_model.backend` import MlBackend。这样不会产生循环依赖：该模块只 import `quantlab.base.backend`。

H. `quantlab/utils/module.py`（R-5）
  - `load_model_from_config` 先解析出类 `cls`，再返回 `cls(cls.config_cls(**config))`。
  - DLConfig 的 import 如果不再使用，就删掉。

I. `quantlab/ml_model/backend.py` 的 docstring
  - 删掉「全仓无人使用 MlBackend、`_auto_train` 对 MLConfig 未实现」这一段，改为说明它是 `MLModel._write_checkpoint`/`_read_checkpoint` 的持久化后端（首个调用点来自 260914-lno）。
  - 保留 2026-09-07 那一段。

J. 测试（英文 docstring、autouse 离线 wandb、合成 FakePanel），逐条实现 behavior 块
  - `tests/test_model_hierarchy.py`：契约、配置类型、删除锁（含 MLModel 的 AST 无 deepcopy 引用）、加载器配置类、DLModel ndarray 推理。
  - `tests/test_model_cv.py`：在 A2 golden 基础上追加 `_cv_folds` 几何、两个分支共用生成器（monkeypatch `_cv_folds`）、ML stub 头的 CV 结果、均值 summary、并行等价、DL 结果只含日期且不开 summary run。
  - `tests/test_ml_models.py`：MLModel 编排语义与 `_loss` 默认值。
    - stub 头的 `_fit_model` 记录四个参数的形状、首格取值、是否为 None，并把 `self.model` 设为可 pickle 的 dict。
    - `_init_model` 计数调用次数，并把 num_labels 存进 dict，这样 load 之后 `_forward` 不需要先 collect。
    - wandb 断言通过在类上 monkeypatch `_init_wandb` 实现。
  - `tests/test_metrics.py`：
    - 设置 `pytestmark = pytest.mark.filterwarnings("error::RuntimeWarning")`。
    - 覆盖手算用例、scipy 参考一致性、AST 无循环锁。
    - 参考面板 S=12、约 20% NaN，保证参与参考计算的行至少有 3 个有效值。
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_model_layer.py tests/test_dl_models.py tests/test_ml_backend.py tests/test_factor_hierarchy.py tests/test_model_hierarchy.py tests/test_model_cv.py tests/test_ml_models.py tests/test_metrics.py -q</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && GD=$(git rev-parse --git-dir) && BASE=$(cat "$GD/quick-260914-lno-base") && test -n "$BASE" && NS1=$(git diff --numstat "$BASE" -- tests/test_model_layer.py) && test "$NS1" = "$(printf '3\t3\ttests/test_model_layer.py')" && git diff --quiet "$BASE" -- tests/test_dl_models.py && NS2=$(git diff --numstat "$BASE" -- quantlab/dl_model/rnn.py) && test "$NS2" = "$(printf '2\t2\tquantlab/dl_model/rnn.py')" && NS3=$(git diff --numstat "$BASE" -- quantlab/dl_model/rnn_classification.py) && test "$NS3" = "$(printf '2\t2\tquantlab/dl_model/rnn_classification.py')" && SHA=$(git log --format=%H --grep='pin DL train_cv fold geometry before extraction' "$BASE"..HEAD) && test -n "$SHA" && test "$(git diff-tree --no-commit-id --name-only -r "$SHA")" = "tests/test_model_cv.py" && git diff --quiet "$BASE" "$SHA" -- quantlab/ && echo DIFF-GATES-OK</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && GD=$(git rev-parse --git-dir) && BASE=$(cat "$GD/quick-260914-lno-base") && test -n "$BASE" && git diff -U0 "$BASE" -- uv.lock > "$GD/quick-260914-lno-uvlock.diff" && ! grep -q '^[-+]version = ' "$GD/quick-260914-lno-uvlock.diff" && grep -q '"scipy>=' pyproject.toml && ! grep -q '没有调用点' quantlab/ml_model/backend.py && echo DEP-DOC-GATES-OK</automated>
  </verify>
  <done>
    - 八个测试文件全部通过。其中四个既有回归文件仍为 56 passed，语义未改；A2 的 golden 在重构前后都是绿的，并且有一个独立的、早于任何生产代码改动的提交。
    - DIFF-GATES-OK 与 DEP-DOC-GATES-OK 均有输出。
    - stub ML 头真正经过 `BaseModel.train()` 和 `BaseModel.train_cv()` 写出了 `.joblib`，一个新实例能 load 并 predict；CV 均值写进了 `{cls}_cv_summary` run。
    - 提交 `feat(260914-lno): split BaseModel into DLModel/MLModel with a shared public interface and one CV fold generator`，以及对应的 `test(260914-lno): ...`；只 stage 本任务 files 列表中的路径。
  </done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: XGBoostRegressor(MLModel)——xgb.train + 原生 EarlyStopping(save_best) + 逐轮 wandb 回调 + nthread 透传，含真实学习断言与顺序/并行交叉验证</name>
  <files>quantlab/ml_model/xgb.py, tests/test_xgb_model.py, tests/test_model_hierarchy.py</files>
  <read_first>quantlab/base/model.py 中 Task 1 写出的 MLModel 与 BaseModel.train_cv 全部内容（抽象钩子签名、_fit、_evaluate、_cv_folds、_train_one_fold、_cv_mean_metrics），tests/test_ml_models.py 与 tests/test_model_cv.py 的 FakePanel 与伪 recorder 写法，quantlab/utils/metrics.py 的 regression_panel_metrics</read_first>
  <behavior>
    - 学习测试：
      - 面板：30 个标的、160 个时间戳，前 120 个用于训练（val_size=0.2），后 40 个用于测试。
      - 标签 = 0.1*f_signal + 0.05*噪声，另有两个纯噪声因子。
      - 配置：`early_stopping=True`、patience=20、`{"num_boost_round": 300, "max_depth": 3, "eta": 0.1}`。
      - 用 `predict(test_x)` 算出的测试截面 IC > 0.5，summary 中的 `test_ic` 也 > 0.5。
      - 对照组：标签与因子无关，|test IC| < 0.2。
    - 早停开、纯噪声标签（num_boost_round=500，patience=10）：
      - 从磁盘 `.joblib` 读回的 Booster，`num_boosted_rounds()` 小于 500，且等于它的 `best_iteration + 1`。
      - summary 中的 `best_iteration` 等于该 `best_iteration`，`best_score` 是 float。
    - 早停关（num_boost_round=25）：恰好 25 轮，summary 中没有 `best_iteration`。
    - 早停开但 `val_size=0.0`：
      - 恰好跑满 num_boost_round 轮，不崩溃，并记一条 warning 说明早停被跳过。
      - 逐轮 log 里没有 `val-` 键，summary 里没有 `val_` 键。
    - 验证段有时间戳、但标签全为 NaN：按「没有验证段」处理，不崩溃，恰好跑满 num_boost_round 轮。
    - wandb 逐轮回调：
      - `step` 从 0 开始连续，每轮同时含 `train-rmse` 与 `val-rmse`。
      - 早停触发时，最后一个 step 就是实际运行的最后一轮，由此锁住「回调排在 EarlyStopping 之前」。
      - 实际轮数与 `best_iteration + patience + 1` 的关系，先由执行者在探针中确认后再写成断言。
    - 超参 `{"num_boost_round": 7, "max_depth": 2, "eval_metric": "mae", "nthread": 1}`，早停关：
      - 恰好 7 轮；Booster 配置中 max_depth 为 "2"、`generic_param.nthread` 为 "1"；逐轮键为 `train-mae`。
      - `num_boost_round` 不进入 xgboost params。
      - `config.hyperparameters` 未被修改。
      - 未传 seed 时等于 `config.random_seed`；未传 eval_metric 时默认为 "rmse"；未传 nthread 时 params 中不出现 nthread，交给 xgboost 默认值。
    - 多标签（第二个标签 = -0.1*f_second + 噪声）：predict 返回 `(T,S,2)`，标签 1 的测试 IC > 0.5。
    - NaN/inf：
      - 10% 标签格为 NaN，若干特征格为 ±inf，训练不崩溃。
      - 私有展平 helper 返回的行数等于标签全部有限的行数。
      - `_preprocess` 输出中没有 inf，NaN 增量等于 inf 个数，且输入数组未被修改。
      - 测试集 predict 全部有限。
    - 持久化：未 collect 的新实例 `load(.joblib)` 后，predict 结果与训练实例 `np.array_equal`。
    - `XGBoostRegressor(DLConfig(...))` 抛 TypeError。
    - CV 顺序模式（R3-5）：
      - 设置：160 个时间戳、信号面板、`train_periods=60, gap_periods=2`、`early_stopping=True`、patience=10、`{"num_boost_round": 60, "max_depth": 3, "nthread": 1}`。
      - 返回 8 个结果；日期等于 `XGBoostRegressor._cv_folds(...)` 的输出。
      - 每折目录含 `.joblib` 与 `config.json`。
      - 未 collect 的新实例 `load` 每折后，predict 返回 `[T,S,L]`。
      - 每折 Booster 轮数 ≤ 60，且等于该折 `best_iteration + 1`，说明每折都跑了原生早停。
      - 每折结果含有限的 `test_ic`；`{cls}_cv_summary` run 的 `cv_mean_test_ic` 等于各折 `test_ic` 的均值，且 > 0.3。
    - CV 并行模式：
      - 用 `parallel=True, njobs=2` 与相同超参（`nthread=1`），另用一个 model_save_dir。
      - 折 checkpoint 的相对路径集合与顺序模式相同。
      - 每折加载后的模型，在同一份测试输入上的预测与顺序模式对应折 `np.allclose(atol=1e-6)`。
      - 并行 CV 之后 `config.hyperparameters["nthread"]` 仍为 1，没有被改写。
    - 契约扩展：
      - `XGBoostRegressor.__abstractmethods__ == frozenset()`，且 `issubclass(XGBoostRegressor, MLModel)`。
      - 四个公开方法都未被覆盖。
      - `load_model_from_config` 用真实点分路径 `quantlab.ml_model.xgb.XGBoostRegressor` 构造出的 `type(config) is MLConfig`。
  </behavior>
  <action>
第 0 步：探针（R2-4 的硬性要求）。在仓库**之外**写一个一次性脚本，不提交，确认 xgboost 3.4.1 的以下行为，并把输出原样贴进 SUMMARY：
- `EarlyStopping(save_best=True)` 会把返回的 Booster 截断到 `best_iteration + 1`；
- `best_iteration`/`best_score` 在截断和 joblib 往返后仍然存在；
- 提前停止时，实际运行轮数与 `best_iteration + patience` 的关系；
- 回调顺序对「记录器能否看到停止那一轮」的影响；
- `nthread` 在 `save_config()` 里的位置。
实测结果与 context 中的记录不一致时，以实测为准调整实现和断言，并在 SUMMARY 里写明差异。

新建 `quantlab/ml_model/xgb.py`，写中文 docstring。文件名必须是 `xgb.py`：叫 `xgboost.py` 会遮蔽顶层的 xgboost 包。模块顶部 `import xgboost as xgb`。

类 `XGBoostRegressor(MLModel)`（R2-4、R3-3）只实现四个 ML 抽象钩子，外加超参处理与 wandb 回调；不自己做配置类型检查。

1. 类常量
   - `DEFAULT_PARAMS`：`objective="reg:squarederror"`、`tree_method="hist"`、`eta=0.05`、`max_depth=6`、`subsample=0.8`、`colsample_bytree=0.8`、`device="cpu"`、`eval_metric="rmse"`。不含 seed，也**不含 nthread**：不传时交给 xgboost 默认使用全部核。
   - `DEFAULT_NUM_BOOST_ROUND = 1000`。

2. `__init__(config)`：先调用 super，再把 `_params` 和 `_num_boost_round` 置为 None。

3. `_init_model(num_features, num_labels, hyperparameters)`
   - 在 hyperparameters 的**副本**上 pop 出 `num_boost_round`（默认 1000），转成 int；小于 1 时抛 ValueError。不得修改 config 里的原 dict。
   - `_params` = DEFAULT_PARAMS，再合并 `{"seed": self.config.random_seed}`，最后合并副本；用户键优先，包括用户传入的 `nthread`，原样透传、绝不改写。
   - 返回 None。

4. `_preprocess(data)`：生成**新的** float32 数组，把 ±inf 替换为 NaN，保留 NaN。绝不原地修改调用方的数组。

5. 私有静态 helper `_to_rows(x, y)`：把 `[T,S,F]` 和 `[T,S,L]` 展平成 `(T*S, F)` 和 `(T*S, L)`，只保留所有标签都有限的行，返回 `(x_rows, y_rows)`。

6. 私有回调类 `_WandbEvalCallback(xgb.callback.TrainingCallback)`
   - 构造时持有模型头的引用，调用时再读取 `head._wandb_recorder`。并行 CV 下每折是 deepcopy 出来的头，回调在 `_fit_model` 里现场用 `self` 构造，所以指向的是折副本。
   - `after_iteration(model, epoch, evals_log)`：把每个 `{data_name}-{metric}` 的最新值转成 float；recorder 存在时 `log(dict, step=epoch)`。始终返回 False。
   - 逐轮键使用 xgboost 原生的连字符形式，刻意与 `_evaluate` 写入 summary 的下划线形式区分。

7. `_fit_model(train_x, train_y, val_x, val_y)`，按顺序：
   1. 用 `_to_rows` 得到训练行。没有行时抛 ValueError（训练段没有标签有限的行）；否则构建 `dtrain`。
   2. 如果 `val_x` 不为 None，同样处理：行数 >0 时构建 `dval`；行数为 0 时 `logger.warning`，按没有验证段处理。
   3. evals 为 `[(dtrain, "train")]`，有 `dval` 时再追加 `(dval, "val")`。
   4. callbacks 第一个元素是 `_WandbEvalCallback(self)`。**顺序必须如此**：回调容器是短路调用，排在 EarlyStopping 之后的回调会漏掉触发停止的那一轮（规划时实测）。
   5. `self.config.early_stopping` 为真且有 `dval` 时，再追加 `xgb.callback.EarlyStopping(rounds=self.config.early_stopping_patience, data_name="val", save_best=True)`。
   6. 早停为真但没有 `dval` 时，`logger.warning` 说明早停被跳过，并训练满全部轮数。
   7. 执行 `self.model = xgb.train(self._params, dtrain, num_boost_round=self._num_boost_round, evals=evals, callbacks=callbacks, verbose_eval=False)`。
   8. 早停实际启用且 recorder 存在时，执行 `summary.update({"best_iteration": int(self.model.best_iteration), "best_score": float(self.model.best_score)})`。

8. `_forward(x)`：`self.model.inplace_predict(x.reshape(T*S, F)).reshape(T, S, -1)`。

9. 类 docstring 需要写明：
   - patience 按 boosting 轮数计；早停判据是验证集的 `eval_metric`，默认 RMSE，不是 IC。
   - 多标签时主标签 0 用于头条指标。
   - wandb 两类键的区别。
   - 交叉验证：`train_cv` 继承自 BaseModel，每折在折内尾部验证段上独立做原生早停。`parallel=True` 时各折在 joblib threading 后端上并发运行，xgboost 默认用满全部核，会造成 CPU 超额订阅，建议用户在 hyperparameters 里设 `nthread ≈ 核数 // njobs`；本类不会替用户改写。
   - 用法示例：MLConfig + SpotReturn 标签，`early_stopping=True`、`early_stopping_patience=50`、`hyperparameters={"num_boost_round": 1000}`，以及一个 `train_cv(train_periods=..., gap_periods=..., parallel=True, njobs=4)` 配合 `nthread` 的示例。

测试：新建 `tests/test_xgb_model.py`，英文 docstring，autouse 离线 wandb。FakePanel 支持按公式生成信号标签，并能注入 NaN/inf。逐条实现 behavior 块。
- 学习测试和 CV 测试都用 `quantlab.utils.metrics.regression_panel_metrics` 计算 IC。
- 早停测试读磁盘上的 `.joblib`。
- CV 的 wandb 断言在类上 monkeypatch `_init_wandb`，并把 recorder 登记到测试作用域的 list 里。
- 顺序模式与并行模式分别使用不同的 model_save_dir。

`tests/test_model_hierarchy.py` 扩展两处：
- 把 XGBoostRegressor 加进「出厂头不覆盖公开四方法」的参数化；
- 加一个用真实点分路径的加载器用例。
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && uv run pytest tests/test_xgb_model.py tests/test_model_hierarchy.py tests/test_model_cv.py tests/test_ml_models.py tests/test_metrics.py tests/test_model_layer.py tests/test_dl_models.py tests/test_ml_backend.py tests/test_factor_hierarchy.py -q</automated>
  </verify>
  <done>
    - behavior 块的每一条都有对应测试且全部通过：
      - 学习：测试 IC > 0.5，对照组 |IC| < 0.2。
      - 早停开时，磁盘上的 Booster 轮数 = best_iteration + 1 < num_boost_round；早停关时恰好 num_boost_round 轮。
      - 逐轮回调看得到触发停止的那一轮。
      - CV 顺序模式与并行模式产出相同的折 checkpoint 和一致的预测，均值 summary 正确，nthread 未被改写。
    - 第 0 步的探针输出已贴进 SUMMARY。
    - quantlab/ml_model/xgb.py 存在，quantlab/ml_model/xgboost.py 不存在。
    - 提交 `feat(260914-lno): add XGBoostRegressor with native early stopping and CV support` 与 `test(260914-lno): ...`，只 stage 本任务 files 列表中的路径。
  </done>
</task>

<task type="auto">
  <name>Task 3: 文档同步三层类结构、两套钩子契约、ML 原生早停与 ML 交叉验证，清除「ML 路径未实现 / MlBackend 无调用点」等过期陈述</name>
  <files>example/model.md, example/backend.md, example/README.md, example/factor.md, README.md, CLAUDE.md, train_model.py, tests/test_ml_backend.py</files>
  <read_first>
    - 待改文档：
      - example/model.md 全文
      - example/backend.md 第 1-30 行与第 450-505 行
      - example/README.md 第 40-65 行
      - example/factor.md 第 725-750 行
      - README.md 第 190-225 行与第 360-380 行
      - CLAUDE.md 的 Technology Stack / Component Responsibilities / Pattern Overview / Layers / Error Handling 各节
      - train_model.py 第 60-80 行
      - tests/test_ml_backend.py 第 1-17 行
    - 源码依据：Task 1/2 最终写出的 quantlab/base/model.py 与 quantlab/ml_model/xgb.py。文档中每一处行为描述都必须对照源码重新推导，不得从旧文档抄写。
  </read_first>
  <action>
所有行为描述都以 Task 1/2 的源码为准（LD-E、R-7、R2-6、R3-6）。本任务只改散文，不改可执行代码。

1. `example/model.md`
   1. 更新文件头「代码位置」：列出 `BaseModel`/`DLModel`/`MLModel`、`quantlab/ml_model/xgb.py`、`quantlab/utils/metrics.py`。
   2. 在文件靠前处加一段「2026-09-14 重构：名字对照」：
      - `_train_dl` → `DLModel._fit`
      - `_predict_nn` → `DLModel._predict`
      - `_auto_train` 已删除，`train`/`train_cv` 直接调 `_fit`
      - `to_tensor` 迁到 DLModel，并包装 BaseModel 的 `to_array`
      - `_train_fold_with_config` 改为消费 `_cv_folds` 的折 dict
      有了这段对照，后文「常见坑」里的历史叙述可以保留旧名字而不误导读者。例外：凡写到 to_tensor 所属类的地方（第 53、299、743、801、831 行附近），一律改为 `DLModel.to_tensor`，因为方法的归属确实变了。
   3. 重写描述**当前**结构的各节：
      - 「一句话」。
      - 「基类替你做了什么」表：公开四方法来自 BaseModel，`_fit` 由变体实现，`train_cv` 使用唯一的 `_cv_folds` 并返回逐折结果。
      - 「子类的五方法契约」拆成「DLModel 的五个张量钩子」和新增的「MLModel 的四个钩子与可覆盖默认实现」两节。后者写清：
        - `_fit_model` 负责原生早停，并在返回前留下最优模型；
        - `_loss`、`_compute_metrics`、`_evaluate`（返回带前缀的指标 dict）；
        - `val_x` 为 None 的含义；
        - 为什么 ML 不用 DL 那种按 epoch 的循环：成本（增量验证是线性的，每 epoch 全量重算是二次的）、粒度（每棵树 vs 每个 epoch）、回滚（切片树 vs 拷贝模型）。
      - 「保存与加载」：
        - 按 `checkpoint_suffix` 分流；
        - `.joblib` 经 MlBackend 读写，本质是 pickle，只加载自己信任的文件；
        - `load()` 先校验后缀；
        - ML 路径不调用 `_init_model`。
      - 「关于 W&B」：
        - DL 头按 epoch 记录，`step=epoch`；
        - XGB 头逐轮记录 `train-rmse`/`val-rmse`，用连字符，`step=iteration`；
        - 训练结束写 summary：`{split}_loss`、`{split}_{mse,rmse,mae,r2,ic,rank_ic}`（下划线），以及 `best_iteration`/`best_score`；
        - CV 结束时单独开一个 `{cls}_cv_summary` run，写入 `cv_mean_test_*` 与 `cv_n_folds`。
   4. 把「已知的不完整之处 #1」替换为「已实现：ML / 树模型路径」，保留编号，注明 2026-09-14 实现。
   5. 新增「ML / 树模型」一节，包含两个示例：
      - 一个合成 FakePanel + XGBoostRegressor 的 CPU 最小示例，用 `num_boost_round`、`early_stopping=True`、`early_stopping_patience`，并在同一脚本里再跑一次小规模的 `train_cv`。执行者必须把它写成仓库**之外**的一次性脚本（放在自己的临时目录，不提交），用 `WANDB_MODE=disabled uv run python <脚本>` 实际运行，把真实输出贴进文档；临时路径按现有文档的写法缩写。
      - 一个 MLConfig + SpotReturn 标签的真实数据用法示例。它依赖本机不存在的行情数据，按本目录约定明确标注「此例未实际运行」。
   6. 在「ML / 树模型」节下新增小节「ML 交叉验证」，写清以下内容：
      - 折几何：滚动前推；`test_periods = train_periods // 5`；折数公式；超出数据范围的折被跳过；`train_periods` 的单位是时间点个数。
      - `gap_periods` 的用途：隔开标签自身的前视窗口。`SpotReturn` 用 `shift` 生成标签，不留 gap 时，训练段末尾那几根 bar 的标签已经包含了测试段开头的信息。
      - 每折内部：训练段尾部按 `val_size` 切出验证段，驱动该折的原生早停；每折有独立的 wandb run 和 `.joblib`。
      - 返回值：逐折结果 list 的键；均值写在 `{cls}_cv_summary` run 里。选择独立 run 的原因：每折 run 在 `_fit` 结束时已经 finish。
      - 并行：`parallel=True` 在 joblib threading 后端上为每折 `copy.deepcopy(self)`，面板数据会被复制 njobs 份。xgboost 默认用满全部核，建议 `nthread ≈ 核数 // njobs`，代码不会替用户改写。
   7. 「扩展：接入一个新模型」里的 TinyRegressor 示例改为继承 DLModel，同样在仓库外实际重跑，并替换「真实输出」代码块：来自已删除方法的日志行不会再出现。绝不手写输出。

2. `example/backend.md` 与 `example/README.md`
   - backend.md 第 501 行附近、README.md 第 60 行附近：把 MlBackend 的现状改成「`MLModel` 的 checkpoint 持久化后端」。
   - backend.md 第 23 行关于 BaseModel 装数据的描述保留不动，它仍然正确。
   - README.md 第 54 行 `_train_dl(backtest=...)` 的历史叙述，改为 `DLModel._fit(backtest=...)`。

3. `example/factor.md` 第 734 与 744 行附近、`train_model.py` 第 70 行的注释：把 to_tensor 的归属类改为 DLModel。train_model.py 只改注释行。

4. `README.md`
   - `quantlab/base/` 条目：列出 BaseModel/DLModel/MLModel。
   - `quantlab/dl_model/` 条目：写明各头继承 DLModel。
   - `quantlab/ml_model/` 条目：改为「MlBackend + XGBoostRegressor（`xgb.py`，未来收益回归，原生早停，支持 train_cv）」。
   - 第 373 行附近的 MLConfig 条目：写明 early_stopping 与 early_stopping_patience（对树模型按轮数计），以及 XGBoostRegressor。

5. `CLAUDE.md`
   - Component Responsibilities 表：
     - BaseModel 行改为「框架无关的共享生命周期、公开接口与 CV 折几何」；
     - 新增 DLModel、MLModel、XGBoostRegressor 三行；
     - 新增 metrics 模块一行。
   - Pattern Overview 中「model layer is torch-centric / every concrete model implements the same 5-method contract」那句，改为三层结构的描述。
   - Layers 中 Model 层的 Location/Purpose 一并更新。
   - Technology Stack：
     - joblib 条目改为经 `ml_model/backend.py:MlBackend` 持久化非 torch 模型；
     - Key Dependencies 增加 xgboost（`ml_model/xgb.py`，原生早停）与 scipy（`utils/metrics.py` 的截面 RankIC）。
   - Error Handling 节里以 MLConfig 分支为例的那句，换成仍然成立的例子：`DLModel._fit(backtest=True)` 抛 NotImplementedError 并点名 Phase 6。
   - 不改 Developer Profile 等由工具管理的节。

6. `tests/test_ml_backend.py`：只改模块 docstring 里说 MlBackend 无调用点的那两句，改为它现在是 MLModel 的持久化后端。不改任何代码与断言。
  </action>
  <verify>
    <automated>cd /Users/daizhaorong/projects/quantlab && test "$(tr '\n' ' ' < README.md | tr -s ' ' | grep -o 'no concrete' | wc -l | tr -d ' ')" = 0 && ! grep -q 'no concrete' CLAUDE.md && ! grep -q 'ML training not implemented' CLAUDE.md && ! grep -q 'ML training not implemented' example/model.md && ! grep -q '没有任何调用点' example/backend.md && ! grep -q '零调用点但' example/README.md && ! grep -q '仓库里没有任何地方使用它' example/model.md && ! grep -q 'zero-call-site' tests/test_ml_backend.py && ! grep -rq 'BaseModel.to_tensor' example/ train_model.py && echo STALE-GATES-OK</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && test "$(grep -o 'MLModel' example/model.md | wc -l)" -ge 5 && test "$(grep -o 'DLModel' example/model.md | wc -l)" -ge 5 && grep -q 'XGBoostRegressor' example/model.md && grep -q 'num_boost_round' example/model.md && grep -q 'ML 交叉验证' example/model.md && grep -q 'cv_mean_test_' example/model.md && grep -q 'nthread' example/model.md && grep -q 'gap_periods' example/model.md && grep -q 'XGBoostRegressor' README.md && grep -q 'XGBoostRegressor' CLAUDE.md && grep -q 'DLModel' CLAUDE.md && echo POSITIVE-GATES-OK</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && GD=$(git rev-parse --git-dir) && BASE=$(cat "$GD/quick-260914-lno-base") && test -n "$BASE" && T=$(mktemp -d) && for f in train_model.py tests/test_ml_backend.py; do git show "$BASE:$f" > "$T/$(basename "$f")" || exit 1; uv run python -c "import ast,sys; body=lambda p: [ast.dump(n) for n in ast.parse(open(p).read()).body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str))]; sys.exit(0 if body(sys.argv[1]) == body(sys.argv[2]) else 1)" "$T/$(basename "$f")" "$f" || { echo "NOT PROSE-ONLY: $f"; exit 1; }; done && echo PROSE-ONLY-OK</automated>
    <automated>cd /Users/daizhaorong/projects/quantlab && GD=$(git rev-parse --git-dir) && test -f "$GD/quick-260914-lno-baseline-failures" && { uv run pytest -q -p no:cacheprovider -rf > "$GD/quick-260914-lno-final-pytest.txt" 2>&1; true; } && grep -Eq '[0-9]+ passed' "$GD/quick-260914-lno-final-pytest.txt" && { grep '^FAILED' "$GD/quick-260914-lno-final-pytest.txt" | sed 's/ - .*//' | sort -u > "$GD/quick-260914-lno-final-failures"; true; } && test -z "$(comm -13 "$GD/quick-260914-lno-baseline-failures" "$GD/quick-260914-lno-final-failures")" && test "$(grep -Eo '[0-9]+ passed' "$GD/quick-260914-lno-final-pytest.txt" | tail -1 | grep -Eo '[0-9]+')" -gt "$(grep -Eo '[0-9]+ passed' "$GD/quick-260914-lno-baseline-pytest.txt" | tail -1 | grep -Eo '[0-9]+')" && tail -1 "$GD/quick-260914-lno-final-pytest.txt" && echo NO-NEW-FAILURES</automated>
  </verify>
  <done>
    - STALE-GATES-OK、POSITIVE-GATES-OK、PROSE-ONLY-OK、NO-NEW-FAILURES 四道门全部输出。
    - 全量测试的失败集合是 Task 1 步骤 A 基线失败集合的子集，passed 数严格增加。
    - example/model.md 中标注为真实输出的代码块都来自本次实际运行。
    - 提交 `docs(260914-lno): document the model hierarchy, native early stopping and ML cross-validation`，只 stage 本任务 files 列表中的路径。
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| 磁盘 checkpoint -> `load()` | `.joblib` 经 `MlBackend.read` 走 joblib/pickle 反序列化；`.pth` 经 `torch.load` |
| `config.json` 的 `name` -> `get_cls_from_path` | 点分路径被 importlib 导入（既有机制；本计划只改变配置类的选择方式） |
| 训练进程 -> wandb | 在线模式下，config 字典、逐轮 eval 值、最终指标、CV 均值会被发送到 W&B |
| 包管理 -> 环境 | 在 pyproject 中声明 scipy |
| 执行者 -> 用户工作树 | 用户有未提交改动，提交步骤可能误带 |

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-260914-01 | Tampering / Elevation | `MLModel._read_checkpoint` -> `MlBackend.read` (pickle) | medium | accept | 单用户本地研究后端：checkpoint 由同一用户训练产生，不从网络接收。Task 3 在 example/model.md「保存与加载」中写明 `.joblib` 是 pickle，只能加载自己信任的文件。`load()` 的后缀校验防止把错误类型的文件交给错误的反序列化器。 |
| T-260914-02 | Tampering | `load_model_from_config` -> `cls.config_cls(**config)` | low | mitigate | 用 `cls.config_cls` 取代硬编码的 DLConfig，再由 BaseModel setter 做 isinstance 检查，MLConfig 字典不能再静默建出 DLConfig 并走 torch 路径。由 tests/test_model_hierarchy.py 的加载器用例和配置类型用例锁住。 |
| T-260914-03 | Information Disclosure | `_init_wandb` / `_WandbEvalCallback` / `MLModel._evaluate` / CV summary run | low | accept | 发送的是既有 config 字典（路径、超参）和数值指标，不含凭证；本计划不新增任何敏感字段。测试一律使用 `WANDB_MODE=disabled`。 |
| T-260914-04 | Denial of Service | `train_cv(parallel=True)` 的 CPU 超额订阅与 deepcopy 内存；`_fit_model` 的 DMatrix | low | accept | 并行折复制面板数据、并发运行 xgboost，是用户显式选择 `parallel=True` 的代价。xgb.py 的 docstring 与 example/model.md 给出 `nthread ≈ 核数 // njobs` 的建议，并有测试断言 `nthread` 不被改写。DMatrix 是 `_fit_model` 的局部变量，不挂在实例上，也不妨碍 deepcopy。 |
| T-260914-05 | Tampering | 提交步骤误带用户未提交文件 | medium | mitigate | 每次提交只用显式路径 `git add`，禁止 `-A`/`.`/`-a` 与 `git stash`。提交前执行者以 `git status --short` 确认 `quantlab/enums/data.py`、`test.py`、`quantlab/acquisition/wrds.py` 均未被 stage。 |
| T-260914-SC | Tampering | `uv add "scipy>=1.18.1"` | low | mitigate | 不引入新代码：规划时 scipy 1.18.1 已作为 scikit-learn 的依赖锁在 uv.lock 中。Task 1 的 verify 断言 uv.lock diff 中没有任何 `version = ` 行变化。无 [ASSUMED]/[SUS] 包，因此不需要人工合法性检查点。 |
</threat_model>

<verification>
- 以下命令全部通过：`uv run pytest tests/test_model_layer.py tests/test_dl_models.py tests/test_ml_backend.py tests/test_factor_hierarchy.py tests/test_model_hierarchy.py tests/test_model_cv.py tests/test_ml_models.py tests/test_metrics.py tests/test_xgb_model.py -q`。
- 全量 `uv run pytest` 与 Task 1 步骤 A 的基线失败集合相比，没有新增失败，且 passed 数严格增加（Task 3 的 NO-NEW-FAILURES 门）。
- 以下门全部有输出：Task 1 的 DIFF-GATES-OK（其中包含「golden 提交早于抽取」的检查）与 DEP-DOC-GATES-OK；Task 3 的 STALE-GATES-OK、POSITIVE-GATES-OK、PROSE-ONLY-OK。
- `git status --short` 中 `quantlab/enums/data.py`、`test.py`、`quantlab/acquisition/wrds.py` 仍为未 stage 的原状。
</verification>

<success_criteria>
- 用户可以用 MLConfig 训练 XGBoostRegressor 预测未来收益率：
  - 最优模型由 xgboost 原生早停选出：patience 按轮数计，判据是验证集 RMSE，save_best 截断。
  - wandb 逐轮记录 eval 曲线；summary 记录最终的 loss、IC/RankIC 等指标，以及 best_iteration。
  - `.joblib` 可以 load 回来并 predict 出 `[T,S,L]`。
- 用户可以对 XGBoostRegressor 做滚动交叉验证：
  - 顺序与并行两个分支经同一个折生成器产出相同的折；每折独立做原生早停并落盘。
  - 返回逐折结果；CV 均值写入 wandb summary；nthread 原样透传。
- 模型层是 BaseModel / DLModel / MLModel 三层：两套抽象钩子各自独立，公开的 train/train_cv/load/predict 只实现一次，并由契约测试锁定。
- DL 的按 epoch 早停语义与 train_cv 折几何逐字不变，分别由既有测试和重构前捕获的 golden 锁定。
- 既有 56 条回归测试零语义改动并全部通过；全量测试没有新增失败。
- 文档与 CLAUDE.md 描述的就是现在的代码。
</success_criteria>

<output>
完成后创建 `.planning/quick/260914-lno-basemodel-mlconfig-dl-save-load-predict-/260914-lno-SUMMARY.md`，其中必须记录：
- 全量测试的基线汇总行与最终汇总行，以及基线 SHA；
- R-2 的「调用改名，不留别名」决定、R2-2 的「DL 早停保持内联」决定、R3-4 的「均值写入独立 `{cls}_cv_summary` run」决定；
- A2 golden 提交的 SHA，证明它早于任何生产代码改动；
- scipy 声明前后 uv.lock 的 version 行 diff 为空的证据；
- Task 2 第 0 步 xgboost 探针的原样输出；
- example/model.md 中两个实际运行示例的运行命令。
</output>
