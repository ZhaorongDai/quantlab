# Requirements: quantlab

**Defined:** 2026-09-04
**Core Value:** 一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。

## v1 Requirements

### Cleanup（现有代码库整理，前置工作）

- [ ] **CLEAN-01**: 仓库通过全新 Git 历史初始化，不包含现有已泄露 Tiingo API Key 的提交
- [ ] **CLEAN-02**: 删除/清理测试代码、临时脚本、明显冗余或不适合长期维护的实现（如 `vecbt/bt.py` 未完整传参的 `backtest_from_signals`、`get_binance_instruments.py` 与 `utils/binance.py` 的重复解析逻辑）
- [ ] **CLEAN-03**: `README.md` 内容与实际代码结构保持一致
- [ ] **CLEAN-04**: `pyproject.toml`/`uv.lock`（通过 `uv` 管理）正确声明并锁定实际依赖，`uv sync` 后环境可用

### Security

- [ ] **SEC-01**: Tiingo 等第三方 API Key 统一通过环境变量读取，代码中不出现硬编码密钥

### Data（数据获取与预处理）

- [ ] **DATA-01**: 用户可以从 Tiingo 拉取美股日频行情数据并写入 xarray/Zarr 存储
- [ ] **DATA-02**: 用户可以复用/整理现有币安现货数据接入，写入同一套存储抽象
- [ ] **DATA-03**: 数据层抽象（`Dataset`/`DataBackend`）在设计上支持按市场（美股/加密...）与频率（日频/分钟频/tick）扩展——新增一种市场或频率不需要改动上层因子/模型/回测代码；v1 至少用两种不同的市场或频率组合验证该扩展性（如美股日频 + 币安现货）
- [ ] **DATA-04**: 提供数据清洗/预处理模块（复用现有 `my_ops` 标准化算子），输出统一的 `xarray.Dataset`
- [x] **DATA-05**: 用户可以获得标普 500 与纳斯达克 100 的日频 point-in-time 成分面板（`xarray.Dataset`，dims `timestamp`/`symbol`，布尔变量 `is_member`），在各自可回溯区间内无幸存者偏差；超出可回溯起点的查询必须显式报错而非静默返回不完整名单
- [x] **DATA-06**: 成分股数据类与行情数据类共享同一 `BaseDataset` 抽象——成分股类不继承任何 OHLCV 专用成员（`_to_kunquant`/`_to_nautilus`），新增一类指数不需要改动上层代码
- [ ] **DATA-07**: 外部调用方可以不指名 vendor 类、`Dataset` 子类或 `ingest_*.py` 脚本，就通过注册表把已落盘的 raw parquet 层转换为 Zarr 层，并拿到一个描述本次转换结果的对象（写入/跳过的窗口、pinned 符号数、存储路径、是否续跑或被取消）
- [ ] **DATA-08**: 任何调用方可以在任何内存被分配之前，问出一次转换的预测峰值内存与每个超预算窗口的补救建议；该答案是一个可被调用方渲染的值，而不是这一层打印的日志行

### Factor（因子计算）

- [x] **FACTOR-01**: KunQuant 后端支持批量计算 Alpha158 因子集，输出 `xarray.Dataset`
- [x] **FACTOR-02**: KunQuant 后端保留流式（`cal_stream`）计算能力，为未来实时数据接入预留接口
- [x] **FACTOR-03**: 新增 Polars 批量因子计算后端接口，用于实现新因子（不要求复刻 Alpha101/Alpha158 已有公式），不需要支持流式
- [x] **FACTOR-04**: 因子计算模块间的数据传输统一使用 `xarray.Dataset`，不使用 DataFrame 作为流水线传输格式

### Model（收益预测模型）

- [ ] **MODEL-01**: 提供一个简单基线收益预测模型（v1，如线性回归），输入因子 `xarray.Dataset`，输出未来收益或收益排名预测
- [ ] **MODEL-02**: 模型训练/推理直接消费 `xarray.Dataset`，不经过 DataFrame 中转

### Portfolio（组合优化）

- [ ] **PORT-01**: 提供一个简单基线组合优化模型（v1，如均值方差或等权），输入收益/排名预测，输出每个标的的目标持仓百分比
- [ ] **PORT-02**: 组合优化支持多空持仓，不加杠杆（净敞口与毛敞口均 ≤100%）

### Backtest（回测）

- [ ] **BT-01**: 基于 vectorbt 的向量化回测打通：输入目标持仓，输出回测结果（收益曲线、关键指标等）
- [ ] **BT-02**: 预留事件驱动回测扩展能力（NautilusTrader），整理现有 `backtest/test_strategy.py` 集成使其至少可运行，不要求完整生产化

### Config（可复现性）

- [ ] **CFG-01**: 流水线各阶段（数据/因子/模型/组合优化/回测）的参数通过配置文件驱动，同一份配置可复现同一次实验结果

### Architecture（架构契约）

- [ ] **ARCH-01**: 数据模块、因子模块、收益模型、组合优化模块之间通过明确的输入输出契约组合（数据→因子→收益预测/排名→目标持仓百分比），任一模块可独立替换而不影响其他模块
- [ ] **ARCH-02**: 架构同时兼容单标的时序策略与多标的截面多因子策略两种使用方式

### Quality（代码质量）

- [ ] **QUAL-01**: 核心模块（数据/因子/模型/组合优化/回测）具备单元测试，测试风格与项目代码风格一致
- [ ] **QUAL-02**: 代码遵循 Python 之禅与模块化组织，函数/类具备必要的说明及参数/返回值类型标注，不包含不必要的代码

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Platform（平台化）

- **PLAT-01**: 后台服务化（提供 API 层）
- **PLAT-02**: 多用户支持与隔离
- **PLAT-03**: 因子在线编辑与测试工具
- **PLAT-04**: 模型在线测试工具
- **PLAT-05**: 组合回测的在线交互能力
- **PLAT-06**: 网页前端

### Data v2（数据深化）

- **DATA-V2-01**: Tick 级别数据的完整生产级接入（历史回补、多市场统一 tick 存储）
- **DATA-V2-02**: 分钟频数据的完整历史回补与多市场覆盖（v1 只需架构验证，不要求全量历史）

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| 网页前端 | 用户明确要求当前阶段只做后台 |
| 后台服务化 / 多用户 | 属于"预估未来平台化发展"，非当前里程碑；架构需要为此预留空间，但不在 v1 实现 |
| 因子 / 模型在线编辑测试平台 | 同上，未来平台化功能，非 v1 |
| Tick 数据完整生产级接入 | v1 只需架构上可扩展支持，完整实现（历史回补、全市场覆盖）留到后续阶段，避免 v1 范围过大 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 1 | Pending |
| CLEAN-02 | Phase 1 | Pending |
| CLEAN-03 | Phase 1 | Pending |
| CLEAN-04 | Phase 1 | Pending |
| SEC-01 | Phase 1 | Pending |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DATA-04 | Phase 2 | Pending |
| DATA-05 | Phase 03.1 | Complete |
| DATA-06 | Phase 03.1 | Complete |
| DATA-07 | Phase 03.5 | Pending |
| DATA-08 | Phase 03.5 | Pending |
| FACTOR-01 | Phase 3 | Complete — accepted defect (gap 2 dismissed; re-open if VWAP-derived features reach a Phase-4 model or Phase-6 backtest — see 03-VERIFICATION.md § Gap Dispositions) |
| FACTOR-02 | Phase 3 | Complete |
| FACTOR-03 | Phase 3 | Complete |
| FACTOR-04 | Phase 3 | Complete |
| MODEL-01 | Phase 4 | Pending |
| MODEL-02 | Phase 4 | Pending |
| PORT-01 | Phase 5 | Pending |
| PORT-02 | Phase 5 | Pending |
| BT-01 | Phase 6 | Pending |
| BT-02 | Phase 6 | Pending |
| CFG-01 | Phase 6 | Pending |
| ARCH-01 | Phase 6 | Pending |
| ARCH-02 | Phase 6 | Pending |
| QUAL-01 | Phase 7 | Pending |
| QUAL-02 | Phase 7 | Pending |

**Coverage:**

- v1 requirements: 26 total
- Mapped to phases: 26 (all v1 requirements covered)
- Unmapped: 0

---
*Requirements defined: 2026-09-04*
*Last updated: 2026-09-11 — Phase 03.5 added DATA-07/DATA-08*
