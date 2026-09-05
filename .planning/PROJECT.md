# quantlab

## What This Is

一个端到端的量化研究后端平台：从多市场、多频率行情数据出发，经因子计算、收益预测、组合优化，生成目标持仓并完成回测，全流程通过配置文件驱动、可复现。当前阶段只做后台，面向未来平台化（服务化、多用户、因子/模型在线编辑与测试、网页前端）预留架构空间，但不在本阶段实现。

## Core Value

一条打通的、config 驱动可复现的量化流水线（数据→因子→收益模型→组合优化→目标持仓→回测→结果），模块间用清晰的输入输出契约组合，任何一环都能独立替换/扩展而不需要推倒重来。

## Requirements

### Validated

(None yet — ship to validate)

<!-- 现有仓库中的原型代码（分层架构、KunQuant 因子引擎、xarray 存储、torch 模型训练、vectorbt/nautilus 回测）是有价值的设计参考和部分可复用实现，但尚未作为"已验证需求"锁定——用户计划先审查、清理测试/临时/冗余代码后再决定复用范围，因此不视为 Validated，相关内容记录在 Context 中。 -->

### Active

- [ ] 数据层：从设计上支持多市场（美股 / 币安现货加密）与多频率（日频、分钟频、tick）扩展；v1 至少打通美股日频（Tiingo）与已有的币安现货数据接入
- [ ] 数据清洗与预处理模块，统一输出为 xarray.Dataset（`[timestamp, symbol]` 规范形状）
- [ ] 因子计算：KunQuant 后端复用 Alpha158（批量 + 流式，含未来实时数据接入能力）
- [ ] 因子计算：新增 Polars 批量计算后端接口（仅面向新增因子，不要求复刻 alpha101/158 公式），不做流式
- [ ] 收益模型：v1 用简单基线模型（如线性回归）输出未来收益或收益排名预测，模型训练直接消费 xarray（不经过 DataFrame）
- [ ] 组合优化模型：v1 用简单基线方法（如均值方差/等权），支持多空、不加杠杆（净/毛敞口 ≤100%），输出每个标的的目标持仓百分比
- [ ] 生成目标持仓并落盘
- [ ] 向量化回测（基于 vectorbt）打通，输出回测结果
- [ ] 预留事件驱动回测扩展点（NautilusTrader），整理现有集成但非 v1 交付重点
- [ ] 全流程参数尽可能通过配置文件驱动，保证实验可复现
- [ ] 敏感凭证（如 Tiingo API Key）一律通过环境变量读取，不硬编码
- [ ] 架构同时兼容单标的时序策略与多标的截面多因子策略
- [ ] 核心模块具备符合项目风格的单元测试
- [ ] 代码风格：清晰模块化、遵循 Python 之禅、必要的函数/类注释与参数/返回值标注；清理不适合长期维护的测试/临时/冗余代码
- [ ] Git 历史重置为全新仓库（不保留当前含已泄露 API Key 的提交）

### Out of Scope

- 网页前端 — 用户明确要求当前阶段只做后台
- 后台服务化、多用户、因子/模型在线编辑测试平台 — 用户列为"预估未来平台化发展"，非当前里程碑，但架构设计需要为此预留空间
- Tick 级数据的完整生产级接入（历史回补、实时流全覆盖）— 架构需要支持，但完整实现范围留待后续阶段细化，避免 v1 范围过大

## Context

**现有代码库现状（brownfield）：**
仓库中已有一份约 6000 行的个人量化研究原型（README 显示旧项目名 "Crypto Quant Trading Models"，且已与当前代码结构脱节）。已建立的分层架构：`DataBackend → Dataset → Factor/Label → Model → Backtest`，每层一个抽象基类（`base/`），具体实现在同名顶层包（`dataset/`、`factor/`、`label/`、`dl_model/`/`ml_model/`、`backtest/`/`vecbt/`）。`xarray.Dataset`（`[timestamp, symbol]`）已经是层间唯一的内存交换格式。因子计算通过 KunQuant 声明式算子图（`Builder/Input/Output`）编译为原生代码执行，`factor/alpha101.py`、`factor/alpha158.py` 直接复用 `KunQuant.predefined` 现成因子库。Polars 目前仅用作数据只读视图（`_get_lazyframe()`），未参与因子计算——这是本次要新增的能力。配置用 dataclass（`base/config.py`），但目前存在需要清理的问题。

**已发现的具体问题（清理阶段需要处理）：**
- `config/__init__.py` 的工厂函数硬编码了不同开发机的绝对路径（Linux `/home/zhrdai/...` 与本机 macOS 路径混杂），新环境必须手改才能跑通
- `pyproject.toml` 声明零依赖、`uv.lock` 也过期（锁的是旧项目 `crypto-quant` 的 3 个包），而实际代码 import 了约 20 个第三方包（torch、xarray、polars、KunQuant、vectorbt、nautilus_trader、wandb 等），`uv sync` 目前无法得到可用环境
- **`scripts/download_stock_data_from_tiingo.py` 硬编码了一个 Tiingo API Key，且该文件已随 `Initial commit` 推送到 GitHub（`origin/main`），Key 处于泄露状态**——用户需尽快在 Tiingo 后台吊销/轮换；清理阶段要把该脚本改为从环境变量读取，且新的 git 历史不应再包含这个 key
- `README.md` 内容与实际代码结构不符（提到不存在的 `models/lstm_model.py`、`examples/train_models.py` 等）
- `vecbt/bt.py:backtest_from_signals` 调用 `vbt.Portfolio.from_signals()` 时未传参数，运行会直接报错
- `get_binance_instruments.py` 与 `utils/binance.py` 存在重复实现的解析逻辑
- 无测试、无 lint/CI 配置

**代码分级——哪些是可复用的参考实现，哪些是临时代码：**
用户明确要求：规划/执行时要区分仓库里的"关键代码"（`base/`、`dataset/`、`factor/`、`label/`、`my_ops/` 等已建立的分层架构，是新代码应该遵循的设计参考）和"临时代码"（目前 `scripts/` 目录下的脚本，如 `scripts/download_stock_data_from_tiingo.py`，是一次性/探索性代码，不代表项目架构风格）。**尤其是数据持久化相关的新代码，不要参照 `scripts/` 里的临时实现（如直接用 Polars 写 parquet 的写法），而要按项目已有风格（`Dataset`/`DataBackend` 抽象、xarray+Zarr 落盘）重新实现。** 这条约束对 Phase 2（数据基础设施）尤其重要。Phase 1 对 `scripts/download_stock_data_from_tiingo.py` 只做最小的安全/路径修复（不重构、不作为架构参考），不违反这条约束。

**技术栈背景：**
- 包管理使用 `uv`（用户已确认，`pyproject.toml`/`uv.lock` 已存在但内容过期，需要重建）
- KunQuant 本身只提供 `Alpha101`、`Alpha158` 两套预定义因子库，没有 Alpha191——用户确认此前提到的"alpha191"是对 alpha158 的笔误/混称，沿用现有 Alpha158 即可

## Constraints

- **数据格式**: 模块间统一使用 xarray（Zarr 落盘），不使用 DataFrame 作为流水线层间传输格式；模型训练直接消费 xarray — 用户明确要求，是贯穿整个流水线的硬约束
- **因子计算后端**: 双后端支持——KunQuant（批量 + 流式，保留未来实时数据接入能力）为主，Polars 为新因子的补充计算路径（仅批量，不需要流式）；能用 xarray/KunQuant 完成的处理，优先不用 Polars — 用户明确的技术选型优先级
- **回测技术栈**: 向量化回测优先用 vectorbt 打通；事件驱动回测（NautilusTrader，现有 `backtest/test_strategy.py` 已有雏形）作为预留扩展能力，非 v1 交付重点
- **凭证安全**: API Key 等敏感信息一律通过环境变量读取，不硬编码 — 现有代码已经因硬编码 Tiingo Key 造成一次真实泄露
- **可复现性**: 全流程参数尽量通过配置文件驱动 — 用户明确要求，服务于实验可复现
- **架构契约**: 数据模块输出数据、因子模块输出因子、收益模型输出未来收益/收益排名预测、组合优化模型输出每个标的目标持仓百分比——各模块通过清晰的输入输出契约组合 — 便于未来插拔式扩展与平台化
- **包管理**: 使用 `uv` — 用户明确要求，延续现有项目的包管理方式
- **范围**: 当前阶段只实现后台，不做网页前端 — 用户明确排除

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v1 因子集沿用 KunQuant 预定义的 Alpha158，不新建 Alpha191 | KunQuant 本身无 Alpha191 预定义集；用户确认之前所说"alpha191"是笔误，指的就是 alpha158 | — Pending |
| 新增的 Polars 因子计算后端只覆盖未来新因子，不要求把 alpha101/alpha158 现有公式在 Polars 里重新实现一遍 | 避免同一套因子公式维护两份实现的成本 | — Pending |
| 持仓模型采用多空、不加杠杆（净/毛敞口 ≤100%） | 用户确认 | — Pending |
| 数据层从设计上支持多市场（美股/加密）与多频率（日/分钟/tick），不是只做美股日频 | 用户强调需要从一开始就把扩展性设计进去，避免后续推倒重来 | — Pending |
| v1 交付全链路（数据→因子→收益模型→组合优化→目标持仓→回测），每一步先用简单/基线实现打通 | 用户确认，保证架构契约在所有模块间都被验证过，后续再逐步替换/增强各环节复杂度 | — Pending |
| Git 历史重置为全新仓库，不保留当前含泄露 Key 的提交 | 现有 `Initial commit` 已推送 GitHub 且包含硬编码的 Tiingo Key | — Pending（将在清理阶段执行，执行前会再次与用户确认） |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-09-04 after initialization*
