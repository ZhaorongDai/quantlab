"""点时点、规则驱动的股票池过滤器，实现成一个**因子包装类**（260915-p91）。

`UniverseFilteredFactor` 包住任意一个 KunQuant 因子或标签，自己也是一个
`FactorKunQuant`。所以它能原样放进 `MLConfig.factors` / `MLConfig.labels`，也能
原样放进回测配置里的模型——模型层与回测层**一行都不用改**。

为什么需要它：美股 Alpha101 -> 5 日 Return -> XGBoostRegressor ->
`USEquityCrossectionSelectStockVectorBt` 这条链在 ~7,700 个 Tiingo `us_all`
标的上跑出过 244 天 +886,077,331% 的「收益」，且 `best_iteration=0`。原因是
认股权证、权利、单位、测试代码和低价股（`AACIW`、`AAC-WS`、`AACBR`、`AACBU`、
`ZWZZT` 0.007 -> 10.05）——57 次强平里约 50 次是这类标的。用一个真实交易员会用的
点时点规则过滤股票池，就能把它们同时从**截面**和**交易清单**里去掉。

锁定语义（LS-1..LS-5，用户已锁，不要重新讨论）：

LS-1 股票池掩码，点时点。标的在 t 时刻在池内，当且仅当：
  (a) 代码是普通股（一条与日期无关的静态规则）；
  (b) **原始** `close[t] >= min_price`（默认 5）；
  (c) 截至 t 的 `window` 根（默认 20）**原始** `close*volume` 均值
      `>= min_dollar_volume`（默认 1,000,000）；窗口不满即出池。
  一律用**原始**（raw）close/volume，绝不用复权列：复权历史会被拆股和分红压低，
  拿复权价判断「当时是不是低价股」是错的。不使用 `anomaly_flag`。
  掩码在池内为 1.0、出池为 NaN，铺在 `(timestamp, symbol)` 上。
  t 之后的任何一根 bar 都不会改变 t 时刻的掩码。

LS-2 截面算子只看池内标的，时序算子看完整历史。图改写：每个
  `CrossSectionalOp` 的每个输入 v 换成 `Div(v, universe_mask)`，掩码作为一个额外
  的图 `Input`，批量（`runGraph`）与流式（`StreamContext`）两条路都要喂。
  除以 1.0 不改变数值，除以 NaN 得到 NaN——于是出池标的在截面算子眼里根本不存在。

LS-3 算完之后，因子输出**和标签**在掩码为 NaN 的位置置 NaN。标签来自**未过滤**的
  价格，且只看标签**自己那个时间戳 t** 的掩码，绝不看 t+h 的池状态。
  标的轴（2026-09-15 用户决定，取代原先的「整窗 NaN 就删列」）：**只删**被静态
  代码规则排除的标的（权证、单位、权利、测试代码），且在**每一个**窗口里都删；
  仅仅因为价格或成交额阈值而整窗 NaN 的标的**保留**为 NaN 列。
  为什么：按「整窗 NaN」删列会让标的轴依赖日期窗口，而
  `DLModel._align_prediction_symbols` 在面板缺少训练过的标的时会直接报错。代码
  规则与日期无关，所以它删掉的标的在任何窗口里都不存在，也就不可能被训练过。

LS-4 回测器不改，`price_dataset` 保持未过滤。掉出池的持仓拿到全 NaN 特征，
  `predict_panel` 于是给出 NaN 预测，下一个调仓日不再被选中，并在再下一根 bar 的
  开盘卖出。`rebalance_periods=5` 时最多晚 4 根 bar，`=1` 时就是次日。

LS-5 已接受的代价：截面算子之上的时序算子（如 `correlation(rank(x), rank(y), 10)`）
  在标的（重新）入池后的那个窗口内是 NaN。这与真实交易一致。

两个 KunQuant 硬性约束，新调用方必须遵守：

- **批量一律 `start=0`。** KunQuant 0.1.11 的 `CrossSectionalDataHolder` 在
  `num_time` 被赋值之前就用它算 `base_time`，导致 `start>0` 时**所有**
  `GenericCrossSectionalOp` 结果错误且不确定（见
  `quantlab/my_ops/preprocess.py:CrossSectionalZScore` 坑 1）。
- **标的数必须与 SIMD 块宽对齐**（本机 aarch64 上 8 的倍数；实测 16 可以、13 不行）。

与 `quantlab/dataset/masking.py:UniverseMask` 的区别：那个做的是**指数成分对齐**
（某天谁是成分股），这里做的是**规则化的可交易性过滤**（够不够贵、够不够活、是不是
普通股）。两者正交，可以叠加使用。
"""

import collections
import re
from typing import Self

import KunQuant.runner.KunRunner as kr
import numpy as np
import pandas as pd
import xarray as xr
from KunQuant.Op import Builder, CrossSectionalOp, Input, OpBase
from KunQuant.ops import Div
from KunQuant.Stage import Function

from quantlab.acquisition.universe import (
    _BABY_BOND_PATTERN,
    _PREFERRED_SHARE_PATTERN,
)
from quantlab.base.factor import FactorKunQuant
from quantlab.dataset.backend import XrBackend
from quantlab.utils.module import load_factor_from_config
from quantlab.utils.timer import Timer

#: 掩码在 KunQuant 图里的输入名。模块级常量，因为 `_mask_cross_sectional_inputs`
#: 在类定义之前就要用到它。
_MASK_INPUT_NAME = "universe_mask"


def _mask_cross_sectional_inputs(
    ops: list[OpBase],
) -> tuple[list[OpBase], bool]:
    """在每个截面算子的每个输入前插入 `Div(v, universe_mask)`（LS-2）。

    返回 `(拓扑排序后的 ops, 是否真的用到了掩码)`。

    图里**没有**截面算子时原样返回、第二项为 False：KunQuant 会把没有任何
    `Output` 消费的输入**剪掉**，此时若仍然声明并喂入掩码，`queryBufferHandle`
    会抛 `RuntimeError: Cannot find the buffer name`。所以「有没有截面算子」必须
    真实地决定「要不要声明掩码输入」。

    每个不同的输入只包一次（`cache`）：同一个节点喂给两个截面算子时，共用一个
    `Div` 节点，避免重复计算。

    **就地修改 `op.inputs`**，所以调用方必须传进来一张**新鲜**的图——
    `UniverseFilteredFactor._get_factor_func` 每次都重新调用内层的
    `_get_factor_func()` 正是为此。复用已改写过的图会把掩码套两层。

    规划期已验证：KunQuant 0.1.11 的 `ops/CompOp.py` 里没有任何会分解出截面算子的
    `CompositiveOp`，`DiffWithWeightedSum` 本身就是 `GenericCrossSectionalOp`，
    `TsRank` 是时序算子——所以在分解**之前**改写是完备的。
    """
    cross_sectional = [op for op in ops if isinstance(op, CrossSectionalOp)]
    if not cross_sectional:
        return list(ops), False

    builder = Builder()
    with builder:
        mask = Input(_MASK_INPUT_NAME)
        cache: dict[OpBase, OpBase] = {}
        for op in cross_sectional:
            for index, source in enumerate(op.inputs):
                if source not in cache:
                    cache[source] = Div(source, mask)
                op.inputs[index] = cache[source]

    return Function.topo_sort_ops(list(ops) + builder.ops), True


class UniverseFilteredFactor(FactorKunQuant):
    """给任意 KunQuant 因子/标签套上点时点股票池过滤（LS-1..LS-5）。

    用法::

        from quantlab.factor.universe_filter import UniverseFilteredFactor

        factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
        labels = [UniverseFilteredFactor(Return(label_config))]

    **因子和标签都要包。** 只包因子会让标签仍然带着出池标的的行；只包标签会让
    截面算子继续被垃圾标的污染。

    `config` 属性返回的是**内层因子的同一个 config 对象**，这是「无缝替换」的关键：
    模型层与回测层写日期（`_reset_factors_config`、`_redate_factors`）、读
    `config.window`（预热按 bar 计数）、读 `config.kwargs["n_forward_periods"]`
    （样本内外划分）、读 `config.data_columns`（数据指纹）时，全部落到内层上。
    """

    #: 掩码读的**原始**列名（LS-1）。绝不是 `adjClose` / `adjVolume`。
    PRICE_COLUMN = "close"
    VOLUME_COLUMN = "volume"

    #: 掩码在图里的输入名。
    MASK_INPUT = _MASK_INPUT_NAME

    #: `_reset_dataset_config` 往前多取数据的量：每根 bar 折算的日历日，加上一个
    #: 固定余量。回测器的预热只按 `config.window` 数 bar，不知道这里还需要
    #: `window` 根来填满成交额窗口；不多取的话，窗口（或预热段）的前 `window-1`
    #: 根会因为缺成交额历史而整体出池。
    LOOKBACK_DAYS_PER_BAR = 2
    LOOKBACK_PAD_DAYS = 10

    #: 非普通股代码规则（LS-1 的 (a)）。`(标签, 正则)` 对，对**大写**代码做
    #: `re.search`。
    #:
    #: **测量**（2026-09-15，`data/data/reference/universe.parquet` 里
    #: `category == "us_all"` 的 14,481 个不同代码）：
    #:
    #:     nasdaq_fifth_letter    2310   AACIW AACBR AACBU ABEOW ...
    #:     six_char_warrant         13   AACTWS ACNDWS EONRWS GRAFWS ...
    #:     delimited_suffix       1056   AAC-WS AAC-U ACP-R ACP-R-W ...
    #:     when_issued_or_called    71   DD-WD JNJ-WD AED-CL DSXN-CL ...
    #:     test_symbol_zzzt          7   ZAZZT ZBZZT ZCZZT ZJZZT ZVZZT ZWZZT ZXZZT
    #:     test_symbol_xtest        72   ATEST-* CTEST-* MTEST-* NTEST-* PTEST-*
    #:     test_symbol_zxyz          1   ZXYZ-A
    #:     preferred_share           0   （us_all 采集期已剔除；nasdaq_all 建的库仍需要）
    #:     baby_bond                 0   （同上，A4/D-02）
    #:
    #: 并集 3,519 / 14,481（24.3%），剩下 10,962 个普通股。
    #:
    #: **证伪一（机械）**：把每一组与 `sp500_constituent` + `nasdaq100_constituent`
    #: 的 966 个不同代码求交——九组**全部为 0**。指数成分股必然是普通股，所以任何
    #: 一个命中都会是一次错误排除。
    #:
    #: **证伪二（人工复核）**：2,310 个五字母命中里有 284 个在名单里找不到佐证
    #: （没有 4 字母词根、没有同词根的其他 W/R/U、也没有 `ROOT-WS`/`-U`/`-R`
    #: 兄弟），中位挂牌 1.9 年，其中 36 个 ≥ 5 年。逐个按 NASDAQ 第五字符约定
    #: （R=权利、U=单位、W=认股权证）判读：**没有一个是普通股**，所以**不设**
    #: `COMMON_TICKER_ALLOWLIST`。挂牌最久的那批恰好是「3 字符词根 + 双写后缀」
    #: 这一形状，而上面那条佐证规则按 4 字符词根去找，结构上就看不见它们——
    #: TMCWW/TMC、HTZWW/HTZ、VLYWW/VLY、XOSWW/XOS、RNWWW/RNW、SMXWW/SMX、
    #: ZEOWW/ZEO、BNCWW/BNC、FGIWW/FGI、UHGWW/UHG、WGSWW/WGS、AUROW/AUR、
    #: QSIAW/QSI，13 个短词根 13 个都在 us_all 里。其余（THWWW、GSMGW、CMPOW、
    #: SBNYW、ETHZW、MCAGR、IMAQU…）的普通股已退市或被并购，不在 us_all 里，与
    #: 「SPAC 权证/单位」完全吻合。
    #:
    #: 未被任何规则命中的带分隔符代码共 45 个，尾巴只有 6 种：`-A`(21)、`-B`(15)、
    #: `-1`(3)、`-C`(3)、`-V`(2)、`-T`(1)——全是普通股或类别股，正是下面这段陷阱
    #: 说的那批。
    #:
    #: **这条规则要避开的陷阱**：`BRK-A`、`BRK-B`、`BF-A`、`BF-B`、`HEI-A`、
    #: `LEN-B`、`MOG-A`、`UA-C`、`MKC-V`、`CWEN-A`、`PBR-A` 都是**带连字符的普通
    #: 股**，`GOOGL`、`CMCSA`、`RYAAY` 都是**五个字母的普通股**。只看「有连字符」
    #: 或「五个字母」的规则会把伯克希尔从全市场名单里悄悄删掉。
    NON_COMMON_TICKER_PATTERNS: tuple[tuple[str, str], ...] = (
        # NASDAQ 第五字符约定：W=warrant, R=right, U=unit。只对**恰好五个字母**
        # 生效，所以 4 字母的 ACIW / AAWW / ACHR / AMKR / ALTR 不受影响。
        ("nasdaq_fifth_letter", r"^[A-Z]{4}[WRU]$"),
        # 六字符权证：AACTWS、ACNDWS。
        ("six_char_warrant", r"^[A-Z]{4}WS$"),
        # 带分隔符的后缀：AAC-WS、AAC-U、ACP-R、ACP-R-W、FINS-R-W。
        ("delimited_suffix", r"[-.](?:WS|WT|W|U|UN|R|RT)(?:[-.]|$)"),
        # 已宣告/待发行：DD-WD、JNJ-WD、IAA-WI、SITC-WI、AED-CL。
        ("when_issued_or_called", r"-(?:WD|WI|CL)$"),
        # 交易所测试代码：ZWZZT / ZVZZT / ZXZZT。
        ("test_symbol_zzzt", r"^Z[A-Z]ZZT$"),
        # 测试代码家族：ATEST / CTEST / MTEST / NTEST / PTEST（可带后缀）。
        ("test_symbol_xtest", r"^[A-Z]TEST(?:-|$)"),
        ("test_symbol_zxyz", r"^ZXYZ(?:-|$)"),
        # 优先股与小额债券。`us_all` 里已在采集期被剔除（计数 0），但按 A4/D-02，
        # 用 `nasdaq_all` 建的库里仍然有，所以这两条必须在。单一真源：直接引用
        # `quantlab/acquisition/universe.py` 的常量，不在这里抄一份。
        ("preferred_share", _PREFERRED_SHARE_PATTERN),
        ("baby_bond", _BABY_BOND_PATTERN),
    )

    def __init__(
        self,
        factor: FactorKunQuant,
        min_price: float = 5.0,
        min_dollar_volume: float = 1_000_000.0,
        window: int = 20,
        exclude_non_common: bool = True,
    ):
        if not isinstance(factor, FactorKunQuant):
            raise TypeError(
                f"UniverseFilteredFactor wraps a FactorKunQuant, got "
                f"{type(factor).__name__}. A Polars factor's cross-sectional "
                f"expressions are polars expressions, not a KunQuant op graph, "
                f"so they cannot be rewritten -- and masking only the OUTPUTS "
                f"would leave every out-of-universe symbol sitting inside each "
                f"rank/zscore, which is exactly what this class exists to "
                f"prevent."
            )
        if isinstance(factor, UniverseFilteredFactor):
            raise TypeError(
                f"UniverseFilteredFactor cannot wrap another "
                f"{type(factor).__name__}: the inner wrapper would mask the "
                f"cross-sections a second time, and the two masks' parameters "
                f"would silently compose. Wrap the innermost factor once, with "
                f"the parameters you want."
            )
        if window < 1:
            raise ValueError(
                f"window must be >= 1 bar, got {window}; it is the number of "
                f"bars the trailing dollar-volume mean is taken over."
            )

        # 刻意**不**调 `super().__init__()`：`Factor.__init__` 会把内层的 config
        # 再过一遍 Factor 的 setter，而那个 setter 会把 `config.name` 覆写成本
        # 包装类的 import path，内层因子就再也无法按自己的类重建了。
        self.factor = factor
        self.min_price = float(min_price)
        self.min_dollar_volume = float(min_dollar_volume)
        self.window = int(window)
        self.exclude_non_common = bool(exclude_non_common)

        self.data_backend = XrBackend()
        self._stream_context: kr.StreamContext = None
        self._lib = None
        self._buffer_name_to_id: dict = {}

        # `_get_factor_func` 编译时置位：图里到底有没有截面算子。
        self._uses_mask = False
        # 最近一次 `cal()` / `read()` / `cal_stream()` 算出的掩码。
        self._universe_mask: xr.DataArray | None = None
        self._stream_dollar_volume = collections.deque(maxlen=self.window)

        self._reset_dataset_config()

    def __repr__(self) -> str:
        return (
            f"UniverseFilteredFactor({self.factor!r}, "
            f"min_price={self.min_price}, "
            f"min_dollar_volume={self.min_dollar_volume}, "
            f"window={self.window}, "
            f"exclude_non_common={self.exclude_non_common})"
        )

    # ------------------------------------------------------------------
    # 配置委托：模型层与回测层写进来的一切都落到内层 config 上
    # ------------------------------------------------------------------

    @property
    def config(self):
        """内层因子的 config，**同一个对象**（不是副本）。

        这是无缝替换的关键。调用方拿到的 `factor.config` 就是内层的那个，所以：

        - 模型 `_reset_factors_config` / 回测器 `_redate_factors` 写的日期直接生效；
        - `config.window` 喂回测器的 `_warmup_start`（按 bar 计数的预热）；
        - `config.kwargs["n_forward_periods"]` 喂 `_label_horizon_bars`；
        - `isinstance(config, FactorConfig)` 成立，数据指纹走 `data_columns` 分支。
        """
        return self.factor.config

    @config.setter
    def config(self, value):
        self.factor.config = value
        self._reset_dataset_config()

    def _get_factor_names(self) -> tuple[str, ...]:
        """因子名完全由内层决定：包装不改变产出哪些因子，只改变它们的值。"""
        return self.factor._get_factor_names()

    def _reset_dataset_config(self) -> None:
        """先让内层重置，然后**只往前**加宽数据集的起始日期。

        回测器的预热只按 `config.window` 数 bar（`_warmup_start`），它并不知道
        掩码还需要 `window` 根 bar 才能算出第一个成交额均值。不加宽的话，窗口
        （或预热段）最前面的 `window-1` 根会因为窗口不满而整体出池——表现为回测
        开头几天一个标的都选不出来，且不报任何错。

        只加宽、从不收窄：内层若因为自己的大 `window` 已经取得更早，就保持它。
        """
        self.factor._reset_dataset_config()

        config = self.config
        widened = pd.to_datetime(config.start_date) - pd.DateOffset(
            days=self.LOOKBACK_DAYS_PER_BAR * self.window
            + self.LOOKBACK_PAD_DAYS
        )
        widened_date = widened.strftime("%Y-%m-%d")

        dataset_config = config.dataset.config
        # 两侧都是零填充 ISO 日期，所以字符串比较就是日期比较。
        if widened_date < dataset_config.start_date:
            dataset_config.start_date = widened_date

    # ------------------------------------------------------------------
    # 图改写（LS-2）
    # ------------------------------------------------------------------

    def _get_factor_func(self) -> Function:
        """内层的图，每个截面算子的输入前插入 `Div(v, universe_mask)`。

        每次都向内层要一张**新鲜**的图：`_mask_cross_sectional_inputs` 会就地改写
        `op.inputs`，缓存或复用会把掩码套成两层。
        """
        ops = self.factor._get_factor_func().ops
        rewritten, uses_mask = _mask_cross_sectional_inputs(ops)
        self._uses_mask = uses_mask
        return Function(rewritten)

    # ------------------------------------------------------------------
    # 掩码本身（LS-1）
    # ------------------------------------------------------------------

    @classmethod
    def is_common_ticker(cls, symbol) -> bool:
        """代码看起来是不是普通股（与日期无关的静态规则）。"""
        ticker = str(symbol).upper()
        return not any(
            re.search(pattern, ticker)
            for _, pattern in cls.NON_COMMON_TICKER_PATTERNS
        )

    def compute_universe_mask(self, panel: xr.Dataset) -> xr.DataArray:
        """按 LS-1 从**原始** close/volume 算出 `(timestamp, symbol)` 掩码。

        池内 1.0、出池 NaN。窗口不满或窗口内含 NaN 都算出池：
        `min_periods=window` 数的是窗口内**有效**观测数，所以窗口里任何一个 NaN
        都让结果变 NaN，正是「有缺失就不敢说它够活」。

        缺列时点名报错，而不是悄悄返回一个全出池（或全入池）的掩码——后者会让
        整条流水线安静地产出空结果（T-p91-02）。
        """
        for column in (self.PRICE_COLUMN, self.VOLUME_COLUMN):
            if column not in panel.data_vars:
                raise ValueError(
                    f"UniverseFilteredFactor needs the RAW column {column!r} "
                    f"to decide universe membership, and the dataset panel "
                    f"does not carry it (present: "
                    f"{sorted(map(str, panel.data_vars))}). The mask reads RAW "
                    f"close/volume, never the adjusted columns: adjusted "
                    f"history is depressed by splits and dividends, so a "
                    f"penny stock today can look like a $50 stock in 2015."
                )

        panel = panel.sortby("timestamp")
        close = (
            panel[self.PRICE_COLUMN]
            .transpose("timestamp", "symbol")
            .astype("float64")
        )
        volume = (
            panel[self.VOLUME_COLUMN]
            .transpose("timestamp", "symbol")
            .astype("float64")
        )

        dollar_volume = close * volume
        average = dollar_volume.rolling(
            timestamp=self.window, min_periods=self.window
        ).mean()

        # NaN 参与比较得到 False，也就是出池——正是想要的。
        in_universe = (close >= self.min_price) & (
            average >= self.min_dollar_volume
        )

        if self.exclude_non_common:
            common = xr.DataArray(
                np.array(
                    [
                        self.is_common_ticker(symbol)
                        for symbol in panel["symbol"].values
                    ],
                    dtype=bool,
                ),
                dims=["symbol"],
                coords={"symbol": panel["symbol"]},
            )
            in_universe = in_universe & common

        return xr.where(in_universe, 1.0, np.nan).rename(self.MASK_INPUT)

    # ------------------------------------------------------------------
    # 批量计算
    # ------------------------------------------------------------------

    def cal(self) -> Self:
        """`FactorKunQuant.cal` 的镜像，外加喂掩码输入。

        `start` 恒为 0：KunQuant 0.1.11 的截面算子在 `start>0` 时结果错误
        （见模块 docstring）。
        """
        input_dict, symbols, timestamps = self.config.dataset.to_kunquant(
            data_columns=self.config.data_columns
        )
        num_time = next(iter(input_dict.values())).shape[0]

        # `_make()` 会调 `_get_factor_func()`，也就是在这一步才知道 `_uses_mask`。
        self._lib = self._make()
        modu = self._lib.getModule(f"{self.__class__.__name__}")  # type: ignore

        # `to_kunquant` 刚刚读过数据集，这里拿到的是同一份缓存，不会再读一次盘。
        self._universe_mask = self.compute_universe_mask(
            self.config.dataset.get_xarray_dataset()
        )

        if self._uses_mask:
            aligned = (
                self._universe_mask.reindex(
                    timestamp=timestamps, symbol=symbols
                )
                .transpose("timestamp", "symbol")
                .values
            )
            input_dict[self.MASK_INPUT] = np.ascontiguousarray(
                aligned, dtype=np.float32
            )

        executor = kr.createMultiThreadExecutor(self.config.njobs)
        with Timer(f" {self.__class__.__name__}: cal"):
            out_dict = kr.runGraph(executor, modu, input_dict, 0, num_time)

        self._lib = None
        self._to_xarray_dataset(out_dict, timestamps, symbols)
        return self

    def read(self, overwrite: bool = False) -> Self:
        """从因子库读回落盘结果，并重新算出掩码。

        掩码来自**数据集**（原始 close/volume），不是因子库——因子库里只有因子值。
        所以这里要先读数据集再读因子库。

        **注意落盘的是打掩码之前的值。** 继承来的 `save()` / `update()` 写的是
        改写后的图的输出，输出掩码（LS-3）是 `read()` 之后再施加的。因此
        `factor_data_strategy="read"` 用的因子库**必须**是经由本包装类写出来的：
        用未包装的内层因子写出来的库，其截面值已经被出池标的污染了，再怎么在读
        的时候打掩码也救不回来。
        """
        self.config.dataset.read(overwrite=overwrite)
        self._universe_mask = self.compute_universe_mask(
            self.config.dataset.get_xarray_dataset()
        )
        super().read(overwrite=overwrite)
        return self

    # ------------------------------------------------------------------
    # 流式计算
    # ------------------------------------------------------------------

    def init_stream(self) -> Self:
        """编译改写后的图（STREAM layout），并额外绑定掩码输入的 buffer。

        只在图里真的有截面算子时才绑定：没有的话 KunQuant 已经把掩码输入剪掉了，
        `queryBufferHandle` 会抛 `RuntimeError: Cannot find the buffer name`。
        """
        super().init_stream()
        if self._uses_mask:
            self._buffer_name_to_id[self.MASK_INPUT] = (
                self._stream_context.queryBufferHandle(self.MASK_INPUT)
            )
        return self

    def cal_stream(
        self, data: dict[str, np.ndarray], timestamp: int, symbols: list[str]
    ) -> Self:
        """推进一根 bar：先算这一行掩码并推进去，再走内层的流式计算。

        `data` 里除了 `config.data_columns`，还**必须**带上原始 `close` 与
        `volume`：继承来的推送只发 `data_columns`，这两个键是掩码专用的额外输入。

        成交额均值用 `np.mean`（不是 `nanmean`）对**攒满**的窗口求：窗口没满就整行
        NaN，窗口里有 NaN 也得到 NaN。这与批量路径 `min_periods=window` 的语义
        逐位一致——两条路必须给出同一个掩码，否则流式和批量的因子值会悄悄分叉。
        """
        missing = [
            column
            for column in (self.PRICE_COLUMN, self.VOLUME_COLUMN)
            if column not in data
        ]
        if missing:
            raise ValueError(
                f"{type(self).__name__}.cal_stream: the bar dict is missing "
                f"the RAW column(s) {missing}, which decide universe "
                f"membership. They are EXTRA keys beyond config.data_columns "
                f"({tuple(self.config.data_columns)}): the inherited push only "
                f"sends data_columns, so they must be supplied explicitly."
            )

        close = np.asarray(data[self.PRICE_COLUMN], dtype=np.float64).reshape(-1)
        volume = np.asarray(
            data[self.VOLUME_COLUMN], dtype=np.float64
        ).reshape(-1)
        self._stream_dollar_volume.append(close * volume)

        if len(self._stream_dollar_volume) == self.window:
            average = np.mean(np.stack(self._stream_dollar_volume), axis=0)
        else:
            average = np.full(close.shape, np.nan)

        in_universe = (close >= self.min_price) & (
            average >= self.min_dollar_volume
        )
        if self.exclude_non_common:
            in_universe = in_universe & np.array(
                [self.is_common_ticker(symbol) for symbol in symbols],
                dtype=bool,
            )
        row = np.where(in_universe, 1.0, np.nan).astype(np.float32)

        if self._stream_context is None:
            self.init_stream()

        # 掩码必须在 `run()` 之前推进去；`super().cal_stream` 推完数据列就会 run。
        if self._uses_mask:
            self._stream_context.pushData(
                self._buffer_name_to_id[self.MASK_INPUT],
                np.ascontiguousarray(row),
            )

        super().cal_stream(data, timestamp, symbols)

        self._universe_mask = xr.DataArray(
            row.reshape(1, -1).astype("float64"),
            dims=["timestamp", "symbol"],
            coords={"timestamp": [timestamp], "symbol": list(symbols)},
            name=self.MASK_INPUT,
        )
        return self

    # ------------------------------------------------------------------
    # 输出掩码（LS-3）
    # ------------------------------------------------------------------

    def _assert_computed(self) -> None:
        if self._universe_mask is None:
            raise RuntimeError(
                f"{type(self).__name__}: no universe mask has been computed "
                f"yet, so the outputs cannot be masked. Call cal(), read() or "
                f"cal_stream() first."
            )

    def _get_xarray_dataset(self) -> xr.Dataset:
        """`get_features()` / `get_labels()` 的共同入口，先检查掩码是否已算。

        不加这一步的话，还没 `cal()` 就调 `get_features()` 会先撞上存储后端的
        `AttributeError: 'XrBackend' object has no attribute 'data'`，那个报错
        既不点名原因也不说该做什么。
        """
        self._assert_computed()
        return super()._get_xarray_dataset()

    def _mask_panel(self, data: xr.Dataset) -> xr.Dataset:
        """在掩码为 NaN 的位置置 NaN，然后**只**删掉被代码规则排除的标的。

        标的轴规则见模块 docstring 的 LS-3（2026-09-15 用户决定）：
        **绝不**在标的轴上 `dropna`。只因为价格/成交额阈值而整窗 NaN 的标的保留为
        NaN 列，这样标的轴就与日期窗口无关，DL 头跨窗口预测才不会因为缺少训练过的
        标的而报错。
        """
        self._assert_computed()

        mask = self._universe_mask.reindex(
            timestamp=data["timestamp"], symbol=data["symbol"]
        )
        data = data.where(mask.notnull())

        if self.exclude_non_common:
            keep = [
                symbol
                for symbol in data["symbol"].values
                if self.is_common_ticker(str(symbol))
            ]
            data = data.sel(symbol=keep)
        return data

    def _get_features(self, data: xr.Dataset) -> xr.Dataset:
        return self._mask_panel(self.factor._get_features(data))

    def _get_labels(self, data: xr.Dataset) -> xr.Dataset:
        """标签先过内层的变换，**再**打掩码。

        顺序是load-bearing 的：`Return._get_labels` 做的是
        `shift(timestamp=-n)`，先位移再打掩码，掩码打的就是标签**自己**那个时间戳
        t（LS-3）；反过来先打掩码再位移，等于用 t+n 的池状态决定 t 的标签，那是
        一个前视错误。
        """
        return self._mask_panel(self.factor._get_labels(data))

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------

    def get_config(self) -> dict:
        """包装类自己的配置；内层因子的配置原样嵌在 `factor` 键下。

        刻意**没有**顶层 `dataset` 键：数据集属于内层因子，重复一份会变成第二个
        数据集对象、第二次读盘，两份还可能漂移。
        """
        return {
            "name": self.import_path,
            "factor": self.factor.get_config(),
            "min_price": float(self.min_price),
            "min_dollar_volume": float(self.min_dollar_volume),
            "window": int(self.window),
            "exclude_non_common": bool(self.exclude_non_common),
        }

    #: `get_config()` 里除 `name` / `factor` 之外的参数键。
    _PARAMETER_KEYS = frozenset(
        {"min_price", "min_dollar_volume", "window", "exclude_non_common"}
    )

    @classmethod
    def from_config(cls, config: dict) -> "UniverseFilteredFactor":
        """从 `get_config()` 的产物重建，内层因子递归走 `load_factor_from_config`。

        缺键或多键都报 `ValueError` 点名（D-25/WR-06 先例）：**绝不**用当前的默认
        值去填一个缺失的参数。默认值日后改了，一份旧配置就会悄悄重建成另一个股票
        池，而 D-25 承诺的是「所有参数一致」（T-p91-01）。
        """
        config = dict(config)
        config.pop("name", None)

        inner = config.pop("factor", None)
        if inner is None:
            raise ValueError(
                f"{cls.__name__}.from_config: the config has no 'factor' key, "
                f"so there is no inner factor to rebuild."
            )

        missing = sorted(cls._PARAMETER_KEYS - set(config))
        unknown = sorted(set(config) - cls._PARAMETER_KEYS)
        if missing or unknown:
            raise ValueError(
                f"{cls.__name__}.from_config: refusing to rebuild -- "
                f"missing key(s) {missing}, unknown key(s) {unknown}. Missing "
                f"parameters are NOT filled from the current defaults, which "
                f"may differ from what the stored run used (D-25/WR-06)."
            )

        return cls(load_factor_from_config(inner), **config)
