from KunQuant.Op import Builder
from KunQuant.ops import *


class WindowedZScore(WindowedCompositiveOp):
    """
    窗口Z值标准化（**时序**标准化：每个标的拿自己过去 window 根的统计量标准化自己）
    Z-score = (value - rolling_mean) / rolling_std

    **不做任何缺失值处理。** 这条 docstring 以前写的是"先将缺失值替换为 0,
    然后进行滚动标准化"，但 `decompose()` 里从来就只有 `WindowedAvg` /
    `WindowedStddev` / `Sub` / `Div`，没有任何替换。2026-09-07 改的是这段
    **文字**而不是代码——补一个 fillna(0) 会改变这个 op 产出的每一个因子值，
    也就是改变已经落盘的每一份因子和依赖它们训出来的每一个模型；而这里从来
    没有人真的依赖过那句描述（它描述的行为一天都没有存在过）。想要填充语义
    的，请在调用方显式做，不要偷偷塞进标准化里。

    窗口未填满时前 window-1 根输出 NaN，这是 KunQuant 滚动 op 的正常行为，
    不是缺失值处理的替代品。
    """

    # `options` matches KunQuant CompositiveOp (passes/Decompose.py:15).
    def decompose(self, options: dict) -> list[OpBase]:
        window: int = self.attrs["window"]  # type: ignore
        b = Builder(self.get_parent())
        with b:
            # 计算滚动均值和标准差
            rolling_mean = WindowedAvg(self.inputs[0], window)
            rolling_std = WindowedStddev(self.inputs[0], window)

            # 计算Z值标准化
            # Z = (x - mean) / std
            diff = Sub(self.inputs[0], rolling_mean)
            z_score = Div(diff, rolling_std)

        return b.ops


class CrossSectionalZScore(GenericCrossSectionalOp):
    """
    截面Z值标准化（**截面**标准化：每个时间点上，跨所有标的算统计量标准化）
    Z-score = (value - cs_mean) / cs_std

    在每个时间点上，对全部标的做 NaN 感知的均值和**样本**标准差（ddof=1，
    与 pandas `.std()`、KunQuant `WindowedStddev` 一致），输出
    `(x - mean) / sd`。

    **缺失值：** 输入为 NaN 的位置输出 NaN；某个截面有效值少于 2 个，或者
    sd == 0（常数截面）时，整行输出 NaN。本 op 不做任何 fillna——和
    `WindowedZScore` 一样，想要填充语义请在调用方显式做。

    **为什么是 `GenericCrossSectionalOp` 而不是 `CompositiveOp`：**
    `decompose()` 只能展开成时序 op，表达不了"跨标的"的计算，所以这里按
    KunQuant 上游 doc/NewOperators.md 的 "Cross-Sectional Operators" 路线，
    直接提供一段 C++ 循环体（`generate_body`）。

    **坑 1（start>0 结果错误）：** KunQuant 0.1.11 的 `CrossSectionalDataHolder`
    （cpp/Kun/LayoutMappers.hpp）在 `num_time` 被赋值之前就用它算了
    `base_time`，导致 `kr.runGraph(..., start>0, ...)` 对**所有**
    `GenericCrossSectionalOp` 给出错误且不确定的结果（内置的 `Scale` 是对的，
    上游 main 目前仍有这个 bug）。本仓库唯一的批量调用方
    `quantlab/base/factor.py` 的 `FactorKunQuant.cal` 始终传 start=0，所以是
    安全的。新的调用方不得传非零 start。

    **坑 2（C++ 函数去重只看类名 + layout）：** `passes/CodegenCpp.py` 生成的
    C++ 函数只按类名加 layout 去重。因此 `generate_body` 不能依赖 attrs；
    需要参数化的截面 op 必须每组参数一个类，否则不同参数会共用同一份代码。

    **坑 3（标的数必须 SIMD 对齐）：** 标的数必须与 KunQuant 的 SIMD 块宽对齐。
    这是 KunQuant 的通用限制，不是本 op 特有的，TS 和 STREAM 两种 layout 都受
    影响。在本机 aarch64 上实测 16 个标的可以、13 个不行，且 aarch64 不支持
    `allow_unaligned`。见 example/factor.md "常见坑" 第 1 条。

    **轴向：** 本 op 与 `WindowedZScore` 沿不同的轴做标准化（截面 vs 时序），
    选哪个由策略类型决定。本 op 不是 `WindowedZScore` 的替代品，两者永远不要
    "对齐"。目前没有任何因子类默认使用它：美股因子仍按 D-09 输出原始值，接入
    因子类推迟到 ARCH-02。见 example/factor.md "标准化算子与截面 vs 时序"。
    """

    def __init__(self, v: OpBase) -> None:
        super().__init__([v], None)

    def generate_head(self) -> str:
        return ""

    def generate_body(self) -> str:
        return """
        T sum = 0;
        size_t n = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            if (!std::isnan(v)) { sum += v; n++; }
        }
        T mean = n > 0 ? sum / n : NAN;
        T ss = 0;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            if (!std::isnan(v)) { T d = v - mean; ss += d * d; }
        }
        T sd = n > 1 ? std::sqrt(ss / (n - 1)) : NAN;
        for (size_t i = 0; i < num_stocks; i++) {
            T v = input_0[i];
            output_0[i] = (std::isnan(v) || !(sd > 0)) ? NAN : (v - mean) / sd;
        }
        """
