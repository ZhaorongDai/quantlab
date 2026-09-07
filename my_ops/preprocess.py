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
