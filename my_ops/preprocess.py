from KunQuant.Op import Builder
from KunQuant.ops import *


class WindowedZScore(WindowedCompositiveOp):
    """
    窗口Z值标准化,先将缺失值替换为0,然后进行滚动标准化
    Z-score = (value - rolling_mean) / rolling_std
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


class WindowedRobustStandardization(WindowedCompositiveOp):
    """
    滚动鲁棒标准化,使用中位数和中位数绝对偏差(MAD)
    Robust Z-score = (value - rolling_median) / (1.4826 * rolling_MAD)
    对异常值更加鲁棒，适用于包含噪声的金融数据
    """

    # `options` matches KunQuant CompositiveOp (passes/Decompose.py:15).
    def decompose(self, options: dict) -> List[OpBase]:
        window: int = self.attrs["window"]  # type: ignore
        b = Builder(self.get_parent())
        with b:
            # 第一步：计算滚动中位数
            rolling_median = WindowedQuantile(self.inputs[0], window, 0.5)

            # 第二步：计算与中位数的绝对偏差
            abs_deviation = Abs(Sub(self.inputs[0], rolling_median))

            # 第三步：计算MAD（中位数绝对偏差）
            mad = WindowedQuantile(abs_deviation, window, 0.5)

            # 第四步：计算鲁棒标准化
            # 使用1.4826常数使MAD与正态分布标准差一致
            mad_scaled = MulConst(mad, 1.4826)

            # 避免除零，添加小的常数
            mad_safe = Add(mad_scaled, ConstantOp(1e-8))

            # 计算鲁棒Z值
            numerator = Sub(self.inputs[0], rolling_median)
            robust_zscore = Div(numerator, mad_safe)

        return b.ops
