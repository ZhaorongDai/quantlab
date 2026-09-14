import joblib
from pathlib import Path
from typing import Self

from quantlab.base.backend import ModelBackend


class MlBackend(ModelBackend):
    """joblib 持久化，给非 torch 的模型用（`MLConfig` 那条路，比如 xgboost）。

    它是 `base/backend.py:ModelBackend` 的实现，跟 `DataBackend` 是对称的两半：
    一个管数据落在哪，一个管模型落在哪。整个类没有任何维度、坐标、时间轴的概念
    ——它能和 `XrBackend` 长在同一套设计里，恰恰因为契约里没有一句话假设「数据是
    个带 timestamp/symbol 的面板」。

    它是 `quantlab/base/model.py:MLModel` 的 checkpoint 持久化后端：
    `MLModel._write_checkpoint` 调 `MlBackend().to_internal(model).write(path)`，
    `MLModel._read_checkpoint` 调 `MlBackend().read(path).get_model()`
    （首个调用点来自 260914-lno，此前它是尚未建成路线的脚手架）。joblib 本质是
    pickle，只加载自己信任的文件。

    2026-09-07 修：`read` / `write` / `to_internal` 三个方法以前都隐式返回
    `None`，而 ABC 上写的是 `-> Self`，所以任何链式写法当场
    `AttributeError: 'NoneType' object has no attribute ...`——这个类显然一次都
    没有被真正跑过。现在三个都 `return self`，跟 `dataset/backend.py` 的两个
    `DataBackend` 实现一致；`**kwargs` 也补上了，同样是 ABC 早就声明的。
    由 `tests/test_ml_backend.py` 锁。
    """

    def get_model(self):
        return self.model

    def write(self, path: str, **kwargs) -> Self:
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        joblib.dump(self.model, path, **kwargs)
        return self

    def read(self, path: str, **kwargs) -> Self:
        self.model = joblib.load(path, **kwargs)
        return self

    def to_internal(self, model) -> Self:
        self.model = model
        return self
