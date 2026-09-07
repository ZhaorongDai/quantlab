import joblib
from pathlib import Path
from typing import Self

from base.backend import ModelBackend


class MlBackend(ModelBackend):
    """joblib 持久化，给非 torch 的模型用（`MLConfig` 那条路，比如 xgboost）。

    它是 `base/backend.py:ModelBackend` 的实现，跟 `DataBackend` 是对称的两半：
    一个管数据落在哪，一个管模型落在哪。整个类没有任何维度、坐标、时间轴的概念
    ——它能和 `XrBackend` 长在同一套设计里，恰恰因为契约里没有一句话假设「数据是
    个带 timestamp/symbol 的面板」。

    目前全仓没有调用点：`BaseModel._auto_train` 对 `MLConfig` 还是
    `NotImplementedError`。这是**尚未建成的既定路线的脚手架**，不是废弃入口
    （`BaseModel.predict()` 签名里的 `np.ndarray` 分支就是为它留的）。

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
