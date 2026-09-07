import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from base.config import DLConfig
from base.model import BaseModel


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class MLPRegressor(BaseModel):
    def __init__(self, config: DLConfig):
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def _train_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        num_times, num_symbols, num_features = x.shape
        num_labels = y.shape[2]

        x = x.reshape(num_times, -1)
        y = y.reshape(num_times, -1)

        self.model.train()  # type: ignore
        self.optim.zero_grad()  # type: ignore
        outputs = self.model(x)  # type: ignore
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optim.step()  # type: ignore

        with torch.no_grad():
            pred = self.model(x).cpu().numpy()  # type: ignore
            train_y_np = y.cpu().numpy()

        metrics = {
            "train_loss": loss.item(),
            "train_R²": r2_score(train_y_np, pred),
            "train_MSE": mean_squared_error(train_y_np, pred),
            "train_RMSE": np.sqrt(mean_squared_error(train_y_np, pred)),
            "train_MAE": mean_absolute_error(train_y_np, pred),
        }

        self._wandb_recorder.log(metrics, step=epoch)

    def _init_model(
        self,
        num_symbols: int,
        num_features: int,
        num_labels: int,
        hyperparameters: dict,
    ) -> nn.Module:
        """签名必须带 `hyperparameters`——`BaseModel._init_model_and_optim()` 是
        按关键字传的（`hyperparameters=self.config.hyperparameters`）。

        以前这里少了这个形参，于是 `MLPRegressor` 连一次 `train()` 都跑不到：
        `TypeError: _init_model() got an unexpected keyword argument
        'hyperparameters'`。两个隐藏层的宽度沿用原来硬编码的 512 / 256 作为默认
        值，所以补上形参不改变任何既有配置下的模型结构。
        """
        input_size = num_symbols * num_features
        output_size = num_symbols * num_labels
        hidden_size1 = hyperparameters.get("hidden_size1", 512)
        hidden_size2 = hyperparameters.get("hidden_size2", 256)
        return MLP(input_size, hidden_size1, hidden_size2, output_size).to(
            self.device
        )

    def _init_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _test_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        x = x.to(self.device)
        y = y.to(self.device)

        num_times, num_symbols, num_features = x.shape
        x = x.reshape(num_times, -1)
        y = y.reshape(num_times, -1)

        self.model.eval()  # type: ignore

        with torch.no_grad():
            y_pred_tensor = self.model(x)  # type: ignore
            test_loss = self.criterion(y_pred_tensor, y).cpu().numpy()
            y_pred = y_pred_tensor.cpu().numpy()
            test_y_np = y.cpu().numpy()

        metrics = {
            "test_loss": test_loss,
            "test_R²": r2_score(test_y_np, y_pred),
            "test_MSE": mean_squared_error(test_y_np, y_pred),
            "test_RMSE": np.sqrt(mean_squared_error(test_y_np, y_pred)),
            "test_MAE": mean_absolute_error(test_y_np, y_pred),
        }
        self._wandb_recorder.log(metrics, step=epoch)

    def _val_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """必须存在，而且必须**返回**一个能 `float()` 的损失。

        它以前根本没有实现，所以 `MLPRegressor.__abstractmethods__` 里始终留着
        `{'_val_one_batch'}`，这个类连实例化都做不到：
        `TypeError: Can't instantiate abstract class MLPRegressor`。

        返回值的契约在 2026-09-07 被收紧过：`base/model.py` 的 epoch 循环现在把
        每个 batch 的返回值按样本数加权累加成「一个 epoch 的验证损失」再跟早停
        阈值比较（`val_loss_sum += float(val_loss) * batch_samples`）。返回 None
        会当场 `TypeError`，所以这里返回的是 `loss.detach()` 而不是只记 metrics。
        """
        x = x.to(self.device)
        y = y.to(self.device)

        num_times = x.shape[0]
        x = x.reshape(num_times, -1)
        y = y.reshape(num_times, -1)

        with torch.no_grad():
            y_pred_tensor = self.model(x)  # type: ignore
            val_loss = self.criterion(y_pred_tensor, y)
            y_pred = y_pred_tensor.cpu().numpy()
            val_y_np = y.cpu().numpy()

        metrics = {
            "val_loss": val_loss.item(),
            "val_R²": r2_score(val_y_np, y_pred),
            "val_MSE": mean_squared_error(val_y_np, y_pred),
            "val_RMSE": np.sqrt(mean_squared_error(val_y_np, y_pred)),
            "val_MAE": mean_absolute_error(val_y_np, y_pred),
        }
        self._wandb_recorder.log(metrics, step=epoch)
        return val_loss.detach()

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """入参是**张量**，不是 `xr.Dataset`。

        `BaseModel._preprocess` 的契约是 `(torch.Tensor) -> torch.Tensor`，两个
        调用点（`_train_dl` 里对四个张量批量调用、`_predict_nn` 里对推理输入调用）
        传进来的都是 `to_tensor()` 的产物。以前这里写的是 `data.fillna(0.0)`，
        标注也写着 `xr.Dataset`——真跑起来是
        `AttributeError: 'Tensor' object has no attribute 'fillna'`。
        `torch.nan_to_num` 是同语义的张量版本，也跟 `dl_model/rnn*.py` 一致。
        """
        return torch.nan_to_num(data, nan=0.0)

