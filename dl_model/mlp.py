import numpy as np
import torch
import torch.nn as nn
import xarray as xr
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

    def _train_one_epoch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
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
        self, num_symbols: int, num_features: int, num_labels: int
    ) -> nn.Module:
        input_size = num_symbols * num_features
        output_size = num_symbols * num_labels
        return MLP(input_size, 512, 256, output_size).to(self.device)

    def _init_optim(self, model):
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _test_one_epoch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
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

    def _preprocess(self, data: xr.Dataset) -> xr.Dataset:
        data = data.fillna(0.0)
        return data

