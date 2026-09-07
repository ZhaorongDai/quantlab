import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from base.config import DLConfig
from base.model import BaseModel


class ModelRBaseCrypto(nn.Module):
    """
    Base recurrent model block, analogous to `ModelRBase` from the Kaggle notebook.
    It consists of GRU/LSTM layers followed by fully connected layers.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        dropout_rates: list[float],
        hidden_sizes_linear: list[int],
        dropout_rates_linear: list[float],
        model_type: str,
    ) -> None:
        super().__init__()
        self.num_layers = len(hidden_sizes)

        self.recurrent_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_size = input_size
        for i in range(self.num_layers):
            if model_type == "gru":
                layer = nn.GRU(
                    current_size,
                    hidden_sizes[i],
                    num_layers=1,
                    batch_first=True,
                )
            elif model_type == "lstm":
                layer = nn.LSTM(
                    current_size,
                    hidden_sizes[i],
                    num_layers=1,
                    batch_first=True,
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            self.recurrent_layers.append(layer)
            self.dropout_layers.append(nn.Dropout(dropout_rates[i]))
            current_size = hidden_sizes[i]

        n_input_linear = current_size
        fc_layers = []
        current_linear_size = n_input_linear
        if hidden_sizes_linear:
            for i in range(len(hidden_sizes_linear)):
                fc_layers.append(
                    nn.Linear(current_linear_size, hidden_sizes_linear[i])
                )
                fc_layers.append(nn.ReLU())
                fc_layers.append(nn.Dropout(dropout_rates_linear[i]))
                current_linear_size = hidden_sizes_linear[i]

        fc_layers.append(
            nn.Linear(current_linear_size, 1)
        )  # Output is always 1 for a single target
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, T, _ = x.shape
        recurrent_output = x
        for i, layer in enumerate(self.recurrent_layers):
            recurrent_output, _ = layer(recurrent_output)
            recurrent_output = self.dropout_layers[i](recurrent_output)

        output = recurrent_output.reshape(D * T, -1)
        output = self.fc(output)
        output = output.reshape(D, T, 1)
        return output


class ModelRCrypto(nn.Module):
    """
    Recurrent model with auxiliary targets, analogous to `ModelR` from the Kaggle notebook.
    It uses multiple `ModelRBaseCrypto` instances, one for each target (primary + auxiliaries).
    The outputs for the auxiliary targets are then combined to predict the primary target.
    """

    def __init__(
        self,
        input_size: int,
        num_labels: int,
        hidden_sizes: list[int],
        dropout_rates: list[float],
        hidden_sizes_linear: list[int],
        dropout_rates_linear: list[float],
        model_type: str,
    ):
        super().__init__()
        if num_labels < 2:
            raise ValueError(
                "ModelRCrypto requires at least 2 labels (1 primary, 1+ auxiliary)."
            )

        self.num_labels = num_labels
        self.num_aux_labels = num_labels - 1

        # Create a base model for each label
        self.base_models = nn.ModuleList(
            [
                ModelRBaseCrypto(
                    input_size,
                    hidden_sizes,
                    dropout_rates,
                    hidden_sizes_linear,
                    dropout_rates_linear,
                    model_type,
                )
                for _ in range(self.num_labels)
            ]
        )

        # The final output layer combines the predictions for the auxiliary targets
        # to make a final prediction for the primary target.
        self.out = nn.Linear(self.num_aux_labels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        D, T, _ = x.shape

        # Get predictions from each base model for its corresponding target
        # The first model predicts the primary target, the rest predict auxiliary targets
        aux_preds = [
            self.base_models[i + 1](x) for i in range(self.num_aux_labels)
        ]
        aux_preds_cat = torch.cat(
            aux_preds, dim=-1
        )  # Shape: (D, T, num_aux_labels)

        # Combine auxiliary predictions to form the final prediction for the primary target
        primary_pred_final = self.out(
            aux_preds_cat.reshape(D * T, self.num_aux_labels)
        )
        primary_pred_final = primary_pred_final.reshape(D, T, 1)

        # We also get the direct prediction for the primary target from its own base model
        primary_pred_direct = self.base_models[0](x)

        # The full set of predictions includes the direct one for the primary target
        # and all auxiliary predictions. This is used for calculating the loss.
        all_direct_preds = torch.cat([primary_pred_direct] + aux_preds, dim=-1)

        return primary_pred_final, all_direct_preds


class RNNRegressor(BaseModel):
    """
    A regressor that uses `ModelRCrypto` for predictions, integrated with the
    project's BaseModel interface and training loop.
    """

    def __init__(self, config: DLConfig):
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def _train_one_epoch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        self.optim.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Loss calculation now mirrors the Kaggle implementation
        # 1. Loss for all direct predictions (primary + auxiliaries) vs their targets
        loss_direct = self.criterion(all_direct_preds, y)

        # 2. Loss for the final combined prediction vs the primary target
        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)

        # Total loss is the sum of the two components
        loss = loss_direct + loss_primary_final

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            # For metrics, we still evaluate the final primary prediction
            primary_pred_np = np.nan_to_num(primary_pred.cpu().numpy())
            primary_y_np = primary_y.cpu().numpy()

        metrics = {
            "train_loss": loss.item(),
            "train_loss_direct": loss_direct.item(),
            "train_loss_final": loss_primary_final.item(),
            "train_primary_r2": r2_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "train_primary_MSE": mean_squared_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "train_primary_RMSE": np.sqrt(
                mean_squared_error(
                    primary_y_np.flatten(), primary_pred_np.flatten()
                )
            ),
            "train_primary_MAE": mean_absolute_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            # "train_primary_IC": pearsonr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
            # "train_primary_rank_IC": spearmanr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
        }

        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)

    def _init_model(
        self,
        num_symbols: int,
        num_features: int,
        num_labels: int,
        hyperparameters: dict,
    ) -> nn.Module:
        return ModelRCrypto(
            input_size=num_features,
            num_labels=num_labels,
            hidden_sizes=[256, 128, 64],
            dropout_rates=[0.1, 0.1, 0.1],
            hidden_sizes_linear=[32],
            dropout_rates_linear=[0.1],
            model_type="gru",
        )

    def _init_optim(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.config.lr)

    def _test_one_epoch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Consistent loss calculation for testing
        loss_direct = self.criterion(all_direct_preds, y)
        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)
        test_loss = loss_direct + loss_primary_final

        primary_pred_np = primary_pred.cpu().numpy()
        primary_y_np = primary_y.cpu().numpy()

        metrics = {
            "test_loss": test_loss.item(),
            "test_loss_direct": loss_direct.item(),
            "test_loss_final": loss_primary_final.item(),
            "test_primary_r2": r2_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "test_primary_MSE": mean_squared_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "test_primary_RMSE": np.sqrt(
                mean_squared_error(
                    primary_y_np.flatten(), primary_pred_np.flatten()
                )
            ),
            "test_primary_MAE": mean_absolute_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            # "test_primary_IC": pearsonr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
            # "test_primary_rank_IC": spearmanr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
        }
        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)

    def _val_one_epoch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """必须**返回**验证损失，不能只记 metrics。

        `BaseModel._val_one_epoch` 声明的就是 `-> torch.Tensor`，但这里以前什么都
        不返回。2026-09-07 之后这条从「注解不实」升级成硬故障：epoch 循环现在无条件
        对每个验证 batch 执行 `val_loss_sum += float(val_loss) * batch_samples`，
        于是 `RNNRegressor.train()` 在第 0 个 epoch 就是
        `TypeError: float() argument must be a string or a real number, not
        'NoneType'`——跟 `early_stopping` 开不开无关。

        同目录的 `rnn_classification.py:RNNClassifier._val_one_epoch` 本来就返回
        `val_loss.detach()`，这里对齐它。
        """
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Consistent loss calculation for testing
        loss_direct = self.criterion(all_direct_preds, y)
        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)
        val_loss = loss_direct + loss_primary_final

        primary_pred_np = primary_pred.cpu().numpy()
        primary_y_np = primary_y.cpu().numpy()

        metrics = {
            "val_loss": val_loss.item(),
            "val_loss_direct": loss_direct.item(),
            "val_loss_final": loss_primary_final.item(),
            "val_primary_r2": r2_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "val_primary_MSE": mean_squared_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "val_primary_RMSE": np.sqrt(
                mean_squared_error(
                    primary_y_np.flatten(), primary_pred_np.flatten()
                )
            ),
            "val_primary_MAE": mean_absolute_error(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            # "val_primary_IC": pearsonr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
            # "val_primary_rank_IC": spearmanr(
            #     primary_y_np.flatten(), primary_pred_np.flatten()
            # )[0],
        }
        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)
        return val_loss.detach()

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def _preprocess_stream(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Update the model with a new batch of data (online learning).

        This method performs a single optimization step on the provided data,
        using a small learning rate (`config.lr_refit`) to fine-tune the model.
        `lr_refit` defaults to 0.0, i.e. online updating is OFF unless a caller
        opts in; the guard below returns immediately in that case. The field
        was added to `DLConfig` on 2026-09-07 -- before that this method raised
        `AttributeError` on its first line.

        Args:
            x: Input features tensor of shape (batch_size, num_symbols, num_features).
            y: Target labels tensor of shape (batch_size, num_symbols, num_labels).
        """
        if self.config.lr_refit <= 0.0:
            return

        if not hasattr(self, "model") or self.model is None:
            raise RuntimeError(
                "Model has not been initialized. Please train the model first."
            )

        self.model.train()
        x = x.to(self.device)
        y = y.to(self.device)

        # A dedicated optimizer with the refit learning rate, CACHED on the
        # instance. Building a fresh AdamW per call reset Adam's moment
        # estimates every single step -- silently degrading online training to
        # SGD with an odd warmup. `_get_refit_optim()` rebuilds only when
        # `self.model` is replaced (`load()`, `_init_model()`) or `lr_refit`
        # changes; see its docstring.
        optimizer = self._get_refit_optim()
        optimizer.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Calculate loss consistently with the training step
        loss_direct = self.criterion(all_direct_preds, y)
        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)
        loss = loss_direct + loss_primary_final

        loss.backward()
        optimizer.step()
