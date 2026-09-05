import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
            nn.Linear(current_linear_size, 2)
        )  # Output 2 classes for up/down classification
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        D, T, _ = x.shape
        recurrent_output = x
        for i, layer in enumerate(self.recurrent_layers):
            recurrent_output, _ = layer(recurrent_output)
            recurrent_output = self.dropout_layers[i](recurrent_output)

        output = recurrent_output.reshape(D * T, -1)
        output = self.fc(output)
        output = output.reshape(D, T, 2)
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
        self.out = nn.Linear(self.num_aux_labels * 2, 2)  # Output 2 classes

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        D, T, _ = x.shape

        # Get predictions from each base model for its corresponding target
        # The first model predicts the primary target, the rest predict auxiliary targets
        aux_preds = [
            self.base_models[i + 1](x) for i in range(self.num_aux_labels)
        ]
        aux_preds_cat = torch.cat(
            aux_preds, dim=-1
        )  # Shape: (D, T, num_aux_labels * 2)

        # Combine auxiliary predictions to form the final prediction for the primary target
        primary_pred_final = self.out(
            aux_preds_cat.reshape(D * T, self.num_aux_labels * 2)
        )
        primary_pred_final = primary_pred_final.reshape(D, T, 2)

        # We also get the direct prediction for the primary target from its own base model
        primary_pred_direct = self.base_models[0](x)

        # The full set of predictions includes the direct one for the primary target
        # and all auxiliary predictions. This is used for calculating the loss.
        all_direct_preds = torch.cat([primary_pred_direct] + aux_preds, dim=-1)

        return primary_pred_final, all_direct_preds


class RNNClassifier(BaseModel):
    """
    A classifier that uses `ModelRCrypto` for up/down predictions, integrated with the
    project's BaseModel interface and training loop.
    """

    def __init__(self, config: DLConfig):
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()

    def _convert_returns_to_labels(self, y: torch.Tensor) -> torch.Tensor:
        """
        Convert future returns to binary labels (0 for down, 1 for up).

        Args:
            y: Future returns tensor of shape (batch_size, time_steps, num_labels)

        Returns:
            Binary labels tensor of shape (batch_size, time_steps, num_labels)
        """
        return (y > 0).long()

    def _train_one_epoch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        self.optim.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Convert future returns to binary labels (0 for down, 1 for up)
        y_labels = self._convert_returns_to_labels(y)

        # Loss calculation for classification
        # 1. Loss for all direct predictions (primary + auxiliaries) vs their targets
        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):  # Loop over each label
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        # 2. Loss for the final combined prediction vs the primary target
        primary_y = y_labels[:, :, 0]  # Primary target class labels
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )

        # Total loss is the sum of the two components
        loss = loss_direct + loss_primary_final

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            # For metrics, we use the final primary prediction
            primary_pred_probs = torch.softmax(primary_pred, dim=-1)
            primary_pred_classes = torch.argmax(primary_pred_probs, dim=-1)
            primary_pred_np = primary_pred_classes.cpu().numpy()
            primary_y_np = primary_y.cpu().numpy()

            # Get probabilities for positive class (class 1) for ROC-AUC
            primary_pred_probs_np = primary_pred_probs[:, :, 1].cpu().numpy()

        metrics = {
            "train_loss": loss.item(),
            "train_loss_direct": loss_direct.item(),
            "train_loss_final": loss_primary_final.item(),
            "train_accuracy": accuracy_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "train_precision": precision_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "train_recall": recall_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "train_f1": f1_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "train_roc_auc": roc_auc_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
            "train_ap": average_precision_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
        }

        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)

        return loss.detach()

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
            hidden_sizes=hyperparameters["hidden_sizes"],
            dropout_rates=hyperparameters["dropout_rates"],
            hidden_sizes_linear=hyperparameters["hidden_sizes_linear"],
            dropout_rates_linear=hyperparameters["dropout_rates_linear"],
            model_type=hyperparameters["model_type"],
        )

    def _init_optim(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.config.lr)

    def _test_one_epoch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Convert future returns to binary labels (0 for down, 1 for up)
        y_labels = self._convert_returns_to_labels(y)

        # Consistent loss calculation for testing
        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):  # Loop over each label
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]  # Primary target class labels
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )
        test_loss = loss_direct + loss_primary_final

        primary_pred_probs = torch.softmax(primary_pred, dim=-1)
        primary_pred_classes = torch.argmax(primary_pred_probs, dim=-1)
        primary_pred_np = primary_pred_classes.cpu().numpy()
        primary_y_np = primary_y.cpu().numpy()

        # Get probabilities for positive class (class 1) for ROC-AUC
        primary_pred_probs_np = primary_pred_probs[:, :, 1].cpu().numpy()

        metrics = {
            "test_loss": test_loss.item(),
            "test_loss_direct": loss_direct.item(),
            "test_loss_final": loss_primary_final.item(),
            "test_accuracy": accuracy_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "test_precision": precision_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "test_recall": recall_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "test_f1": f1_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "test_roc_auc": roc_auc_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
            "test_ap": average_precision_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
        }
        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)

        return test_loss.detach()

    def _val_one_epoch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Convert future returns to binary labels (0 for down, 1 for up)
        y_labels = self._convert_returns_to_labels(y)

        # Consistent loss calculation for validation
        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):  # Loop over each label
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]  # Primary target class labels
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )
        val_loss = loss_direct + loss_primary_final

        primary_pred_probs = torch.softmax(primary_pred, dim=-1)
        primary_pred_classes = torch.argmax(primary_pred_probs, dim=-1)
        primary_pred_np = primary_pred_classes.cpu().numpy()
        primary_y_np = primary_y.cpu().numpy()

        # Get probabilities for positive class (class 1) for ROC-AUC
        primary_pred_probs_np = primary_pred_probs[:, :, 1].cpu().numpy()

        metrics = {
            "val_loss": val_loss.item(),
            "val_loss_direct": loss_direct.item(),
            "val_loss_final": loss_primary_final.item(),
            "val_accuracy": accuracy_score(
                primary_y_np.flatten(), primary_pred_np.flatten()
            ),
            "val_precision": precision_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "val_recall": recall_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "val_f1": f1_score(
                primary_y_np.flatten(),
                primary_pred_np.flatten(),
                average="weighted",
                zero_division=0,
            ),
            "val_roc_auc": roc_auc_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
            "val_ap": average_precision_score(
                primary_y_np.flatten(), primary_pred_probs_np.flatten()
            )
            if len(np.unique(primary_y_np.flatten())) > 1
            else 0.5,
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
        using a small learning rate (`lr_refit`) to fine-tune the model.

        Args:
            x: Input features tensor of shape (batch_size, num_symbols, num_features).
            y: Target returns tensor of shape (batch_size, num_symbols, num_labels).
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

        # Use a dedicated optimizer with the refit learning rate
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr_refit
        )
        optimizer.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Convert future returns to binary labels (0 for down, 1 for up)
        y_labels = self._convert_returns_to_labels(y)

        # Calculate loss consistently with the training step
        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):  # Loop over each label
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]  # Primary target class labels
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )
        loss = loss_direct + loss_primary_final

        loss.backward()
        optimizer.step()

    def _vecbt(self, prices: pd.Series, signals: pd.Series):
        short_entries = signals[signals == 0]
        long_entries = signals[signals == 1]
        short_exits = long_entries[long_entries == 1]
        long_exits = long_entries[short_entries == 1]
