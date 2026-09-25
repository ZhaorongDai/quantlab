"""Recurrent up/down classification head for the torch model layer.

``RNNClassifier`` is a ``DLModel`` that turns each future-return label into a
binary class (1 when the return is positive, else 0) and predicts it with a
two-logit ``ModelRCrypto``: one GRU or LSTM tower per label, plus a linear
layer that recombines the auxiliary towers into a second estimate of the
primary label. ``predict_panel`` exposes the probability of an up move per
label, which the backtest layer uses as a ranking score.

The recurrent layers run with ``batch_first=True`` over the tensors exactly
as the data loader yields them, so the batch axis is time and the sequence
axis is the symbol axis of each bar.
"""

import numpy as np
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

from quantlab.base.config import DLConfig
from quantlab.base.model import DLModel


class ModelRBaseCrypto(nn.Module):
    """Stack of GRU or LSTM layers followed by a two-logit linear head.

    Args:
        input_size: Number of input features per sequence element.
        hidden_sizes: Hidden width of each recurrent layer, in order.
        dropout_rates: Dropout applied after each recurrent layer; same
            length as ``hidden_sizes``.
        hidden_sizes_linear: Widths of the hidden linear layers after the
            recurrent stack. May be empty.
        dropout_rates_linear: Dropout after each hidden linear layer; same
            length as ``hidden_sizes_linear``.
        model_type: ``"gru"`` or ``"lstm"``.

    Raises:
        ValueError: If ``model_type`` is neither ``"gru"`` nor ``"lstm"``.

    Example:
        >>> block = ModelRBaseCrypto(
        ...     input_size=3, hidden_sizes=[8, 8], dropout_rates=[0.0, 0.0],
        ...     hidden_sizes_linear=[8], dropout_rates_linear=[0.0],
        ...     model_type="gru",
        ... )
        >>> block(torch.zeros(4, 2, 3)).shape
        torch.Size([4, 2, 2])
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
        """Build the recurrent stack, the dropouts and the linear head."""
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

        # Two logits (down, up) per sequence element.
        fc_layers.append(nn.Linear(current_linear_size, 2))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[D, T, input_size]`` to ``[D, T, 2]`` logits.

        Example:
            >>> block = ModelRBaseCrypto(3, [8], [0.0], [], [], "lstm")
            >>> block.forward(torch.zeros(4, 2, 3)).shape
            torch.Size([4, 2, 2])
        """
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
    """One two-logit ``ModelRBaseCrypto`` per label, with auxiliaries recombined.

    Label 0 is the primary target. Every label, primary included, gets its own
    tower that emits two logits for it. The logits of the auxiliary towers
    are also fed through one linear layer to produce a second, combined pair
    of logits for the primary target.

    Args:
        input_size: Number of input features per sequence element.
        num_labels: Total number of labels; must be at least 2.
        hidden_sizes: Hidden width of each recurrent layer in every tower.
        dropout_rates: Dropout after each recurrent layer.
        hidden_sizes_linear: Widths of the hidden linear layers in every tower.
        dropout_rates_linear: Dropout after each hidden linear layer.
        model_type: ``"gru"`` or ``"lstm"``.

    Raises:
        ValueError: If ``num_labels`` is less than 2.

    Example:
        >>> net = ModelRCrypto(
        ...     input_size=3, num_labels=2, hidden_sizes=[8, 8],
        ...     dropout_rates=[0.0, 0.0], hidden_sizes_linear=[8],
        ...     dropout_rates_linear=[0.0], model_type="gru",
        ... )
        >>> combined, direct = net(torch.zeros(4, 2, 3))
        >>> combined.shape, direct.shape
        (torch.Size([4, 2, 2]), torch.Size([4, 2, 4]))
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
        """Build one tower per label and the combining output layer."""
        super().__init__()
        if num_labels < 2:
            raise ValueError(
                "ModelRCrypto requires at least 2 labels (1 primary, 1+ auxiliary)."
            )

        self.num_labels = num_labels
        self.num_aux_labels = num_labels - 1

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

        # Combines the auxiliary logits into a second pair of logits for the
        # primary target.
        self.out = nn.Linear(self.num_aux_labels * 2, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(combined_primary, all_direct)`` for a ``[D, T, F]`` input.

        Returns:
            A pair. ``combined_primary`` has shape ``[D, T, 2]`` and holds the
            primary target's logits estimated from the auxiliary towers.
            ``all_direct`` has shape ``[D, T, 2 * num_labels]``; channels
            ``[2i, 2i + 1]`` are the direct logits of label ``i``.

        Example:
            >>> net = ModelRCrypto(3, 2, [8], [0.0], [], [], "gru")
            >>> combined, direct = net.forward(torch.zeros(4, 2, 3))
            >>> direct.shape
            torch.Size([4, 2, 4])
        """
        D, T, _ = x.shape

        # Towers 1.. predict the auxiliary labels; tower 0 the primary one.
        aux_preds = [
            self.base_models[i + 1](x) for i in range(self.num_aux_labels)
        ]
        aux_preds_cat = torch.cat(aux_preds, dim=-1)  # [D, T, num_aux * 2]

        primary_pred_final = self.out(
            aux_preds_cat.reshape(D * T, self.num_aux_labels * 2)
        )
        primary_pred_final = primary_pred_final.reshape(D, T, 2)

        primary_pred_direct = self.base_models[0](x)

        # Direct logits for every label, in label order; the loss is computed
        # against these.
        all_direct_preds = torch.cat([primary_pred_direct] + aux_preds, dim=-1)

        return primary_pred_final, all_direct_preds


class RNNClassifier(DLModel):
    """Classify the sign of future returns with ``ModelRCrypto``.

    Labels arrive as future returns and are converted on the fly to class 1
    (return above zero) or class 0. The loss is cross-entropy summed over
    every label's direct logits plus cross-entropy of the combined primary
    logits against label 0. The optimizer is AdamW with ``config.lr``. At
    least two labels are required.

    All five keys must be present in ``config.hyperparameters``:
    ``hidden_sizes``, ``dropout_rates``, ``hidden_sizes_linear``,
    ``dropout_rates_linear`` and ``model_type`` (``"gru"`` or ``"lstm"``).

    ``predict`` returns the raw ``(combined_primary, all_direct)`` logits of
    the module. ``predict_panel`` returns, per label, the softmax probability
    of class 1; these are probabilities in ``[0, 1]``, not return magnitudes.

    Example:
        >>> config = DLConfig(
        ...     factors=[alpha],            # factor objects
        ...     labels=[ret_30, ret_60],    # label objects, primary first
        ...     model_save_dir="checkpoints",
        ...     factor_data_strategy="read",
        ...     label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-02-09",
        ...     test_start="2024-02-10", test_end="2024-02-29",
        ...     epochs=2, batch_size=8, num_workers=0,
        ...     hyperparameters={
        ...         "hidden_sizes": [8, 8], "dropout_rates": [0.0, 0.0],
        ...         "hidden_sizes_linear": [8], "dropout_rates_linear": [0.0],
        ...         "model_type": "gru",
        ...     },
        ... )
        >>> model = RNNClassifier(config)
        >>> model.collect().train().name
        'RNNClassifier_total.pth'
        >>> combined, direct = model.predict(torch.zeros(5, 2, 3))
        >>> combined.shape, direct.shape
        (torch.Size([5, 2, 2]), torch.Size([5, 2, 4]))
    """

    def __init__(self, config: DLConfig):
        """Store the config and create the cross-entropy criterion."""
        super().__init__(config)
        self.criterion = nn.CrossEntropyLoss()

    def _convert_returns_to_labels(self, y: torch.Tensor) -> torch.Tensor:
        """Return ``1`` where ``y > 0`` and ``0`` elsewhere, as a long tensor."""
        return (y > 0).long()

    def _train_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Run one optimizer step on a batch, log train metrics, return the loss."""
        self.optim.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        y_labels = self._convert_returns_to_labels(y)

        # Cross-entropy of each label's direct logit pair against its class.
        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        # Cross-entropy of the combined primary logits against label 0.
        primary_y = y_labels[:, :, 0]
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )

        loss = loss_direct + loss_primary_final

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            # Metrics are reported on the combined primary estimate.
            primary_pred_probs = torch.softmax(primary_pred, dim=-1)
            primary_pred_classes = torch.argmax(primary_pred_probs, dim=-1)
            primary_pred_np = primary_pred_classes.cpu().numpy()
            primary_y_np = primary_y.cpu().numpy()

            # Probability of class 1 (up), for ROC-AUC and average precision.
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
        """Build a ``ModelRCrypto`` from ``hyperparameters``.

        Every key listed on the class is required; a missing one raises
        ``KeyError``. ``num_symbols`` is not needed because the module is
        applied per sequence element.
        """
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
        """Return an AdamW optimizer over ``model`` with ``config.lr``."""
        return torch.optim.AdamW(model.parameters(), lr=self.config.lr)

    def _test_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one test batch, log the test metrics and return the loss."""
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        y_labels = self._convert_returns_to_labels(y)

        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]
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

        # Probability of class 1 (up), for ROC-AUC and average precision.
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

    def _val_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one validation batch, log its metrics and return the loss.

        The epoch loop weights the returned loss by the batch size to form
        the per-epoch validation loss that drives early stopping.
        """
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        y_labels = self._convert_returns_to_labels(y)

        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]
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

        # Probability of class 1 (up), for ROC-AUC and average precision.
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
        """Replace NaN with 0.0 in a tensor produced by ``to_tensor``."""
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def _preprocess_stream(self, data: torch.Tensor) -> torch.Tensor:
        """Replace NaN with 0.0 in a streaming input tensor."""
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Take one online fine-tuning step on a new batch.

        Uses a dedicated AdamW optimizer with learning rate
        ``config.lr_refit``, cached on the instance so that its moment
        estimates persist across calls. Returns immediately when
        ``config.lr_refit`` is ``0.0`` (the default), which disables online
        updating. ``y`` holds future returns and is converted to classes
        the same way as in training.

        Args:
            x: Features of shape ``[batch, num_symbols, num_features]``.
            y: Future returns of shape ``[batch, num_symbols, num_labels]``.

        Raises:
            RuntimeError: If no model has been trained or loaded yet.

        Example:
            >>> model.config.lr_refit = 1e-4
            >>> model.update(torch.randn(4, 2, 3), torch.randn(4, 2, 2))
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

        # The refit optimizer is cached on the instance; a fresh optimizer per
        # call would reset Adam's moment estimates every step.
        optimizer = self._get_refit_optim()
        optimizer.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        y_labels = self._convert_returns_to_labels(y)

        D, T, num_labels = all_direct_preds.shape
        all_direct_preds_reshaped = all_direct_preds.reshape(D * T, num_labels)
        y_labels_reshaped = y_labels.reshape(D * T, -1)

        loss_direct = 0
        for i in range(y_labels.shape[-1]):
            loss_direct += self.criterion(
                all_direct_preds_reshaped[:, i * 2 : (i + 1) * 2],
                y_labels_reshaped[:, i],
            )

        primary_y = y_labels[:, :, 0]
        primary_pred_reshaped = primary_pred.reshape(D * T, 2)
        primary_y_reshaped = primary_y.reshape(D * T)
        loss_primary_final = self.criterion(
            primary_pred_reshaped, primary_y_reshaped
        )
        loss = loss_direct + loss_primary_final

        loss.backward()
        optimizer.step()

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Adapt ``predict_panel``: label ``i`` is the up probability of its logit pair.

        Only ``all_direct`` of the module's output pair is used, shape
        ``[T, S, 2 * L]``, with channels ``[2i, 2i + 1]`` belonging to label
        ``i``. Each label's two logits are passed through a
        softmax and the probability of class 1 (up) is returned, giving a
        ``[T, S, L]`` array. Label 0 is the primary tower's direct prediction,
        not the combined estimate.

        Raises:
            ValueError: If the channel count is not twice the number of
                labels, which means the module does not match the label
                configuration.
        """
        _, direct = self.predict(x)
        num_times, num_symbols, num_channels = direct.shape
        num_labels = self.num_labels
        if num_channels != 2 * num_labels:
            raise ValueError(
                f"{self.class_name}: all_direct_preds has {num_channels} "
                f"channels, expected 2 * {num_labels} labels = {2 * num_labels}"
            )
        probs = torch.softmax(
            direct.reshape(num_times, num_symbols, num_channels // 2, 2), dim=-1
        )[..., 1]
        return probs.detach().cpu().numpy()
