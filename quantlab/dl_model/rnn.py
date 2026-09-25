"""Recurrent regression head with auxiliary targets for the torch model layer.

``RNNRegressor`` is a ``DLModel`` built on ``ModelRCrypto``: one GRU or LSTM
tower per label, where the first label is the primary target and the others
are auxiliary targets whose direct predictions are also combined linearly
into a second estimate of the primary target. The head sits between the
factor and label layers, which supply ``[num_times, num_symbols, *]``
tensors, and the backtest layer, which consumes ``predict_panel``.

The recurrent layers run with ``batch_first=True`` over the tensors exactly
as the data loader yields them, so the batch axis is time and the sequence
axis is the symbol axis of each bar.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from quantlab.base.config import DLConfig
from quantlab.base.model import DLModel


class ModelRBaseCrypto(nn.Module):
    """Stack of GRU or LSTM layers followed by a linear head with one output.

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
        torch.Size([4, 2, 1])
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

        # One regression output per sequence element.
        fc_layers.append(nn.Linear(current_linear_size, 1))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[D, T, input_size]`` to ``[D, T, 1]``.

        The linear head is applied to every sequence element, so the output
        keeps the batch and sequence axes of the input.

        Example:
            >>> block = ModelRBaseCrypto(3, [8], [0.0], [], [], "lstm")
            >>> block.forward(torch.zeros(4, 2, 3)).shape
            torch.Size([4, 2, 1])
        """
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
    """One ``ModelRBaseCrypto`` per label, with the auxiliaries recombined.

    Label 0 is the primary target. Every label, primary included, gets its own
    tower that predicts it directly. The direct predictions of the auxiliary
    labels are also fed through one linear layer to produce a second,
    combined estimate of the primary target.

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
        (torch.Size([4, 2, 1]), torch.Size([4, 2, 2]))
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

        # Combines the auxiliary predictions into a second estimate of the
        # primary target.
        self.out = nn.Linear(self.num_aux_labels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(combined_primary, all_direct)`` for a ``[D, T, F]`` input.

        Returns:
            A pair. ``combined_primary`` has shape ``[D, T, 1]`` and is the
            primary target estimated from the auxiliary predictions.
            ``all_direct`` has shape ``[D, T, num_labels]`` with channel
            ``i`` the direct prediction of label ``i`` by its own tower.

        Example:
            >>> net = ModelRCrypto(3, 2, [8], [0.0], [], [], "gru")
            >>> combined, direct = net.forward(torch.zeros(4, 2, 3))
            >>> direct.shape
            torch.Size([4, 2, 2])
        """
        D, T, _ = x.shape

        # Towers 1.. predict the auxiliary labels; tower 0 the primary one.
        aux_preds = [
            self.base_models[i + 1](x) for i in range(self.num_aux_labels)
        ]
        aux_preds_cat = torch.cat(aux_preds, dim=-1)  # [D, T, num_aux_labels]

        primary_pred_final = self.out(
            aux_preds_cat.reshape(D * T, self.num_aux_labels)
        )
        primary_pred_final = primary_pred_final.reshape(D, T, 1)

        primary_pred_direct = self.base_models[0](x)

        # Direct predictions for every label, in label order; the loss is
        # computed against these.
        all_direct_preds = torch.cat([primary_pred_direct] + aux_preds, dim=-1)

        return primary_pred_final, all_direct_preds


class RNNRegressor(DLModel):
    """Regress future returns with ``ModelRCrypto`` inside the ``DLModel`` loop.

    The loss is the sum of two MSE terms: every direct prediction against its
    own label, and the combined primary estimate against label 0. The
    optimizer is AdamW with ``config.lr``. At least two labels are required.

    Hyperparameters read from ``config.hyperparameters``, with defaults:
    ``hidden_sizes`` (``[256, 128, 64]``), ``dropout_rates``
    (``[0.1, 0.1, 0.1]``), ``hidden_sizes_linear`` (``[32]``),
    ``dropout_rates_linear`` (``[0.1]``) and ``model_type`` (``"gru"``).

    ``predict`` returns the raw ``(combined_primary, all_direct)`` pair of the
    module; ``predict_panel`` exposes one variable per label taken from the
    direct channels.

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
        >>> model = RNNRegressor(config)
        >>> model.collect().train().name
        'RNNRegressor_total.pth'
        >>> combined, direct = model.predict(torch.zeros(5, 2, 3))
        >>> combined.shape, direct.shape
        (torch.Size([5, 2, 1]), torch.Size([5, 2, 2]))
    """

    def __init__(self, config: DLConfig):
        """Store the config and create the MSE criterion."""
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def _train_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        """Run one optimizer step on a batch and log the training metrics."""
        self.optim.zero_grad()

        primary_pred, all_direct_preds = self.model(x)  # type: ignore

        # Direct predictions (primary and auxiliaries) against their labels,
        # plus the combined primary estimate against label 0.
        loss_direct = self.criterion(all_direct_preds, y)

        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)

        loss = loss_direct + loss_primary_final

        loss.backward()
        self.optim.step()

        with torch.no_grad():
            # Metrics are reported on the combined primary estimate.
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
        """Build a ``ModelRCrypto`` from ``hyperparameters``.

        Every key is optional; missing keys fall back to the defaults listed
        on the class. ``num_symbols`` is not needed because the module is
        applied per sequence element.
        """
        return ModelRCrypto(
            input_size=num_features,
            num_labels=num_labels,
            hidden_sizes=hyperparameters.get("hidden_sizes", [256, 128, 64]),
            dropout_rates=hyperparameters.get(
                "dropout_rates", [0.1, 0.1, 0.1]
            ),
            hidden_sizes_linear=hyperparameters.get(
                "hidden_sizes_linear", [32]
            ),
            dropout_rates_linear=hyperparameters.get(
                "dropout_rates_linear", [0.1]
            ),
            model_type=hyperparameters.get("model_type", "gru"),
        )

    def _init_optim(self, model):
        """Return an AdamW optimizer over ``model`` with ``config.lr``."""
        return torch.optim.AdamW(model.parameters(), lr=self.config.lr)

    def _test_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        """Evaluate one test batch with the training loss and log the metrics."""
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

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
        }
        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)

    def _val_one_batch(
        self, epoch: int, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate one validation batch, log its metrics and return the loss.

        The epoch loop weights the returned loss by the batch size to form
        the per-epoch validation loss that drives early stopping, so this
        must return a tensor.
        """
        primary_pred, all_direct_preds = self.model(x)  # type: ignore

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
        }
        if self._wandb_recorder:
            self._wandb_recorder.log(metrics, step=epoch)
        return val_loss.detach()

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Replace NaN with 0.0 in a tensor produced by ``to_tensor``."""
        data = torch.nan_to_num(data, nan=0.0)
        return data

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Adapt ``predict_panel``: label ``i`` is direct channel ``i``.

        The module returns a pair, which the generic ``DLModel`` path cannot
        consume. Only ``all_direct`` is used, shape ``[T, S, L]``, so label 0
        is the primary tower's direct prediction rather than the combined
        estimate.
        """
        _, direct = self.predict(x)
        return direct.detach().cpu().numpy()

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
        updating.

        Args:
            x: Features of shape ``[batch, num_symbols, num_features]``.
            y: Labels of shape ``[batch, num_symbols, num_labels]``.

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

        loss_direct = self.criterion(all_direct_preds, y)
        primary_y = y[:, :, 0].unsqueeze(-1)
        loss_primary_final = self.criterion(primary_pred, primary_y)
        loss = loss_direct + loss_primary_final

        loss.backward()
        optimizer.step()
