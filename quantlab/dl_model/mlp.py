"""Feed-forward regression head for the torch model layer.

``MLPRegressor`` is a ``DLModel`` (the torch training loop defined in
``quantlab.base.model``). It flattens every bar of the factor panel into one
row, with all symbols side by side, and maps that row to every label of every
symbol with a two-hidden-layer ``MLP``. A *panel* is an ``xarray.Dataset``
indexed by ``timestamp`` and ``symbol``. The factor and label layers supply
the ``[num_times, num_symbols, *]`` tensors cut from their panels, and the
backtest layer consumes the predictions through ``predict_panel``.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from quantlab.base.config import DLConfig
from quantlab.base.model import DLModel


class MLP(nn.Module):
    """Two-hidden-layer perceptron with ReLU activations.

    Parameters
    ----------
    input_size : int
        Width of the input rows.
    hidden_size1 : int
        Width of the first hidden layer.
    hidden_size2 : int
        Width of the second hidden layer.
    output_size : int
        Width of the output rows.

    Examples
    --------
    >>> net = MLP(input_size=6, hidden_size1=16, hidden_size2=8, output_size=4)
    >>> net(torch.zeros(5, 6)).shape
    torch.Size([5, 4])
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        """Build the three linear layers and their activations."""
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        """Map a ``[batch, input_size]`` matrix to ``[batch, output_size]``.

        Examples
        --------
        >>> net = MLP(6, 16, 8, 4)
        >>> net.forward(torch.zeros(5, 6)).shape
        torch.Size([5, 4])
        """
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


class MLPRegressor(DLModel):
    """Regress future returns with an ``MLP`` over the flattened cross-section.

    Each bar ``t`` becomes one training row: the ``[num_symbols, num_features]``
    slice is flattened to ``num_symbols * num_features`` inputs and the target
    is the flattened ``[num_symbols, num_labels]`` slice. The network therefore
    encodes symbol position, so predictions must be made on the same symbol
    axis the model was trained on; ``DLModel`` aligns the panel for
    ``predict_panel``. The loss is mean squared error and the optimizer is
    Adam with ``config.lr``.

    Hyperparameters read from ``config.hyperparameters``: ``hidden_size1``
    (default 512) and ``hidden_size2`` (default 256).

    The public ``predict`` takes the flattened
    ``[num_times, num_symbols * num_features]`` matrix and returns the
    flattened ``[num_times, num_symbols * num_labels]`` output, while
    ``predict_panel`` works on the ``(timestamp, symbol)`` panel.

    Parameters
    ----------
    config : DLConfig
        Factors, labels, date ranges and training settings. See ``DLConfig``.

    Examples
    --------
    >>> config = DLConfig(
    ...     factors=[alpha],            # factor objects
    ...     labels=[fwd_return],        # label objects
    ...     model_save_dir="checkpoints",
    ...     factor_data_strategy="read",
    ...     label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-02-09",
    ...     test_start="2024-02-10", test_end="2024-02-29",
    ...     epochs=2, batch_size=8, num_workers=0,
    ...     hyperparameters={"hidden_size1": 16, "hidden_size2": 8},
    ... )
    >>> model = MLPRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'MLPRegressor_total.pth'
    >>> flat = torch.zeros(5, model.num_symbols * model.num_factors)
    >>> out = model.predict(flat)  # shape [5, num_symbols * num_labels]
    """

    def __init__(self, config: DLConfig):
        """Store the config and create the MSE criterion."""
        super().__init__(config)
        self.criterion = nn.MSELoss()

    def _train_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        """Run one optimizer step on a flattened batch and log train metrics.

        The batch arrives as ``[batch, num_symbols, num_*]`` and is flattened
        to one row per bar before the forward pass.
        """
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
        """Build the ``MLP`` sized for the flattened panel and move it to the device.

        The input width is ``num_symbols * num_features`` and the output width
        ``num_symbols * num_labels``. Hidden widths come from
        ``hyperparameters`` with defaults of 512 and 256.
        """
        input_size = num_symbols * num_features
        output_size = num_symbols * num_labels
        hidden_size1 = hyperparameters.get("hidden_size1", 512)
        hidden_size2 = hyperparameters.get("hidden_size2", 256)
        return MLP(input_size, hidden_size1, hidden_size2, output_size).to(
            self.device
        )

    def _init_optim(self, model):
        """Return an Adam optimizer over ``model`` with ``config.lr``."""
        return torch.optim.Adam(model.parameters(), lr=self.config.lr)

    def _test_one_batch(self, epoch: int, x: torch.Tensor, y: torch.Tensor):
        """Evaluate one flattened test batch and log the test metrics."""
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
        """Evaluate one flattened validation batch and return its detached loss.

        The epoch loop weights the returned loss by the batch size to form
        the per-epoch validation loss that drives early stopping, so this
        must return a tensor, not just log metrics.
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

    def _predict_panel_array(self, x: np.ndarray) -> np.ndarray:
        """Predict a ``[T, S, F]`` array as ``[T, S, L]`` for ``predict_panel``.

        ``T`` is bars, ``S`` symbols, ``F`` features and ``L`` labels. The
        network consumes ``[T, S * F]`` and emits ``[T, S * L]``, both in C
        order with the symbol axis outermost, exactly as the training step
        flattens them. Reshaping the output back therefore restores the
        per-symbol layout. The public ``predict`` keeps the flat shapes.
        """
        num_times, num_symbols, num_features = x.shape
        flat = x.reshape(num_times, num_symbols * num_features)
        out = self.predict(flat)
        out = out.detach().cpu().numpy()  # type: ignore[union-attr]
        return out.reshape(num_times, num_symbols, -1)

    def _preprocess(self, data: torch.Tensor) -> torch.Tensor:
        """Replace NaN with 0.0 in a tensor produced by ``to_tensor``."""
        return torch.nan_to_num(data, nan=0.0)
