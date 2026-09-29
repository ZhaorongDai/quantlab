"""GATs: Qlib's graph attention model over one bar's cross-section.

``GATsNet`` reproduces Qlib's ``GATModel`` (``qlib/contrib/model/pytorch_gats_ts.py``):
a recurrent encoder per symbol, one fully connected attention head over
every symbol of the bar, a residual and a two-layer output head.
``GATsRegressor`` is the ``TorchModel`` head that trains it with Qlib's
Alpha158 benchmark settings. See ``docs/research/qlib-gats-master.md`` §1.
"""

import copy

import torch
from torch import nn

from quantlab.base.torch_model import TorchModel
from quantlab.utils.torch_training import cs_rank_norm, masked_mse


class GATsNet(nn.Module):
    """Qlib's ``GATModel`` with an output of any width.

    Each symbol's window ``[N, F]`` goes through an LSTM or GRU, and the
    hidden state of the last bar ``h`` (``[S_t, H]``) is kept. With
    ``x = W h + b`` (``transformation``), the attention score of symbol
    ``i`` on symbol ``j`` is ``LeakyReLU(a[:H]·x_j + a[H:]·x_i)`` with
    slope 0.01, normalised by a softmax over every ``j`` of the bar, self
    included. The output is ``fc_out(LeakyReLU(fc(h + A h)))``: the
    untransformed ``h`` is aggregated, as in Qlib. The parameter names are
    Qlib's, so a Qlib ``GATModel`` state dict loads into a one-label net.

    The scores are computed as a sum of two ``[S_t, 1]`` projections
    rather than Qlib's ``[S_t, S_t, 2H]`` concatenation, which is the same
    value without the memory that grows with the square of the
    cross-section.

    Parameters
    ----------
    num_features : int
        Features per bar, ``F``.
    num_labels : int
        Outputs per symbol, ``L``; Qlib's model has one.
    hidden_size : int, default 64
        ``H``, the encoder's hidden size.
    num_layers : int, default 2
        Encoder layers.
    dropout : float, default 0.0
        Dropout between encoder layers.
    base_model : {"LSTM", "GRU"}, default "LSTM"
        The encoder.

    Raises
    ------
    ValueError
        If ``base_model`` is neither ``"LSTM"`` nor ``"GRU"``.

    Examples
    --------
    >>> net = GATsNet(num_features=4, num_labels=2, hidden_size=8)
    >>> net(torch.randn(5, 3, 4)).shape
    torch.Size([5, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        base_model: str = "LSTM",
    ):
        super().__init__()
        encoders = {"LSTM": nn.LSTM, "GRU": nn.GRU}
        if base_model not in encoders:
            raise ValueError(
                f"base_model must be one of {list(encoders)}, got {base_model!r}"
            )
        self.rnn = encoders[base_model](
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden_size = hidden_size
        self.transformation = nn.Linear(hidden_size, hidden_size)
        self.a = nn.Parameter(torch.randn(hidden_size * 2, 1))
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_labels)
        self.leaky_relu = nn.LeakyReLU()

    def attention(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return the ``[S_t, S_t]`` attention weights, each row summing to 1.

        Examples
        --------
        >>> weights = net.attention(torch.randn(5, 8))
        >>> weights.shape, bool(torch.allclose(weights.sum(dim=1), torch.ones(5)))
        (torch.Size([5, 5]), True)
        """
        x = self.transformation(hidden)
        size = self.hidden_size
        # score[i, j] = a[:H]·x_j + a[H:]·x_i
        scores = (x @ self.a[:size]).T + x @ self.a[size:]
        return torch.softmax(self.leaky_relu(scores), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map one bar's ``[S_t, N, F]`` windows to ``[S_t, L]`` outputs.

        Examples
        --------
        >>> net.forward(torch.randn(3, 6, 4)).shape
        torch.Size([3, 2])
        """
        out, _ = self.rnn(x)
        hidden = out[:, -1, :]
        hidden = self.attention(hidden) @ hidden + hidden
        return self.fc_out(self.leaky_relu(self.fc(hidden)))


class GATsRegressor(TorchModel):
    """Qlib's GATs on the cross-section of every bar.

    Each step is one bar: every symbol with a finite feature, each with its
    last ``window_bars`` bars, goes through ``GATsNet``. The training target
    is Qlib's ``CSRankNorm`` of the label (``cs_rank_norm``), the loss is
    the MSE over symbols with a label, the optimizer is Adam, and gradient
    values are clipped at 3. Training keeps the epoch with the lowest
    validation loss and stops after ``early_stop`` epochs without a strict
    improvement, as Qlib does; without a validation segment it runs every
    epoch and keeps the last.

    Hyperparameters, with Qlib's Alpha158 benchmark values as defaults:
    ``window_bars`` (20), ``hidden_size`` (64), ``num_layers`` (2),
    ``dropout`` (0.7), ``base_model`` (``"LSTM"``), ``lr`` (1e-4),
    ``epochs`` (200) and ``early_stop`` (10).

    Known differences from Qlib's implementation:

    - no pretrained LSTM: Qlib copies the encoder and ``fc_out`` from its
      LSTM benchmark's checkpoint, here every weight starts random;
    - the features are the model's factors, not Qlib's 20 filtered Alpha158
      columns, and no ``RobustZScoreNorm`` is fitted on the training span:
      the default ``_transform_feature`` clips to ±3 and fills NaN with 0,
      where Qlib forward- and back-fills gaps inside a window;
    - the bars of an epoch are shuffled, as in Qlib's Alpha360 variant; the
      Alpha158 variant visits them in time order;
    - a symbol whose label is missing stays in the bar's cross-section as
      context and only leaves the loss; Qlib drops it from that day's
      training and validation input;
    - the output has one column per label instead of one;
    - the last training batch is kept, where Qlib drops it.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Examples
    --------
    >>> head = GATsRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"window_bars": 5, "epochs": 20, "early_stop": 5},
    ... )).collect()
    >>> head.train().name
    'GATsRegressor_total.pth'
    """

    #: Qlib's Alpha158 benchmark values, used for every unset hyperparameter.
    DEFAULTS: dict = {
        "window_bars": 20,
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.7,
        "base_model": "LSTM",
        "lr": 1e-4,
        "epochs": 200,
        "early_stop": 10,
    }

    def _setting(self, key: str):
        """``hyperparameters[key]``, or its ``DEFAULTS`` value when unset."""
        return self.config.hyperparameters.get(key, self.DEFAULTS[key])

    @property
    def window_bars(self) -> int:
        """Bars in each symbol's window, ``hyperparameters["window_bars"]`` (20).

        Examples
        --------
        >>> head.window_bars
        5
        """
        return int(self._setting("window_bars"))

    @property
    def epochs(self) -> int:
        """The epoch cap, ``hyperparameters["epochs"]``, 200 when unset.

        Raises
        ------
        ValueError
            If the value is not a positive integer.

        Examples
        --------
        >>> head.epochs
        20
        """
        if "epochs" not in self.config.hyperparameters:
            return int(self.DEFAULTS["epochs"])
        return super().epochs

    @property
    def early_stop(self) -> int:
        """Epochs without a better validation loss before training stops (10).

        Examples
        --------
        >>> head.early_stop
        5
        """
        return int(self._setting("early_stop"))

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> GATsNet:
        """Build ``GATsNet`` from the encoder hyperparameters."""
        return GATsNet(
            num_features=num_features,
            num_labels=num_labels,
            hidden_size=int(self._setting("hidden_size")),
            num_layers=int(self._setting("num_layers")),
            dropout=float(self._setting("dropout")),
            base_model=str(self._setting("base_model")),
        )

    def _init_optim(self, model: torch.nn.Module):
        """Adam at ``hyperparameters["lr"]``, 1e-4 when unset."""
        return torch.optim.Adam(model.parameters(), lr=float(self._setting("lr")))

    def _loss(self, output: torch.Tensor, batch) -> torch.Tensor:
        """MSE between the output and the rank target over symbols with a label."""
        return masked_mse(output, batch.y, batch.mask)

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Qlib's ``CSRankNorm`` of the bar's labels; no symbol is dropped."""
        return cs_rank_norm(y), None

    def _on_fit_start(self) -> None:
        """Reset the best validation loss, its weights and the patience count."""
        self._best_loss = float("inf")
        self._best_state = None
        self._bad_epochs = 0

    def _should_stop(self, epoch: int, train_loss: float, val_loss: float | None) -> bool:
        """Keep the weights of a strictly better validation loss; stop after ``early_stop`` misses."""
        if val_loss is None:
            return False
        if val_loss < self._best_loss:
            self._best_loss, self._bad_epochs = val_loss, 0
            self._best_state = copy.deepcopy(self.model.state_dict())
            return False
        self._bad_epochs += 1
        return self._bad_epochs >= self.early_stop

    def _on_fit_end(self) -> None:
        """Restore the weights of the best validation epoch, if there was one."""
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        self._best_state = None
