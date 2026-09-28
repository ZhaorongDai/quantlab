"""MASTER: a market-guided transformer over one bar's cross-section.

``MASTERNet`` reproduces the network of ``SJTU-DMTai/MASTER`` (``master.py``,
Li et al., "MASTER: Market-Guided Stock Transformer for Stock Price
Forecasting", AAAI 2024): a market gate that rescales the stock features,
attention over time within each stock, attention across stocks at every
time step, and a temporal aggregation queried by the last step.
``MASTERRegressor`` is the ``TorchModel`` head that trains it with the
official settings. See ``docs/research/qlib-gats-master.md`` §2.
"""

import math

import torch
from torch import nn

from quantlab.base.config import ModelConfig
from quantlab.base.model import TorchModel
from quantlab.torch_model.training import (
    TrainLossThreshold,
    cs_zscore,
    drop_extreme,
    masked_mse,
)


class PositionalEncoding(nn.Module):
    """Adds the fixed sinusoidal encoding of each step's position, up to ``max_len`` steps.

    Examples
    --------
    >>> PositionalEncoding(d_model=4)(torch.zeros(2, 3, 4))[0, 1]
    tensor([0.8415, 0.5403, 0.0100, 0.9999])
    """

    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` (``[S_t, N, D]``) plus the encoding of its N steps."""
        return x + self.pe[: x.shape[1], :]


class AttentionBlock(nn.Module):
    """MASTER's pre-LayerNorm multi-head attention block with a feed-forward layer.

    ``x`` is ``[S_t, N, D]``. With ``across_symbols=False`` (MASTER's
    ``TAttention``) each symbol attends over its own N steps and the scores
    are not scaled; with ``across_symbols=True`` (``SAttention``) the
    symbols attend to each other at every step and the scores are divided
    by ``sqrt(D / nhead)``. Both return
    ``norm2(LN(x) + MHA(LN(x)))`` plus the feed-forward layer of that sum.

    Examples
    --------
    >>> block = AttentionBlock(d_model=8, nhead=2, dropout=0.0, across_symbols=True)
    >>> block(torch.randn(5, 3, 8)).shape
    torch.Size([5, 3, 8])
    """

    def __init__(self, d_model: int, nhead: int, dropout: float, across_symbols: bool):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.across_symbols = across_symbols
        self.temperature = math.sqrt(d_model / nhead) if across_symbols else 1.0
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Attend over steps or over symbols; ``[S_t, N, D]`` in and out."""
        x = self.norm1(x)
        q, k, v = self.qtrans(x), self.ktrans(x), self.vtrans(x)
        if self.across_symbols:
            q, k, v = (t.transpose(0, 1) for t in (q, k, v))  # [N, S_t, D]
        batch, length, _ = q.shape
        split = lambda t: t.reshape(batch, length, self.nhead, -1).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)  # [B, heads, length, D / heads]
        weights = torch.softmax(q @ k.transpose(-1, -2) / self.temperature, dim=-1)
        out = (self.attn_dropout(weights) @ v).transpose(1, 2).reshape(batch, length, -1)
        if self.across_symbols:
            out = out.transpose(0, 1)
        xt = self.norm2(x + out)
        return xt + self.ffn(xt)


class Gate(nn.Module):
    """``F · softmax(Linear(market) / beta)``: weights for F features that sum to F.

    Examples
    --------
    >>> weights = Gate(2, 5, beta=5.0)(torch.randn(3, 2))
    >>> weights.shape, [round(w, 4) for w in weights.sum(dim=-1).tolist()]
    (torch.Size([3, 5]), [5.0, 5.0, 5.0])
    """

    def __init__(self, d_input: int, d_output: int, beta: float):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.beta = beta

    def forward(self, market: torch.Tensor) -> torch.Tensor:
        """Map ``[S_t, G]`` market features to ``[S_t, F]`` feature weights."""
        return self.d_output * torch.softmax(self.trans(market) / self.beta, dim=-1)


class TemporalAttention(nn.Module):
    """Aggregates a symbol's N steps with weights ``softmax_t((W z_t)·(W z_N))``.

    Examples
    --------
    >>> TemporalAttention(8)(torch.randn(5, 3, 8)).shape
    torch.Size([5, 8])
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Map ``[S_t, N, D]`` to ``[S_t, D]``."""
        h = self.trans(z)
        weights = torch.softmax(h @ h[:, -1, :].unsqueeze(-1), dim=1)  # [S_t, N, 1]
        return (weights * z).sum(dim=1)


class MASTERNet(nn.Module):
    """The MASTER network with its market gate columns anywhere among the features.

    The input is one bar's ``[S_t, N, F]`` windows. The columns
    ``gate_columns`` are the market features: their values at the last
    step go through ``Gate`` to weights that rescale the other ``F - G``
    columns at every step. The rescaled features then pass
    ``Linear(F - G, D)``, ``PositionalEncoding``, attention over time
    within each symbol (4 heads, unscaled), attention across symbols at
    every step (2 heads, scaled), ``TemporalAttention`` and ``Linear(D, L)``.
    The submodule names and order are the official ones, so an official
    ``MASTER`` state dict loads into a one-label net whose gate columns are
    the trailing ones.

    Parameters
    ----------
    num_features : int
        F, every column of the input, gate columns included.
    num_labels : int
        L, outputs per symbol; the official network has one.
    gate_columns : list of int
        Positions of the market features among the F columns.
    d_model : int, default 256
        D.
    t_nhead, s_nhead : int, default 4 and 2
        Heads of the attention over time and across symbols.
    dropout : float, default 0.5
        Dropout of both attention blocks.
    beta : float, default 5.0
        Gate temperature; a smaller beta selects features more sharply.

    Raises
    ------
    ValueError
        If ``gate_columns`` is empty, covers every column or repeats one,
        or a head count does not divide ``d_model``.

    Examples
    --------
    >>> net = MASTERNet(num_features=5, num_labels=2, gate_columns=[3, 4],
    ...                 d_model=8, t_nhead=2, s_nhead=2)
    >>> net(torch.randn(6, 4, 5)).shape
    torch.Size([6, 2])
    """

    def __init__(
        self,
        num_features: int,
        num_labels: int,
        gate_columns: list[int],
        d_model: int = 256,
        t_nhead: int = 4,
        s_nhead: int = 2,
        dropout: float = 0.5,
        beta: float = 5.0,
    ):
        super().__init__()
        gate = sorted(int(c) for c in gate_columns)
        if not gate or len(set(gate)) != len(gate) or len(gate) >= num_features:
            raise ValueError(
                f"gate_columns must name some but not all of the {num_features} "
                f"columns, each once; got {list(gate_columns)}"
            )
        for heads in (t_nhead, s_nhead):
            if d_model % heads:
                raise ValueError(f"d_model={d_model} is not divisible by {heads} heads")
        stock = [c for c in range(num_features) if c not in set(gate)]
        self.register_buffer("gate_index", torch.tensor(list(gate_columns)), persistent=False)
        self.register_buffer("stock_index", torch.tensor(stock), persistent=False)
        self.d_model = d_model
        self.beta = beta
        self.feature_gate = Gate(len(gate), len(stock), beta=beta)
        self.layers = nn.Sequential(
            nn.Linear(len(stock), d_model),
            PositionalEncoding(d_model),
            AttentionBlock(d_model, t_nhead, dropout, across_symbols=False),
            AttentionBlock(d_model, s_nhead, dropout, across_symbols=True),
            TemporalAttention(d_model),
            nn.Linear(d_model, num_labels),
        )

    def gate(self, market: torch.Tensor) -> torch.Tensor:
        """Return the ``[S_t, F - G]`` feature weights for ``[S_t, G]`` market features.

        Each row sums to ``F - G``.

        Examples
        --------
        >>> net.gate(torch.zeros(1, 2)).sum().item()
        3.0
        """
        return self.feature_gate(market)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map one bar's ``[S_t, N, F]`` windows to ``[S_t, L]`` outputs."""
        stock = x[..., self.stock_index]
        market = x[:, -1, self.gate_index]
        return self.layers(stock * self.gate(market).unsqueeze(1))


class MASTERRegressor(TorchModel):
    """MASTER on the cross-section of every bar, gated by the ``gate_features`` factors.

    Each step is one bar: every symbol with a finite feature, each with its
    last ``window_bars`` bars, goes through ``MASTERNet``. The factors named
    by ``hyperparameters["gate_features"]`` are the market features (for
    example those of ``quantlab.factor.market``, the same at every symbol);
    every other factor is a stock feature they gate. In training the label's
    top and bottom ``drop_extreme`` share leave the loss, and the rest is
    z-scored per bar; validation and test targets are z-scored only. The
    loss is the MSE over symbols with a target, the optimizer is Adam, and
    gradient values are clipped at 3. Training stops at the first epoch
    whose training loss is at or below ``train_loss_threshold``, or after
    ``epochs``, and keeps the last weights, as the official code does.

    Hyperparameters, with the official values as defaults:
    ``gate_features`` (required), ``window_bars`` (8), ``d_model`` (256),
    ``t_nhead`` (4), ``s_nhead`` (2), ``dropout`` (0.5), ``beta`` (5.0; the
    paper uses 2 for CSI800), ``lr`` (1e-5), ``epochs`` (40),
    ``train_loss_threshold`` (0.95) and ``drop_extreme`` (0.025).

    Known differences from the official implementation:

    - the market features are whatever ``gate_features`` names, for US
      equities the SPY/QQQ/IWM features of the market factor rather than
      the CSI300/500/800 indices;
    - a symbol dropped by ``drop_extreme`` leaves the loss but stays in the
      bar's cross-section as context; the official code removes it from
      that day's input too;
    - no ``RobustZScoreNorm`` is fitted on the training span: the default
      ``_transform_feature`` clips to ±3 and fills NaN with 0, where the
      official data forward- and back-fills gaps inside a window;
    - the output has one column per label instead of one.

    Parameters
    ----------
    config : ModelConfig
        Factors, labels, dates and hyperparameters.

    Raises
    ------
    ValueError
        If ``gate_features`` is missing or empty, names a factor the model
        does not have (the error lists them), or names every factor.

    Examples
    --------
    >>> head = MASTERRegressor(ModelConfig(
    ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
    ...     factor_data_strategy="read", label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-05-31",
    ...     test_start="2024-06-01", test_end="2024-07-18",
    ...     hyperparameters={"gate_features": ["f_b"], "window_bars": 5,
    ...                      "d_model": 16, "epochs": 5},
    ... )).collect()
    >>> head.train().name
    'MASTERRegressor_total.pth'
    """

    #: The official values, used for every unset hyperparameter.
    DEFAULTS: dict = {
        "window_bars": 8,
        "d_model": 256,
        "t_nhead": 4,
        "s_nhead": 2,
        "dropout": 0.5,
        "beta": 5.0,
        "lr": 1e-5,
        "epochs": 40,
        "train_loss_threshold": 0.95,
        "drop_extreme": 0.025,
    }

    def __init__(self, config: ModelConfig):
        """Build the head and check ``gate_features`` against the factors."""
        super().__init__(config)
        self.gate_columns

    def _setting(self, key: str):
        """``hyperparameters[key]``, or its ``DEFAULTS`` value when unset."""
        return self.config.hyperparameters.get(key, self.DEFAULTS[key])

    @property
    def gate_columns(self) -> list[int]:
        """Positions of the ``gate_features`` factors in ``get_factor_names()``.

        Raises
        ------
        ValueError
            If ``gate_features`` is missing or empty, names a factor the
            model does not have, or names every factor.

        Examples
        --------
        >>> head.get_factor_names(), head.gate_columns
        (['f_a', 'f_b'], [1])
        """
        gate = list(self.config.hyperparameters.get("gate_features") or [])
        names = [str(name) for name in self.get_factor_names()]
        if not gate:
            raise ValueError(
                f"{self.class_name}: hyperparameters['gate_features'] must name the "
                f"market factors that gate the others"
            )
        missing = [name for name in gate if name not in names]
        if missing:
            raise ValueError(
                f"{self.class_name}: gate_features {missing} are not among the "
                f"model's factors"
            )
        if len(set(gate)) >= len(names):
            raise ValueError(
                f"{self.class_name}: gate_features names every factor; at least one "
                f"stock feature must remain to be gated"
            )
        return [names.index(name) for name in gate]

    @property
    def window_bars(self) -> int:
        """Bars in each symbol's window, ``hyperparameters["window_bars"]`` (8).

        Examples
        --------
        >>> head.window_bars
        5
        """
        return int(self._setting("window_bars"))

    @property
    def epochs(self) -> int:
        """The epoch cap, ``hyperparameters["epochs"]``, 40 when unset.

        Raises
        ------
        ValueError
            If the value is not a positive integer.

        Examples
        --------
        >>> head.epochs
        5
        """
        if "epochs" not in self.config.hyperparameters:
            return int(self.DEFAULTS["epochs"])
        return super().epochs

    def _init_model(self, num_features: int, num_labels: int, hyperparameters: dict) -> MASTERNet:
        """Build ``MASTERNet`` with the gate columns of ``gate_features``."""
        return MASTERNet(
            num_features=num_features,
            num_labels=num_labels,
            gate_columns=self.gate_columns,
            d_model=int(self._setting("d_model")),
            t_nhead=int(self._setting("t_nhead")),
            s_nhead=int(self._setting("s_nhead")),
            dropout=float(self._setting("dropout")),
            beta=float(self._setting("beta")),
        )

    def _init_optim(self, model: torch.nn.Module):
        """Adam at ``hyperparameters["lr"]``, 1e-5 when unset."""
        return torch.optim.Adam(model.parameters(), lr=float(self._setting("lr")))

    def _loss(self, output: torch.Tensor, batch) -> torch.Tensor:
        """MSE between the output and the target over symbols with a target."""
        return masked_mse(output, batch.y, batch.mask)

    def _transform_target(self, y: torch.Tensor, training: bool):
        """Drop both tails of the first label in training, then z-score the bar."""
        if not training:
            return cs_zscore(y), None
        keep = drop_extreme(y, float(self._setting("drop_extreme")))
        return cs_zscore(y[keep]), keep

    def _on_fit_start(self) -> None:
        """Start the training-loss threshold rule for this fit."""
        self._stop_rule = TrainLossThreshold(
            float(self._setting("train_loss_threshold")), self.epochs
        )

    def _should_stop(self, epoch: int, train_loss: float, val_loss: float | None) -> bool:
        """Stop once the epoch's training loss is at or below the threshold."""
        return self._stop_rule.update(train_loss)
