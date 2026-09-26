"""RealMLP regression head backed by pytabkit.

``RealMLPRegressor`` is a ``TabkitRegressor`` that wraps
``pytabkit.RealMLP_TD_Regressor``, the RealMLP network with the tuned
defaults from Holzmüller et al., "Better by Default" (NeurIPS 2024). It
trains on the flattened ``(num_times * num_symbols, num_features)`` rows of
the factor panel and predicts future returns as ``[num_times, num_symbols,
num_labels]``. A *panel* is an ``xarray.Dataset`` indexed by ``timestamp``
and ``symbol``. Early stopping is pytabkit's own, driven by the validation
segment the pipeline holds out, and the stopping epoch is recorded to
Weights and Biases (W&B, the experiment tracker) after training.

pytabkit is itself a torch library. On macOS, torch and xgboost bundle
clashing OpenMP runtimes, so a process that mixes them must set
``OMP_NUM_THREADS=1`` before importing either; the same applies here.
"""

import math

import numpy as np
from loguru import logger
from pytabkit import RealMLP_TD_Regressor
from pytabkit.models.training.lightning_callbacks import Callback

from quantlab.ml_model.tabkit import TabkitRegressor, active_callbacks


class _WandbEpochCallback(Callback):
    """Log every epoch's training loss and validation error to a W&B run.

    A Lightning callback that pytabkit's ``TabNNModule`` receives through
    ``quantlab.ml_model.tabkit.active_callbacks``. The training loss is the
    mean of the per-batch losses ``training_step`` returns. The validation
    error is recomputed from the module's own validation predictions the
    way its ``on_validation_epoch_end`` computes it, for every name in
    ``val_metric_names``. Keys are ``train-loss`` and ``val-<metric>``
    with ``step`` equal to the epoch (1-based), so they never collide with
    the underscored final ``train_*``/``val_*`` values the base class writes
    to the summary. At the end of the fit the best validation error and the
    number of epochs trained go to the summary as ``best_val_<metric>`` and
    ``epochs_trained``.

    The callback reads ``head._wandb_recorder`` on each call, so a
    deep-copied cross-validation fold logs to its own run. Any failure in
    it is reported once as a warning and never interrupts training.

    Parameters
    ----------
    head : RealMLPRegressor
        The model whose recorder receives the rows.
    """

    def __init__(self, head: "RealMLPRegressor"):
        """Keep a reference to the head and reset the running totals."""
        super().__init__()
        self._head = head
        self._loss_sum = 0.0
        self._loss_count = 0
        self._best: dict[str, float] = {}
        self._epochs = 0
        self._failed = False

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Reset the running training loss."""
        self._loss_sum = 0.0
        self._loss_count = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        """Accumulate the batch loss ``training_step`` returned."""
        try:
            value = outputs["loss"] if isinstance(outputs, dict) else outputs
            self._loss_sum += float(value)
            self._loss_count += 1
        except Exception as exc:  # reporting only
            self._warn_once(exc)

    def on_validation_end(self, trainer, pl_module) -> None:
        """Log the epoch's mean training loss and validation errors."""
        recorder = self._head._wandb_recorder
        if recorder is None or trainer.sanity_checking:
            return
        try:
            epoch = int(pl_module.progress.epoch)  # already advanced past this epoch
            row = {"epoch": epoch}
            if self._loss_count:
                row["train-loss"] = self._loss_sum / self._loss_count
            for name, error in self._validation_errors(pl_module).items():
                row[f"val-{name}"] = error
                best = self._best.get(name, math.inf)
                if error < best:
                    self._best[name] = error
            self._epochs = epoch
            recorder.log(row, step=epoch)
        except Exception as exc:  # reporting only
            self._warn_once(exc)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Write the best validation errors and the epochs trained to the summary."""
        recorder = self._head._wandb_recorder
        if recorder is None:
            return
        summary = {f"best_val_{name}": value for name, value in self._best.items()}
        if self._epochs:
            summary["epochs_trained"] = self._epochs
        if summary:
            recorder.summary.update(summary)

    @staticmethod
    def _validation_errors(pl_module) -> dict[str, float]:
        """Return ``{metric: mean validation error}`` from the module's last predictions."""
        import torch
        from pytabkit.models.training.lightning_modules import postprocess_multiquantile
        from pytabkit.models.training.metrics import Metrics

        preds = getattr(pl_module, "val_preds", None)
        if not preds:
            return {}
        y_pred = pl_module._postprocess_ens_pred(torch.cat(preds, dim=-2))
        y_pred = postprocess_multiquantile(y_pred, **pl_module.config)
        y = pl_module.val_dl.val_y[:: pl_module.config.get("n_ens", 1)]
        errors = {}
        for name in pl_module.val_metric_names:
            values = [
                float(Metrics.apply(y_pred[i, :, :], y[i, :, :], name))
                for i in range(y_pred.shape[0])
            ]
            errors[name] = float(np.mean(values))
        return errors

    def _warn_once(self, exc: Exception) -> None:
        if not self._failed:
            self._failed = True
            logger.warning(
                f"{self._head.class_name}: per-epoch W&B logging failed and is "
                f"off for the rest of this fit (training continues): {exc}"
            )


class RealMLPRegressor(TabkitRegressor):
    """Predict future returns with a pytabkit RealMLP (tuned defaults).

    One ``RealMLP_TD_Regressor`` is fitted on the flattened rows. Each label
    is one output of a multi-output regression, and headline metrics are
    computed on the primary label, index 0. Row conversion, NaN handling and
    the hyperparameter record are inherited from ``TabkitRegressor``.

    Hyperparameters are the constructor arguments of
    ``RealMLP_TD_Regressor`` (``n_epochs``, ``hidden_sizes``, ``lr``,
    ``device``, ``n_threads``, ...).

    With ``config.early_stopping`` set and a validation segment that has at
    least one finite-label row, pytabkit's early stopping watches the
    validation loss with ``early_stopping_additive_patience =
    config.early_stopping_patience`` and a multiplicative patience of
    ``1.0``, so patience counts epochs without improvement. The fitted model
    is already rolled back to the best epoch, so the ``.joblib`` checkpoint
    is the best model. The stopping epoch is written to the run summary as
    ``stop_epoch``. Without a usable validation segment a warning is logged
    and all ``n_epochs`` are trained. Every W&B run also gets a per-epoch
    curve: the mean training loss (``train-loss``) and the validation
    error pytabkit stops on (``val-rmse``), at ``step=epoch``, plus
    ``best_val_rmse`` and ``epochs_trained`` in the summary, through a
    Lightning callback injected into pytabkit's trainer (see
    ``quantlab.ml_model.tabkit.active_callbacks``).

    ``train_cv`` (rolling walk-forward cross-validation) is inherited. Each
    fold does its own early stopping and writes its own ``.joblib``. With
    ``parallel=True`` the folds run on threads while pytabkit uses every
    physical core by default, so set ``n_threads`` in the hyperparameters to
    roughly ``os.cpu_count() // njobs`` to avoid oversubscribing the CPU.

    Parameters
    ----------
    config : MLConfig
        Factors, labels, date ranges, early-stopping settings and
        hyperparameters. See ``MLConfig``.

    Examples
    --------
    >>> config = MLConfig(
    ...     factors=[alpha],            # factor objects
    ...     labels=[fwd_return],        # label objects
    ...     model_save_dir="checkpoints",
    ...     factor_data_strategy="read",
    ...     label_data_strategy="read",
    ...     train_start="2024-01-01", train_end="2024-02-09",
    ...     test_start="2024-02-10", test_end="2024-02-29",
    ...     early_stopping=True, early_stopping_patience=5,
    ...     hyperparameters={"n_epochs": 50, "n_threads": 4},
    ... )
    >>> model = RealMLPRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'RealMLPRegressor_total.joblib'
    >>> model.predict(np.zeros((5, 2, 3), dtype="float32")).shape
    (5, 2, 1)
    >>> model.train_cv(train_periods=500, gap_periods=5, parallel=True, njobs=4)
    """

    DEFAULT_PARAMS: dict = {
        "device": "cpu",
        "val_fraction": 0.0,
        "verbosity": 0,
    }

    def _early_stopping_params(self) -> dict:
        """Return the pytabkit early-stopping keys implied by the config."""
        if not self.config.early_stopping:
            return {}
        return {
            "use_early_stopping": True,
            "early_stopping_additive_patience": int(
                self.config.early_stopping_patience
            ),
            "early_stopping_multiplicative_patience": 1.0,
        }

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> RealMLP_TD_Regressor:
        """Resolve the parameters and return an unfitted estimator.

        ``num_features`` and ``num_labels`` are unused; pytabkit infers both
        from the arrays passed to ``fit``.

        Raises
        ------
        TypeError
            From pytabkit, if a hyperparameter key is not a
            ``RealMLP_TD_Regressor`` constructor argument.
        """
        return RealMLP_TD_Regressor(**self._resolve_params(hyperparameters))

    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Fit the estimator and record the stopping epoch in the run summary.

        The validation rows are passed to pytabkit only when the segment has
        at least one row with finite labels.

        Raises
        ------
        ValueError
            If the training segment has no row with finite labels.
        """
        x_rows, y_rows = self._training_rows(train_x, train_y)
        val_rows = self._validation_rows(val_x, val_y)

        callbacks = [] if self._wandb_recorder is None else [_WandbEpochCallback(self)]
        with active_callbacks(lightning_callbacks=callbacks):
            if val_rows is None:
                self.model.fit(x_rows, y_rows)
            else:
                self.model.fit(x_rows, y_rows, X_val=val_rows[0], y_val=val_rows[1])

        if self._wandb_recorder is not None and val_rows is not None:
            stop_epoch = self._stop_epoch()
            if stop_epoch is not None:
                self._wandb_recorder.summary.update({"stop_epoch": stop_epoch})

    def _stop_epoch(self) -> int | None:
        """Return the epoch pytabkit stopped at, or None if it did not report one.

        pytabkit stores ``fit_params_["stop_epoch"]`` as a dict keyed by the
        validation metric; the first value is taken.
        """
        fit_params = getattr(self.model, "fit_params_", None) or {}
        stop = fit_params.get("stop_epoch")
        if isinstance(stop, dict):
            stop = next(iter(stop.values()), None)
        return None if stop is None else int(stop)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[T, S, L]`` from a preprocessed ``[T, S, F]`` array.

        ``T`` is bars, ``S`` symbols, ``F`` features and ``L`` labels. Missing
        feature values are imputed with ``0.0`` first, as in training.
        """
        n_times, n_symbols, n_features = x.shape
        rows = self._impute_features(x.reshape(n_times * n_symbols, n_features))
        pred = self.model.predict(rows)
        return np.asarray(pred, dtype=np.float32).reshape(n_times, n_symbols, -1)
