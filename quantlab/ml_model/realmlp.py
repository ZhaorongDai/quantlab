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

import numpy as np
from pytabkit import RealMLP_TD_Regressor

from quantlab.ml_model.tabkit import TabkitRegressor


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
    and all ``n_epochs`` are trained. pytabkit exposes no per-epoch callback,
    so no per-epoch curve is logged.

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
