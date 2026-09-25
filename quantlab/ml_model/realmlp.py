"""RealMLP regression head backed by pytabkit.

``RealMLPRegressor`` is an ``MLModel`` that wraps
``pytabkit.RealMLP_TD_Regressor``, the RealMLP network with the tuned
defaults from Holzmüller et al., "Better by Default" (NeurIPS 2024). It
trains on the flattened ``(num_times * num_symbols, num_features)`` rows of
the factor panel and predicts future returns as ``[num_times, num_symbols,
num_labels]``. Early stopping is pytabkit's own, driven by the validation
segment the pipeline holds out, and the stopping epoch is recorded to
Weights and Biases after training.

pytabkit is itself a torch library, so on macOS the same OpenMP guard that
applies to torch and xgboost in one process applies here (see
``docs/model.md``).
"""

import numpy as np
from loguru import logger
from pytabkit import RealMLP_TD_Regressor

from quantlab.base.config import MLConfig
from quantlab.base.model import MLModel


class RealMLPRegressor(MLModel):
    """Predict future returns with a pytabkit RealMLP (tuned defaults).

    Training flattens the ``[T, S, F]`` features and ``[T, S, L]`` labels to
    rows, drops every row with a non-finite label, and fits one
    ``RealMLP_TD_Regressor`` on the rest. Each label is one output of a
    multi-output regression; headline metrics are computed on the primary
    label, index 0.

    pytabkit refuses NaN in numerical columns, at fit and at predict, so
    non-finite feature values are imputed with ``0.0`` in ``_to_rows`` and
    ``_forward``. Factors are normally z-scored before they reach a model,
    which makes ``0.0`` the column mean; a head that needs another imputation
    overrides ``_impute_features``. Label NaN is kept by ``_preprocess`` so
    the row drop and ``MLModel._loss`` still see it.

    Hyperparameters come from ``config.hyperparameters`` and are the
    constructor arguments of ``RealMLP_TD_Regressor`` (``n_epochs``,
    ``hidden_sizes``, ``lr``, ``device``, ``n_threads``, ...). Every key
    overrides the matching entry of ``DEFAULT_PARAMS``; ``random_state``
    defaults to ``config.random_seed``. An unknown key raises ``TypeError``
    from pytabkit at ``_init_model``. The user's dict is never modified, and
    the parameters actually used are recorded under
    ``resolved_hyperparameters`` in the checkpoint's ``config.json`` and in
    the run config.

    ``DEFAULT_PARAMS`` pins ``val_fraction=0.0``: pytabkit would otherwise
    carve a second validation set out of the training rows, and the
    pipeline's trailing ``val_size`` split is meant to be the only one.

    With ``config.early_stopping`` set and a validation segment that has at
    least one finite-label row, pytabkit's early stopping watches the
    validation loss with ``early_stopping_additive_patience =
    config.early_stopping_patience`` and a multiplicative patience of
    ``1.0``, so patience counts epochs without improvement. The fitted model
    is already the best epoch, so the ``.joblib`` checkpoint is the best
    model, and the stopping epoch is written to the run summary as
    ``stop_epoch``. Without a usable validation segment a warning is logged
    and all ``n_epochs`` are trained. pytabkit exposes no per-epoch callback,
    so no per-epoch curve is logged.

    ``train_cv`` is inherited: each fold does its own early stopping and
    writes its own ``.joblib``. With ``parallel=True`` the folds run on
    threads while pytabkit uses every physical core by default, so set
    ``n_threads`` in the hyperparameters to roughly
    ``os.cpu_count() // njobs``.

    Example:
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

    def __init__(self, config: MLConfig):
        """Store the config; parameters are resolved later by ``_init_model``."""
        super().__init__(config)
        self._params: dict | None = None

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

        The merge order is ``DEFAULT_PARAMS``, then ``random_state`` from the
        config seed, then the early-stopping keys implied by
        ``config.early_stopping``, then the user's hyperparameters, which win.

        Raises:
            TypeError: From pytabkit, if a hyperparameter key is not a
                ``RealMLP_TD_Regressor`` constructor argument.
        """
        self._params = {
            **self.DEFAULT_PARAMS,
            "random_state": self.config.random_seed,
            **self._early_stopping_params(),
            **dict(hyperparameters),
        }
        return RealMLP_TD_Regressor(**self._params)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the constructor arguments handed to ``RealMLP_TD_Regressor``."""
        if self._params is None:
            return None
        return dict(self._params)

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Return a float32 copy with infinities replaced by NaN."""
        out = np.array(data, dtype=np.float32, copy=True)
        out[np.isinf(out)] = np.nan
        return out

    @staticmethod
    def _impute_features(x: np.ndarray) -> np.ndarray:
        """Return ``x`` with every non-finite value replaced by ``0.0``."""
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @classmethod
    def _to_rows(cls, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flatten ``[T, S, F]`` and ``[T, S, L]`` to rows with finite labels.

        Rows whose label has any non-finite value are dropped; the surviving
        feature rows are then imputed, because pytabkit refuses NaN.
        """
        n_times, n_symbols, n_features = x.shape
        x_rows = x.reshape(n_times * n_symbols, n_features)
        y_rows = y.reshape(n_times * n_symbols, y.shape[-1])
        keep = np.isfinite(y_rows).all(axis=1)
        return cls._impute_features(x_rows[keep]), y_rows[keep]

    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Fit the estimator and record the stopping epoch in the run summary.

        Raises:
            ValueError: If the training segment has no row with finite labels.
        """
        x_rows, y_rows = self._to_rows(train_x, train_y)
        if x_rows.shape[0] == 0:
            raise ValueError(
                "The training segment has no rows with finite labels."
            )

        val_rows = None
        if val_x is not None:
            val_x_rows, val_y_rows = self._to_rows(val_x, val_y)
            if val_x_rows.shape[0] > 0:
                val_rows = (val_x_rows, val_y_rows)
            else:
                logger.warning(
                    f"{self.class_name}: the validation segment has no rows "
                    "with finite labels; training without a validation set."
                )

        if val_rows is None and self.config.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"epochs."
            )

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

        ``fit_params_["stop_epoch"]`` is a dict keyed by the validation
        metric; the first value is taken.
        """
        fit_params = getattr(self.model, "fit_params_", None) or {}
        stop = fit_params.get("stop_epoch")
        if isinstance(stop, dict):
            stop = next(iter(stop.values()), None)
        return None if stop is None else int(stop)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[T, S, L]`` from a preprocessed ``[T, S, F]`` array."""
        n_times, n_symbols, n_features = x.shape
        rows = self._impute_features(x.reshape(n_times * n_symbols, n_features))
        pred = self.model.predict(rows)
        return np.asarray(pred, dtype=np.float32).reshape(n_times, n_symbols, -1)
