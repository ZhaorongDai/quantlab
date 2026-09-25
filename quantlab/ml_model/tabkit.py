"""Shared base for model heads backed by pytabkit estimators.

``TabkitRegressor`` holds what every pytabkit head needs and pytabkit does
not do itself: the ``[T, S, *]`` panel to row conversion, NaN handling
(pytabkit refuses NaN in numerical columns at fit and at predict), the
validation-row logic with the pipeline's warnings, hyperparameter merging
and the ``resolved_hyperparameters`` record. Concrete heads
(``realmlp.py``, ``xgb_td.py``) build the estimator(s), fit them and
predict.

The module is named ``tabkit.py`` rather than ``pytabkit.py`` so it does
not shadow the ``pytabkit`` package inside this package.
"""

from abc import abstractmethod

import numpy as np
from loguru import logger

from quantlab.base.config import MLConfig
from quantlab.base.model import MLModel


class TabkitRegressor(MLModel):
    """``MLModel`` base for pytabkit regression heads.

    Training flattens the ``[T, S, F]`` features and ``[T, S, L]`` labels to
    rows and drops every row with a non-finite label. Non-finite feature
    values are imputed with ``0.0`` (see ``_impute_features``), because
    pytabkit refuses NaN in numerical columns; factors are normally z-scored
    before they reach a model, which makes ``0.0`` the column mean. Label
    NaN is kept by ``_preprocess`` so the row drop and ``MLModel._loss``
    still see it.

    Hyperparameters come from ``config.hyperparameters`` and are the
    constructor arguments of the pytabkit estimator. The merge order is the
    head's ``DEFAULT_PARAMS``, then ``random_state`` from
    ``config.random_seed``, then the keys ``_early_stopping_params`` derives
    from ``config.early_stopping``, then the user's dict, which wins and is
    never modified. An unknown key raises ``TypeError`` from pytabkit at
    ``_init_model``. The merged dict is recorded under
    ``resolved_hyperparameters`` in the checkpoint's ``config.json`` and in
    the run config.

    Every head pins ``val_fraction=0.0`` in its ``DEFAULT_PARAMS``: pytabkit
    would otherwise carve a second validation set out of the training rows,
    and the pipeline's trailing ``val_size`` split is meant to be the only
    one. When that split has finite-label rows it is passed to ``fit`` as
    ``X_val``/``y_val`` and pytabkit selects the best iteration on it;
    whether training also halts early is the head's early-stopping mapping.

    Subclasses implement ``_init_model``, ``_fit_model`` and ``_forward``;
    ``_early_stopping_params`` is an optional hook.

    Examples
    --------
    A head is used like any other ``MLModel``; see ``RealMLPRegressor``
    and ``XGBTDRegressor`` for the estimator-specific parts::

        >>> issubclass(RealMLPRegressor, TabkitRegressor)
        True
        >>> RealMLPRegressor.DEFAULT_PARAMS["val_fraction"]
        0.0
    """

    #: Estimator constructor arguments applied before the config seed, the
    #: early-stopping keys and the user's hyperparameters.
    DEFAULT_PARAMS: dict = {}

    def __init__(self, config: MLConfig):
        """Store the config; parameters are resolved later by ``_init_model``."""
        super().__init__(config)
        self._params: dict | None = None

    def _early_stopping_params(self) -> dict:
        """Return estimator constructor keys implied by ``config.early_stopping``.

        The default returns ``{}``. A head whose estimator takes early
        stopping as constructor arguments overrides this.
        """
        return {}

    def _resolve_params(self, hyperparameters: dict) -> dict:
        """Merge defaults, seed, early-stopping keys and user overrides.

        The result is stored on ``self._params`` and returned; the user's
        dict is copied, never modified.
        """
        self._params = {
            **self.DEFAULT_PARAMS,
            "random_state": self.config.random_seed,
            **self._early_stopping_params(),
            **dict(hyperparameters),
        }
        return dict(self._params)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the constructor arguments handed to the estimator."""
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

    def _training_rows(
        self, train_x: np.ndarray, train_y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the training rows.

        Raises
        ------
        ValueError
            If the training segment has no row with finite labels.
        """
        x_rows, y_rows = self._to_rows(train_x, train_y)
        if x_rows.shape[0] == 0:
            raise ValueError(
                "The training segment has no rows with finite labels."
            )
        return x_rows, y_rows

    def _validation_rows(
        self, val_x: np.ndarray | None, val_y: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Return the validation rows, or None when there is no usable segment.

        A validation segment without a finite-label row counts as absent.
        Both absent cases log a warning when ``config.early_stopping`` is
        set, because early stopping is then skipped and every iteration is
        trained, as in ``XGBoostRegressor``.
        """
        rows = None
        if val_x is not None:
            val_x_rows, val_y_rows = self._to_rows(val_x, val_y)
            if val_x_rows.shape[0] > 0:
                rows = (val_x_rows, val_y_rows)
            else:
                logger.warning(
                    f"{self.class_name}: the validation segment has no rows "
                    "with finite labels; training without a validation set."
                )
        if rows is None and self.config.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"iterations."
            )
        return rows

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Resolve the parameters and return the unfitted estimator(s)."""

    @abstractmethod
    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Fit ``self.model`` on the rows and record the run's summary."""

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for a preprocessed ``[T, S, F]`` input."""
