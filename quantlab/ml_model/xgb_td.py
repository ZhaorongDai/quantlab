"""XGBoost regression head with pytabkit's tuned defaults.

``XGBTDRegressor`` is a ``TabkitRegressor`` that wraps
``pytabkit.XGB_TD_Regressor``, XGBoost with the tuned defaults from
Holzmüller et al., "Better by Default" (NeurIPS 2024). It is the pytabkit
counterpart of ``XGBoostRegressor`` (``xgb.py``), which drives ``xgboost``
directly. This head takes pytabkit's parameter set and preprocessing. The
other one takes xgboost's raw parameters, stops early on a concordance
correlation (CCC) loss and logs feature-importance charts. Both predict
future returns as ``[num_times, num_symbols, num_labels]`` from the
flattened factor panel, an ``xarray.Dataset`` indexed by ``timestamp`` and
``symbol``.

pytabkit's XGBoost estimator is single-output, so one estimator is fitted
per label.
"""

import numpy as np
from loguru import logger
from pytabkit import XGB_TD_Regressor

from quantlab.ml_model.tabkit import TabkitRegressor, active_callbacks
from quantlab.ml_model.xgb import _WandbEvalCallback, record_feature_importance


class _XGBTDEstimator(XGB_TD_Regressor):
    """``XGB_TD_Regressor`` whose early-stopping patience is settable.

    pytabkit fixes ``early_stopping_rounds`` at 300 inside its tuned
    defaults and offers no constructor argument for it, so the head sets
    ``early_stopping_rounds`` on the instance after construction. ``None``
    removes the key: every ``n_estimators`` round is then trained, and the
    validation set (when given) still selects the best round. The attribute
    is not a scikit-learn parameter, so ``get_params`` does not report it;
    it is pickled with the estimator.

    Examples
    --------
    >>> est = _XGBTDEstimator(n_estimators=100)
    >>> est.early_stopping_rounds = 10
    >>> est.get_config()["early_stopping_rounds"]
    10
    """

    early_stopping_rounds: int | None = None

    def _get_default_params(self) -> dict:
        """Return pytabkit's tuned defaults with the patience applied."""
        params = dict(super()._get_default_params())
        if self.early_stopping_rounds is None:
            params.pop("early_stopping_rounds", None)
        else:
            params["early_stopping_rounds"] = int(self.early_stopping_rounds)
        return params


class XGBTDRegressor(TabkitRegressor):
    """Predict future returns with pytabkit's tuned-default XGBoost.

    ``self.model`` is a list with one fitted estimator per label, in
    ``get_label_names()`` order, because pytabkit's XGBoost estimator does
    not support multi-output regression. Every estimator is trained on the
    same rows (rows with any non-finite label are dropped for all labels).
    Headline metrics are computed on the primary label, index 0. Row
    conversion, NaN handling and the hyperparameter record are inherited
    from ``TabkitRegressor``.

    Hyperparameters are the constructor arguments of ``XGB_TD_Regressor``
    (``n_estimators``, ``max_depth``, ``lr``, ``subsample``, ``n_threads``,
    ...). pytabkit's tuned defaults fill in whatever is not given:
    1000 rounds, depth 9, learning rate 0.05, subsample 0.7.

    With a validation segment that has at least one finite-label row,
    pytabkit always selects the round with the lowest validation error, so
    the ``.joblib`` checkpoint predicts with the best round, and that round
    count is written to the run summary as ``best_n_estimators`` (primary
    label) and ``best_n_estimators/{label}`` (every label). With
    ``config.early_stopping`` set, training additionally halts after
    ``config.early_stopping_patience`` rounds without improvement; the value
    is recorded as ``early_stopping_rounds`` in
    ``resolved_hyperparameters`` (``None`` when off). Without a usable
    validation segment a warning is logged, all rounds are trained and the
    estimator is pinned to predict with all of them (see
    ``_pin_all_rounds``). No per-round curve is logged.

    ``train_cv`` (rolling walk-forward cross-validation) is inherited. Each
    fold selects its own best round and writes its own ``.joblib``. With
    ``parallel=True`` the folds run on threads while pytabkit uses every
    physical core by default, so set ``n_threads`` in the hyperparameters to
    roughly ``os.cpu_count() // njobs`` to avoid oversubscribing the CPU.

    Every W&B run gets, per label, the validation curve of each boosting
    round (``val-rmse``, or ``val-rmse/<label>`` with several labels, at
    ``step=round``), the selected round (``best_n_estimators``), the rounds
    trained (``num_boosted_rounds``) and the per-factor importance with its
    charts, exactly as ``XGBoostRegressor`` records them. The callback is
    injected into pytabkit's inner ``xgboost.train`` call (see
    ``quantlab.ml_model.tabkit.active_callbacks``).

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
    ...     early_stopping=True, early_stopping_patience=50,
    ...     hyperparameters={"n_estimators": 500, "n_threads": 4},
    ... )
    >>> model = XGBTDRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'XGBTDRegressor_total.joblib'
    >>> model.predict(np.zeros((5, 2, 3), dtype="float32")).shape
    (5, 2, 1)
    >>> model.train_cv(train_periods=500, gap_periods=5, parallel=True, njobs=4)
    """

    DEFAULT_PARAMS: dict = {
        "val_fraction": 0.0,
        "verbosity": 0,
    }

    def _early_stopping_rounds(self) -> int | None:
        """Return the patience to inject, or None when early stopping is off."""
        if not self.config.early_stopping:
            return None
        return int(self.config.early_stopping_patience)

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ) -> list[_XGBTDEstimator]:
        """Resolve the parameters and return one unfitted estimator per label.

        ``num_features`` is unused; pytabkit infers it from the arrays passed
        to ``fit``.

        Raises
        ------
        TypeError
            From pytabkit, if a hyperparameter key is not an
            ``XGB_TD_Regressor`` constructor argument.
        """
        params = self._resolve_params(hyperparameters)
        rounds = self._early_stopping_rounds()
        estimators = []
        for _ in range(num_labels):
            estimator = _XGBTDEstimator(**params)
            estimator.early_stopping_rounds = rounds
            estimators.append(estimator)
        return estimators

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the constructor arguments plus the injected patience."""
        resolved = super()._resolved_hyperparameters()
        if resolved is None:
            return None
        return {**resolved, "early_stopping_rounds": self._early_stopping_rounds()}

    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Fit one estimator per label and record the best rounds in the summary.

        Label ``i`` is fitted on column ``i`` of the label rows. Without a
        usable validation segment each estimator is pinned to predict with
        all of its rounds.

        Raises
        ------
        ValueError
            If the training segment has no row with finite labels.
        """
        x_rows, y_rows = self._training_rows(train_x, train_y)
        val_rows = self._validation_rows(val_x, val_y)
        names = list(self.get_label_names())

        for i, estimator in enumerate(self.model):
            suffix = self._key_suffix(names, i)
            self._last_log_step = None
            with active_callbacks(xgb_callbacks=self._round_callbacks(suffix)):
                if val_rows is None:
                    estimator.fit(x_rows, y_rows[:, i])
                    self._pin_all_rounds(estimator)
                else:
                    estimator.fit(
                        x_rows, y_rows[:, i], X_val=val_rows[0], y_val=val_rows[1][:, i]
                    )
            if self._wandb_recorder is not None:
                self._record_booster(estimator, names[i], suffix)

        if self._wandb_recorder is not None and val_rows is not None:
            best = self._best_n_estimators()
            summary = {
                f"best_n_estimators/{name}": rounds
                for name, rounds in zip(names, best)
                if rounds is not None
            }
            if best and best[0] is not None:
                summary["best_n_estimators"] = best[0]
            if summary:
                self._wandb_recorder.summary.update(summary)

    @staticmethod
    def _key_suffix(names: list[str], i: int) -> str:
        """Return the per-label key suffix: empty for one label, ``/<label>`` otherwise."""
        return "" if len(names) == 1 else f"/{names[i]}"

    def _round_callbacks(self, suffix: str) -> list:
        """Return the per-round W&B callback to inject, or none without a run.

        pytabkit evaluates the validation set every round (``val-rmse``);
        the callback logs it at ``step=round`` so the curve of each label
        sits beside the plain ``XGBoostRegressor``'s.
        """
        if self._wandb_recorder is None:
            return []
        return [_WandbEvalCallback(self, suffix=suffix)]

    def _record_booster(self, estimator: _XGBTDEstimator, label: str, suffix: str) -> None:
        """Log the fitted Booster's rounds and feature importance for ``label``.

        The Booster is pytabkit's ``sub_split_interfaces[0].model``. Its
        trained round count goes to the summary as
        ``num_boosted_rounds{suffix}``; the importance goes through
        ``record_feature_importance`` with the same suffix. Reporting only:
        a missing Booster is a warning, never an error.
        """
        try:
            booster = estimator.alg_interface_.sub_split_interfaces[0].model
        except (AttributeError, IndexError) as exc:
            logger.warning(
                f"{self.class_name}: no fitted Booster found for label {label!r}; "
                f"rounds and feature importance were not recorded: {exc}"
            )
            return
        self._wandb_recorder.summary.update(
            {f"num_boosted_rounds{suffix}": int(booster.num_boosted_rounds())}
        )
        record_feature_importance(
            booster,
            [str(name) for name in self.get_factor_names()],
            self._wandb_recorder,
            getattr(self, "_last_log_step", None),
            self.class_name,
            suffix=suffix,
        )

    @staticmethod
    def _pin_all_rounds(estimator: _XGBTDEstimator) -> None:
        """Make an estimator fitted without a validation set predict with every round.

        pytabkit 1.7.3 leaves the inner split interface's ``fit_params`` at
        ``None`` after such a fit, and its ``predict`` then looks up
        ``n_estimators`` in an empty dict and raises ``KeyError``. Setting
        the round count to the Booster's trained rounds restores prediction
        without changing what was trained.
        """
        for sub in estimator.alg_interface_.sub_split_interfaces:
            if sub.fit_params is None:
                sub.fit_params = [
                    {"n_estimators": int(sub.model.num_boosted_rounds())}
                ]

    def _best_n_estimators(self) -> list[int | None]:
        """Return each estimator's selected round count, None where unreported."""
        out = []
        for estimator in self.model:
            fit_params = getattr(estimator, "fit_params_", None) or {}
            rounds = fit_params.get("n_estimators")
            out.append(None if rounds is None else int(rounds))
        return out

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[T, S, L]`` from a preprocessed ``[T, S, F]`` array.

        Missing feature values are imputed with ``0.0`` first, as in
        training, and each estimator fills one label column.
        """
        n_times, n_symbols, n_features = x.shape
        rows = self._impute_features(x.reshape(n_times * n_symbols, n_features))
        columns = [
            np.asarray(estimator.predict(rows), dtype=np.float32).reshape(-1)
            for estimator in self.model
        ]
        return np.stack(columns, axis=-1).reshape(n_times, n_symbols, -1)
