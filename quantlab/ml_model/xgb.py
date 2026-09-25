"""XGBoost regression head for the tree-model layer.

``XGBoostRegressor`` is an ``MLModel`` (the numpy-based model base class in
``quantlab.base.model``). It trains an XGBoost ``Booster`` with ``xgb.train``
on the flattened ``(num_times * num_symbols, num_features)`` rows of the
factor panel, where a *panel* is an ``xarray.Dataset`` indexed by
``timestamp`` and ``symbol``. It predicts future returns as
``[num_times, num_symbols, num_labels]``.

Early stopping uses xgboost's native callback on a loss derived from the
*concordance correlation coefficient* (CCC), which measures how closely
predictions match the labels in both correlation and scale
(``pooled_ccc_loss``). Per-factor feature importance is recorded to Weights
and Biases (W&B, the experiment tracker) after training.

The module is named ``xgb.py`` rather than ``xgboost.py`` so it does not
shadow the ``xgboost`` package inside this package.
"""

import numpy as np
import wandb
import xgboost as xgb
from loguru import logger

from quantlab.base.config import MLConfig
from quantlab.base.model import MLModel

#: scikit-learn style aliases mapped to the native ``xgb.train`` parameter
#: names. Aliases are rewritten on the user's dict before it is merged with
#: the defaults, because xgboost handles them inconsistently: ``learning_rate``
#: next to ``eta`` wins or loses by dict order, ``n_estimators`` is ignored
#: with a warning, and ``random_state`` is silently ignored when ``seed`` is
#: present.
_PARAM_ALIASES: dict[str, str] = {
    "n_estimators": "num_boost_round",
    "learning_rate": "eta",
    "random_state": "seed",
    "n_jobs": "nthread",
    "reg_alpha": "alpha",
    "reg_lambda": "lambda",
}

#: ``Booster.get_score`` importance types written to the run summary after
#: training: split count, mean gain per split and total gain.
_IMPORTANCE_TYPES: tuple[str, ...] = ("weight", "gain", "total_gain")

#: Number of factors drawn in the importance bar chart. The accompanying
#: table still lists every factor; only the chart is truncated.
_IMPORTANCE_CHART_TOP_N = 30

#: Key prefix of the chart objects. It is distinct from the ``importance_``
#: prefix of the per-factor summary scalars so the two never collide.
_IMPORTANCE_CHART_PREFIX = "feature_importance"


def pooled_ccc_loss(y_true, y_pred) -> float:
    """Return ``1 - ccc``, the pooled concordance correlation loss.

    ``ccc = 2 * cov / (var_pred + var_true + (mu_pred - mu_true)^2)`` with
    population moments (``ddof=0``). Both inputs are flattened and scored as
    one pool, without any per-timestamp grouping, so the loss also rewards
    predicting the market-wide move of each day and penalises a correctly
    shrunk prediction whose variance is below the label's.

    A CCC of 1 means perfect agreement, 0 no agreement and -1 perfect
    inverse agreement, so the loss ranges from 0 (best) to 2 (worst).

    Only positions where both inputs are finite are used. When no such
    position remains, or the denominator is exactly zero, the loss ``1.0``
    (no agreement) is returned rather than NaN, so that early stopping can
    still compare rounds.

    Parameters
    ----------
    y_true : array_like
        Observed values, any shape.
    y_pred : array_like
        Predicted values, the same number of elements as ``y_true``.

    Returns
    -------
    float
        ``1 - ccc``.

    Raises
    ------
    ValueError
        If the two inputs have different lengths after flattening.

    Examples
    --------
    >>> pooled_ccc_loss([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    0.0
    >>> pooled_ccc_loss([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    0.014084507042253502
    >>> pooled_ccc_loss([1, 2, 3], [3, 2, 1])
    2.0
    """
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(
            f"pooled_ccc_loss expects two vectors of the same length, got "
            f"{true.shape} vs {pred.shape}"
        )

    mask = np.isfinite(true) & np.isfinite(pred)
    if int(mask.sum()) < 1:
        return 1.0
    true = true[mask]
    pred = pred[mask]

    mu_true = np.mean(true)
    mu_pred = np.mean(pred)
    var_true = np.var(true)
    var_pred = np.var(pred)
    cov = np.mean((pred - mu_pred) * (true - mu_true))

    denominator = var_pred + var_true + (mu_pred - mu_true) ** 2
    if denominator == 0.0:
        return 1.0
    return float(1.0 - 2.0 * cov / denominator)


def ccc_loss_metric(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    """Score a Booster's predictions for ``xgb.train(custom_metric=...)``.

    The score is ``pooled_ccc_loss`` of the predictions against the labels.
    Only the primary label (column 0) is scored. With multiple labels
    ``dtrain.get_label()`` returns a ``(n_rows, n_labels)`` array and the
    prediction has the same layout; both are reshaped to ``(n_rows, -1)``
    before column 0 is taken, which is a no-op for a single label.

    Parameters
    ----------
    predt : np.ndarray
        The Booster's predictions for ``dtrain``.
    dtrain : xgb.DMatrix
        The evaluated ``DMatrix``, whose labels are read back.

    Returns
    -------
    tuple[str, float]
        ``("ccc_loss", value)`` as xgboost expects from a custom metric.

    Raises
    ------
    ValueError
        If the label and prediction element counts differ.

    Examples
    --------
    >>> dm = xgb.DMatrix(np.zeros((3, 2)), label=np.array([1.0, 2.0, 3.0]))
    >>> ccc_loss_metric(np.array([1.1, 1.9, 3.2]), dm)
    ('ccc_loss', 0.014084507042253502)
    """
    label = np.asarray(dtrain.get_label(), dtype=np.float64)
    pred = np.asarray(predt, dtype=np.float64)
    if label.size != pred.size:
        raise ValueError(
            f"ccc_loss: the DMatrix carries {label.size} label values but the "
            f"prediction has {pred.size}; they must match."
        )
    n_rows = dtrain.num_row()
    return "ccc_loss", pooled_ccc_loss(
        label.reshape(n_rows, -1)[:, 0], pred.reshape(n_rows, -1)[:, 0]
    )


class _WandbEvalCallback(xgb.callback.TrainingCallback):
    """Log every boosting round's eval results to the head's current W&B run.

    The callback holds a reference to the head and reads
    ``head._wandb_recorder`` on each round, so a deep-copied cross-validation
    fold logs to its own run.

    Keys use xgboost's hyphenated form
    (``train-rmse``, ``val-ccc_loss``) with ``step`` equal to the round
    index, which distinguishes these curves from the underscored final
    values ``MLModel._evaluate`` writes to the summary. The last logged round
    is stored on the head as ``_last_log_step`` so the feature-importance
    charts can be logged on the same step.

    Parameters
    ----------
    head : XGBoostRegressor
        The model whose recorder receives the per-round values.
    """

    def __init__(self, head: "XGBoostRegressor"):
        """Keep a reference to the head whose recorder receives the rows."""
        super().__init__()
        self._head = head

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        """Log the latest value of every metric and return ``False`` to continue.

        Called by xgboost after each boosting round.

        Parameters
        ----------
        model : xgb.Booster
            The Booster being trained (unused).
        epoch : int
            Index of the round just finished, used as the W&B step.
        evals_log : dict
            xgboost's history, ``{data_name: {metric: [value per round]}}``.

        Returns
        -------
        bool
            Always ``False``; returning ``True`` would stop training.

        Examples
        --------
        >>> booster = xgb.train(
        ...     params, dtrain, evals=[(dtrain, "train"), (dval, "val")],
        ...     callbacks=[_WandbEvalCallback(head)],
        ... )
        """
        recorder = self._head._wandb_recorder
        if recorder is not None:
            row = {
                f"{data_name}-{metric}": float(values[-1])
                for data_name, metrics in evals_log.items()
                for metric, values in metrics.items()
            }
            recorder.log(row, step=epoch)
            self._head._last_log_step = epoch
        return False


class XGBoostRegressor(MLModel):
    """Predict future returns with an XGBoost Booster.

    Training flattens the ``[T, S, F]`` features and ``[T, S, L]`` labels to
    rows, drops every row with a non-finite label, converts infinite feature
    values to NaN (which xgboost treats as missing) and calls ``xgb.train``.
    Each label is one output of a multi-output regression; headline metrics
    are computed on the primary label, index 0.

    Hyperparameters come from ``config.hyperparameters``. ``num_boost_round``
    (default 1000) is taken out separately; every other key overrides the
    matching entry of ``DEFAULT_PARAMS``, and ``seed`` defaults to
    ``config.random_seed``. scikit-learn style aliases such as
    ``learning_rate`` or ``n_estimators`` are rewritten to the native names
    first; giving both an alias and its native name raises ``ValueError``.
    The user's dict is never modified, and the parameters actually used are
    recorded under ``resolved_hyperparameters`` in the checkpoint's
    ``config.json`` and in the run config.

    With ``config.early_stopping`` set and a validation segment that has at
    least one finite-label row, ``xgb.callback.EarlyStopping`` watches the
    validation ``ccc_loss`` (see ``pooled_ccc_loss``); the built-in
    ``eval_metric`` (RMSE by default) is logged as a curve only. Patience
    counts boosting rounds. ``save_best=True`` means the returned Booster is
    already truncated to ``best_iteration + 1`` trees, so the ``.joblib``
    checkpoint is the best model, and ``best_iteration`` and ``best_score``
    (a CCC loss) are written to the run summary. Without a usable validation
    segment a warning is logged and all rounds are trained.

    After training, per-factor importance (``weight``, ``gain`` and
    ``total_gain``) is written to the run summary as
    ``importance_{type}/{factor}`` and logged as a sorted table plus a
    top-30 bar chart. This is best effort: a Booster that cannot report an
    importance type only logs a warning, and the checkpoint is never lost
    because of it.

    ``train_cv`` (rolling walk-forward cross-validation) is inherited. Each
    fold does its own native early stopping and writes its own ``.joblib``.
    With ``parallel=True`` the folds run on threads while xgboost itself uses
    every core, so set ``nthread`` in the hyperparameters to roughly
    ``os.cpu_count() // njobs`` to avoid oversubscribing the CPU. The value
    is passed through unchanged.

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
    ...     hyperparameters={"num_boost_round": 20, "max_depth": 3},
    ... )
    >>> model = XGBoostRegressor(config)
    >>> checkpoint = model.collect().train()
    >>> checkpoint.name
    'XGBoostRegressor_total.joblib'
    >>> model.predict(np.zeros((5, 2, 3), dtype="float32")).shape
    (5, 2, 1)
    >>> model.train_cv(train_periods=500, gap_periods=5, parallel=True, njobs=4)
    """

    DEFAULT_PARAMS: dict = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eta": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "device": "cpu",
        "eval_metric": "rmse",
    }
    DEFAULT_NUM_BOOST_ROUND = 1000

    def __init__(self, config: MLConfig):
        """Initialize the head; see the class docstring for parameters.

        Training parameters are resolved later, by ``_init_model``.
        """
        super().__init__(config)
        self._params: dict | None = None
        self._num_boost_round: int | None = None
        # Round index of the last per-round log, reused as the step of the
        # feature-importance charts.
        self._last_log_step: int | None = None

    @staticmethod
    def _normalize_aliases(hyperparameters: dict) -> dict:
        """Return a copy of ``hyperparameters`` with aliases renamed to native keys.

        Raises
        ------
        ValueError
            If an alias and its native key are both present.
        """
        user = dict(hyperparameters)
        for alias, canonical in _PARAM_ALIASES.items():
            if alias not in user:
                continue
            if canonical in user:
                raise ValueError(
                    f"hyperparameters set both {alias!r} and {canonical!r}, which "
                    f"are the same XGBoost parameter; keep only one of them."
                )
            user[canonical] = user.pop(alias)
        return user

    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Resolve the training parameters and return ``None``.

        The Booster itself is built by ``xgb.train`` inside ``_fit_model``.
        Aliases are normalised first, then ``num_boost_round`` is split off,
        then the remaining keys override ``DEFAULT_PARAMS`` and the seed.

        Raises
        ------
        ValueError
            If ``num_boost_round`` is below 1.
        """
        user = self._normalize_aliases(hyperparameters)
        num_boost_round = int(
            user.pop("num_boost_round", self.DEFAULT_NUM_BOOST_ROUND)
        )
        if num_boost_round < 1:
            raise ValueError(
                f"num_boost_round must be >= 1, got {num_boost_round}"
            )
        self._num_boost_round = num_boost_round
        self._params = {
            **self.DEFAULT_PARAMS,
            "seed": self.config.random_seed,
            **user,
        }
        return None

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the parameters handed to ``xgb.train`` plus ``num_boost_round``.

        Returns None before ``_init_model`` has run.
        """
        if self._params is None:
            return None
        return {**self._params, "num_boost_round": self._num_boost_round}

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Return a float32 copy with infinities replaced by NaN.

        xgboost treats NaN as a missing value, so no imputation is needed.
        """
        out = np.array(data, dtype=np.float32, copy=True)
        out[np.isinf(out)] = np.nan
        return out

    @staticmethod
    def _to_rows(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Flatten ``[T, S, F]`` and ``[T, S, L]`` to rows with finite labels.

        ``T`` is bars, ``S`` symbols, ``F`` features and ``L`` labels. Rows
        whose label has any non-finite value are dropped; NaN features stay.
        """
        n_times, n_symbols, n_features = x.shape
        x_rows = x.reshape(n_times * n_symbols, n_features)
        y_rows = y.reshape(n_times * n_symbols, y.shape[-1])
        keep = np.isfinite(y_rows).all(axis=1)
        return x_rows[keep], y_rows[keep]

    def _fit_model(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        val_x: np.ndarray | None,
        val_y: np.ndarray | None,
    ) -> None:
        """Train the Booster with ``xgb.train`` and record the run's summary.

        The validation rows are used only when the segment has at least one
        row with finite labels. With early stopping, the best iteration and
        its score are written to the W&B summary, followed by the feature
        importance.

        Raises
        ------
        ValueError
            If the training segment has no row with finite labels.
        """
        # A deep-copied cross-validation fold would otherwise inherit the
        # last step of a previously trained head.
        self._last_log_step = None
        x_rows, y_rows = self._to_rows(train_x, train_y)
        if x_rows.shape[0] == 0:
            raise ValueError(
                "The training segment has no rows with finite labels."
            )
        dtrain = xgb.DMatrix(x_rows, label=y_rows)
        evals = [(dtrain, "train")]

        dval = None
        if val_x is not None:
            val_x_rows, val_y_rows = self._to_rows(val_x, val_y)
            if val_x_rows.shape[0] > 0:
                dval = xgb.DMatrix(val_x_rows, label=val_y_rows)
                evals.append((dval, "val"))
            else:
                logger.warning(
                    f"{self.class_name}: the validation segment has no rows "
                    "with finite labels; training without a validation set."
                )

        # The logging callback must precede EarlyStopping: xgboost short-
        # circuits its callback list, so a callback placed after EarlyStopping
        # misses the round that triggered the stop.
        callbacks: list[xgb.callback.TrainingCallback] = [
            _WandbEvalCallback(self)
        ]
        use_early_stopping = bool(self.config.early_stopping) and dval is not None
        if use_early_stopping:
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.config.early_stopping_patience,
                    data_name="val",
                    save_best=True,
                )
            )
        elif self.config.early_stopping:
            logger.warning(
                f"{self.class_name}: early_stopping=True but there is no usable "
                f"validation segment; early stopping skipped, training all "
                f"{self._num_boost_round} rounds."
            )

        self.model = xgb.train(
            self._params,
            dtrain,
            num_boost_round=self._num_boost_round,
            evals=evals,
            custom_metric=ccc_loss_metric,
            callbacks=callbacks,
            verbose_eval=False,
        )

        if use_early_stopping and self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(
                {
                    "best_iteration": int(self.model.best_iteration),
                    "best_score": float(self.model.best_score),
                }
            )

        if self._wandb_recorder is not None:
            self._record_feature_importance()

    def _record_feature_importance(self) -> None:
        """Write per-factor importance to the run summary and log the charts.

        ``Booster.get_score`` keys features as ``f{i}`` by column index, and
        columns follow ``get_factor_names()``, so ``f{i}`` maps to the
        ``i``-th factor. Factors that were never split on get ``0.0``. For
        every type in ``_IMPORTANCE_TYPES`` the scalars go to the summary as
        ``importance_{type}/{factor}``, and one ``log`` call at
        ``_last_log_step`` carries a full table sorted by importance plus a
        bar chart of the top ``_IMPORTANCE_CHART_TOP_N`` factors.

        This is reporting only, so a failure here must never lose the
        checkpoint. ``booster="gblinear"`` has no split importance and is
        skipped with an info message. A type that raises ``XGBoostError`` or
        returns non-scalar scores is skipped with a warning, and a failure
        while building or logging the charts is also only a warning.

        Raises
        ------
        ValueError
            If a score key is not ``f<index>`` for one of the factors, which
            means the Booster was not trained on these columns.
        """
        booster_type = str((self._params or {}).get("booster", "gbtree"))
        if booster_type == "gblinear":
            logger.info(
                f"{self.class_name}: booster='gblinear' has no split feature "
                f"importance; importance recording skipped."
            )
            return

        names = [str(name) for name in self.get_factor_names()]
        charts: dict = {}
        for importance_type in _IMPORTANCE_TYPES:
            try:
                scores = self.model.get_score(  # type: ignore[union-attr]
                    importance_type=importance_type
                )
            except xgb.core.XGBoostError as exc:
                logger.warning(
                    f"{self.class_name}: feature importance {importance_type!r} "
                    f"is unavailable for this Booster and was skipped: {exc}"
                )
                continue
            values = {name: 0.0 for name in names}
            non_scalar_key = None
            for key, score in scores.items():
                digits = key[1:] if key.startswith("f") else ""
                index = int(digits) if digits.isdigit() else -1
                if not 0 <= index < len(names):
                    raise ValueError(
                        f"{self.class_name}: Booster.get_score returned feature "
                        f"key {key!r}, which is not f<index> for one of the "
                        f"{len(names)} factors {names}."
                    )
                if not np.isscalar(score):
                    non_scalar_key = key
                    break
                values[names[index]] = float(score)
            if non_scalar_key is not None:
                logger.warning(
                    f"{self.class_name}: feature importance {importance_type!r} "
                    f"returned a non-scalar score for {non_scalar_key!r} "
                    f"(one value per output); skipped."
                )
                continue
            self._wandb_recorder.summary.update(
                {
                    f"importance_{importance_type}/{name}": value
                    for name, value in values.items()
                }
            )

            try:
                # Stable sort: ties keep factor order, so never-split factors
                # at 0.0 end up last in factor order.
                ordered = sorted(
                    values.items(), key=lambda item: item[1], reverse=True
                )
                full_table = wandb.Table(
                    columns=["factor", "importance"],
                    data=[[name, value] for name, value in ordered],
                )
                top = ordered[:_IMPORTANCE_CHART_TOP_N]
                top_chart = wandb.plot.bar(
                    wandb.Table(
                        columns=["factor", "importance"],
                        data=[[name, value] for name, value in top],
                    ),
                    "factor",
                    "importance",
                    title=(
                        f"feature importance ({importance_type}, "
                        f"top {len(top)} of {len(ordered)})"
                    ),
                )
            except Exception as exc:
                logger.warning(
                    f"{self.class_name}: building the feature importance charts "
                    f"for {importance_type!r} failed; they were skipped "
                    f"(the summary entries are unaffected): {exc}"
                )
                continue
            # Chart and table enter the payload together or not at all.
            charts[f"{_IMPORTANCE_CHART_PREFIX}/{importance_type}"] = top_chart
            charts[f"{_IMPORTANCE_CHART_PREFIX}_table/{importance_type}"] = (
                full_table
            )

        if charts:
            try:
                self._wandb_recorder.log(charts, step=self._last_log_step)
            except Exception as exc:
                logger.warning(
                    f"{self.class_name}: logging the feature importance charts "
                    f"failed; they were skipped (the summary entries and the "
                    f"checkpoint are unaffected): {exc}"
                )

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Predict ``[T, S, L]`` from a preprocessed ``[T, S, F]`` array.

        ``inplace_predict``, which skips building a ``DMatrix``, is tried
        first. Boosters that do not support it
        (``gblinear``) fall back to ``predict`` on a ``DMatrix``. NaN is
        treated as missing on both paths.
        """
        n_times, n_symbols, n_features = x.shape
        rows = x.reshape(n_times * n_symbols, n_features)
        try:
            pred = self.model.inplace_predict(rows)  # type: ignore[union-attr]
        except xgb.core.XGBoostError as exc:
            if "Inplace predict is not supported" not in str(exc):
                raise
            pred = self.model.predict(xgb.DMatrix(rows))  # type: ignore[union-attr]
        return np.asarray(pred).reshape(n_times, n_symbols, -1)
