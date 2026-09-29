"""Library variant of the model layer: ``LibraryModel``, its rows and its checkpoint backend.

``LibraryModel`` is the base class of heads whose library trains itself (XGBoost, pytabkit):
there is no epoch loop, the head's ``_fit_model`` receives flat ``Rows`` (one per
``(bar, symbol)`` cell with a valid training target) and uses the library's native early
stopping. Checkpoints are ``.joblib`` files written through ``MlBackend``. Shipped heads live in
``quantlab/model/predefined``.
"""

from abc import abstractmethod
from pathlib import Path
from typing import NamedTuple, Self

import joblib
import numpy as np
import torch
from loguru import logger

from quantlab.base.backend import ModelBackend
from quantlab.base.model import LIBRARY_RESERVED_HYPERPARAMETERS, BaseModel
from quantlab.model.torch_data import TrainingPanel
from quantlab.model.training_target import TrainingTargetMixin
from quantlab.utils.timer import Timer


class Rows(NamedTuple):
    """The training or validation rows of one fit.

    Rows are ordered by bar, then by symbol.

    Attributes
    ----------
    x : np.ndarray
        Features after ``_transform_feature``, ``[n, F]`` float32. NaN is kept
        for the library's own missing-value handling.
    y : np.ndarray
        Training target, ``[n, L]`` float32, finite everywhere: the output of
        the head's ``_transform_target`` on the row's bar.
    y_raw : np.ndarray
        Raw labels, ``[n, L]`` float32.
    where : tuple[np.ndarray, np.ndarray]
        Timestamp and symbol index of each row in the collected panel,
        two ``[n]`` int64 arrays.

    Examples
    --------
    >>> rows = Rows(
    ...     x=np.zeros((2, 3), dtype=np.float32), y=np.zeros((2, 1), dtype=np.float32),
    ...     y_raw=np.zeros((2, 1), dtype=np.float32),
    ...     where=(np.array([0, 0]), np.array([0, 1])),
    ... )
    >>> len(rows.x), rows.where[1].tolist()
    (2, [0, 1])
    """

    x: np.ndarray
    y: np.ndarray
    y_raw: np.ndarray
    where: tuple[np.ndarray, np.ndarray]


class MlBackend(ModelBackend):
    """Persist one model object with joblib.

    ``write``, ``read`` and ``to_internal`` all return ``self`` so calls can
    be chained. ``write`` creates missing parent directories. The constructor
    takes no arguments; the backend is empty until ``read`` or
    ``to_internal`` gives it a model.

    Examples
    --------
    >>> MlBackend().to_internal({"coef": 2.5}).write("ckpt/model.joblib")
    MlBackend()
    >>> MlBackend().read("ckpt/model.joblib").get_model()
    {'coef': 2.5}
    """

    def get_model(self):
        """Return the held model object.

        Raises
        ------
        AttributeError
            If nothing has been loaded with ``read`` or ``to_internal`` yet.

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5}).get_model()
        {'coef': 2.5}
        """
        return self.model

    def write(self, path: str, **kwargs) -> Self:
        """Dump the held model to ``path`` with ``joblib.dump`` and return ``self``.

        Missing parent directories of ``path`` are created.

        Parameters
        ----------
        path : str
            Destination file, conventionally ending in ``.joblib``.
        **kwargs
            Forwarded to ``joblib.dump``, for example ``compress=3``.

        Returns
        -------
        MlBackend
            This backend, for chaining.

        Examples
        --------
        >>> backend = MlBackend().to_internal({"coef": 2.5})
        >>> backend.write("ckpt/model.joblib", compress=3)
        MlBackend()
        """
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        joblib.dump(self.model, path, **kwargs)
        return self

    def read(self, path: str, **kwargs) -> Self:
        """Load the model at ``path`` with ``joblib.load`` and return ``self``.

        Parameters
        ----------
        path : str
            A file previously written by ``write`` or ``joblib.dump``.
        **kwargs
            Forwarded to ``joblib.load``.

        Returns
        -------
        MlBackend
            This backend, now holding the loaded model.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> MlBackend().read("ckpt/model.joblib").get_model()
        {'coef': 2.5}
        """
        self.model = joblib.load(path, **kwargs)
        return self

    def to_internal(self, model) -> Self:
        """Adopt an in-memory model object and return ``self``.

        Parameters
        ----------
        model : object
            Any picklable object, typically a fitted estimator.

        Returns
        -------
        MlBackend
            This backend, now holding ``model``.

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5})
        MlBackend()
        """
        self.model = model
        return self


class LibraryModel(TrainingTargetMixin, BaseModel):
    """Numpy variant for tree models and other non-torch libraries.

    There is no epoch loop and no copy-based rollback. Training, early
    stopping and the choice of the best model are left to the library's own
    mechanism inside ``_fit_model``: boosting libraries decide early stopping
    per round with cached validation scores, and rolling back to the best
    round is a matter of keeping the first ``k`` trees, both of which an
    outer epoch loop would only make coarser and slower.

    The base builds the rows. The training target is computed once per fit,
    one bar at a time, by ``_transform_target`` (``training=True`` on the
    training bars only), exactly as for a torch head. Only cells with a valid
    training target become rows; NaN features are kept, so the library's own
    missing-value handling decides what they mean. Each row carries its cell
    in ``where`` (see ``Rows``).

    A head implements three hooks: ``_init_model``, ``_fit_model`` and
    ``_forward``. ``_transform_feature`` (inf to NaN), ``_transform_target``
    (the raw label), ``_loss`` (MSE), ``_resolved_hyperparameters`` and the
    inherited ``_compute_metrics`` have defaults that may be overridden.
    ``{split}_loss`` is ``_loss`` on the training target per bar, averaged
    over bars; the other metrics score the raw first label. Checkpoints are
    ``.joblib`` files written through ``MlBackend``; they are pickles, so only
    load files you trust.

    Examples
    --------
    A minimal head that predicts the first feature for every label::

        >>> class FirstFeatureHead(LibraryModel):
        ...     def _init_model(self, num_features, num_labels, hyperparameters):
        ...         return {"num_labels": num_labels}
        ...     def _fit_model(self, train_rows, val_rows):
        ...         pass
        ...     def _forward(self, x):
        ...         return np.repeat(x[:, :1], self.model["num_labels"], axis=-1)
        >>> head = FirstFeatureHead(ModelConfig(
        ...     factors=[factor], labels=[label], model_save_dir="checkpoints",
        ...     factor_data_strategy="read", label_data_strategy="read",
        ...     train_start="2024-01-01", train_end="2024-01-30",
        ...     test_start="2024-01-31", test_end="2024-02-09",
        ... ))
        >>> head.collect().train().suffix
        '.joblib'
    """

    checkpoint_suffix = ".joblib"
    reserved_hyperparameters = LIBRARY_RESERVED_HYPERPARAMETERS

    @property
    def early_stopping(self) -> bool:
        """``hyperparameters["early_stopping"]``, False when unset.

        Whether the head turns on its library's native early stopping.

        Examples
        --------
        >>> head.early_stopping
        False
        """
        return bool(self.config.hyperparameters.get("early_stopping", False))

    @property
    def early_stopping_patience(self) -> int:
        """``hyperparameters["early_stopping_patience"]``, 5 when unset.

        Rounds (or the library's own unit) without improvement before the
        library's early stopping triggers.

        Examples
        --------
        >>> head.early_stopping_patience
        5
        """
        return int(self.config.hyperparameters.get("early_stopping_patience", 5))

    @abstractmethod
    def _init_model(
        self, num_features: int, num_labels: int, hyperparameters: dict
    ):
        """Prepare the model for the given shape; the result becomes ``self.model``.

        Tree libraries often build the real model only inside ``_fit_model``,
        in which case this may just resolve the hyperparameters and return
        None. ``load()`` does not call it: the checkpoint holds the whole
        model. ``hyperparameters`` holds the reserved keys too; pass it
        through ``self.head_hyperparameters`` before handing it to a library.
        """

    @abstractmethod
    def _fit_model(self, train_rows: Rows, val_rows: Rows | None) -> None:
        """Fit the model on the training rows.

        ``val_rows`` is None when there is no validation segment
        (``val_size == 0``) or it has no cell with a valid training target.
        Early stopping and rollback to the best model are the hook's job,
        using the library's native mechanism and honouring
        ``early_stopping`` and ``early_stopping_patience``. On return
        ``self.model`` must be the model to save.
        """

    @abstractmethod
    def _forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``[n, L]`` predictions for ``[n, F]`` rows from ``_transform_feature``."""

    def _transform_feature(self, x: np.ndarray) -> np.ndarray:
        """Turn raw ``[n, F]`` float32 feature rows into the library's input.

        The result must keep the shape. The default returns a copy with
        infinities replaced by NaN, which tree libraries read as missing; a
        library that refuses NaN overrides this to impute. Applied to the
        training and validation rows and at prediction alike; never modify
        ``x`` in place.
        """
        return np.where(np.isinf(x), np.float32(np.nan), x)

    def _resolved_hyperparameters(self) -> dict | None:
        """Return the hyperparameters actually in effect, or None to record nothing.

        A head that merges user overrides into library defaults in
        ``_init_model`` overrides this to expose the merged result. When not
        None, ``_fit`` adds it to the wandb run config and ``get_config`` adds
        it to ``config.json`` as ``resolved_hyperparameters``, so a run stays
        reproducible after defaults change. It is a record, not an input:
        ``config.hyperparameters`` is left as the user wrote it, and the
        config loader drops the key when rebuilding.
        """
        return None

    def get_config(self) -> dict:
        """Return ``BaseModel.get_config()`` plus any resolved hyperparameters.

        Examples
        --------
        >>> "resolved_hyperparameters" in head.get_config()
        False
        """
        cfg = super().get_config()
        resolved = self._resolved_hyperparameters()
        if resolved is not None:
            cfg["resolved_hyperparameters"] = dict(resolved)
        return cfg

    def _loss(self, target: np.ndarray, pred: np.ndarray) -> float:
        """Return the MSE over every label of one bar's ``[n, L]`` rows.

        ``target`` is the bar's training target, finite everywhere. Returns
        NaN when there are no rows.
        """
        target = np.asarray(target, dtype=np.float64)
        pred = np.asarray(pred, dtype=np.float64)
        if target.size == 0:
            return float("nan")
        diff = pred - target
        return float(np.sum(diff * diff) / diff.size)

    def _features(self, x: np.ndarray) -> np.ndarray:
        """Return ``_transform_feature(x)`` after checking it kept the shape."""
        out = np.asarray(self._transform_feature(x))
        if out.shape != x.shape:
            raise ValueError(
                f"{self.class_name}._transform_feature must keep the shape "
                f"{x.shape}, got {out.shape}"
            )
        return out

    def _forward_rows(self, x: np.ndarray) -> np.ndarray:
        """Run ``_forward`` on raw ``[n, F]`` rows and check it returns ``[n, L]``."""
        if len(x) == 0:
            return np.empty((0, self.num_labels), dtype=np.float32)
        pred = np.asarray(self._forward(self._features(x)))
        if pred.shape != (len(x), self.num_labels):
            raise ValueError(
                f"{self.class_name}._forward must return [n, L] = "
                f"{[len(x), self.num_labels]} predictions, got {list(pred.shape)}"
            )
        return pred

    def _rows(self, panel: TrainingPanel, bars) -> Rows:
        """Return the rows of ``bars``: every cell with a valid training target."""
        cells = panel.mask & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        return Rows(
            x=self._features(panel.x[t, s].numpy()),
            y=panel.target[t, s].numpy(),
            y_raw=panel.y_raw[t, s].numpy(),
            where=(t.numpy(), s.numpy()),
        )

    @staticmethod
    def _bar_mask(panel: TrainingPanel, bars) -> torch.Tensor:
        """Return ``[T]`` booleans, True on ``bars``."""
        chosen = torch.zeros(panel.present.shape[0], dtype=torch.bool)
        chosen[torch.as_tensor(np.asarray(bars, dtype=np.int64))] = True
        return chosen

    def _evaluate(self, split: str, panel: TrainingPanel, bars) -> dict[str, float]:
        """Evaluate one split and write the prefixed metrics to the wandb summary.

        Every present cell of ``bars`` is predicted. ``{split}_loss`` is
        ``_loss`` on each bar's training target, averaged over the bars
        that have one, so every bar weighs the same; the other keys are
        ``_compute_metrics`` on the raw labels. They go to the run summary
        (final values, no step), so they do not interfere with per-round
        ``log(step=...)`` curves.
        """
        num_times, num_symbols = panel.present.shape
        pred = np.full((num_times, num_symbols, self.num_labels), np.nan, dtype=np.float64)
        cells = panel.present & self._bar_mask(panel, bars)[:, None]
        t, s = torch.nonzero(cells, as_tuple=True)
        pred[t.numpy(), s.numpy()] = self._forward_rows(panel.x[t, s].numpy())

        target = panel.target.numpy()
        mask = panel.mask.numpy()
        losses = [
            self._loss(target[bar, mask[bar]], pred[bar, mask[bar]])
            for bar in bars
            if mask[bar].any()
        ]
        metrics = {f"{split}_loss": float(np.mean(losses)) if losses else float("nan")}
        y_raw = panel.y_raw.numpy()
        for key, value in self._compute_metrics(
            y_raw[bars], pred[bars], split, panel.timestamps[bars]
        ).items():
            metrics[f"{split}_{key}"] = value
        if self._wandb_recorder is not None:
            self._wandb_recorder.summary.update(metrics)
        return metrics

    def _fit(self, checkpoint: Path) -> dict:
        """Build the rows, fit once with ``_fit_model``, evaluate, save and finish the run.

        The validation segment is the trailing ``val_size`` share of the
        training window, and the purge of ``_fit_segments`` drops the last L
        bars before validation and before test, as in the torch variant.
        The training target is computed once, before the fit:
        ``_transform_target`` sees ``training=True`` on the training bars and
        ``training=False`` on the validation and test bars. Empty splits skip
        evaluation: no validation segment means no ``val_*`` metrics, and an
        empty test segment no ``test_*`` metrics.

        Returns
        -------
        dict
            The ``train_*``, ``val_*`` and ``test_*`` metrics, with the
            values ``_evaluate`` wrote to the wandb summary.

        Raises
        ------
        ValueError
            If any of the four ``train_*`` / ``test_*`` dates is
            unset, ``val_size`` or the purge leaves no timestamps to fit
            on, or no training cell has a valid training target.
        """
        config = self.config
        if not all(
            (config.train_start, config.train_end, config.test_start, config.test_end)
        ):
            raise ValueError(
                "Training and testing start and end dates must be specified."
            )

        data = self.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        segments = self._fit_segments(data)
        stamps = data.timestamp.values
        train_bars, val_bars, test_bars = [
            np.searchsorted(stamps, part.timestamp.values) for part in segments
        ]
        panel = self._training_panel(data)
        self._fill_target(panel, train_bars, training=True)
        self._fill_target(panel, val_bars, training=False)
        self._fill_target(panel, test_bars, training=False)

        train_rows = self._rows(panel, train_bars)
        if len(train_rows.x) == 0:
            raise ValueError(
                f"{self.class_name}: the training segment has no cell with a "
                f"valid training target."
            )
        val_rows = self._rows(panel, val_bars) if len(val_bars) else None
        if val_rows is not None and len(val_rows.x) == 0:
            logger.warning(
                f"{self.class_name}: the validation segment has no cell with a "
                f"valid training target; training without a validation set."
            )
            val_rows = None

        self.model = self._init_model(
            num_features=self.num_factors,
            num_labels=self.num_labels,
            hyperparameters=config.hyperparameters,
        )
        resolved = self._resolved_hyperparameters()
        if resolved is not None and self._wandb_recorder is not None:
            # The run was opened in `_init_wandb`, before the hyperparameters
            # were resolved; record them now.
            self._wandb_recorder.config.update(
                {"resolved_hyperparameters": dict(resolved)},
                allow_val_change=True,
            )

        with Timer(f"{self.class_name}: fit_model"):
            self._fit_model(train_rows, val_rows)

        with Timer(f"{self.class_name}: evaluate"):
            metrics = self._evaluate("train", panel, train_bars)
            for split, bars in (("val", val_bars), ("test", test_bars)):
                if len(bars):
                    metrics.update(self._evaluate(split, panel, bars))

        self._save_model(checkpoint)
        if self._wandb_recorder is not None:
            self._wandb_recorder.finish()
        return metrics

    def _predict(self, data: torch.Tensor | np.ndarray) -> np.ndarray:
        """Return ``[T, S, L]`` predictions for a ``[T, S, F]`` input.

        Every cell with a finite feature is a row for ``_forward``; the
        others are NaN. Tensors are converted to numpy. A floating input
        keeps its dtype, and the predictions are float64.

        Raises
        ------
        TypeError
            If ``data`` is neither a tensor nor an array.
        ValueError
            If ``_transform_feature`` changes the shape of the rows or
            ``_forward`` does not return ``[n, L]``.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Unsupported data type: {type(data)}")
        x = data if np.issubdtype(data.dtype, np.floating) else data.astype(np.float64)
        present = np.isfinite(x).any(axis=-1)
        out = np.full(x.shape[:-1] + (self.num_labels,), np.nan, dtype=np.float64)
        out[present] = self._forward_rows(x[present])
        return out

    def _predict_panel_array(
        self, x: np.ndarray, timestamps: np.ndarray, symbols: np.ndarray
    ) -> np.ndarray:
        """Return ``predict(x)`` as an array."""
        return np.asarray(self.predict(x))

    def _write_checkpoint(self, path: Path) -> None:
        """Serialize ``self.model`` to ``path`` through ``MlBackend``."""
        MlBackend().to_internal(self.model).write(str(path))

    def _read_checkpoint(self, path: Path) -> None:
        """Load the whole model from ``path`` through ``MlBackend``.

        ``_init_model`` is not called: the file holds the complete model, and
        rebuilding an empty one first would require collecting data to know
        the feature count.
        """
        self.model = MlBackend().read(str(path)).get_model()
