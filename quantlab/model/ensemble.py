"""The base class of every ensemble of models, shipped or user-written.

``BaseEnsemble`` holds what does not depend on where an ensemble's members
come from: the ``Predictor`` members derived from the members (labels,
label delays, training and test windows, with the label configs and the
windows checked identical across members), prediction by combining the
members' predictions, the ensemble directory ``train()`` writes, and the
``ensemble.json`` manifest that ``load`` and ``check_checkpoint`` read.

A concrete ensemble builds its members and implements ``get_config`` /
``from_config``; nothing else is required, and the defaults work for members
of different classes over different factors. Optional hooks:

- ``_combine``: how the members' prediction panels become the ensemble's,
  ``average_predictions`` (per-bar z-score, equal-weight mean) by default.
  Both ``predict_window`` and the ensemble-level evaluation files use it.
- ``collect``, ``_member_predictions``, ``_member_panel_predictions``: how
  members collect their data and features, each member on its own by
  default; an ensemble whose members read the same data shares it.
- ``fingerprint_inputs``, ``training_fingerprint_inputs``: which data it
  reports reading.
- ``_member_seed``: the seed recorded for each member in the manifest.

Shipped ensembles are in ``quantlab/model/predefined`` (``SeedEnsemble``,
``ModelEnsemble``).

The ensemble composes models and inherits none: each member is a complete
model (a ``BaseModel``) with its own checkpoint.

On disk ``train()`` writes::

    {model_save_dir}/{EnsembleClass}_trial_{timestamp}/
        member_0/            one member's usual run directory
        member_1/
        ...
        metrics.json         IC metrics of the combined prediction
        ic_series.csv        their per-bar series, in the single-model layout
        test_predictions.zarr  the combined test-segment prediction
        config.json          what every member shares: dates and labels
        ensemble.json        the manifest, written last

``train_cv`` writes one ensemble directory per walk-forward fold::

    {model_save_dir}/{EnsembleClass}_cv_{timestamp}/
        cv_folds.json        the fold manifest ``run_cv`` replays
        fold_0/              an ensemble directory as above, without metrics.json
        fold_1/
        ...

``ensemble.json`` is ``{"format_version": 1, "members": [{"name": ...,
"checkpoint": ..., "seed": ...}, ...]}``: each member's class as a dotted
path, its checkpoint relative to the manifest's directory, and its seed
(null when the ensemble does not vary seeds). The format names nothing
specific to one kind of ensemble.
"""

import dataclasses
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Self

import wandb
import xarray as xr
from loguru import logger

from quantlab.base.model import BaseModel
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.ensemble import average_predictions
from quantlab.utils.jsonable import to_jsonable
from quantlab.utils.metrics import ic_panel_metrics


def _class_path(obj) -> str:
    """Return the dotted import path of ``obj``'s class."""
    return f"{type(obj).__module__}.{type(obj).__qualname__}"


class BaseEnsemble(ABC):
    """Base class of an ensemble that combines the predictions of several models.

    Subclass it, build the members and implement ``get_config`` and
    ``from_config``; override ``_combine`` to replace the equal-weight
    z-score average (see the module docstring for every hook).

    Parameters
    ----------
    members : sequence
        At least two models (``BaseModel`` instances) whose label configs and
        training and test windows are identical.

    Attributes
    ----------
    members : list
        The member models, in manifest order.

    Raises
    ------
    ValueError
        If there are fewer than two members, or two members differ in
        label configs or in training or test window.

    Examples
    --------
    A concrete ensemble passes its members to this constructor; given a
    ``SeedEnsemble`` of a model over one forward-return label::

        >>> len(ensemble.members), ensemble.label_delays
        (3, (1,))
    """

    #: Name of the manifest ``train()`` writes last and ``load()`` reads.
    MANIFEST_FILENAME = "ensemble.json"
    #: Name of the file holding what every member shares.
    CONFIG_FILENAME = "config.json"
    #: The manifest format this class writes and reads.
    MANIFEST_FORMAT_VERSION = 1
    #: Name of the IC metrics file of the combined prediction.
    METRICS_FILENAME = "metrics.json"
    #: Name of the per-bar IC series file of the combined prediction.
    IC_SERIES_FILENAME = "ic_series.csv"
    #: Name of the zarr store holding the combined test-segment prediction.
    TEST_PREDICTIONS_FILENAME = "test_predictions.zarr"

    def __init__(self, members: Sequence):
        """Initialize the ensemble; see the class docstring for parameters."""
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"{self.class_name} needs at least two members, got {len(members)}"
            )
        self._check_members_agree(members)
        self.members = members
        self._wandb_recorder = None

    def __repr__(self) -> str:
        """Return ``ClassName(members=[...])``."""
        names = ", ".join(type(member).__name__ for member in self.members)
        return f"{self.class_name}(members=[{names}])"

    @property
    def class_name(self) -> str:
        """The ensemble's class name, used in directory names and messages.

        Examples
        --------
        >>> ensemble.class_name
        'SeedEnsemble'
        """
        return type(self).__name__

    @property
    def import_path(self) -> str:
        """The ensemble's class as a dotted import path, the ``name`` of its config.

        Examples
        --------
        >>> ensemble.import_path
        'quantlab.model.predefined.seed_ensemble.SeedEnsemble'
        """
        return _class_path(self)

    def _check_members_agree(self, members: list) -> None:
        """Refuse members that would not combine into one prediction.

        The members must carry the same labels, compared by each label's
        ``get_config()`` (which holds its variables and delay), and share one
        training and test window, since the backtester's in-sample split has
        one training window.

        Raises
        ------
        ValueError
            Naming the first member that differs from member 0 and how.
        """
        first = members[0]
        reference = {
            "labels": [label.get_config() for label in first.labels],
            "train_bounds": tuple(first.train_bounds),
            "test_bounds": tuple(first.test_bounds),
        }
        for k, member in enumerate(members[1:], start=1):
            own = {
                "labels": [label.get_config() for label in member.labels],
                "train_bounds": tuple(member.train_bounds),
                "test_bounds": tuple(member.test_bounds),
            }
            for key, value in reference.items():
                if own[key] != value:
                    raise ValueError(
                        f"{self.class_name}: member {k} ({type(member).__name__}) "
                        f"has {key} {own[key]!r}, member 0 has {value!r}; every "
                        f"member must share them"
                    )

    # ------------------------------------------------------------------
    # Predictor members derived from the members
    # ------------------------------------------------------------------

    @property
    def labels(self) -> list:
        """The label objects every member predicts, in config order.

        Examples
        --------
        >>> [label.get_factor_names() for label in ensemble.labels]
        [('fwd_ret_1',)]
        """
        return list(self.members[0].labels)

    @property
    def train_bounds(self) -> tuple:
        """The training window ``(train_start, train_end)`` every member shares.

        Examples
        --------
        >>> ensemble.train_bounds
        ('2024-01-01', '2024-02-02')
        """
        return tuple(self.members[0].train_bounds)

    @property
    def test_bounds(self) -> tuple:
        """The test window ``(test_start, test_end)`` every member shares.

        Examples
        --------
        >>> ensemble.test_bounds
        ('2024-02-05', '2024-02-09')
        """
        return tuple(self.members[0].test_bounds)

    @property
    def label_delays(self) -> tuple[int, ...]:
        """Each label's ``delay`` in bars, in the order of ``labels``.

        Examples
        --------
        >>> ensemble.label_delays
        (1,)
        """
        return tuple(self.members[0].label_delays)

    @property
    def model_save_dir(self) -> Path:
        """The directory ``train()`` creates the ensemble directory in.

        The first member's ``config.model_save_dir``.

        Examples
        --------
        >>> ensemble.model_save_dir.name
        'models'
        """
        return Path(self.members[0].config.model_save_dir)

    def _member_seed(self, k: int) -> int | None:
        """The seed recorded for member ``k`` in the manifest; None by default."""
        return None

    def collect(self) -> Self:
        """Load every member's features and labels into its data backend.

        Each member collects its own data. An ensemble whose members read the
        same data overrides this to collect once.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Examples
        --------
        >>> ensemble.collect() is ensemble
        True
        """
        for member in self.members:
            member.collect()
        return self

    def _member_predictions(self, start, end) -> list[xr.Dataset]:
        """Return each member's predictions from ``start`` to ``end``.

        Each member requests its own features (``predict_window``). An
        ensemble whose members read the same features overrides this to
        request them once.
        """
        return [member.predict_window(start, end) for member in self.members]

    def _member_panel_predictions(self) -> list[xr.Dataset]:
        """Return each member's prediction over its whole collected panel.

        Each member predicts the panel in its own data backend, so every bar
        is predicted with all the history collected before it. Used to
        evaluate the combined prediction after training.
        """
        return [
            member.predict_panel(
                member.data_backend.get_xarray_dataset(["timestamp", "symbol"])
            )
            for member in self.members
        ]

    def _combine(self, predictions: list[xr.Dataset]) -> xr.Dataset:
        """Combine the members' prediction panels into the ensemble's prediction.

        The default is ``average_predictions``: z-scored over symbols per
        member, variable and bar, then averaged with equal weights, ignoring
        NaN, in z-score units. Override it for another rule, such as fixed
        weights or a rank average; it sees only the predictions, so a rule
        whose parameters are learned in training does not fit here.

        Parameters
        ----------
        predictions : list[xr.Dataset]
            One panel per member, in member order, each on
            ``(timestamp, symbol)`` with one variable per label name. The
            members' coordinates may differ.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> import xarray as xr
        >>> def panel(values):
        ...     return xr.Dataset(
        ...         {"ret": (("timestamp", "symbol"), np.array([values]))},
        ...         coords={"timestamp": pd.date_range("2024-01-01", periods=1),
        ...                 "symbol": ["A", "B", "C"]},
        ...     )
        >>> BaseEnsemble._combine(None, [panel([1.0, 2.0, 3.0]), panel([30.0, 10.0, 20.0])])["ret"].values
        array([[ 0. , -0.5,  0.5]])
        """
        return average_predictions(predictions)

    def predict_window(self, start, end) -> xr.Dataset:
        """Predict every bar from ``start`` to ``end`` by combining the members.

        The members' predictions are combined by ``_combine``, by default
        the equal-weight mean of their per-bar z-scores. Every member must
        be trained or loaded.

        Parameters
        ----------
        start, end : str
            First and last bar to predict, inclusive.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``.

        Examples
        --------
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
        """
        combined = self._combine(self._member_predictions(start, end))
        return combined.sel(timestamp=slice(start, end))

    def fingerprint_inputs(self, start, end) -> list[tuple]:
        """Return the data ``predict_window(start, end)`` reads, for fingerprinting.

        The union of the members' entries, each key prefixed with
        ``member[{k}]:`` so that entries of different members never collide.

        Parameters
        ----------
        start, end : str
            The window passed to ``predict_window``.

        Returns
        -------
        list[tuple]
            ``(key, factor, strategy, first, last)`` entries, member by member.

        Examples
        --------
        >>> [key for key, *_ in BaseEnsemble.fingerprint_inputs(
        ...     ensemble, "2024-02-12", "2024-03-11")][:2]
        ['member[0]:factor[0]:PastReturnFactor', 'member[1]:factor[0]:PastReturnFactor']
        """
        return [
            (f"member[{k}]:{key}", *rest)
            for k, member in enumerate(self.members)
            for key, *rest in member.fingerprint_inputs(start, end)
        ]

    def training_fingerprint_inputs(self) -> list[tuple]:
        """Return the data ``collect()`` reads, for fingerprinting.

        The union of the members' entries, each key prefixed with
        ``member[{k}]:``.

        Returns
        -------
        list[tuple]
            ``(key, item, strategy, first, last)`` entries, member by member.

        Examples
        --------
        >>> [key for key, *_ in BaseEnsemble.training_fingerprint_inputs(ensemble)][:2]
        ['member[0]:train_factor[0]:PastReturnFactor', 'member[0]:train_label[0]:ForwardReturnLabel']
        """
        return [
            (f"member[{k}]:{key}", *rest)
            for k, member in enumerate(self.members)
            for key, *rest in member.training_fingerprint_inputs()
        ]

    # ------------------------------------------------------------------
    # Training and the ensemble directory
    # ------------------------------------------------------------------

    def _new_directory(self, kind: str = "trial") -> Path:
        """Create and return a fresh ``{class}_{kind}_{%Y%m%d_%H%M%S_%f}`` directory.

        The directory is created under ``model_save_dir``; when the name is
        taken, ``_1``, ``_2``, ... are appended. ``train`` passes ``"trial"``
        and ``train_cv`` passes ``"cv"``.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base = f"{self.class_name}_{kind}_{stamp}"
        root = self.model_save_dir.absolute()
        root.mkdir(parents=True, exist_ok=True)
        name, suffix = base, 1
        while True:
            try:
                (root / name).mkdir()
            except FileExistsError:
                name = f"{base}_{suffix}"
                suffix += 1
                continue
            return root / name

    def _shared_config(self) -> dict:
        """Return what every member shares, for the ensemble-level ``config.json``."""
        train_start, train_end = self.train_bounds
        test_start, test_end = self.test_bounds
        return {
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "labels": [label.get_config() for label in self.labels],
        }

    def train(self) -> Path:
        """Train every member into one ensemble directory and write its manifest.

        Every member's hyperparameters are checked first. Then a new
        ``{class}_trial_{timestamp}`` directory is created under
        ``model_save_dir`` and filled by ``_train_into``: member k is
        trained, in order, into ``member_{k}/`` under a wandb run
        ``{MemberClass}_member_{k}`` in a project named after the directory;
        each member writes its usual checkpoint, ``config.json``,
        ``metrics.json``, ``ic_series.csv`` and ``test_predictions.zarr``
        there, and reseeds its generators from its own ``random_seed`` right
        before it trains. The ensemble directory then gets the evaluation
        files of the combined prediction (``metrics.json``,
        ``ic_series.csv``, ``test_predictions.zarr``, see
        ``_write_evaluation_files``), ``config.json`` with the shared
        training and test dates and label configs (it is not a model
        config), and last, atomically, ``ensemble.json``. If a member or the
        ensemble evaluation fails, the error propagates, no manifest is
        written and the files already written stay. Call ``collect()``
        first.

        Returns
        -------
        Path
            Absolute path of the ``ensemble.json`` written.

        Examples
        --------
        >>> manifest = ensemble.collect().train()
        >>> manifest.name, manifest.parent.name.startswith("SeedEnsemble_trial_")
        ('ensemble.json', True)
        >>> sorted(p.name for p in manifest.parent.iterdir())
        ['config.json', 'ensemble.json', 'ic_series.csv', 'member_0', 'member_1', 'member_2', 'metrics.json', 'test_predictions.zarr']
        >>> sorted(json.loads((manifest.parent / "metrics.json").read_text()))
        ['test_ic', 'test_icir', 'test_rank_ic', 'test_rank_icir', 'train_ic', 'train_icir', 'train_rank_ic', 'train_rank_icir']
        """
        for member in self.members:
            member._check_hyperparameters()
        directory = self._new_directory()
        manifest, _ = self._train_into(directory, project_name=directory.name)
        return manifest

    def _train_into(
        self,
        run_dir: Path | str,
        project_name: str,
        *,
        run_tag: str | None = None,
        write_metrics: bool = True,
    ) -> tuple[Path, dict]:
        """Train every member, evaluate the combination and write the manifest into ``run_dir``.

        Member k trains into ``run_dir/member_{k}`` under the wandb run
        ``{MemberClass}_member_{k}`` (``{MemberClass}_{run_tag}_member_{k}``
        with a ``run_tag``) in the wandb project ``project_name``.
        Then come the evaluation files of the combined prediction (see
        ``_write_evaluation_files``; ``metrics.json`` only with
        ``write_metrics``), ``config.json`` and last ``ensemble.json``.
        Hyperparameters are not checked here: ``train`` checks them first,
        and ``train_cv`` once before its folds.

        Parameters
        ----------
        run_dir : Path or str
            The ensemble directory. It is created with its parents when
            missing; its ``member_{k}`` subdirectories must not exist yet.
        project_name : str
            wandb project of the members' runs.
        run_tag : str, optional
            Inserted into every member's wandb run name, so that runs of
            several ensemble directories in one project stay apart;
            ``train_cv`` passes ``fold_{i}``.
        write_metrics : bool, default True
            Write the ensemble's ``metrics.json``; a caller that records the
            metrics elsewhere passes False.

        Returns
        -------
        tuple[Path, dict]
            The absolute ``ensemble.json`` path and the ensemble metrics,
            with NaN where a metric is undefined.
        """
        directory = Path(run_dir).absolute()
        directory.mkdir(parents=True, exist_ok=True)
        entries = []
        tag = "" if run_tag is None else f"_{run_tag}"
        for k, member in enumerate(self.members):
            checkpoint, _ = member._train_into(
                directory / f"member_{k}",
                project_name=project_name,
                experiment_name=f"{member.class_name}{tag}_member_{k}",
            )
            entries.append(
                {
                    "name": _class_path(member),
                    "checkpoint": checkpoint.relative_to(directory).as_posix(),
                    "seed": self._member_seed(k),
                }
            )
        metrics = self._write_evaluation_files(directory, write_metrics=write_metrics)
        write_json_atomically(
            directory / self.CONFIG_FILENAME,
            to_jsonable(self._shared_config()),
            indent=2,
        )
        manifest = directory / self.MANIFEST_FILENAME
        write_json_atomically(
            manifest,
            {"format_version": self.MANIFEST_FORMAT_VERSION, "members": entries},
            indent=2,
        )
        return manifest, metrics

    def train_cv(self, train_periods: int, expanding: bool = False) -> list[dict]:
        """Run a walk-forward cross-validation of the ensemble and return per-fold results.

        The folds are those ``BaseModel.train_cv`` trains for the first
        member: laid out by ``BaseModel._cv_folds`` over the first member's
        collected timestamps between its ``start_date`` and ``end_date``,
        sliding or, with ``expanding=True``, growing from the first fold's
        start, and each training window loses its last L bars, L being the
        largest ``lookahead_bars()`` among the labels. Every member's
        hyperparameters are checked once, before any directory is created.

        A new ``{class}_cv_{timestamp}`` directory is created under
        ``model_save_dir``. For fold i, every member's config gets the
        fold's dates before the purge (the members purge them themselves,
        as a single model's fold does), and ``_train_into`` fills
        ``fold_{i}/`` like ``train()`` fills its directory: member k in
        ``member_{k}/`` under the wandb run ``{MemberClass}_fold_{i}_member_{k}``,
        the evaluation files of the combined prediction, ``config.json`` and
        ``ensemble.json``. The fold's ensemble metrics go into the manifest
        instead of a ``metrics.json``. Folds train one after another. After
        the last fold the members keep its dates, as a model does after its
        own ``train_cv``. The fold means of the ensemble metrics, keyed
        ``cv_mean_{key}``, and ``cv_n_folds`` go to the summary of a
        separate ``{class}_cv_summary`` wandb run in the same project.

        Last, ``cv_folds.json`` is written atomically into the CV directory
        as ``{"format_version": 2, "folds": [...], "cv_mean": {...}}`` (NaN
        and inf as null), the format ``BaseModel.train_cv`` writes, so a
        backtester's ``run_cv`` replays it with the ensemble as its model.
        Call ``collect()`` first.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in the first fold's training segment, and
            in every fold's when sliding. The test segment is one fifth of
            it.
        expanding : bool, default False
            Train every fold from the first fold's start instead of sliding
            a fixed-length window.

        Returns
        -------
        list[dict]
            One dict per fold: ``fold``, the purged ``train_start``,
            ``train_end``, ``test_start`` and ``test_end``, the absolute
            ``checkpoint`` path of the fold's ``ensemble.json`` and the
            fold's ensemble metrics (``{split}_ic``, ``{split}_rank_ic``,
            ``{split}_icir``, ``{split}_rank_icir``).

        Raises
        ------
        ValueError
            If ``train_periods`` is below 5, no timestamp falls inside the
            date range, the purge leaves a fold no training bar, or a
            member's hyperparameters are invalid.

        Examples
        --------
        >>> results = ensemble.collect().train_cv(train_periods=30)
        >>> len(results), sorted(results[0])[:6]
        (8, ['checkpoint', 'fold', 'test_end', 'test_ic', 'test_icir', 'test_rank_ic'])
        >>> Path(results[0]["checkpoint"]).relative_to(ensemble.model_save_dir).parts[1:]
        ('fold_0', 'ensemble.json')
        """
        for member in self.members:
            member._check_hyperparameters()
        first = self.members[0]
        if train_periods < 5:
            raise ValueError(
                f"{self.class_name}: train_cv(train_periods={train_periods}) needs "
                f"at least 5 training bars, since each fold tests on "
                f"train_periods // 5 bars."
            )
        start_date, end_date = first.config.start_date, first.config.end_date
        data = first.data_backend.get_xarray_dataset(["timestamp", "symbol"])
        timestamps = data.sel(timestamp=slice(start_date, end_date)).timestamp.values
        if len(timestamps) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        folds = BaseModel._cv_folds(timestamps, train_periods, expanding=expanding)
        lookahead = first._purge_bars()
        records = [
            BaseModel._purged_fold(timestamps, fold, lookahead) for fold in folds
        ]
        logger.info(
            f"{self.class_name}: {len(folds)} walk-forward folds from {start_date} "
            f"to {end_date} with {train_periods} training periods"
        )

        directory = self._new_directory("cv")
        results = []
        for fold, record in zip(folds, records):
            for member in self.members:
                member.config = dataclasses.replace(
                    member.config,
                    train_start=fold["train_start"],
                    train_end=fold["train_end"],
                    test_start=fold["test_start"],
                    test_end=fold["test_end"],
                )
            manifest, metrics = self._train_into(
                directory / f"fold_{fold['fold']}",
                project_name=directory.name,
                run_tag=f"fold_{fold['fold']}",
                write_metrics=False,
            )
            results.append({**record, "checkpoint": str(manifest), **metrics})

        means = BaseModel._cv_mean_metrics(results)
        if means:
            self._init_wandb(directory.name, f"{self.class_name}_cv_summary")
            if self._wandb_recorder is not None:
                self._wandb_recorder.summary.update(means)
                self._wandb_recorder.finish()

        write_json_atomically(
            directory / BaseModel.CV_FOLDS_FILENAME,
            {
                "format_version": BaseModel.CV_FOLDS_FORMAT_VERSION,
                "folds": to_jsonable(results),
                "cv_mean": to_jsonable(means),
            },
            indent=2,
        )
        return results

    def _init_wandb(self, project_name: str, experiment_name: str) -> None:
        """Open a wandb run with ``get_config()`` as its config."""
        self._wandb_recorder = wandb.init(
            project=project_name, name=experiment_name, config=self.get_config()
        )

    def _write_evaluation_files(
        self, run_dir: Path, *, write_metrics: bool = True
    ) -> dict:
        """Evaluate the combined prediction of the trained members and write its files.

        Every member predicts its whole collected panel
        (``_member_panel_predictions``) and the predictions are combined by
        ``_combine``, the same rule ``predict_window`` uses. The splits are the first member's: its
        collected panel cut by its ``_fit_segments`` into the purged train,
        validation and test segments a single model evaluates, a split being
        skipped when it has no bars (so no ``val_*`` without a validation
        segment). On each split ``ic_panel_metrics`` scores the combined
        prediction of the first label against that label's raw values in
        the first member's panel. No error metric is computed: the default combination
        is in z-score units, not in the target's.

        Written into ``run_dir``:

        - ``metrics.json`` (only with ``write_metrics``): ``{split}_ic``,
          ``{split}_rank_ic``, ``{split}_icir`` and ``{split}_rank_icir``,
          NaN and inf as null.
        - ``ic_series.csv``: the per-bar series behind them, in the layout
          of a single model's file (``BaseModel._write_ic_series``).
        - ``test_predictions.zarr``: the combined prediction on the test
          bars, one variable per label; not written when the test segment
          has no bars.

        Returns
        -------
        dict
            The metrics, with NaN where a metric is undefined.
        """
        first = self.members[0]
        data = first.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        combined = self._combine(self._member_panel_predictions())
        label = first.get_label_names()[0]
        metrics, series = {}, {}
        segments = first._fit_segments(data)
        for split, part in zip(("train", "val", "test"), segments):
            stamps = part.timestamp.values
            if len(stamps) == 0:
                continue
            pred = combined[label].reindex(timestamp=stamps, symbol=data.symbol.values)
            values, per_bar = ic_panel_metrics(
                pred.values,
                data[label].sel(timestamp=stamps).values,
                return_series=True,
            )
            metrics.update({f"{split}_{key}": value for key, value in values.items()})
            series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"])

        if write_metrics:
            write_json_atomically(
                run_dir / self.METRICS_FILENAME, to_jsonable(metrics), indent=2
            )
        first._write_ic_series(run_dir / self.IC_SERIES_FILENAME, series)
        test_stamps = segments[2].timestamp.values
        if len(test_stamps):
            combined.reindex(timestamp=test_stamps).to_zarr(
                run_dir / self.TEST_PREDICTIONS_FILENAME, mode="w"
            )
        return metrics

    # ------------------------------------------------------------------
    # The manifest: check and load
    # ------------------------------------------------------------------

    def _member_checkpoints(self, path: Path | str) -> list[Path]:
        """Read and validate a manifest; return each member's checkpoint path.

        Raises
        ------
        FileNotFoundError
            If the manifest or a member checkpoint does not exist.
        ValueError
            If the manifest is not a JSON object of the known format, lists a
            different number of members, names a different member class or
            records a different seed than this ensemble's member.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"{self.class_name}: manifest {path} not found")
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(
                f"{self.class_name}: {path} is not a JSON {self.MANIFEST_FILENAME} "
                f"manifest ({exc})"
            ) from None
        if not isinstance(saved, dict) or not isinstance(saved.get("members"), list):
            raise ValueError(
                f"{self.class_name}: {path} is not an {self.MANIFEST_FILENAME} "
                f"manifest: expected an object with a 'members' list"
            )
        version = saved.get("format_version")
        if version != self.MANIFEST_FORMAT_VERSION:
            raise ValueError(
                f"{self.class_name}: {path} has format_version {version!r}; this "
                f"version reads format_version {self.MANIFEST_FORMAT_VERSION}"
            )
        entries = saved["members"]
        if len(entries) != len(self.members):
            raise ValueError(
                f"{self.class_name}: {path} lists {len(entries)} members, but "
                f"this ensemble has {len(self.members)}"
            )
        checkpoints = []
        for k, (entry, member) in enumerate(zip(entries, self.members)):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} is not an object"
                )
            for key in ("name", "checkpoint", "seed"):
                if key not in entry:
                    raise ValueError(
                        f"{self.class_name}: {path} member {k} has no {key!r}"
                    )
            if not isinstance(entry["checkpoint"], str):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} 'checkpoint' must be a "
                    f"path string, got {entry['checkpoint']!r}"
                )
            if entry["name"] != _class_path(member):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} is a {entry['name']}, "
                    f"but this ensemble's member {k} is a {_class_path(member)}"
                )
            if entry["seed"] != self._member_seed(k):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} was trained with seed "
                    f"{entry['seed']!r}, but this ensemble's member {k} has seed "
                    f"{self._member_seed(k)!r}"
                )
            checkpoint = path.parent / entry["checkpoint"]
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    f"{self.class_name}: {path} member {k} checkpoint "
                    f"{checkpoint} not found"
                )
            checkpoints.append(checkpoint)
        return checkpoints

    def check_checkpoint(self, path: Path | str) -> None:
        """Check an ``ensemble.json`` and every member checkpoint it lists, loading nothing.

        The manifest must be a JSON object of format version 1 listing as
        many members as the ensemble has, each with this ensemble's member
        class and seed; every member checkpoint, resolved relative to the
        manifest's directory, must exist and pass that member's
        ``check_checkpoint``.

        Parameters
        ----------
        path : Path or str
            The ``ensemble.json`` that ``train()`` returned.

        Raises
        ------
        FileNotFoundError
            If the manifest or a member checkpoint does not exist.
        ValueError
            If the manifest is malformed, of an unknown ``format_version``,
            or does not match this ensemble's members, or a member's own
            check fails.

        Examples
        --------
        >>> ensemble.check_checkpoint(manifest) is None
        True
        """
        for member, checkpoint in zip(self.members, self._member_checkpoints(path)):
            member.check_checkpoint(checkpoint)

    def load(self, path: Path | str) -> Self:
        """Restore every member from an ``ensemble.json``.

        Every member checkpoint is checked (see ``check_checkpoint``) before
        any is loaded.

        Parameters
        ----------
        path : Path or str
            The ``ensemble.json`` that ``train()`` returned.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Raises
        ------
        FileNotFoundError
            If the manifest or a member checkpoint does not exist.
        ValueError
            As ``check_checkpoint``.

        Examples
        --------
        >>> SeedEnsemble(model, [0, 1, 2]).load(manifest).members[0].model is not None
        True
        """
        checkpoints = self._member_checkpoints(path)
        for member, checkpoint in zip(self.members, checkpoints):
            member.check_checkpoint(checkpoint)
        for member, checkpoint in zip(self.members, checkpoints):
            member.load(checkpoint)
        return self

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    @abstractmethod
    def get_config(self) -> dict:
        """Return a JSON-ready dict naming the ensemble class in ``"name"``."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild the ensemble from the dict ``get_config()`` returned."""
