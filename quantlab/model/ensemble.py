"""The base class of every ensemble of models, shipped or user-written.

``BaseEnsemble`` holds what does not depend on where an ensemble's members
come from: the ``Predictor`` members derived from the members (labels,
label delays and scales, training and test windows), prediction by
combining the members' predictions, and the ``"ensemble"`` trained unit
(``quantlab.runs.trained_run``) that ``train()`` writes and ``load`` and
``check_checkpoint`` read.

Each label is combined over the members that predict it (ADR 0013): the
ensemble's labels are the union of the members' labels in first-appearance
order, a label several members predict is standardised per bar and
averaged, and a label only one member predicts is passed through unchanged.
So seed ensembles, ensembles of different models and a return model paired
with a volatility model are one rule. A label predicted by several members
must have the same config in each. The training end is the latest
member's, the test window the intersection of the members'.

A concrete ensemble builds its members and implements ``get_config`` /
``from_config``; nothing else is required, and the defaults work for members
of different classes over different factors. Optional hooks:

- ``_combine``: how the members' prediction panels become the ensemble's;
  by default each label is averaged over its members with
  ``average_predictions`` (per-bar z-score, equal-weight mean) or passed
  through from its only member. Both ``predict_window`` and the
  ensemble-level evaluation files use it. An override should keep
  ``label_scales`` true, overriding it too when needed.
- ``_collect``, ``_member_predictions``, ``_member_panel_predictions``: how
  members collect their data and features, each member on its own by
  default; an ensemble whose members read the same data shares it.
  ``collect`` records what ``_collect`` reads, on the ensemble's unit.
- ``_member_seed``: the seed recorded for each member in ``run.json``.

Shipped ensembles are in ``quantlab/model/predefined`` (``SeedEnsemble``,
``ModelEnsemble``).

The ensemble composes models and inherits none: each member is a complete
model (a ``BaseModel``) with its own checkpoint.

On disk ``train()`` writes an ``"ensemble"`` unit::

    {model_save_dir}/{EnsembleClass}_trial_{timestamp}/
        member_0/            one member's ``"model"`` unit
        member_1/
        ...
        ic_series.csv        the first label's per-bar series, in the single-model layout
        test_predictions.zarr  the combined test-segment prediction
        run.json             members and seeds, windows, metrics; written last

``train_cv`` writes a ``"walk_forward"`` unit whose folds are such units::

    {model_save_dir}/{EnsembleClass}_trial_{timestamp}/
        fold_0/              an ensemble unit as above
        fold_1/
        ...
        run.json             the folds and the fold means ``cv_mean``

The record names each member's directory and seed (null when the ensemble
does not vary seeds) and nothing specific to one kind of ensemble; each
member's class is in its own ``config.json``.
"""

from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.base.component import Component, code_of
from quantlab.base.model import BaseModel, record_training_reads
from quantlab.runs.trained_run import (
    TrainedRun,
    evaluation_paths,
    fold_directory,
    member_directory,
    new_trial_directory,
    write_ensemble_run,
)
from quantlab.utils.ensemble import average_predictions, member_correlation
from quantlab.utils.metrics import (
    ic_panel_metrics,
    scores_volatility_level,
    volatility_level_metrics,
)


def _as_time(value) -> pd.Timestamp:
    """Order a date bound: ``value`` as a ``pd.Timestamp`` (a ``numpy.str_`` too)."""
    return pd.Timestamp(str(value) if isinstance(value, str) else value)


def _covering(bounds) -> tuple:
    """Return the ``(start, end)`` window covering every ``(start, end)`` of ``bounds``."""
    return (
        min((b[0] for b in bounds), key=_as_time),
        max((b[1] for b in bounds), key=_as_time),
    )


class BaseEnsemble(Component, ABC):
    """Base class of an ensemble that combines the predictions of several models.

    Subclass it, build the members, declare ``config_cls`` (a dataclass
    whose member fields are declared with ``component()``) with a ``config``
    property building it, and override ``from_config`` to construct the
    ensemble from the rebuilt fields; override ``_combine`` to replace the
    per-label rule (see the module docstring for every hook).

    Parameters
    ----------
    members : sequence
        At least two models (``BaseModel`` instances). A label two members
        predict must have the same config in both, and their test windows
        must overlap.

    Attributes
    ----------
    members : list
        The member models, in member order.

    Raises
    ------
    ValueError
        If there are fewer than two members, two members predict a label of
        one name with different configs, or the members' test windows do
        not overlap.

    Examples
    --------
    A concrete ensemble passes its members to this constructor; given a
    ``SeedEnsemble`` of a model over one forward-return label::

        >>> len(ensemble.members), ensemble.label_delays
        (3, (1,))
    """

    def __init__(self, members: Sequence):
        """Initialize the ensemble; see the class docstring for parameters."""
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"{self.class_name} needs at least two members, got {len(members)}"
            )
        self._check_members_agree(members)
        self.members = members
        self.test_bounds  # refuses members whose test windows do not overlap
        # What the last collect() read; see training_record.
        self._training_record: dict = {}

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

    def _check_members_agree(self, members: list) -> None:
        """Refuse members that predict a label of one name with different configs.

        Labels are compared by ``get_config()``, which holds their class,
        variables and delay, and by ``kind``, which decides how the label is
        evaluated; every variable name a label outputs is checked.

        Raises
        ------
        ValueError
            Naming the first member whose label differs from an earlier
            member's label of the same name.
        """
        seen: dict[str, tuple[int, dict, str]] = {}
        for k, member in enumerate(members):
            for label in member.labels:
                config, kind = label.get_config(), getattr(label, "kind", "return")
                for name in label.get_factor_names():
                    if name not in seen:
                        seen[name] = (k, config, kind)
                        continue
                    first, reference, reference_kind = seen[name]
                    if (config, kind) != (reference, reference_kind):
                        raise ValueError(
                            f"{self.class_name}: member {k} ({type(member).__name__}) "
                            f"predicts label {name!r} with config {config!r} (kind "
                            f"{kind!r}), but member {first} predicts it with "
                            f"{reference!r} (kind {reference_kind!r}); a label "
                            f"several members predict must have one config"
                        )

    # ------------------------------------------------------------------
    # Predictor members derived from the members
    # ------------------------------------------------------------------

    def _label_owners(self) -> dict[str, list[int]]:
        """Map each label name to the indices of the members predicting it.

        The names follow first appearance over the members in order.
        """
        owners: dict[str, list[int]] = {}
        for k, member in enumerate(self.members):
            for name in member.get_label_names():
                owners.setdefault(str(name), []).append(k)
        return owners

    @property
    def labels(self) -> list:
        """The union of the members' label objects, in first-appearance order.

        A label several members predict appears once, as the first member
        predicting it holds it.

        Examples
        --------
        >>> [label.get_factor_names() for label in ensemble.labels]
        [('fwd_ret_1',)]
        """
        labels, seen = [], set()
        for member in self.members:
            for label in member.labels:
                names = {str(name) for name in label.get_factor_names()}
                if names <= seen:
                    continue
                seen |= names
                labels.append(label)
        return labels

    @property
    def label_scales(self) -> dict[str, str]:
        """Each label name's prediction scale: ``"raw"`` or ``"standardized"``.

        A label averaged over several members is ``"standardized"`` (the
        default combine is in per-bar z-score units); a label only one
        member predicts keeps that member's scale.

        Examples
        --------
        >>> ensemble.label_scales
        {'fwd_ret_1': 'standardized'}
        """
        return {
            name: (
                "standardized"
                if len(owners) > 1
                else self.members[owners[0]].label_scales[name]
            )
            for name, owners in self._label_owners().items()
        }

    @property
    def train_bounds(self) -> tuple:
        """The training window ``(train_start, train_end)`` covering every member's.

        The earliest member start and the latest member end, so a bar
        after ``train_end`` was seen in training by no member.

        Examples
        --------
        >>> ensemble.train_bounds
        ('2024-01-01', '2024-02-02')
        """
        return _covering([member.train_bounds for member in self.members])

    @property
    def fitted_train_bounds(self) -> tuple:
        """The training window actually fitted, covering every member's.

        The earliest member start and the latest member end of the members'
        ``fitted_train_bounds``, so a bar after the end was fitted by no
        member. Known after ``train()`` or ``load()``.

        Raises
        ------
        RuntimeError
            If the members have been neither trained nor loaded.

        Examples
        --------
        >>> ensemble.fitted_train_bounds
        ('2024-01-01', '2024-01-31T00:00:00.000000000')
        """
        return _covering([member.fitted_train_bounds for member in self.members])

    @property
    def test_bounds(self) -> tuple:
        """The test window ``(test_start, test_end)`` every member tests on.

        The intersection of the members' test windows: the latest start and
        the earliest end.

        Raises
        ------
        ValueError
            If the members' test windows do not overlap.

        Examples
        --------
        >>> ensemble.test_bounds
        ('2024-02-05', '2024-02-09')
        """
        bounds = [tuple(member.test_bounds) for member in self.members]
        start = max((b[0] for b in bounds), key=_as_time)
        end = min((b[1] for b in bounds), key=_as_time)
        if _as_time(start) > _as_time(end):
            raise ValueError(
                f"{self.class_name}: the members' test windows {bounds!r} do not "
                f"overlap, so the ensemble has no test window"
            )
        return start, end

    @property
    def label_delays(self) -> tuple[int, ...]:
        """Each label's ``delay`` in bars, in the order of ``labels``.

        Examples
        --------
        >>> ensemble.label_delays
        (1,)
        """
        return tuple(label.config.delay for label in self.labels)

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
        """The seed recorded for member ``k`` in ``run.json``; None by default."""
        return None

    def collect(self) -> Self:
        """Load every member's features and labels into its data backend.

        The members' reads (``_collect``) are recorded once, by a
        ``DataRecorder`` keyed by component path within the ensemble, and
        written into the ``run.json`` of the ensemble or walk-forward unit
        ``train()`` or ``train_cv()`` writes next; its members and folds
        record none.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Examples
        --------
        >>> ensemble.collect() is ensemble
        True
        >>> sorted(ensemble.training_record)[:1]
        ['model.factors.0.dataset']
        """
        for member in self.members:
            member._check_hyperparameters()
        self._training_record = record_training_reads(self)
        return self

    @property
    def training_record(self) -> dict:
        """What the last ``collect()`` read, by component path; empty before it.

        Examples
        --------
        >>> sorted(ensemble.collect().training_record)[:1]
        ['model.factors.0.dataset']
        """
        return dict(self._training_record)

    def _provenance(self) -> dict:
        """What the ensemble's top unit records: ``training_record`` and the code record."""
        return {"data_fingerprint": self.training_record, "code": code_of(self)}

    def _collect(self) -> None:
        """Collect every member's own data; see ``BaseModel._collect``.

        An ensemble whose members read the same data overrides this to
        collect once.
        """
        for member in self.members:
            member._collect()

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

        The default groups by label. A label several members predict is
        ``average_predictions`` of their panels: z-scored over symbols per
        member and bar, then averaged with equal weights, ignoring NaN, in
        z-score units. A label one member predicts is that member's
        prediction, unchanged. Override it for another rule, such as fixed
        weights or a rank average (and ``label_scales`` with it when the
        scales change); it sees only the predictions, so a rule whose
        parameters are learned in training does not fit here.

        Parameters
        ----------
        predictions : list[xr.Dataset]
            One panel per member, in member order, each on
            ``(timestamp, symbol)`` with one variable per label name the
            member predicts. The members' coordinates may differ.

        Returns
        -------
        xr.Dataset
            One variable per label name on ``(timestamp, symbol)``, in
            first-appearance order, over the union of the coordinates.

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
        >>> vol = panel([0.2, 0.3, 0.1]).rename(ret="vol")
        >>> out = BaseEnsemble._combine(None, [panel([1.0, 2.0, 3.0]), vol])
        >>> list(out.data_vars), out["vol"].values
        (['ret', 'vol'], array([[0.2, 0.3, 0.1]]))
        """
        owners: dict[str, list[int]] = {}
        for k, panel in enumerate(predictions):
            for name in panel.data_vars:
                owners.setdefault(str(name), []).append(k)
        if all(len(ks) == len(predictions) for ks in owners.values()):
            return average_predictions(predictions)
        parts = []
        for name, ks in owners.items():
            if len(ks) > 1:
                parts.append(average_predictions([predictions[k][[name]] for k in ks]))
            else:
                parts.append(predictions[ks[0]][[name]])
        return xr.merge(parts, join="outer", compat="override", combine_attrs="drop_conflicts")

    def predict_window(self, start, end) -> xr.Dataset:
        """Predict every bar from ``start`` to ``end`` by combining the members.

        The members' predictions are combined by ``_combine``: by default a
        label several members predict is the equal-weight mean of their
        per-bar z-scores and a label one member predicts is its prediction.
        Every member must be trained or loaded.

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

    # ------------------------------------------------------------------
    # Training and the ensemble directory
    # ------------------------------------------------------------------

    def train(self) -> Path:
        """Train every member into one ensemble unit and return its ``run.json``.

        Every member's hyperparameters are checked first. Then a new
        ``{class}_trial_{timestamp}`` directory under ``model_save_dir`` is
        filled by ``_train_into``: member k is trained, in order, into the
        ``"model"`` unit ``member_{k}/`` under a tracking run
        ``{MemberClass}_member_{k}`` of its own tracker, grouped by the
        directory's name, and reseeds its generators from its own
        ``random_seed`` right before it trains. The directory then gets the
        evaluation files of the combined prediction (``ic_series.csv``,
        ``test_predictions.zarr``, see ``_write_evaluation_files``) and
        last, atomically, ``run.json``, which makes it an ``"ensemble"``
        unit (``quantlab.runs.trained_run``) recording the members, the
        windows and the metrics. If a member or the ensemble evaluation
        fails, the error propagates, no ``run.json`` is written and the
        files already written stay. Call ``collect()`` first.

        Returns
        -------
        Path
            Absolute path of the unit's ``run.json``, the ensemble's
            load-mode checkpoint.

        Examples
        --------
        >>> checkpoint = ensemble.collect().train()
        >>> run = TrainedRun.open(checkpoint)
        >>> run.checkpoint == checkpoint, run.path.name.startswith("SeedEnsemble_trial_")
        (True, True)
        >>> sorted(p.name for p in run.path.iterdir())
        ['ic_series.csv', 'member_0', 'member_1', 'member_2', 'run.json', 'test_predictions.zarr']
        >>> sorted(run.metrics)
        ['test_ic', 'test_icir', 'test_member_correlation', 'test_rank_ic', 'test_rank_icir', 'train_ic', 'train_icir', 'train_member_correlation', 'train_rank_ic', 'train_rank_icir']
        """
        for member in self.members:
            member._check_hyperparameters()
        directory = new_trial_directory(self.model_save_dir, self.class_name)
        return self._train_into(
            directory, group=directory.name, provenance=self._provenance()
        )

    def _train_into(
        self,
        run_dir: Path | str,
        group: str,
        *,
        run_tag: str | None = None,
        provenance: dict | None = None,
    ) -> Path:
        """Train every member, evaluate the combination and write the unit into ``run_dir``.

        Member k trains into ``member_{k}/`` of ``run_dir`` under the
        tracking run ``{MemberClass}_member_{k}``
        (``{MemberClass}_{run_tag}_member_{k}`` with a ``run_tag``) in the
        group ``group``. Then come the evaluation files of the combined
        prediction (see ``_write_evaluation_files``) and last ``run.json``.
        Hyperparameters are not checked here: ``train`` checks them first,
        and ``train_cv`` once before its folds.

        Parameters
        ----------
        run_dir : Path or str
            The unit's directory. It is created with its parents when
            missing; its member directories must not exist yet.
        group : str
            Tracking group of the members' runs, the trial directory's name.
        run_tag : str, optional
            Inserted into every member's run name, so that runs of several
            ensemble units in one group stay apart; ``train_cv`` passes
            ``fold_{i}``.
        provenance : dict, optional
            ``_provenance()``, for the top unit (``train``); a fold records
            none.

        Returns
        -------
        Path
            The absolute path of the unit's ``run.json``.
        """
        directory = Path(run_dir).absolute()
        directory.mkdir(parents=True, exist_ok=True)
        tag = "" if run_tag is None else f"_{run_tag}"
        for k, member in enumerate(self.members):
            member._train_into(
                member_directory(directory, k),
                group=group,
                experiment_name=f"{member.class_name}{tag}_member_{k}",
            )
        metrics = self._write_evaluation_files(directory)
        return write_ensemble_run(
            directory,
            seeds=[self._member_seed(k) for k in range(len(self.members))],
            train_window=self.train_bounds,
            fitted_train_window=self.fitted_train_bounds,
            test_window=self.test_bounds,
            metrics=metrics,
            provenance=provenance,
        )

    def train_cv(
        self,
        train_periods: int,
        expanding: bool = False,
        test_periods: int | None = None,
    ) -> TrainedRun:
        """Run a walk-forward cross-validation of the ensemble and return the walk-forward unit.

        The folds are those ``BaseModel.train_cv`` trains for the first
        member: laid out by ``quantlab.utils.walk_forward.walk_forward_folds`` over the first member's
        collected timestamps between its ``start_date`` and ``end_date``,
        sliding or, with ``expanding=True``, growing from the first fold's
        start, and each training window loses its last L bars, L being the
        largest ``lookahead_bars()`` among every member's labels. Every member's
        hyperparameters are checked once, before any directory is created.

        A new ``{class}_trial_{timestamp}`` directory under
        ``model_save_dir`` becomes a ``"walk_forward"`` unit, laid out as a
        model's. For fold i, every member's config gets the fold's dates
        before the purge (the members purge them themselves, as a single
        model's fold does), and ``_train_into`` writes the ``"ensemble"``
        unit ``fold_{i}/`` as ``train()`` writes its directory, member k
        under the tracking run ``{MemberClass}_fold_{i}_member_{k}``. Folds
        train one after another. After the last fold the members keep its
        dates, as a model does after its own ``train_cv``. The fold means of
        the ensemble metrics, keyed ``cv_mean_{key}``, and ``cv_n_folds`` go
        to the summary of a separate ``{class}_cv_summary`` run, opened
        through the first member's tracker in the members' project and
        group, and into the unit's ``run.json``, which a backtester's
        ``run_cv`` replays with the ensemble as its model. Call
        ``collect()`` first.

        Parameters
        ----------
        train_periods : int
            Number of timestamps in the first fold's training segment, and
            in every fold's when sliding.
        expanding : bool, default False
            Train every fold from the first fold's start instead of sliding
            a fixed-length window.
        test_periods : int, optional
            Number of timestamps in each fold's test segment, and the step
            from one fold to the next. Defaults to ``train_periods // 5``.

        Returns
        -------
        TrainedRun
            The ``"walk_forward"`` unit, whose ``folds`` are ``"ensemble"``
            units with the fold's ensemble metrics (``{split}_ic``,
            ``{split}_rank_ic``, ``{split}_icir``, ``{split}_rank_icir``,
            ``{split}_member_correlation``).

        Raises
        ------
        ValueError
            If ``test_periods`` is below 1, or is not given and
            ``train_periods`` is below 5, no timestamp falls inside the
            date range, the purge leaves a fold no training bar, or a
            member's hyperparameters are invalid.

        Examples
        --------
        >>> cv = ensemble.collect().train_cv(train_periods=30)
        >>> cv.kind, len(cv.folds), cv.folds[0].kind
        ('walk_forward', 8, 'ensemble')
        >>> [member.seed for member in cv.folds[0].members]
        [0, 1, 2]
        """
        for member in self.members:
            member._check_hyperparameters()
        first = self.members[0]
        folds = first._walk_forward_folds(
            train_periods,
            expanding,
            test_periods,
            max(member._purge_bars() for member in self.members),
            self.class_name,
        )

        trial = new_trial_directory(self.model_save_dir, self.class_name)
        for fold in folds:
            for member in self.members:
                member.config = BaseModel._with_fold_dates(member.config, fold)
            self._train_into(
                fold_directory(trial, fold.index),
                group=trial.name,
                run_tag=f"fold_{fold.index}",
            )
        # Through the first member's tracker, beside the members' runs.
        return first._finish_walk_forward(
            trial,
            folds,
            name=f"{self.class_name}_cv_summary",
            config=self.get_config(),
            provenance=self._provenance(),
        )

    def _write_evaluation_files(self, run_dir: Path) -> dict:
        """Evaluate the combined prediction of the trained members and write its files.

        Every member predicts its whole collected panel
        (``_member_panel_predictions``) and the predictions are combined by
        ``_combine``, the same rule ``predict_window`` uses. Each label is
        scored against the truth of the first member predicting it: that
        member's collected panel, cut by its ``_fit_segments`` into the
        purged train, validation and test segments a single model evaluates,
        a split being skipped when it has no bars (so no ``val_*`` without a
        validation segment). On each split ``ic_panel_metrics`` scores the
        combined prediction of the label against the label's raw values,
        and, for a label at least two members predict,
        ``member_correlation`` measures how much their predictions of it
        agree. No error metric is computed: an averaged label is in z-score
        units, not in the target's. A label whose ``kind`` is
        ``"volatility"`` and whose combined prediction is on its own scale
        (``label_scales`` ``"raw"``, a label one raw member predicts) also
        gets the ``volatility_level_metrics`` ``qlike`` and
        ``variance_ratio``.

        The metrics, which ``run.json`` records, are for the first label
        ``{split}_ic``, ``{split}_rank_ic``, ``{split}_icir``,
        ``{split}_rank_icir`` and, when shared, ``{split}_member_correlation``
        (the mean over bars of the mean pairwise Pearson correlation of the
        members' predictions over their common finite symbols), and
        ``{split}_qlike`` / ``{split}_variance_ratio`` for a raw volatility
        label; for every other label the same keys as
        ``{split}_{label}_{metric}``.

        Written into ``run_dir``:

        - ``ic_series.csv``: the first label's per-bar series behind them, in
          the layout of a single model's file (``BaseModel._write_ic_series``).
        - ``test_predictions.zarr``: the combined prediction, one variable
          per label, on the first member's test bars inside the ensemble's
          ``test_bounds``; not written when there are none.

        Returns
        -------
        dict
            The metrics, with NaN where a metric is undefined.
        """
        predictions = self._member_panel_predictions()
        combined = self._combine(predictions)
        scales = self.label_scales
        # Members agree on every label of one name (_check_members_agree), so
        # the first label object naming a variable speaks for all of them.
        objects = {str(name): obj for obj in reversed(self.labels) for name in obj.get_factor_names()}
        metrics, series = {}, {}
        for i, (label, owners) in enumerate(self._label_owners().items()):
            level = scores_volatility_level(objects.get(label), scales.get(label))
            member = self.members[owners[0]]
            data = member.data_backend.get_xarray_dataset(
                ["timestamp", "symbol"]
            ).sortby(["timestamp", "symbol"])
            prefix = "" if i == 0 else f"{label}_"
            for split, part in zip(("train", "val", "test"), member._fit_segments(data)):
                stamps = part.timestamp.values
                if len(stamps) == 0:
                    continue
                pred = combined[label].reindex(timestamp=stamps, symbol=data.symbol.values)
                values, per_bar = ic_panel_metrics(
                    pred.values,
                    data[label].sel(timestamp=stamps).values,
                    return_series=True,
                )
                if level:
                    values.update(volatility_level_metrics(
                        pred.values, data[label].sel(timestamp=stamps).values
                    ))
                metrics.update(
                    {f"{split}_{prefix}{key}": value for key, value in values.items()}
                )
                if len(owners) > 1:
                    metrics[f"{split}_{prefix}member_correlation"], _ = member_correlation(
                        [
                            predictions[k][label]
                            .reindex(timestamp=stamps, symbol=data.symbol.values)
                            .values
                            for k in owners
                        ]
                    )
                if i == 0:
                    series[split] = (stamps, per_bar["ic"], per_bar["rank_ic"])

        ic_series_path, test_predictions_path = evaluation_paths(run_dir)
        first = self.members[0]
        first._write_ic_series(ic_series_path, series)
        data = first.data_backend.get_xarray_dataset(["timestamp", "symbol"]).sortby(
            ["timestamp", "symbol"]
        )
        test_stamps = first._fit_segments(data)[2].timestamp.values
        test_start, test_end = self.test_bounds
        test_stamps = test_stamps[
            (test_stamps >= np.datetime64(_as_time(test_start)))
            & (test_stamps <= np.datetime64(_as_time(test_end)))
        ]
        if len(test_stamps):
            combined.reindex(timestamp=test_stamps).to_zarr(
                test_predictions_path, mode="w"
            )
        return metrics

    # ------------------------------------------------------------------
    # The trained unit: check and load
    # ------------------------------------------------------------------

    def _open_unit(self, path: Path | str) -> TrainedRun:
        """Open an ``"ensemble"`` unit whose members match this ensemble's.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If ``path`` is not an ``"ensemble"`` unit ``TrainedRun`` can
            open, or it holds a different number of members, a member of a
            different class or a member trained with a different seed than
            this ensemble's.
        """
        run = TrainedRun.open(path)
        if run.kind != "ensemble":
            raise ValueError(
                f"{self.class_name}: {path} is a {run.kind!r} trained run, not an "
                f"ensemble"
            )
        if len(run.members) != len(self.members):
            raise ValueError(
                f"{self.class_name}: {path} holds {len(run.members)} members, but "
                f"this ensemble has {len(self.members)}"
            )
        for k, (saved, member) in enumerate(zip(run.members, self.members)):
            name = saved.config.get("name")
            if name != member.import_path:
                raise ValueError(
                    f"{self.class_name}: {path} member {k} is a {name}, but this "
                    f"ensemble's member {k} is a {member.import_path}"
                )
            if saved.seed != self._member_seed(k):
                raise ValueError(
                    f"{self.class_name}: {path} member {k} was trained with seed "
                    f"{saved.seed!r}, but this ensemble's member {k} has seed "
                    f"{self._member_seed(k)!r}"
                )
        return run

    def check_checkpoint(self, path: Path | str) -> TrainedRun:
        """Check an ensemble unit and every member checkpoint in it, loading nothing.

        The unit, read through ``TrainedRun``, must hold as many members as
        the ensemble has, each with this ensemble's member class and seed,
        and every member checkpoint must pass that member's
        ``check_checkpoint``.

        Parameters
        ----------
        path : Path or str
            The unit's ``run.json`` that ``train()`` returned, or its
            directory.

        Returns
        -------
        TrainedRun
            The ensemble unit.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the unit cannot be opened or does not match this ensemble's
            members, or a member's own check fails.

        Examples
        --------
        >>> ensemble.check_checkpoint(checkpoint).kind
        'ensemble'
        """
        run = self._open_unit(path)
        for member, saved in zip(self.members, run.members):
            member.check_checkpoint(saved.checkpoint)
        return run

    def load(self, path: Path | str) -> Self:
        """Restore every member from an ensemble unit.

        Every member checkpoint is checked (see ``check_checkpoint``) before
        any is loaded.

        Parameters
        ----------
        path : Path or str
            The unit's ``run.json`` that ``train()`` returned, or its
            directory.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            As ``check_checkpoint``.

        Examples
        --------
        >>> SeedEnsemble(model, [0, 1, 2]).load(checkpoint).members[0].model is not None
        True
        """
        run = self.check_checkpoint(path)
        for member, saved in zip(self.members, run.members):
            member.load(saved.checkpoint)
        return self
