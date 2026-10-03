"""Trained runs: the files a training run leaves on disk, and how they are read back.

Training writes one directory per trained unit. A unit holds its files and,
written last, ``run.json``, which describes the unit. There are three kinds:

- ``"model"``: one checkpoint, the ``config.json`` that rebuilds the model,
  and, when the fit was evaluated, the per-bar IC series ``ic_series.csv``
  and the test-segment predictions ``test_predictions.zarr``. ``run.json``
  holds the training window as configured and as fitted after the purge,
  the test window, what the model was trained on and the metrics.
- ``"ensemble"``: its members, each a ``"model"`` unit in ``member_{k}/``,
  and the evaluation files of the combined prediction. ``run.json`` holds
  the members (directory and seed), the three windows and the metrics. Its
  load-mode checkpoint is its ``run.json``.
- ``"walk_forward"``: its folds, each a ``"model"`` or ``"ensemble"`` unit
  in ``fold_{i}/``. ``run.json`` holds the folds (directory, index, three
  windows, metrics) and the fold means of the metrics, ``cv_mean``.

``train`` writes a unit at ``{model_save_dir}/{Class}_trial_{timestamp}/``,
``train_cv`` a ``"walk_forward"`` unit there. Paths inside ``run.json`` are
relative to the unit, so a unit copied elsewhere still opens.

``TrainedRun.open`` is how a run is read: from the unit's directory, its
``run.json``, or a model's checkpoint; members and folds are opened with
it, as child ``TrainedRun`` objects. A unit without ``run.json``, or written
in a ``format_version`` this module does not know, is refused with a
message to retrain it. The writing functions are for the model layer.

This module imports no other quantlab layer.

Examples
--------
>>> run = TrainedRun.open("models/MyHead_trial_20240601_120000_000000")
>>> run.kind, run.checkpoint.name
('model', 'MyHead_total.joblib')
>>> run.train_window, run.fitted_train_window
(('2024-01-01', '2024-02-09'), ('2024-01-01', '2024-02-07T00:00:00.000000000'))

A walk-forward run opens the same way; its folds are units of their own:

>>> cv = TrainedRun.open("models/MyHead_trial_20240601_130000_000000")
>>> cv.kind, len(cv.folds), cv.folds[0].kind, cv.folds[0].index
('walk_forward', 4, 'model', 0)
"""

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.jsonable import to_jsonable

#: The ``run.json`` structure this module writes and reads.
FORMAT_VERSION = 1

_RUN_FILE = "run.json"
_CONFIG_FILE = "config.json"
_IC_SERIES_FILE = "ic_series.csv"
_TEST_PREDICTIONS_FILE = "test_predictions.zarr"


@dataclass(frozen=True)
class TrainedRun:
    """One trained unit, as its ``run.json`` describes it.

    Every window is ``(start, end)``, both ends inclusive, with ``None`` for a
    date the model was not given. A ``"walk_forward"`` unit has no windows of
    its own (``None``); its folds have.

    Attributes
    ----------
    path : Path
        The unit's directory.
    kind : str
        ``"model"``, ``"ensemble"`` or ``"walk_forward"``.
    train_window : tuple or None
        The training window as configured, before the purge; an ensemble's
        covers its members'.
    fitted_train_window : tuple or None
        The training window actually fitted: ``train_window`` less the bars
        the purge drops before the test window.
    test_window : tuple or None
        The test window; an ensemble's is the one every member tests on.
    metrics : dict
        The ``train_*`` / ``val_*`` / ``test_*`` metrics of the fit, ``None``
        where a metric is undefined; empty when the fit was not evaluated
        and for a ``"walk_forward"`` unit.
    checkpoint : Path or None
        The file ``load`` reads: a ``"model"`` unit's checkpoint, an
        ``"ensemble"`` unit's ``run.json``; None for a ``"walk_forward"``
        unit.
    trained_on : dict or None
        What a ``"model"`` unit was trained on: ``factor_names``,
        ``label_names`` and the sorted training ``symbols``.
    ic_series : Path or None
        The per-bar IC series, when written.
    test_predictions : Path or None
        The test-segment prediction store, when written.
    members : tuple of TrainedRun
        An ``"ensemble"`` unit's members, in member order; empty otherwise.
    folds : tuple of TrainedRun
        A ``"walk_forward"`` unit's folds, in fold order; empty otherwise.
    cv_mean : dict or None
        A ``"walk_forward"`` unit's fold means, keyed ``cv_mean_{metric}``,
        with ``cv_n_folds``; empty when no fold has metrics, None for other
        kinds.
    index : int or None
        The fold's index in the walk, for a unit read as a fold.
    seed : int or None
        The seed its ensemble recorded, for a unit read as a member.

    Examples
    --------
    >>> run = TrainedRun.open(checkpoint)
    >>> run.checkpoint == checkpoint, sorted(run.trained_on)
    (True, ['factor_names', 'label_names', 'symbols'])
    >>> ensemble = TrainedRun.open(ensemble_dir)
    >>> ensemble.kind, [member.seed for member in ensemble.members]
    ('ensemble', [0, 1, 2])
    """

    path: Path
    kind: str
    train_window: tuple | None
    fitted_train_window: tuple | None
    test_window: tuple | None
    metrics: dict
    checkpoint: Path | None
    trained_on: dict | None
    ic_series: Path | None
    test_predictions: Path | None
    members: tuple = ()
    folds: tuple = ()
    cv_mean: dict | None = None
    index: int | None = None
    seed: int | None = None

    @classmethod
    def open(cls, path: Path | str) -> "TrainedRun":
        """Read the trained unit at ``path``, with its members and folds.

        Parameters
        ----------
        path : Path or str
            The unit's directory, its ``run.json``, or the checkpoint of a
            ``"model"`` unit.

        Returns
        -------
        TrainedRun

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the unit, or one of its members or folds, has no ``run.json``
            or a ``format_version`` other than ``FORMAT_VERSION``, or
            ``path`` is a file other than the unit's ``run.json`` or
            checkpoint.

        Examples
        --------
        >>> TrainedRun.open(checkpoint) == TrainedRun.open(checkpoint.parent)
        True
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        directory = path if path.is_dir() else path.parent
        record_path = directory / _RUN_FILE
        if not record_path.is_file():
            raise ValueError(
                f"{directory} has no {_RUN_FILE}, so it is not a trained run "
                f"this quantlab can read; retrain it"
            )
        record = json.loads(record_path.read_text(encoding="utf-8"))
        version = record.get("format_version") if isinstance(record, dict) else None
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{record_path} has format_version {version}, but this quantlab "
                f"reads format_version {FORMAT_VERSION}; retrain it"
            )
        if path.is_file() and path.name not in (_RUN_FILE, record.get("checkpoint")):
            raise ValueError(
                f"{path} is not the checkpoint of the trained run in {directory}"
            )
        kind = record["kind"]
        if kind == "model":
            checkpoint = directory / record["checkpoint"]
        elif kind == "ensemble":
            checkpoint = record_path
        else:
            checkpoint = None
        return cls(
            path=directory,
            kind=kind,
            train_window=_window(record.get("train_window")),
            fitted_train_window=_window(record.get("fitted_train_window")),
            test_window=_window(record.get("test_window")),
            metrics=dict(record.get("metrics") or {}),
            checkpoint=checkpoint,
            trained_on=record.get("trained_on"),
            ic_series=_optional(directory, record.get("ic_series")),
            test_predictions=_optional(directory, record.get("test_predictions")),
            members=tuple(
                dataclasses.replace(
                    cls.open(directory / entry["directory"]), seed=entry["seed"]
                )
                for entry in record.get("members", ())
            ),
            folds=tuple(
                _checked_fold(record_path, entry, cls.open(directory / entry["directory"]))
                for entry in record.get("folds", ())
            ),
            cv_mean=record.get("cv_mean") if kind == "walk_forward" else None,
        )

    @property
    def config(self) -> dict:
        """The ``config.json`` that rebuilds the model of a ``"model"`` unit.

        Examples
        --------
        >>> TrainedRun.open(checkpoint).config["name"]
        'tests.backtest_fixtures.FirstFeatureHead'
        """
        return json.loads((self.path / _CONFIG_FILE).read_text(encoding="utf-8"))


def _checked_fold(record_path: Path, entry: dict, fold: "TrainedRun") -> "TrainedRun":
    """Return ``fold`` with its index, after checking the walk-forward record's copy of it.

    Raises
    ------
    ValueError
        If the windows or metrics the walk-forward record holds for the fold
        differ from the fold's own ``run.json``.
    """
    copy = {
        "train_window": _window(entry.get("train_window")),
        "fitted_train_window": _window(entry.get("fitted_train_window")),
        "test_window": _window(entry.get("test_window")),
        "metrics": dict(entry.get("metrics") or {}),
    }
    own = {key: getattr(fold, key) for key in copy}
    if copy != own:
        raise ValueError(
            f"{record_path} records fold {entry['fold']} differently from "
            f"{fold.path / _RUN_FILE}; the run was altered after training, retrain it"
        )
    return dataclasses.replace(fold, index=entry["fold"])


def _window(value) -> tuple | None:
    """Return a recorded ``[start, end]`` as a tuple, or None when absent."""
    return None if value is None else tuple(value)


def _optional(directory: Path, name: str | None) -> Path | None:
    """Return ``directory / name``, or None when no name is recorded."""
    return None if name is None else directory / name


def new_trial_directory(root: Path | str, class_name: str) -> Path:
    """Return a fresh, absolute ``{class_name}_trial_{%Y%m%d_%H%M%S_%f}`` path under ``root``.

    The path does not exist yet and is not created: when the name is taken,
    ``_1``, ``_2``, ... are appended. It is absolute, so the checkpoints a
    unit records never depend on the working directory.

    Examples
    --------
    >>> new_trial_directory("models", "MyHead").name.startswith("MyHead_trial_")
    True
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = f"{class_name}_trial_{stamp}"
    root = Path(root).absolute()
    name, suffix = base, 1
    while (root / name).exists():
        name = f"{base}_{suffix}"
        suffix += 1
    return root / name


def fold_directory(unit: Path | str, index: int) -> Path:
    """Return where fold ``index`` of a walk-forward unit is written.

    Examples
    --------
    >>> fold_directory("trial", 3).as_posix()
    'trial/fold_3'
    """
    return Path(unit) / f"fold_{index}"


def member_directory(unit: Path | str, k: int) -> Path:
    """Return where member ``k`` of an ensemble unit is written.

    Examples
    --------
    >>> member_directory("trial", 0).as_posix()
    'trial/member_0'
    """
    return Path(unit) / f"member_{k}"


def evaluation_paths(directory: Path | str) -> tuple[Path, Path]:
    """Return where a unit's per-bar IC series and test-segment predictions are written.

    Examples
    --------
    >>> [path.as_posix() for path in evaluation_paths("unit")]
    ['unit/ic_series.csv', 'unit/test_predictions.zarr']
    """
    return Path(directory) / _IC_SERIES_FILE, Path(directory) / _TEST_PREDICTIONS_FILE


def write_model_config(directory: Path | str, config: dict) -> None:
    """Write the ``config.json`` that rebuilds a ``"model"`` unit's model.

    Examples
    --------
    >>> write_model_config(unit, model.get_config())
    >>> TrainedRun.open(unit).config == model.get_config()
    True
    """
    with open(Path(directory) / _CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)


def _evaluation_names(directory: Path) -> dict:
    """Return the ``ic_series`` / ``test_predictions`` entries, each a file's name when it exists."""
    ic_series, test_predictions = evaluation_paths(directory)
    return {
        "ic_series": ic_series.name if ic_series.exists() else None,
        "test_predictions": (
            test_predictions.name if test_predictions.exists() else None
        ),
    }


def _write_run(directory: Path, record: dict) -> Path:
    """Write ``record`` as the unit's ``run.json``, atomically; NaN and inf become null."""
    path = directory / _RUN_FILE
    write_json_atomically(
        path,
        to_jsonable({"format_version": FORMAT_VERSION, **record}),
        indent=2,
    )
    return path


def write_model_run(
    directory: Path | str,
    *,
    checkpoint: Path | str,
    train_window: tuple,
    fitted_train_window: tuple,
    test_window: tuple,
    trained_on: dict,
    metrics: dict | None,
) -> None:
    """Write the ``run.json`` of a ``"model"`` unit, atomically, as its last file.

    The evaluation files are recorded when the ``evaluation_paths`` of
    ``directory`` exist; NaN and inf metrics become null.

    Parameters
    ----------
    directory : Path or str
        The unit's directory, already holding the checkpoint.
    checkpoint : Path or str
        The checkpoint file, inside ``directory``.
    train_window, fitted_train_window, test_window : tuple
        ``(start, end)`` pairs, see ``TrainedRun``.
    trained_on : dict
        ``factor_names``, ``label_names`` and ``symbols``.
    metrics : dict or None
        The fit's metrics; None when the fit was not evaluated.

    Examples
    --------
    >>> write_model_run(unit, checkpoint=unit / "head.joblib",
    ...                 train_window=("2024-01-01", "2024-02-09"),
    ...                 fitted_train_window=("2024-01-01", "2024-02-07"),
    ...                 test_window=("2024-02-12", "2024-02-23"),
    ...                 trained_on={"factor_names": ["f"], "label_names": ["y"], "symbols": ["A"]},
    ...                 metrics=None)
    >>> TrainedRun.open(unit).metrics
    {}
    """
    directory = Path(directory)
    _write_run(
        directory,
        {
            "kind": "model",
            "checkpoint": Path(checkpoint).name,
            "train_window": list(train_window),
            "fitted_train_window": list(fitted_train_window),
            "test_window": list(test_window),
            "trained_on": trained_on,
            "metrics": metrics or {},
            **_evaluation_names(directory),
        },
    )


def write_ensemble_run(
    directory: Path | str,
    *,
    seeds: Sequence,
    train_window: tuple,
    fitted_train_window: tuple,
    test_window: tuple,
    metrics: dict,
) -> Path:
    """Write the ``run.json`` of an ``"ensemble"`` unit, atomically, as its last file.

    Member k is the ``"model"`` unit in ``member_directory(directory, k)``,
    which must already be written; ``seeds`` holds one entry per member.

    Parameters
    ----------
    directory : Path or str
        The unit's directory.
    seeds : sequence
        Each member's seed, None when the ensemble does not vary seeds.
    train_window, fitted_train_window, test_window : tuple
        ``(start, end)`` pairs of the ensemble, see ``TrainedRun``.
    metrics : dict
        The metrics of the combined prediction.

    Returns
    -------
    Path
        The ``run.json`` written, the ensemble's load-mode checkpoint.

    Examples
    --------
    >>> path = write_ensemble_run(unit, seeds=[0, 1],
    ...                           train_window=("2024-01-01", "2024-02-09"),
    ...                           fitted_train_window=("2024-01-01", "2024-02-07"),
    ...                           test_window=("2024-02-12", "2024-02-23"),
    ...                           metrics={})
    >>> TrainedRun.open(path).checkpoint == path
    True
    """
    directory = Path(directory)
    return _write_run(
        directory,
        {
            "kind": "ensemble",
            "members": [
                {
                    "directory": member_directory(directory, k).name,
                    "seed": seed,
                }
                for k, seed in enumerate(seeds)
            ],
            "train_window": list(train_window),
            "fitted_train_window": list(fitted_train_window),
            "test_window": list(test_window),
            "metrics": metrics,
            **_evaluation_names(directory),
        },
    )


def write_walk_forward_run(
    directory: Path | str, *, folds: Sequence[int], cv_mean: dict
) -> TrainedRun:
    """Write the ``run.json`` of a ``"walk_forward"`` unit and return the unit.

    Fold i is the unit in ``fold_directory(directory, i)``, which must
    already be written; its windows and metrics are copied from its own
    ``run.json`` into the walk-forward record.

    Parameters
    ----------
    directory : Path or str
        The unit's directory.
    folds : sequence of int
        The fold indices, in fold order.
    cv_mean : dict
        The fold means of the metrics; NaN and inf become null.

    Returns
    -------
    TrainedRun
        The walk-forward unit, as ``TrainedRun.open`` reads it.

    Examples
    --------
    >>> run = write_walk_forward_run(trial, folds=[0, 1], cv_mean={})
    >>> run.kind, [fold.index for fold in run.folds]
    ('walk_forward', [0, 1])
    """
    directory = Path(directory)
    entries = []
    for index in folds:
        fold = TrainedRun.open(fold_directory(directory, index))
        entries.append(
            {
                "fold": index,
                "directory": fold.path.name,
                "kind": fold.kind,
                "train_window": list(fold.train_window),
                "fitted_train_window": list(fold.fitted_train_window),
                "test_window": list(fold.test_window),
                "metrics": fold.metrics,
            }
        )
    _write_run(
        directory,
        {"kind": "walk_forward", "folds": entries, "cv_mean": cv_mean},
    )
    return TrainedRun.open(directory)
