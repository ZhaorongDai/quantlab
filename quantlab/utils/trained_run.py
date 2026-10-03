"""Trained runs: the files a training run leaves on disk, and how they are read back.

Training writes one directory per trained unit. A unit holds its files and,
written last, ``run.json``, which describes the unit: its kind, its training
window as configured and as fitted after the purge, its test window, its
metrics, and the names of its other files. A unit of kind ``"model"`` holds
one checkpoint, the ``config.json`` that rebuilds the model, and, when the fit
was evaluated, the per-bar IC series ``ic_series.csv`` and the test-segment
predictions ``test_predictions.zarr``. Paths inside ``run.json`` are relative
to the unit, so a unit copied elsewhere still opens.

``TrainedRun.open`` is how a run is read: from the unit's directory, its
``run.json``, or its checkpoint. A unit without ``run.json``, or written in a
``format_version`` this module does not know, is refused with a message to
retrain it. The writing functions are for the model layer.

This module imports no other quantlab layer.

Examples
--------
>>> run = TrainedRun.open("models/MyHead_trial_20240601_120000_000000")
>>> run.kind, run.checkpoint.name
('model', 'MyHead_total.joblib')
>>> run.train_window, run.fitted_train_window
(('2024-01-01', '2024-02-09'), ('2024-01-01', '2024-02-07T00:00:00.000000000'))
"""

import json
from dataclasses import dataclass
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
    date the model was not given.

    Attributes
    ----------
    path : Path
        The unit's directory.
    kind : str
        ``"model"``.
    train_window : tuple
        The training window as configured, before the purge.
    fitted_train_window : tuple
        The training window actually fitted: ``train_window`` less the bars
        the purge drops before the test window.
    test_window : tuple
        The test window.
    metrics : dict
        The ``train_*`` / ``val_*`` / ``test_*`` metrics of the fit, ``None``
        where a metric is undefined; empty when the fit was not evaluated.
    checkpoint : Path or None
        The checkpoint file of a ``"model"`` unit.
    trained_on : dict or None
        What a ``"model"`` unit was trained on: ``factor_names``,
        ``label_names`` and the sorted training ``symbols``.
    ic_series : Path or None
        The per-bar IC series, when written.
    test_predictions : Path or None
        The test-segment prediction store, when written.

    Examples
    --------
    >>> run = TrainedRun.open(checkpoint)
    >>> run.checkpoint == checkpoint, sorted(run.trained_on)
    (True, ['factor_names', 'label_names', 'symbols'])
    """

    path: Path
    kind: str
    train_window: tuple
    fitted_train_window: tuple
    test_window: tuple
    metrics: dict
    checkpoint: Path | None
    trained_on: dict | None
    ic_series: Path | None
    test_predictions: Path | None

    @classmethod
    def open(cls, path: Path | str) -> "TrainedRun":
        """Read the trained unit at ``path``.

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
            If the unit has no ``run.json``, its ``format_version`` is not
            ``FORMAT_VERSION``, or ``path`` is a file other than the
            unit's ``run.json`` or checkpoint.

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
        checkpoint = (
            directory / record["checkpoint"] if record.get("checkpoint") else None
        )
        if path.is_file() and path.name not in (_RUN_FILE, record.get("checkpoint")):
            raise ValueError(
                f"{path} is not the checkpoint of the trained run in {directory}"
            )
        return cls(
            path=directory,
            kind=record["kind"],
            train_window=tuple(record["train_window"]),
            fitted_train_window=tuple(record["fitted_train_window"]),
            test_window=tuple(record["test_window"]),
            metrics=dict(record.get("metrics") or {}),
            checkpoint=checkpoint,
            trained_on=record.get("trained_on"),
            ic_series=_optional(directory, record.get("ic_series")),
            test_predictions=_optional(directory, record.get("test_predictions")),
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


def _optional(directory: Path, name: str | None) -> Path | None:
    """Return ``directory / name``, or None when no name is recorded."""
    return None if name is None else directory / name


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
    ``directory`` exist; NaN and inf metrics
    become null.

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
    ic_series, test_predictions = evaluation_paths(directory)
    write_json_atomically(
        directory / _RUN_FILE,
        to_jsonable(
            {
                "format_version": FORMAT_VERSION,
                "kind": "model",
                "checkpoint": Path(checkpoint).name,
                "train_window": list(train_window),
                "fitted_train_window": list(fitted_train_window),
                "test_window": list(test_window),
                "trained_on": trained_on,
                "metrics": metrics or {},
                "ic_series": ic_series.name if ic_series.exists() else None,
                "test_predictions": (
                    test_predictions.name if test_predictions.exists() else None
                ),
            }
        ),
        indent=2,
    )
