"""Walk-forward training: one walk-forward cross-validation of a model or an ensemble.

``train_walk_forward`` runs a ``train_cv`` call for anything satisfying the
public protocol ``WalkForwardTrainable``; ``BaseModel`` and ``BaseEnsemble``
are its two adapters, and their ``train_cv`` delegates here. It owns the
procedure, so the two cannot drift apart:

1. check the unit's hyperparameters, before any directory exists;
2. lay out the folds over the unit's bars with
   ``quantlab.model.split.walk_forward_folds`` and its purge length;
3. create the trial directory ``{class}_trial_{timestamp}/`` under the
   unit's ``model_save_dir`` and train fold i, in order, into ``fold_{i}/``
   (the unit applies the fold's dates, trains and restores its own dates);
4. average every ``train_*`` / ``val_*`` / ``test_*`` metric of the folds'
   trained runs (``cv_mean_metrics``), write the means to a
   ``{class}_cv_summary`` tracking run in the trial's group and record the
   walk-forward trained run (``quantlab.runs.trained_run``) with the unit's
   provenance.

The module imports no quantlab layer above ``quantlab.utils`` except the
trained-run module.

Examples
--------
>>> run = train_walk_forward(model.collect(), train_periods=20)
>>> run.kind, len(run.folds)
('walk_forward', 4)
"""

import dataclasses
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from loguru import logger

from quantlab.runs.trained_run import (
    TrainedRun,
    fold_directory,
    new_trial_directory,
    write_walk_forward_run,
)
from quantlab.model.split import Fold, walk_forward_folds

#: Prefixes of the metric keys averaged over folds, one per split.
METRIC_PREFIXES = ("train_", "val_", "test_")


@runtime_checkable
class WalkForwardTrainable(Protocol):
    """What ``train_walk_forward`` needs from the unit it trains; all public.

    Attributes
    ----------
    class_name : str
        Names the trial directory, the summary run and error messages.
    model_save_dir : Path
        The directory the trial directory is created in.
    purge_bars : int
        Bars each fold's training window loses (an ensemble's: the largest
        over its members).
    tracker : Tracker
        The tracker the CV summary run is opened through (an ensemble's:
        its first member's).
    tracking_project : str
        The default project of that run.

    Examples
    --------
    >>> isinstance(model, WalkForwardTrainable), isinstance(ensemble, WalkForwardTrainable)
    (True, True)
    """

    @property
    def class_name(self) -> str: ...

    @property
    def model_save_dir(self) -> Path: ...

    @property
    def purge_bars(self) -> int: ...

    @property
    def tracker(self) -> Any: ...

    @property
    def tracking_project(self) -> str: ...

    def walk_forward_bars(self) -> np.ndarray:
        """The collected bars between the unit's start and end dates, maybe none."""
        ...

    def check_hyperparameters(self) -> None:
        """Raise ``ValueError`` if a hyperparameter is invalid."""
        ...

    def train_fold(self, fold: Fold, run_dir: Path, group: str) -> None:
        """Train on ``fold``'s dates into ``run_dir``, then restore the unit's own dates."""
        ...

    def get_config(self) -> dict:
        """The unit's config, the summary run's config."""
        ...

    def provenance(self) -> dict:
        """What the walk-forward run records about the data and code trained on."""
        ...


def train_walk_forward(
    unit: WalkForwardTrainable,
    train_periods: int,
    expanding: bool = False,
    test_periods: int | None = None,
) -> TrainedRun:
    """Run a walk-forward cross-validation of ``unit`` and return the walk-forward run.

    See the module docstring for the procedure; the parameters are those
    of ``walk_forward_folds``.

    Parameters
    ----------
    unit : WalkForwardTrainable
        The collected model or ensemble.
    train_periods : int
        Number of bars in the first fold's training window, and in every
        fold's when sliding.
    expanding : bool, default False
        Train every fold from the first fold's start.
    test_periods : int, optional
        Number of bars in each fold's test window, and the step between
        folds. Defaults to ``train_periods // 5``.

    Returns
    -------
    TrainedRun
        The ``"walk_forward"`` run: its ``folds`` and ``cv_mean``.

    Raises
    ------
    ValueError
        If a hyperparameter is invalid, ``walk_forward_folds`` refuses the
        settings (checked first), or the unit has no bar in its date range;
        nothing is created on disk then.

    Examples
    --------
    >>> run = train_walk_forward(model.collect(), train_periods=20, test_periods=10)
    >>> len(run.folds), sorted(run.cv_mean)[:1]
    (2, ['cv_mean_test_ic'])
    """
    name = unit.class_name
    unit.check_hyperparameters()
    bars = unit.walk_forward_bars()
    try:
        folds = walk_forward_folds(
            bars,
            train_periods,
            test_periods=test_periods,
            expanding=expanding,
            purge_bars=unit.purge_bars,
        )
    except ValueError as error:
        raise ValueError(f"{name}: train_cv: {error}") from None
    if len(bars) == 0:
        raise ValueError(f"{name}: train_cv: No data found between its start and end dates")
    logger.info(
        f"{name}: {len(folds)} walk-forward folds with {train_periods} training periods"
    )
    for fold in folds:
        logger.info(
            f"Fold {fold.index}: Train [{fold.fitted_train_window[0]} to "
            f"{fold.fitted_train_window[1]}], Test [{fold.test_window[0]} "
            f"to {fold.test_window[1]}]"
        )

    trial = new_trial_directory(unit.model_save_dir, name)
    for fold in folds:
        unit.train_fold(fold, fold_directory(trial, fold.index), trial.name)

    indices = [fold.index for fold in folds]
    fold_runs = [TrainedRun.open(fold_directory(trial, i)) for i in indices]
    means = cv_mean_metrics([run.metrics for run in fold_runs])
    if means:
        with unit.tracker.start_run(
            project=unit.tracking_project,
            group=trial.name,
            name=f"{name}_cv_summary",
            config=unit.get_config(),
        ) as run:
            run.summarize(means)
    return write_walk_forward_run(
        trial, folds=indices, cv_mean=means, provenance=unit.provenance()
    )


def fold_config(config, fold: Fold):
    """Return ``config`` with ``fold``'s training window before the purge and its test window.

    A model given these dates purges the training window itself.

    Parameters
    ----------
    config : dataclass
        A model config with ``train_start``, ``train_end``, ``test_start``
        and ``test_end``.
    fold : Fold
        The fold whose dates to apply.

    Returns
    -------
    dataclass
        A copy of ``config``; ``config`` is not modified.

    Examples
    --------
    >>> fold = Fold(0, ("2024-01-01", "2024-01-10"), ("2024-01-01", "2024-01-08"),
    ...             ("2024-01-11", "2024-01-12"))
    >>> fold_config(model.config, fold).test_start
    '2024-01-11'
    """
    return dataclasses.replace(
        config,
        train_start=fold.train_window[0],
        train_end=fold.train_window[1],
        test_start=fold.test_window[0],
        test_end=fold.test_window[1],
    )


def cv_mean_metrics(results: list[dict]) -> dict:
    """Average every ``train_*`` / ``val_*`` / ``test_*`` metric over folds.

    Each mean is keyed ``cv_mean_{key}``. Only finite numeric values count
    (a recorded null is skipped); a metric with no finite value in any fold
    averages to NaN. ``cv_n_folds`` is added.

    Parameters
    ----------
    results : list[dict]
        Each fold's metrics.

    Returns
    -------
    dict
        The means and ``cv_n_folds``; empty when no fold carries a metric,
        in which case no summary run is opened.

    Examples
    --------
    >>> cv_mean_metrics([{"test_ic": 0.1}, {"test_ic": 0.3}, {"test_ic": None}])
    {'cv_mean_test_ic': 0.2, 'cv_n_folds': 3}
    """
    keys: list[str] = []
    for result in results:
        for key, value in result.items():
            # None is a metric recorded as null, undefined in that fold.
            numeric = value is None or (
                isinstance(value, (int, float, np.integer, np.floating))
                and not isinstance(value, bool)
            )
            if key.startswith(METRIC_PREFIXES) and numeric and key not in keys:
                keys.append(key)
    if not keys:
        return {}

    means: dict = {}
    for key in keys:
        finite = [
            float(r[key])
            for r in results
            if r.get(key) is not None and np.isfinite(float(r[key]))
        ]
        means[f"cv_mean_{key}"] = sum(finite) / len(finite) if finite else float("nan")
    means["cv_n_folds"] = len(results)
    return means
