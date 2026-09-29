"""Seed ensembles: one model config trained under several random seeds.

``SeedEnsemble(model, seeds)`` builds one member per seed, each a new
instance of the model's class on the model's config with ``random_seed``
replaced. The members read the same data, so they share one collected panel
and one feature request per prediction window; their predictions are
averaged after a per-bar cross-sectional z-score (``average_predictions``).
The ensemble satisfies the backtester's ``Predictor`` protocol, so it is
backtested, loaded and rebuilt from a run's ``config.json`` like a single
model.
"""

import dataclasses
from numbers import Integral
from typing import Self

import xarray as xr

from quantlab.model.ensemble import BaseEnsemble
from quantlab.utils.module import get_cls_from_path


class SeedEnsemble(BaseEnsemble):
    """An ensemble of one model trained under several random seeds.

    Member k is ``type(model)`` built on ``model.config`` with
    ``random_seed=seeds[k]``. ``collect()`` collects once, on the first
    member, and the other members share its data backend, so the panel is
    held once. ``predict_window`` requests features once, through the first
    member (its warm-up included), and hands them to every member's
    ``predict_panel``. ``train()``, ``load()`` and ``check_checkpoint()``
    work on an ``ensemble.json`` manifest (see ``BaseEnsemble``) that
    records each member's seed; ``train_cv()`` writes one such directory
    per walk-forward fold and a ``cv_folds.json`` that a backtester's
    ``run_cv`` replays.

    Parameters
    ----------
    model : BaseModel
        The model to replicate. It is not trained itself; its config is the
        template of every member.
    seeds : sequence of int
        At least two distinct seeds, one per member, in member order.

    Attributes
    ----------
    model : BaseModel
        The model given, whose config ``get_config`` records.
    seeds : tuple[int, ...]
        The members' seeds, in member order.
    members : list[BaseModel]
        One model per seed.

    Raises
    ------
    TypeError
        If a seed is not an integer.
    ValueError
        If fewer than two seeds are given or a seed repeats.

    Examples
    --------
    Given a model ``model`` over one factor and one forward-return label::

        >>> ensemble = SeedEnsemble(model, [0, 1, 2])
        >>> [member.config.random_seed for member in ensemble.members]
        [0, 1, 2]
        >>> manifest = ensemble.collect().train()
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars), out.sizes["timestamp"]
        (['fwd_ret_1'], 21)
    """

    def __init__(self, model, seeds):
        """Initialize the ensemble; see the class docstring for parameters."""
        self.seeds = self._checked_seeds(seeds)
        self.model = model
        super().__init__(
            [
                type(model)(dataclasses.replace(model.config, random_seed=seed))
                for seed in self.seeds
            ]
        )

    @staticmethod
    def _checked_seeds(seeds) -> tuple[int, ...]:
        """Return ``seeds`` as a tuple of ints after checking count and uniqueness."""
        seeds = list(seeds)
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, Integral):
                raise TypeError(
                    f"SeedEnsemble seeds must be integers, got {seed!r} "
                    f"({type(seed).__name__})"
                )
        seeds = [int(seed) for seed in seeds]
        if len(seeds) < 2:
            raise ValueError(
                f"SeedEnsemble needs at least two seeds, got {seeds}"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"SeedEnsemble seeds must be distinct, got {seeds}")
        return tuple(seeds)

    def _member_seed(self, k: int) -> int:
        """The seed of member ``k``, recorded in the manifest."""
        return self.seeds[k]

    def collect(self) -> Self:
        """Collect the panel once and share it with every member.

        The first member collects (see ``BaseModel.collect``); every other
        member's ``data_backend`` becomes that same object.

        Returns
        -------
        Self
            The ensemble itself, for chaining.

        Examples
        --------
        >>> ensemble.collect() is ensemble
        True
        >>> ensemble.members[1].data_backend is ensemble.members[0].data_backend
        True
        """
        first = self.members[0]
        first.collect()
        for member in self.members[1:]:
            member.data_backend = first.data_backend
        return self

    def _member_predictions(self, start, end) -> list[xr.Dataset]:
        """Request the features once and predict them with every member."""
        features = self.members[0]._collect_all_features(start, end)
        return [
            member.predict_panel(features).sel(timestamp=slice(start, end))
            for member in self.members
        ]

    def _member_panel_predictions(self) -> list[xr.Dataset]:
        """Predict the one shared collected panel with every member."""
        data = self.members[0].data_backend.get_xarray_dataset(["timestamp", "symbol"])
        return [member.predict_panel(data) for member in self.members]

    def fingerprint_inputs(self, start, end) -> list[tuple]:
        """Return the data ``predict_window(start, end)`` reads, for fingerprinting.

        The first member's entries, unchanged: every member reads the same
        features, which are requested once, so the keys are those of a
        single model.

        Parameters
        ----------
        start, end : str
            The window passed to ``predict_window``.

        Returns
        -------
        list[tuple]
            ``(key, factor, strategy, first, last)`` entries in factor order.

        Examples
        --------
        >>> [key for key, *_ in ensemble.fingerprint_inputs("2024-02-12", "2024-03-11")]
        ['factor[0]:PastReturnFactor']
        """
        return self.members[0].fingerprint_inputs(start, end)

    def training_fingerprint_inputs(self) -> list[tuple]:
        """Return the data ``collect()`` reads, for fingerprinting.

        The first member's entries, unchanged: it is the only member that
        collects.

        Returns
        -------
        list[tuple]
            ``(key, item, strategy, first, last)`` entries, factors first.

        Examples
        --------
        >>> [key for key, *_ in ensemble.training_fingerprint_inputs()]
        ['train_factor[0]:PastReturnFactor', 'train_label[0]:ForwardReturnLabel']
        """
        return self.members[0].training_fingerprint_inputs()

    def get_config(self) -> dict:
        """Return the wrapped model's config and the seeds as a JSON-ready dict.

        Returns
        -------
        dict
            ``{"name": ..., "seeds": [...], "model": model.get_config()}``.

        Examples
        --------
        >>> config = SeedEnsemble(model, [0, 1]).get_config()
        >>> config["name"], config["seeds"], config["model"]["name"]
        ('quantlab.model.predefined.seed_ensemble.SeedEnsemble', [0, 1], 'tests.backtest_fixtures.SeededHead')
        """
        return {
            "name": self.import_path,
            "seeds": list(self.seeds),
            "model": self.model.get_config(),
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild a seed ensemble from the dict ``get_config()`` returned.

        The wrapped model is rebuilt by ``from_config`` of the class its
        config names, and the seeds are applied to it.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, for example read back from a
            backtest run's ``config.json``.

        Returns
        -------
        Self
            An untrained ensemble; call ``load`` to restore a manifest.

        Examples
        --------
        >>> rebuilt = SeedEnsemble.from_config(ensemble.get_config())
        >>> rebuilt.seeds, type(rebuilt.members[0]).__name__
        ((0, 1, 2), 'SeededHead')
        """
        model_config = config["model"]
        model = get_cls_from_path(model_config["name"]).from_config(model_config)
        return cls(model, config["seeds"])
