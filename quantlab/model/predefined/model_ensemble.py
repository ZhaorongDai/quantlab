"""Model ensembles: models of any classes over any factors, combined into one prediction.

``ModelEnsemble(members)`` takes the member models as given, for example an
XGBoost regressor and a GATs network over different factors. Every member
collects its own data and requests its own features; their predictions are
combined label by label: a label several members predict is averaged after a
per-bar cross-sectional z-score (``average_predictions``), a label one member
predicts is passed through, so a return model and a volatility model make one
predictor. A label several members predict must have one config.
The ensemble satisfies the backtester's ``Predictor`` protocol, so it is
backtested, loaded and rebuilt from a run's ``config.json`` like a single
model.
"""

import dataclasses
from typing import Self

from quantlab.core.component import component
from quantlab.model.ensemble import BaseEnsemble


@dataclasses.dataclass(frozen=True)
class ModelEnsembleConfig:
    """What a ``ModelEnsemble`` is rebuilt from: its member models.

    Examples
    --------
    >>> len(ModelEnsemble([xgb, gats]).config.members)
    2
    """

    #: The member models, in order.
    members: list = component(many=True)


class ModelEnsemble(BaseEnsemble):
    """An ensemble of models given one by one, of any classes and over any factors.

    Every hook keeps ``BaseEnsemble``'s default: each member collects and predicts
    on its own, and the predictions are combined label by
    label (averaged when shared, passed through otherwise). Subclass it and override ``_combine`` for another
    combination rule. ``train()`` writes an ensemble unit whose ``run.json``
    records a null seed per member; ``train_cv()`` writes one such unit per
    walk-forward fold.

    Parameters
    ----------
    members : sequence of BaseModel
        At least two untrained or trained models. A label two members
        predict must have the same config in both, and their test windows
        must overlap.

    Attributes
    ----------
    members : list[BaseModel]
        The models given, in order.

    Raises
    ------
    ValueError
        If fewer than two members are given, two members predict a label of
        one name with different configs, or the test windows do not overlap.

    Examples
    --------
    Given two models ``xgb`` and ``gats`` over different factors and the same
    forward-return label::

        >>> ensemble = ModelEnsemble([xgb, gats])
        >>> checkpoint = ensemble.collect().train()
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars)
        ['fwd_ret_1']
    """

    #: The config dataclass the ensemble is serialised and rebuilt with.
    config_cls = ModelEnsembleConfig

    @property
    def config(self) -> ModelEnsembleConfig:
        """The member models; ``get_config()`` serialises it.

        Examples
        --------
        >>> config = ModelEnsemble([xgb, gats]).get_config()
        >>> config["name"], [m["name"] for m in config["members"]]
        ('quantlab.model.predefined.model_ensemble.ModelEnsemble', ['quantlab.model.predefined.xgb.XGBoostRegressor', 'quantlab.model.predefined.gats.GATsRegressor'])
        """
        return ModelEnsembleConfig(members=list(self.members))

    @classmethod
    def from_config(cls, config: dict, run_dir=None) -> Self:
        """Rebuild a model ensemble from the dict ``get_config()`` returned.

        Each member is rebuilt by the component rule.

        Parameters
        ----------
        config : dict
            The dict ``get_config()`` returned, for example read back from a
            backtest run's ``config.json``.
        run_dir : str or os.PathLike, optional
            The run directory the config was read from.

        Returns
        -------
        Self
            An untrained ensemble; call ``load`` to restore a trained unit.

        Examples
        --------
        >>> rebuilt = ModelEnsemble.from_config(ensemble.get_config())
        >>> [type(m).__name__ for m in rebuilt.members]
        ['XGBoostRegressor', 'GATsRegressor']
        """
        return cls(cls._rebuilt_fields(config, run_dir)["members"])
