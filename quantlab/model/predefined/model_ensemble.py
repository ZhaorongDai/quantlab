"""Model ensembles: models of any classes over any factors, combined into one prediction.

``ModelEnsemble(members)`` takes the member models as given, for example an
XGBoost regressor and a GATs network over different factors. Every member
collects its own data and requests its own features; their predictions are
averaged after a per-bar cross-sectional z-score (``average_predictions``).
The members must share their labels and their training and test windows.
The ensemble satisfies the backtester's ``Predictor`` protocol, so it is
backtested, loaded and rebuilt from a run's ``config.json`` like a single
model.
"""

from typing import Self

from quantlab.model.ensemble import BaseEnsemble
from quantlab.utils.module import get_cls_from_path


class ModelEnsemble(BaseEnsemble):
    """An ensemble of models given one by one, of any classes and over any factors.

    Every hook keeps ``BaseEnsemble``'s default: each member collects, predicts
    and is fingerprinted on its own, and the predictions are combined by
    ``average_predictions``. Subclass it and override ``_combine`` for another
    combination rule. ``train()`` writes an ``ensemble.json`` manifest with a
    null seed per member; ``train_cv()`` writes one such directory per
    walk-forward fold.

    Parameters
    ----------
    members : sequence of BaseModel
        At least two untrained or trained models whose label configs and
        training and test windows are identical.

    Attributes
    ----------
    members : list[BaseModel]
        The models given, in order.

    Raises
    ------
    ValueError
        If fewer than two members are given, or two members differ in label
        configs or in training or test window.

    Examples
    --------
    Given two models ``xgb`` and ``gats`` over different factors and the same
    forward-return label::

        >>> ensemble = ModelEnsemble([xgb, gats])
        >>> manifest = ensemble.collect().train()
        >>> out = ensemble.predict_window("2024-02-12", "2024-03-11")
        >>> list(out.data_vars)
        ['fwd_ret_1']
    """

    def get_config(self) -> dict:
        """Return every member's config as a JSON-ready dict.

        Returns
        -------
        dict
            ``{"name": ..., "members": [member.get_config(), ...]}``.

        Examples
        --------
        >>> config = ModelEnsemble([xgb, gats]).get_config()
        >>> config["name"], [m["name"] for m in config["members"]]
        ('quantlab.model.predefined.model_ensemble.ModelEnsemble', ['quantlab.model.predefined.xgb.XGBoostRegressor', 'quantlab.model.predefined.gats.GATsRegressor'])
        """
        return {
            "name": self.import_path,
            "members": [member.get_config() for member in self.members],
        }

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """Rebuild a model ensemble from the dict ``get_config()`` returned.

        Each member is rebuilt by ``from_config`` of the class its config names.

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
        >>> rebuilt = ModelEnsemble.from_config(ensemble.get_config())
        >>> [type(m).__name__ for m in rebuilt.members]
        ['XGBoostRegressor', 'GATsRegressor']
        """
        return cls(
            [
                get_cls_from_path(member["name"]).from_config(member)
                for member in config["members"]
            ]
        )
