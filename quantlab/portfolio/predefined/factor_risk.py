"""A covariance estimator reading a factor risk model's stores: ``B F B' + diag(D)`` at each bar.

``FactorRiskStoreEstimator`` estimates nothing itself (ADR 0024). At a bar it reads
the factor covariance ``F`` and the specific risks of that bar from the
estimate store of a ``quantlab.risk.base.FactorRiskModel``, builds each
symbol's exposures ``B`` from the bar's exposures as the model gives them
(``context.risk_exposures``) and returns the model's forecast
(``FactorRiskModel.forecast``, a ``FactorRiskForecast``), which the
mean-variance optimiser turns into a low-rank risk term.
"""

import pandas as pd
import xarray as xr

from quantlab.portfolio.base import CovarianceEstimator, PortfolioContext
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig
from quantlab.risk.base import FactorRiskForecast, FactorRiskModel


class FactorRiskStoreEstimator(CovarianceEstimator):
    """The covariance at a bar from a factor risk model's estimate store.

    ``required_risk_model()`` is the factor risk model, so the decision
    inputs take its exposures from it (``FactorRiskModel.exposures``: the
    exposures factor's store under ``exposure_data_strategy="read"``,
    computed under ``"cal"``), exactly as its stores, factor attribution and
    bias statistics do, and hand their values at the bar in
    ``context.risk_exposures``. ``estimate(context)`` reads the bar's row
    of the estimate store and returns the model's forecast from it and those
    exposures (``FactorRiskModel.forecast``): ``B F B' + diag(D)`` of
    one-bar returns, in factor form, over the symbols the model covers at
    the bar (coverage is the model's rule, shared with factor attribution
    and the bias statistics). It reads only what every factor risk model
    provides, so it works with any. There is no staleness filter: the model
    reads no return window, and a locked position is priced like any
    other.

    Parameters
    ----------
    config : FactorRiskStoreEstimatorConfig
        The factor risk model.

    Examples
    --------
    With ``use4`` a ``Use4RiskModel`` whose stores are built and ``context``
    a backtest's context at a bar inside them:

    >>> risk = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4))
    >>> risk.required_risk_model() is use4
    True
    >>> estimate = risk.estimate(context)
    >>> exposures, factor_covariance, specific_variance = estimate.factor_form()
    >>> exposures.shape[1] == factor_covariance.shape[0]
    True
    """

    config_cls = FactorRiskStoreEstimatorConfig

    def required_risk_model(self) -> FactorRiskModel:
        """The factor risk model, whose exposures ``estimate`` reads from ``context.risk_exposures``.

        Examples
        --------
        >>> risk.required_risk_model() is use4
        True
        """
        return self.config.risk_model

    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> FactorRiskForecast:
        """Return ``B F B' + diag(D)`` at the context's bar over the covered symbols.

        See the class docstring for the coverage.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; its ``risk_exposures`` hold the model's
            exposures at the bar.
        volatility : None
            Refused: a factor risk model takes no predicted volatilities.

        Returns
        -------
        FactorRiskForecast
            The model's forecast at the bar over the context's symbols it
            covers, of one-bar returns.

        Raises
        ------
        ValueError
            If ``volatility`` is given, the context has no
            ``risk_exposures`` or the estimate store does not cover the bar.

        Examples
        --------
        >>> estimate = risk.estimate(context)
        >>> estimate.exposures.shape[1] == estimate.factor_covariance.shape[0]
        True
        """
        if volatility is not None:
            raise ValueError(
                f"{type(self).__name__} takes no predicted volatilities: its risk is the "
                f"factor risk model's; leave the optimiser's volatility_label unset."
            )
        if context.risk_exposures is None:
            raise ValueError(
                f"{type(self).__name__}: the context has no risk exposures; the decision "
                f"inputs take them from required_risk_model()."
            )
        model = self.config.risk_model
        return model.forecast(
            self._row(context.timestamp), context.risk_exposures.reindex(symbol=context.symbols)
        )

    def _row(self, timestamp) -> xr.Dataset:
        """Return the estimate store's row at ``timestamp``.

        Read through ``RiskStore.read`` on every call, so a store built or
        extended since the last bar is seen.

        Raises
        ------
        ValueError
            If the store has no row at ``timestamp``.
        """
        store = self.config.risk_model.estimate
        bar = pd.Timestamp(timestamp)
        rows = store.read(bar, bar)
        if rows.sizes["timestamp"] != 1:
            raise ValueError(
                f"{type(self).__name__}: the estimate store at {store.path} has no row "
                f"at {bar}."
            )
        return rows.isel(timestamp=0).load()
