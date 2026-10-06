"""A covariance estimator reading a factor risk model's stores: ``B F B' + diag(D)`` at each bar.

``FactorRiskStoreEstimator`` estimates nothing itself (ADR 0024). At a bar it reads
the factor covariance ``F`` and the specific risks of that bar from the
estimate store of a ``quantlab.risk.base.FactorRiskModel``, builds each
symbol's exposures ``B`` from the bar's values of the model's exposures
factor (``context.factors``) and returns a ``FactorCovarianceEstimate``,
which the mean-variance optimiser turns into a low-rank risk term.
"""

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.portfolio.base import FactorCovarianceEstimate, PortfolioContext, CovarianceEstimator
from quantlab.portfolio.config import FactorRiskStoreEstimatorConfig


class FactorRiskStoreEstimator(CovarianceEstimator):
    """The covariance at a bar from a factor risk model's estimate store.

    ``required_factors()`` is the factor risk model's exposures factor, so the
    backtest computes it over its window and hands its values at the bar in
    ``context.factors``. ``estimate(context)`` reads the bar's row of the
    estimate store and returns ``B F B' + diag(D)`` in factor form, with
    ``B`` on the model's factors (``FactorRiskModel.factor_names``): 1 on the
    country factor, 1 on the symbol's industry, its style exposures. The
    estimate is of one-bar returns.

    A symbol is covered when it has every style exposure, an industry among
    the model's (when the model has industries) and a specific risk at the
    bar. A factor whose variance is not known at the bar (an industry with
    too few observations) is left out of ``F``, and so is every symbol
    exposed to it. There is no staleness filter: the model reads no return
    window, and a locked position is priced like any other.

    Parameters
    ----------
    config : FactorRiskStoreEstimatorConfig
        The factor risk model.

    Examples
    --------
    With ``use4`` a ``Use4RiskModel`` whose stores are built and ``context``
    a backtest's context at a bar inside them:

    >>> risk = FactorRiskStoreEstimator(FactorRiskStoreEstimatorConfig(risk_model=use4))
    >>> risk.required_factors() == [use4.config.exposures]
    True
    >>> estimate = risk.estimate(context)
    >>> exposures, factor_covariance, specific_variance = estimate.factor_form()
    >>> exposures.shape[1] == factor_covariance.shape[0]
    True
    """

    config_cls = FactorRiskStoreEstimatorConfig

    def required_factors(self) -> list:
        """The factor risk model's exposures factor.

        Examples
        --------
        >>> risk.required_factors() == [use4.config.exposures]
        True
        """
        return [self.config.risk_model.config.exposures]

    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> FactorCovarianceEstimate:
        """Return ``B F B' + diag(D)`` at the context's bar over the covered symbols.

        See the class docstring for the coverage.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; its ``factors`` hold the exposures at the bar.
        volatility : None
            Refused: a factor risk model takes no predicted volatilities.

        Returns
        -------
        FactorCovarianceEstimate
            The estimate in factor form, of one-bar returns.

        Raises
        ------
        ValueError
            If ``volatility`` is given, the context has no ``factors`` or the
            estimate store does not cover the bar.

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
        if context.factors is None:
            raise ValueError(
                f"{type(self).__name__}: the context has no factors; the exposures come "
                f"from required_factors()."
            )
        row = self._row(context.timestamp)
        model = self.config.risk_model
        names = list(model.factor_names)
        symbols = context.symbols
        exposures, covered = model.exposure_matrix(context.factors.reindex(symbol=symbols))

        specific = row["specific_risk"].reindex(symbol=symbols).values.astype(np.float64)
        covered &= np.isfinite(specific)
        covariance = row["factor_covariance"].sel(factor_i=names, factor_j=names).values
        kept = self._finite_factors(covariance)
        # A symbol exposed to a factor without a covariance is not covered.
        covered &= ~(exposures[:, ~kept] != 0).any(axis=1)
        index = np.flatnonzero(covered)
        return FactorCovarianceEstimate(
            symbols=symbols[index],
            exposures=np.nan_to_num(exposures[np.ix_(index, np.flatnonzero(kept))]),
            factor_covariance=covariance[np.ix_(kept, kept)],
            specific_variance=specific[index] ** 2,
        )

    @staticmethod
    def _finite_factors(covariance: np.ndarray) -> np.ndarray:
        """Return the factors kept, a set whose covariance is all finite.

        A factor without a variance is dropped first; then, while a pair has
        no correlation, the factor missing the most pairs (a greedy choice).
        """
        kept = np.isfinite(np.diag(covariance))
        while True:
            missing = ~np.isfinite(covariance) & kept[:, None] & kept[None, :]
            if not missing.any():
                return kept
            kept[np.argmax(missing.sum(axis=1))] = False

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
