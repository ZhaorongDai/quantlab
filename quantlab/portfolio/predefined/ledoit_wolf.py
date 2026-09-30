"""A Ledoit-Wolf shrunk sample covariance of trailing one-bar returns.

``LedoitWolfRiskModel`` estimates the covariance of one-bar returns at a bar
from the ``lookback_bars`` returns ending there, shrunk toward a scaled
identity with the Ledoit-Wolf coefficient, so the estimate stays well
conditioned when there are more symbols than bars. The shrunk covariance is
split into correlations and volatilities, and the volatilities can be
replaced by given ones (a model's forecast) before it is put back together.
"""

import numpy as np
import xarray as xr
from sklearn.covariance import ledoit_wolf

from quantlab.base.config import LedoitWolfConfig
from quantlab.base.portfolio import CovarianceEstimate, PortfolioContext, RiskModel


class LedoitWolfRiskModel(RiskModel):
    """Ledoit-Wolf shrunk covariance of the trailing one-bar returns.

    A symbol is covered when every one of the ``lookback_bars`` returns in
    the context's window is finite; the others have too little history
    and are left out of the estimate. The covered symbols' sample
    covariance is shrunk (``sklearn.covariance.ledoit_wolf``), converted to
    correlations ``C`` and scaled back by volatilities ``D``: the given
    ones where ``volatility`` is passed (a symbol without a finite positive
    one is left out), else the shrunk covariance's own. The estimate is
    ``D C D``, of one-bar returns; ``factor_form()`` is ``None``.

    Parameters
    ----------
    config : LedoitWolfConfig
        ``lookback_bars``, at least 2.

    Raises
    ------
    ValueError
        If ``lookback_bars`` is below 2.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> from quantlab.base.config import LedoitWolfConfig
    >>> from quantlab.base.portfolio import PortfolioContext
    >>> rng = np.random.default_rng(0)
    >>> symbols = ["AAA", "BBB", "CCC"]
    >>> window = rng.normal(0.0, 0.01, size=(60, 3))
    >>> window[:5, 2] = np.nan  # CCC listed five bars into the window
    >>> context = PortfolioContext(
    ...     timestamp=pd.Timestamp("2024-03-25"),
    ...     predictions=xr.Dataset(coords={"symbol": symbols}),
    ...     eligible=xr.DataArray([True] * 3, dims="symbol", coords={"symbol": symbols}),
    ...     current_weights=xr.DataArray(np.zeros(3), dims="symbol", coords={"symbol": symbols}),
    ...     returns=xr.DataArray(window, dims=("timestamp", "symbol"), coords={
    ...         "timestamp": pd.bdate_range("2024-01-01", periods=60), "symbol": symbols}),
    ... )
    >>> risk = LedoitWolfRiskModel(LedoitWolfConfig(lookback_bars=60))
    >>> estimate = risk.estimate(context)
    >>> estimate.symbols.tolist(), estimate.covariance.shape
    (['AAA', 'BBB'], (2, 2))
    >>> given = xr.DataArray([0.02, 0.03, 0.04], dims="symbol", coords={"symbol": symbols})
    >>> np.sqrt(risk.estimate(context, volatility=given).variance).round(6)
    array([0.02, 0.03])
    """

    config_cls = LedoitWolfConfig

    def __init__(self, config: LedoitWolfConfig):
        """Initialize the risk model; see the class docstring for parameters."""
        super().__init__(config)
        if config.lookback_bars < 2:
            raise ValueError(
                f"lookback_bars must be >= 2 to estimate a covariance, got "
                f"{config.lookback_bars}"
            )

    def estimate(
        self, context: PortfolioContext, volatility: xr.DataArray | None = None
    ) -> CovarianceEstimate:
        """Estimate the covariance of one-bar returns from the context's return window.

        See the class docstring for the rule and an example.

        Parameters
        ----------
        context : PortfolioContext
            The bar's context; the last ``lookback_bars`` rows of its
            ``returns`` window are the history.
        volatility : xr.DataArray, optional
            Per-symbol one-bar volatilities on ``symbol``, replacing the
            historical ones.

        Returns
        -------
        CovarianceEstimate
            The covariance over the covered symbols, symmetric and positive
            definite.

        Examples
        --------
        >>> estimate = risk.estimate(context)
        >>> bool(np.allclose(estimate.covariance, estimate.covariance.T))
        True
        >>> bool((np.linalg.eigvalsh(estimate.covariance) > 0).all())
        True
        """
        symbols = context.symbols
        if context.returns is None:
            window = np.empty((0, len(symbols)))
        else:
            window = np.asarray(
                context.returns.sel(symbol=symbols).values, dtype=np.float64
            )[-self.config.lookback_bars :]
        covered = (len(window) == self.config.lookback_bars) & np.isfinite(window).all(axis=0)
        if volatility is not None:
            given = np.asarray(volatility.sel(symbol=symbols).values, dtype=np.float64)
            covered &= np.isfinite(given) & (given > 0)
        index = np.flatnonzero(covered)
        if index.size == 0:
            return CovarianceEstimate(symbols=symbols[:0], covariance=np.empty((0, 0)))

        shrunk, _ = ledoit_wolf(window[:, index])
        sd = np.sqrt(np.diag(shrunk))
        correlation = shrunk / np.outer(sd, sd)
        scale = given[index] if volatility is not None else sd
        covariance = correlation * np.outer(scale, scale)
        return CovarianceEstimate(symbols=symbols[index], covariance=covariance)
