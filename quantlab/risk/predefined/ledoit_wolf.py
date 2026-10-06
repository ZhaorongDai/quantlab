"""The Ledoit-Wolf shrunk covariance of a window of returns, as a function.

``ledoit_wolf_covariance`` is the estimate alone: a window of one-bar
returns in, a covariance out, with no context, coverage rule or store. The
portfolio layer's ``LedoitWolfEstimator`` decides which symbols a bar covers
and calls it on their window; anything else that needs the same estimate
(the bias statistics of a backtest's holdings, say) calls it directly.
"""

import numpy as np
from sklearn.covariance import ledoit_wolf


def ledoit_wolf_covariance(
    returns: np.ndarray, volatility: np.ndarray | None = None
) -> np.ndarray:
    """Return the Ledoit-Wolf shrunk covariance of ``returns``, its volatilities optionally replaced.

    The sample covariance of the demeaned window (divided by ``T``) is
    shrunk toward a scaled identity with the Ledoit-Wolf (2004) coefficient
    (``sklearn.covariance.ledoit_wolf``), so the estimate stays well
    conditioned when there are more symbols than bars. It is then split into
    correlations ``C`` and volatilities ``D`` and put back together as ``D C
    D``, with ``D`` the given ``volatility`` when there is one, else the
    shrunk covariance's own.

    Parameters
    ----------
    returns : np.ndarray
        ``[T, N]`` one-bar returns, every value finite, ``T`` at least 2,
        no column constant.
    volatility : np.ndarray, optional
        ``[N]`` one-bar volatilities, every one finite and positive.

    Returns
    -------
    np.ndarray
        ``[N, N]``: the covariance of one-bar returns, symmetric and
        positive definite.

    Raises
    ------
    ValueError
        If ``returns`` is not two-dimensional with at least two bars, holds
        a value that is not finite or a constant column, or ``volatility``
        does not have one finite positive value per column.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> window = rng.normal(0.0, [0.01, 0.02, 0.03], size=(250, 3))
    >>> np.sqrt(np.diag(ledoit_wolf_covariance(window))).round(3)
    array([0.011, 0.02 , 0.029])
    >>> covariance = ledoit_wolf_covariance(window, volatility=np.array([0.02, 0.02, 0.02]))
    >>> np.sqrt(np.diag(covariance)).round(6)
    array([0.02, 0.02, 0.02])
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2 or len(returns) < 2:
        raise ValueError(
            f"ledoit_wolf_covariance(): returns must be [T, N] with T at least 2, got "
            f"shape {returns.shape}."
        )
    if not np.isfinite(returns).all():
        raise ValueError("ledoit_wolf_covariance(): every return must be finite.")
    if (returns.max(axis=0) == returns.min(axis=0)).any():
        raise ValueError(
            "ledoit_wolf_covariance(): a constant column has no variance to correlate by."
        )
    shrunk, _ = ledoit_wolf(returns)
    sd = np.sqrt(np.diag(shrunk))
    correlation = shrunk / np.outer(sd, sd)
    if volatility is None:
        return correlation * np.outer(sd, sd)
    volatility = np.asarray(volatility, dtype=np.float64)
    if volatility.shape != (returns.shape[1],) or not (
        np.isfinite(volatility).all() and (volatility > 0).all()
    ):
        raise ValueError(
            f"ledoit_wolf_covariance(): volatility must hold {returns.shape[1]} finite "
            f"positive values, one per column, got {volatility!r}."
        )
    return correlation * np.outer(volatility, volatility)


__all__ = ["ledoit_wolf_covariance"]
