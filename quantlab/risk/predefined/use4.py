"""USE4-style factor risk model: the factor set, the regression and the estimate chain.

Follows Menchero, Orr and Wang, *The Barra US Equity Model (USE4),
Methodology Notes* (MSCI, 2011), §3-5 and Appendix B: one country factor,
one factor per industry (the point-in-time Fama-French 48 industries) and
the 12 USE4 style factors, 61 factors, by default on the ``BarraStyle``
outputs; any factor's outputs can stand in through ``Use4RiskConfig``. Its
regression store holds, for every bar ``t``, the cross-sectional regression
of the excess returns of ``t`` on the exposures of ``t-1``:

- the excess return of a symbol is its adjusted close of ``t`` over that of
  ``t-1``, less one, less the risk-free rate of ``t-1`` (the rate of a day is
  published the next business day; ``BarraStyle`` lags it the same way);
- exposures: 1 on the country factor, 1 on the symbol's industry and 0 on
  every other industry, and its style exposures;
- weighted least squares over the estimation universe of ``t-1`` (every
  symbol without one), weighted by ``config.weighting`` of the market cap of
  ``t-1``; with both a country factor and industries, subject to the
  cap-weighted industry factor returns summing to 0, which removes their
  collinearity and makes the country factor the cap-weighted market (USE4
  Methodology Notes eq. 3.3);
- an industry with fewer than ``min_industry_members`` fitted members is left
  out of the bar: it has no factor return that bar, its members are not in
  the fit, and their specific returns carry no industry term (our choice);
- a fitted return more than ``return_outlier_sigma`` robust standard
  deviations (1.4826 times the median absolute deviation) from the
  cross-sectional median is trimmed to that bound for the fit only. A robust
  bound does not move when the outlier grows, so a vendor price error cannot
  move a factor return (our choice);
- the specific return of every symbol with all exposures and a return, fitted
  or not, is its untrimmed excess return less the fitted factor part.

"""

import dataclasses
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed

from quantlab.dataset.base import InsufficientHistoryError
from quantlab.dataset.config import FF48_INDUSTRY_NAMES
from quantlab.risk.base import FactorRiskModel, covered_factors
from quantlab.risk.config import REGRESSION_WEIGHTINGS, Use4RiskConfig
from quantlab.utils.date_range import check_range
from quantlab.utils.returns import one_bar_returns
from quantlab.utils.symbol_axis import sort_symbol_axis
from quantlab.utils.timer import Timer

#: Scales a median absolute deviation to the standard deviation of a normal.
MAD_TO_SIGMA = 1.4826


def _exponential_weights(length: int, half_life: float) -> np.ndarray:
    """Return ``length`` weights halving every ``half_life`` bars, the last one 1."""
    return 0.5 ** (np.arange(length - 1, -1, -1) / half_life)


def _pairwise_moments(
    window: np.ndarray, half_life: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exponentially weighted moments of each pair of columns over their common bars.

    ``window`` is ``[L, K]`` with NaN where a value is missing, the last row
    the latest. For each pair ``(i, j)`` the weights are those of the bars
    where both are present. Returns the covariance, the variance of ``i``
    and of ``j`` over those bars (``[K, K]`` each, about the weighted means,
    normalised by the sum of the weights) and the count of those bars.
    """
    weights = _exponential_weights(len(window), half_life)[:, None]
    present = np.isfinite(window).astype(np.float64)
    # Moments do not change with a shift; centring each column on its own
    # weighted mean first keeps the one-pass formulas below from cancelling.
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = np.nansum(window * weights, axis=0) / (present * weights).sum(axis=0)
    values = np.where(present > 0, window - np.nan_to_num(centre), 0.0)
    weighted = values * weights
    total = (present * weights).T @ present
    sums = weighted.T @ present  # [i, j]: sum of w x_i over the bars with x_j
    squares = (weighted * values).T @ present
    products = weighted.T @ values
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_i, mean_j = sums / total, sums.T / total
        covariance = products / total - mean_i * mean_j
        variance_i = squares / total - mean_i**2
        variance_j = squares.T / total - mean_j**2
    return covariance, variance_i, variance_j, present.T @ present


def _bartlett_weights(lags: int) -> np.ndarray:
    """Return the Newey-West (1987) weights ``1 - l / (lags + 1)`` of lags 1 to ``lags``."""
    return 1.0 - np.arange(1, lags + 1) / (lags + 1)


def _lagged_moments(
    window: np.ndarray, half_life: float, lags: int, least: int
) -> np.ndarray:
    """Exponentially weighted lagged covariances of each pair of columns.

    ``window`` is ``[L, K]`` with NaN where a value is missing, the last row
    the latest. Returns ``[lags, K, K]``: entry ``[l - 1, i, j]`` is the
    weighted mean of ``x_i(t) x_j(t - l)`` over the bars ``t`` where both are
    present, each column taken about its weighted mean over the window, each
    product weighted by the weight of ``t``. A lag with fewer than ``least``
    such bars is 0.
    """
    weights = _exponential_weights(len(window), half_life)[:, None]
    present = np.isfinite(window).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = np.nansum(window * weights, axis=0) / (present * weights).sum(axis=0)
    values = np.where(present > 0, window - np.nan_to_num(centre), 0.0)
    moments = np.zeros((lags, window.shape[1], window.shape[1]))
    for lag in range(1, min(lags, len(window) - 1) + 1):
        later = weights[lag:]
        total = (present[lag:] * later).T @ present[:-lag]
        products = (values[lag:] * later).T @ values[:-lag]
        count = present[lag:].T @ present[:-lag]
        with np.errstate(divide="ignore", invalid="ignore"):
            moments[lag - 1] = np.where(count >= least, products / total, 0.0)
    return moments


def _newey_west_multiplier(
    window: np.ndarray, half_life: float, lags: int, least: int
) -> np.ndarray:
    """Return each column's Newey-West multiplier ``C_NW`` over ``window``.

    ``C_NW = 1 + 2 sum_l (1 - l / (L + 1)) rho_l`` over lags 1 to ``L`` (USE4
    eq. 5.2), ``rho_l`` the autocorrelation at lag ``l`` of the column's
    returns weighted with ``half_life`` (the weight of the later bar of each
    pair), each column about its weighted mean over the window. A lag with
    fewer than ``least`` pairs, or a column without a variance, adds
    nothing; a negative multiplier is 0 (our choice). ``window`` is ``[L,
    N]`` with NaN where a return is missing, the last row the latest.
    """
    multiplier = np.ones(window.shape[1])
    # Only columns with a return in the window can be adjusted.
    active = np.flatnonzero(np.isfinite(window).any(axis=0))
    window = np.ascontiguousarray(window[:, active])
    weights = _exponential_weights(len(window), half_life)
    present = np.isfinite(window)
    mask = present.astype(np.float64)
    total = weights @ mask
    with np.errstate(divide="ignore", invalid="ignore"):
        centre = (weights @ np.where(present, window, 0.0)) / total
        values = np.where(present, window - np.nan_to_num(centre), 0.0)
        variance = np.einsum("t,ts,ts->s", weights, values, values) / total
    adjustment = np.ones(window.shape[1])
    bartlett = _bartlett_weights(lags)
    for lag in range(1, min(lags, len(window) - 1) + 1):
        later = weights[lag:]
        pairs = np.einsum("ts,ts->s", mask[lag:], mask[:-lag])
        with np.errstate(divide="ignore", invalid="ignore"):
            autocovariance = np.einsum(
                "t,ts,ts->s", later, values[lag:], values[:-lag]
            ) / np.einsum("t,ts,ts->s", later, mask[lag:], mask[:-lag])
            rho = autocovariance / variance
        usable = (pairs >= least) & np.isfinite(rho) & (variance > 0)
        adjustment += 2.0 * bartlett[lag - 1] * np.where(usable, rho, 0.0)
    multiplier[active] = np.clip(adjustment, 0.0, None)
    return multiplier


def _excess_returns(price: np.ndarray, risk_free: np.ndarray) -> np.ndarray:
    """Return ``[T, S]`` excess returns; row 0 is NaN (no previous bar)."""
    excess = one_bar_returns(price)
    excess[1:] -= risk_free[:-1]
    return excess


class Use4RiskModel(FactorRiskModel):
    """USE4-style factor risk model: country, industries and styles; EWMA, Newey-West, eigenfactors, VRA.

    See the module docstring for the regression; ``FactorRiskModel`` for the
    contract every factor risk model meets. This model's stores:

    The regression store (``regression``) holds:

    - ``factor_return`` on ``(timestamp, factor)``: the factor returns, NaN
      for an industry left out of the bar and on a bar without a regression;
    - ``specific_return`` on ``(timestamp, symbol)``;
    - ``r_squared`` on ``timestamp``: the weighted R-squared of the fit,
      ``1 - sum(w e^2) / sum(w y^2)`` over the trimmed returns ``y`` (our
      choice: uncentred, as the country factor plays the intercept);
    - ``estu_count`` on ``timestamp``: the symbols in the fit;
    - ``industry_members`` on ``(timestamp, industry)``: estimation-universe
      symbols of each industry with exposures and a return, before thin
      industries are left out;
    - ``industry_excluded`` on ``(timestamp, industry)``: whether the
      industry was left out of the bar.

    Its warm-up is one bar: row ``t`` reads the exposures and the price of
    the previous priced bar. A bar has no regression (all NaN, a count of 0)
    when it has no more fitted symbols than factors to fit.

    The estimate store (``estimate``) is computed from the regression store
    and holds forecasts of the next one-bar return:

    - ``factor_covariance`` on ``(timestamp, factor_i, factor_j)``: ``F_ij =
      rho_ij sigma_i sigma_j`` (USE4 eq. 4.1), the volatilities ``sigma``
      from the last ``volatility_window`` factor returns weighted with
      ``volatility_half_life`` with ``volatility_lags`` Newey-West lags, the
      correlations ``rho`` from the last ``correlation_window`` weighted with
      ``correlation_half_life`` with ``correlation_lags`` lags (USE4 §4.1),
      then the eigenfactor risk adjustment (USE4 §4.2, Appendix B): the
      eigenvariances of ``F`` rescaled by the volatility bias of the
      eigenfactors of ``eigen_simulations`` histories simulated from ``F``
      and estimated the same way, as eq. B7, or with ``eigen_scale`` the
      scaled eq. B8; each bar's draws are seeded by ``eigen_seed`` and the
      bar, so a row is the same however the store is built. The adjusted
      block is positive semi-definite (see ``_eigen_adjusted``); then the
      volatility regime adjustment, the whole matrix times ``lambda_F^2``
      (USE4 §4.3, eqs. 4.3-4.5), which leaves the correlations alone;
    - ``specific_risk`` on ``(timestamp, symbol)`` (USE4 §5.1-5.2), for
      every symbol with a time-series value and, with a structural model,
      every symbol with exposures at the bar:

      1. the time-series volatility: the last ``specific_window`` specific
         returns weighted with ``specific_half_life``, times the square root
         of the Newey-West multiplier ``C_NW`` (eq. 5.2);
      2. combined with a structural volatility (eqs. 5.3-5.5): a blending
         coefficient per symbol from how many returns it has and how
         fat-tailed they are, a daily regression of the log time-series
         volatility on the exposures and the history length ``log(1 + h /
         252)`` over every symbol with a time series (USE4 fits the
         symbols whose coefficient is 1 on the exposures alone; see
         ``structural_fit`` and ``structural_history_window``), and
         ``structural_bias`` (by default the smearing estimate) times the
         exponential of each symbol's fitted value;
         ``structural_model="fill"`` (the default) gives it to
         the symbols without a time-series value, ``"blend"`` blends every
         symbol as USE4 eq. 5.5 (see ``Use4RiskConfig.structural_model``
         and ``blending_min_observations``);
      3. shrunk toward the cap-weighted mean of the symbol's size group with
         intensity ``q |s - m| / (d + q |s - m|)`` (eqs. 5.6-5.9, ``q`` the
         ``shrinkage``);
      4. times ``lambda_S``, the volatility regime adjustment (§5.3, eqs.
         5.10-5.12).

      The time series, the blending coefficient and the history length
      read only their windows;
      the structural fit and the shrinkage use the exposures and market caps
      of the bar.

    - with the volatility regime adjustment (``vra_half_life``),
      ``factor_volatility_multiplier`` and ``specific_volatility_multiplier``
      on ``timestamp``: ``lambda_F`` and ``lambda_S``, each the root of the
      exponentially weighted mean square of the cross-sectional bias
      statistics of the last ``vra_window`` bars, a bar's returns against
      the forecasts of the bar before (see ``Use4RiskConfig.vra_half_life``).
      The forecasts it reads are the store's before the adjustment, so a
      row reads the ``vra_window`` rows before it as well; the specific ones
      include Newey-West when ``specific_lags`` is set, where USE4 removes it
      (our choice; off by default).

    Newey-West, with Newey and West's (1987) Bartlett weights ``b_l = 1 - l
    / (L + 1)`` over lags 1 to ``L`` (USE4 publishes no weights; our
    choice):

    - a factor variance and a specific variance are multiplied by ``C_NW =
      1 + 2 sum_l b_l rho_l``, at least 0, ``rho_l`` the autocorrelation at
      lag ``l`` (USE4 eq. 5.2), weighted with its own half-life over its own
      window: ``volatility_autocorrelation_half_life`` and ``_window``
      (504 and 1512 by default, longer than the volatilities' so the
      multiplier is less noisy) and ``specific_autocorrelation_half_life``
      and ``_window`` (the specific adjustment is off by default,
      ``specific_lags=0``; see ``Use4RiskConfig``);
    - a correlation is that of the covariance ``G_0 + sum_l b_l (G_l +
      G_l')``, ``G_l[i, j]`` the weighted mean of ``x_i(t) x_j(t - l)`` over
      the bars ``t`` of the correlation window both have, each column about
      its weighted mean, weighted by the weight of ``t``, ``G_0`` the
      covariance described below; it divides by each factor's variance over
      the pair's bars plus its own lag terms.

    Each ``rho_l`` and ``G_l`` is taken over the bars of the window that
    have both returns; a lag with fewer than ``min_observations`` such bars
    adds nothing. With 0 lags the
    estimates are the plain exponentially weighted moments. The forecasts
    stay one-bar: Newey-West turns a sum of serially correlated bars' returns
    into a per-bar variance that scales linearly with the horizon.

    The weights halve every half-life back from the bar and stop at the
    window, so a row reads only its window. Moments are taken about the
    weighted mean, normalised by the sum of the weights. A missing factor
    return (an industry left out of a bar) leaves its bar out: each
    correlation uses the bars both factors have (pairwise, our choice where
    USE4 uses the EM algorithm), so the estimate need not be positive
    semi-definite; the eigenfactor adjustment sets its negative
    eigenvalues to 0, and with ``eigen_simulations=0`` it stays as it is.
    A variance, a correlation or a specific volatility with
    fewer than ``min_observations`` bars is NaN. Its warm-up is the longest
    window less one bar, plus ``vra_window`` with the volatility regime
    adjustment, counted on the regression store's bars.

    Parameters
    ----------
    config : Use4RiskConfig
        The exposures factor (a ``BarraStyle`` by default), the dataset it
        reads and the method's parameters.

    Raises
    ------
    TypeError
        If ``config`` is not a ``Use4RiskConfig``.
    ValueError
        If a parameter is invalid.

    Examples
    --------
    With ``style`` a ``BarraStyle`` whose store is built and ``prices`` the
    dataset it reads:

    >>> model = Use4RiskModel(Use4RiskConfig(
    ...     exposures=style, dataset=prices, exposure_data_strategy="read",
    ...     regression_path="risk/use4_regression.zarr",
    ...     estimate_path="risk/use4_estimate.zarr",
    ... ))
    >>> len(model.factor_names), model.factor_names[:2]
    (61, ('country', 'industry_1'))
    >>> model.regression.build("2012-01-01", "2024-12-31")
    >>> model.estimate.build("2018-01-01", "2024-12-31")
    """

    #: The config class ``from_config`` rebuilds this model with.
    config_cls = Use4RiskConfig

    # Narrower type annotation for readers and type checkers only.
    config: Use4RiskConfig

    def _validate(self, config: Use4RiskConfig) -> None:
        """Raise ``ValueError`` for a parameter the regression or the estimates cannot use."""
        super()._validate(config)
        owner = self.class_name
        if len(set(config.style_names)) != len(config.style_names):
            raise ValueError(f"{owner}: style_names must be distinct.")
        if config.industry_name is None and config.industries:
            raise ValueError(f"{owner}: industries need an industry_name.")
        if config.industry_name is not None and not config.industries:
            raise ValueError(f"{owner}: industry_name needs the industries.")
        if len(set(config.industries)) != len(config.industries):
            raise ValueError(f"{owner}: industries must be distinct.")
        if not (config.country or config.industries or config.style_names):
            raise ValueError(f"{owner}: the model has no factor.")
        if config.weighting not in REGRESSION_WEIGHTINGS:
            raise ValueError(
                f"{owner}: weighting must be one of {REGRESSION_WEIGHTINGS}, got "
                f"{config.weighting!r}."
            )
        if config.min_industry_members < 1:
            raise ValueError(f"{owner}: min_industry_members must be at least 1.")
        if not config.return_outlier_sigma > 0:
            raise ValueError(f"{owner}: return_outlier_sigma must be positive.")
        for prefix in (
            "volatility",
            "correlation",
            "specific",
            "specific_autocorrelation",
            "volatility_autocorrelation",
            "vra",
        ):
            half_life = getattr(config, f"{prefix}_half_life")
            window = getattr(config, f"{prefix}_window")
            if half_life is not None and not half_life > 0:
                raise ValueError(f"{owner}: {prefix}_half_life must be positive.")
            if window is not None and window < 2:
                raise ValueError(f"{owner}: {prefix}_window must be at least 2.")
        for prefix, window in (
            ("volatility", self._volatility_autocorrelation(config)[1]),
            ("correlation", config.correlation_window),
            ("specific", config.specific_autocorrelation_window),
        ):
            if not 0 <= getattr(config, f"{prefix}_lags") < window:
                raise ValueError(
                    f"{owner}: {prefix}_lags must be at least 0 and below its window, "
                    f"{window}; got {getattr(config, f'{prefix}_lags')}."
                )
        if config.structural_model not in ("fill", "blend", "off"):
            raise ValueError(
                f"{owner}: structural_model must be 'fill', 'blend' or 'off', got "
                f"{config.structural_model!r}."
            )
        if config.njobs < 1:
            raise ValueError(f"{owner}: njobs must be at least 1, got {config.njobs}.")
        if config.structural_bias != "smearing" and not (
            isinstance(config.structural_bias, (int, float)) and config.structural_bias > 0
        ):
            raise ValueError(
                f"{owner}: structural_bias must be positive or 'smearing', got "
                f"{config.structural_bias!r}."
            )
        if config.structural_fit not in ("blending", "series"):
            raise ValueError(
                f"{owner}: structural_fit must be 'blending' or 'series', got "
                f"{config.structural_fit!r}."
            )
        if config.structural_history_window is not None and config.structural_history_window < 1:
            raise ValueError(f"{owner}: structural_history_window must be at least 1 or None.")
        if config.blending_min_observations < 0 or config.blending_ramp < 1:
            raise ValueError(
                f"{owner}: blending_min_observations must be at least 0 and blending_ramp "
                f"at least 1."
            )
        if not config.blending_outlier_bound > 0:
            raise ValueError(f"{owner}: blending_outlier_bound must be positive.")
        if config.eigen_simulations < 0:
            raise ValueError(f"{owner}: eigen_simulations must be at least 0.")
        if config.eigen_scale is not None and not config.eigen_scale > 0:
            raise ValueError(f"{owner}: eigen_scale must be positive or None.")
        if config.eigen_fit_skip < 0:
            raise ValueError(f"{owner}: eigen_fit_skip must be at least 0.")
        if config.shrinkage < 0 or config.shrinkage_groups < 1:
            raise ValueError(
                f"{owner}: shrinkage must be at least 0 and shrinkage_groups at least 1."
            )
        windows = self._estimate_windows(config)
        if not 2 <= config.min_observations <= min(windows):
            raise ValueError(
                f"{owner}: min_observations must be at least 2 and at most the shortest "
                f"window, {min(windows)}; got {config.min_observations}."
            )

    @property
    def exposure_names(self) -> tuple[str, ...]:
        """The style exposures and, with industries, the industry code.

        Examples
        --------
        >>> model.exposure_names[-2:]
        ('style_growth', 'industry')
        """
        config = self.config
        industry = (config.industry_name,) if config.industry_name is not None else ()
        return (*config.style_names, *industry)

    @property
    def regression_warmup_bars(self) -> int:
        """One bar: row ``t`` reads the exposures and the price of the previous priced bar.

        Examples
        --------
        >>> model.regression_warmup_bars
        1
        """
        return 1

    @property
    def estimate_warmup_bars(self) -> int:
        """The longest estimate window less one bar, plus ``vra_window`` with the regime adjustment.

        Examples
        --------
        >>> model.estimate_warmup_bars  # the 1512-bar correlation window, 126 VRA bars
        1637
        """
        config = self.config
        return max(self._estimate_windows(config)) - 1 + self._regime_window(config)

    @property
    def factor_names(self) -> tuple[str, ...]:
        """The ``factor`` axis: ``country``, ``industry_<code>`` per industry, styles.

        Examples
        --------
        >>> model.factor_names[:3], model.factor_names[-1], len(model.factor_names)
        (('country', 'industry_1', 'industry_2'), 'style_growth', 61)
        """
        config = self.config
        country = ("country",) if config.country else ()
        industries = tuple(f"industry_{code}" for code in config.industries)
        return (*country, *industries, *config.style_names)

    def factor_groups(self) -> dict[str, str]:
        """Return ``country`` for the country factor, ``industry`` for the industries, ``style`` for the styles.

        Examples
        --------
        >>> groups = model.factor_groups()
        >>> groups["country"], groups["industry_1"], groups["style_growth"]
        ('country', 'industry', 'style')
        """
        config = self.config
        country = {"country": "country"} if config.country else {}
        industries = {f"industry_{code}": "industry" for code in config.industries}
        return {**country, **industries, **{name: "style" for name in config.style_names}}

    def factor_labels(self) -> dict[str, str]:
        """Return ``Country``, each industry's Fama-French 48 name and each style without ``style_``.

        Examples
        --------
        >>> labels = model.factor_labels()
        >>> labels["country"], labels["industry_34"], labels["style_residual_volatility"]
        ('Country', 'Business Services', 'residual volatility')
        """
        config = self.config
        country = {"country": "Country"} if config.country else {}
        industries = {
            f"industry_{code}": FF48_INDUSTRY_NAMES.get(code, f"industry {code}")
            for code in config.industries
        }
        styles = {
            name: name.removeprefix("style_").replace("_", " ") for name in config.style_names
        }
        return {**country, **industries, **styles}

    def exposure_matrix(self, exposures: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Return each symbol's exposures to ``factor_names`` and whether it has them all.

        Parameters
        ----------
        exposures : xr.Dataset
            The exposures factor's values at one bar, on ``symbol``.

        Returns
        -------
        matrix : np.ndarray
            ``[n_symbols, n_factors]``: 1 on the country factor, 1 on the
            symbol's industry and 0 on the others, its style exposures.
        covered : np.ndarray
            Booleans: every style finite and, with industries, an industry
            among ``config.industries``.

        Examples
        --------
        >>> matrix, covered = model.exposure_matrix(style.compute(day, day).isel(timestamp=0))
        >>> matrix.shape[1] == len(model.factor_names)
        True
        """
        config = self.config
        n = exposures.sizes["symbol"]
        styles = np.column_stack(
            [np.asarray(exposures[name].values, dtype=np.float64) for name in config.style_names]
            or [np.zeros((n, 0))]
        )
        parts = [np.ones((n, 1))] if config.country else []
        covered = np.isfinite(styles).all(axis=1)
        if config.industry_name is not None:
            codes = np.asarray(exposures[config.industry_name].values, dtype=np.float64)
            industries = np.asarray(config.industries, dtype=np.float64)
            dummies = (codes[:, None] == industries[None, :]).astype(np.float64)
            covered &= dummies.any(axis=1)
            parts.append(dummies)
        parts.append(styles)
        return np.column_stack(parts), covered

    @staticmethod
    def _volatility_autocorrelation(config: Use4RiskConfig) -> tuple[float, int]:
        """Return the half-life and window of the factor volatilities' autocorrelations."""
        half_life = config.volatility_autocorrelation_half_life
        window = config.volatility_autocorrelation_window
        return (
            config.volatility_half_life if half_life is None else half_life,
            config.volatility_window if window is None else window,
        )

    @staticmethod
    def _regime_window(config: Use4RiskConfig) -> int:
        """Return the bars of bias statistics the volatility regime adjustment reads, 0 without it."""
        return config.vra_window if config.vra_half_life is not None else 0

    @staticmethod
    def _estimate_windows(config: Use4RiskConfig) -> tuple[int, ...]:
        """Return the windows the estimate rows read, in bars.

        An autocorrelation window is read only when its Newey-West lags are
        not 0, the history-length window only with a structural model.
        """
        windows = [
            _covariance_window(_EstimateParameters.of(config)), config.specific_window
        ]
        if config.specific_lags:
            windows.append(config.specific_autocorrelation_window)
        if _history_window(config):
            windows.append(config.structural_history_window)
        return tuple(windows)

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _previous_priced_bar(self, first: pd.Timestamp) -> pd.Timestamp | None:
        """Return the last bar before ``first`` with a price, or ``None``.

        Counts back one bar on the dataset's calendar, further when the bars
        found carry no price (a merged rate series' own days).
        """
        dataset = self.config.dataset
        back = 1
        while True:
            try:
                candidate, exhausted = dataset.bar_before(first, back), False
            except InsufficientHistoryError as exc:
                if exc.available == 0:
                    return None
                candidate, exhausted = dataset.bar_before(first, exc.available), True
            prices = self.prices(candidate, first - pd.Timedelta(1, "ns"))
            if prices.sizes["timestamp"]:
                return pd.Timestamp(prices["timestamp"].values[-1])
            if exhausted:
                return None
            back *= 2

    # ------------------------------------------------------------------
    # Regression
    # ------------------------------------------------------------------

    def _compute_regression(self, start, end) -> xr.Dataset:
        """Return the regression rows of the priced bars from ``start`` to ``end``."""
        config = self.config
        first, _ = check_range(start, end, f"{self.class_name}.regression.compute()")
        previous = self._previous_priced_bar(first)
        if previous is None:
            warnings.warn(
                f"{self.class_name}.regression.compute(): no priced bar before "
                f"{start!r}; the first bar has no regression.",
                UserWarning,
                stacklevel=4,
            )
        read_from = first if previous is None else previous
        prices = self.prices(read_from, end)
        exposures = self.exposures(read_from, end)
        symbols = sort_symbol_axis(
            set(prices["symbol"].values.tolist()) & set(exposures["symbol"].values.tolist())
        )
        bars = prices["timestamp"].values
        prices = prices.sel(symbol=symbols)
        exposures = exposures.reindex(timestamp=bars, symbol=symbols)

        def values(panel, name):
            return panel[name].transpose("timestamp", "symbol").values.astype(np.float64)

        shape = (len(bars), len(symbols))
        excess = _excess_returns(
            values(prices, config.price_column), values(prices, config.risk_free_column)
        )
        cap = values(prices, config.market_cap_column)
        if config.style_names:
            styles = np.stack([values(exposures, n) for n in config.style_names], axis=-1)
        else:
            styles = np.zeros((*shape, 0))
        if config.industry_name is None:
            industry = np.full(shape, -1, dtype=np.int64)
        else:
            code_index = {float(code): j for j, code in enumerate(config.industries)}
            codes = values(exposures, config.industry_name)
            industry = np.vectorize(lambda c: code_index.get(c, -1), otypes=[np.int64])(codes)
        if config.estu_name is None:
            estu = np.ones(shape, dtype=bool)
        else:
            estu = values(exposures, config.estu_name) == 1.0

        rows = np.flatnonzero(bars >= first.to_datetime64())
        n_industries = len(config.industries)
        factor_return = np.full((len(rows), len(self.factor_names)), np.nan)
        specific = np.full((len(rows), len(symbols)), np.nan)
        r_squared = np.full(len(rows), np.nan)
        estu_count = np.zeros(len(rows), dtype=np.int64)
        members = np.zeros((len(rows), n_industries), dtype=np.int64)
        excluded = np.ones((len(rows), n_industries), dtype=bool)
        with Timer(f"{self.class_name}: regression"):
            for row, i in enumerate(rows):
                if i == 0:
                    continue
                fit = self._regress_bar(
                    excess[i], styles[i - 1], industry[i - 1], estu[i - 1], cap[i - 1]
                )
                factor_return[row], specific[row] = fit["factor_return"], fit["specific"]
                r_squared[row], estu_count[row] = fit["r_squared"], fit["count"]
                members[row], excluded[row] = fit["members"], fit["excluded"]
        return xr.Dataset(
            {
                "factor_return": (("timestamp", "factor"), factor_return),
                "specific_return": (("timestamp", "symbol"), specific),
                "r_squared": (("timestamp",), r_squared),
                "estu_count": (("timestamp",), estu_count),
                "industry_members": (("timestamp", "industry"), members),
                "industry_excluded": (("timestamp", "industry"), excluded),
            },
            coords={
                "timestamp": bars[rows],
                "factor": list(self.factor_names),
                "symbol": symbols,
                "industry": list(config.industries),
            },
        )

    def _regress_bar(
        self,
        excess: np.ndarray,
        styles: np.ndarray,
        industry: np.ndarray,
        estu: np.ndarray,
        cap: np.ndarray,
    ) -> dict:
        """Fit one bar: excess returns of ``t`` on the exposures of ``t-1``.

        ``industry`` holds each symbol's position in ``config.industries``, -1
        for none, and ``estu`` whether it is in the estimation universe.
        Returns the factor returns over every factor, the specific returns
        over every symbol and the diagnostics.
        """
        config = self.config
        n_industries = len(config.industries)
        has_industry = config.industry_name is not None
        covered = np.isfinite(styles).all(axis=1) & ((industry >= 0) | (not has_industry))
        candidates = covered & estu & np.isfinite(excess) & np.isfinite(cap) & (cap > 0)
        slot = np.where(industry >= 0, industry, 0)
        members = np.bincount(industry[candidates & (industry >= 0)], minlength=n_industries)
        kept = members >= config.min_industry_members
        fit = candidates & (kept[slot] if has_industry else True)
        kept_index = np.flatnonzero(kept)
        result = {
            "factor_return": np.full(len(self.factor_names), np.nan),
            "specific": np.full(len(excess), np.nan),
            "r_squared": np.nan,
            "count": 0,
            "members": members,
            "excluded": ~kept,
        }

        # Industry dummies of the kept industries. With a country factor the
        # last one is expressed through the others, f_last = -sum_j (c_j /
        # c_last) f_j, so the cap-weighted industry factor returns sum to 0.
        dummies = (industry[fit][:, None] == kept_index[None, :]).astype(np.float64)
        ratio = np.zeros(0)
        restricted = dummies
        if config.country and len(kept_index):
            industry_cap = cap[fit] @ dummies
            ratio = industry_cap[:-1] / industry_cap[-1]
            restricted = dummies[:, :-1] - dummies[:, -1:] * ratio[None, :]
        country = [np.ones(fit.sum())] if config.country else []
        design = np.column_stack([*country, restricted, styles[fit]])
        if fit.sum() <= design.shape[1]:
            return result

        y = excess[fit]
        median = np.median(y)
        bound = config.return_outlier_sigma * MAD_TO_SIGMA * np.median(np.abs(y - median))
        # More than half the returns equal (stale prices) leaves no spread to
        # measure outliers by; nothing is trimmed then.
        if bound > 0:
            y = np.clip(y, median - bound, median + bound)
        weight = {
            "sqrt_cap": np.sqrt(cap[fit]),
            "cap": cap[fit],
            "equal": np.ones(fit.sum()),
        }[config.weighting]
        root = np.sqrt(weight)
        solution = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]

        n_country, n_free = len(country), restricted.shape[1]
        industry_returns = np.full(n_industries, np.nan)
        if len(kept_index):
            free = solution[n_country : n_country + n_free]
            industry_returns[kept_index] = (
                np.append(free, -ratio @ free) if config.country else free
            )
        style_returns = solution[n_country + n_free :]
        country_return = solution[:n_country]
        factor_return = np.concatenate([country_return, industry_returns, style_returns])

        residual = y - design @ solution
        with np.errstate(divide="ignore", invalid="ignore"):
            r_squared = 1.0 - (weight * residual**2).sum() / (weight * y**2).sum()

        # Specific returns of every covered symbol with a return, from the
        # untrimmed return; a left-out industry contributes nothing.
        fitted = styles @ style_returns + (country_return.sum() if config.country else 0.0)
        if has_industry:
            fitted = fitted + np.nan_to_num(industry_returns)[slot]
        specific = np.where(covered & np.isfinite(excess), excess - fitted, np.nan)
        result.update(
            factor_return=factor_return, specific=specific, r_squared=r_squared,
            count=int(fit.sum()),
        )
        return result

    # ------------------------------------------------------------------
    # Estimates
    # ------------------------------------------------------------------

    def _compute_estimate(self, start, end) -> xr.Dataset:
        """Return the estimate rows of the regression store's bars from ``start`` to ``end``.

        Raises
        ------
        ValueError
            If the regression store has no recorded range, or its recorded
            range does not contain the bars read (``RiskStore.read``).
        """
        config = self.config
        owner = f"{self.class_name}.estimate.compute()"
        first, last = check_range(start, end, owner)
        regression = self.regression
        recorded = regression.store_range()
        if recorded is None:
            raise ValueError(
                f"{owner}: the regression store has no recorded range; build it with "
                f"regression.build(start, end) first."
            )
        bars = regression.read(*recorded)["timestamp"].values
        begin = int(np.searchsorted(bars, first.to_datetime64(), side="left"))
        stop = int(np.searchsorted(bars, last.to_datetime64(), side="right"))
        warmup = self.estimate_warmup_bars
        if begin < warmup:
            warnings.warn(
                f"{owner}: {warmup} warm-up bar(s) are needed before {start!r} but the "
                f"regression store holds only {begin}; the first rows use shorter "
                f"windows.",
                UserWarning,
                stacklevel=4,
            )
        read_from = max(begin - warmup, 0)
        names = list(self.factor_names)
        if begin >= stop:
            rows = regression.read(recorded[0], recorded[0]).isel(timestamp=slice(0, 0))
        else:
            rows = regression.read(bars[read_from], end).load()
        # The volatility regime multipliers read the forecasts of the
        # ``vra_window`` bars before the first row, computed here too and
        # dropped at the end.
        lead = min(self._regime_window(config), begin) if begin < stop else 0
        offset = begin - lead - read_from
        count = max(stop - begin, 0) + lead
        timestamps = rows["timestamp"].values[offset : offset + count]
        symbols = rows["symbol"].values
        structural = count and config.structural_model != "off"
        refine = count and (structural or config.shrinkage > 0)
        if structural:
            # A symbol with exposures has a specific risk, a return or not.
            exposures = self.exposures(timestamps[0], timestamps[-1])
            symbols = np.asarray(sort_symbol_axis(
                set(symbols.tolist()) | set(exposures["symbol"].values.tolist())
            ))
        factor_returns = rows["factor_return"].transpose("timestamp", "factor").values
        specific_returns = (
            rows["specific_return"].reindex(symbol=symbols).transpose("timestamp", "symbol").values
        )
        covariance = np.full((count, len(names), len(names)), np.nan)
        specific_risk = np.full((count, len(symbols)), np.nan)
        blending = np.full((count, len(symbols)), np.nan)
        history = np.zeros((count, len(symbols)))
        parameters = _EstimateParameters.of(config)
        longest = max(self._estimate_windows(config))
        # Each row reads only its windows, so chunks of rows are independent
        # and the result does not depend on njobs. A chunk gets the rows its
        # windows reach, no more.
        ends = offset + np.arange(count) + 1  # rows before ``end`` end at the bar
        chunks = [
            chunk for chunk in np.array_split(ends, min(count, config.njobs * 4) or 1)
            if len(chunk)
        ]
        # A row's eigenfactor simulations draw from a stream of its own bar.
        bar_seeds = timestamps.astype("datetime64[ns]").astype(np.int64)
        tasks = []
        for chunk in chunks:
            first = max(int(chunk[0]) - longest, 0)
            tasks.append(delayed(_estimate_rows)(
                factor_returns[first : chunk[-1]],
                specific_returns[first : chunk[-1]],
                chunk - first,
                bar_seeds[chunk - offset - 1],
                parameters,
            ))
        with Timer(f"{self.class_name}: estimate"):
            if config.njobs == 1:
                results = [task[0](*task[1], **task[2]) for task in tasks]
            else:
                results = Parallel(n_jobs=config.njobs)(tasks)
        row = 0
        for chunk_covariance, chunk_specific, chunk_blending, chunk_history in results:
            covariance[row : row + len(chunk_covariance)] = chunk_covariance
            specific_risk[row : row + len(chunk_specific)] = chunk_specific
            blending[row : row + len(chunk_blending)] = chunk_blending
            history[row : row + len(chunk_history)] = chunk_history
            row += len(chunk_covariance)
        regime = count and self._regime_window(config) > 0
        if refine or regime:
            cap = (
                self.prices(timestamps[0], timestamps[-1])[config.market_cap_column]
                .reindex(timestamp=timestamps, symbol=symbols)
                .transpose("timestamp", "symbol")
                .values
            )
        if refine:
            if structural:
                exposures = exposures.reindex(timestamp=timestamps, symbol=symbols).load()
            with Timer(f"{self.class_name}: specific risk refinements"):
                for row in range(count):
                    matrix, covered = (
                        self.exposure_matrix(exposures.isel(timestamp=row)) if structural
                        else (None, None)
                    )
                    specific_risk[row] = self._refined_specific_risk(
                        specific_risk[row], blending[row], matrix, covered, cap[row],
                        history[row],
                    )
        variables = {}
        if regime:
            if config.estu_name is None:
                estu = np.ones((count, len(symbols)), dtype=bool)
            else:
                estu = (
                    self.exposures(timestamps[0], timestamps[-1])[config.estu_name]
                    .reindex(timestamp=timestamps, symbol=symbols)
                    .transpose("timestamp", "symbol")
                    .values
                    == 1.0
                )
            factor_multiplier, specific_multiplier = _regime_multipliers(
                covariance,
                specific_risk,
                factor_returns[offset : offset + count],
                specific_returns[offset : offset + count],
                cap,
                estu,
                config.vra_half_life,
                config.vra_window,
                config.min_observations,
            )
            covariance = covariance * factor_multiplier[:, None, None] ** 2
            specific_risk = specific_risk * specific_multiplier[:, None]
            variables = {
                "factor_volatility_multiplier": (("timestamp",), factor_multiplier[lead:]),
                "specific_volatility_multiplier": (("timestamp",), specific_multiplier[lead:]),
            }
        return xr.Dataset(
            {
                "factor_covariance": (
                    ("timestamp", "factor_i", "factor_j"), covariance[lead:]
                ),
                "specific_risk": (("timestamp", "symbol"), specific_risk[lead:]),
                **variables,
            },
            coords={
                "timestamp": timestamps[lead:],
                "factor_i": names,
                "factor_j": names,
                "symbol": symbols,
            },
        )

    def _refined_specific_risk(
        self,
        time_series: np.ndarray,
        blending: np.ndarray,
        matrix: np.ndarray,
        covered: np.ndarray,
        cap: np.ndarray,
        history: np.ndarray,
    ) -> np.ndarray:
        """Return one bar's specific volatilities: structural blend, then shrinkage.

        ``time_series`` and ``blending`` are each symbol's time-series
        volatility and blending coefficient, ``matrix`` and ``covered`` its
        exposures (``exposure_matrix``), ``cap`` its market cap and
        ``history`` its specific returns in the history-length window. See
        ``Use4RiskConfig.structural_model``, ``structural_fit``,
        ``structural_bias``, ``structural_history_window`` and ``shrinkage``.
        """
        config = self.config
        sigma = time_series.copy()
        if config.structural_model != "off":
            if _history_window(config):
                matrix = np.column_stack([matrix, np.log1p(history / 252.0)])
            structural = _structural_volatility(
                time_series, blending, matrix, covered, cap, config.weighting,
                config.structural_bias, config.structural_fit,
            )
            # A symbol without a time series is all structural.
            gamma = np.where(np.isfinite(time_series), np.nan_to_num(blending), 0.0)
            if config.structural_model == "blend":
                # USE4 eq. 5.5.
                blended = np.where(
                    gamma >= 1.0,
                    time_series,
                    gamma * np.nan_to_num(time_series) + (1 - gamma) * structural,
                )
            else:
                blended = np.where(np.isfinite(time_series), time_series, structural)
            # Where no structural value exists the time series stands.
            sigma = np.where(np.isfinite(blended), blended, time_series)
        if config.shrinkage > 0:
            sigma = _shrunk(sigma, cap, config.shrinkage, config.shrinkage_groups)
        return sigma


@dataclass(frozen=True)
class _EstimateParameters:
    """The numbers the estimate rows are computed from, without the model's components.

    What a worker process receives in place of the config, whose exposures
    factor and dataset need not pickle.
    """

    volatility_half_life: float
    volatility_window: int
    volatility_lags: int
    volatility_autocorrelation_half_life: float
    volatility_autocorrelation_window: int
    correlation_half_life: float
    correlation_window: int
    correlation_lags: int
    specific_half_life: float
    specific_window: int
    specific_lags: int
    specific_autocorrelation_half_life: float
    specific_autocorrelation_window: int
    min_observations: int
    structural_model: str
    structural_history_window: int | None
    blending_min_observations: int
    blending_ramp: int
    blending_outlier_bound: float
    eigen_simulations: int
    eigen_seed: int
    eigen_scale: float | None
    eigen_fit_skip: int

    @classmethod
    def of(cls, config: Use4RiskConfig) -> "_EstimateParameters":
        """Return the parameters of ``config``, its defaults resolved."""
        half_life, window = Use4RiskModel._volatility_autocorrelation(config)
        fields = {f.name for f in dataclasses.fields(cls)}
        values = {name: getattr(config, name) for name in fields}
        values.update(
            volatility_autocorrelation_half_life=half_life,
            volatility_autocorrelation_window=window,
        )
        return cls(**values)


def _estimate_rows(
    factor_returns: np.ndarray,
    specific_returns: np.ndarray,
    ends: np.ndarray,
    bars: np.ndarray,
    parameters: _EstimateParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the factor covariances, time-series specific risks, blending coefficients and history lengths.

    One row per bar, each bar ending before one of ``ends``; ``bars`` are
    the rows' timestamps in nanoseconds, which seed their simulations. A
    history length is a symbol's specific returns in the last
    ``structural_history_window`` rows, 0 without that window.
    """
    covariance = np.stack([
        _factor_covariance(factor_returns[:end], parameters, bar)
        for end, bar in zip(ends, bars)
    ])
    specific = np.stack([_specific_risk(specific_returns[:end], parameters) for end in ends])
    if parameters.structural_model != "off":
        blending = np.stack([_blending(specific_returns[:end], parameters) for end in ends])
    else:
        blending = np.ones_like(specific)
    window = _history_window(parameters)
    if window:
        present = np.isfinite(specific_returns)
        history = np.stack([present[max(end - window, 0) : end].sum(axis=0) for end in ends])
    else:
        history = np.zeros_like(specific)
    return covariance, specific, blending, history.astype(np.float64)


def _history_window(config) -> int | None:
    """Return the history-length regressor's window, ``None`` when it is not used."""
    if config.structural_model == "off":
        return None
    return config.structural_history_window


def _factor_covariance(
    history: np.ndarray, config: "_EstimateParameters", bar: int
) -> np.ndarray:
    """Return the ``[K, K]`` factor covariance at the last row of ``history``, at ``bar``.

    The Newey-West estimate, then the eigenfactor risk adjustment.
    """
    covariance = _sample_factor_covariance(history, config)
    if config.eigen_simulations:
        covariance = _eigen_adjusted(covariance, len(history), config, bar)
    return covariance


def _covariance_window(config: "_EstimateParameters") -> int:
    """Return the bars of factor returns ``_sample_factor_covariance`` reads, its longest window."""
    windows = [config.volatility_window, config.correlation_window]
    if config.volatility_lags:
        windows.append(config.volatility_autocorrelation_window)
    return max(windows)


#: An eigenvalue at most this times the largest is taken as 0 by the
#: eigenfactor risk adjustment (our choice).
_EIGEN_TOLERANCE = 1e-10

#: Simulated histories of the eigenfactor risk adjustment estimated together:
#: 16 histories of 1512 bars of 61 factors are about 12 MB a copy (our
#: choice; the results do not depend on it beyond rounding).
_EIGEN_BATCH = 16


def _eigen_adjusted(
    covariance: np.ndarray, available: int, config: "_EstimateParameters", bar: int
) -> np.ndarray:
    """Return ``covariance`` with the eigenfactor risk adjustment (USE4 Appendix B).

    Over the factors whose covariances are all finite (``covered_factors``;
    the others are left as they are): ``F0 = U0 D0 U0'`` (eq. B2), negative
    eigenvalues of a pairwise estimate set to 0 (our choice), which makes
    ``F0`` the covariance the simulations are drawn from. Each of
    ``eigen_simulations`` simulations draws a complete history ``f_m = U0
    b_m`` (eq. B3), ``b_m`` normal with variances ``D0``, of
    ``min(available, longest window)`` bars (our choice: what the sample
    estimate reads, not every bar of history), and estimates its covariance
    ``F_m = U_m D_m U_m'`` with ``_sample_factor_covariance`` (eqs. B4-B5).
    ``v(k) = sqrt(mean_m (U_m' F0 U_m)(k) / D_m(k))`` (eqs. B6-B7), the
    eigenfactors numbered from the lowest variance; with ``eigen_scale``
    ``a``, ``v`` is fitted by a parabola in ``k`` over the eigenfactors past
    the first ``eigen_fit_skip`` and replaced by ``a (v_P - 1) + 1`` (eq.
    B8; with fewer than three such eigenfactors, ``v`` itself is scaled).
    The result is ``U0 v^2 D0 U0'`` (eqs. B9-B10), made exactly symmetric.
    An eigenvalue of ``F0`` that is 0 stays 0; an eigenfactor whose
    simulated eigenvalue is 0 in some simulation is not adjusted. A
    simulation whose estimate is not all finite (Newey-West lag terms can
    make a short window's variance negative) is left out of the mean; with
    no more bars than factors there are no simulations, as every simulated
    covariance would be singular (our choices). The simulations run
    ``_EIGEN_BATCH`` at a time (``_complete_factor_covariances``).
    """
    kept = covered_factors(covariance)
    if not kept.any():
        return covariance
    block = covariance[np.ix_(kept, kept)]
    block = (block + block.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    # The covariance the simulations are drawn from, their "truth" (eq. B6).
    generating = (eigenvectors * eigenvalues) @ eigenvectors.T
    positive = eigenvalues > eigenvalues.max() * _EIGEN_TOLERANCE
    length = min(available, _covariance_window(config))
    n_factors = len(eigenvalues)
    rng = np.random.default_rng([config.eigen_seed, int(bar) % 2**64])
    ratio = np.zeros(n_factors)
    usable = np.ones(n_factors, dtype=bool)
    used = 0
    scale = np.sqrt(eigenvalues)
    # With no more bars than factors every simulated covariance is singular.
    simulations = config.eigen_simulations if length > n_factors else 0
    for first in range(0, simulations, _EIGEN_BATCH):
        # One draw of [batch, length, K] gives the numbers of that many
        # draws of [length, K] in turn.
        draws = rng.standard_normal((min(_EIGEN_BATCH, simulations - first), length, n_factors))
        estimated = _complete_factor_covariances((draws * scale) @ eigenvectors.T, config)
        estimated = estimated[np.isfinite(estimated).all(axis=(1, 2))]
        used += len(estimated)
        variances, rotation = np.linalg.eigh((estimated + estimated.swapaxes(1, 2)) / 2)
        true = (rotation * (generating @ rotation)).sum(axis=1)
        good = variances > variances.max(axis=1, keepdims=True) * _EIGEN_TOLERANCE
        ratio += np.where(good, true / np.where(good, variances, 1.0), 0.0).sum(axis=0)
        usable &= good.all(axis=0)
    valid = positive & usable & (used > 0)
    bias = np.where(valid, np.sqrt(ratio / max(used, 1)), 1.0)
    if config.eigen_scale is not None:
        scaled = config.eigen_scale * (_parabola(bias, valid, config.eigen_fit_skip) - 1.0) + 1.0
        bias = np.where(valid, scaled, 1.0)
    adjusted = (eigenvectors * np.where(positive, bias**2 * eigenvalues, 0.0)) @ eigenvectors.T
    out = covariance.copy()
    out[np.ix_(kept, kept)] = (adjusted + adjusted.T) / 2
    return out


def _regime_multipliers(
    covariance: np.ndarray,
    specific_risk: np.ndarray,
    factor_returns: np.ndarray,
    specific_returns: np.ndarray,
    cap: np.ndarray,
    estu: np.ndarray,
    half_life: float,
    window: int,
    least: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return each row's factor and specific volatility multipliers (USE4 eqs. 4.3-4.4, 5.10-5.11).

    Row ``t`` of every array is one bar: its forecasts (``covariance``,
    ``specific_risk``), the returns realized over it, and the market caps and
    estimation universe of the bar. The bias statistic of bar ``t`` compares
    its returns with the forecasts of row ``t - 1``: ``B_F`` is the root mean
    square of the standardized factor returns over the factors with both,
    ``B_S`` that of the specific returns weighted by the caps of ``t - 1``
    over its estimation universe. A row's multiplier is ``sqrt(sum_t w_t
    B_t^2)`` over the bias statistics of the last ``window`` bars up to and
    including it, ``w_t`` halving every ``half_life`` bars back and
    normalised to 1; 1 with fewer than ``least`` of them.
    """
    count = len(covariance)
    factor_bias = np.full(count, np.nan)
    specific_bias = np.full(count, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        volatility = np.sqrt(np.diagonal(covariance, axis1=1, axis2=2))
        factor_outcome = factor_returns[1:] / volatility[:-1]
        specific_outcome = specific_returns[1:] / specific_risk[:-1]
    factor_ok = np.isfinite(factor_outcome) & (volatility[:-1] > 0)
    squares = np.where(factor_ok, factor_outcome, 0.0) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        factor_bias[1:] = np.sqrt(squares.sum(axis=1) / factor_ok.sum(axis=1))
    weight = np.where(
        np.isfinite(specific_outcome) & (specific_risk[:-1] > 0)
        & np.isfinite(cap[:-1]) & (cap[:-1] > 0) & estu[:-1],
        cap[:-1],
        0.0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        specific_bias[1:] = np.sqrt(
            (weight * np.nan_to_num(specific_outcome) ** 2).sum(axis=1) / weight.sum(axis=1)
        )

    def multipliers(bias: np.ndarray) -> np.ndarray:
        out = np.ones(count)
        decay = _exponential_weights(window, half_life)
        for row in range(count):
            first = max(row - window + 1, 0)
            values = bias[first : row + 1]
            weights = decay[window - len(values) :]
            present = np.isfinite(values)
            if present.sum() >= least:
                out[row] = np.sqrt(
                    (weights[present] * values[present] ** 2).sum() / weights[present].sum()
                )
        return out

    return multipliers(factor_bias), multipliers(specific_bias)


def _complete_moments(
    histories: np.ndarray, half_life: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[B, L, K]`` complete histories (no value missing) about their weighted means, and the weights.

    The weights halve every ``half_life`` bars back from the last row and are
    normalised to sum to 1.
    """
    weights = _exponential_weights(histories.shape[1], half_life)
    weights = weights / weights.sum()
    centred = histories - (weights @ histories)[:, None, :]
    return centred, weights


def _complete_lag_sum(
    centred: np.ndarray, weights: np.ndarray, lags: int, least: int
) -> np.ndarray:
    """Return ``sum_l b_l G_l`` of ``_lagged_moments`` for ``[B, L, K]`` complete histories."""
    length = centred.shape[1]
    total = np.zeros((len(centred), centred.shape[2], centred.shape[2]))
    bartlett = _bartlett_weights(lags)
    for lag in range(1, min(lags, length - 1) + 1):
        if length - lag < least:
            break
        later = weights[lag:]
        moment = (centred[:, lag:] * later[:, None]).swapaxes(1, 2) @ centred[:, :-lag]
        total += bartlett[lag - 1] * moment / later.sum()
    return total


def _complete_newey_west_multiplier(
    histories: np.ndarray, half_life: float, lags: int, least: int
) -> np.ndarray:
    """Return ``_newey_west_multiplier`` of each column of ``[B, L, K]`` complete histories."""
    centred, weights = _complete_moments(histories, half_life)
    variance = weights @ centred**2
    adjustment = np.ones_like(variance)
    bartlett = _bartlett_weights(lags)
    for lag in range(1, min(lags, histories.shape[1] - 1) + 1):
        if histories.shape[1] - lag < least:
            break
        later = weights[lag:]
        autocovariance = later @ (centred[:, lag:] * centred[:, :-lag]) / later.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = autocovariance / variance
        adjustment += 2.0 * bartlett[lag - 1] * np.where(
            np.isfinite(rho) & (variance > 0), rho, 0.0
        )
    return np.clip(adjustment, 0.0, None)


def _complete_factor_covariances(
    histories: np.ndarray, config: "_EstimateParameters"
) -> np.ndarray:
    """Return ``_sample_factor_covariance`` of each of ``[B, L, K]`` complete histories.

    The same estimator: with no value missing, every pair of factors shares
    every bar, so the moments are batched matrix products; the results agree
    up to rounding.
    """
    least = config.min_observations
    volatility = histories[:, -config.volatility_window :]
    centred, weights = _complete_moments(volatility, config.volatility_half_life)
    variance = weights @ centred**2
    if config.volatility_lags:
        variance = variance * _complete_newey_west_multiplier(
            histories[:, -config.volatility_autocorrelation_window :],
            config.volatility_autocorrelation_half_life,
            config.volatility_lags,
            least,
        )
    if volatility.shape[1] < least:
        variance[:] = np.nan
    correlation_window = histories[:, -config.correlation_window :]
    centred, weights = _complete_moments(correlation_window, config.correlation_half_life)
    covariance = (centred * weights[:, None]).swapaxes(1, 2) @ centred
    if config.correlation_lags:
        summed = _complete_lag_sum(centred, weights, config.correlation_lags, least)
        covariance = covariance + summed + summed.swapaxes(1, 2)
    own = np.diagonal(covariance, axis1=1, axis2=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.sqrt(own[:, :, None] * own[:, None, :])
    if correlation_window.shape[1] < least:
        correlation[:] = np.nan
    index = np.arange(histories.shape[2])
    correlation[:, index, index] = 1.0
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    return correlation * sigma[:, :, None] * sigma[:, None, :]


def _parabola(values: np.ndarray, valid: np.ndarray, skip: int) -> np.ndarray:
    """Return a parabola in ``k = 1, 2, ...`` fitted to ``values`` past the first ``skip``.

    Only ``valid`` values are fitted; with fewer than three, ``values``
    themselves are returned.
    """
    k = np.arange(1, len(values) + 1, dtype=np.float64)
    fit = valid & (k > skip)
    if fit.sum() < 3:
        return values
    return np.polyval(np.polyfit(k[fit], values[fit], 2), k)


def _sample_factor_covariance(history: np.ndarray, config: "_EstimateParameters") -> np.ndarray:
    """Return the Newey-West ``[K, K]`` factor covariance at the last row of ``history``."""
    least = config.min_observations
    volatility_window = np.ascontiguousarray(history[-config.volatility_window :])
    covariance, _, _, observed = _pairwise_moments(
        volatility_window, config.volatility_half_life
    )
    variance = np.diag(covariance)
    if config.volatility_lags:
        variance = variance * _newey_west_multiplier(
            history[-config.volatility_autocorrelation_window :],
            config.volatility_autocorrelation_half_life,
            config.volatility_lags,
            least,
        )
    variance = np.where(np.diag(observed) >= least, variance, np.nan)
    correlation_window = np.ascontiguousarray(history[-config.correlation_window :])
    covariance, variance_i, variance_j, observed = _pairwise_moments(
        correlation_window, config.correlation_half_life
    )
    if config.correlation_lags:
        lagged = _lagged_moments(
            correlation_window, config.correlation_half_life, config.correlation_lags, least
        )
        summed = np.einsum("l,lij->ij", _bartlett_weights(config.correlation_lags), lagged)
        covariance = covariance + summed + summed.T
        own = 2.0 * np.diag(summed)
        variance_i = variance_i + own[:, None]
        variance_j = variance_j + own[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / np.sqrt(variance_i * variance_j)
    correlation[observed < least] = np.nan
    np.fill_diagonal(correlation, 1.0)
    sigma = np.sqrt(np.clip(variance, 0.0, None))
    return correlation * np.outer(sigma, sigma)


def _specific_risk(history: np.ndarray, config: "_EstimateParameters") -> np.ndarray:
    """Return each symbol's specific volatility at the last row of ``history``."""
    window = np.ascontiguousarray(history[-config.specific_window :])
    weights = _exponential_weights(len(window), config.specific_half_life)[:, None]
    present = np.isfinite(window)
    values = np.where(present, window, 0.0)
    total = (weights * present).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (weights * values).sum(axis=0) / total
        variance = (weights * present * (values - mean) ** 2).sum(axis=0) / total
    if config.specific_lags:
        variance = variance * _newey_west_multiplier(
            history[-config.specific_autocorrelation_window :],
            config.specific_autocorrelation_half_life,
            config.specific_lags,
            config.min_observations,
        )
    enough = present.sum(axis=0) >= config.min_observations
    return np.where(enough, np.sqrt(variance), np.nan)


def _blending(history: np.ndarray, config: "_EstimateParameters") -> np.ndarray:
    """Return each symbol's blending coefficient at the last row of ``history``.

    ``min(1, max(0, (h - m) / r)) * min(1, max(0, exp(1 - Z)))`` over the
    last ``specific_window`` specific returns; see
    ``Use4RiskConfig.blending_min_observations``. A symbol whose returns
    have no interquartile range is 0.
    """
    window = history[-config.specific_window :]
    gamma = np.zeros(window.shape[1])
    active = np.flatnonzero(np.isfinite(window).sum(axis=0) > config.blending_min_observations)
    if not len(active):
        return gamma
    window = window[:, active]
    count = np.isfinite(window).sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        low, high = np.nanpercentile(window, [25, 75], axis=0)
        robust = (high - low) / 1.35
        bound = config.blending_outlier_bound * robust
        spread = np.nanstd(np.clip(window, -bound, bound), axis=0, ddof=1)
        tails = np.abs(spread / robust - 1.0)
        coverage = np.clip((count - config.blending_min_observations) / config.blending_ramp, 0.0, 1.0)
        value = coverage * np.clip(np.exp(1.0 - tails), 0.0, 1.0)
    gamma[active] = np.where(np.isfinite(value) & (robust > 0), value, 0.0)
    return gamma


def _structural_volatility(
    time_series: np.ndarray,
    blending: np.ndarray,
    matrix: np.ndarray,
    covered: np.ndarray,
    cap: np.ndarray,
    weighting: str,
    bias: float | str,
    fit_set: str,
) -> np.ndarray:
    """Return each covered symbol's structural specific volatility (USE4 eqs. 5.3-5.4).

    The log time-series volatility of the covered symbols with a time-series
    value and a market cap (``fit_set="blending"``: and a blending
    coefficient of 1) is regressed on the columns of ``matrix``, weighted as
    ``weighting``; a covered symbol's structural volatility is ``bias``
    times the exponential of its fitted value, ``bias="smearing"`` the
    unweighted mean of ``exp(residual)`` over the fitted symbols. NaN for
    every symbol when there are no more such symbols than columns.
    """
    structural = np.full(len(time_series), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        fit = (
            covered & np.isfinite(time_series)
            & (time_series > 0) & np.isfinite(cap) & (cap > 0)
        )
    if fit_set == "blending":
        fit &= np.nan_to_num(blending) >= 1.0
    if fit.sum() <= matrix.shape[1]:
        return structural
    weight = {"sqrt_cap": np.sqrt(cap[fit]), "cap": cap[fit], "equal": np.ones(fit.sum())}[weighting]
    root = np.sqrt(weight)
    log_volatility = np.log(time_series[fit])
    coefficients = np.linalg.lstsq(
        matrix[fit] * root[:, None], log_volatility * root, rcond=None
    )[0]
    if bias == "smearing":
        bias = float(np.mean(np.exp(log_volatility - matrix[fit] @ coefficients)))
    structural[covered] = bias * np.exp(matrix[covered] @ coefficients)
    return structural


def _shrunk(sigma: np.ndarray, cap: np.ndarray, q: float, groups: int) -> np.ndarray:
    """Return the specific volatilities shrunk toward their size group's mean (USE4 eqs. 5.6-5.9).

    The symbols with a volatility and a market cap are cut into ``groups``
    equal-count groups by market cap. Within a group, the target is the
    cap-weighted mean ``m``, ``d`` the population standard deviation of the
    volatilities, and a volatility ``s`` moves to ``v m + (1 - v) s`` with
    ``v = q |s - m| / (d + q |s - m|)``. A symbol without a market cap is
    left as it is.
    """
    shrunk = sigma.copy()
    usable = np.flatnonzero(np.isfinite(sigma) & np.isfinite(cap) & (cap > 0))
    order = usable[np.argsort(cap[usable], kind="stable")]
    for group in np.array_split(order, groups):
        if not len(group):
            continue
        weight = cap[group] / cap[group].sum()
        target = weight @ sigma[group]
        distance = np.abs(sigma[group] - target)
        spread = np.sqrt(np.mean((sigma[group] - target) ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = np.where(
                spread + q * distance > 0, q * distance / (spread + q * distance), 0.0
            )
        shrunk[group] = intensity * target + (1.0 - intensity) * sigma[group]
    return shrunk


__all__ = ["Use4RiskModel"]
