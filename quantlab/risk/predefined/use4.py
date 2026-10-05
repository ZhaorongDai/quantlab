"""USE4-style factor risk model on the ``BarraStyle`` exposures.

Follows Menchero, Orr and Wang, *The Barra US Equity Model (USE4),
Methodology Notes* (MSCI, 2011), §3 and Appendix B: one country factor, one
factor per industry (the point-in-time Fama-French 48 industries) and the 12
USE4 style factors, 61 factors, fitted over the estimation universe with
square-root-of-cap weights (p.12) and the cap-weighted industry factor
returns constrained to sum to 0 (eq. 3.3). The regression itself is
``FactorRiskModel``'s; this model fixes the factor set, through the defaults
of ``Use4RiskConfig``.
"""

from quantlab.risk.base import FactorRiskModel
from quantlab.risk.config import Use4RiskConfig


class Use4RiskModel(FactorRiskModel):
    """USE4-style factor risk model: country, Fama-French 48 industries, 12 styles.

    ``FactorRiskModel`` with ``Use4RiskConfig``, whose defaults are USE4's
    factor set on the outputs of ``BarraStyle`` (``style_*``, ``industry``,
    ``estu``). See ``FactorRiskModel`` for the stores.

    Parameters
    ----------
    config : Use4RiskConfig
        The exposures factor (a ``BarraStyle``), the dataset it reads and the
        regression's parameters.

    Examples
    --------
    With ``style`` a ``BarraStyle`` whose store is built and ``prices`` the
    dataset it reads:

    >>> model = Use4RiskModel(Use4RiskConfig(
    ...     exposures=style, dataset=prices, exposure_data_strategy="read",
    ...     regression_path="risk/use4_regression.zarr",
    ... ))
    >>> len(model.factor_names), model.factor_names[:2]
    (61, ('country', 'industry_1'))
    >>> model.regression.build("2012-01-01", "2024-12-31")
    """

    #: The config class ``from_config`` rebuilds this model with.
    config_cls = Use4RiskConfig

    # Narrower type annotation for readers and type checkers only.
    config: Use4RiskConfig


__all__ = ["Use4RiskModel"]
