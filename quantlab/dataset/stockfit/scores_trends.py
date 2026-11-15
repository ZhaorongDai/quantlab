"""StockFit's scores and trends, accumulated from snapshots, keyed by CIK.

``/api/financials/scores`` (the Piotroski F-score with its nine criteria and
the Altman Z-score with its zone, from the latest annual filings) and
``/api/earnings/trends`` (margin trajectories, multi-year CAGRs and earnings
quality ratios) carry no history: each pull is what the vendor says today.
The raw tier keeps every pull that changed, so a history accumulates from a
company's first pull on.

``StockFitScoresTrendsDataset`` shows each pull's values from the day of the
pull (``quantlab.dataset.stockfit.available.pull_day``: its New York day when
made before the NYSE close, otherwise the next trading day) and nothing
before a company's first pull. ``period_end`` is the fiscal year the values
describe (``period`` of the scores, ``latestPeriod`` of the trends) and
``release_date`` the day it was first pulled. Text categories are coded:
the Altman zone as ``distress`` 0, ``grey`` 1, ``safe`` 2; a margin direction
as ``compressing`` -1, ``stable`` 0, ``expanding`` 1; a Piotroski criterion
as 1 when met, 0 when not. Nested trend fields are named by their path
(``margins_gross_latest``, ``cagr_revenue3yr``, ``quality_roe``).

Examples
--------
Build the scores store::

    config = StockFitScoresTrendsConfig(
        zarr_file_path="/data/quantlab/market/stockfit/stockfit_scores/stockfit_scores.zarr",
        raw_data_dir_path="/data/quantlab/downloads/stockfit",
        table="scores",
    )
    StockFitScoresTrendsDataset(config).update().panel("2026-11-02", "2026-11-27")["piotroskiFScore"]
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

from quantlab.dataset.config import DatasetConfig, StockFitScoresTrendsConfig
from quantlab.dataset.stockfit.available import pull_day
from quantlab.dataset.stockfit.period_panel import (
    PeriodRow,
    VersionedPeriodDataset,
    as_float,
)

#: The Altman zones, coded.
ALTMAN_ZONES: dict[str, float] = {"distress": 0.0, "grey": 1.0, "safe": 2.0}

#: The margin directions, coded.
DIRECTIONS: dict[str, float] = {"compressing": -1.0, "stable": 0.0, "expanding": 1.0}

#: The nine Piotroski criteria, in the vendor's order.
PIOTROSKI_CRITERIA: tuple[str, ...] = (
    "positiveNetIncome", "positiveOperatingCashFlow", "risingRoa", "cashFlowExceedsNetIncome",
    "decliningLeverage", "risingCurrentRatio", "noShareDilution", "risingGrossMargin", "risingAssetTurnover",
)

#: The scores' variables.
SCORE_FIELDS: tuple[str, ...] = ("piotroskiFScore", "altmanZScore", "altmanZone", *PIOTROSKI_CRITERIA)

#: The trends' nested fields, by path.
_TREND_PATHS: tuple[tuple[str, ...], ...] = (
    *((("margins", margin, part)) for margin in ("gross", "operating", "net") for part in ("latest", "threeYearAgo", "direction")),
    ("netMarginStdDev",), ("netMarginStdDevRecent",), ("historicalMaxNetMargin",),
    *(("cagr", name) for name in ("revenue3yr", "revenue5yr", "netIncome3yr", "netIncome5yr", "eps3yr", "eps5yr", "freeCashFlow3yr")),
    *(("quality", name) for name in ("fcfToNetIncome", "interestCoverage", "debtToOperatingIncome", "roe")),
)

#: The trends' variables, named by path.
TREND_FIELDS: tuple[str, ...] = tuple("_".join(path) for path in _TREND_PATHS)


def _dig(body: Mapping, path: tuple[str, ...]):
    for name in path:
        body = body.get(name) if isinstance(body, Mapping) else None
    return body


def _scores(body) -> tuple[str | None, dict[str, float]]:
    details = body.get("piotroskiDetails") or {}
    values = {
        "piotroskiFScore": as_float(body.get("piotroskiFScore")),
        "altmanZScore": as_float(body.get("altmanZScore")),
        "altmanZone": ALTMAN_ZONES.get(body.get("altmanZone"), float("nan")),
    }
    values.update({name: as_float(details.get(name)) for name in PIOTROSKI_CRITERIA})
    return body.get("period"), values


def _trends(body) -> tuple[str | None, dict[str, float]]:
    values = {}
    for path, name in zip(_TREND_PATHS, TREND_FIELDS):
        value = _dig(body, path)
        values[name] = DIRECTIONS.get(value, float("nan")) if path[-1] == "direction" else as_float(value)
    return body.get("latestPeriod"), values


#: The tables a store may hold: endpoint, variables, and how a pull becomes its period and values.
SNAPSHOT_TABLES: dict[str, tuple[str, tuple[str, ...], Callable]] = {
    "scores": ("scores", SCORE_FIELDS, _scores),
    "trends": ("trends", TREND_FIELDS, _trends),
}


class StockFitScoresTrendsDataset(VersionedPeriodDataset):
    """Daily panel of StockFit's scores or trends, each pull shown from its day, keyed by CIK.

    See the module docstring for what a cell shows and when.

    Parameters
    ----------
    config : StockFitScoresTrendsConfig
        ``table`` is ``"scores"`` or ``"trends"``.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitScoresTrendsDataset(config).update().panel("2026-11-02", "2026-11-06")["altmanZone"]
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitScoresTrendsConfig

    #: Each value is shown from the day it was pulled, so none is flagged.
    flags_restatements = False

    def _normalize_config(self, config: DatasetConfig) -> StockFitScoresTrendsConfig:
        """Normalise as the base class does, then check the table.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitScoresTrendsConfig``.
        ValueError
            If ``table`` is not one of ``SNAPSHOT_TABLES``.

        Examples
        --------
        ::

            StockFitScoresTrendsDataset(dataclasses.replace(config, table="growth"))
            # ValueError: StockFitScoresTrendsDataset: table must be one of ('scores', 'trends') ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitScoresTrendsConfig):
            raise TypeError(f"{self.class_name} needs a StockFitScoresTrendsConfig, got {type(config).__name__}.")
        if config.table not in SNAPSHOT_TABLES:
            raise ValueError(f"{self.class_name}: table must be one of {tuple(SNAPSHOT_TABLES)}, got {config.table!r}.")
        return config

    def _endpoint(self) -> str:
        """Return the table's endpoint."""
        return SNAPSHOT_TABLES[self.config.table][0]

    def _variables(self) -> dict[str, dict]:
        """Return the table's variables, each with its table."""
        return {name: {"table": self.config.table} for name in SNAPSHOT_TABLES[self.config.table][1]}

    def _period_rows(self, body, cik: int, pulled_at: datetime) -> dict[str, PeriodRow]:
        """Return the pull's one row, by the fiscal year it describes, placed at the day of the pull."""
        if not isinstance(body, Mapping):
            return {}
        period, values = SNAPSHOT_TABLES[self.config.table][2](body)
        if not period:
            return {}
        return {str(period)[:10]: PeriodRow(filed=pull_day(pulled_at), accepted_at=None, values=values)}
