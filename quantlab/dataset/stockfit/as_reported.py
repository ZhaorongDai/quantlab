"""StockFit's as-reported XBRL facts as a panel of chosen us-gaap concepts, keyed by CIK.

``/api/financials/as-reported`` serves, per fiscal period, every XBRL fact
the filer reported, keyed by the concept it used: us-gaap concepts and the
filer's own extensions. For each concept and period it serves the most
recently filed value, with the dates of the period's original filing and no
trail of the values it replaced.

``as_reported_concepts`` lists the us-gaap concepts at least a share of
companies use (5% by default), read from the raw tier, so the list can be
regenerated whenever the raw tier grows; a store takes its list as a config
parameter (``StockFitAsReportedConfig.concepts``). Filer extensions stay in
the raw tier.

``StockFitAsReportedDataset`` shows on every trading day the latest fiscal
period filed by then (quarterly or annual), each concept a float64 variable
named without its ``us-gaap:`` prefix (``Revenues``). Values from a company's
first pull are placed at the period's filing date (accepted before the NYSE
close, that day; otherwise the next trading day) and **may include
restatements made after it**: the panel flags them with
``may_include_restatement = 1``. From the first pull on, each pull is
compared with the one before and a changed value shows from its filing date
or from the day of the pull that first served it, unflagged
(``quantlab.dataset.stockfit.period_panel``).

Examples
--------
Regenerate the list, then build the quarterly store::

    root = "/data/quantlab/downloads/stockfit"
    config = StockFitAsReportedConfig(
        zarr_file_path="/data/quantlab/market/stockfit/stockfit_as_reported_quarter/stockfit_as_reported_quarter.zarr",
        raw_data_dir_path=root,
        concepts=as_reported_concepts(root),
    )
    panel = StockFitAsReportedDataset(config).update().panel("2024-01-02", "2024-03-28")
    panel["Revenues"].sel(symbol=320193)
"""

from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from pathlib import Path

from quantlab.dataset.config import DatasetConfig, StockFitAsReportedConfig
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.period_panel import (
    PeriodRow,
    VersionedPeriodDataset,
    as_float,
    parse_accepted,
)

#: The prefix of the concepts a panel may hold.
US_GAAP = "us-gaap:"

#: The periods a store may hold, and the fiscal periods each keeps.
PERIODS: dict[str, frozenset[str]] = {
    "quarter": frozenset({"Q1", "Q2", "Q3", "Q4"}),
    "annual": frozenset({"FY"}),
}


def endpoint(period: str) -> str:
    """Return the raw-tier endpoint of the as-reported facts of one period type.

    Examples
    --------
    >>> endpoint("quarter")
    'as_reported_quarter'
    """
    return f"as_reported_{period}"


def as_reported_concepts(vendor_root: str | Path, period: str = "quarter", min_share: float = 0.05) -> tuple[str, ...]:
    """Return the us-gaap concepts at least ``min_share`` of the companies in the raw tier use.

    A company uses a concept when its newest as-reported pull of ``period``
    serves a value of it for any period. The share is taken over the
    companies with such a pull.

    Parameters
    ----------
    vendor_root : str or Path
        ``<download-dir>/stockfit``.
    period : str, default "quarter"
        ``"quarter"`` or ``"annual"``.
    min_share : float, default 0.05
        The least share of companies, in ``(0, 1]``.

    Returns
    -------
    tuple of str
        The concepts, ``us-gaap:`` prefix included, sorted; empty when
        nothing is pulled.

    Examples
    --------
    >>> as_reported_concepts("/nonexistent/stockfit")
    ()
    """
    used: Counter[str] = Counter()
    companies = 0
    name = endpoint(period)
    for cik in raw.entities(vendor_root, name):
        if not raw.pulls(vendor_root, name, cik):
            continue
        companies += 1
        used.update({
            concept
            for entry in raw.read_pull(vendor_root, name, cik) or ()
            for concept, value in (entry.get("facts") or {}).items()
            if concept.startswith(US_GAAP) and value is not None
        })
    return tuple(sorted(concept for concept, count in used.items() if count >= min_share * companies)) if companies else ()


def variable_name(concept: str) -> str:
    """Return a concept's variable name: the concept without its ``us-gaap:`` prefix.

    Examples
    --------
    >>> variable_name("us-gaap:NetIncomeLoss")
    'NetIncomeLoss'
    """
    return concept.removeprefix(US_GAAP)


class StockFitAsReportedDataset(VersionedPeriodDataset):
    """Daily point-in-time panel of StockFit's as-reported us-gaap facts, keyed by CIK.

    See the module docstring for what a cell shows and when.

    Parameters
    ----------
    config : StockFitAsReportedConfig
        ``concepts`` (us-gaap, prefix included) and ``period``.

    Examples
    --------
    With ``config`` as in the module example::

        ds = StockFitAsReportedDataset(config).update()
        ds.panel("2024-01-02", "2024-01-05")["may_include_restatement"]
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitAsReportedConfig

    def _normalize_config(self, config: DatasetConfig) -> StockFitAsReportedConfig:
        """Normalise as the base class does, then check the as-reported fields.

        Raises
        ------
        TypeError
            If ``config`` is not a ``StockFitAsReportedConfig``.
        ValueError
            If ``concepts`` is empty or holds a concept outside us-gaap, or
            ``period`` or ``stale_after_days`` is invalid.

        Examples
        --------
        A filer extension stays in the raw tier::

            StockFitAsReportedDataset(dataclasses.replace(config, concepts=("plug:Custom",)))
            # ValueError: StockFitAsReportedDataset: only us-gaap concepts ...
        """
        config = super()._normalize_config(config)
        if not isinstance(config, StockFitAsReportedConfig):
            raise TypeError(f"{self.class_name} needs a StockFitAsReportedConfig, got {type(config).__name__}.")
        if not config.concepts:
            raise ValueError(
                f"{self.class_name}: concepts is empty; regenerate the list with as_reported_concepts()."
            )
        outside = [concept for concept in config.concepts if not concept.startswith(US_GAAP)]
        if outside:
            raise ValueError(
                f"{self.class_name}: only us-gaap concepts are paneled; filer extensions stay in the raw tier, got {outside}."
            )
        if config.period not in PERIODS:
            raise ValueError(f"{self.class_name}: period must be one of {tuple(PERIODS)}, got {config.period!r}.")
        if config.stale_after_days is not None and config.stale_after_days < 1:
            raise ValueError(
                f"{self.class_name}: stale_after_days must be at least 1 or None, got {config.stale_after_days}."
            )
        return config

    def _endpoint(self) -> str:
        """Return the as-reported endpoint of the configured period."""
        return endpoint(self.config.period)

    def _variables(self) -> dict[str, dict]:
        """Return one variable per concept, named without its prefix."""
        return {variable_name(concept): {"concept": concept} for concept in self.config.concepts}

    def _period_rows(self, body, cik: int, pulled_at: datetime) -> dict[str, PeriodRow]:
        """Return each period of the configured type with a filing date, placed at its original filing."""
        keep = PERIODS[self.config.period]
        rows: dict[str, PeriodRow] = {}
        for entry in body or ():
            if entry.get("fiscalPeriod") not in keep or not entry.get("period") or not entry.get("dateFiled"):
                continue
            facts = entry.get("facts") or {}
            rows[entry["period"]] = PeriodRow(
                filed=date.fromisoformat(entry["dateFiled"][:10]),
                accepted_at=parse_accepted(entry.get("acceptedAt")),
                values={variable_name(concept): as_float(facts.get(concept)) for concept in self.config.concepts},
            )
        return rows
