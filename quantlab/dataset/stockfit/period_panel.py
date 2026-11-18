"""Panels of StockFit rows kept without a restatement trail, one row per company and fiscal period.

The as-reported facts, the derived metrics, the share classes, the scores and
the trends are served as they are today: each period's latest values, with no
trail of what they replaced. Their panels are built here the same way:

- each pull of a company (``quantlab.dataset.stockfit.raw.pulls``) is read
  into one row per fiscal period end, with the filing date the row is placed
  at (its own, or one borrowed from the statements) and its values;
- the pulls are compared one after another
  (``quantlab.dataset.stockfit.vintages.observed_versions``): the first
  pull's rows count from their filing date and are flagged
  (``may_include_restatement``), since they may hold a restatement made
  after it; a row a later pull adds or changes counts from its filing date
  when filed since the pull before, and from the day of the pull otherwise;
- each version is placed on NYSE's calendar by StockFit's rule
  (``quantlab.dataset.stockfit.available``; a pull's day by
  ``available.pull_day``), and on every trading day the panel shows the
  latest fiscal period available by then, in its latest version known that
  day. A later filing of an older period never replaces a newer one, and a
  row whose period ends after its filing date (a maturity or schedule row
  the vendor serves as a period, ``vintages.reported_by``) is never shown.

``VersionedPeriodDataset`` is that panel; a subclass says which endpoint it
reads and how one pull becomes rows (``_period_rows``). ``release_date``
(the shown period's filing date) and ``period_end`` say which period a cell
shows. The raw tier is taken to be complete through the day before the
endpoint's newest pull.

Examples
--------
``StockFitAsReportedDataset`` (``quantlab.dataset.stockfit.as_reported``) is
one::

    StockFitAsReportedDataset(config).update().panel("2026-01-02", "2026-03-31")["Revenues"]
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import polars as pl

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
)
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.available import nyse_calendar, place_released, pull_day
from quantlab.dataset.stockfit.vintages import ServedRow, Version, observed_versions, reported_by
from quantlab.utils.timer import Timer

#: The variables saying which period a cell shows.
DATE_VARIABLES: tuple[str, ...] = ("release_date", "period_end")

#: The variable flagging values that may include a later restatement.
FLAG = "may_include_restatement"


@dataclass(frozen=True)
class PeriodRow:
    """One fiscal period of one pull: where it is placed and its values.

    Parameters
    ----------
    filed : date
        The filing date the row is placed at.
    accepted_at : datetime or None
        The filing's EDGAR acceptance time (UTC), when known: a row accepted
        at or after the NYSE close counts from the next trading day.
    values : Mapping[str, float]
        The row's values by variable, NaN where none is served.

    Examples
    --------
    >>> from datetime import date
    >>> PeriodRow(date(2026, 11, 9), None, {"value": 1.0}).filed
    datetime.date(2026, 11, 9)
    """

    filed: date
    accepted_at: datetime | None
    values: Mapping[str, float]


class VersionedPeriodDataset(ReleasePanelDataset):
    """Daily point-in-time panel of one StockFit endpoint served without a restatement trail, keyed by CIK.

    See the module docstring for what a cell shows and when. A subclass sets
    ``config_cls`` and implements ``_endpoint``, ``_variables`` and
    ``_period_rows``; ``flags_restatements`` says whether the panel carries
    ``may_include_restatement``.

    Examples
    --------
    With ``config`` a ``StockFitAsReportedConfig``::

        StockFitAsReportedDataset(config).update().panel("2026-01-02", "2026-01-05")["period_end"]
    """

    #: A raw tier's newest pull counts from the next trading day: an update before then writes nothing.
    SKIPS_EMPTY_UPDATE = True

    #: Whether the panel carries the ``may_include_restatement`` flag.
    flags_restatements: bool = True

    # -- what a subclass provides -------------------------------------------

    def _endpoint(self) -> str:
        """Return the raw-tier endpoint the panel reads."""
        raise NotImplementedError

    def _variables(self) -> dict[str, dict]:
        """Return the value variables and their attributes, in panel order."""
        raise NotImplementedError

    def _period_rows(self, body, cik: int, pulled_at: datetime) -> dict[str, PeriodRow]:
        """Return one pull's rows by fiscal period end (ISO date)."""
        raise NotImplementedError

    def _begin(self) -> None:
        """Prepare a build of the released rows; nothing by default."""

    def _end(self) -> None:
        """Finish a build of the released rows; nothing by default."""

    # -- the shared part ----------------------------------------------------

    def _ciks(self) -> list[str]:
        """Return the companies with a pull of the endpoint, only those of ``symbols`` when set."""
        found = sorted(raw.entities(self.config.raw_data_dir_path, self._endpoint()), key=int)
        if self.config.symbols is not None:
            wanted = {int(symbol) for symbol in self.config.symbols}
            found = [cik for cik in found if int(cik) in wanted]
        return found

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions through the day before the endpoint's newest pull.

        Raises
        ------
        ValueError
            If the endpoint was never pulled.
        """
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self.config.raw_data_dir_path, self._endpoint(),
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _versions(self) -> list[tuple[int, Version, datetime | None]]:
        """Return every version of every company's rows, with the acceptance time it is placed at."""
        root, endpoint = self.config.raw_data_dir_path, self._endpoint()
        found: list[tuple[int, Version, datetime | None]] = []
        for cik in self._ciks():
            pulls: list[tuple[datetime, dict[str, ServedRow]]] = []
            accepted: dict[tuple[str, date], datetime | None] = {}
            for pulled_at in raw.pulls(root, endpoint, cik):
                rows = self._period_rows(raw.read_pull(root, endpoint, cik, pulled_at), int(cik), pulled_at)
                pulls.append((pulled_at, {key: ServedRow(row.filed, row.values) for key, row in rows.items()}))
                for key, row in rows.items():
                    accepted.setdefault((key, row.filed), row.accepted_at)
            found.extend(
                (int(cik), version, None if version.seen_at is not None else accepted.get((version.key, version.filed)))
                for version in observed_versions(pulls)
                if reported_by(version.key, version.filed)  # else a schedule row, not a period of its filing
            )
        return found

    def _place(self, versions: list[tuple[int, Version, datetime | None]]) -> list[np.datetime64 | None]:
        """Return each version's available date: its filing's, or its pull day's; ``None`` when not reached yet."""
        frame = pl.DataFrame(
            {
                "accepted_at": [accepted for _, _, accepted in versions],
                "date_filed": [
                    version.filed if version.seen_at is None else pull_day(version.seen_at)
                    for _, version, _ in versions
                ],
                "_version": list(range(len(versions))),
            },
            schema={"accepted_at": pl.Datetime("us", "UTC"), "date_filed": pl.Date, "_version": pl.Int64},
        )
        placed = dict(
            place_released(frame, self._trading_calendar(), drop_earlier=False)
            .select("_version", "available")
            .iter_rows()
        )
        return [None if (day := placed.get(index)) is None else np.datetime64(day, "ns") for index in range(len(versions))]

    def _released(self) -> pl.DataFrame:
        """Return the rows the panel ever shows, each at its available date.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the CIK), ``available``, the date variables, the
            flag when carried and the value variables, sorted by
            ``available``.
        """
        variables = list(self._variables())
        names = ["symbol", "available", *DATE_VARIABLES, *([FLAG] if self.flags_restatements else []), *variables]
        columns: dict[str, list] = {name: [] for name in names}
        with Timer(f"{self.class_name}: shown rows"):
            self._begin()
            versions = self._versions()
            self._end()
            placed = self._place(versions)
            companies: dict[int, dict[str, list[tuple[np.datetime64, Version]]]] = {}
            for (cik, version, _), when in zip(versions, placed):
                if when is not None:
                    companies.setdefault(cik, {}).setdefault(version.key, []).append((when, version))
            for cik, periods in companies.items():
                for at, release, end, version in _company_rows(periods):
                    columns["symbol"].append(cik)
                    columns["available"].append(at)
                    columns["release_date"].append(np.datetime64(release, "ns"))
                    columns["period_end"].append(np.datetime64(end, "ns"))
                    if self.flags_restatements:
                        columns[FLAG].append(float(version.may_include_restatement))
                    for name in variables:
                        columns[name].append(raw.as_number(version.values.get(name)))
        for name in ("available", *DATE_VARIABLES):
            columns[name] = np.asarray(columns[name], dtype="datetime64[ns]")
        schema = {
            "symbol": pl.Int64,
            "available": pl.Datetime("ns"),
            **dict.fromkeys(DATE_VARIABLES, pl.Datetime("ns")),
            **({FLAG: pl.Float64} if self.flags_restatements else {}),
            **dict.fromkeys(variables, pl.Float64),
        }
        return pl.DataFrame(columns, schema=schema).sort("available", "symbol")

    def _spec(self) -> ReleaseSpec:
        """Return the value variables, the flag when carried and the date variables."""
        variables: dict[str, dict] = {
            "release_date": {"description": "filing date of the fiscal period shown"},
            "period_end": {"description": "fiscal period end of the period shown"},
        }
        if self.flags_restatements:
            variables[FLAG] = {
                "description": (
                    "1 when the shown values come from the company's first pull, so they may include "
                    "a restatement made after their filing date; 0 when a later pull first served them."
                )
            }
        variables.update(self._variables())
        return ReleaseSpec(variables=variables, dates=DATE_VARIABLES)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {self._endpoint()} row is shown"


def _company_rows(
    periods: dict[str, list[tuple[np.datetime64, Version]]],
) -> list[tuple[np.datetime64, date, date, Version]]:
    """Return one company's shown rows: ``(available, release, period end, version)``, one per change."""
    placed = []
    for key, found in periods.items():
        found = sorted(found, key=lambda item: item[0])  # stable: pull order breaks ties
        placed.append((found[0][0], date.fromisoformat(key), min(version.filed for _, version in found), found))
    placed.sort(key=lambda item: (item[0], item[1]))
    starts: list[tuple[np.datetime64, tuple]] = []
    for item in placed:
        if starts and item[1] <= starts[-1][1][1]:
            continue  # a later filing of an older period never replaces a newer one
        if starts and starts[-1][0] == item[0]:
            starts.pop()
        starts.append((item[0], item))
    rows = []
    for index, (start, (_, end, release, found)) in enumerate(starts):
        until = starts[index + 1][0] if index + 1 < len(starts) else None
        changes = sorted({start} | {when for when, _ in found if start < when and (until is None or when < until)})
        for at in changes:
            version = [version for when, version in found if when <= at][-1]
            rows.append((at, release, end, version))
    return rows
