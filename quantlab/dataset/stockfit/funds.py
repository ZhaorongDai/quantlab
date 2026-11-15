"""StockFit's fund data as daily panels on the fund-series axis.

A fund series is one series of a fund trust, named by the trust's CIK and
the series id (``<cik>-<series id>``, ``quantlab.dataset.stockfit.raw.fund_series_name``):
two series of one trust are two symbols, and an ETF's share class is not a
series. Each panel reads the newest pull of every series of its endpoints in
StockFit's raw tier (``quantlab.dataset.stockfit.raw``) and shows, on every
NYSE session, the series' latest row available by then:

- ``StockFitFundProfileDataset``: the N-PORT summary (``/api/fund/profile``):
  total assets, liabilities and net assets, the three monthly returns, sales,
  redemptions and the holding count of each quarter;
- ``StockFitFundFlowsDataset``: the N-PORT monthly flows (``/api/fund/flows``):
  sales, reinvestments, redemptions and returns of each month of the quarter;
- ``StockFitFundFeesDataset``: the N-1A prospectus fee table
  (``/api/fund/fees``) of each filing;
- ``StockFitFundNcenDataset``: the N-CEN fields of each reporting period
  (``/api/fund/fund-health`` ``ncenRecords`` and ``/api/fund/fee-analysis``
  ``feeRecords``): index tracking, securities lending, red flags, expense
  limits, commissions and in-kind creation.

Available dates. N-PORT carries no filing date, only its quarter end
(``reportDate``); public N-PORT is due 60 days after the quarter, so a
quarter is available on the first session on or after quarter end + 60
days. N-CEN is due 75 days after the fund's fiscal year, so its fields are
available from ``reportDate`` + 75 days. A fee table is available on its
filing date (``filedDate``), as a row with only a filing date is in every
StockFit panel. A fund files N-PORT and N-CEN on its own schedule, often
earlier than the deadline, so these panels may show a quarter some days
after it was public; they never show it before.

Look-ahead caveats. Values are as StockFit serves them on the pull day: an
amended N-PORT or N-CEN replaces the original with no trace, so a backfilled
quarter can show an amendment's values from the original's deadline. The
fee panel shows the share class of the ticker the series was asked for by.
N-CSR performance, N-PORT holdings, daily ETF holdings, N-CEN service
providers and the adviser lookup stay in the raw tier only.

The days are NYSE's sessions up to the day before the panel's endpoints
were last pulled (the oldest of their newest watermarks), so ``update()``
appends a day only once a refresh after it has run.

Examples
--------
Build the N-PORT summary of every fund series and read one::

    config = StockFitFundConfig(
        zarr_file_path="/data/market/stockfit/stockfit_fund_profile_1d/stockfit_fund_profile_1d.zarr",
        raw_data_dir_path="/data/downloads/stockfit",
        start_date="2019-01-02",
    )
    panel = StockFitFundProfileDataset(config).update().panel("2024-01-02", "2024-12-31")
    panel["net_assets"].sel(symbol="36405-2839")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

import polars as pl

from quantlab.dataset._support.point_in_time import (
    ReleasePanelDataset,
    ReleaseSpec,
    TradingCalendar,
)
from quantlab.dataset.config import StockFitFundConfig
from quantlab.dataset.stockfit import raw
from quantlab.dataset.stockfit.available import nyse_calendar
from quantlab.dataset.stockfit.raw import as_number

__all__ = [
    "NCEN_LAG",
    "NPORT_LAG",
    "FundSource",
    "StockFitFundFeesDataset",
    "StockFitFundFlowsDataset",
    "StockFitFundNcenDataset",
    "StockFitFundPanel",
    "StockFitFundProfileDataset",
]

#: Public N-PORT is due 60 days after the quarter end (the SEC deadline).
NPORT_LAG = timedelta(days=60)

#: N-CEN is due 75 days after the fund's fiscal year end (the SEC deadline).
NCEN_LAG = timedelta(days=75)


@dataclass(frozen=True)
class FundSource:
    """One endpoint a fund panel reads, and the fields it takes from each row.

    Parameters
    ----------
    endpoint : str
        The endpoint's directory in the raw tier.
    fields : Mapping[str, tuple[str, dict]]
        Vendor field to the panel variable and its attributes, in panel
        order. A boolean field becomes 1.0 or 0.0.
    records : str or None, default None
        The key of the body's list of rows (``"ncenRecords"``); ``None`` when
        the pull is the list itself.

    Examples
    --------
    >>> FundSource("fund_flows", {"netAssets": ("net_assets", {"unit": "USD"})}).records is None
    True
    """

    endpoint: str
    fields: dict[str, tuple[str, dict]] = field(default_factory=dict)
    records: str | None = None


def _usd(description: str) -> dict:
    return {"unit": "USD", "description": description}


def _percent(description: str) -> dict:
    return {"unit": "percent", "description": description}


def _ratio(description: str) -> dict:
    return {"unit": "ratio", "description": description}


def _flag(description: str) -> dict:
    return {"unit": "1 yes, 0 no", "description": description}


class StockFitFundPanel(ReleasePanelDataset):
    """Daily point-in-time panel of StockFit fund rows, by fund series, on NYSE's sessions.

    A subclass sets ``SOURCES`` (the endpoints and fields), ``DATE_FIELD``
    and ``DATE_VARIABLE`` (the vendor date a row is placed from and its
    variable) and ``LAG`` (added to that date; ``None`` places a row on the
    date itself). Rows of several sources are joined by series and date. Of
    rows of one series placed on one session, the one with the latest date
    is shown. See the module docstring for the rules.

    Parameters
    ----------
    config : StockFitFundConfig
        StockFit's raw tier, the window and the series kept.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitFundFlowsDataset(config).update().panel("2024-01-02", "2024-12-31")["flow_sales_mon1"]
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = StockFitFundConfig

    #: The endpoints read, in join order.
    SOURCES: tuple[FundSource, ...] = ()

    #: The vendor field each row is dated by.
    DATE_FIELD = "reportDate"

    #: The variable holding that date.
    DATE_VARIABLE = "report_date"

    #: Added to the date before placing; ``None`` places a row on its date.
    LAG: timedelta | None = None

    # -- the release table --------------------------------------------------------

    def _variables(self) -> dict[str, dict]:
        variables = {name: attrs for source in self.SOURCES for name, attrs in source.fields.values()}
        return {**variables, self.DATE_VARIABLE: {"description": f"The row's {self.DATE_FIELD}."}}

    def _spec(self) -> ReleaseSpec:
        """Return the variables, the row's date among them; staleness is measured from that date."""
        return ReleaseSpec(variables=self._variables(), dates=(self.DATE_VARIABLE,), stamp=self.DATE_VARIABLE)

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no StockFit {self.SOURCES[0].endpoint} row is available"

    def _vendor_root(self):
        return self.config.raw_data_dir_path

    def _series(self, endpoint: str) -> list[str]:
        """The series of ``endpoint`` with a pull, those of the config's ``symbols`` when set."""
        found = raw.entities(self._vendor_root(), endpoint)
        if self.config.symbols is not None:
            wanted = {str(symbol) for symbol in self.config.symbols}
            found = [series for series in found if series in wanted]
        return found

    def _trading_calendar(self) -> TradingCalendar:
        """Return NYSE's sessions up to the day before the endpoints read were last pulled."""
        if self._calendar_cache is None:
            self._calendar_cache = nyse_calendar(
                self._vendor_root(), tuple(source.endpoint for source in self.SOURCES),
                start=self.config.start_date, end=self.config.end_date, owner=self.class_name,
            )
        return self._calendar_cache

    def _rows(self, source: FundSource, series: str) -> list[dict]:
        """The rows of one series' newest pull of ``source``."""
        body = raw.read_pull(self._vendor_root(), source.endpoint, series)
        if source.records is not None:
            body = body.get(source.records) if isinstance(body, dict) else None
        return [row for row in body or [] if isinstance(row, dict)]

    def _keep(self, row: dict, series: str) -> bool:
        """Whether a row of ``series`` is shown; every row by default."""
        return True

    def _source_frame(self, source: FundSource) -> pl.DataFrame:
        """One source's rows: ``symbol``, ``date`` and its variables, the first row per series and date."""
        schema = {"symbol": pl.String, "date": pl.Date, **{name: pl.Float64 for name, _ in source.fields.values()}}
        records = [
            {
                "symbol": series,
                "date": str(row.get(self.DATE_FIELD) or "")[:10] or None,
                **{name: as_number(row.get(vendor_field)) for vendor_field, (name, _) in source.fields.items()},
            }
            for series in self._series(source.endpoint)
            for row in self._rows(source, series)
            if self._keep(row, series)
        ]
        frame = pl.DataFrame(
            records, schema={**schema, "date": pl.String}, orient="row"
        ) if records else pl.DataFrame(schema={**schema, "date": pl.String})
        return (
            frame.with_columns(pl.col("date").str.to_date(strict=False))
            .filter(pl.col("date").is_not_null())
            .unique(["symbol", "date"], keep="first", maintain_order=True)
        )

    def _released(self) -> pl.DataFrame:
        """Return every series' rows at their available dates, one per series and session.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the fund series), ``available``, the date variable
            and every variable, sorted by ``available``.
        """
        frame = None
        for source in self.SOURCES:
            part = self._source_frame(source)
            frame = part if frame is None else frame.join(part, on=["symbol", "date"], how="full", coalesce=True)
        placed = self._trading_calendar().place(frame, on="date", lag=self.LAG)
        return (
            placed.with_columns(pl.col("date").cast(pl.Datetime("ns")).alias(self.DATE_VARIABLE))
            .sort("date")
            .unique(["symbol", "available"], keep="last", maintain_order=True)
            .select("symbol", "available", *self._variables())
            .sort("available", "symbol")
        )


class StockFitFundProfileDataset(StockFitFundPanel):
    """The N-PORT summary of every fund series, each quarter from quarter end + 60 days.

    A row whose ``seriesId`` is not the series' own (a ticker StockFit
    resolved to another series) is not shown. See the module docstring.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitFundProfileDataset(config).update().panel("2024-01-02", "2024-12-31")["net_assets"]
    """

    SOURCES = (FundSource("fund_profile", {
        "totalAssets": ("total_assets", _usd("Total assets at the quarter end (N-PORT).")),
        "totalLiabilities": ("total_liabilities", _usd("Total liabilities at the quarter end (N-PORT).")),
        "netAssets": ("net_assets", _usd("Net assets at the quarter end (N-PORT).")),
        "returnMon1": ("return_mon1", _percent("Return of the quarter's first month (N-PORT).")),
        "returnMon2": ("return_mon2", _percent("Return of the quarter's second month (N-PORT).")),
        "returnMon3": ("return_mon3", _percent("Return of the quarter's third month (N-PORT).")),
        "totalSales": ("total_sales", _usd("Shares sold over the quarter, incl. reinvestments (N-PORT).")),
        "totalRedemptions": ("total_redemptions", _usd("Shares redeemed over the quarter (N-PORT).")),
        "holdingCount": ("holding_count", {"unit": "holdings", "description": "Holdings at the quarter end."}),
    }),)
    LAG = NPORT_LAG

    def _keep(self, row: dict, series: str) -> bool:
        """Keep the series' own rows."""
        return row.get("seriesId") in (None, raw.fund_series_key(series)[1])


class StockFitFundFlowsDataset(StockFitFundPanel):
    """The N-PORT monthly flows of every fund series, each quarter from quarter end + 60 days.

    Month 1 is the quarter's first month. See the module docstring.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitFundFlowsDataset(config).update().panel("2024-01-02", "2024-12-31")["flow_redemption_mon3"]
    """

    SOURCES = (FundSource("fund_flows", {
        **{
            f"{vendor}Mon{month}": (f"{name}_mon{month}", _usd(f"{what} in the quarter's month {month} (N-PORT)."))
            for vendor, name, what in (
                ("flowSales", "flow_sales", "Shares sold"),
                ("flowReinvest", "flow_reinvest", "Shares sold in reinvestment of distributions"),
                ("flowRedemption", "flow_redemption", "Shares redeemed or repurchased"),
            )
            for month in (1, 2, 3)
        },
        **{f"returnMon{m}": (f"return_mon{m}", _percent(f"Return of the quarter's month {m} (N-PORT)."))
           for m in (1, 2, 3)},
        "netAssets": ("net_assets", _usd("Net assets at the quarter end (N-PORT).")),
        "totalAssets": ("total_assets", _usd("Total assets at the quarter end (N-PORT).")),
    }),)
    LAG = NPORT_LAG


class StockFitFundFeesDataset(StockFitFundPanel):
    """The N-1A prospectus fee table of every fund series, from each filing's date.

    Fees are fractions of net assets (0.0003 is 3 basis points), for the
    share class of the ticker the series was asked for by. See the module
    docstring.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitFundFeesDataset(config).update().panel("2024-01-02", "2024-12-31")["total_expense_ratio"]
    """

    SOURCES = (FundSource("fund_fees", {
        "managementFee": ("management_fee", _ratio("Management fee (N-1A).")),
        "distribution12b1Fee": ("distribution_12b1_fee", _ratio("Distribution (12b-1) fee (N-1A).")),
        "otherExpenses": ("other_expenses", _ratio("Other expenses (N-1A).")),
        "acquiredFundFees": ("acquired_fund_fees", _ratio("Acquired fund fees and expenses (N-1A).")),
        "totalExpenseRatio": ("total_expense_ratio", _ratio("Total annual operating expenses (N-1A).")),
        "feeWaiver": ("fee_waiver", _ratio("Fee waiver or expense reimbursement (N-1A).")),
        "netExpenseRatio": ("net_expense_ratio", _ratio("Expenses after the waiver (N-1A).")),
        "maxSalesChargePurchase": ("max_sales_charge_purchase", _ratio("Maximum sales charge on purchases (N-1A).")),
        "maxDeferredSalesCharge": ("max_deferred_sales_charge", _ratio("Maximum deferred sales charge (N-1A).")),
        "redemptionFee": ("redemption_fee", _ratio("Redemption fee (N-1A).")),
    }),)
    DATE_FIELD = "filedDate"
    DATE_VARIABLE = "filed_date"


class StockFitFundNcenDataset(StockFitFundPanel):
    """The N-CEN fields of every fund series, each reporting period from its end + 75 days.

    Yes/no answers are 1.0 or 0.0. The fund-health ``liquidity``,
    ``fairValue``, ``defaults`` and provider lists describe the latest filing
    only and stay raw. See the module docstring.

    Examples
    --------
    With ``config`` as in the module example::

        StockFitFundNcenDataset(config).update().panel("2024-01-02", "2024-12-31")["did_lend"]
    """

    SOURCES = (
        FundSource("fund_health", {
            "isIndexFund": ("is_index_fund", _flag("The fund is an index fund (N-CEN).")),
            "indexReturnDiffBefore": ("index_return_diff_before", _percent(
                "Return difference to the index, before fees (N-CEN).")),
            "indexReturnDiffAfter": ("index_return_diff_after", _percent(
                "Return difference to the index, after fees (N-CEN).")),
            "indexStdevBefore": ("index_stdev_before", _percent(
                "Standard deviation of the return difference, before fees (N-CEN).")),
            "indexStdevAfter": ("index_stdev_after", _percent(
                "Standard deviation of the return difference, after fees (N-CEN).")),
            "isLendingAuthorized": ("is_lending_authorized", _flag("Securities lending is authorized (N-CEN).")),
            "didLend": ("did_lend", _flag("The fund lent securities in the period (N-CEN).")),
            "isLendingLiquidated": ("is_lending_liquidated", _flag(
                "A borrower default made the fund liquidate collateral (N-CEN).")),
            "hasMaterialWeakness": ("has_material_weakness", _flag(
                "The auditor reported a material weakness (N-CEN).")),
            "hasLegalProceedings": ("has_legal_proceedings", _flag("Material legal proceedings (N-CEN).")),
            "hasValuationChanges": ("has_valuation_changes", _flag("Changes in valuation methods (N-CEN).")),
            "hasAccountingChanges": ("has_accounting_changes", _flag("Changes in accounting principles (N-CEN).")),
            "hasPaymentErrors": ("has_payment_errors", _flag("Errors in net asset value or dividends (N-CEN).")),
        }, records="ncenRecords"),
        FundSource("fund_fee_analysis", {
            "isExpenseLimitation": ("is_expense_limitation", _flag("An expense limitation is in place (N-CEN).")),
            "isExpenseReducedOrWaived": ("is_expense_reduced_or_waived", _flag(
                "Expenses were reduced or waived (N-CEN).")),
            "isFeesWaivedRecoupable": ("is_fees_waived_recoupable", _flag(
                "Waived fees may be recouped (N-CEN).")),
            "monthlyAvgNetAssets": ("monthly_avg_net_assets", _usd("Average monthly net assets (N-CEN).")),
            "aggregateCommission": ("aggregate_commission", _usd("Aggregate brokerage commissions (N-CEN).")),
            "creationUnitShares": ("creation_unit_shares", {
                "unit": "shares", "description": "Shares per creation unit (N-CEN)."}),
            "avgPctPurchasedInKind": ("avg_pct_purchased_in_kind", _percent(
                "Average share of creations paid in kind (N-CEN).")),
            "avgPctRedeemedInKind": ("avg_pct_redeemed_in_kind", _percent(
                "Average share of redemptions paid in kind (N-CEN).")),
            "hasLineOfCredit": ("has_line_of_credit", _flag("The fund has a line of credit (N-CEN).")),
            "lineOfCreditSize": ("line_of_credit_size", _usd("Size of the line of credit (N-CEN).")),
        }, records="feeRecords"),
    )
    LAG = NCEN_LAG
