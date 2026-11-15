"""A company panel keyed by SEC CIK read on Sharadar's permatickers, through the CIK map.

Company panels of a vendor without a security master (StockFit's statements,
footnotes, insiders) are keyed by the filer's SEC CIK, which names a
company, not a security. ``CikMapDataset`` reads such a panel on the
permaticker axis of Sharadar's securities, the axis prices, factors and
backtests use. The map is Sharadar's ticker table (TICKERS): each SEP row's
CIK is the ``CIK=`` of its ``secfilings`` URL, and the security is priced
from its ``firstpricedate`` to its ``lastpricedate``.

- A company's value lands on every security its CIK has on the day: both
  classes of a two-class company carry the company's statements.
- A security shows its CIK's values only on the days it is priced (an
  unknown first or last price date counts as priced throughout); before
  and after, its cells are empty, as for a company that reported nothing.
- A CIK with no Sharadar security on a day contributes nothing that day,
  and a security without a CIK, or whose CIK the company panel lacks, is
  not on the axis.

Each day is decided from that day's securities alone. One caveat: TICKERS
holds one row per security, its current ``secfilings`` URL, so a security's
CIK is today's and is applied to its whole history. A security whose filer
changed CIK (a new holding company after a reorganization, say) shows the
current CIK's values on days the old CIK filed for it; TICKERS does not
date the change, so the map cannot. Short of that, the map adds no
look-ahead to the company panel. The map is a dataset view
(``DatasetView``) of the company panel: a factor given it in place of a
permaticker panel changes nothing else. Its calendar is the company panel's;
tradable and delisting bars are ``MarketDataset``'s rules over the view's
own panel, since the company panel holds no prices. A read records the
company panel (through its own dataset) and the map's cells in an open
``DataRecorder``. The view keeps no store and has no build path: the company
panel and TICKERS are updated by their own steps.

Examples
--------
With ``statements`` a company panel on the CIK axis (such as
``StockFitStatementsDataset``)::

    view = CikMapDataset(statements, "/data/quantlab/downloads/sharadar")
    panel = view.panel("2024-01-02", "2024-03-28")      # permatickers
    panel["revenue"].sel(symbol=195146)                  # GOOGL: Alphabet's revenue
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from quantlab.dataset.base import BaseDataset, MarketDataset
from quantlab.dataset.config import CikMapConfig
from quantlab.dataset.sharadar.tables import scan_raw_table, table
from quantlab.dataset.view import DatasetView
from quantlab.runs.record import record_read
from quantlab.utils.date_range import check_range
from quantlab.utils.symbol_axis import sort_symbol_axis

#: The SEC CIK of a TICKERS row, from its ``secfilings`` URL.
_CIK = pl.col("secfilings").str.extract(r"CIK=(\d+)").cast(pl.Int64).alias("cik")


class CikMapDataset(DatasetView):
    """A company panel on the CIK axis read on permatickers; see the module docstring.

    Parameters
    ----------
    config : CikMapConfig or BaseDataset
        The config, or the company panel with ``raw_data_dir_path`` beside it.
    raw_data_dir_path : str or os.PathLike, optional
        ``<download-dir>/sharadar``, holding the ``tickers`` raw table, when
        ``config`` is the company panel.

    Raises
    ------
    ValueError
        If ``dataset`` is not a dataset, or a permaticker has two SEP rows
        in TICKERS (on the first read).

    Examples
    --------
    With ``statements`` a company panel on the CIK axis::

        view = CikMapDataset(statements, "/data/quantlab/downloads/sharadar")
        view.inputs() == (statements,)
        # True
    """

    # Narrower type annotation for readers and type checkers only.
    config: CikMapConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = CikMapConfig

    def __init__(self, config: "CikMapConfig | BaseDataset", raw_data_dir_path=None):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, CikMapConfig):
            config = CikMapConfig(dataset=config, raw_data_dir_path=str(raw_data_dir_path))
        elif raw_data_dir_path is not None:
            raise ValueError(f"{type(self).__name__}: give raw_data_dir_path in the config, not beside it.")
        super().__init__(config)

    @property
    def company(self) -> BaseDataset:
        """The company panel on the CIK axis.

        Examples
        --------
        With ``statements`` a company panel on the CIK axis::

            CikMapDataset(statements, "/data/quantlab/downloads/sharadar").company is statements
            # True
        """
        return self.config.dataset

    def inputs(self) -> tuple[BaseDataset]:
        """Return the company panel, the one dataset ``panel`` reads (TICKERS is a raw table).

        Examples
        --------
        With ``view`` as in the class example::

            view.inputs() == (view.company,)
            # True
        """
        return (self.company,)

    def _normalize_config(self, config: CikMapConfig) -> CikMapConfig:
        """Return ``config`` with ``name`` set, refusing a company panel that is not a dataset."""
        if not isinstance(config.dataset, BaseDataset):
            raise ValueError(
                f"{self.class_name}: dataset must be a dataset, got {type(config.dataset).__name__}."
            )
        return super()._normalize_config(config)

    # -- the map ----------------------------------------------------------------

    def _company_labels(self) -> dict[int, object]:
        """Return each CIK of the company panel's store, mapped to its label there (int or text)."""
        return {int(label): label for label in self.company.stored_symbols()}

    def securities(self) -> pl.DataFrame:
        """Return every SEP security with a CIK the company panel holds.

        Returns
        -------
        pl.DataFrame
            ``permaticker``, ``cik``, ``firstpricedate`` and ``lastpricedate``
            (null when unknown), by permaticker.

        Raises
        ------
        ValueError
            If a permaticker has two SEP rows in TICKERS.

        Examples
        --------
        With ``view`` as in the class example::

            view.securities().filter(pl.col("cik") == 1652044)["permaticker"].to_list()
            # [195146, 119496]  GOOGL, GOOG
        """
        ciks = list(self._company_labels())
        securities = (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(pl.col("table").is_in(table("sep").tickers_labels))
            .select("permaticker", _CIK, "firstpricedate", "lastpricedate")
            .filter(pl.col("cik").is_in(ciks))
            .unique()
            .sort("permaticker")
            .collect()
        )
        twice = securities.filter(pl.col("permaticker").is_duplicated())
        if twice.height:
            raise ValueError(
                f"{self.class_name}: permaticker(s) {twice['permaticker'].unique().sort().head(5).to_list()} "
                f"have several SEP rows in TICKERS at {self.config.raw_data_dir_path}; re-pull TICKERS."
            )
        return securities

    def stored_symbols(self) -> list:
        """Return every permaticker whose CIK the company panel holds, sorted.

        Examples
        --------
        With ``view`` as in the class example::

            view.stored_symbols()[:3]
            # [101, 102, 303]
        """
        return sort_symbol_axis(self.securities()["permaticker"].to_list())

    def _priced(self, securities: pl.DataFrame, timestamps: np.ndarray) -> np.ndarray:
        """Return ``[time, security]`` booleans: the security is priced on the bar's day."""
        days = pd.DatetimeIndex(timestamps).normalize().values.astype("datetime64[D]")[:, None]
        first = securities["firstpricedate"].to_numpy().astype("datetime64[D]")[None, :]
        last = securities["lastpricedate"].to_numpy().astype("datetime64[D]")[None, :]
        return (np.isnat(first) | (first <= days)) & (np.isnat(last) | (days <= last))

    def panel(
        self,
        start,
        end,
        symbols: Sequence | None = None,
        variables: Sequence[str] | None = None,
    ) -> xr.Dataset:
        """Return the company panel from ``start`` to ``end`` on permatickers.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence of int, optional
            Permatickers to return, in the order given; ``None`` returns
            every one ``stored_symbols`` names.
        variables : sequence of str, optional
            Variables to keep, in the order given; ``None`` keeps every one.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``, ``symbol`` the int64
            permaticker; a security's cells are empty on the days it is not
            priced.

        Raises
        ------
        KeyError
            If ``symbols`` names a permaticker without a CIK the company
            panel holds.

        Examples
        --------
        With ``view`` as in the class example, GOOG and GOOGL carry
        Alphabet's revenue::

            panel = view.panel("2024-01-02", "2024-03-28", symbols=[195146, 119496], variables=["revenue"])
            panel["symbol"].values.tolist()
            # [195146, 119496]
        """
        check_range(start, end, f"{self.class_name}.panel()")
        securities = self.securities()
        if symbols is not None:
            wanted = list(dict.fromkeys(int(s) for s in symbols))
            off = sorted(set(wanted) - set(securities["permaticker"].to_list()))
            if off:
                raise KeyError(
                    f"{self.class_name}.panel(): permaticker(s) {off} have no CIK that "
                    f"{self.company.class_name}'s store holds."
                )
            order = pl.DataFrame({"permaticker": wanted}, schema={"permaticker": pl.Int64})
            securities = order.join(securities, on="permaticker", how="left", maintain_order="left")
        labels = self._company_labels()
        ciks = [labels[cik] for cik in sort_symbol_axis(securities["cik"].unique().to_list())]
        company = self.company.panel(start, end, symbols=ciks, variables=variables)
        mapped = company.sel(symbol=[labels[cik] for cik in securities["cik"].to_list()]).assign_coords(
            symbol=np.asarray(securities["permaticker"].to_list(), dtype=np.int64)
        )
        priced = xr.DataArray(
            self._priced(securities, company["timestamp"].values),
            dims=("timestamp", "symbol"),
            coords={"timestamp": mapped["timestamp"], "symbol": mapped["symbol"]},
        )
        fills = self._empty_values()
        panel = mapped.copy()
        for name, values in mapped.data_vars.items():
            fill = fills.get(name, np.datetime64("NaT", "ns") if values.dtype.kind == "M" else np.nan)
            panel[name] = values.where(priced, fill)
        self._record_map(securities, priced, start, end)
        return panel

    def _empty_values(self) -> dict:
        """Return the company panel's empty value of each variable that has one of its own."""
        widen = getattr(self.company, "_widen_fill_values", None)
        return dict(widen()) if callable(widen) else {}

    def _record_map(self, securities: pl.DataFrame, priced: xr.DataArray, start, end) -> None:
        """Record the map's cells read, each security's CIK on its priced days, in an open ``DataRecorder``."""

        def as_panel() -> xr.Dataset:
            ciks = np.asarray(securities["cik"].to_list(), dtype=float)[None, :]
            return xr.Dataset({"cik": priced.where(priced, False).astype(float) * ciks}).where(priced)

        record_read(self, as_panel(), store=str(self.config.raw_data_dir_path), reread=as_panel, part="cik_map")

    # -- what the view changes about the company panel's answers ---------------

    def ticker_lookup(self) -> None:
        """Return ``None``: the company panel's lookup, if any, would name CIKs, not permatickers.

        Examples
        --------
        With ``view`` as in the class example::

            view.ticker_lookup() is None
            # True
        """
        return None

    def head(self, n: int) -> pl.LazyFrame:
        """Return the company panel's first ``n`` rows on the permaticker axis.

        Each row is repeated once for every security its CIK has priced on
        the row's day, ``symbol`` that security's int64 permaticker; a row
        whose CIK has none that day is dropped.

        Parameters
        ----------
        n : int
            The number of company-panel rows to read.

        Returns
        -------
        pl.LazyFrame
            The mapped rows, in the company panel's order.

        Examples
        --------
        With ``view`` as in the class example::

            view.head(5).collect()["symbol"].dtype
            # Int64
        """
        frame = self.company.head(n).collect()
        securities = self.securities().select(
            pl.col("cik").alias("symbol"),
            "permaticker",
            pl.col("firstpricedate").cast(pl.Datetime("ns")),
            pl.col("lastpricedate").cast(pl.Datetime("ns")),
        )
        day = pl.col("timestamp").dt.truncate("1d")
        rows = (
            frame.with_row_index("_row")
            .with_columns(pl.col("symbol").cast(pl.Int64))
            .join(securities, on="symbol", how="inner")
            .filter(
                (pl.col("firstpricedate").is_null() | (pl.col("firstpricedate") <= day))
                & (pl.col("lastpricedate").is_null() | (day <= pl.col("lastpricedate")))
            )
            .sort("_row", "permaticker")
        )
        return rows.with_columns(pl.col("permaticker").alias("symbol")).select(frame.columns).lazy()

    def tradable_bars(self, prices: xr.Dataset, fill_column: str) -> xr.DataArray:
        """Return ``MarketDataset``'s rule over the view's own panel: the company panel holds no prices.

        Examples
        --------
        With ``view`` as in the class example and ``panel`` one of its panels
        holding ``revenue``::

            view.tradable_bars(panel, "revenue").dims
            # ('timestamp', 'symbol')
        """
        return MarketDataset.tradable_bars(self, prices, fill_column)

    def delisting_bars(self, prices: xr.Dataset, valuation_column: str) -> xr.DataArray:
        """Return ``MarketDataset``'s rule over the view's own panel, looked up by permaticker.

        Examples
        --------
        With ``view`` as in the class example and ``panel`` one of its panels
        holding ``revenue``::

            view.delisting_bars(panel, "revenue").dims
            # ('timestamp', 'symbol')
        """
        return MarketDataset.delisting_bars(self, prices, valuation_column)

    def _own_delisting_symbols(self) -> list:
        """Return none: the view delists by ``MarketDataset``'s rule, not a rule of its own."""
        return []

    def _to_kunquant(self, data: xr.Dataset, data_columns: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
        """Export the columns as they are named: the company panel names no OHLCV columns."""
        return MarketDataset._kunquant_arrays(data, data_columns)

    def copy(self) -> Self:
        """Return a view over a copy of the company panel and the same raw tier.

        Examples
        --------
        With ``view`` as in the class example::

            other = view.copy()
            other == view, other.company is view.company
            # (True, False)
        """
        return type(self)(self.company.copy(), self.config.raw_data_dir_path)
