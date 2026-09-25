"""Point-in-time index-membership panels, independent of any specific index.

An index such as the S&P 500 changes its members over time. A
*point-in-time* membership record says which symbols were in the index on
each past date, as known on that date. Using today's member list for the
past instead would introduce *survivorship bias*: backtests would only see
companies that survived to the present.

``IndexConstituentDataset`` turns a table of membership intervals (one row
per ``symbol``, ``start_date``, ``end_date``) into a *panel*, an
``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, with one boolean
variable ``is_member``, and stores it like any other dataset. It knows
nothing about a particular index. A concrete index in
``quantlab/dataset/constituent.py`` supplies the interval table and the
earliest date its source can answer for. The main consumer is a *universe
mask*: a filter that restricts a price panel to the symbols that were index
members on each date.
"""

from abc import abstractmethod

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from loguru import logger

from quantlab.base.config import ConstituentDatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset._support.cleaning import clean_membership_panel
from quantlab.enums.constant import Date
from quantlab.utils.symbol_axis import sort_symbol_axis


class IndexConstituentDataset(BaseDataset):
    """Daily point-in-time index-membership panel built from intervals.

    A subclass implements two hooks: ``_pit_coverage_start`` (the earliest
    date the source can answer membership for) and ``_build_intervals`` (the
    membership intervals). Everything else is shared. The config setter
    raises ``start_date`` to the coverage start, ``_densify`` turns the
    intervals into the ``is_member`` grid, and ``_clean`` validates the panel
    instead of running the price-data (OHLCV) cleaner. The class derives
    from ``BaseDataset`` directly, so it has no conversions to bars, KunQuant
    arrays or Nautilus objects.

    Two conventions matter to a consumer. The ``timestamp`` axis covers every
    calendar day, weekends and holidays included, so to combine it with a
    trading-day price panel, select it onto the price timestamps rather than
    assuming the axes line up. Intervals include both end dates: a symbol
    removed on date ``D`` reads ``True`` on ``D`` and ``False`` on ``D + 1``.

    Parameters
    ----------
    config : ConstituentDatasetConfig
        The dataset config. ``as_of`` pins the right edge of the panel when
        some membership is still open; see ``_densify``.

    Examples
    --------
    A minimal index over an in-memory interval table::

        class DemoPanel(IndexConstituentDataset):
            def _pit_coverage_start(self) -> str:
                return "2020-01-01"

            def _build_intervals(self) -> pl.DataFrame:
                return pl.DataFrame(
                    [
                        ("AAA", "2020-01-01", None),
                        ("BBB", "2020-01-01", "2020-01-05"),
                        ("CCC", "2020-01-04", None),
                    ],
                    schema=["symbol", "start_date", "end_date"],
                    orient="row",
                )

    >>> config = ConstituentDatasetConfig(
    ...     zarr_file_path="data/demo.zarr",
    ...     cache_dir="data/demo_cache",
    ...     start_date="2020-01-01",
    ...     end_date="2020-01-08",
    ...     as_of="2020-01-08",
    ... )
    >>> panel = DemoPanel(config).from_raw_data().get_xarray_dataset()
    >>> panel["is_member"].to_pandas().astype(int)
    symbol      AAA  BBB  CCC
    timestamp
    2020-01-01    1    1    0
    2020-01-02    1    1    0
    2020-01-03    1    1    0
    2020-01-04    1    1    1
    2020-01-05    1    1    1
    2020-01-06    1    0    1
    2020-01-07    1    0    1
    2020-01-08    1    0    1
    """

    #: The config class a saved ``config.json`` is rebuilt with.
    config_cls = ConstituentDatasetConfig

    @property
    def config(self) -> ConstituentDatasetConfig:
        """Return the dataset config.

        Redefined here so that the setter can adjust the start date after the
        shared base-class setter has run, and to give the property the
        narrower config type.

        Examples
        --------
        >>> ds.config.start_date
        '2020-01-01'
        """
        return self._config  # type: ignore[return-value]

    @config.setter
    def config(self, config: ConstituentDatasetConfig):
        """Assign the config, then raise ``start_date`` to the coverage start.

        The base setter runs first because it fills in ``name`` and the
        default ``start_date`` and ``end_date``. The adjustment reads the
        resolved ``start_date`` and would otherwise see ``None``.

        Parameters
        ----------
        config : ConstituentDatasetConfig
            The config to assign. It is modified in place.

        Examples
        --------
        A requested date before the coverage start is raised, with a
        warning, at assignment time:

        >>> config.start_date = "2019-06-01"
        >>> ds.config = config
        >>> ds.config.start_date
        '2020-01-01'
        """
        BaseDataset.config.fset(self, config)  # type: ignore[attr-defined]
        self._clamp_coverage_start()

    def _clamp_coverage_start(self) -> None:
        """Raise ``config.start_date`` to this index's coverage start.

        The source cannot answer membership before ``_pit_coverage_start()``.
        Without this adjustment, the inherited default start date
        (``Date.START_DATE``) would add decades of all-False rows, which read
        as "not a member" when the truth is "unknown". A warning is logged
        only when the caller explicitly asked for an earlier date; the
        default is adjusted silently so that ordinary construction does not
        warn.
        """
        requested = self._config.start_date
        coverage_start = self._pit_coverage_start()
        if requested is not None and requested >= coverage_start:
            return

        self._config.start_date = coverage_start
        if requested != Date.START_DATE:
            logger.warning(
                f"{self.class_name}: requested start_date {requested} is "
                f"before this index's point-in-time coverage start "
                f"{coverage_start}; the panel's left edge was clamped to "
                f"{coverage_start}. Membership before that date cannot be "
                f"answered from the source."
            )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        """Validate the membership panel instead of running the OHLCV cleaner."""
        return clean_membership_panel(data)

    def _raw_data_to_xr(self) -> xr.Dataset:
        """Build the interval table and densify it into the daily panel."""
        return self._densify(self._build_intervals())

    def _densify(self, intervals: pl.DataFrame) -> xr.Dataset:
        """Densify a membership-interval table into a daily boolean panel.

        The symbol axis holds every symbol that appears anywhere in the
        table, sorted, and is computed before any date filtering. So a symbol
        whose membership ended before the window still gets an all-False
        column, and selecting a symbol that was never a member raises. Labels
        keep the table's own type: ticker strings give a string axis in
        alphabetical order, and integer identifiers (such as CRSP PERMNOs,
        permanent security numbers) give an int64 axis in numeric order.
        ``sort_symbol_axis`` decides the order.

        The first date is ``max(config.start_date, coverage start)``. The
        last date is ``min(config.end_date, horizon)``. ``horizon`` is the
        latest date in the table; if any interval is still open it is
        extended to ``config.as_of``, or to today when ``as_of`` is unset.
        Set ``as_of`` when the panel must be reproducible. Cutting back an
        explicitly requested ``end_date`` logs a warning; the inherited
        default (``Date.END_DATE``) is cut back silently. Each interval
        includes both of its end dates, and a null ``end_date`` runs to the
        last date of the panel.

        Parameters
        ----------
        intervals : pl.DataFrame
            A frame with ``symbol``, ``start_date`` and ``end_date`` columns.
            A null ``end_date`` means the symbol is still a member.

        Returns
        -------
        xr.Dataset
            A dataset with one boolean ``is_member`` variable on
            ``(timestamp, symbol)``, with one ``timestamp`` per calendar day.

        Raises
        ------
        ValueError
            If the table is empty, any row has a null ``start_date``, or the
            resolved date range is empty.
        """
        rows = intervals.select(
            ["symbol", "start_date", "end_date"]
        ).to_dicts()
        if not rows:
            raise ValueError(
                f"{self.class_name}: the membership-interval table is empty; "
                f"there is no membership history to densify."
            )

        # Reject null start dates: pd.Timestamp(None) is NaT, every comparison
        # with NaT is False, so max() below would return an arbitrary value and
        # the fill loop would produce an all-False column. str() is only for
        # the error message; the axis keeps the table's type.
        undated = sorted(
            {str(row["symbol"]) for row in rows if row["start_date"] is None}
        )
        if undated:
            raise ValueError(
                f"{self.class_name}: interval rows with a null start_date "
                f"cannot be densified: {undated}. A membership with no start "
                f"date is not a membership -- silently treating it as one "
                f"corrupts the panel's horizon and yields an all-False column "
                f"indistinguishable from 'never a member'."
            )

        # sort_symbol_axis orders integer identifiers numerically even when
        # they arrive as digit strings; plain sorted() would put "14593"
        # before "7000".
        symbols = sort_symbol_axis({row["symbol"] for row in rows})
        column_of = {symbol: i for i, symbol in enumerate(symbols)}

        left = pd.Timestamp(
            max(self.config.start_date, self._pit_coverage_start())
        )

        observed = max(
            pd.Timestamp(row["start_date"]) for row in rows
        )
        closed_ends = [
            pd.Timestamp(row["end_date"])
            for row in rows
            if row["end_date"] is not None
        ]
        if closed_ends:
            observed = max(observed, max(closed_ends))
        has_open_membership = any(row["end_date"] is None for row in rows)
        if has_open_membership:
            # An open membership is still current, so the last recorded change
            # is only a lower bound on the last date. config.as_of fixes it;
            # without it today's date is used, which is not reproducible.
            today = (
                pd.Timestamp(self.config.as_of)
                if self.config.as_of is not None
                else pd.Timestamp.today().normalize()
            )
            horizon = max(observed, today)
        else:
            horizon = observed

        requested_right = pd.Timestamp(self.config.end_date)
        right = min(requested_right, horizon)
        if right < requested_right and self.config.end_date != Date.END_DATE:
            logger.warning(
                f"{self.class_name}: requested end_date "
                f"{self.config.end_date} is beyond what the source supports; "
                f"the panel's right edge was truncated to "
                f"{right.strftime('%Y-%m-%d')}. Membership past that date "
                f"would be fabricated, not observed."
            )

        if right < left:
            raise ValueError(
                f"{self.class_name}: empty membership window -- resolved left "
                f"edge {left.strftime('%Y-%m-%d')} is after resolved right "
                f"edge {right.strftime('%Y-%m-%d')} (index coverage starts "
                f"{self._pit_coverage_start()}). An empty panel is never a "
                f"valid answer."
            )

        timestamps = pd.date_range(left, right, freq="D")
        values = np.zeros((len(timestamps), len(symbols)), dtype=bool)

        for row in rows:
            start = pd.Timestamp(row["start_date"])
            end = (
                pd.Timestamp(row["end_date"])
                if row["end_date"] is not None
                else right
            )
            # Include both ends: the day a removal takes effect is still a
            # membership day.
            mask = (timestamps >= start) & (timestamps <= end)
            values[mask, column_of[row["symbol"]]] = True

        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], values)},
            coords={"timestamp": timestamps, "symbol": np.asarray(symbols)},
        )

    @abstractmethod
    def _pit_coverage_start(self) -> str:
        """Return the ISO date before which the source cannot answer membership.

        The panel never starts earlier than this, whatever the config asks.
        """

    @abstractmethod
    def _build_intervals(self) -> pl.DataFrame:
        """Return the membership intervals as a polars frame.

        The frame has ``symbol``, ``start_date`` and ``end_date`` columns; a
        null ``end_date`` means the symbol is still a member.
        """
