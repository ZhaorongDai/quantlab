"""Point-in-time index-membership panels, the index-agnostic half.

``IndexConstituentDataset`` turns a table of membership intervals
(``symbol``, ``start_date``, ``end_date``) into a dense daily boolean panel
with a single ``is_member`` variable on ``(timestamp, symbol)``, and stores
it like any other dataset. It knows nothing about any particular index; a
concrete index in ``quantlab/dataset/constituent.py`` supplies the interval
table and the earliest date its source can answer. A universe mask applied
to a price panel is the main consumer of the result.
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
    membership intervals). Everything else is shared: the config setter
    clamps ``start_date`` to the coverage start, ``_densify`` turns the
    intervals into the ``is_member`` grid, and ``_clean`` validates the panel
    instead of running the OHLCV cleaner. The class subclasses
    ``BaseDataset`` directly, so it has no bar, KunQuant or Nautilus exits.

    Two axis conventions matter to a consumer. The ``timestamp`` axis is a
    contiguous calendar-day range, weekends and holidays included, so a join
    against a trading-day price panel must select the panel onto the price
    timestamps rather than assume the axes align. Membership intervals are
    closed on both ends: a symbol removed on date ``D`` reads ``True`` on
    ``D`` and ``False`` on ``D + 1``.

    Example:
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

        Redefined here so the setter can clamp the start date after the
        shared lifecycle has run, and to narrow the return type to the
        membership-panel config.

        Example:
            >>> ds.config.start_date
            '2020-01-01'
        """
        return self._config  # type: ignore[return-value]

    @config.setter
    def config(self, config: ConstituentDatasetConfig):
        """Assign the config, then raise ``start_date`` to the coverage start.

        The base setter runs first because it fills ``name`` and the default
        ``start_date``/``end_date``; the clamp reads the resolved
        ``start_date`` and would otherwise see ``None``.

        Example:
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

        Membership before ``_pit_coverage_start()`` cannot be answered from
        the source, and without the clamp the inherited ``Date.START_DATE``
        default would prepend decades of all-False rows that read as "not a
        member" rather than "unknown". A warning is logged only when the
        caller explicitly asked for an earlier date; the default sentinel is
        clamped silently so that every default construction does not warn.
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

        The symbol axis is the sorted all-time union of the table's symbols,
        computed before any date filtering, so a symbol whose membership
        ended before the window still gets an all-False column and a
        requested symbol that was never a member fails loudly on selection.
        Labels keep the table's own dtype: tickers give a string axis in
        lexicographic order, integer identifiers an int64 axis in numeric
        order, with ``sort_symbol_axis`` deciding the order.

        The left edge is ``max(config.start_date, coverage start)``. The right
        edge is ``min(config.end_date, horizon)``, where ``horizon`` is the
        latest date in the table, extended to ``config.as_of`` (or today when
        ``as_of`` is unset) when any interval is still open. Pin ``as_of``
        when the panel must be reproducible. Narrowing an explicitly
        requested ``end_date`` logs a warning; the inherited ``Date.END_DATE``
        sentinel is truncated silently. Each interval fills inclusively on
        both ends, and a null ``end_date`` fills through the right edge.

        Args:
            intervals: A frame with ``symbol``, ``start_date`` and
                ``end_date`` columns; a null ``end_date`` means still a
                member.

        Returns:
            A dataset with one boolean ``is_member`` variable on
            ``(timestamp, symbol)`` over a contiguous daily ``timestamp``.

        Raises:
            ValueError: If the table is empty, any row has a null
                ``start_date``, or the resolved window is empty.
        """
        rows = intervals.select(
            ["symbol", "start_date", "end_date"]
        ).to_dicts()
        if not rows:
            raise ValueError(
                f"{self.class_name}: the membership-interval table is empty; "
                f"there is no membership history to densify."
            )

        # A null start_date must be rejected here: pd.Timestamp(None) is NaT,
        # every comparison against NaT is False, so max() below would return
        # an arbitrary value and the fill loop would emit an all-False row.
        # str() renders the label for the error message only; the axis
        # itself keeps the table's dtype.
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

        # Labels enter the axis as the table spells them; sort_symbol_axis
        # orders integer identifiers numerically even when they arrive as
        # digit strings, where a bare sorted() would put "14593" before
        # "7000".
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
            # An open membership is current, so the last observed change is
            # only a lower bound on the right edge. config.as_of pins it;
            # None falls back to the wall clock, which is not reproducible.
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
            # Inclusive on both ends: the removal's effective date is still a
            # membership day.
            mask = (timestamps >= start) & (timestamps <= end)
            values[mask, column_of[row["symbol"]]] = True

        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], values)},
            coords={"timestamp": timestamps, "symbol": np.asarray(symbols)},
        )

    @abstractmethod
    def _pit_coverage_start(self) -> str:
        """Return the ISO date before which membership cannot be answered.

        The panel never starts earlier than this, whatever the config asks.
        """

    @abstractmethod
    def _build_intervals(self) -> pl.DataFrame:
        """Return the membership intervals as a polars frame.

        The frame has ``symbol``, ``start_date`` and ``end_date`` columns; a
        null ``end_date`` means the symbol is still a member.
        """
