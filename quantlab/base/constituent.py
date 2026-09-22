"""Index-membership panel datasets — the interval-to-daily-grid layer (D-03).

This module holds the shared, index-agnostic half of the constituent data
layer. It deliberately knows nothing about any particular index: no URL, no
index name, no fetcher import. A concrete index binds itself by implementing
the two abstract hooks below, in `dataset/constituent.py`, which is what makes
"adding a new index needs no change to the upper layers" (DATA-06)
structurally true rather than merely intended.
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
    """A daily, point-in-time index-membership panel.

    Holds four things and nothing else:

    1. The interval-to-daily-panel densification (`_densify`), which turns a
       `(symbol, start_date, end_date)` membership-interval table into a dense
       `(timestamp, symbol) -> bool` `is_member` grid.
    2. The point-in-time left-edge clamp (`_clamp_coverage_start`), applied at
       config-assignment time so the resolved window is assertable on the
       config object itself, not only observable in the panel.
    3. The membership cleaning route (`_clean`), which replaces the inherited
       OHLCV-shaped `clean_market_data()` default.
    4. The two abstract per-index hooks, `_pit_coverage_start()` and
       `_build_intervals()` — the complete contract a new index satisfies.

    **What it deliberately excludes, and why that exclusion is load-bearing.**
    It inherits none of `MarketDataset`'s bar/catalog/KunQuant members
    (`to_nautilus`/`_to_nautilus`, `to_kunquant`/`_to_kunquant`,
    `_write_catalog`). A membership panel has no bar representation, is never
    written to a nautilus `ParquetDataCatalog`, and is not a compiled-graph
    input. That absence is the entire reason D-03 split the dataset hierarchy
    instead of making this a `MarketDataset` carrying two meaningless
    `raise NotImplementedError` stubs. `hasattr` on any of those five names is
    False here, and a test asserts it.

    **Two axis conventions a reader must know.**

    - **CALENDAR days, not trading days.** The `timestamp` axis is a
      contiguous `pd.date_range(..., freq="D")`, so it carries weekend and
      holiday rows whose value is the last trading day's membership carried
      forward. The one real consequence: a downstream join against OHLCV data,
      which only has trading days, must reindex or `.sel()` the panel onto the
      price panel's timestamps rather than assuming the two axes align. A
      trading-day axis would need a session calendar this repository does not
      have.
    - **Membership intervals are CLOSED on both ends.** A symbol removed with
      `effective_date` D reads True on D and False on D+1. This matches
      `UniverseCatalog.get_symbols_as_of`'s
      `start_date <= as_of_date` combined with
      `end_date.is_null() | (end_date >= as_of_date)` comparison exactly. A
      divergence here would make the panel and that query disagree by one day
      at every removal in history, and nothing at runtime would notice.
    """

    # D-26: the config class `quantlab/utils/module.py` rebuilds this dataset with.
    config_cls = ConstituentDatasetConfig

    # The property is redefined here (rather than left inherited) purely so
    # the setter can run `_clamp_coverage_start()` after the shared lifecycle.
    # Its narrowed return type also records that this hierarchy takes the
    # membership-panel config -- no `catalog_path`, no `market`/`frequency`.
    @property
    def config(self) -> ConstituentDatasetConfig:
        return self._config  # type: ignore[return-value]

    @config.setter
    def config(self, config: ConstituentDatasetConfig):
        # The shared lifecycle must run FIRST: it is what fills `name`, and
        # the `start_date`/`end_date` defaults out of `enums.constant.Date`.
        # The clamp below then reads the resolved `start_date`, so running it
        # before the base setter would clamp a `None`.
        BaseDataset.config.fset(self, config)  # type: ignore[attr-defined]
        self._clamp_coverage_start()

    def _clamp_coverage_start(self) -> None:
        """Raise `config.start_date` to this index's coverage start.

        Point-in-time membership before `_pit_coverage_start()` cannot be
        answered from the source at all, so a panel must never begin earlier
        (CONFLICT 5). Without the clamp, the inherited `Date.START_DATE`
        default ("1900-01-01") would prepend 76 years of all-False rows that
        read as "nobody was a member" rather than "unknown".

        The clamp announces itself only when the caller ASKED for the earlier
        date. A pre-clamp value equal to the inherited `Date.START_DATE`
        sentinel is clamped silently, and the two cases differ for exactly one
        reason: in the sentinel case the caller requested nothing, so warning
        would fire on every single default construction and train readers to
        ignore the log entirely.
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
        """Route the panel to membership validation, not market-data cleaning."""
        return clean_membership_panel(data)

    def _raw_data_to_xr(self) -> xr.Dataset:
        return self._densify(self._build_intervals())

    def _densify(self, intervals: pl.DataFrame) -> xr.Dataset:
        """Densify a membership-interval table into a daily boolean panel.

        See the class docstring for the two axis conventions. The resolution
        rules, in order:

        - **Symbol axis**: `sort_symbol_axis(set(...))` over the ALL-TIME
          union, computed before any date filtering. A symbol whose entire
          membership predates the panel window still gets a (fully False)
          column. That is the survivorship-bias guarantee, and it is also what
          makes `XrBackend.filter_by_symbol`'s `.sel(symbol=[...])` safe: a
          requested symbol that was never a member raises `KeyError` loudly
          instead of silently vanishing.

          **The labels keep the interval table's own dtype.** This method used
          to `str()` them, so a PERMNO-keyed universe came back as digit
          strings in lexicographic order and selected NOTHING against the
          int64 price panel (03.11-05). It does not convert them now: a
          ticker-keyed index still gets a textual axis in lexicographic order,
          a PERMNO-keyed one gets an int64 axis in numeric order, and the
          order contract itself lives once in
          `quantlab/utils/symbol_axis.py:sort_symbol_axis`.
        - **Left edge**: `max(config.start_date, _pit_coverage_start())`. The
          config setter has already clamped it; the `max` here keeps the panel
          correct even if a caller mutates the config afterwards.
        - **Right edge**: `min(config.end_date, horizon)`. Let `observed` be
          the latest date appearing anywhere in the interval table. If ANY
          interval is still open, `horizon = max(observed, today)`, where
          `today` is `config.as_of` when set and the wall clock otherwise;
          otherwise `horizon = observed`. Pin `as_of` when the panel must be
          reproducible -- with the clock, two rebuilds of one config produce
          two differently-shaped stores. The today-branch is load-bearing: an open
          membership is by definition current, so the last change event is only
          a LOWER BOUND on the right edge, and a panel stopping at `observed`
          would end weeks or months short of the present — exactly the live
          edge where a universe mask gets used. Conversely `Date.END_DATE`
          ("2100-01-01") is what the `min` protects against: without it, a
          default config would materialise ~45,000 days of fabricated future
          membership.
        - **Fill**: inclusive on BOTH ends; an interval with a null `end_date`
          fills through the resolved right edge rather than being dropped.

        Narrowing an explicitly requested right edge emits a `logger.warning`
        naming both dates. An `end_date` still at the inherited `Date.END_DATE`
        sentinel is truncated silently, for the same reason the left-edge
        sentinel is: the caller requested nothing.
        """
        rows = intervals.select(
            ["symbol", "start_date", "end_date"]
        ).to_dicts()
        if not rows:
            raise ValueError(
                f"{self.class_name}: the membership-interval table is empty; "
                f"there is no membership history to densify."
            )

        # A null `start_date` must be rejected, not tolerated. `max()` over
        # `pd.Timestamp(row["start_date"])` would not raise on one:
        # `pd.Timestamp(None)` is NaT and EVERY comparison against NaT is
        # False, so `max()` silently returns whichever value it happened to
        # hold first rather than the true maximum -- corrupting the horizon
        # for every symbol. The same null then reaches the fill loop below and
        # produces a silently all-False mask row. `start_date` can legitimately
        # be null today: it comes from `effective_date`, which is null if a
        # change-log row lacks a date.
        # The `str()` here is deliberate and is NOT one of the three the
        # PERMNO migration removed (03.11-05). It renders an ERROR MESSAGE
        # PAYLOAD, and a message is text whatever the axis is: on an
        # integer-keyed universe, dropping it would print `{np.int64(7000)}`
        # at the exact moment the operator most needs to read which row has no
        # date. The three that DID go are the ones whose values reach
        # `column_of` and `coords` below, i.e. the ones that decide the axis.
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

        # The labels enter the axis AS THE INTERVAL TABLE SPELLS THEM. A
        # ticker-keyed index (Wikipedia) still yields a textual axis in
        # lexicographic order; a PERMNO-keyed one (CRSP) yields an int64 axis
        # in NUMERIC order, which is what lines it up with the price panel
        # 03.11-03 produced. `sort_symbol_axis` is the single implementation of
        # that order contract -- a bare `sorted()` here would put `"14593"`
        # before `"7000"`, and today's five-digit PERMNOs make that invisible.
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
            # `config.as_of` pins this edge; `None` means "today". The default
            # is convenient but NON-REPRODUCIBLE -- rebuilding the same config
            # on two days yields two differently-shaped stores, and `save()`
            # overwrites with mode="w" -- so a pinned rebuild is available for
            # anyone who needs the artefact to be a function of the config
            # alone (CLAUDE.md 可复现性).
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
            # Inclusive on BOTH ends -- deliberately the same convention as
            # `UniverseCatalog.get_symbols_as_of`: the removal's effective
            # date IS a membership day.
            mask = (timestamps >= start) & (timestamps <= end)
            values[mask, column_of[row["symbol"]]] = True

        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], values)},
            coords={"timestamp": timestamps, "symbol": np.asarray(symbols)},
        )

    @abstractmethod
    def _pit_coverage_start(self) -> str:
        """The ISO date before which this index's membership cannot be
        answered. The panel never starts earlier than this, whatever the
        config asks for."""

    @abstractmethod
    def _build_intervals(self) -> pl.DataFrame:
        """Return this index's membership intervals as `(symbol, start_date,
        end_date)` rows, where a null `end_date` means "still a member"."""
