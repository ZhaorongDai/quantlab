"""The shared shape of a Sharadar panel of SEC filings placed at their filing date.

Some Sharadar tables are events, not states: an 8-K (EVENTS) or an insider's
Form 4 (SF2) is filed on one day and says something about that day only. A
panel of them has a value on the day a filing became public and nothing
in between. ``SharadarFilingPanelDataset`` is that shape on a Sharadar
table (the shape itself is ``FilingPanelDataset`` in
``quantlab.dataset._support.point_in_time``); a subclass says which raw
rows count and how one day's filings of one security combine.

Each filing is placed at its *available date*, the first SEP trading day on
or after its filing date, so a weekend or holiday filing lands on the next
trading day and the panel lines up with the price panels bar for bar. Its
``symbol`` axis is the int64 permaticker. A day with no filing holds the
subclass's empty value (``False``, ``0.0``), never NaN: nothing was filed.
A filing made after the close lands on its filing date's bar, so a signal
formed at a bar's close should read the panel one bar back.
"""

from __future__ import annotations

import polars as pl

from quantlab.dataset._support.point_in_time import FilingPanelDataset
from quantlab.dataset.config import DatasetConfig, SharadarDatasetConfig
from quantlab.dataset.sharadar.permatickers import map_raw_table
from quantlab.dataset.sharadar.point_in_time import SepReleasePanelDataset
from quantlab.dataset.sharadar.universe import normalize_universe, universe
from quantlab.utils.timer import Timer


class SharadarFilingPanelDataset(SepReleasePanelDataset, FilingPanelDataset):
    """Daily panel of one Sharadar filing table, each filing at its available date.

    A subclass sets ``config_cls`` and ``TABLE`` and implements
    ``_empty`` (each variable's value on a day without a filing),
    ``_filings`` (the raw rows that count, with their values) and
    ``_combine`` (how one day's rows of one security add up).
    The windows, the spec and the widening fill are ``FilingPanelDataset``'s.

    Parameters
    ----------
    config : SharadarDatasetConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        filing table, ``sep`` (the calendar) and ``tickers``.
    """

    #: Code of the filing table (a key of ``quantlab.dataset.sharadar.tables.TABLES``).
    TABLE: str

    #: The columns naming one row of a security, whose rows under two of its
    #: tickers are settled (``PermatickerResolver.settle``); ``None`` when a
    #: security has several rows a day (SF2's filing lines).
    KEY: tuple[str, ...] | None = None

    def _normalize_config(self, config: DatasetConfig) -> SharadarDatasetConfig:
        """Normalise as the base class does, then check the table and the universe fields.

        Raises
        ------
        TypeError
            If ``config`` is not this dataset's ``config_cls``.
        ValueError
            If ``table`` is not ``TABLE`` or a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``).
        """
        config = super()._normalize_config(config)
        if not isinstance(config, self.config_cls):
            raise TypeError(
                f"{self.class_name} needs a {self.config_cls.__name__}, got "
                f"{type(config).__name__}."
            )
        if config.table != self.TABLE:
            raise ValueError(
                f"{self.class_name}: table must be {self.TABLE!r}, got {config.table!r}."
            )
        return normalize_universe(config, self.class_name)

    # -- the subclass's part --------------------------------------------------

    def _filings(self, raw: pl.DataFrame) -> pl.DataFrame:
        """Return the raw rows that count, with their values.

        Parameters
        ----------
        raw : pl.DataFrame
            The filing table's raw rows in the window, with a
            ``permaticker`` column added.

        Returns
        -------
        pl.DataFrame
            ``permaticker``, ``date`` (the filing date) and one column per
            variable, any number of rows per pair.
        """
        raise NotImplementedError

    def _combine(self) -> list[pl.Expr]:
        """Return one aggregation per variable, combining a security's filings of one day."""
        raise NotImplementedError

    # -- the release table ------------------------------------------------------

    def _raw_tables(self) -> tuple[str, ...]:
        """Return the filing table and SEP, the calendar."""
        return (self.TABLE, "sep")

    def _released(self) -> pl.DataFrame:
        """Return every day with a filing, per security, combined.

        Returns
        -------
        pl.DataFrame
            ``symbol``, ``available`` (the trading day the filings land on)
            and one column per variable, one row per pair, sorted by both.

        Raises
        ------
        ValueError
            If a raw ticker has two permatickers (``map_raw_table``); a row
            with none is left out and reported in ``<store>.unmapped.json``.
        """
        root = self.config.raw_data_dir_path
        calendar = self._trading_calendar()
        with Timer(f"{self.class_name}: place filings"):
            frame = map_raw_table(
                root,
                self.TABLE,
                owner=self.class_name,
                store_path=self.config.zarr_file_path,
                query=lambda scan: scan.filter(pl.col("date") <= pl.lit(calendar.through)),
                key=self.KEY,
            )
            frame = frame.filter(pl.col("permaticker").is_in(universe(self.config)))
            # A filing before the calendar's first day has no trading day of
            # its own to land on; rolling it onto the first would misdate it.
            return (
                calendar.place(self._filings(frame), drop_earlier=True)
                .group_by(pl.col("permaticker").alias("symbol"), "available")
                .agg(self._combine())
                .sort("symbol", "available")
            )

    def _nothing_shown(self) -> str:
        """Return what an empty window lacks."""
        return f"no {self.TABLE!r} filing"
