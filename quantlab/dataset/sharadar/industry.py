"""Point-in-time Fama-French 48 industry of each Sharadar security, keyed by permaticker.

TICKERS gives each security's *current* SIC code (``siccode``); its
``famaindustry``, ``sicindustry`` and ``sector`` are current snapshots too,
and none of them is used here. The SIC history is in ACTIONS: a change is a
pair of rows on one date, ``sicchangefrom`` (the old code) and
``sicchangeto`` (the new one). The history is rebuilt by walking back from
the current code: on a day, a security's SIC is the ``sicchangefrom`` of its
first change dated after that day, or the current code when no change is
later. A change dated on a weekend or holiday therefore shows from the next
trading day. Every change in ACTIONS is walked through, even one dated after
the days the panel holds: TICKERS is pulled whole, so its current code
already reflects it. ``sicchangeto`` is not read. On the 2026-10-05 pull it disagrees
with the next change's ``sicchangefrom`` for 19 of 2,927 changes, and the
last change's ``sicchangeto`` differs from the current ``siccode`` for 36 of
2,575 securities. In both cases the walk back keeps the code that held
afterwards, the current one.

``SharadarIndustryDataset`` classifies each day's SIC into the Fama-French
48 industries (``quantlab.dataset._support.ff48``; a SIC inside no range is
48, "Other"), merges the thin industries named in
``config.industry_merge``, and stores the code as one float variable,
``industry``, on SEP's trading days. A security's industry is shown from its
TICKERS ``firstpricedate`` to its ``lastpricedate`` and is NaN outside them,
or where its SIC is unknown. The ``industry`` variable's ``names`` attribute maps
each code that can appear (as a string) to French's short name.

ACTIONS is keyed by the same current ticker as SEP, so its rows are mapped
through SEP's TICKERS rows; a row of a ticker outside SEP is another security
and is dropped. Two changes of one security on one date are refused. The
store grows with ``update()``: new days are appended and a stored day is
never rewritten.

Examples
--------
Build the store and read a year::

    config = SharadarIndustryConfig(
        zarr_file_path="/data/quantlab/market/sharadar/sharadar_industry_1d/sharadar_industry_1d.zarr",
        raw_data_dir_path="/data/quantlab/downloads/sharadar",
    )
    SharadarIndustryDataset(config).update()
    panel = SharadarIndustryDataset(config).panel("2024-01-02", "2024-12-31")
    panel["industry"].attrs["names"]["44"]  # 'Banks'
"""

from __future__ import annotations

import dataclasses

import numpy as np
import polars as pl

from quantlab.dataset._support.ff48 import (
    apply_merge,
    check_merge,
    ff48_codes,
    short_names,
)
from quantlab.dataset.config import DatasetConfig, SharadarIndustryConfig
from quantlab.dataset.sharadar.spells import ERROR_SAMPLE, SpellPanelDataset
from quantlab.dataset.sharadar.permatickers import PermatickerResolver
from quantlab.dataset.sharadar.tables import (
    scan_raw_table,
    table,
)
from quantlab.dataset.sharadar.universe import universe

#: The ACTIONS type holding a SIC change's old code.
SIC_CHANGE_FROM = "sicchangefrom"

#: Name of the panel's one variable.
VARIABLE = "industry"


class SharadarIndustryDataset(SpellPanelDataset):
    """Daily panel of each security's point-in-time Fama-French 48 industry code.

    Parameters
    ----------
    dataset_config : SharadarIndustryConfig
        ``raw_data_dir_path`` is ``<download-dir>/sharadar``, holding the
        ``tickers``, ``actions`` and ``sep`` (the calendar) raw tables.

    Examples
    --------
    With ``config`` as in the module example::

        ds = SharadarIndustryDataset(config).update()
        list(ds.panel("2024-01-02", "2024-01-05").data_vars)  # ['industry']
    """

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = SharadarIndustryConfig
    VARIABLE = VARIABLE
    RAW_TABLES = ("sep", "actions")

    def _normalize_config(self, config: DatasetConfig) -> SharadarIndustryConfig:
        """Normalise as ``SpellPanelDataset`` does, then check the merge mapping.

        Raises
        ------
        TypeError
            If ``config`` is not a ``SharadarIndustryConfig``.
        ValueError
            If ``table`` is not ``"sep"``, a universe field is invalid
            (``quantlab.dataset.sharadar.universe.normalize_universe``) or
            ``industry_merge`` is refused (``quantlab.dataset._support.ff48.check_merge``).

        Examples
        --------
        A merge naming a code outside 1..48 is refused::

            SharadarIndustryDataset(dataclasses.replace(config, industry_merge=((27, 49),)))
            # ValueError: SharadarIndustryDataset: the industry merge mapping names unknown code(s) [49]; ...
        """
        config = super()._normalize_config(config)
        return dataclasses.replace(
            config, industry_merge=check_merge(config.industry_merge, self.class_name)
        )

    def _variable_attrs(self) -> dict:
        """Return the scheme and each code's short name."""
        return {"scheme": "Fama-French 48", "names": short_names(self.config.industry_merge)}

    def _securities(self) -> pl.DataFrame:
        """Return each kept security's current SIC and price dates, from SEP's TICKERS rows."""
        return (
            scan_raw_table(self.config.raw_data_dir_path, "tickers")
            .filter(
                pl.col("table").is_in(table("sep").tickers_labels)
                & pl.col("permaticker").is_in(universe(self.config))
            )
            .select("permaticker", "siccode", "firstpricedate", "lastpricedate")
            .unique()
            .collect()
        )

    def _changes(self, securities: pl.DataFrame) -> pl.DataFrame:
        """Return each kept security's SIC changes: ``permaticker``, ``date`` and the old ``sic``.

        Raises
        ------
        ValueError
            If a security has two changes on one date.
        """
        root = self.config.raw_data_dir_path
        # ACTIONS is keyed by ticker, each raw file by the tickers of its own
        # pull; a row mapping to no SEP security is another one's, dropped.
        resolver = PermatickerResolver(root, "sep")
        changes = (
            scan_raw_table(root, "actions", annotate=resolver.annotate)
            # Every change, however late: the current code walked back from
            # is TICKERS', which already holds them all.
            .filter(pl.col("action") == SIC_CHANGE_FROM)
            .select("date", "ticker", pl.col("value").alias("sic"), "permaticker")
            .collect()
            .filter(pl.col("permaticker").is_not_null())
            .join(securities.select("permaticker"), on="permaticker", how="semi")
        )
        repeated = (
            changes.group_by("permaticker", "date").len().filter(pl.col("len") > 1)
            .sort("permaticker", "date")
        )
        if repeated.height:
            sample = [
                f"{row['permaticker']}@{row['date']}"
                for row in repeated.head(ERROR_SAMPLE).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {repeated.height} (permaticker, date) "
                f"pair(s) have several SIC changes in ACTIONS, first {sample}; "
                f"refusing rather than choosing one."
            )
        return changes.select("permaticker", "date", "sic")

    def _build_spells(self) -> pl.DataFrame:
        """Return every security's industry spells.

        Returns
        -------
        pl.DataFrame
            ``symbol`` (the permaticker), ``start`` (inclusive, null when
            open), ``end`` (exclusive, null when open), ``industry`` (the
            merged code, NaN for an unknown SIC), ``first`` and ``last``
            (the security's first and last price dates).

        Raises
        ------
        ValueError
            If a permaticker has two SEP TICKERS rows, or two SIC changes
            on one date.
        """
        securities = self._securities()
        twice = securities.filter(pl.col("permaticker").is_duplicated())
        if twice.height:
            raise ValueError(
                f"{self.class_name}: permaticker(s) "
                f"{twice.get_column('permaticker').unique().sort().head(ERROR_SAMPLE).to_list()} "
                f"have several SEP rows in TICKERS; re-pull TICKERS."
            )
        changes = self._changes(securities).sort("permaticker", "date")
        # Before each change its old code held, back to the previous change.
        before = changes.select(
            "permaticker",
            pl.col("date").shift(1).over("permaticker").alias("start"),
            pl.col("date").alias("end"),
            pl.col("sic").cast(pl.Float64),
        )
        # From the last change on (or always, without one) the current code holds.
        current = securities.join(
            changes.group_by("permaticker").agg(pl.col("date").max().alias("start")),
            on="permaticker",
            how="left",
        ).select(
            "permaticker",
            "start",
            pl.lit(None, dtype=pl.Date).alias("end"),
            pl.col("siccode").cast(pl.Float64).alias("sic"),
        )
        spells = pl.concat([before, current]).join(
            securities.select(
                "permaticker",
                pl.col("firstpricedate").alias("first"),
                pl.col("lastpricedate").alias("last"),
            ),
            on="permaticker",
        )
        sic = spells.get_column("sic").fill_null(np.nan).to_numpy()
        industry = apply_merge(ff48_codes(sic), self.config.industry_merge)
        spells = spells.select(
            pl.col("permaticker").alias("symbol"), "start", "end", "first", "last"
        ).with_columns(pl.Series(VARIABLE, industry))
        return spells

