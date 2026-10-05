"""Which permatickers a Sharadar panel converts: the rules every Sharadar panel shares.

A panel either names an explicit *roster* (``permatickers``, the members of
an index through ``roster_universe``, or both), which is converted whole and
never filtered, or converts the *market universe* of its table, filtered by
the TICKERS ``category`` (``category_filter``). The price panels and the
fundamentals panel apply the same rules through these two functions.
"""

from __future__ import annotations

import dataclasses
from datetime import date

import polars as pl

from quantlab.dataset.config import SharadarDatasetConfig
from quantlab.dataset.sharadar.membership import ROSTER_UNIVERSES, sp500_intervals
from quantlab.dataset.sharadar.tables import scan_raw_table, table


def normalize_universe(config: SharadarDatasetConfig, owner: str) -> SharadarDatasetConfig:
    """Check and normalise a Sharadar config's universe fields.

    Refuses any ``symbols`` value, an empty ``permatickers`` and an unknown
    ``roster_universe``. ``permatickers`` is returned as a tuple of ints,
    and ``category_filter="default"`` as the table's own default
    (``SharadarTable.categories``).

    Parameters
    ----------
    config : SharadarDatasetConfig
        The config to check.
    owner : str
        Named in error messages.

    Raises
    ------
    ValueError
        For an invalid field.

    Examples
    --------
    SEP's default category filter is resolved::

        normalize_universe(config, "demo").category_filter
        # ('Domestic Common Stock', 'Domestic Common Stock Primary Class',
        #  'Domestic Common Stock Secondary Class')
    """
    if config.symbols is not None:
        raise ValueError(
            f"{owner}: config.symbols is not selectable on a "
            f"Sharadar panel; got {config.symbols!r}. The symbol axis is "
            f"the permaticker, and a ticker can be renamed or reused. Use "
            f"config.permatickers instead."
        )
    if config.permatickers is not None:
        permatickers = tuple(int(value) for value in config.permatickers)
        if not permatickers:
            raise ValueError(
                f"{owner}: config.permatickers is empty. Pass "
                f"None for every security, or name at least one."
            )
        config = dataclasses.replace(config, permatickers=permatickers)
    if config.roster_universe is not None and config.roster_universe not in ROSTER_UNIVERSES:
        raise ValueError(
            f"{owner}: roster_universe {config.roster_universe!r} "
            f"is not a Sharadar universe; known: {ROSTER_UNIVERSES}."
        )
    if config.category_filter == "default":
        config = dataclasses.replace(
            config, category_filter=table(config.table).categories
        )
    elif isinstance(config.category_filter, str):
        raise ValueError(
            f"{owner}: config.category_filter must be 'default', "
            f"None or a tuple of categories; got {config.category_filter!r}."
        )
    if config.category_filter is not None:
        categories = tuple(str(value) for value in config.category_filter)
        if not categories:
            raise ValueError(
                f"{owner}: config.category_filter is empty. Pass "
                f"None to keep every category, or name at least one."
            )
        config = dataclasses.replace(config, category_filter=categories)
    return config


def universe(config: SharadarDatasetConfig) -> list[int]:
    """Return the permatickers a conversion of ``config`` keeps.

    An explicit roster (``permatickers``, the members of ``roster_universe``
    inside the window, or both) is kept whole. Without one, the market
    universe is every permaticker whose TICKERS ``category`` (on the rows of
    ``config.table``) is in ``category_filter``, or every one when that is
    ``None``.

    Examples
    --------
    An explicit roster is kept as named::

        universe(dataclasses.replace(config, permatickers=(101,)))  # [101]
    """
    root = config.raw_data_dir_path
    if config.permatickers is None and config.roster_universe is None:
        tickers = scan_raw_table(root, "tickers").filter(
            pl.col("table").is_in(table(config.table).tickers_labels)
        )
        if config.category_filter is not None:
            tickers = tickers.filter(
                pl.col("category").is_in(list(config.category_filter))
            )
        return tickers.select("permaticker").unique().collect().to_series().to_list()
    roster = set(config.permatickers or ())
    if config.roster_universe is not None:
        start = date.fromisoformat(config.start_date)
        end = date.fromisoformat(config.end_date)
        spells = sp500_intervals(root).filter(
            (pl.col("start_date") <= end) & (pl.col("end_date") >= start)
        )
        roster.update(spells.get_column("permaticker").to_list())
    return sorted(roster)
