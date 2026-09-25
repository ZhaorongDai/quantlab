"""PERMNO-to-ticker intervals derived from CRSP's security-information history.

CRSP (the Center for Research in Security Prices) is a US stock database
sold through WRDS (Wharton Research Data Services). A PERMNO is CRSP's
permanent integer id for one security; unlike a ticker, it never changes and
is never reused. The CRSP price panel uses the PERMNO as its ``symbol``
axis, so a ticker is only a display name here, never an identity.

``CrspSymbology`` turns the ``stksecurityinfohist`` reference table (CRSP's
history of each security's descriptive fields) into a table of intervals
saying which ticker each PERMNO used over which dates. The conversion writes
that table into the ticker *sidecar*, a JSON file next to the Zarr store,
which ``quantlab.dataset.crsp.tickers`` reads back.

The daily price table cannot spell a share class on its own (it has no
``shareclass`` and no ``tradingsymbol`` column), so the names come from the
downloaded reference tables rather than from the daily rows.

The naming rule, applied per interval:

1. ``base`` is ``ticker`` stripped and upper-cased.
2. ``cls`` is ``shareclass``, unless it is null, empty, ``"None"`` or
   ``"NONE"`` (the last two occur as literal text in the live tables).
3. If ``cls`` is set and ``tradingsymbol == base + cls``, the symbol is
   ``base.cls`` (BRK with trading symbol BRKB and class B gives ``BRK.B``).
   Otherwise it is ``base`` (GOOGL, META, FB).
4. An interval whose ticker is null or empty keeps the previous interval's
   symbol for that PERMNO. The interval for a delisting day usually looks
   like this, and carrying the name forward lets the sidecar name a
   delisted security on its last day.

Nothing here merges two PERMNOs or decides which rows enter a panel.
"""

from __future__ import annotations

import polars as pl

#: The delimiter between a base ticker and its share class. It is the same
#: ``.`` that the constituent universes and the TAQ (tick data) download use;
#: it is repeated here rather than imported so this daily-data module does
#: not depend on a tick-data module.
SUFFIX_DELIMITER = "."

#: Share-class values that mean "no class". ``"None"`` and ``"NONE"`` are
#: text, not nulls: the live tables hold a real SQL null on some rows and
#: the four-character string on others.
_NO_CLASS = ("", "None", "NONE")


class CrspSymbology:
    """Ticker intervals per PERMNO, computed once from ``stksecurityinfohist``.

    The one public method, ``symbol_intervals``, caches its result on the
    instance because a conversion reads it more than once.

    Parameters
    ----------
    security_info : pl.DataFrame
        The ``stksecurityinfohist`` table as ``CrspReference.table`` returns
        it.

    Attributes
    ----------
    security_info : pl.DataFrame
        The table given to the constructor.

    Examples
    --------
    >>> from quantlab.dataset.crsp.reference import CrspReference
    >>> from quantlab.dataset.crsp.symbology import CrspSymbology
    >>> ref = CrspReference("data/downloads/us_equity/1d/wrds_crsp/_reference")
    >>> symbology = CrspSymbology(ref.table("stksecurityinfohist"))
    >>> symbology.symbol_intervals().filter(pl.col("permno") == 13407)
    shape: (2, 4)
    ┌────────┬────────┬────────────┬────────────┐
    │ permno ┆ symbol ┆ start_date ┆ end_date   │
    │ ---    ┆ ---    ┆ ---        ┆ ---        │
    │ i64    ┆ str    ┆ date       ┆ date       │
    ╞════════╪════════╪════════════╪════════════╡
    │ 13407  ┆ FB     ┆ 2012-05-18 ┆ 2022-06-08 │
    │ 13407  ┆ META   ┆ 2022-06-09 ┆ 2025-12-31 │
    └────────┴────────┴────────────┴────────────┘
    """

    SUFFIX_DELIMITER = SUFFIX_DELIMITER

    def __init__(self, security_info: pl.DataFrame) -> None:
        """Initialize from the security-info table; nothing is computed until first use."""
        self.security_info = security_info
        self._intervals: pl.DataFrame | None = None

    # -- intervals ----------------------------------------------------------

    def symbol_intervals(self) -> pl.DataFrame:
        """Return ``(permno, symbol, start_date, end_date)``, one row per interval.

        The rows are sorted by ``(permno, start_date)``. ``permno`` is
        ``Int64``, ``symbol`` is ``String`` and both dates are ``Date``. The
        result is computed once and cached. These four columns are also the
        schema of the ticker sidecar written next to a converted store.

        ``symbol`` is null only where a PERMNO's first interval already has
        no ticker, so there is no earlier name to carry forward. The nulls
        are kept so "this PERMNO had no ticker yet" stays visible; call
        ``.drop_nulls("symbol")`` to keep only named intervals.

        Returns
        -------
        pl.DataFrame
            One row per interval with columns ``permno``, ``symbol``,
            ``start_date`` and ``end_date``.

        Examples
        --------
        >>> symbology.symbol_intervals().filter(pl.col("permno") == 83443)
        shape: (2, 4)
        ┌────────┬────────┬────────────┬────────────┐
        │ permno ┆ symbol ┆ start_date ┆ end_date   │
        │ ---    ┆ ---    ┆ ---        ┆ ---        │
        │ i64    ┆ str    ┆ date       ┆ date       │
        ╞════════╪════════╪════════════╪════════════╡
        │ 83443  ┆ BRK    ┆ 1996-05-09 ┆ 2002-01-01 │
        │ 83443  ┆ BRK.B  ┆ 2002-01-02 ┆ 2025-12-31 │
        └────────┴────────┴────────────┴────────────┘
        """
        if self._intervals is not None:
            return self._intervals

        frame = self.security_info.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("secinfostartdt").cast(pl.Date).alias("start_date"),
            pl.col("secinfoenddt").cast(pl.Date).alias("end_date"),
            pl.col("ticker").cast(pl.String),
            pl.col("tradingsymbol").cast(pl.String),
            pl.col("shareclass").cast(pl.String),
        ).sort(["permno", "start_date"])

        base = pl.col("ticker").str.strip_chars().str.to_uppercase()
        cls = (
            pl.when(
                pl.col("shareclass").is_null()
                | pl.col("shareclass").str.strip_chars().is_in(_NO_CLASS)
            )
            .then(None)
            .otherwise(pl.col("shareclass").str.strip_chars().str.to_uppercase())
        )
        trading = pl.col("tradingsymbol").str.strip_chars().str.to_uppercase()

        frame = frame.with_columns(base.alias("_base"), cls.alias("_cls"))
        frame = frame.with_columns(
            pl.when(
                pl.col("_cls").is_not_null()
                & trading.is_not_null()
                & (trading == pl.col("_base") + pl.col("_cls"))
            )
            .then(pl.col("_base") + pl.lit(SUFFIX_DELIMITER) + pl.col("_cls"))
            .otherwise(pl.col("_base"))
            .alias("_symbol")
        )
        # An empty ticker means the same as a null one; both must reach the
        # forward fill below instead of becoming the symbol "".
        frame = frame.with_columns(
            pl.when(
                pl.col("_base").is_null() | (pl.col("_base").str.len_chars() == 0)
            )
            .then(None)
            .otherwise(pl.col("_symbol"))
            .alias("_symbol")
        )
        # Rule 4: carry the name forward within one PERMNO, never across two.
        frame = frame.with_columns(
            pl.col("_symbol").forward_fill().over("permno").alias("_symbol")
        )

        self._intervals = frame.select(
            pl.col("permno"),
            pl.col("_symbol").alias("symbol"),
            pl.col("start_date"),
            pl.col("end_date"),
        ).sort(["permno", "start_date"])
        return self._intervals
