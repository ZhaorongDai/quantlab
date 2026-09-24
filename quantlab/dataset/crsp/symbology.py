"""PERMNO-to-ticker intervals derived from CRSP's security-info history.

The CRSP price panel's ``symbol`` axis is the integer PERMNO, so a ticker is
never an identity here; it is a display name. ``CrspSymbology`` turns the
``stksecurityinfohist`` reference table into an interval table saying which
ticker each PERMNO wore over which dates. The conversion writes that table
into the ticker sidecar that ``quantlab.dataset.crsp.tickers`` reads back.

The daily table cannot spell a share class on its own (it has no
``shareclass`` and no ``tradingsymbol`` column), which is why the names come
from the reference tier rather than from the daily rows.

The naming rule, applied per interval:

1. ``base`` is ``ticker`` stripped and upper-cased.
2. ``cls`` is ``shareclass``, unless it is null, empty, ``"None"`` or
   ``"NONE"`` (the last two occur as literal text in the live tables).
3. If ``cls`` is set and ``tradingsymbol == base + cls``, the symbol is
   ``base.cls`` (BRK with trading symbol BRKB and class B gives ``BRK.B``).
   Otherwise it is ``base`` (GOOGL, META, FB).
4. An interval whose ticker is null or empty carries the previous interval's
   symbol for that PERMNO. A delisting-day interval is typically like this,
   and the carry is what lets the sidecar name a dead security's last day.

Nothing here merges two PERMNOs or decides which rows enter a panel.
"""

from __future__ import annotations

import polars as pl

#: The delimiter between a base ticker and its share class. The same ``.``
#: the constituent universes and the TAQ acquisition use; restated rather
#: than imported so that a daily-panel module does not depend on a
#: tick-data module.
SUFFIX_DELIMITER = "."

#: Share-class values that mean "no class". ``"None"`` and ``"NONE"`` are
#: text, not nulls: the live tables carry both a real SQL null and, on some
#: rows, the four-character string.
_NO_CLASS = ("", "None", "NONE")


class CrspSymbology:
    """Ticker intervals per PERMNO, computed once from ``stksecurityinfohist``.

    ``security_info`` is the reference table as ``CrspReference.table()``
    returns it. The single public method, ``symbol_intervals()``, is cached
    on the instance because a conversion reads it more than once.

    Example:
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
        """Bind the security-info table; nothing is computed until first use."""
        self.security_info = security_info
        self._intervals: pl.DataFrame | None = None

    # -- intervals ----------------------------------------------------------

    def symbol_intervals(self) -> pl.DataFrame:
        """Return ``(permno, symbol, start_date, end_date)``, one row per interval.

        Sorted by ``(permno, start_date)``; ``permno`` is ``Int64``,
        ``symbol`` is ``String`` and both dates are ``Date``. The result is
        computed once and cached. These four columns are also the schema of
        the ticker sidecar written beside a converted store.

        ``symbol`` is null only where a PERMNO's first interval already has no
        ticker, so there is nothing earlier to carry forward. The nulls are
        kept so that "this PERMNO never had a ticker" remains visible; call
        ``.drop_nulls("symbol")`` for named intervals only.

        Example:
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
        # An empty-string ticker is the same absence as a null one; both must
        # reach the carry below rather than becoming the symbol "".
        frame = frame.with_columns(
            pl.when(
                pl.col("_base").is_null() | (pl.col("_base").str.len_chars() == 0)
            )
            .then(None)
            .otherwise(pl.col("_symbol"))
            .alias("_symbol")
        )
        # The carry (rule 4): forward within one PERMNO, never across two.
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
