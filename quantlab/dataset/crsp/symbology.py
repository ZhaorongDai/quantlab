"""PERMNO -> ticker, as intervals: the CRSP symbology (D-04, D-18).

**One job: say which ticker a PERMNO wore over which dates.** Nothing here
decides what a panel column is called any more. Since 03.11-03 the price
panel's `symbol` axis IS the int64 PERMNO (D-01), so a ticker is no longer an
identity: it cannot collide with another security's, cannot need a share-class
suffix to be told apart, and cannot decide whether a delisting row has a column
to live in. The four mechanisms that existed for those cases -- the interval
collision pass, the row-level `(date, symbol)` tie-break, the class respelling
and the delisting SYMBOL carry -- were DELETED in 03.11-07 rather than kept as
guards, because on a PERMNO axis the failure they guarded against is not merely
unlikely, it is unspellable: two securities never reach one cell.

What remains is a lookup table for HUMANS. `symbol_intervals()` is the single
data source for the ticker sidecar (plan 09), which puts readable names beside
a panel keyed on numbers.

`dsf_v2` cannot spell a share class on its own: it carries no `shareclass` and
no `tradingsymbol` (live check `C3_columns`). Both come from
`stksecurityinfohist`, which is why this is an interval table built from the
reference tier rather than a per-row read of the daily one.

**The rule, in order** (RESEARCH Q4, every case from a live row):

1. `base = ticker.strip().upper()`;
2. `cls = shareclass`, unless it is null, empty, `"None"` or `"NONE"`;
3. if `cls` is set AND `tradingsymbol == base + cls`, the symbol is
   `base.cls` -- BRK + BRKB + B gives `BRK.B`, with the same `.` delimiter the
   constituent universes and `WrdsTaqNbboAcquisition.SUFFIX_DELIMITER` already
   use. Otherwise it is `base` (GOOGL, META, FB);
4. an interval whose ticker is null or empty CARRIES the previous interval's
   symbol for that PERMNO. Lehman's 2008-09-18 delisting interval is exactly
   this (live `L3_3`). On a PERMNO axis this carry no longer decides whether
   that row is in the panel -- the row is keyed on 80599 either way -- it
   decides only whether the sidecar can put a NAME on a dead security's last
   day.

**What this module deliberately does NOT do.** It never merges two PERMNOs and
it never decides which rows enter a panel. It answers one question, about
names; identity is the PERMNO and is settled before anything here runs.
"""

from __future__ import annotations

import polars as pl

#: The delimiter between a base ticker and its share class. The same `.` the
#: constituent universes use and the same one
#: `quantlab/acquisition/wrds/taq.py:WrdsTaqNbboAcquisition.SUFFIX_DELIMITER`
#: declares -- restated rather than imported, because importing a TAQ constant
#: into the CRSP dataset layer would make a tick-acquisition module a
#: dependency of a daily-panel conversion.
SUFFIX_DELIMITER = "."

#: Share-class values that mean "no class". `"None"` and `"NONE"` are TEXT,
#: not nulls: the live tables carry both a real SQL NULL and, on some rows,
#: the four-character string.
_NO_CLASS = ("", "None", "NONE")


class CrspSymbology:
    """`stksecurityinfohist` intervals as a PERMNO -> ticker interval table.

    `security_info` is the reference table as `CrspReference.table(...)`
    returns it. There is exactly one public method, `symbol_intervals()`. This
    is a class rather than a function because the table is computed once and
    read more than once per conversion, and the cache has to live somewhere.
    """

    SUFFIX_DELIMITER = SUFFIX_DELIMITER

    def __init__(self, security_info: pl.DataFrame) -> None:
        self.security_info = security_info
        self._intervals: pl.DataFrame | None = None

    # -- intervals ----------------------------------------------------------

    def symbol_intervals(self) -> pl.DataFrame:
        """`(permno, symbol, start_date, end_date)`, one row per interval.

        Sorted by `(permno, start_date)`; `permno` Int64, `symbol` String,
        both dates Date. Computed once and cached.

        **This schema is the ticker sidecar's schema** (plan 09), and that
        sidecar is now this method's only consumer -- the panel axis is the
        PERMNO and needs no name to be built. Stated here rather than only at
        the sidecar because the four columns above are what a reader of that
        JSON gets, and they are decided in this select.

        `symbol` is null only where a PERMNO's FIRST interval already has no
        ticker -- there is then no previous interval to carry, and inventing
        one would be a guess about identity. A caller that wants only named
        intervals asks for `.drop_nulls("symbol")`; the nulls are kept here so
        that "this PERMNO never had a ticker" stays a distinguishable fact
        rather than an absent row.
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
        # An empty-string ticker is the same absence a NULL one is; both must
        # reach the carry below rather than becoming the symbol `""`.
        frame = frame.with_columns(
            pl.when(
                pl.col("_base").is_null() | (pl.col("_base").str.len_chars() == 0)
            )
            .then(None)
            .otherwise(pl.col("_symbol"))
            .alias("_symbol")
        )
        # The carry (rule 4), forward within one PERMNO and never across two.
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
