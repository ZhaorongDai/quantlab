"""PERMNO -> ticker, as intervals: the CRSP symbology (D-04).

The raw CRSP tier is keyed by PERMNO, the stable security id, and the ticker
is derived HERE, at conversion time. That is the phase's identity decision:
a rename (FB -> META, PERMNO 13407 throughout) then never touches a watermark,
a resume point or a shard path -- it only changes which column a row lands in.

`dsf_v2` cannot spell a share class on its own: it carries no `shareclass` and
no `tradingsymbol` (live check `C3_columns`). Both come from
`stksecurityinfohist`, which is why symbology is an interval join rather than
a per-row read of the daily table.

**The rule, in order** (RESEARCH Q4, every case from a live row):

1. an override PERMNO takes its fixed symbol for every interval -- QQQ (86755)
   is `QQQ` for its whole history, including the 2004-2011 `QQQQ` spell;
2. `base = ticker.strip().upper()`;
3. `cls = shareclass`, unless it is null, empty, `"None"` or `"NONE"`;
4. if `cls` is set AND `tradingsymbol == base + cls`, the symbol is
   `base.cls` -- BRK + BRKB + B gives `BRK.B`, with the same `.` delimiter
   the constituent universes and `WrdsTaqNbboAcquisition.SUFFIX_DELIMITER`
   already use. Otherwise it is `base` (GOOGL, META, FB);
5. an interval whose ticker is null or empty CARRIES the previous interval's
   symbol for that PERMNO. Lehman's 2008-09-18 delisting interval is exactly
   this (live `L3_3`): without the carry, the delisting return -- the single
   most consequential row a delisted security has -- would silently lose its
   symbol and vanish from the panel.

**What this module deliberately does NOT do yet.** Plan 05 adds the collision
pass (two PERMNOs mapping to one symbol on overlapping dates), the carry past
a PERMNO's last interval, and row-level resolution. The two public signatures
below -- `symbol_intervals()` and `label_rows()` -- are the seam those land
behind, and are meant to stay stable across that change.
"""

from __future__ import annotations

import polars as pl

#: The delimiter between a base ticker and its share class. The same `.` the
#: constituent universes use and the same one
#: `quantlab/acquisition/wrds_taq.py:WrdsTaqNbboAcquisition.SUFFIX_DELIMITER`
#: declares -- restated rather than imported, because importing a TAQ constant
#: into the CRSP dataset layer would make a tick-acquisition module a
#: dependency of a daily-panel conversion.
SUFFIX_DELIMITER = "."

#: Share-class values that mean "no class". `"None"` and `"NONE"` are TEXT,
#: not nulls: the live tables carry both a real SQL NULL and, on some rows,
#: the four-character string.
_NO_CLASS = ("", "None", "NONE")


class CrspSymbology:
    """Turn `stksecurityinfohist` intervals into `(permno, symbol, window)`.

    `security_info` is the reference table as `CrspReference.table(...)`
    returns it. `overrides` maps a PERMNO string to a fixed symbol and is
    applied BEFORE every other rule.
    """

    SUFFIX_DELIMITER = SUFFIX_DELIMITER

    def __init__(
        self,
        security_info: pl.DataFrame,
        overrides: dict[str, str] | None = None,
    ) -> None:
        self.security_info = security_info
        self.overrides = {
            str(permno): str(symbol) for permno, symbol in (overrides or {}).items()
        }
        #: What `label_rows` could not label, for the caller to surface.
        #: Per PERMNO: `{"rows": n, "first": date, "last": date}`.
        self.report: dict[str, dict] = {"unlabelled": {}}
        self._intervals: pl.DataFrame | None = None

    # -- intervals ----------------------------------------------------------

    def symbol_intervals(self) -> pl.DataFrame:
        """`(permno, symbol, start_date, end_date)`, one row per interval.

        `symbol` is null only where a PERMNO's FIRST interval already has no
        ticker -- there is then no previous interval to carry, and inventing
        one would be a guess about identity.
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
        # The carry (rule 5), forward within one PERMNO and never across two.
        frame = frame.with_columns(
            pl.col("_symbol").forward_fill().over("permno").alias("_symbol")
        )

        if self.overrides:
            override = pl.col("permno").cast(pl.String).replace_strict(
                self.overrides, default=None, return_dtype=pl.String
            )
            frame = frame.with_columns(
                pl.coalesce(override, pl.col("_symbol")).alias("_symbol")
            )

        self._intervals = frame.select(
            pl.col("permno"),
            pl.col("_symbol").alias("symbol"),
            pl.col("start_date"),
            pl.col("end_date"),
        ).sort(["permno", "start_date"])
        return self._intervals

    # -- labelling ----------------------------------------------------------

    def label_rows(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Add `symbol` to daily rows by as-of joining each row's DATE onto
        its PERMNO's intervals.

        The join is per PERMNO on the interval START, then filtered by the
        interval END -- a plain as-of join alone would attach the last
        interval to every later row, which is the delisting-carry behaviour
        plan 05 makes deliberate rather than accidental.

        Rows left without a symbol are DROPPED and counted in
        `self.report["unlabelled"]`. Dropping is the honest answer here: a row
        with no symbol has no column to live in, and a placeholder label would
        put unrelated securities into one series.
        """
        intervals = self.symbol_intervals().drop_nulls("symbol")

        rows = frame.with_columns(
            pl.col("timestamp").dt.date().alias("_as_of")
        ).sort("_as_of")
        intervals = intervals.sort("start_date")

        labelled = rows.join_asof(
            intervals,
            left_on="_as_of",
            right_on="start_date",
            by="permno",
            strategy="backward",
        )
        labelled = labelled.with_columns(
            pl.when(
                pl.col("end_date").is_not_null()
                & (pl.col("_as_of") > pl.col("end_date"))
            )
            .then(None)
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        unlabelled = labelled.filter(pl.col("symbol").is_null())
        if unlabelled.height:
            summary = (
                unlabelled.group_by("permno")
                .agg(
                    pl.len().alias("rows"),
                    pl.col("_as_of").min().alias("first"),
                    pl.col("_as_of").max().alias("last"),
                )
                .sort("permno")
            )
            self.report["unlabelled"] = {
                str(record["permno"]): {
                    "rows": int(record["rows"]),
                    "first": str(record["first"]),
                    "last": str(record["last"]),
                }
                for record in summary.to_dicts()
            }
        else:
            self.report["unlabelled"] = {}

        return labelled.filter(pl.col("symbol").is_not_null()).drop(
            ["_as_of", "start_date", "end_date"]
        )
