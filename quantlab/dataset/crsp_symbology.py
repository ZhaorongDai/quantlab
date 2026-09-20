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
   symbol and vanish from the panel;
6. the COLLISION PASS, last. Two DIFFERENT PERMNOs whose symbols came out
   equal over overlapping dates are two securities heading for one column.
   Every colliding interval whose row carries a share class is respelled
   `base.cls`; the ones with no class keep the bare symbol. That is the only
   rule that can separate BRK.A from BRK.B before 2002-01-02, when NEITHER
   row has a `tradingsymbol` (live `C5`), and the WIN pair of `L5_1`, where
   one issue has a class and the other has none.

**The same function feeds the price panel AND the membership panels.** Plan 08
labels `dsf_v2` rows with it and plan 09 labels `dsp500list_v2` / Nasdaq-100
spells with it. If they derived symbols separately, a universe mask could name
`BRK.B` on a day the price panel called that column `BRK`, and the mask would
silently select nothing.

**What this module deliberately does NOT do.** It never MERGES two PERMNOs. The
interval pass respells what it can; whatever still lands on one `(date, symbol)`
cell goes to `resolve_collisions`, which resolves by stated rule or refuses.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from quantlab.enums.data import TRADEABLE_TICKER_PATTERN

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

#: Stands in for an open interval's missing `secinfoenddt` while the collision
#: pass tests overlaps. An open interval overlaps everything that starts after
#: it, which is what a null end MEANS; comparing against a null would instead
#: make every such pair silently non-overlapping.
_OPEN_END = date(9999, 12, 31)


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
        #: Everything this module CHANGED or COULD NOT DO, for the caller to
        #: write beside the panel (the plan-08 `{zarr}.crsp_symbols.json`
        #: sidecar). All five values are JSON-serializable, dates as ISO text:
        #:
        #: - `class_suffixed` -- intervals the collision pass respelled;
        #: - `nonconforming_symbols` -- symbols outside
        #:   `TRADEABLE_TICKER_PATTERN`, kept but listed;
        #: - `unlabelled` -- per PERMNO, rows dropped for want of a symbol;
        #: - `delisting_carried` -- per PERMNO, delisting rows labelled from
        #:   the PERMNO's last interval;
        #: - `collisions` -- every `(date, symbol)` cell resolved, and by
        #:   which rule.
        self.report: dict[str, object] = {
            "class_suffixed": [],
            "nonconforming_symbols": [],
            "unlabelled": {},
            "delisting_carried": {},
            "collisions": [],
        }
        self._intervals: pl.DataFrame | None = None

    # -- intervals ----------------------------------------------------------

    def symbol_intervals(self) -> pl.DataFrame:
        """`(permno, symbol, start_date, end_date)`, one row per interval.

        Sorted by `(permno, start_date)`; `permno` Int64, `symbol` String,
        both dates Date. Computed once and cached, because the price panel
        and every membership panel ask the same instance for it.

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

        frame = self._class_collision_pass(frame)

        self._intervals = frame.select(
            pl.col("permno"),
            pl.col("_symbol").alias("symbol"),
            pl.col("start_date"),
            pl.col("end_date"),
        ).sort(["permno", "start_date"])

        self.report["nonconforming_symbols"] = [
            symbol
            for symbol in sorted(
                set(self._intervals["symbol"].drop_nulls().to_list())
            )
            if not TRADEABLE_TICKER_PATTERN.match(symbol)
        ]
        return self._intervals

    def _class_collision_pass(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Respell the intervals of DIFFERENT PERMNOs that came out with one
        symbol over overlapping dates (rule 6).

        Deterministic and single-pass: the overlapping set is computed once,
        off the symbols rules 1-5 produced, and each colliding interval is
        respelled from its OWN share class. It is never iterated to a fixed
        point -- a second round could only rename an interval a first round
        already separated, and "the symbol depends on how many times we
        looked" is not a property a panel axis may have.

        Three deliberate exemptions:

        - an interval with NO class keeps the bare symbol. CRSP did not give
          it a class, and inventing one would rename a security (`WIN` and
          `BF`, live `L5_1`/`L6_1`, are exactly this);
        - an OVERRIDDEN PERMNO is untouched: `symbol_overrides` is a fixed
          symbol for a whole history (D-15), which a suffix would contradict;
        - a symbol that ALREADY ends in its own class is left alone, so a
          pair that rules 1-4 spelled `ABC.B` cannot become `ABC.B.B`. Such a
          pair is still a collision; it is `resolve_collisions`'s to refuse,
          because the spelling cannot separate them.
        """
        named = frame.filter(pl.col("_symbol").is_not_null())
        if self.overrides:
            named = named.filter(
                ~pl.col("permno").cast(pl.String).is_in(list(self.overrides))
            )
        if named.height < 2:
            return frame

        windows = named.select(
            pl.col("permno"),
            pl.col("_symbol"),
            pl.col("start_date"),
            pl.col("end_date").fill_null(_OPEN_END).alias("_end"),
        )
        overlaps = windows.join(windows, on="_symbol", how="inner", suffix="_r")
        overlaps = overlaps.filter(
            (pl.col("permno") != pl.col("permno_r"))
            & (pl.col("start_date") <= pl.col("_end_r"))
            & (pl.col("start_date_r") <= pl.col("_end"))
        )
        if overlaps.is_empty():
            return frame

        colliding = overlaps.select(
            "permno", "_symbol", "start_date"
        ).unique().with_columns(pl.lit(True).alias("_collides"))

        frame = frame.join(
            colliding, on=["permno", "_symbol", "start_date"], how="left"
        ).with_columns(pl.col("_collides").fill_null(False))

        suffix = pl.lit(SUFFIX_DELIMITER) + pl.col("_cls")
        frame = frame.with_columns(
            pl.when(
                pl.col("_collides")
                & pl.col("_cls").is_not_null()
                & ~pl.col("_symbol").str.ends_with(suffix)
            )
            .then(pl.col("_symbol") + suffix)
            .otherwise(pl.col("_symbol"))
            .alias("_respelled")
        )

        moved = frame.filter(pl.col("_respelled") != pl.col("_symbol"))
        self.report["class_suffixed"] = [
            {
                "permno": int(record["permno"]),
                "symbol": str(record["_respelled"]),
                "start_date": str(record["start_date"]),
                "end_date": None
                if record["end_date"] is None
                else str(record["end_date"]),
            }
            for record in moved.sort(["permno", "start_date"]).to_dicts()
        ]
        return frame.with_columns(
            pl.col("_respelled").alias("_symbol")
        ).drop(["_collides", "_respelled"])

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
