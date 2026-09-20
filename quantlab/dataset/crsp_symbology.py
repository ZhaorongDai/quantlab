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

from datetime import date, datetime

import polars as pl
from loguru import logger

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

#: `dsf_v2.dlydelflg` on the delisting row itself (live `L3_1`). The ONE row
#: per dead security that carries the delisting return.
_DELISTING_FLAG = "Y"

#: How many colliding cells the refusal names before it stops listing. A
#: systematic breakage produces thousands; twenty is enough to recognise the
#: pattern, and the total is always stated.
_MAX_LISTED_COLLISIONS = 20


def _as_date(value) -> date:
    """A `Date`/`Datetime` cell as a plain `date`."""
    return value.date() if isinstance(value, datetime) else value


def _member_spans(member_intervals: pl.DataFrame | None):
    """`[(permno, start, end), ...]` from a universe's membership intervals.

    A null start or end means "open on that side", so it becomes the extreme
    rather than being skipped -- an open membership covers every date after
    its start, which is exactly the case a null end records.
    """
    if member_intervals is None or member_intervals.is_empty():
        return []
    spans = []
    for record in member_intervals.to_dicts():
        start = record.get("start_date")
        end = record.get("end_date")
        spans.append(
            (
                int(record["permno"]),
                date.min if start is None else _as_date(start),
                _OPEN_END if end is None else _as_date(end),
            )
        )
    return spans


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

        `frame` carries at least `permno` (Int64), `timestamp` (Datetime) and
        `dlydelflg` (String). The join is per PERMNO on the interval START,
        then filtered by the interval END -- a plain as-of join alone would
        attach the last interval to every later row, silently labelling years
        of rows a security never traded.

        **The delisting carry** (D-10, D-19, RESEARCH Pitfall 3). `DelDlyDt`
        is the trading day AFTER the delisting date, while the last
        `secinfoenddt` can be the delisting date itself (live `L3_1`/`L3_2`).
        A `dlydelflg='Y'` row dated past the PERMNO's last interval therefore
        falls outside every interval -- and it is the delisting RETURN, the
        single most consequential row a dead security has. It takes the last
        interval's symbol, and the carry is counted in
        `self.report["delisting_carried"]`. Without it, every delisting loss
        would vanish and survivorship bias would walk back in one row at a
        time.

        Any OTHER row with no covering interval is DROPPED and counted in
        `self.report["unlabelled"]`. Dropping is the honest answer there: a
        row with no symbol has no column to live in, and a placeholder label
        would put unrelated securities into one series.
        """
        intervals = self.symbol_intervals().drop_nulls("symbol")

        rows = frame
        borrowed_flag = "dlydelflg" not in rows.columns
        if borrowed_flag:
            # A frame without the flag simply has no delisting rows to carry.
            rows = rows.with_columns(
                pl.lit(None, dtype=pl.String).alias("dlydelflg")
            )
        rows = rows.with_columns(
            pl.col("timestamp").dt.date().alias("_as_of")
        ).sort(["permno", "_as_of"])
        ordered = intervals.sort(["permno", "start_date"])

        labelled = rows.join_asof(
            ordered,
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

        last_interval = ordered.group_by("permno").agg(
            pl.col("symbol").last().alias("_last_symbol"),
            pl.col("end_date").last().alias("_last_end"),
        )
        labelled = labelled.join(last_interval, on="permno", how="left")
        labelled = labelled.with_columns(
            (
                pl.col("symbol").is_null()
                & pl.col("_last_symbol").is_not_null()
                & pl.col("_last_end").is_not_null()
                & (pl.col("_as_of") > pl.col("_last_end"))
                & (
                    pl.col("dlydelflg").str.strip_chars().str.to_uppercase()
                    == _DELISTING_FLAG
                )
            ).alias("_carried")
        )
        labelled = labelled.with_columns(
            pl.when(pl.col("_carried"))
            .then(pl.col("_last_symbol"))
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )

        self.report["delisting_carried"] = self._per_permno(
            labelled.filter(pl.col("_carried")), symbol_column="symbol"
        )
        unlabelled = labelled.filter(pl.col("symbol").is_null())
        self.report["unlabelled"] = self._per_permno(unlabelled)
        if unlabelled.height:
            logger.warning(
                f"{type(self).__name__}: dropped {unlabelled.height} daily "
                f"row(s) across {unlabelled['permno'].n_unique()} PERMNO(s) "
                f"with no covering symbol interval and no delisting carry; "
                f"see report['unlabelled'] for the per-PERMNO windows."
            )

        dropped_helpers = ["_as_of", "start_date", "end_date",
                           "_last_symbol", "_last_end", "_carried"]
        if borrowed_flag:
            dropped_helpers.append("dlydelflg")
        return labelled.filter(pl.col("symbol").is_not_null()).drop(
            dropped_helpers
        )

    @staticmethod
    def _per_permno(
        frame: pl.DataFrame, symbol_column: str | None = None
    ) -> dict[str, dict]:
        """`{permno: {"rows": n, "first": iso, "last": iso}}` for a set of
        rows, with the symbol added when one is asked for. ISO text, because
        the report is written to a JSON sidecar."""
        if frame.is_empty():
            return {}
        aggregates = [
            pl.len().alias("rows"),
            pl.col("_as_of").min().alias("first"),
            pl.col("_as_of").max().alias("last"),
        ]
        if symbol_column is not None:
            aggregates.append(pl.col(symbol_column).first().alias("symbol"))
        summary = frame.group_by("permno").agg(aggregates).sort("permno")
        return {
            str(record["permno"]): {
                key: (int(value) if key == "rows" else str(value))
                for key, value in record.items()
                if key != "permno"
            }
            for record in summary.to_dicts()
        }

    # -- row-level collisions ------------------------------------------------

    def resolve_collisions(
        self,
        frame: pl.DataFrame,
        member_intervals: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        """Make `(timestamp, symbol)` identify exactly ONE security, by rule
        or not at all (D-04).

        `frame` is `label_rows`'s output, so it carries `symbol`.
        `member_intervals` is `(permno, start_date, end_date)`, closed on both
        ends -- the configured universe, when there is one.

        The rules, in order, per colliding `(date, symbol)` cell:

        1. `active_over_delisting` -- exactly one PERMNO is still trading
           (`dlydelflg != 'Y'`). Ticker reuse looks exactly like this: the old
           security's delisting row and the new one's first row share a day.
           The ticker belongs to whoever is still trading under it;
        2. `universe_member` -- exactly one of them is in the universe on that
           date. The panel is being built for that universe;
        3. otherwise **refuse**, naming the cells.

        There is deliberately no fourth rule. Averaging, summing or taking the
        first row would put two companies' prices in one series and leave no
        trace -- the panel would still look well-formed, and every return
        across the seam would be fabricated (T-03.10-16). A raise stops the
        conversion where the user can still fix it.

        A frame with no collision is returned UNCHANGED, same rows in the same
        order: the caller's sort carries the adjustment anchor.
        """
        self.report["collisions"] = []
        if frame.is_empty():
            return frame

        work = frame.with_row_index("_row")
        crowded = (
            work.group_by(["timestamp", "symbol"])
            .agg(pl.col("permno").n_unique().alias("_permnos"))
            .filter(pl.col("_permnos") > 1)
            .drop("_permnos")
        )
        if crowded.is_empty():
            return frame

        cells = work.join(crowded, on=["timestamp", "symbol"], how="inner")
        cells = cells.sort(["timestamp", "symbol", "permno", "_row"])
        spans = _member_spans(member_intervals)

        resolutions: list[dict] = []
        unresolved: list[tuple[date, str, list[int]]] = []
        discard: set[int] = set()

        for (stamp, symbol), group in cells.group_by(
            ["timestamp", "symbol"], maintain_order=True
        ):
            records = group.to_dicts()
            day = _as_date(stamp)
            permnos = sorted({int(record["permno"]) for record in records})
            active = sorted(
                {
                    int(record["permno"])
                    for record in records
                    if str(record.get("dlydelflg") or "").strip().upper()
                    != _DELISTING_FLAG
                }
            )
            kept, rule = None, None
            if len(active) == 1:
                kept, rule = active[0], "active_over_delisting"
            elif spans:
                members = sorted(
                    {
                        permno
                        for permno, start, end in spans
                        if permno in permnos and start <= day <= end
                    }
                )
                if len(members) == 1:
                    kept, rule = members[0], "universe_member"
            if kept is None:
                unresolved.append((day, str(symbol), permnos))
                continue
            resolutions.append(
                {
                    "date": day.isoformat(),
                    "symbol": str(symbol),
                    "kept": kept,
                    "dropped": [p for p in permnos if p != kept],
                    "rule": rule,
                }
            )
            discard.update(
                int(record["_row"])
                for record in records
                if int(record["permno"]) != kept
            )

        if unresolved:
            raise ValueError(self._collision_message(unresolved))

        self.report["collisions"] = resolutions
        if not discard:
            return frame
        logger.warning(
            f"{type(self).__name__}: resolved {len(resolutions)} "
            f"(date, symbol) collision(s) by rule; see report['collisions'] "
            f"for which PERMNO kept each cell."
        )
        return work.filter(~pl.col("_row").is_in(list(discard))).drop("_row")

    def _collision_message(
        self, unresolved: list[tuple[date, str, list[int]]]
    ) -> str:
        """The refusal. Names the cells, says nothing was merged, and gives
        the two things the user can actually do about it."""
        shown = unresolved[:_MAX_LISTED_COLLISIONS]
        listing = "\n".join(
            f"  ({day.isoformat()}, {symbol!r}, {permnos})"
            for day, symbol, permnos in shown
        )
        more = (
            ""
            if len(unresolved) <= _MAX_LISTED_COLLISIONS
            else f" (first {_MAX_LISTED_COLLISIONS} shown)"
        )
        return (
            f"{type(self).__name__}: {len(unresolved)} (date, symbol) cell(s) "
            f"hold more than one PERMNO and no rule resolves them{more}:\n"
            f"{listing}\n"
            f"NOTHING was merged -- two securities in one column would make "
            f"every return across the seam fabricated. Either restrict "
            f"`permnos` to one of the colliding securities, or supply a "
            f"`collision_universe` (the `member_intervals` argument) so the "
            f"universe member on that date wins."
        )
