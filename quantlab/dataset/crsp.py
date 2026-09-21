"""The CRSP Stock v2 daily panel: PERMNO shards -> a drop-in `[timestamp,
symbol]` store (phase 03.10).

**Raw stays CRSP-as-is; everything derived happens here.** The acquisition
tier writes `dsf_v2` rows verbatim, keyed by PERMNO. This module is where the
ticker is resolved, the total-return series is built, and the twelve Tiingo
EOD variables appear -- so an existing factor, label or backtester reads a
CRSP panel without knowing it is one (D-07).

**The adjustment is GLOBAL, computed ONCE per instance over
`[config.start_date, config.end_date]`** (RESEARCH Pattern 4, Pitfall 1).
`from_raw_data_chunked` calls `_raw_axes_in_range()` once and then
`_raw_data_to_xr_window()` per window; computing the anchor inside a window
would give every chunk its OWN anchor and put a fabricated return at every
chunk seam. The cached derivation is the reason `granularity="year"` and
`granularity="month"` produce the same store.

**The anchor MOVES when the window is extended, so a store records its own**
(D-08, RESEARCH Pitfall 2). Pushing `end_date` forward gives every
still-listed PERMNO a later last priced day, which multiplies all of its
earlier `adj*` values by a constant -- ratios survive, levels do not. Tiingo
restates adjusted history the same way. What differs here is the WRITER: the
chunk ledger appends windows and never rewrites a finished one, so extending
a store in place would leave its old windows on the old anchor and the new
ones on the new anchor, with a fabricated return at the join that no reader
could see. Every conversion therefore writes `{zarr}.crsp_adjustment.json`
beside the store before the first append and compares it on every later run;
a mismatch -- or a missing sidecar -- REFUSES before anything is written. The
remedy is a rebuild, which is cheap: CRSP publishes once a year.

**The arithmetic**, per PERMNO, sorted by date (D-08):

- `close = abs(dlyprc)`, with CRSP's NO-PRICE sentinel excluded BEFORE the
  `abs()`: a delisting-AMOUNT row (`dlyprcflg` in `_NO_PRICE_FLAGS`, or a bare
  `dlyprc == 0.0`) carries a settlement amount rather than a market price, and
  `abs(0.0)` is still 0.0 -- so an unguarded `abs()` publishes a $0.00 trade on
  a day that had none. Not `dlyclose` either: that is null on bid/ask days,
  through the whole pre-1992 Nasdaq era and on delisting rows, while `dlyret`
  is computed from `dlyprc` -- so using `dlyclose` would put a return chain and
  a price series that disagree into one panel.
- `G_t = prod_{s<=t}(1 + dlyret_s)`, a NULL return contributing 1. CIZ returns
  span gaps back to `DlyPrevDt` (`DlyRetDurFlg`), so the next valid return
  already covers the missing day; filling a null with 0 would double-count it.
- the anchor `A` is the PERMNO's LAST row carrying a USABLE LEVEL: a strictly
  positive `close` AND a non-null `dlycumfacshr`, so every quantity read off the
  anchor comes from ONE row that carries all of them. A non-null test alone was
  the loophole -- the sentinel above is the NUMBER 0.0, which is not null, so it
  became the anchor and zeroed the security's whole adjusted history, while the
  same row's NULL `dlycumfacshr` made its whole `adjVolume` NaN. A PERMNO with
  no qualifying row, or one whose `G_A` is 0.0 or non-finite (a `dlyret` of
  -1.0), makes the conversion REFUSE by PERMNO rather than publish a zeroed,
  all-NaN or infinite column -- nothing downstream would raise on any of them;
- `adjClose_t = close_A * G_t / G_A`, `factor_t = adjClose_t / close_t`,
  `adjOpen/High/Low = raw * factor_t`;
- `adjVolume_t = volume_t * dlycumfacshr_t / dlycumfacshr_A`;
- `splitFactor_t = dlycumfacpr_{t-1} / dlycumfacpr_t`, and 1.0 on the PERMNO's
  first row;
- `divCash_t = dlyorddivamt_t + dlynonorddivamt_t`, on the ex-date.

**`abs()` is a GUARD, not a transform** (D-09). Legacy CRSP encoded "no trade,
this is a bid/ask midpoint" as a NEGATIVE price; CIZ does not -- the live check
counted ZERO negative `dlyprc` against 122,471 `BA` rows in 2000 (`L10_1`), so
`abs()` is a no-op on every row this pipeline will meet. It stays because a
legacy-shaped row entering the panel as a negative price would invert every
ratio downstream in silence. The no-trade SIGNAL is therefore
`dlyprcflg == 'BA'`, surfaced as `prc_is_bidask` (D-19) -- a sign test would
flag nothing at all.

**The sentinel guard, by contrast, is NOT a no-op** (`_NO_PRICE_FLAGS`). The
delisting-AMOUNT shape is what modern CIZ writes -- 5 of 5 delisting rows in the
tier this phase pulled -- and it is the one row per delisted security that the
old `abs(dlyprc)` turned into a fabricated $0.00 close and an anchor of zero.

**A missing return is NaN, never 0** (D-09). `ret` keeps the null; only the
internal cumulative product treats it as a factor of 1, because CIZ returns
span gaps back to `DlyPrevDt` (`DlyRetDurFlg` D3/D4) and the next valid return
already covers the missing day.

The delisting return needs no special case: CIZ already puts it on its own
daily row, so chaining `dlyret` carries it. Adding `stkdelists.delret` on top
would apply the loss twice (D-10) -- Lehman's 2008-09-18 row already holds
`dlyret = -0.6`, and `stkdelists` stays EVENT data that nothing compounds.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import xarray as xr
from loguru import logger

from quantlab.base.config import CrspDatasetConfig, DatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset.crsp_reference import CrspReference
from quantlab.dataset.crsp_symbology import CrspSymbology
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.symbol_axis import sort_symbol_axis

#: Everything this panel carries BEYOND the Tiingo twelve, with its CRSP source
#: and the unit the panel states it in. FLOAT64 like every other variable
#: (RESEARCH Pitfall 7): a dense `[timestamp, symbol]` panel is a cartesian
#: product, so a symbol that did not exist yet needs a NaN to say so -- which
#: an integer `permno` or a boolean flag has no room for.
#:
#: `permno` is NOT here, and its absence is the point (D-01): the panel's
#: `symbol` COORDINATE is the int64 PERMNO, so a `permno` data variable would
#: be the same number twice -- once as the axis and once as a float64 copy of
#: it that nothing keeps in step.
#:
#: - ``permco``             -- CRSP's COMPANY id. It survives because it is a
#:   different identity from the axis: one company, several securities.
#: - ``ret``              -- ``dlyret``, the daily TOTAL return (dividends
#:   included, and the delisting return on its own row). NaN where CRSP has
#:   none; never 0 (D-09).
#: - ``retx``               -- ``dlyretx``, the same return WITHOUT dividends.
#: - ``shrout``             -- ``shrout`` x 1000. CRSP stores thousands of
#:   shares; the panel states shares.
#: - ``market_cap``         -- ``dlycap`` x 1000. CRSP stores thousands of
#:   USD; the panel states USD.
#: - ``bid``/``ask``        -- ``dlybid``/``dlyask``, the quote the midpoint
#:   came from on a no-trade day.
#: - ``prc_is_bidask``      -- 1.0 when ``dlyprcflg == "BA"`` (the price is a
#:   bid/ask midpoint, i.e. NO TRADE), 0.0 for any other flag, NaN when the
#:   flag itself is null. The D-19 no-trade indicator.
#: - ``is_delisting``       -- 1.0 when ``dlydelflg == "Y"`` (this row carries
#:   the delisting return), 0.0 otherwise, NaN when the flag is null.
#: - ``numtrd``             -- ``dlynumtrd``, the trade count.
#: - ``cumfacpr``/``cumfacshr`` -- ``dlycumfacpr``/``dlycumfacshr``, CRSP's own
#:   cumulative price and share factors (1.0 on the last trading day).
#: - ``facprc``             -- ``dlyfacprc``, the day's price factor: 1.0 on an
#:   ordinary day, 4.0 on AAPL's 2020-08-31 4:1 split (unlike legacy ``facpr``,
#:   which is 0 on ordinary days).
#: - ``close_trade``        -- ``dlyclose``, the CLOSING-TRADE price. Null on
#:   bid/ask days, through the whole pre-1992 Nasdaq era and on delisting rows,
#:   which is exactly why `close` is ``abs(dlyprc)`` and this is a separate
#:   variable rather than the panel's close.
CRSP_EXTRA_VARIABLES: tuple[str, ...] = (
    "permco",
    "ret",
    "retx",
    "shrout",
    "market_cap",
    "bid",
    "ask",
    "prc_is_bidask",
    "is_delisting",
    "numtrd",
    "cumfacpr",
    "cumfacshr",
    "facprc",
    "close_trade",
)

#: The `dsf_v2` columns a security filter may name. Every one of them is a
#: per-day TYPE column: it says what the security WAS on that date, which is
#: what makes a per-date verdict possible at all.
#:
#: The list is CLOSED on purpose (T-03.10-28). `ticker`, `permno` or a price
#: column would also filter, but a ticker-picked panel that looks like a
#: type-filtered one is exactly the silent substitution the report cannot
#: catch -- the roster filter is `config.permnos`, the ticker filter is
#: `config.symbols`, and this is the TYPE filter.
#:
#: The last four (`primaryexch`, `conditionaltype`, `tradingstatusflg`,
#: `exchangetier`) are filterable but appear in NO preset: they change over a
#: security's life, so filtering on them punches holes in a series and can
#: drop the delisting row (RESEARCH Q3). A user who wants an NYSE-only panel
#: may still ask for one explicitly.
FILTERABLE_COLUMNS: tuple[str, ...] = (
    "sharetype",
    "securitytype",
    "securitysubtype",
    "usincflg",
    "issuertype",
    "primaryexch",
    "conditionaltype",
    "tradingstatusflg",
    "exchangetier",
)

#: The named security filters. A value is `{column: allowed values}`; every
#: listed column must match, and a NULL value never matches.
#:
#: **Why `equity_common` also carries a `sharetype` allow-list** (the flagged
#: D-17 reading). D-17 states the predicate `securitytype='EQTY' AND
#: securitysubtype='COM'` AND states that the filter drops ADRs and units. The
#: live S&P rows show those two goals disagree: an ADR reads
#: `AD/EQTY/COM/CORP/N` and a unit reads `UG/EQTY/COM/CORP/N`
#: (`03.10-LIVE-CHECK-2.json` key `L11_1`), so both SATISFY the two-column
#: predicate. Adding `sharetype in (NS, SB, CE)` -- every ShareType code the
#: CRSP flag dictionary defines (`L2_2`) except `AD` and `UG` -- delivers
#: every drop and every keep D-17 lists: REITs stay (including the eight with
#: `SB`), non-US-incorporated common stays, ADRs/units/funds/ETFs/unknown
#: types go.
#:
#: The reading is REVERSIBLE and costs nothing to undo: the literal two-column
#: predicate is `{"securitytype": ["EQTY"], "securitysubtype": ["COM"]}` as a
#: `security_filter` dict, and the filter report makes the difference visible
#: either way.
#:
#: `shrcd_10_11` is the legacy `shrcd in (10, 11)` replication, which is NOT
#: the default precisely because it drops REITs and non-US issuers that are
#: legitimate S&P 500 and Nasdaq-100 members (RESEARCH Pitfall 5).
SECURITY_FILTER_PRESETS: dict[str, dict[str, tuple[str, ...]]] = {
    "equity_common": {
        "securitytype": ("EQTY",),
        "securitysubtype": ("COM",),
        "sharetype": ("NS", "SB", "CE"),
    },
    "shrcd_10_11": {
        "sharetype": ("NS",),
        "securitytype": ("EQTY",),
        "securitysubtype": ("COM",),
        "usincflg": ("Y",),
        "issuertype": ("ACOR", "CORP"),
    },
    "none": {},
}

#: `{zarr_file_path}.crsp_filter_report.json` -- what the security filter
#: removed, by type combination and by PERMNO (D-17).
FILTER_REPORT_SUFFIX: str = ".crsp_filter_report.json"

#: The `dsf_v2` flag marking the row that carries the delisting return.
_DELISTING_FLAG = "Y"

#: The `dlyprcflg` values that mean "this row carries NO market price".
#:
#: CIZ writes two delisting shapes, and only one of them is a price:
#:
#: - `DP` (delisting PRICE) is a REAL price -- Lehman 2008-09-18 is
#:   `dlyprc = 0.052`, an actual value a holding was worth. It stays a price.
#: - `DA` (delisting AMOUNT) carries a settlement AMOUNT, not a market price.
#:   CRSP writes `dlyprc = 0.000000` there as a SENTINEL and leaves `dlyclose`,
#:   `dlyvol`, `dlycumfacpr` and `dlycumfacshr` all NULL.
#:
#: `DA` is the shape modern CIZ actually writes: **5 of 5** delisting rows in
#: the raw tier this phase pulled (`TR` 138,888 / `DA` 5 / `DP` 0). Reading its
#: 0.0 as a close publishes a fabricated $0.00 trade AND -- because 0.0 is not
#: NULL -- lets the sentinel row become the adjustment anchor, which zeroes the
#: security's whole adjusted history (03.10-REVIEW.md CR-01/CR-02).
_NO_PRICE_FLAGS: tuple[str, ...] = ("DA",)

#: The order the report spells a type combination in.
_TYPE_COLUMNS: tuple[str, ...] = (
    "sharetype",
    "securitytype",
    "securitysubtype",
    "issuertype",
    "usincflg",
)


def resolve_security_filter(
    value: str | dict, *, owner: str = "CrspStockDataset"
) -> dict[str, tuple[str, ...]]:
    """`config.security_filter` -> `{column: allowed values}`, or `ValueError`.

    A preset NAME resolves to its entry in `SECURITY_FILTER_PRESETS`; a
    mapping is validated against `FILTERABLE_COLUMNS` and returned with its
    value lists frozen into tuples.

    Every refusal names BOTH the offending value and the legal ones. An
    unknown preset must never fall back to the default or to "keep
    everything": both would be a silently different panel, and a panel that is
    silently different is indistinguishable from one that is right.
    """
    if isinstance(value, str):
        try:
            preset = SECURITY_FILTER_PRESETS[value]
        except KeyError:
            raise ValueError(
                f"{owner}: security_filter {value!r} is not a preset. The "
                f"presets are {sorted(SECURITY_FILTER_PRESETS)}. Pass one of "
                f"those names, or an explicit "
                f"{{column: allowed values}} mapping over "
                f"{list(FILTERABLE_COLUMNS)}."
            ) from None
        return {column: tuple(allowed) for column, allowed in preset.items()}

    if not isinstance(value, dict):
        raise ValueError(
            f"{owner}: security_filter must be a preset name "
            f"({sorted(SECURITY_FILTER_PRESETS)}) or a "
            f"{{column: allowed values}} mapping over "
            f"{list(FILTERABLE_COLUMNS)}; got {type(value).__name__}."
        )

    resolved: dict[str, tuple[str, ...]] = {}
    for column, allowed in value.items():
        if column not in FILTERABLE_COLUMNS:
            raise ValueError(
                f"{owner}: security_filter names the column {column!r}, which "
                f"is not filterable. The filterable columns are "
                f"{list(FILTERABLE_COLUMNS)} -- all per-day TYPE columns, so "
                f"the verdict can be a per-date one. To restrict the ROSTER "
                f"use config.permnos; to restrict the TICKERS use "
                f"config.symbols."
            )
        if isinstance(allowed, (str, bytes)) or not isinstance(
            allowed, (list, tuple, set, frozenset)
        ):
            raise ValueError(
                f"{owner}: security_filter[{column!r}] must be a list of "
                f"allowed values, got {allowed!r}. A bare string would be "
                f"iterated CHARACTER by character and match nothing."
            )
        values = tuple(str(item) for item in allowed)
        if not values:
            raise ValueError(
                f"{owner}: security_filter[{column!r}] is an empty allow-list, "
                f"which matches no row and would silently empty the panel. "
                f"Drop the key to stop filtering on {column!r}, or use the "
                f"'none' preset to keep every security."
            )
        resolved[column] = values
    return resolved


class CrspStockDataset(StockDataset):
    """A dense daily panel built from the CRSP PERMNO-keyed raw tier.

    Reuses `StockDataset`'s vendor-root assertion, `month=` hive schema,
    single-vendor provenance check, window predicates and `has_raw_data`.
    Overrides the axes and the window densifier, because both must run against
    the SYMBOLOGY-LABELLED, globally-adjusted derivation rather than against
    the raw frame -- the raw tier has no `symbol` axis a panel can use (its
    `symbol` column is the PERMNO).
    """

    #: The config class `quantlab/utils/module.py` rebuilds this dataset with.
    config_cls = CrspDatasetConfig

    #: The twelve variables a Tiingo daily panel carries, in `TiingoColumns.EOD`
    #: order. Derived from that constant rather than restated, so the drop-in
    #: promise is checked against the thing it promises compatibility with.
    TIINGO_VARIABLES: tuple[str, ...] = tuple(TiingoColumns.EOD.split(","))

    #: Extra variables this panel adds beyond the Tiingo twelve -- the
    #: module-level `CRSP_EXTRA_VARIABLES`, bound here so a subclass can
    #: narrow or extend the set without the module constant moving.
    EXTRA_VARIABLES: tuple[str, ...] = CRSP_EXTRA_VARIABLES

    #: The sidecar recording WHICH anchor a store's `adj*` columns were built
    #: against. A SIBLING of the store, exactly like `ChunkLedger`'s
    #: `.chunks.json` -- not a file inside the zarr directory, which a
    #: `mode="w"` rewrite would drop and a zarr reader would surface as a
    #: stray array.
    ADJUSTMENT_SIDECAR_SUFFIX: str = ".crsp_adjustment.json"

    #: The rule the sidecar names. Written into the file rather than only
    #: implied by it, so a future change of rule is visible to a reader of an
    #: OLD store instead of only in this module's history.
    ADJUSTMENT_RULE: str = "total_return_backward_from_last_close"

    @BaseDataset.config.setter
    def config(self, config: DatasetConfig):
        BaseDataset.config.fset(self, config)
        if not isinstance(config, CrspDatasetConfig):
            raise TypeError(
                f"{self.class_name} needs a CrspDatasetConfig, got "
                f"{type(config).__name__}."
            )
        if config.frequency != "1d":
            raise ValueError(
                f"{self.class_name}: frequency must be '1d' (CRSP Stock v2 "
                f"daily); got {config.frequency!r}."
            )
        if config.vendor != "wrds":
            raise ValueError(
                f"{self.class_name}: vendor must be 'wrds' (CRSP is reached "
                f"through the WRDS account); got {config.vendor!r}."
            )
        if config.permnos is not None:
            permnos = tuple(str(permno) for permno in config.permnos)
            bad = [permno for permno in permnos if not permno.isdigit()]
            if bad:
                raise ValueError(
                    f"{self.class_name}: config.permnos must hold PERMNO digit "
                    f"strings; {bad} are not. The TICKER filter is "
                    f"config.symbols -- permnos selects the raw tier, before "
                    f"symbology has run."
                )
            # WR-01. An empty tuple carried TWO possible meanings and no way to
            # tell them apart, so it is refused by NAME rather than given one of
            # them: it selects no security, and the roster gate that reads it
            # would otherwise have read it as "every PERMNO in the raw tier".
            # The refusal names both, because the defect was that they were
            # indistinguishable -- a caller who saw only "empty roster refused"
            # would still not know which one they had written.
            if not permnos:
                raise ValueError(
                    f"{self.class_name}: config.permnos is an EMPTY tuple, "
                    f"which selects no security -- and which would otherwise "
                    f"be read as 'every PERMNO in the raw tier', silently "
                    f"widening the panel to every security on disk under a "
                    f"store name chosen for none of them. The two meanings are "
                    f"not distinguishable from '()', so neither is assumed: "
                    f"pass None to mean 'every PERMNO in the raw tier', or a "
                    f"non-empty roster to name the securities you want."
                )
            config.permnos = permnos

        # The filter is validated HERE, at assignment, rather than where it is
        # first applied: a malformed filter is a config error, and a config
        # error that only surfaces after a raw tier has been scanned is one the
        # user pays for twice. `_security_filter` holds the RESOLVED mapping;
        # `config.security_filter` keeps what the user wrote (a preset NAME
        # stays a name), so a round-tripped config reads back as it was
        # written. A dict value is normalised in place to tuples, which is
        # what makes the JSON round trip exact (D-12).
        self._security_filter = resolve_security_filter(
            config.security_filter, owner=self.class_name
        )
        if isinstance(config.security_filter, dict):
            config.security_filter = dict(self._security_filter)
        if config.collision_universe is not None:
            from quantlab.dataset.crsp_membership import CrspMembership

            if config.collision_universe not in CrspMembership.INDEXES:
                raise ValueError(
                    f"{self.class_name}: collision_universe "
                    f"{config.collision_universe!r} is not a CRSP universe; "
                    f"this vendor serves {CrspMembership.INDEXES}."
                )

        # Invalidated on every config assignment: the derivation is scoped to
        # `[start_date, end_date]`, so a re-dated config must not reuse the
        # previous window's anchor.
        self._derivation_cache: pl.DataFrame | None = None
        self._symbology: CrspSymbology | None = None
        self._filter_report: dict | None = None
        # The membership spells of `collision_universe`, memoised for the
        # roster exemption in `_apply_security_filter` -- its one reader. Reset
        # here for the same reason the derivation is: a re-dated or re-rostered
        # config must not reuse the previous universe's spells.
        self._member_intervals_cache: pl.DataFrame | None = None

    # -- the global derivation ---------------------------------------------

    def _derivation(self) -> pl.DataFrame:
        """The labelled, adjusted frame for the WHOLE configured window.

        Computed once per instance and cached. Every window densifier slices
        this, so every chunk of one conversion shares one anchor per PERMNO.

        **Also the ANCHOR GATE** (WR-02). It sits here, on the FIRST derivation
        of an instance, because this is the one method BOTH conversion entry
        points call: `from_raw_data_chunked` reaches it through
        `_raw_axes_in_range`, and `from_raw_data` through `_raw_data_to_xr`.
        While the gate lived in `_raw_axes_in_range` only the chunked path was
        gated, so `from_raw_data().save()` could extend a store in place onto a
        second anchor -- and, having written no sidecar, latched that store into
        a permanent refusal for every later chunked run. The cache check above
        keeps it once per instance: a re-dated config resets the cache in the
        `config` setter, which is exactly when the anchor must be re-checked.
        """
        cached = getattr(self, "_derivation_cache", None)
        if cached is not None:
            return cached

        self._assert_anchor_unchanged(self._adjustment_record())

        frame = self._scan_raw()
        # `is not None`, NOT truthiness (WR-01). The config setter above already
        # refuses an empty tuple, so on every path through a constructor the two
        # spellings agree -- and the explicit test is kept anyway, because
        # `_derivation` is reachable with a config the setter never inspected: a
        # field assigned after construction, a `dataclasses.replace`, a JSON
        # round trip into an object whose branch did not run, a future change of
        # the field's default. On a truthiness gate every one of those paths
        # converts the ENTIRE raw tier while the config says the roster is
        # empty, and nothing warns. That is the class of bug that comes back
        # silently, so the gate states the condition it means.
        if self.config.permnos is not None:
            frame = frame.filter(
                pl.col("symbol").is_in(list(self.config.permnos))
            )
        frame = frame.collect()

        reference = CrspReference(self.config.reference_dir)
        self._symbology = CrspSymbology(
            reference.table("stksecurityinfohist")
        )
        # The panel's `symbol` IS the PERMNO (D-01), and the raw frame's
        # `symbol` column ALREADY holds it -- `wrds_crsp.py:317-319` verbatim:
        # "Raw `symbol` is the PERMNO as a string, and a typed `permno` Int64
        # column rides along." So the panel's identity axis is reached by a
        # CAST of the column the raw tier already wrote, not by dropping it and
        # labelling the rows with a date-valid ticker.
        #
        # One column, one spelling, from here on: the int64 `symbol` and the
        # Int64 `permno` are now the same number, which is why `permno` leaves
        # `CRSP_EXTRA_VARIABLES` -- a float64 copy of the coordinate is not an
        # extra variable, it is a second source of truth.
        frame = frame.with_columns(pl.col("symbol").cast(pl.Int64))

        frame = frame.sort(["permno", "timestamp"])
        derived = frame.with_columns(
            # The SENTINEL is excluded BEFORE the abs, not after: a
            # delisting-AMOUNT row (`_NO_PRICE_FLAGS`) carries no market price,
            # and CRSP's way of saying so is `dlyprc = 0.000000`. `abs(0.0)` is
            # still 0.0, so an `abs()` that ran first would publish a $0.00
            # trade on a day that had none. The flag test is case-insensitive
            # and whitespace-stripped, exactly as `_apply_security_filter`
            # treats `dlydelflg`; the bare `== 0.0` arm catches a sentinel
            # written under a flag this tuple does not yet name.
            pl.when(
                pl.col("dlyprcflg")
                .str.strip_chars()
                .str.to_uppercase()
                .is_in(list(_NO_PRICE_FLAGS))
                .fill_null(False)
                | (pl.col("dlyprc") == 0.0).fill_null(False)
            )
            .then(None)
            .otherwise(pl.col("dlyprc").abs())
            .alias("close"),
            (1.0 + pl.col("dlyret").fill_null(0.0))
            .cum_prod()
            .over("permno")
            .alias("_G"),
        )
        # The anchor is the PERMNO's last row carrying a USABLE LEVEL -- a
        # strictly positive close AND the share factor every adjusted volume is
        # scaled by -- not simply its last row, and not merely its last
        # non-null one. "Non-null" was the loophole: CRSP's no-price sentinel is
        # the number 0.0, which passes `is_not_null()` and then makes
        # `adjClose = 0.0 * _G / _G_anchor` exactly 0.0 on every day of that
        # security's history (CR-01). `dlycumfacshr` is in the SAME predicate so
        # that every quantity read off the anchor comes from ONE row that
        # carries all of them; selecting on `close` alone and then reading a
        # NULL `dlycumfacshr` off that row is what made `adjVolume` NaN for a
        # whole delisted history while raw `volume` was fully populated (CR-02).
        anchor = (
            derived.filter(
                pl.col("close").is_not_null()
                & (pl.col("close") > 0.0)
                & pl.col("dlycumfacshr").is_not_null()
            )
            .group_by("permno")
            .agg(
                pl.col("close").last().alias("_close_anchor"),
                pl.col("_G").last().alias("_G_anchor"),
                pl.col("dlycumfacshr").last().alias("_cumfacshr_anchor"),
            )
        )
        derived = derived.join(anchor, on="permno", how="left")
        self._assert_anchor_usable(derived)

        derived = derived.with_columns(
            # NaN wherever there is no close. The chain `_G` is defined on a
            # priceless day (a null return contributes 1), so without this
            # mask the anchor's level would be published as that day's
            # adjusted price -- inventing a price on a day that had none, and
            # one that a return series would then diff against.
            pl.when(pl.col("close").is_null())
            .then(None)
            .otherwise(
                pl.col("_close_anchor") * pl.col("_G") / pl.col("_G_anchor")
            )
            .alias("adjClose")
        )
        derived = derived.with_columns(
            (pl.col("adjClose") / pl.col("close")).alias("_factor"),
            (pl.col("dlycumfacshr") / pl.col("_cumfacshr_anchor"))
            .alias("_volume_factor"),
        )
        derived = derived.with_columns(
            pl.col("dlycumfacpr")
            .shift(1)
            .over("permno")
            .alias("_prev_cumfacpr")
        )
        # AFTER the chain, never before: dropping a filtered day before the
        # cumulative product would make the next kept day's adjusted move span
        # a return the panel no longer shows.
        derived = self._apply_security_filter(derived)
        # There is no identity-resolution step here, and no place one could go
        # (D-01). Both things the deleted one existed for are unspellable on a
        # PERMNO axis: a same-day collision needs two PERMNOs in one
        # `(date, symbol)` cell, and the raw tier already asserts
        # `(permno, dlycaldt)` uniqueness (`wrds_crsp.py:818-840`); a seam needs
        # a symbol column to change company, and a PERMNO column never does.
        return self._finalise(derived)

    def _assert_anchor_usable(self, derived: pl.DataFrame) -> None:
        """Refuse, by PERMNO, rather than publish an unusable adjusted column.

        Two DISTINCT causes, two messages, because they need different
        remedies:

        1. **No usable anchor row.** No row of the PERMNO inside the window
           carries both a strictly positive `dlyprc` and a non-null
           `dlycumfacshr`, so `_close_anchor` is null and every `adj*` value
           would be NaN -- or, before the sentinel guard above existed, exactly
           0.0. The remedy is a different window or a different roster.
        2. **A return chain that reaches zero.** `_G` is `cum_prod(1 + dlyret)`,
           so a `dlyret` of exactly -1.0 -- a legal CRSP total loss -- makes
           `_G_anchor` exactly 0.0 and `close_A * _G_t / 0.0` `inf` before the
           loss and `NaN` after it, under IEEE semantics and with nothing
           raised. The remedy is to inspect that security's returns.

        **Why a refusal and not a NaN column.** `alpha158` and `fret` read the
        five `adj*` names and nothing else. A zeroed column turns into `0/0 ->
        NaN` returns, `x/0 -> inf` ratios and a cross-sectional rank pinned to
        the bottom every day; an all-NaN `adjVolume` silently drops the security
        from every liquidity screen. Both happen for exactly the securities that
        delisted, which is the survivorship bias D-10 exists to remove coming
        back through a different door -- and neither raises anywhere downstream.

        The messages carry `class_name`, PERMNO digits, the configured dates and
        CRSP column names only: no credential, no path outside the configured
        store, and never the frame.
        """
        missing = (
            derived.filter(pl.col("_close_anchor").is_null())
            .get_column("permno")
            .unique()
            .sort()
            .to_list()
        )
        if missing:
            raise ValueError(
                f"{self.class_name}: {len(missing)} PERMNO(s) have NO usable "
                f"adjustment anchor in [{self.config.start_date}, "
                f"{self.config.end_date}]: {missing[:10]}. The anchor is a "
                f"PERMNO's last row inside that window carrying BOTH a strictly "
                f"positive dlyprc AND a non-null dlycumfacshr. CRSP writes "
                f"dlyprc = 0.000000 on a delisting-AMOUNT row "
                f"(dlyprcflg in {list(_NO_PRICE_FLAGS)}) as a NO-PRICE "
                f"sentinel, and leaves dlycumfacshr NULL there, so such a row "
                f"is not a level and cannot anchor a series. Refusing rather "
                f"than publishing an all-zero adjClose or an all-NaN adjVolume "
                f"for these securities. Widen [start_date, end_date] to include "
                f"a day on which they traded, or drop them from config.permnos."
            )

        degenerate = (
            derived.filter(
                (
                    (pl.col("_G_anchor") == 0.0)
                    | pl.col("_G_anchor").is_infinite()
                    | pl.col("_G_anchor").is_nan()
                ).fill_null(True)
            )
            .get_column("permno")
            .unique()
            .sort()
            .to_list()
        )
        if degenerate:
            raise ValueError(
                f"{self.class_name}: {len(degenerate)} PERMNO(s) have an "
                f"adjustment anchor whose cumulative return chain is 0.0 or "
                f"non-finite in [{self.config.start_date}, "
                f"{self.config.end_date}]: {degenerate[:10]}. adjClose is "
                f"close_anchor * G_t / G_anchor with G = cum_prod(1 + dlyret), "
                f"so a dlyret of exactly -1.0 -- a legal CRSP total loss -- "
                f"makes G_anchor 0.0, every earlier day inf and every later day "
                f"NaN, without raising anywhere downstream. Refusing rather "
                f"than publishing an infinite adjusted series. Inspect dlyret "
                f"for these PERMNO(s) in the raw tier."
            )

    # -- the security filter (D-06, D-17) -----------------------------------

    def _member_intervals(self) -> pl.DataFrame | None:
        """`permno_intervals(collision_universe)`, read ONCE per instance.

        `None` when no universe is configured, which is also "there is no
        membership fact to consult" for this frame's reader.

        **One reader, since 03.11-07**: the roster exemption in
        `_apply_security_filter` (GAP-C, operator P8 -- a named roster is not
        overruled by a type filter). The memo once served a second reader as
        well, a collision tie-break that asked which of two PERMNOs sharing a
        `(date, symbol)` cell was the universe member; that question cannot be
        asked on a PERMNO axis and its machinery is gone. The memo stays
        anyway, for the reason it was worth having with one reader too: the
        reference tier is read from disk, and a conversion that read it twice
        could be handed two different answers if it were rewritten underneath a
        long run. One read, one opinion about who was a member.

        Invalidated in the `config` setter beside `_derivation_cache`.
        """
        if self.config.collision_universe is None:
            return None
        cached = getattr(self, "_member_intervals_cache", None)
        if cached is None:
            from quantlab.dataset.crsp_membership import CrspMembership

            cached = CrspMembership(
                CrspReference(self.config.reference_dir)
            ).permno_intervals(self.config.collision_universe)
            self._member_intervals_cache = cached
        return cached

    def _roster_sources(self) -> list[str]:
        """Which EXPLICIT ROSTERS are in play, for the filter report.

        Computed separately from the exemption expression so the `'none'` preset
        -- which rejects nothing, so can rescue nothing -- still records that a
        roster WAS configured without paying for the per-row membership join.
        """
        sources: list[str] = []
        # `is not None` for the same reason `_derivation`'s gate is (WR-01): a
        # roster field is either unset or a list of names, and "unset" is
        # spelled None. An empty tuple reaching here through a setter-bypassing
        # path reports a roster of zero PERMNOs, which is what it is, rather
        # than reporting no roster at all.
        if self.config.permnos is not None:
            sources.append(
                f"config.permnos: {len(self.config.permnos)} PERMNO(s) named "
                f"explicitly, exempt on every date"
            )
        if self.config.collision_universe is not None:
            intervals = self._member_intervals()
            spells = 0 if intervals is None else intervals.height
            members = (
                0
                if intervals is None
                else intervals.get_column("permno").n_unique()
            )
            sources.append(
                f"config.collision_universe={self.config.collision_universe!r}: "
                f"{members} member PERMNO(s) over {spells} membership spell(s), "
                f"exempt on the dates inside a spell"
            )
        return sources

    def _roster_exemption(
        self, derived: pl.DataFrame
    ) -> tuple[pl.Expr, list[str]]:
        """`(the rows an explicit roster exempts from the type filter, sources)`.

        **The rule this implements** (GAP-C, the operator's decision of
        2026-09-20, `03.10-11-SUMMARY.md`: 「优先保证成分股不缺」). The security
        filter screens an UNSPECIFIED population; it must not overrule an
        explicit roster:

        - `config.permnos` -- the user NAMED these securities, so every date of
          each is exempt. A `--permnos` run is a roster, not a screen.
        - `config.collision_universe` -- the index provider already decided
          membership, so a member is exempt on the dates INSIDE its membership
          spell and on no others. Scoping the exemption to the spell is what
          keeps it from becoming a blanket widening: a PERMNO's pre-membership
          or post-membership era is still an unspecified population.
        - neither -- a literal false. Nothing is exempt, and the filter behaves
          exactly as it did before this method existed.

        **Why this is not a filter bug but a specification one.** `equity_common`
        rejects `sharetype='UG'`, which in CRSP marks the publicly traded
        PARTNERSHIP era of a security -- and that era belongs to real S&P 500
        members: Blackstone 2007-06-22..2019-06-30, KKR 2010..2018, Carnival
        everything from 2003, Royal Dutch Petroleum (`AD`) its entire life. Seven
        of the 1,956 historical S&P 500 member PERMNOs are hit, and the loss is
        MID-SECURITY, which downstream is indistinguishable from a late IPO.

        **The universe arm is materialised as a Series literal, deliberately.**
        A per-date interval test is a join, and the two join-free alternatives
        are both worse: a disjunction over every spell is O(spells) passes over
        the frame (thousands, for a real index history), and a `(permno, date)`
        membership set is millions of pairs for a decade-long panel. So the join
        happens here, once, and its boolean result is returned as
        `pl.lit(series)` -- which keeps the contract a plain polars expression
        the caller folds into `_keep`. It is aligned to the row order of the
        `derived` argument, so the caller must evaluate it against THAT frame.
        """
        sources = self._roster_sources()
        terms: list[pl.Expr] = []

        # `is not None`, as everywhere this field is read (WR-01). An empty
        # roster contributes a term that matches nothing, which is the honest
        # reading of "these securities, of which there are none" -- never the
        # blanket exemption a truthiness test's else-branch would give it.
        if self.config.permnos is not None:
            # `permno` is the identity column (Int64); the config holds the
            # digit STRINGS the CLI and the raw tier speak, normalised by the
            # config setter, so the cast is the whole of the comparison.
            terms.append(
                pl.col("permno")
                .cast(pl.String)
                .is_in(list(self.config.permnos))
                .fill_null(False)
            )

        intervals = self._member_intervals()
        if intervals is not None:
            terms.append(pl.lit(self._member_days(derived, intervals)))

        if not terms:
            return pl.lit(False), sources

        expression = terms[0]
        for term in terms[1:]:
            expression = expression | term
        return expression, sources

    @staticmethod
    def _member_days(
        derived: pl.DataFrame, intervals: pl.DataFrame
    ) -> pl.Series:
        """Per ROW of `derived`: is this `(permno, timestamp)` inside a spell?

        Both ends inclusive, which is `permno_intervals`' own contract. A row
        whose PERMNO appears in no spell at all falls out of the inner join and
        reads False through the left join back, rather than null -- "not a
        member" and "no membership data for this security" are the same answer
        to the only question the filter asks.
        """
        rows = derived.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Date).alias("_day"),
        ).with_row_index("_row")
        spells = intervals.select(
            pl.col("permno").cast(pl.Int64),
            pl.col("start_date"),
            pl.col("end_date"),
        )
        hit = (
            rows.join(spells, on="permno", how="inner")
            .filter(
                (pl.col("_day") >= pl.col("start_date"))
                & (pl.col("_day") <= pl.col("end_date"))
            )
            .select(pl.col("_row").unique())
            .with_columns(pl.lit(True).alias("_member"))
        )
        return (
            rows.join(hit, on="_row", how="left")
            .sort("_row")
            .get_column("_member")
            .fill_null(False)
        )

    def _apply_security_filter(self, derived: pl.DataFrame) -> pl.DataFrame:
        """Drop the rows the configured filter rejects, and say what went.

        The verdict is PER ROW, read off `dsf_v2`'s own per-day type columns,
        so a security that stopped being common stock keeps exactly the era in
        which it was (D-17). Every listed column must match and a NULL never
        matches -- "unknown type" is not "the type you asked for".

        **The delisting row inherits its PERMNO's previous verdict** (D-10).
        A delisted security's last row is precisely where CRSP's type columns
        go blank, and that row carries the delisting RETURN. Judging it on its
        own blank types would drop the -60% day and let survivorship bias back
        in through the filter, one row at a time.

        The PERMNO axis (D-01) did NOT make this inheritance redundant, and the
        distinction is worth stating because a neighbouring mechanism WAS made
        redundant by it and deleted in 03.11-07. That one was about the NAME: a
        delisting row's `ticker` also goes blank, and on a ticker axis a row
        with no name had no column to live in, so symbology carried the
        previous interval's symbol onto it. Keyed on the PERMNO, the row has
        its column regardless -- nothing needs carrying to keep it in the
        panel. What still goes blank is the row's TYPE columns, and this filter
        reads those, so without the inheritance below every delisted
        security silently loses its final, largest-magnitude day and the panel
        stays perfectly well-formed while it happens (T-03.11-09). The
        alternative considered and rejected: exempt `dlydelflg == 'Y'` rows
        from the filter outright. That is wider than the fact warrants -- a
        security legitimately excluded by type on its last trading day would be
        readmitted on its delisting day -- whereas inheriting says exactly what
        is true, that the verdict has not changed because nothing observable
        about the security did.

        **An EXPLICIT ROSTER overrides the verdict** (GAP-C, `_roster_exemption`).
        The exemption is OR-ed in AFTER the delisting carry, which is the only
        ordering that leaves both mechanisms whole: a rescued delisting row is
        still rescued by the carry it inherited, and a rescued ordinary row does
        not become its PERMNO's carried verdict for the next day.

        The report is BUILT here and WRITTEN once per conversion from
        `_raw_axes_in_range`, for the same reason the anchor record is: that
        hook runs exactly once, after the derivation has succeeded.
        """
        rows_total = derived.height
        if not self._security_filter:
            self._filter_report = self._build_filter_report(
                rows_total, derived.head(0), derived.head(0),
                self._roster_sources(), kept=derived,
            )
            return derived

        predicate = pl.lit(True)
        for column, allowed in self._security_filter.items():
            predicate = predicate & pl.col(column).is_in(list(allowed)).fill_null(
                False
            )

        derived = derived.sort(["permno", "timestamp"]).with_columns(
            predicate.alias("_keep_raw")
        )
        derived = derived.with_columns(
            pl.when(
                pl.col("dlydelflg").str.strip_chars().str.to_uppercase()
                == pl.lit(_DELISTING_FLAG)
            )
            .then(
                pl.coalesce(
                    pl.col("_keep_raw").shift(1).over("permno"),
                    pl.col("_keep_raw"),
                )
            )
            .otherwise(pl.col("_keep_raw"))
            .alias("_carried")
        )

        exemption, sources = self._roster_exemption(derived)
        derived = derived.with_columns(
            exemption.fill_null(False).alias("_roster_exempt")
        )
        derived = derived.with_columns(
            (pl.col("_carried") | pl.col("_roster_exempt")).alias("_keep")
        )

        # Exactly "what would have been dropped and was not": the carried
        # verdict is the whole of the pre-exemption decision, so a row the
        # delisting carry already rescued is NOT a roster rescue.
        rescued = derived.filter(
            ~pl.col("_carried") & pl.col("_roster_exempt")
        )
        dropped = derived.filter(~pl.col("_keep"))
        self._filter_report = self._build_filter_report(
            rows_total, dropped, rescued, sources,
            kept=derived.filter(pl.col("_keep")),
        )
        if dropped.height:
            logger.warning(
                f"{self.class_name}: the security filter dropped "
                f"{dropped.height} of {rows_total} row(s) across "
                f"{dropped['permno'].n_unique()} PERMNO(s); see "
                f"{FILTER_REPORT_SUFFIX} beside the store for the per-type and "
                f"per-PERMNO breakdown."
            )
        if rescued.height:
            logger.warning(
                f"{self.class_name}: an explicit roster KEPT {rescued.height} "
                f"row(s) across {rescued['permno'].n_unique()} PERMNO(s) that "
                f"the security filter would have dropped; see "
                f"{FILTER_REPORT_SUFFIX} beside the store, key "
                f"'roster_overrides', for the per-PERMNO date ranges. The "
                f"filter screens an unspecified population and does not "
                f"overrule a named roster."
            )
        return derived.filter(pl.col("_keep")).drop(
            ["_keep_raw", "_carried", "_roster_exempt", "_keep"]
        )

    @staticmethod
    def _type_combination() -> pl.Expr:
        """`"sharetype/securitytype/securitysubtype/issuertype/usincflg"`.

        A null component renders as `"None"` rather than turning the whole key
        null, because "which combination was dropped" is exactly the question
        a row with missing types needs answered.
        """
        parts = [
            pl.col(name).fill_null(pl.lit("None")) for name in _TYPE_COLUMNS
        ]
        expression = parts[0]
        for part in parts[1:]:
            expression = expression + pl.lit("/") + part
        return expression.alias("_types")

    def _build_filter_report(
        self,
        rows_total: int,
        dropped: pl.DataFrame,
        rescued: pl.DataFrame | None = None,
        sources: list[str] | tuple[str, ...] = (),
        kept: pl.DataFrame | None = None,
    ) -> dict:
        """The `{zarr}.crsp_filter_report.json` payload.

        **`roster_overrides` is always present**, with zero counts when no
        roster is configured, rather than omitted. A reader can then tell "no
        override happened" from "this store predates the feature" by the key's
        presence alone -- omitting it would make a store written before GAP-C was
        closed indistinguishable from one written after it with nothing rescued,
        and those two need different actions.

        **`admitted_without_ticker` follows the same rule, and for the same
        reason** (D-14 / RULING 1). It is always written, empty or not, so that
        "nobody was admitted without a ticker" and "this store predates the
        count" stay distinguishable. It is the only place the panel says out
        loud that the PERMNO axis widened the admission rule, so the WARNING
        beside it fires here rather than at a call site -- both branches of
        `_apply_security_filter` reach this method exactly once per conversion,
        and neither of them could emit it without duplicating the other.

        Every pre-existing key keeps its pre-existing meaning: a rescued row is
        counted in `rows_kept` (it IS kept) and appears in neither
        `dropped_by_type` nor `dropped_permnos` (it was not dropped), so
        `rows_kept + rows_dropped == rows_total` still holds.
        """
        report: dict = {
            "filter": {
                "requested": self._jsonable_filter(self.config.security_filter),
                "resolved": {
                    column: list(allowed)
                    for column, allowed in self._security_filter.items()
                },
            },
            "rows_total": int(rows_total),
            "rows_kept": int(rows_total - dropped.height),
            "rows_dropped": int(dropped.height),
            "dropped_by_type": {},
            "dropped_permnos": {},
            "roster_overrides": {
                "sources": [str(source) for source in sources],
                "rows_rescued": 0 if rescued is None else int(rescued.height),
                "permnos": {},
            },
            "admitted_without_ticker": self._admitted_without_ticker(kept),
        }
        self._warn_admitted_without_ticker(report["admitted_without_ticker"])
        if rescued is not None and not rescued.is_empty():
            report["roster_overrides"]["permnos"] = self._permno_breakdown(
                rescued
            )
        if dropped.is_empty():
            return report

        typed = dropped.with_columns(self._type_combination())
        by_type = (
            typed.group_by("_types")
            .agg(pl.len().alias("rows"))
            .sort("_types")
        )
        report["dropped_by_type"] = {
            str(record["_types"]): int(record["rows"])
            for record in by_type.to_dicts()
        }

        report["dropped_permnos"] = self._permno_breakdown(dropped)
        return report

    def _admitted_without_ticker(self, kept: pl.DataFrame | None) -> dict:
        """Which admitted PERMNOs could NOT have entered a ticker-keyed panel.

        `{"permnos": [...numeric order...], "rows": N}` (D-14 / RULING 1).

        **The predicate is the retired ticker labeller's own, reproduced
        exactly here**, because the question is counterfactual: "would the
        ticker axis have admitted this row". That labeller (deleted with the
        rest of the ticker-identity machinery in 03.11-07 -- this method is the
        only place its behaviour still has to be stated) as-of joined each
        row's date onto the NAMED intervals, `symbol_intervals()` with the null
        symbols dropped, and discarded whatever it could not label. So a PERMNO
        none of whose panel days falls inside an interval that CARRIES a ticker
        is one the old axis never let in. Using the full interval table instead
        (nulls included) would answer a different question and count nobody: a
        never-ticker PERMNO does have intervals, they just have no name on
        them.

        A PERMNO is counted only when NONE of its panel days is covered.
        Partial coverage is not this field's subject -- a security that had a
        ticker for part of the window was admitted for that part on either
        axis, and the rows the ticker axis would have dropped mid-history were
        counted by a separate `unlabelled` tally that went away with the
        labeller.

        **What this is NOT.** It is not a filter. Nothing is excluded here, and
        `FILTERABLE_COLUMNS` stays at its nine TYPE columns. D-10 originally
        asked for an explicit `securitytype`/`sharetype` predicate to replace
        the implicit "must have a ticker" rule; RESEARCH R5c measured that
        1,003 of the 1,012 never-ticker PERMNOs read `EQTY/COM/NS` -- the
        ordinary-common-stock combination -- so no type predicate can separate
        them from real common stock. RULING 1: admit them, and count them.
        """
        empty: dict = {"permnos": [], "rows": 0}
        if kept is None or kept.is_empty() or self._symbology is None:
            return empty

        intervals = self._symbology.symbol_intervals().drop_nulls("symbol")
        rows = (
            kept.select("permno", "timestamp")
            .with_columns(pl.col("timestamp").dt.date().alias("_as_of"))
            .sort(["permno", "_as_of"])
        )
        if intervals.is_empty():
            # No named interval anywhere: every admitted PERMNO is one the
            # ticker axis would have dropped entirely.
            per_permno = rows.group_by("permno").agg(pl.len().alias("rows"))
            return self._render_admitted_without_ticker(per_permno)

        labelled = rows.join_asof(
            intervals.select("permno", "symbol", "start_date", "end_date").sort(
                ["permno", "start_date"]
            ),
            left_on="_as_of",
            right_on="start_date",
            by="permno",
            strategy="backward",
        )
        # A backward as-of join alone attaches the last interval to every later
        # row; the END is what says the row is actually inside it. Same two
        # steps, same order, as the retired labeller's.
        labelled = labelled.with_columns(
            pl.when(
                pl.col("end_date").is_not_null()
                & (pl.col("_as_of") > pl.col("end_date"))
            )
            .then(None)
            .otherwise(pl.col("symbol"))
            .alias("symbol")
        )
        per_permno = labelled.group_by("permno").agg(
            pl.col("symbol").is_not_null().any().alias("_ever_labelled"),
            pl.len().alias("rows"),
        )
        return self._render_admitted_without_ticker(
            per_permno.filter(~pl.col("_ever_labelled"))
        )

    @staticmethod
    def _render_admitted_without_ticker(per_permno: pl.DataFrame) -> dict:
        """`{"permnos": [...], "rows": N}` from a `(permno, rows)` frame."""
        if per_permno.is_empty():
            return {"permnos": [], "rows": 0}
        permnos = sort_symbol_axis(
            int(value) for value in per_permno.get_column("permno").to_list()
        )
        return {
            "permnos": permnos,
            "rows": int(per_permno.get_column("rows").sum()),
        }

    def _warn_admitted_without_ticker(self, admitted: dict) -> None:
        """Say out loud that the admission rule widened, when it did."""
        if not admitted["permnos"]:
            return
        logger.warning(
            f"{self.class_name}: {len(admitted['permnos'])} PERMNO(s) "
            f"({admitted['rows']} row(s)) were ADMITTED to the panel with no "
            f"ticker on any of their days. On the old ticker axis these "
            f"securities could never enter a panel at all -- a row with no "
            f"symbol had no column to live in -- so this is a widening of the "
            f"ADMISSION RULE (D-14), not a filter that stopped working. The "
            f"security filter's verdict is unchanged for every one of them. "
            f"See {FILTER_REPORT_SUFFIX} beside the store, key "
            f"'admitted_without_ticker', for the PERMNO list."
        )

    def _permno_breakdown(self, frame: pl.DataFrame) -> dict:
        """`{PERMNO: {symbol, types, rows, first, last}}` for a set of rows.

        ONE rendering, used for the rows the filter dropped and for the rows a
        roster rescued: "which PERMNO, spelled how, over which dates, on which
        type combination" is the same question in both directions, and two
        renderings of it could drift apart while both looked right.
        """
        typed = frame.with_columns(self._type_combination())
        per_permno = (
            typed.sort(["permno", "timestamp"])
            .group_by("permno")
            .agg(
                pl.col("symbol").last().alias("symbol"),
                pl.col("_types").unique().sort().alias("types"),
                pl.len().alias("rows"),
                pl.col("timestamp").min().alias("first"),
                pl.col("timestamp").max().alias("last"),
            )
            .sort("permno")
        )
        return {
            str(record["permno"]): {
                "symbol": None
                if record["symbol"] is None
                else str(record["symbol"]),
                "types": [str(value) for value in record["types"]],
                "rows": int(record["rows"]),
                "first": str(record["first"])[:10],
                "last": str(record["last"])[:10],
            }
            for record in per_permno.to_dicts()
        }

    @staticmethod
    def _jsonable_filter(value):
        """The configured filter as JSON: a preset name, or lists not tuples."""
        if isinstance(value, dict):
            return {
                str(column): [str(item) for item in allowed]
                for column, allowed in value.items()
            }
        return value

    def filter_report_path(self) -> Path:
        """`{zarr_file_path}.crsp_filter_report.json`, a SIBLING of the store."""
        return Path(str(self.config.zarr_file_path) + FILTER_REPORT_SUFFIX)

    def _finalise(self, derived: pl.DataFrame) -> pl.DataFrame:
        """Project the derivation onto the panel's variables and cache it."""
        frame = derived.with_columns(
            pl.col("dlyopen").alias("open"),
            pl.col("dlyhigh").alias("high"),
            pl.col("dlylow").alias("low"),
            pl.col("dlyvol").alias("volume"),
            (pl.col("dlyopen") * pl.col("_factor")).alias("adjOpen"),
            (pl.col("dlyhigh") * pl.col("_factor")).alias("adjHigh"),
            (pl.col("dlylow") * pl.col("_factor")).alias("adjLow"),
            (pl.col("dlyvol") * pl.col("_volume_factor")).alias("adjVolume"),
            # UNADJUSTED cash per share, on the EX-DATE -- the Tiingo
            # convention, so a consumer reading `divCash` needs no CRSP
            # vocabulary. CRSP splits the day's cash into an ordinary and a
            # non-ordinary component; Tiingo states one number, so they are
            # summed. Both null is 0.0, not NaN: a day with no distribution
            # paid a KNOWN amount of nothing. Anything finer than the daily
            # total (declaration/record/pay dates, distribution codes) stays
            # available raw in `stkdistributions` under `_reference/`.
            (
                pl.col("dlyorddivamt").fill_null(0.0)
                + pl.col("dlynonorddivamt").fill_null(0.0)
            ).alias("divCash"),
            # 1.0 on the PERMNO's FIRST ROW INSIDE THE WINDOW: there is no
            # previous `dlycumfacpr` to divide by there, and an ordinary day's
            # split factor IS 1.0. So a split that fell on a window's opening
            # day reads 1.0 rather than its real ratio -- the same edge every
            # differenced series has, and the reason `facprc` (CRSP's own
            # per-day factor, which needs no previous row) sits beside it.
            pl.coalesce(
                pl.col("_prev_cumfacpr") / pl.col("dlycumfacpr"),
                pl.lit(1.0),
            ).alias("splitFactor"),
            # -- the CRSP extras, in `CRSP_EXTRA_VARIABLES` order ------------
            # `permno`/`permco` are already named; `close` came from the
            # derivation above. Everything else is renamed or rescaled here.
            #
            # NO `fill_null` on `ret`: the stored return keeps CRSP's null as
            # a NaN (D-09). The `fill_null(0.0)` that DOES exist lives inside
            # `_derivation`'s cumulative product and nowhere else, because a
            # gap-spanning return already covers the missing day.
            pl.col("dlyret").alias("ret"),
            pl.col("dlyretx").alias("retx"),
            # CRSP states both in THOUSANDS; the panel states shares and USD,
            # so the x1000 happens once here rather than at every call site.
            (pl.col("shrout") * 1000).alias("shrout"),
            (pl.col("dlycap") * 1000.0).alias("market_cap"),
            pl.col("dlybid").alias("bid"),
            pl.col("dlyask").alias("ask"),
            # The no-trade indicator (D-19), read off the FLAG rather than off
            # the sign of the price: CIZ carries no negative prices, so a sign
            # test would flag nothing. A null flag stays null -- "unknown" is
            # not "was a trade".
            pl.when(pl.col("dlyprcflg").is_null())
            .then(None)
            .when(pl.col("dlyprcflg") == pl.lit("BA"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("prc_is_bidask"),
            # 1.0 marks the row whose `dlyret` IS the delisting return. It is
            # a MARKER, not an instruction: nothing multiplies by it, because
            # the return is already in the chain (D-10).
            pl.when(pl.col("dlydelflg").is_null())
            .then(None)
            .when(pl.col("dlydelflg") == pl.lit("Y"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("is_delisting"),
            pl.col("dlynumtrd").alias("numtrd"),
            pl.col("dlycumfacpr").alias("cumfacpr"),
            pl.col("dlycumfacshr").alias("cumfacshr"),
            pl.col("dlyfacprc").alias("facprc"),
            # The closing TRADE price, kept beside `close = abs(dlyprc)` so a
            # caller who needs "was there a trade, and at what price" has it
            # without re-deriving it from the flag.
            pl.col("dlyclose").alias("close_trade"),
        )
        columns = ["timestamp", "symbol", *self.TIINGO_VARIABLES, *self.EXTRA_VARIABLES]
        frame = frame.select(
            pl.col("timestamp"),
            pl.col("symbol"),
            *[
                pl.col(name).cast(pl.Float64).alias(name)
                for name in (*self.TIINGO_VARIABLES, *self.EXTRA_VARIABLES)
            ],
        ).select(columns)
        self._derivation_cache = frame
        return frame

    # -- the adjustment anchor, and its sidecar ------------------------------

    @classmethod
    def adjustment_sidecar_path(cls, config: CrspDatasetConfig) -> Path:
        """`{zarr_file_path}.crsp_adjustment.json`, a SIBLING of the store."""
        return Path(str(config.zarr_file_path) + cls.ADJUSTMENT_SIDECAR_SUFFIX)

    def _adjustment_record(self) -> dict:
        """What this conversion's `adj*` columns are anchored to.

        The anchor is a PERMNO's last priced day inside
        `[start_date, end_date]`, so the WINDOW identifies it. `product_end`
        rides along because the same window read against a newer CRSP vintage
        can have a later last row for a security that kept trading -- the
        vintage is part of "which anchor", not decoration.
        """
        reference = CrspReference(self.config.reference_dir)
        return {
            "start_date": str(self.config.start_date),
            "end_date": str(self.config.end_date),
            "product_end": str(reference.product_end),
            "rule": self.ADJUSTMENT_RULE,
        }

    def _assert_anchor_unchanged(self, record: dict) -> None:
        """Refuse to write into a store built against a DIFFERENT anchor.

        **Why a refusal rather than a rescale.** Extending `end_date` moves
        every still-listed PERMNO's anchor, which multiplies every earlier
        `adj*` value of that PERMNO by a constant (ratios within a series
        survive; levels do not). That is the same thing Tiingo does when it
        restates adjusted history. The difference is the WRITER: the chunk
        ledger APPENDS windows and never rewrites a finished one, so an
        in-place extension would leave the old windows on the old anchor and
        the new ones on the new anchor -- one column, two anchors, and a
        fabricated return at the join that no reader could see.

        Rebuilding is the remedy because it is cheap: CRSP publishes once a
        year, and a full daily S&P panel re-converts in minutes.

        Runs BEFORE the derivation and before any append -- the one place a
        refusal cannot be half-done.
        """
        import json

        store = Path(str(self.config.zarr_file_path))
        if not store.exists():
            return

        sidecar = self.adjustment_sidecar_path(self.config)
        if not sidecar.exists():
            raise ValueError(
                f"{self.class_name}: the store at {str(store)!r} exists but "
                f"its adjustment sidecar {sidecar.name!r} is missing, so "
                f"which anchor its adjusted columns were computed against is "
                f"unknown. A store written against a different anchor is "
                f"indistinguishable from this one, and appending would splice "
                f"two anchors into one column. Convert into a NEW "
                f"zarr_file_path, or delete {str(store)!r} together with its "
                f"'.crsp_*.json' sidecars and rebuild."
            )

        try:
            recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{self.class_name}: the adjustment sidecar {str(sidecar)!r} "
                f"could not be read ({type(exc).__name__}: {exc}). It records "
                f"which anchor {str(store)!r} was built against; without it "
                f"an append could splice two anchors. Delete the store and "
                f"its '.crsp_*.json' sidecars and rebuild, or convert into a "
                f"new zarr_file_path."
            ) from exc

        if recorded != record:
            raise ValueError(
                f"{self.class_name}: {str(store)!r} was built against the "
                f"adjustment anchor {recorded} (recorded in "
                f"{str(sidecar)!r}) and this conversion would use "
                f"{record}. The anchor is each PERMNO's last priced day in "
                f"the configured window, so a different window (here "
                f"end_date {recorded.get('end_date')!r} -> "
                f"{record.get('end_date')!r}) rescales every earlier adjusted "
                f"value. The chunk ledger appends and never rewrites, so the "
                f"result would be ONE column holding TWO anchors, with a "
                f"fabricated return at the seam. Choose a new "
                f"zarr_file_path, or delete {str(store)!r} together with its "
                f"'.crsp_*.json' sidecars and rebuild the whole window."
            )

    def _write_adjustment_record(self, record: dict) -> None:
        """Record the anchor before the first append, so a store cannot exist
        without it."""
        if Path(str(self.config.zarr_file_path)).exists():
            return
        write_json_atomically(
            self.adjustment_sidecar_path(self.config),
            record,
            indent=2,
            sort_keys=True,
        )

    # -- axes and windows ---------------------------------------------------

    def _raw_axes_in_range(self):
        """Both axes of the CONVERTED panel, from the cached derivation.

        Overridden rather than inherited because `StockDataset`'s version
        reads the raw `symbol` column as text, and this panel's axis is the
        int64 PERMNO (D-01) -- the same labels, cast, and read off the
        DERIVATION rather than off raw so the `config.symbols` restriction and
        the security filter have already been applied.

        One of THREE places the pinned symbol axis is decided -- the others
        being `StockDataset._raw_axes_in_range` and
        `BaseDataset._raw_axes_in_range`. All three take their ORDER from
        `quantlab/utils/symbol_axis.py:sort_symbol_axis`, which is where that
        contract is stated and argued; it is deliberately not restated here.

        Also where the chunked path RECORDS its provenance -- the gate itself
        now lives in `_derivation()`, which both entry points call (WR-02). This
        method is still the right place for the two WRITES: `from_raw_data_chunked`
        calls it ONCE, before `ChunkLedger.assert_consistent` and before any
        append, so the records land before the first irreversible write and are
        never written eleven times by an eleven-window run.
        """
        import pandas as pd

        record = self._adjustment_record()
        derivation = self._derivation()
        # ONE frame decides BOTH axes (WR-07). The `config.symbols` restriction
        # is applied to the derivation FIRST, and the symbol axis and the
        # timestamp axis are then read off that same filtered frame. Do not
        # separate them again: `_raw_data_to_xr_window` applies the restriction
        # too, so a timestamp axis taken from the unrestricted frame made
        # `from_raw_data_chunked` plan windows over every day ANY security
        # traded. A window the requested symbol has no rows in densifies to a
        # zero-length `timestamp` dimension and appends nothing, while
        # `_reconcile_new_listings` was handed `len(timestamps)` from the wider
        # frame -- so a later `widen` rewrite pins the on-disk chunk grid against
        # an extent the store will never reach, and the chunk ledger persists a
        # completed window over days that symbol never traded.
        if self.config.symbols:
            wanted = {int(symbol) for symbol in self.config.symbols}
            derivation = derivation.filter(
                pl.col("symbol").is_in(sorted(wanted))
            )
        # The ONE place the CRSP panel's symbol axis is decided -- its dtype
        # (int64 PERMNO, D-01) and its ORDER (numeric, D-19) both. `sorted()`
        # on the string spelling put 5-digit PERMNOs before 4-digit ones
        # ('10107' < '7000'); `sort_symbol_axis` is the single implementation
        # of "numeric order" this repo has (`quantlab/utils/symbol_axis.py`).
        symbols = sort_symbol_axis(
            int(value)
            for value in derivation.get_column("symbol").unique().to_list()
        )
        timestamps = derivation.get_column("timestamp").unique().to_list()

        # Written LAST, after the derivation has succeeded: a run that fails
        # in the derivation (an unusable adjustment anchor, a changed anchor)
        # must not leave an anchor record for a store that was never created.
        # A store, on the other hand, can never exist without one -- the first
        # append happens after this returns.
        self._write_identity_reports()
        self._write_adjustment_record(record)
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _write_identity_reports(self) -> None:
        """Write the identity sidecars, ONCE per conversion, and only when the
        store does not exist yet.

        One sidecar as of 03.11-07: `.crsp_filter_report.json`. The symbology
        report that sat beside it is gone -- all six of its keys were
        statements about a ticker axis (resolved collisions, PERMNO seams,
        carried labels, class respellings, unlabelled rows), and a sidecar that
        can only ever say "nothing happened" reads like evidence a check ran.
        Plan 09 puts a ticker sidecar here, which answers a different question:
        not what identity resolution DID, but what the numbers are CALLED. The
        plural in this method's name is kept for that arrival.

        Called once per run by each entry point, which is what keeps a windowed
        conversion from writing eleven copies of the same report -- or worse,
        eleven DIFFERENT ones, each describing a single window as if it
        described the store.

        **The store-exists guard is WR-03**, and it is a deliberate trade. Both
        writes run before `ChunkLedger.assert_consistent`, before
        `_reconcile_new_listings`' `on_new_listing='refuse'` arm and before the
        per-window axis check -- any of which still aborts the run after these
        files are on disk. Without the guard, a refused re-conversion replaced
        the SURVIVING store's `.crsp_filter_report.json` (the D-17 audit
        artifact whose whole purpose is to say what the store dropped) with
        numbers for a panel that was never written, and the report describing
        the real store was gone.

        The cost: an APPEND no longer refreshes the reports, so they describe
        the panel as it was FIRST written rather than as it stands. That is the
        lesser harm -- a slightly stale audit trail beats a confidently wrong
        one, and the anchor sidecar has carried exactly this guard since it was
        introduced. The STRONGER form, considered and not taken here because it
        needs a success signal this method cannot see: write to a temp name and
        `os.replace` after the append loop reports success, which would keep the
        reports fresh on every completed run and untouched on every aborted one.
        """
        if Path(str(self.config.zarr_file_path)).exists():
            return
        if self._filter_report is not None:
            write_json_atomically(
                self.filter_report_path(),
                self._filter_report,
                indent=2,
                sort_keys=True,
            )

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Densify ONE window of the cached derivation.

        `symbols` is a list of int64 PERMNOs here, not tickers (D-01) -- the
        base signature says `list[str]` because most vendors key on a ticker,
        and CRSP narrows it.
        """
        start = self._as_datetime(start_date)
        end = self._as_datetime(end_date)
        window = self._derivation().filter(
            (pl.col("timestamp") >= pl.lit(start))
            & (pl.col("timestamp") <= pl.lit(end))
        )
        # `int(...)`, not `str(...)`: the derivation's `symbol` column is Int64
        # (D-01), and `is_in` on a list of strings matches NOTHING against an
        # integer column -- it would filter every window down to zero rows
        # without raising.
        if symbols is not None:
            window = window.filter(
                pl.col("symbol").is_in([int(symbol) for symbol in symbols])
            )
        elif self.config.symbols:
            window = window.filter(
                pl.col("symbol").is_in(
                    [int(symbol) for symbol in self.config.symbols]
                )
            )
        self._assert_unique_panel_keys(window)

        data = (
            window.to_pandas()
            .set_index(["timestamp", "symbol"])
            .to_xarray()
        )
        if symbols is not None:
            data = data.reindex(symbol=list(symbols))
        return data

    def _raw_data_to_xr(self) -> xr.Dataset:
        """The WHOLE window, densified in one go -- and its provenance (WR-02).

        `BaseDataset.from_raw_data()` calls this and nothing else, so it is the
        non-chunked path's only chance to leave the records `_raw_axes_in_range`
        leaves on the chunked one. Overridden rather than left inherited because
        the omission was LATCHED: `_assert_anchor_unchanged` refuses any store
        that exists without an adjustment sidecar, so a store written by
        `from_raw_data().save()` could never be appended to again, and the user
        met a refusal naming a file they had never heard of.

        The writes happen AFTER the derivation has succeeded -- a run that
        fails in the derivation must leave no record of a store that was never
        created -- and before the densified window is returned, which is before
        `save()` creates the store. Both writers skip a path that already
        exists, so this is a first-write only.
        """
        record = self._adjustment_record()
        window = self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )
        self._write_identity_reports()
        self._write_adjustment_record(record)
        return window

    def _assert_unique_panel_keys(self, window: pl.DataFrame) -> None:
        """`(timestamp, symbol)` is unique, or the conversion fails.

        **A BACKSTOP, not the mechanism.** `symbol` IS the PERMNO here (D-01),
        so this pair is `(dlycaldt, permno)` under two other names, and
        `WrdsCrspAcquisition._assert_unique_keys` (`wrds_crsp.py:818-840`)
        already refuses any raw page that duplicates it. A duplicate reaching
        this point therefore means something between the raw tier and here
        MULTIPLIED rows -- a join that fanned out, a window read twice -- not
        that two securities were confused for one.

        It stays because the cost of being wrong is invisible: the inherited
        `dedup_raw_frame(keep="last")` would collapse the duplicates into one
        price series and leave a well-formed panel behind, with nothing
        recording that a choice was made. This is where that stops rather than
        where it gets silently resolved.
        """
        duplicates = (
            window.group_by(["timestamp", "symbol"])
            .agg(pl.len().alias("rows"))
            .filter(pl.col("rows") > 1)
            .sort(["timestamp", "symbol"])
        )
        if duplicates.height:
            sample = [
                f"{str(record['timestamp'])[:10]}/{record['symbol']}"
                f"x{record['rows']}"
                for record in duplicates.head(5).to_dicts()
            ]
            raise ValueError(
                f"{self.class_name}: {duplicates.height} (timestamp, symbol) "
                f"collision(s) in the window, first {sample}. `symbol` IS the "
                f"PERMNO (D-01), so this is a duplicated "
                f"(permno, dlycaldt) key -- the raw tier asserts that pair is "
                f"unique on every page it fetches "
                f"(WrdsCrspAcquisition._assert_unique_keys, "
                f"wrds_crsp.py:818-840), so these rows were multiplied AFTER "
                f"acquisition, not confused between two securities. Refusing "
                f"rather than collapsing them into one series. Inspect the raw "
                f"parquet for these (permno, date) pairs; if the raw tier is "
                f"clean, the fault is in the derivation between them."
            )
