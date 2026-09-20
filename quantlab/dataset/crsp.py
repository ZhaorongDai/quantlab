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

**The arithmetic**, per PERMNO, sorted by date (D-08):

- `close = abs(dlyprc)`. Not `dlyclose`: that is null on bid/ask days, through
  the whole pre-1992 Nasdaq era and on delisting rows, while `dlyret` is
  computed from `dlyprc` -- so using `dlyclose` would put a return chain and a
  price series that disagree into one panel.
- `G_t = prod_{s<=t}(1 + dlyret_s)`, a NULL return contributing 1. CIZ returns
  span gaps back to `DlyPrevDt` (`DlyRetDurFlg`), so the next valid return
  already covers the missing day; filling a null with 0 would double-count it.
- the anchor `A` is the PERMNO's LAST row with a non-null close;
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

import polars as pl
import xarray as xr

from quantlab.base.config import CrspDatasetConfig, DatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset.crsp_reference import CrspReference
from quantlab.dataset.crsp_symbology import CrspSymbology
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns

#: Everything this panel carries BEYOND the Tiingo twelve, with its CRSP source
#: and the unit the panel states it in. FLOAT64 like every other variable
#: (RESEARCH Pitfall 7): a dense `[timestamp, symbol]` panel is a cartesian
#: product, so a symbol that did not exist yet needs a NaN to say so -- which
#: an integer `permno` or a boolean flag has no room for.
#:
#: - ``permno``/``permco``  -- CRSP's security and company ids, the stable
#:   identity behind a ticker column that renames and gets reused.
#: - ``ret``                -- ``dlyret``, the daily TOTAL return (dividends
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
    "permno",
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
            config.permnos = permnos
        # Invalidated on every config assignment: the derivation is scoped to
        # `[start_date, end_date]`, so a re-dated config must not reuse the
        # previous window's anchor.
        self._derivation_cache: pl.DataFrame | None = None
        self._symbology: CrspSymbology | None = None

    # -- the global derivation ---------------------------------------------

    def _derivation(self) -> pl.DataFrame:
        """The labelled, adjusted frame for the WHOLE configured window.

        Computed once per instance and cached. Every window densifier slices
        this, so every chunk of one conversion shares one anchor per PERMNO.
        """
        cached = getattr(self, "_derivation_cache", None)
        if cached is not None:
            return cached

        frame = self._scan_raw()
        if self.config.permnos:
            frame = frame.filter(
                pl.col("symbol").is_in(list(self.config.permnos))
            )
        frame = frame.collect()

        reference = CrspReference(self.config.reference_dir)
        self._symbology = CrspSymbology(
            reference.table("stksecurityinfohist"),
            self.config.symbol_overrides,
        )
        # `symbol` on the raw frame is the PERMNO string; the panel's `symbol`
        # is the ticker, so the raw one is dropped rather than overwritten --
        # `permno` (Int64) already carries the identity.
        frame = frame.drop("symbol")
        frame = self._symbology.label_rows(frame)

        frame = frame.sort(["permno", "timestamp"])
        derived = frame.with_columns(
            pl.col("dlyprc").abs().alias("close"),
            (1.0 + pl.col("dlyret").fill_null(0.0))
            .cum_prod()
            .over("permno")
            .alias("_G"),
        )
        # The anchor is the LAST row with a non-null close, not simply the last
        # row: a PERMNO whose final row has no price would otherwise anchor the
        # entire series on a null and make every adjusted value NaN.
        anchor = (
            derived.filter(pl.col("close").is_not_null())
            .group_by("permno")
            .agg(
                pl.col("close").last().alias("_close_anchor"),
                pl.col("_G").last().alias("_G_anchor"),
                pl.col("dlycumfacshr").last().alias("_cumfacshr_anchor"),
            )
        )
        derived = derived.join(anchor, on="permno", how="left")

        derived = derived.with_columns(
            (pl.col("_close_anchor") * pl.col("_G") / pl.col("_G_anchor"))
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
        return self._finalise(derived)

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
            (
                pl.col("dlyorddivamt").fill_null(0.0)
                + pl.col("dlynonorddivamt").fill_null(0.0)
            ).alias("divCash"),
            # 1.0 on the PERMNO's first row: there is no previous factor to
            # divide by, and an ordinary day's split factor IS 1.0.
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

    # -- axes and windows ---------------------------------------------------

    def _raw_axes_in_range(self):
        """Both axes of the CONVERTED panel, from the cached derivation.

        Overridden rather than inherited because `StockDataset`'s version
        reads the raw `symbol` column, which here is the PERMNO. The axis this
        store is pinned to is the TICKER axis, and it only exists after
        symbology has run.
        """
        import pandas as pd

        derivation = self._derivation()
        symbols = sorted(
            str(value)
            for value in derivation.get_column("symbol").unique().to_list()
        )
        if self.config.symbols:
            wanted = {str(symbol) for symbol in self.config.symbols}
            symbols = [symbol for symbol in symbols if symbol in wanted]
        timestamps = derivation.get_column("timestamp").unique().to_list()
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[str] | None = None
    ) -> xr.Dataset:
        """Densify ONE window of the cached derivation."""
        start = self._as_datetime(start_date)
        end = self._as_datetime(end_date)
        window = self._derivation().filter(
            (pl.col("timestamp") >= pl.lit(start))
            & (pl.col("timestamp") <= pl.lit(end))
        )
        if symbols is not None:
            window = window.filter(
                pl.col("symbol").is_in([str(symbol) for symbol in symbols])
            )
        elif self.config.symbols:
            window = window.filter(
                pl.col("symbol").is_in(
                    [str(symbol) for symbol in self.config.symbols]
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
        return self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )

    def _assert_unique_panel_keys(self, window: pl.DataFrame) -> None:
        """`(timestamp, symbol)` is unique, or the conversion fails.

        Two PERMNOs that resolve to ONE symbol on overlapping dates is a real
        CRSP situation (ticker reuse, an unsuffixed share class), and the
        inherited `dedup_raw_frame(keep="last")` would collapse them
        arbitrarily into one price series. Refusing here keeps that from
        happening silently; plan 05 turns the refusal into resolution RULES,
        which is a decision a phase makes rather than a dedup makes.
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
                f"collision(s) in the window, first {sample}. Two PERMNOs "
                f"resolved to one symbol on one day; refusing rather than "
                f"collapsing two securities' prices into one series. Use "
                f"config.symbol_overrides to separate them."
            )
