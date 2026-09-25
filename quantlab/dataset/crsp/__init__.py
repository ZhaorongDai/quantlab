"""CRSP Stock v2 daily panel on a PERMNO axis.

The acquisition tier stores CRSP's ``dsf_v2`` rows verbatim, one shard per
PERMNO. ``CrspStockDataset`` turns that raw tier into the same dense
``(timestamp, symbol)`` Zarr panel ``StockDataset`` produces for Tiingo: the
twelve Tiingo variables plus the CRSP extras in ``CRSP_EXTRA_VARIABLES``, so a
factor, label or backtester reads a CRSP store without knowing the vendor.
The ``symbol`` coordinate is the integer PERMNO; period-correct tickers are
written to a sidecar file next to the store and read back by
``quantlab.dataset.crsp.tickers``.

Per PERMNO, sorted by date, the conversion derives:

- ``close = abs(dlyprc)``, except that a delisting-amount row (``dlyprcflg``
  in ``_NO_PRICE_FLAGS``, or ``dlyprc == 0``) carries no price and is NaN.
- ``adjClose``: the PERMNO's first usable close inside the window, compounded
  by ``prod(1 + dlyret)`` relative to that anchor row. A null return
  contributes a factor of 1, because a CRSP return already spans any gap back
  to the previous valid price. ``adjOpen``/``adjHigh``/``adjLow`` are scaled
  by ``adjClose / close``, and ``adjVolume`` by ``dlycumfacshr`` relative to
  the anchor row.
- ``splitFactor = dlycumfacpr[t-1] / dlycumfacpr[t]`` (1.0 on the first row)
  and ``divCash = dlyorddivamt + dlynonorddivamt`` on the ex-date.

The adjustment is computed once over the whole configured window before any
date slicing, so a chunked and an unchunked conversion produce the same store.
``ret`` keeps a missing CRSP return as NaN rather than 0. The delisting return
is compounded exactly once: CRSP already puts it on its own daily row, so
nothing adds ``stkdelists.delret`` on top. A security filter keeps only the
security types asked for and writes a report of what it dropped beside the
store. See ``docs/wrds_crsp.md``.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import xarray as xr
from loguru import logger

from quantlab.base.config import CrspDatasetConfig, DatasetConfig
from quantlab.base.data import BaseDataset
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.dataset.crsp.symbology import CrspSymbology
from quantlab.dataset.stock import StockDataset
from quantlab.enums.data import TiingoColumns
from quantlab.utils.atomic import write_json_atomically
from quantlab.utils.symbol_axis import sort_symbol_axis

#: Variables the panel carries beyond the twelve Tiingo ones, with their CRSP
#: source. Every one is float64, because a dense panel needs NaN to say that a
#: symbol did not exist yet. There is no ``permno`` variable: the ``symbol``
#: coordinate already is the PERMNO.
#:
#: ``permco`` is CRSP's company id (one company can have several securities).
#: ``ret`` and ``retx`` are ``dlyret``/``dlyretx``, the daily total return
#: with and without dividends, NaN where CRSP has none. ``shrout`` and
#: ``market_cap`` are ``shrout``/``dlycap`` times 1000, so the panel states
#: shares and USD rather than thousands. ``bid``/``ask`` are
#: ``dlybid``/``dlyask``. ``prc_is_bidask`` is 1.0 when ``dlyprcflg == "BA"``
#: (the price is a bid/ask midpoint, so there was no trade), 0.0 otherwise
#: and NaN when the flag is null; ``is_delisting`` encodes ``dlydelflg == "Y"``
#: the same way. ``numtrd`` is ``dlynumtrd``; ``cumfacpr``/``cumfacshr`` are
#: CRSP's cumulative price and share factors; ``facprc`` is ``dlyfacprc``, the
#: day's own price factor (1.0 on an ordinary day, 4.0 on a 4:1 split);
#: ``close_trade`` is ``dlyclose``, the closing-trade price, which is null on
#: bid/ask days and on delisting rows and is therefore not the panel's
#: ``close``.
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

#: The ``dsf_v2`` columns a security filter may name. Each is a per-day type
#: column, which is what makes a per-date verdict possible. The list is
#: closed: ``ticker``, ``permno`` and the price columns are not filters, and
#: restricting the roster is ``config.permnos``'s job. The last four
#: (``primaryexch``, ``conditionaltype``, ``tradingstatusflg``,
#: ``exchangetier``) appear in no preset because they change over a
#: security's life, so filtering on them punches holes in a series and can
#: drop the delisting row.
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

#: The named security filters. Each value is ``{column: allowed values}``;
#: every listed column must match, and a null value never matches.
#:
#: ``equity_common`` keeps common stock, including REITs and non-US
#: incorporated issuers, and drops ADRs, units, funds and ETFs. Its
#: ``sharetype`` allow-list is what excludes ADRs (``AD``) and units (``UG``),
#: which otherwise satisfy ``securitytype='EQTY'`` and
#: ``securitysubtype='COM'``. ``shrcd_10_11`` replicates the legacy
#: ``shrcd in (10, 11)`` screen; it is not the default because it drops REITs
#: and non-US issuers that are legitimate index members. ``none`` keeps every
#: security.
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

#: Suffix of the filter report written beside the store: what the security
#: filter removed, by type combination and by PERMNO.
FILTER_REPORT_SUFFIX: str = ".crsp_filter_report.json"

#: Suffix of the ticker sidecar written beside the store: a table of
#: ``{PERMNO: [{ticker, start, end}, ...]}`` intervals, so a display layer can
#: spell the integer axis for a human as of a date. Intervals rather than one
#: name per PERMNO, because a renamed company (FB, then META) keeps its
#: PERMNO. Read by ``quantlab.dataset.crsp.tickers.CrspTickerLookup``.
TICKER_SIDECAR_SUFFIX: str = ".crsp_tickers.json"

#: The ``dlydelflg`` value marking the row that carries the delisting return.
_DELISTING_FLAG = "Y"

#: ``dlyprcflg`` values whose row carries no market price. CRSP writes two
#: delisting shapes: ``DP`` (delisting price) is a real price and stays one,
#: while ``DA`` (delisting amount) is a settlement amount, written with
#: ``dlyprc = 0.0`` as a sentinel and null ``dlyclose``, ``dlyvol``,
#: ``dlycumfacpr`` and ``dlycumfacshr``. Reading that 0.0 as a close would
#: publish a trade at $0.00 and, because 0.0 is not null, let the row become
#: the adjustment anchor and zero the security's whole adjusted history.
_NO_PRICE_FLAGS: tuple[str, ...] = ("DA",)

#: The order in which the filter report spells a type combination.
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
    """Resolve a ``security_filter`` value to ``{column: allowed values}``.

    A preset name resolves to its entry in ``SECURITY_FILTER_PRESETS``; a
    mapping is validated against ``FILTERABLE_COLUMNS`` and returned with its
    allow-lists frozen into tuples of strings. An unknown preset raises rather
    than falling back to the default or to "keep everything", because a
    silently different panel is indistinguishable from a correct one.

    Parameters
    ----------
    value : str | dict
        A preset name or a ``{column: allowed values}`` mapping.
    owner : str
        Class name used as the prefix of every error message.

    Returns
    -------
    dict[str, tuple[str, ...]]
        The resolved mapping; empty for the ``"none"`` preset.

    Raises
    ------
    ValueError
        If the preset name is unknown, ``value`` is neither a
        string nor a dict, a column is not filterable, an allow-list is a
        bare string, or an allow-list is empty.

    Examples
    --------
    >>> resolve_security_filter("none")
    {}
    >>> resolve_security_filter("equity_common")["sharetype"]
    ('NS', 'SB', 'CE')
    >>> resolve_security_filter({"securitytype": ["EQTY"], "primaryexch": ["N"]})
    {'securitytype': ('EQTY',), 'primaryexch': ('N',)}
    >>> resolve_security_filter("common")
    Traceback (most recent call last):
    ValueError: CrspStockDataset: security_filter 'common' is not a preset. ...
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
                f"use config.permnos -- the PERMNO is this panel's identity "
                f"(D-01), and there is no ticker-side roster on this vendor."
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
    """Dense daily panel built from the CRSP raw tier, keyed by PERMNO.

    The class reuses ``StockDataset``'s raw-tree scan (vendor root, ``month=``
    hive layout, single-vendor provenance check, window predicates and
    ``has_raw_data``) and overrides the axes and the window densifier, because
    both must run against the adjusted derivation rather than the raw frame:
    the raw tier's ``symbol`` column is the PERMNO as a string, and every
    derived variable is computed here.

    The config must be a ``CrspDatasetConfig`` with ``frequency="1d"`` and
    ``vendor="wrds"``. The inherited ticker-side ``symbols`` field is refused;
    ``permnos`` restricts the conversion instead. Each conversion also writes
    two sidecars beside the store, the filter report and the ticker table (see
    ``filter_report_path`` and ``ticker_sidecar_path``).

    Examples
    --------
    Build the store from a raw tier already pulled through WRDS (the paths
    follow the layout the pull writes under the data root), then read it
    back as a panel whose ``symbol`` axis is the integer PERMNO:

    >>> config = CrspDatasetConfig(
    ...     zarr_file_path="/data/crsp.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/wrds_crsp/wrds",
    ...     reference_dir="/data/downloads/us_equity/1d/wrds_crsp/_reference",
    ...     start_date="2008-01-01",
    ...     end_date="2020-12-31",
    ... )
    >>> ds = CrspStockDataset(config)
    >>> ds.from_raw_data().save()
    >>> panel = CrspStockDataset(config).read().get_xarray_dataset()
    >>> panel.symbol.values.tolist()
    [14593, 80599]
    >>> panel["ret"].sel(symbol=80599).to_pandas().dropna().tail(2)
    timestamp
    2008-09-17   -0.566667
    2008-09-18   -0.600000
    Name: ret, dtype: float64
    """

    #: The config class the module loader rebuilds this dataset with.
    config_cls = CrspDatasetConfig

    #: Factor-config fields refused over this panel, read by the factor base
    #: class. ``BaseFactorConfig.symbols`` is a different field from the
    #: dataset's ``symbols``, but it too selects on a ticker and would raise a
    #: ``KeyError`` from ``.sel`` against the integer PERMNO axis mid-run.
    #: Declared here so the factor layer never learns this class's name.
    REJECTED_FACTOR_CONFIG_FIELDS: dict[str, str] = {
        "symbols": (
            "This panel's symbol axis is the int64 PERMNO (D-01), while that "
            "field is the base-class TICKER-side roster. Restrict the "
            "CONVERSION with config.permnos on the dataset instead -- it is "
            "the raw-side PERMNO roster, and it doubles as the explicit "
            "roster that overrides config.security_filter. A period-correct "
            "ticker for a PERMNO is READ from the ticker sidecar; it is not "
            "something this panel selects on."
        )
    }

    #: The twelve variables a Tiingo daily panel carries, in
    #: ``TiingoColumns.EOD`` order, derived from that constant so the drop-in
    #: promise is checked against its source.
    TIINGO_VARIABLES: tuple[str, ...] = tuple(TiingoColumns.EOD.split(","))

    #: Variables added beyond the Tiingo twelve. Bound here so a subclass can
    #: narrow or extend the set without touching the module constant.
    EXTRA_VARIABLES: tuple[str, ...] = CRSP_EXTRA_VARIABLES

    @BaseDataset.config.setter
    def config(self, config: DatasetConfig):
        """Assign the config, validating the CRSP-specific fields.

        On top of the base-class date normalisation this refuses anything
        that is not a ``CrspDatasetConfig``, a ``frequency`` other than
        ``"1d"``, a ``vendor`` other than ``"wrds"``, any value of the
        ticker-side ``symbols`` field, a ``permnos`` entry that is not a digit
        string, an empty ``permnos`` tuple, an unknown ``security_filter`` and
        an unknown ``roster_universe``. ``permnos`` is normalised to a tuple of
        strings and a dict ``security_filter`` to tuples of strings, so a
        config round-trips through JSON unchanged. Every cache derived from
        the previous config is cleared.

        Validation happens here rather than at first use so that a wrong
        field is reported by name before any raw data has been scanned.

        Raises
        ------
        TypeError
            If ``config`` is not a ``CrspDatasetConfig``.
        ValueError
            For any of the field refusals listed above.

        Examples
        --------
        >>> from dataclasses import replace
        >>> ds.config = replace(config, permnos=(14593,))
        >>> ds.config.permnos
        ('14593',)
        >>> ds.config = replace(config, symbols=("AAPL",))
        Traceback (most recent call last):
        ValueError: CrspStockDataset: config.symbols is not selectable ...
        """
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
        # `symbols` is the base-class ticker-side roster and this panel has no
        # ticker axis for it to select on. It is refused at assignment,
        # whatever it holds (an empty tuple too), so the error names the wrong
        # field instead of surfacing later as a `KeyError` from a `.sel`
        # against an integer index.
        if config.symbols is not None:
            rejected = config.symbols
            raise ValueError(
                f"{self.class_name}: config.symbols is not selectable on a "
                f"CRSP panel; got {rejected!r}. This panel's symbol axis is "
                f"the int64 PERMNO (D-01), while that field is the base-class "
                f"TICKER-side roster -- two names for one axis, which disagree "
                f"the moment a ticker is reused or renamed. Use config.permnos "
                f"instead: it is the raw-side PERMNO roster, and it doubles as "
                f"the explicit roster that overrides config.security_filter. A "
                f"period-correct ticker for a PERMNO is READ from the ticker "
                f"sidecar; it is not something this panel selects on."
            )
        if config.permnos is not None:
            permnos = tuple(str(permno) for permno in config.permnos)
            bad = [permno for permno in permnos if not permno.isdigit()]
            if bad:
                raise ValueError(
                    f"{self.class_name}: config.permnos must hold PERMNO digit "
                    f"strings; {bad} are not. permnos selects the RAW tier, "
                    f"which keys on the PERMNO -- there is no ticker-side "
                    f"roster on this vendor to put a ticker in instead."
                )
            # An empty tuple could mean "no security" or, to the roster gate
            # that reads it, "every PERMNO in the raw tier". Refuse it by name
            # rather than pick one.
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

        # Validated at assignment: a malformed filter is a config error and
        # should not surface only after the raw tier has been scanned.
        # `_security_filter` holds the resolved mapping; `config.security_filter`
        # keeps what the user wrote (a preset name stays a name) so the config
        # round-trips, with a dict normalised to tuples in place.
        self._security_filter = resolve_security_filter(
            config.security_filter, owner=self.class_name
        )
        if isinstance(config.security_filter, dict):
            config.security_filter = dict(self._security_filter)
        if config.roster_universe is not None:
            from quantlab.dataset.crsp.membership import CrspMembership

            if config.roster_universe not in CrspMembership.INDEXES:
                raise ValueError(
                    f"{self.class_name}: roster_universe "
                    f"{config.roster_universe!r} is not a CRSP universe; "
                    f"this vendor serves {CrspMembership.INDEXES}."
                )

        # Every cache below is derived from the configured window and roster,
        # so a reassigned config must not reuse any of it: the derivation and
        # its anchors, the symbology, the filter report, the ticker sidecar
        # payload and the membership spells of `roster_universe`.
        self._derivation_cache: pl.DataFrame | None = None
        self._symbology: CrspSymbology | None = None
        self._filter_report: dict | None = None
        self._ticker_intervals: dict | None = None
        self._member_intervals_cache: pl.DataFrame | None = None

    # -- the global derivation ---------------------------------------------

    def _derivation(self) -> pl.DataFrame:
        """Return the labelled, adjusted frame for the whole configured window.

        The raw tier is scanned across ``[start_date, end_date]`` with no date
        argument, the ``config.permnos`` roster is applied, the ``symbol``
        column is cast to the int64 PERMNO, and the adjustment anchor is
        chosen per PERMNO over that whole frame before
        ``_raw_data_to_xr_window`` slices any dates. That ordering is what
        makes every chunk granularity produce the same store: a window never
        sees an anchor of its own. The result is cached on the instance and
        cleared whenever the config is reassigned. Both conversion entry
        points pass through here, so anything that must happen exactly once
        per conversion belongs here.

        The anchor is each PERMNO's first row inside the window with a
        strictly positive ``close`` and a non-null ``dlycumfacshr``. A store
        that only grows forward therefore keeps every historical value, but
        two movements silently rescale a security's whole adjusted history:
        moving ``start_date`` later, and back-filling the raw tier with rows
        earlier than the previous anchor. Neither is detected; the remedy is
        a rebuild.

        Returns
        -------
        pl.DataFrame
            One row per raw ``(permno, timestamp)`` that survives the security
            filter, projected onto the panel's variables by ``_finalise``.

        Raises
        ------
        ValueError
            If a PERMNO has no usable anchor or its return chain
            is zero or non-finite at the anchor.
        """
        cached = getattr(self, "_derivation_cache", None)
        if cached is not None:
            return cached

        frame = self._scan_raw()
        # `is not None` rather than truthiness: the setter refuses an empty
        # tuple, but a config assigned by another route must not convert the
        # entire raw tier while claiming an empty roster.
        if self.config.permnos is not None:
            frame = frame.filter(
                pl.col("symbol").is_in(list(self.config.permnos))
            )
        frame = frame.collect()

        reference = CrspReference(self.config.reference_dir)
        self._symbology = CrspSymbology(
            reference.table("stksecurityinfohist")
        )
        # The raw tier's `symbol` column already holds the PERMNO as a string;
        # the panel's axis is that same value cast to int64.
        frame = frame.with_columns(pl.col("symbol").cast(pl.Int64))

        frame = frame.sort(["permno", "timestamp"])
        derived = frame.with_columns(
            # Exclude the no-price sentinel before `abs()`: `abs(0.0)` is still
            # 0.0, so a delisting-amount row would otherwise publish a $0.00
            # close. The flag test is case-insensitive and whitespace-stripped,
            # as in `_apply_security_filter`; the bare `== 0.0` arm catches a
            # sentinel written under a flag `_NO_PRICE_FLAGS` does not list.
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
        # The anchor row must carry both a positive close and a share factor,
        # so every quantity read off it comes from one row. A merely non-null
        # close would let the 0.0 sentinel anchor the series and zero it, and
        # a null `dlycumfacshr` on the anchor would make every `adjVolume` NaN.
        anchor = (
            derived.filter(
                pl.col("close").is_not_null()
                & (pl.col("close") > 0.0)
                & pl.col("dlycumfacshr").is_not_null()
            )
            .group_by("permno")
            .agg(
                pl.col("close").first().alias("_close_anchor"),
                pl.col("_G").first().alias("_G_anchor"),
                pl.col("dlycumfacshr").first().alias("_cumfacshr_anchor"),
            )
        )
        derived = derived.join(anchor, on="permno", how="left")
        self._assert_anchor_usable(derived)

        derived = derived.with_columns(
            # NaN wherever there is no close: the chain is defined on a
            # priceless day (a null return contributes 1), so without the mask
            # the anchor's level would be published as a price on a day that
            # had none.
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
        # Filter after the cumulative product, never before: dropping a row
        # first would make the next kept day's adjusted move span a return the
        # panel no longer shows.
        derived = self._apply_security_filter(derived)
        # Built from the kept rows, so the sidecar names exactly the PERMNOs
        # the panel carries.
        self._ticker_intervals = self._build_ticker_intervals(derived)
        return self._finalise(derived)

    def _assert_anchor_usable(self, derived: pl.DataFrame) -> None:
        """Refuse, by PERMNO, rather than publish an unusable adjusted column.

        Two causes get two messages because they need different remedies. A
        PERMNO with no row in the window carrying both a positive ``dlyprc``
        and a non-null ``dlycumfacshr`` has no anchor, so every ``adj*`` value
        would be NaN; the remedy is a different window or roster. A PERMNO
        whose cumulative return chain is 0.0 or non-finite at the anchor (a
        ``dlyret`` of exactly -1.0) would get an infinite series; the remedy
        is to inspect its returns. Nothing downstream raises on a zeroed,
        all-NaN or infinite column, and both cases hit exactly the delisted
        securities the panel exists to keep.

        Raises
        ------
        ValueError
            Naming the offending PERMNOs and the configured window.
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
                f"PERMNO's first row inside that window carrying BOTH a strictly "
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

    # -- the security filter ------------------------------------------------

    def _member_intervals(self) -> pl.DataFrame | None:
        """Return the membership spells of ``roster_universe``, read once.

        ``None`` when no universe is configured. The reference tier is read
        from disk, so the frame is memoised on the instance and cleared with
        the derivation cache when the config is reassigned. The window passed
        is the conversion's own, so a CRSP/Compustat link gap outside it
        cannot affect the result; a gap inside it refuses the conversion,
        since ``CrspDatasetConfig`` carries no ``allow_unlinked`` flag.
        """
        if self.config.roster_universe is None:
            return None
        cached = getattr(self, "_member_intervals_cache", None)
        if cached is None:
            from quantlab.dataset.crsp.membership import CrspMembership

            cached = CrspMembership(
                CrspReference(self.config.reference_dir)
            ).permno_intervals(
                self.config.roster_universe,
                window=(self.config.start_date, self.config.end_date),
            )
            self._member_intervals_cache = cached
        return cached

    def _roster_sources(self) -> list[str]:
        """Describe the explicit rosters in play, for the filter report.

        Computed separately from the exemption expression so that the
        ``"none"`` preset, which rejects nothing, still records that a roster
        was configured without paying for the per-row membership join.
        """
        sources: list[str] = []
        # `is not None`: an empty tuple that bypassed the setter reports a
        # roster of zero PERMNOs, which is what it is, rather than no roster.
        if self.config.permnos is not None:
            sources.append(
                f"config.permnos: {len(self.config.permnos)} PERMNO(s) named "
                f"explicitly, exempt on every date"
            )
        if self.config.roster_universe is not None:
            intervals = self._member_intervals()
            spells = 0 if intervals is None else intervals.height
            members = (
                0
                if intervals is None
                else intervals.get_column("permno").n_unique()
            )
            sources.append(
                f"config.roster_universe={self.config.roster_universe!r}: "
                f"{members} member PERMNO(s) over {spells} membership spell(s), "
                f"exempt on the dates inside a spell"
            )
        return sources

    def _roster_exemption(
        self, derived: pl.DataFrame
    ) -> tuple[pl.Expr, list[str]]:
        """Return the rows an explicit roster exempts from the type filter.

        The security filter screens an unspecified population and must not
        overrule a roster the caller named. Every date of a PERMNO in
        ``config.permnos`` is exempt; a member of ``config.roster_universe``
        is exempt on the dates inside its membership spell and on no others,
        which keeps the exemption from widening into a blanket one. With
        neither configured the expression is a literal false.

        The exemption matters because ``equity_common`` rejects
        ``sharetype='UG'``, the publicly traded partnership era of securities
        that were real index members, and that loss lands mid-history where
        it is indistinguishable from a late listing.

        The universe arm is evaluated here as a join and returned as
        ``pl.lit(series)`` aligned to the row order of ``derived``, so the
        caller must apply the expression to that same frame. One predicate
        per spell, or a materialised ``(permno, date)`` set, would both cost
        far more on a real index history.

        Returns
        -------
        tuple[pl.Expr, list[str]]
            ``(expression, sources)``, with ``sources`` from
            ``_roster_sources``.
        """
        sources = self._roster_sources()
        terms: list[pl.Expr] = []

        # `is not None`, as everywhere this field is read. An empty roster
        # contributes a term that matches nothing, never a blanket exemption.
        if self.config.permnos is not None:
            # `permno` is Int64; the config holds the digit strings the CLI
            # and the raw tier use, so the cast is the whole comparison.
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
        """Return, per row of ``derived``, whether it falls inside a spell.

        Both ends are inclusive, matching ``permno_intervals``. A PERMNO with
        no spell at all reads False rather than null: "not a member" and "no
        membership data" are the same answer to the filter's question.
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
        """Drop the rows the configured filter rejects and build the report.

        The verdict is per row, read off ``dsf_v2``'s own per-day type
        columns, so a security that stopped being common stock keeps exactly
        the era in which it was. Every listed column must match and a null
        never matches. Two adjustments follow. A delisting row inherits its
        PERMNO's previous verdict, because that row is where CRSP's type
        columns go blank and it carries the delisting return; judging it on
        blank types would drop the largest-magnitude day of every delisted
        security. Then the explicit-roster exemption is OR-ed in, after the
        carry, so a rescued ordinary row does not become the carried verdict
        of the next day.

        The report is built here and written once per conversion by
        ``_write_identity_reports``.

        Returns
        -------
        pl.DataFrame
            ``derived`` without the rejected rows and the working columns.
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

        # A row the delisting carry already kept is not a roster rescue.
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
        """Return the ``"sharetype/securitytype/.../usincflg"`` key expression.

        A null component renders as ``"None"`` rather than nulling the whole
        key, because which combination was dropped is exactly what a row with
        missing types needs answered.
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
        """Build the filter report payload.

        ``roster_overrides`` and ``admitted_without_ticker`` are always
        present, with zero counts when nothing happened, so a reader can tell
        "no override" from "this store predates the key". A rescued row counts
        in ``rows_kept`` and appears in neither ``dropped_by_type`` nor
        ``dropped_permnos``, so ``rows_kept + rows_dropped == rows_total``.
        The no-ticker warning is logged here because both branches of
        ``_apply_security_filter`` reach this method exactly once.

        Parameters
        ----------
        rows_total : int
            Row count of the derivation before filtering.
        dropped : pl.DataFrame
            The rows the filter removed.
        rescued : pl.DataFrame | None
            The rows an explicit roster kept that the filter would
            have removed.
        sources : list[str] | tuple[str, ...]
            The roster descriptions from ``_roster_sources``.
        kept : pl.DataFrame | None
            The surviving rows, used for the no-ticker count.
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
        """Count the admitted PERMNOs that have no ticker on any panel day.

        Returns ``{"permnos": [...], "rows": N}`` with the PERMNOs in numeric
        order. A PERMNO is listed only when none of its panel days falls
        inside a ``stksecurityinfohist`` interval that carries a ticker;
        partial coverage is not counted. This is a count, not a filter:
        nothing is excluded, because most never-ticker securities read as
        ordinary common stock and no type predicate separates them.
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
            # No named interval anywhere: every admitted PERMNO counts.
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
        # A backward as-of join alone attaches the last interval to every
        # later row; checking the end is what says the row is inside it.
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
        """Render a ``(permno, rows)`` frame as ``{"permnos": [...], "rows": N}``."""
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
        """Log a warning when securities with no ticker entered the panel."""
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
        """Return ``{PERMNO: {types, rows, first, last}}`` for a set of rows.

        One rendering serves both the dropped rows and the roster-rescued
        rows, since "which PERMNO, over which dates, on which type
        combination" is the same question in both directions. There is no
        ticker field: the key answers who and ``types`` answers why. A
        human-readable name, if ever wanted here, would come from
        ``self._symbology``, not from the ticker sidecar, which carries only
        the panel's own PERMNOs and so knows nothing about a dropped one.
        """
        typed = frame.with_columns(self._type_combination())
        per_permno = (
            typed.sort(["permno", "timestamp"])
            .group_by("permno")
            .agg(
                pl.col("_types").unique().sort().alias("types"),
                pl.len().alias("rows"),
                pl.col("timestamp").min().alias("first"),
                pl.col("timestamp").max().alias("last"),
            )
            .sort("permno")
        )
        return {
            str(record["permno"]): {
                "types": [str(value) for value in record["types"]],
                "rows": int(record["rows"]),
                "first": str(record["first"])[:10],
                "last": str(record["last"])[:10],
            }
            for record in per_permno.to_dicts()
        }

    @staticmethod
    def _jsonable_filter(value):
        """Return the configured filter as JSON data: a preset name, or lists."""
        if isinstance(value, dict):
            return {
                str(column): [str(item) for item in allowed]
                for column, allowed in value.items()
            }
        return value

    def filter_report_path(self) -> Path:
        """Return the path of the filter report written beside the store.

        Examples
        --------
        >>> ds.filter_report_path()
        PosixPath('/data/crsp.zarr.crsp_filter_report.json')
        """
        return Path(str(self.config.zarr_file_path) + FILTER_REPORT_SUFFIX)

    def ticker_sidecar_path(self) -> Path:
        """Return the path of the ticker sidecar written beside the store.

        Examples
        --------
        >>> ds.ticker_sidecar_path()
        PosixPath('/data/crsp.zarr.crsp_tickers.json')
        """
        return Path(str(self.config.zarr_file_path) + TICKER_SIDECAR_SUFFIX)

    def _build_ticker_intervals(self, derived: pl.DataFrame) -> dict:
        """Build the ticker sidecar payload for the PERMNOs ``derived`` keeps.

        The payload is ``{"generated_from", "vintage_product_end",
        "intervals"}``, where ``intervals`` maps ``str(permno)`` to that
        PERMNO's named spells in ascending ``start`` order, each
        ``{"ticker", "start", "end"}`` with both ends inclusive. Intervals
        rather than one name per PERMNO, because a renamed company keeps its
        PERMNO and a last-name-wins map would file its early years under the
        later name. Only the panel's own PERMNOs are written, not the whole
        reference table. ``vintage_product_end`` rides along because a newer
        CRSP vintage can carry a later interval for the same PERMNO.
        Intervals with no ticker are dropped: this sidecar answers "what is
        it called", and no name is the same answer whether the interval is
        absent or nameless.
        """
        empty: dict = {
            "generated_from": "stksecurityinfohist",
            "vintage_product_end": str(CrspReference(
                self.config.reference_dir
            ).product_end),
            "intervals": {},
        }
        if self._symbology is None or derived.is_empty():
            return empty

        panel_permnos = set(
            int(value) for value in derived.get_column("permno").unique().to_list()
        )
        intervals = (
            self._symbology.symbol_intervals()
            .drop_nulls("symbol")
            .filter(pl.col("permno").is_in(sorted(panel_permnos)))
            .sort(["permno", "start_date"])
        )

        payload: dict[str, list[dict]] = {}
        for record in intervals.to_dicts():
            payload.setdefault(str(int(record["permno"])), []).append(
                {
                    "ticker": str(record["symbol"]),
                    "start": str(record["start_date"])[:10],
                    "end": str(record["end_date"])[:10],
                }
            )
        empty["intervals"] = payload
        return empty

    def _finalise(self, derived: pl.DataFrame) -> pl.DataFrame:
        """Project the derivation onto the panel's variables and cache it.

        Renames the raw OHLCV columns, applies the adjustment factors, derives
        ``divCash`` and ``splitFactor``, renames or rescales the CRSP extras,
        casts every variable to float64 and stores the result in
        ``_derivation_cache``.
        """
        frame = derived.with_columns(
            pl.col("dlyopen").alias("open"),
            pl.col("dlyhigh").alias("high"),
            pl.col("dlylow").alias("low"),
            pl.col("dlyvol").alias("volume"),
            (pl.col("dlyopen") * pl.col("_factor")).alias("adjOpen"),
            (pl.col("dlyhigh") * pl.col("_factor")).alias("adjHigh"),
            (pl.col("dlylow") * pl.col("_factor")).alias("adjLow"),
            (pl.col("dlyvol") * pl.col("_volume_factor")).alias("adjVolume"),
            # Unadjusted cash per share on the ex-date, the Tiingo convention.
            # CRSP splits the day's cash into an ordinary and a non-ordinary
            # component, so they are summed. Both null is 0.0, not NaN: a day
            # with no distribution paid a known amount of nothing. Finer
            # detail (record and pay dates, distribution codes) stays raw in
            # `stkdistributions` under the reference tier.
            (
                pl.col("dlyorddivamt").fill_null(0.0)
                + pl.col("dlynonorddivamt").fill_null(0.0)
            ).alias("divCash"),
            # 1.0 on the PERMNO's first row inside the window, where there is
            # no previous `dlycumfacpr` to divide by. A split on the window's
            # opening day therefore reads 1.0; `facprc`, CRSP's own per-day
            # factor, sits beside it for that case.
            pl.coalesce(
                pl.col("_prev_cumfacpr") / pl.col("dlycumfacpr"),
                pl.lit(1.0),
            ).alias("splitFactor"),
            # The CRSP extras, in `CRSP_EXTRA_VARIABLES` order. `permco` is
            # already named and `close` came from the derivation.
            #
            # `ret` keeps CRSP's null as NaN. The only `fill_null(0.0)` on a
            # return is inside `_derivation`'s cumulative product, because a
            # gap-spanning return already covers the missing day.
            pl.col("dlyret").alias("ret"),
            pl.col("dlyretx").alias("retx"),
            # CRSP states both in thousands; the panel states shares and USD.
            (pl.col("shrout") * 1000).alias("shrout"),
            (pl.col("dlycap") * 1000.0).alias("market_cap"),
            pl.col("dlybid").alias("bid"),
            pl.col("dlyask").alias("ask"),
            # The no-trade indicator is read off the flag, not off the sign of
            # the price: CRSP Stock v2 carries no negative prices, so a sign
            # test would flag nothing. A null flag stays null.
            pl.when(pl.col("dlyprcflg").is_null())
            .then(None)
            .when(pl.col("dlyprcflg") == pl.lit("BA"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.0))
            .alias("prc_is_bidask"),
            # A marker, not a multiplier: the delisting return is already in
            # `ret`.
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
            # The closing-trade price, kept beside `close` so a caller can see
            # whether a trade happened and at what price.
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
        """Return the symbol and timestamp axes of the converted panel.

        Overrides ``StockDataset``'s version because this panel's axis is the
        int64 PERMNO, read off the cached derivation so that the ``permnos``
        roster and the security filter have already been applied. Both axes
        come from that one frame: a timestamp axis taken from a wider frame
        would make the chunked path plan windows over days the requested
        securities never traded, pin the chunk grid against an extent the
        store never reaches, and record completed windows for them. The
        symbol order is numeric, via ``sort_symbol_axis``, so a five-digit
        PERMNO does not sort before a four-digit one as strings would.

        This is also where the chunked path writes its sidecars:
        ``from_raw_data_chunked`` calls it once, after the derivation has
        succeeded and before the first append.

        Returns
        -------
        tuple[list[int], pandas.DatetimeIndex]
            ``(symbols, timestamps)``: a sorted list of ints and a
            ``pandas.DatetimeIndex``.
        """
        import pandas as pd

        derivation = self._derivation()
        symbols = sort_symbol_axis(
            int(value)
            for value in derivation.get_column("symbol").unique().to_list()
        )
        timestamps = derivation.get_column("timestamp").unique().to_list()

        # Written last, after the derivation has succeeded: a run that fails
        # in the derivation must not leave sidecars for a store that was never
        # created, and a store can never exist without them because the first
        # append happens after this returns.
        self._write_identity_reports()
        return symbols, pd.DatetimeIndex(sorted(timestamps))

    def _write_identity_reports(self) -> None:
        """Write the filter report and ticker sidecar, once, for a new store.

        Nothing is written when the store already exists. The reports are
        written before the chunk ledger and new-listing checks, any of which
        may still abort the run, and without the guard a refused
        re-conversion would replace the surviving store's report with numbers
        for a panel that was never written. The cost is that an append does
        not refresh the sidecars, so they describe the panel as first
        written; a rebuild (``quantlab.dataset.crsp.rebuild``) deletes them
        first. Both writes pass ``indent=2, sort_keys=True`` explicitly so the
        on-disk shape does not depend on the JSON writer's defaults.
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
        if self._ticker_intervals is not None:
            write_json_atomically(
                self.ticker_sidecar_path(),
                self._ticker_intervals,
                indent=2,
                sort_keys=True,
            )

    def _raw_data_to_xr_window(
        self, start_date, end_date, symbols: list[int] | None = None
    ) -> xr.Dataset:
        """Densify one window of the cached derivation.

        Parameters
        ----------
        start_date
            First timestamp of the window, inclusive.
        end_date
            Last timestamp of the window, inclusive.
        symbols : list[int] | None
            Integer PERMNOs to keep and reindex onto, or ``None`` for
            every PERMNO in the derivation. The base signature says
            ``list[str]`` because most vendors key on a ticker.

        Returns
        -------
        xr.Dataset
            A dataset on ``(timestamp, symbol)`` for that window.
        """
        start = self._as_datetime(start_date)
        end = self._as_datetime(end_date)
        window = self._derivation().filter(
            (pl.col("timestamp") >= pl.lit(start))
            & (pl.col("timestamp") <= pl.lit(end))
        )
        # Cast to int: the derivation's `symbol` column is Int64, and `is_in`
        # on strings would silently match nothing. The `permnos` roster is
        # already applied inside `_derivation()`, so this is the only
        # restriction here.
        if symbols is not None:
            window = window.filter(
                pl.col("symbol").is_in([int(symbol) for symbol in symbols])
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
        """Densify the whole window in one go and write the sidecars.

        ``BaseDataset.from_raw_data`` calls only this, so it is the
        non-chunked path's one chance to leave the same sidecars
        ``_raw_axes_in_range`` leaves on the chunked path. The write happens
        after the derivation has succeeded and before ``save`` creates the
        store; the writer skips an existing store, so this is a first write
        only.
        """
        window = self._raw_data_to_xr_window(
            self.config.start_date, self.config.end_date, symbols=None
        )
        self._write_identity_reports()
        return window

    def _assert_unique_panel_keys(self, window: pl.DataFrame) -> None:
        """Refuse a window in which ``(timestamp, symbol)`` is not unique.

        ``symbol`` is the PERMNO, so this pair is ``(permno, dlycaldt)``, which
        the acquisition already asserts unique on every raw page. A duplicate
        here means something between the raw tier and this point multiplied
        rows (a join that fanned out, a window read twice), not that two
        securities were confused. The inherited deduplication would collapse
        the duplicates into one series silently, which is why this raises.

        Raises
        ------
        ValueError
            Listing the first few colliding keys.
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
                f"wrds/crsp.py:818-840), so these rows were multiplied AFTER "
                f"acquisition, not confused between two securities. Refusing "
                f"rather than collapsing them into one series. Inspect the raw "
                f"parquet for these (permno, date) pairs; if the raw tier is "
                f"clean, the fault is in the derivation between them."
            )
