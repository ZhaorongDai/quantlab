"""The configs of the dataset layer.

Every dataset is constructed from one of these and exposes it as ``self.config``.
They are frozen: a field cannot be assigned after creation, and a changed config
is a new one made with ``dataclasses.replace``. The dataset's config setter
normalises the config it is given into a new one, filling in derived values such
as open-ended dates and the ``name`` field (the owning class's dotted import path,
so the dataset can be rebuilt from the serialised dict; see
``quantlab.core.component``).

Several configs describe US-equity data from WRDS (Wharton Research Data Services,
a university data platform). CRSP (the Center for Research in Security Prices)
supplies daily stock data keyed by PERMNO, a permanent integer id that stays with a
security when its ticker changes. TAQ (Trade and Quote) supplies intraday quotes,
and the NBBO (National Best Bid and Offer) is the best bid and ask across all US
exchanges at each moment.

Fields are documented with ``#:`` comments so the meaning of each one sits beside
its definition.
"""

from dataclasses import dataclass

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig
from quantlab.enums.data import (
    BarInterval,
    Frequency,
    Market,
    ResampleFrequency,
    Vendor,
)


@dataclass(kw_only=True, frozen=True)
class BaseDatasetConfig(FrozenConfig):
    """Fields every dataset shares, whatever it holds.

    Both market panels and constituent (index membership) panels build on
    this class. Market-specific fields live on ``DatasetConfig``. The dates
    and symbols bound what the build path (``from_raw_data``,
    ``from_raw_data_chunked``, ``update``) converts from raw files; reading
    a range of the store is ``panel(start, end, symbols)``, which does not
    use them.

    Examples
    --------
    >>> cfg = BaseDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/stock.zarr",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     symbols=("AAPL", "MSFT"),
    ... )
    >>> cfg.kwargs, cfg.name
    (None, None)
    """

    #: Path of the Zarr store the dataset reads from and writes to.
    zarr_file_path: str
    #: First date to convert from raw files, inclusive. ``None`` means no
    #: lower bound.
    start_date: str | None = None
    #: Last date to convert from raw files, inclusive. ``None`` means no
    #: upper bound.
    end_date: str | None = None
    #: Convert only these symbols. ``None`` means every symbol.
    symbols: tuple | None = None
    #: Free-form options a specific dataset class may read (for example
    #: ``data_type`` for tick data). ``None`` is treated as empty.
    kwargs: dict | None = None
    #: Bar size the panel is resampled onto when it is requested or saved;
    #: ``None`` keeps the store's own bars. Set by ``resample()``.
    resample_freq: ResampleFrequency | None = None
    #: How each variable is aggregated into a resampled bar: one
    #: ``ResampleMethod`` for every variable, or a ``{variable: method}``
    #: dict naming every variable. Required when ``resample_freq`` is set.
    resample_how: dict[str, str] | str | None = None

    #: Dotted import path of the dataset class; filled by the config setter
    #: and used to rebuild the dataset from its serialised config.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class DatasetConfig(BaseDatasetConfig):
    """Config of a market data panel built from a raw download tree.

    ``market`` and ``frequency`` sit here, not on ``BaseDatasetConfig``,
    because a constituent panel has neither. Together with the ``data_type``
    entry of ``kwargs`` they form the key the vendor registry uses to pick the
    converter for this dataset.

    Examples
    --------
    >>> cfg = DatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/stock.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/tiingo",
    ...     market="us_equity",
    ...     frequency="1d",
    ...     vendor="tiingo",
    ...     start_date="2020-01-01",
    ...     symbols=("AAPL",),
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor
    ('us_equity', '1d', 'tiingo')
    """

    #: Root of the raw download tree the panel is converted from.
    raw_data_dir_path: str
    #: The market this panel belongs to.
    market: Market
    #: The acquisition frequency of the raw data.
    frequency: Frequency
    #: The vendor the raw data was downloaded from, when it matters for
    #: conversion.
    vendor: Vendor | None = None


@dataclass(kw_only=True, frozen=True)
class NbboDatasetConfig(DatasetConfig):
    """Config of the intraday bar panel built from WRDS TAQ NBBO quotes.

    ``frequency`` stays ``"tick"`` (the raw tier holds one row per NBBO
    record); the size of the bars the panel is resampled to is the separate
    ``bar_interval``. The four ``drop_*`` and ``keep_*`` fields are the
    resampler's record filter and are config fields, not ``kwargs``, so a
    rebuild from ``config.json`` reproduces the panel exactly.

    The panel's ``symbol`` axis is the integer PERMNO, the same axis as the
    CRSP panels, although the raw TAQ files are keyed by ticker. The
    conversion maps each raw ``(date, ticker)`` to its PERMNO through the
    CRSP symbology in ``reference_dir``, so the inherited ticker-side
    ``symbols`` field is refused by the dataset's config setter; use
    ``permnos`` instead. See ``docs/wrds_taq.md``.

    Examples
    --------
    >>> cfg = NbboDatasetConfig(
    ...     zarr_file_path="/data/us_equity/tick/nbbo_5m.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/tick/wrds",
    ...     reference_dir="/data/downloads/_reference",
    ...     bar_interval="5m",
    ...     start_date="2024-01-02",
    ...     end_date="2024-01-31",
    ...     permnos=("14593",),
    ... )
    >>> cfg.frequency, cfg.bar_interval, cfg.drop_crossed
    ('tick', '5m', True)
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always tick: the raw tier is one row per NBBO record.
    frequency: Frequency = "tick"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"
    #: Directory of the CRSP reference tables (``stksecurityinfohist`` and
    #: friends) the conversion reads its symbology from: the mapping from a
    #: ticker on a date to the PERMNO that traded under it. The same
    #: directory ``scripts/wrds/index.py`` fills for the CRSP panels.
    reference_dir: str
    #: Restrict the panel to these PERMNOs, as digit strings. ``None`` means
    #: every PERMNO the raw tier's tickers resolve to; an empty tuple is
    #: refused at config assignment because it could mean either "none" or
    #: "all". A listed PERMNO is on the axis even when the raw tier has no
    #: record for it, as an all-NaN column.
    permnos: tuple[str, ...] | None = None
    #: Bar size the tick records are resampled to.
    bar_interval: BarInterval = "1m"
    #: Start of the session window, US/Eastern wall clock ``HH:MM``.
    session_start: str = "09:30"
    #: End of the session window, US/Eastern wall clock ``HH:MM``.
    session_end: str = "16:00"
    #: Drop records whose bid exceeds the ask (both sides present).
    drop_crossed: bool = True
    #: Drop records whose bid equals the ask.
    drop_locked: bool = False
    #: Drop records carrying a non-positive price.
    drop_nonpositive_price: bool = True
    #: Keep only these ``qu_cond`` quote conditions. ``None`` keeps every
    #: condition.
    keep_qu_cond: tuple[str, ...] | None = None


#: PERMNO of the QQQ ETF. A module constant because it is also the value a user
#: passes to ``permnos`` when they want the ETF in some other window.
QQQ_PERMNO: str = "86755"

#: PERMNO of the SPY ETF (SPDR S&P 500 ETF Trust), the S&P 500 benchmark.
SPY_PERMNO: str = "84398"

#: Sharadar permaticker of SPY in the SFP table (TICKERS, 2026-10-05 pull),
#: the S&P 500 benchmark on the Sharadar axis.
SPY_PERMATICKER: int = 118691

#: PERMNO of the IWM ETF (iShares Russell 2000 ETF), the small-cap series of
#: the market-feature factor.
IWM_PERMNO: str = "88222"


@dataclass(kw_only=True, frozen=True)
class CrspDatasetConfig(DatasetConfig):
    """Config of the CRSP Stock v2 daily panel.

    The three market fields have defaults rather than being required: CRSP Stock v2
    is US equity, daily, and reached through the ``wrds`` account. They remain
    fields (not constants on the dataset) because the vendor registry resolves
    the converter from ``(market, frequency, data_type)`` read off this object.

    The panel's ``symbol`` axis is the integer PERMNO, so the inherited
    ticker-side ``symbols`` field is refused by the dataset's config setter;
    use ``permnos`` instead. See ``docs/wrds_crsp.md``.

    Examples
    --------
    >>> cfg = CrspDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/crsp.zarr",
    ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
    ...     reference_dir="/data/reference/crsp",
    ...     start_date="2015-01-01",
    ...     end_date="2024-12-31",
    ...     permnos=("14593", "10107"),
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor
    ('us_equity', '1d', 'wrds')
    >>> cfg.security_filter
    'equity_common'
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always daily for this vendor.
    frequency: Frequency = "1d"
    #: Always WRDS for this dataset.
    vendor: Vendor | None = "wrds"

    #: Directory of the CRSP reference tables (``stksecurityinfohist`` and
    #: friends) the conversion reads its symbology (the mapping from PERMNO to
    #: ticker over time) from. It is required rather than derived from
    #: ``raw_data_dir_path`` because the reference tables are downloaded by a
    #: separate step, and a missing download must fail with a clear error.
    reference_dir: str

    #: Restrict the conversion to these PERMNOs, as digit strings. ``None``
    #: means every PERMNO present in the raw tier; an empty tuple is refused at
    #: config assignment because it could mean either "none" or "all". A
    #: PERMNO listed here is an explicit roster: every one of its rows survives
    #: ``security_filter`` regardless of its share or security type, and the
    #: override is recorded under ``roster_overrides`` in the filter report
    #: written next to the store.
    permnos: tuple[str, ...] | None = None

    #: Which securities the panel holds: either the name of a preset
    #: (``"equity_common"``, ``"shrcd_10_11"``, ``"none"``) or an explicit
    #: ``{column: allowed values}`` mapping over the filterable CRSP type
    #: columns. The default keeps common stock, including REITs and non-US
    #: incorporated issuers, and drops ADRs, units, funds, ETFs and unknown
    #: types. The predicate is evaluated per date against the daily type
    #: columns, so a security that changed type keeps only the era in which it
    #: qualified. Filtering happens at conversion time, never in the raw tier,
    #: so a re-filter is a re-conversion rather than a re-download.
    security_filter: str | dict = "equity_common"

    #: Name of an index (one of ``CrspMembership.INDEXES``) whose point-in-time
    #: membership (who was in the index on each date, as known on that date)
    #: acts as an explicit roster for this conversion. During a
    #: PERMNO's membership spell it is exempt from ``security_filter``; outside
    #: its spells the filter applies normally. Overrides are recorded under
    #: ``roster_overrides`` in the filter report. ``None`` means no index roster.
    roster_universe: str | None = None

    @classmethod
    def etf_benchmark(
        cls,
        *,
        permno: str,
        zarr_file_path: str,
        raw_data_dir_path: str,
        reference_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "CrspDatasetConfig":
        """Return the config of a store holding only one benchmark ETF.

        The general form of ``qqq_benchmark``: ``permnos=(permno,)`` selects
        the ETF alone and ``security_filter="none"`` keeps it, because an ETF
        is a fund, which the default filter drops. Use ``SPY_PERMNO`` for the
        S&P 500 and ``QQQ_PERMNO`` for the Nasdaq-100.

        Parameters
        ----------
        permno : str
            The ETF's CRSP PERMNO.
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            Root of the CRSP raw download tree.
        reference_dir : str
            Directory of the CRSP reference tables.
        start_date : str | None
            First date to keep, inclusive.
        end_date : str | None
            Last date to keep, inclusive.

        Examples
        --------
        >>> cfg = CrspDatasetConfig.etf_benchmark(
        ...     permno=SPY_PERMNO,
        ...     zarr_file_path="/data/us_equity/1d/spy.zarr",
        ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        ...     reference_dir="/data/reference/crsp",
        ... )
        >>> cfg.permnos, cfg.security_filter
        (('84398',), 'none')
        """
        return cls(
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            reference_dir=reference_dir,
            start_date=start_date,
            end_date=end_date,
            permnos=(str(permno),),
            security_filter="none",
        )

    @classmethod
    def qqq_benchmark(
        cls,
        *,
        zarr_file_path: str,
        raw_data_dir_path: str,
        reference_dir: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "CrspDatasetConfig":
        """Return the config of a store holding only the QQQ ETF.

        The benchmark gets its own store rather than one more symbol in the
        equity panel: everything in the panel enters cross-sectional ranking
        and model training, and an ETF ranked against its own constituents is
        the index competing with itself. Two settings are fixed here because
        getting either wrong is silent: ``permnos=(QQQ_PERMNO,)`` selects the
        ETF alone, and ``security_filter="none"`` is required because QQQ is a
        fund, which the default filter drops.

        Pass a dataset over this store as ``BacktestConfig.benchmark_dataset``
        to compare a backtest against QQQ.

        Parameters
        ----------
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            Root of the CRSP raw download tree.
        reference_dir : str
            Directory of the CRSP reference tables.
        start_date : str | None
            First date to keep, inclusive.
        end_date : str | None
            Last date to keep, inclusive.

        Examples
        --------
        >>> cfg = CrspDatasetConfig.qqq_benchmark(
        ...     zarr_file_path="/data/us_equity/1d/qqq.zarr",
        ...     raw_data_dir_path="/data/downloads/us_equity/1d/crsp/wrds",
        ...     reference_dir="/data/reference/crsp",
        ...     start_date="2015-01-01",
        ... )
        >>> cfg.permnos, cfg.security_filter
        (('86755',), 'none')
        """
        return cls.etf_benchmark(
            permno=QQQ_PERMNO,
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            reference_dir=reference_dir,
            start_date=start_date,
            end_date=end_date,
        )


@dataclass(kw_only=True, frozen=True)
class SharadarDatasetConfig(DatasetConfig):
    """Config of a Sharadar daily price panel on the permaticker axis.

    Sharadar keys its price tables by ticker and renames a delisted company's
    history when its ticker is reused, so the panel's ``symbol`` axis is the
    *permaticker*, Sharadar's unchanging integer id of one share class (ADR
    0023). The inherited ticker-side ``symbols`` field is refused by the
    dataset; use ``permatickers`` instead.

    Examples
    --------
    >>> cfg = SharadarDatasetConfig(
    ...     zarr_file_path="/data/zarrs/sharadar_sep_1d.zarr",
    ...     raw_data_dir_path="/data/downloads/sharadar",
    ...     start_date="2015-01-01",
    ... )
    >>> cfg.market, cfg.frequency, cfg.vendor, cfg.table
    ('us_equity', '1d', 'sharadar', 'sep')
    """

    #: Always US equity for this vendor.
    market: Market = "us_equity"
    #: Always daily for this vendor.
    frequency: Frequency = "1d"
    #: Always Sharadar for this dataset.
    vendor: Vendor | None = "sharadar"

    #: Code of the price table the panel is built from (a key of
    #: ``quantlab.dataset.sharadar.tables.TABLES``); its raw parquet lives in
    #: ``<raw_data_dir_path>/<table>/``.
    table: str = "sep"

    #: Restrict the conversion to these permatickers, an explicit roster
    #: that ``category_filter`` never touches. An empty tuple is refused.
    permatickers: tuple[int, ...] | None = None

    #: An index whose members form an explicit roster: every permaticker that
    #: was a member at some point in ``[start_date, end_date]`` is converted,
    #: with all its bars and whatever its category. ``"sp500"`` reads the raw
    #: SP500 table. Combined with ``permatickers``, the roster is the union.
    roster_universe: str | None = None

    #: TICKERS ``category`` values an *unrostered* market universe keeps (one
    #: with neither ``permatickers`` nor ``roster_universe``). ``"default"``
    #: is the table's own default, which the dataset resolves: for SEP,
    #: domestic common stock, every share class of it (ADRs, Canadian filers,
    #: preferreds and anything else are dropped); for SFP, every fund
    #: category (``None``). ``None`` keeps every category. Ignored when a
    #: roster is set: a named security is never filtered out.
    category_filter: tuple[str, ...] | str | None = "default"

    @classmethod
    def etf_benchmark(
        cls,
        *,
        permaticker: int,
        zarr_file_path: str,
        raw_data_dir_path: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> "SharadarDatasetConfig":
        """Return the config of an SFP store holding only one benchmark fund.

        A backtest benchmark is a market dataset with exactly one symbol;
        ``table="sfp"`` and ``permatickers=(permaticker,)`` give that, and a
        named permaticker is never filtered by category.

        Parameters
        ----------
        permaticker : int
            The fund's permaticker, from the TICKERS rows of table ``SFP``.
        zarr_file_path : str
            Path of the benchmark's own Zarr store.
        raw_data_dir_path : str
            ``<download-dir>/sharadar``.
        start_date, end_date : str or None
            The window to convert, both inclusive.

        Examples
        --------
        >>> cfg = SharadarDatasetConfig.etf_benchmark(
        ...     permaticker=118691,
        ...     zarr_file_path="/data/zarrs/sharadar_spy_1d.zarr",
        ...     raw_data_dir_path="/data/downloads/sharadar",
        ... )
        >>> cfg.table, cfg.permatickers
        ('sfp', (118691,))
        """
        return cls(
            zarr_file_path=zarr_file_path,
            raw_data_dir_path=raw_data_dir_path,
            start_date=start_date,
            end_date=end_date,
            table="sfp",
            permatickers=(int(permaticker),),
        )


@dataclass(kw_only=True, frozen=True)
class ConstituentDatasetConfig(BaseDatasetConfig):
    """Config of an index-membership panel (a boolean mask over time and symbol).

    A cell is True when the symbol was a member of the index on that date.

    Examples
    --------
    >>> cfg = ConstituentDatasetConfig(
    ...     zarr_file_path="/data/us_equity/1d/sp500_constituent.zarr",
    ...     cache_dir="/data/reference/_cache",
    ...     start_date="2020-01-01",
    ...     end_date="2020-12-31",
    ...     as_of="2024-06-30",
    ... )
    >>> cfg.as_of
    '2024-06-30'
    """

    #: Directory holding the membership source the panel is built from (the
    #: universe catalog cache or the CRSP reference tables).
    cache_dir: str
    #: Date the membership is observed from. ``None`` means today, so a panel
    #: that must be reproducible later should pin it.
    as_of: str | None = None


@dataclass(kw_only=True, frozen=True)
class MergedDatasetConfig(FrozenConfig):
    """Config of a merged dataset: the datasets it merges, in order.

    A merged dataset holds no store of its own, so this config has no path,
    dates or symbols; each input keeps its own config. ``datasets`` is a
    component field, so ``to_dict()`` nests each input's config dict under it
    and ``MergedDataset.from_config`` rebuilds them.

    Examples
    --------
    With ``index`` and ``etf`` two datasets built earlier:

    >>> cfg = MergedDatasetConfig(datasets=[index, etf])
    >>> len(cfg.datasets)
    2
    >>> [d["zarr_file_path"] for d in cfg.to_dict()["datasets"]]
    ['/data/us_equity/1d/sp500.zarr', '/data/us_equity/1d/spy.zarr']
    """

    #: The datasets merged, in order. A list is stored as a tuple.
    datasets: tuple = component(many=True)
    #: Dotted import path of the dataset class; filled by the config setter.
    name: str | None = None


@dataclass(kw_only=True, frozen=True)
class FrameDatasetConfig(BaseDatasetConfig):
    """Config of a dataset held in memory, built from a caller's frame or panel.

    The panel itself is not part of the config: a ``FrameDataset`` is handed its data at
    construction, so no Zarr store is needed and ``zarr_file_path`` defaults to ``None``.
    A config that names a store is how the panel comes back from disk: a
    ``FrameDataset`` built from it reads that store into memory (the rebuild path of a
    saved backtest run, whose ``config.json`` names the store relative to the run
    directory). The dates and symbols are unused, as there is nothing to convert from
    raw files; the resample fields are set by ``resample()`` as on any dataset.

    Examples
    --------
    >>> cfg = FrameDatasetConfig()
    >>> cfg.zarr_file_path is None
    True
    >>> FrameDatasetConfig(zarr_file_path="copies/prices.zarr").zarr_file_path
    'copies/prices.zarr'
    """

    #: Path of a Zarr store holding the panel, read at construction; ``None`` for a
    #: panel handed over in memory.
    zarr_file_path: str | None = None
