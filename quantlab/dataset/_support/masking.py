"""Point-in-time universe masking of a market panel.

Holds ``UniverseMask``, which aligns a boolean ``is_member`` panel from a
constituent dataset against a market-data panel, reports which index members
the market data does not cover, and returns the market panel with every
non-member cell set to NaN. It is a concrete composition of two dataset kinds
rather than an abstract contract, which is why it lives beside the datasets
instead of under ``quantlab/base``.
"""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.utils.symbol_axis import sort_symbol_axis

if TYPE_CHECKING:  # type hints only, so no import cycle at runtime
    from quantlab.base.constituent import IndexConstituentDataset
    from quantlab.base.data import MarketDataset


class UniverseMask:
    """Restrict a market panel to an index's point-in-time membership.

    The two panels are intersected on both axes. Symbols are handled
    asymmetrically on purpose: an index member the market panel lacks is a
    coverage gap that would quietly reintroduce survivorship bias (delisted
    names are the hardest to obtain), so ``report()`` names every one of
    them, whereas a market symbol outside the index is simply out of universe
    and is dropped silently. Timestamps are a plain inner join and are not
    reported, because the membership panel is on a calendar-day axis while
    market data is on trading days, so dropped rows are expected.

    Nothing here fetches data. The class is built from two ``xarray.Dataset``
    objects so it can be used without a store; ``from_datasets()`` is the
    constructor used in a pipeline run.

    Examples
    --------
    >>> mask = UniverseMask.from_datasets(market_dataset, constituent_dataset)
    >>> mask.report()["missing_count"]
    1
    >>> panel = mask.apply()  # non-member cells are NaN

    The method examples below use a market panel of ``AAA``, ``BBB`` and
    ``CCC`` over four business days and a membership panel over six
    calendar days in which ``AAA`` and ``DDD`` are members throughout.
    """

    def __init__(
        self,
        market: xr.Dataset,
        membership: xr.Dataset,
        ticker_lookup: Optional[CrspTickerLookup] = None,
    ) -> None:
        """Store the two panels and an optional ticker lookup.

        Parameters
        ----------
        market : xr.Dataset
            A market panel on ``(timestamp, symbol)``.
        membership : xr.Dataset
            A panel carrying a boolean ``is_member`` variable.
        ticker_lookup : Optional[CrspTickerLookup]
            Used only to spell ``report()``'s warning with
            human-readable tickers. ``None`` prints the axis labels as
            they are.

        Raises
        ------
        ValueError
            If ``membership`` has no ``is_member`` variable, which
            usually means the two panels were passed in the wrong order.
        """
        if "is_member" not in membership.data_vars:
            raise ValueError(
                f"UniverseMask: the membership panel must carry an "
                f"'is_member' variable, got "
                f"{sorted(membership.data_vars)}. Passing the two panels in "
                f"the wrong order is the usual cause."
            )
        self.market = market
        self.membership = membership
        self.ticker_lookup = ticker_lookup

    def __repr__(self) -> str:
        """Return a short summary with the overlapping axis lengths."""
        return (
            f"UniverseMask(timestamps={len(self.timestamps)}, "
            f"symbols={len(self.symbols)})"
        )

    @classmethod
    def from_datasets(
        cls,
        market_dataset: "MarketDataset",
        constituent_dataset: "IndexConstituentDataset",
    ) -> "UniverseMask":
        """Build a mask from two persisted datasets, reading each from its store.

        This is the only constructor that knows where the market store lives,
        so it is where a ticker sidecar beside that store is attached. A store
        without a sidecar yields a lookup that falls back to the axis's own
        labels, so nothing here depends on the vendor.

        Parameters
        ----------
        market_dataset : MarketDataset
            The market-data dataset to mask.
        constituent_dataset : IndexConstituentDataset
            The dataset providing ``is_member``.

        Returns
        -------
        UniverseMask
            A ``UniverseMask`` over the two panels read from disk.

        Examples
        --------
        >>> mask = UniverseMask.from_datasets(market_dataset, constituent_dataset)
        >>> mask
        UniverseMask(timestamps=4, symbols=2)
        >>> mask.missing_members
        ['DDD']
        """
        return cls(
            market_dataset.read().get_xarray_dataset(),
            constituent_dataset.read().get_xarray_dataset(),
            ticker_lookup=CrspTickerLookup.beside_store(
                market_dataset.config.zarr_file_path
            ),
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """The overlapping timestamp axis, a sorted inner join.

        Examples
        --------
        >>> len(mask.timestamps), mask.timestamps[0]
        (4, Timestamp('2024-01-01 00:00:00'))
        """
        market = pd.DatetimeIndex(self.market["timestamp"].values)
        membership = pd.DatetimeIndex(self.membership["timestamp"].values)
        return market.intersection(membership).sort_values()

    @property
    def symbols(self) -> list:
        """The intersected symbol axis, sorted.

        Labels are compared as they are, without conversion, and the element
        type is whatever the two axes share (integers for a PERMNO-keyed
        universe, strings for a ticker-keyed one). Ordering comes from
        ``sort_symbol_axis`` so both axis kinds follow one rule.

        Examples
        --------
        >>> mask.symbols
        ['AAA', 'BBB']
        """
        market = pd.Index(self.market["symbol"].values)
        membership = pd.Index(self.membership["symbol"].values)
        return sort_symbol_axis(market.intersection(membership).tolist())

    @property
    def in_window_members(self) -> list:
        """Symbols that are members at one or more overlapping timestamps.

        Membership is tested only inside the overlapping window: the panel
        carries all-False columns for symbols whose membership falls entirely
        outside it, and those are not coverage gaps. Labels keep the
        membership axis's own dtype.

        Examples
        --------
        >>> mask.in_window_members
        ['AAA', 'DDD']
        """
        overlap = self.timestamps
        if len(overlap) == 0:
            return []
        member = self.membership["is_member"].sel(timestamp=overlap)
        ever = member.any(dim="timestamp")
        return sort_symbol_axis(
            symbol
            for symbol, flag in zip(ever["symbol"].values.tolist(), ever.values)
            if bool(flag)
        )

    @property
    def missing_members(self) -> list:
        """In-window members the market panel does not carry at all.

        A set difference, so both sides must use the same label type; the
        properties feeding it deliberately convert nothing.

        Examples
        --------
        >>> mask.missing_members
        ['DDD']
        """
        market = set(self.market["symbol"].values.tolist())
        return sort_symbol_axis(set(self.in_window_members) - market)

    def report(self) -> dict:
        """Return and log the coverage report.

        The report is a dict with ``in_window_members`` (a count),
        ``missing_count``, ``missing_symbols`` (the axis's own labels, usable
        with ``.sel()``) and ``missing_labels`` (the same entries spelled for
        a human, same length and order). A non-empty list is logged as a
        warning in full; it is never truncated or sampled, because a
        shortened list looks like a complete answer. An empty list is logged
        at info level.

        Labels are looked up as of the last overlapping timestamp, the newest
        spelling in the window being aligned. With no overlap the labels are
        the raw axis values.

        Examples
        --------
        >>> report = mask.report()  # also logs the full missing list
        >>> report["missing_count"], report["missing_symbols"]
        (1, ['DDD'])
        """
        members = self.in_window_members
        missing = self.missing_members
        report = {
            "in_window_members": len(members),
            "missing_count": len(missing),
            "missing_symbols": missing,
            "missing_labels": self._label(missing),
        }

        if missing:
            logger.warning(
                f"UniverseMask: {len(missing)} of {len(members)} in-window "
                f"index member(s) are absent from the market panel entirely "
                f"and are dropped by the alignment. Every dropped name is a "
                f"survivorship-bias hole, so the COMPLETE list follows: "
                f"{report['missing_labels']}"
            )
        else:
            logger.info(
                f"UniverseMask: 0 missing members -- the market panel covers "
                f"all {len(members)} in-window index member(s)."
            )
        return report

    def _label(self, symbols: list) -> list[str]:
        """Spell ``symbols`` for a human, one label per input, never fewer.

        Never raises. Without a lookup, without any overlap, or with an
        unreadable sidecar, every entry falls back to ``str(symbol)``.
        """
        if not symbols:
            return []
        overlap = self.timestamps
        if self.ticker_lookup is None or len(overlap) == 0:
            return [str(symbol) for symbol in symbols]
        return self.ticker_lookup.label(symbols, overlap[-1].date())

    def apply(self) -> xr.Dataset:
        """Return the market panel on the intersected axes, non-members NaN.

        ``report()`` runs first, so masking can never skip the coverage
        report. Every data variable is masked uniformly, boolean flags
        included: outside the universe a flag is undefined rather than False,
        so it becomes NaN (and the variable float64) like everything else.

        Raises
        ------
        ValueError
            If the two panels share no timestamp or no symbol. An
            empty panel would flow into a backtest as "no positions"
            instead of surfacing the misconfiguration.

        Examples
        --------
        >>> masked = mask.apply()
        >>> masked["close"].values  # BBB is never a member, so NaN throughout
        array([[ 0., nan],
               [ 3., nan],
               [ 6., nan],
               [ 9., nan]])
        """
        self.report()

        overlap = self.timestamps
        symbols = self.symbols
        if len(overlap) == 0 or not symbols:
            raise ValueError(
                f"UniverseMask: the two panels do not overlap "
                f"({len(overlap)} shared timestamp(s), {len(symbols)} shared "
                f"symbol(s)), so there is nothing to mask. An empty panel is "
                f"never a useful answer -- it would flow silently into a "
                f"backtest as 'no positions' rather than as the "
                f"misconfiguration it is."
            )

        market = self.market.sel(timestamp=overlap, symbol=symbols)
        mask = self.membership["is_member"].sel(timestamp=overlap, symbol=symbols)
        return market.where(mask)
