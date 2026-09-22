"""Index-membership masking of a market panel (260906-13w D-06/D-07).

Holds one class, `UniverseMask`, which aligns an `is_member` boolean panel
against a market-data panel and reports what the alignment discards.

**Why `dataset/` and not `base/`.** `base/` is reserved for abstract contracts
with subclass seams -- `BaseDataset`, `MarketDataset`,
`IndexConstituentDataset` -- each of which exists so a new dataset kind can
bind itself by implementing hooks. `UniverseMask` has no hooks and no
per-index variation: it is a concrete, hook-free composition of two dataset
kinds that already exist, and putting a hook-free concrete class in `base/`
would dilute exactly the rule that makes that package legible.
`quantlab/dataset/_support/cleaning.py` is the existing precedent for a concrete collaborator
sitting beside the concrete datasets.
"""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr
from loguru import logger

from quantlab.dataset.crsp.tickers import CrspTickerLookup
from quantlab.utils.symbol_axis import sort_symbol_axis

if TYPE_CHECKING:  # import-cycle-free type hints only
    from quantlab.base.constituent import IndexConstituentDataset
    from quantlab.base.data import MarketDataset


class UniverseMask:
    """Restrict a market panel to an index's point-in-time membership.

    Constructed from two `xr.Dataset` objects rather than from two Dataset
    instances, so it is testable offline with no store and no network;
    `from_datasets()` is the pipeline-facing constructor that reads both from
    their stores.

    **The two asymmetries a reader must not "fix".**

    1. *Symbols are intersected in both directions, but only ONE direction is
       reported.* An index member the market panel does not carry is a
       COVERAGE GAP: dropping it silently reintroduces survivorship bias
       through precisely the hardest-to-obtain names, the long-delisted ones,
       so `report()` names every one of them (D-06). A market symbol that is
       not an index member is merely out of universe -- not a gap in
       anything -- so it is dropped without a word.
    2. *Timestamps are a plain inner join and are deliberately NOT reported.*
       The membership panel is on a CALENDAR-day axis (see
       `base/constituent.py`'s class docstring: contiguous `freq="D"`, weekend
       and holiday rows carried forward) while market data is on trading days.
       The dropped rows are therefore expected by construction, and reporting
       them would bury the symbol report under thousands of routine lines.

    Nothing here fetches anything. Per D-07 the assumption is that the market
    vendor serves delisted history, and `report()` is the entire mechanism
    that makes that assumption FALSIFIABLE -- which is why an empty report is
    a result worth logging rather than a no-op. Designing a fallback data
    source is explicitly out of scope; if the report comes back non-empty,
    that is a finding to act on, not a hole to paper over here.
    """

    def __init__(
        self,
        market: xr.Dataset,
        membership: xr.Dataset,
        ticker_lookup: Optional[CrspTickerLookup] = None,
    ) -> None:
        """`ticker_lookup` only ever spells `report()`'s warning for a human.

        Optional because this class is deliberately constructible from two bare
        panels with no store behind them -- that is what makes it testable
        offline. `from_datasets` supplies one; a direct construction gets
        `None` and the report reads exactly as it did before 03.11-09.
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
        """Build from two persisted datasets, reading each from its store.

        The pipeline-facing constructor, keeping this usable as a component
        in a config-driven run rather than only in a test.

        This is also the ONE place a ticker lookup can be attached, because it
        is the only constructor that knows where the market panel LIVES -- and
        the sidecar is a sibling of the store, not a property of the panel in
        memory. A market store with no `.crsp_tickers.json` beside it (Tiingo,
        Alpaca, a CRSP store converted before 03.11-09) yields a lookup that
        falls back to the axis's own spelling, so nothing here branches on a
        vendor.
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
        """The overlapping timestamp axis: a plain inner join, sorted."""
        market = pd.DatetimeIndex(self.market["timestamp"].values)
        membership = pd.DatetimeIndex(self.membership["timestamp"].values)
        return market.intersection(membership).sort_values()

    @property
    def symbols(self) -> list:
        """The intersected symbol axis, sorted.

        Deliberately the same SHAPE as `timestamps` above: a plain index
        intersection that converts nothing. It used to build two sets with
        `str()` applied to every label instead, which on a CRSP pair -- both
        panels keyed by int64 PERMNO since 03.11-03/05 -- produced digit
        strings that
        `apply()`'s `.sel` could not find in an integer index
        (`KeyError: "not all values found in index 'symbol'"`, measured
        2026-09-20).

        The return type is intentionally unparameterised: the element type is
        whatever the two axes agree on -- `int` for a CRSP universe, `str` for
        a Wikipedia one -- and `list[str]` was a claim about only one of them.

        Order comes from `sort_symbol_axis`, the single implementation of the
        numeric-order contract. `Index.sort_values()` would already be numeric
        on an int64 axis, but routing both axis kinds through one source is
        what keeps a string axis's existing lexicographic order stated in the
        same place rather than implied by a different call.
        """
        market = pd.Index(self.market["symbol"].values)
        membership = pd.Index(self.membership["symbol"].values)
        return sort_symbol_axis(market.intersection(membership).tolist())

    @property
    def in_window_members(self) -> list:
        """Every symbol True in `is_member` at one or more timestamps INSIDE
        the overlapping window.

        Restricting to the overlapping window before testing membership is
        what keeps the report meaningful: the panel deliberately carries
        all-False columns for symbols whose entire membership falls outside
        the window (that is its survivorship-bias guarantee), and those are
        not coverage gaps.

        Labels come back in the membership axis's own dtype -- see `symbols`
        for why that matters and `missing_members` for why the two properties
        had to stop converting at the SAME time.
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

        **This is a set difference, so both sides must be spelled the same
        way.** When this property stringified the market axis while
        `in_window_members` returned integers -- or the reverse -- the
        difference was the WHOLE in-window membership: every index member
        reported as a survivorship-bias hole, by count and by name, in a
        report that is loud, complete and entirely wrong (T-03.11-16). The
        three conversions in this class therefore moved together in 03.11-05;
        repairing `symbols` alone would have turned a `KeyError` into that
        silent, plausible-looking answer.
        """
        market = set(self.market["symbol"].values.tolist())
        return sort_symbol_axis(set(self.in_window_members) - market)

    def report(self) -> dict:
        """Return AND log the coverage report (D-06).

        A `logger.warning` carrying the count and the COMPLETE sorted list
        when non-empty; a `logger.info` when empty.
        The list is never truncated and never sampled -- a truncated list is
        worse than none, because it looks like a complete answer.

        That promise survived the PERMNO migration unchanged (03.11-05), and
        the temptation it had to survive was real: a list of bare integers
        reads worse than a list of tickers, and shortening it is the obvious
        way to make the log tidy again. 03.11-09 pays that debt the right way
        round: readability is restored by mapping PERMNOs back to
        period-correct tickers for DISPLAY (`missing_labels`, and the warning),
        not by printing fewer of them. Both lists are the SAME length as
        `missing_symbols` and in the same order -- an entry that cannot be
        named keeps its digits rather than dropping out.

        `missing_symbols` keeps the axis's own labels, unchanged. It is the
        machine half of this report: a caller that wants to `.sel()` those
        symbols out of a panel needs the identity, and a name looked up as of
        one day is not an identity -- that is the whole reason the names went
        into a sidecar instead of a coord.

        The as-of day is the LAST overlapping timestamp: this report is about
        the window being aligned, so the newest spelling in that window is the
        one its reader is looking at. With no overlap at all there is nothing
        to name and the labels are the digits.
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
        """`symbols` spelled for a human -- one label per input, never fewer.

        Never raises and never shortens. With no lookup, no overlap, or an
        unreadable sidecar, every entry falls back to its own spelling, which
        is what this report printed before the sidecar existed.
        """
        if not symbols:
            return []
        overlap = self.timestamps
        if self.ticker_lookup is None or len(overlap) == 0:
            return [str(symbol) for symbol in symbols]
        return self.ticker_lookup.label(symbols, overlap[-1].date())

    def apply(self) -> xr.Dataset:
        """The market panel on the intersected axes, non-member cells NaN.

        Runs `report()` first, so masking can never be a quiet way to skip
        the coverage report.

        Every data variable is masked uniformly, boolean flags included --
        so a panel carrying `anomaly_flag` gets float64 with NaN outside
        membership rather than a surviving `False`. That is deliberate:
        outside the universe a flag is UNDEFINED, not False, and keeping
        False would assert something this mask does not know.
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
