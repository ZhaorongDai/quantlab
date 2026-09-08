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
`dataset/cleaning.py` is the existing precedent for a concrete collaborator
sitting beside the concrete datasets.
"""

from typing import TYPE_CHECKING, Optional

import pandas as pd
import xarray as xr
from loguru import logger

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

    def __init__(self, market: xr.Dataset, membership: xr.Dataset) -> None:
        if "is_member" not in membership.data_vars:
            raise ValueError(
                f"UniverseMask: the membership panel must carry an "
                f"'is_member' variable, got "
                f"{sorted(membership.data_vars)}. Passing the two panels in "
                f"the wrong order is the usual cause."
            )
        self.market = market
        self.membership = membership

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
        """
        return cls(
            market_dataset.read().get_xarray_dataset(),
            constituent_dataset.read().get_xarray_dataset(),
        )

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """The overlapping timestamp axis: a plain inner join, sorted."""
        market = pd.DatetimeIndex(self.market["timestamp"].values)
        membership = pd.DatetimeIndex(self.membership["timestamp"].values)
        return market.intersection(membership).sort_values()

    @property
    def symbols(self) -> list[str]:
        """The intersected symbol axis, sorted."""
        market = {str(symbol) for symbol in self.market["symbol"].values}
        membership = {str(symbol) for symbol in self.membership["symbol"].values}
        return sorted(market & membership)

    @property
    def in_window_members(self) -> list[str]:
        """Every symbol True in `is_member` at one or more timestamps INSIDE
        the overlapping window.

        Restricting to the overlapping window before testing membership is
        what keeps the report meaningful: the panel deliberately carries
        all-False columns for symbols whose entire membership falls outside
        the window (that is its survivorship-bias guarantee), and those are
        not coverage gaps.
        """
        overlap = self.timestamps
        if len(overlap) == 0:
            return []
        member = self.membership["is_member"].sel(timestamp=overlap)
        ever = member.any(dim="timestamp")
        return sorted(
            str(symbol)
            for symbol, flag in zip(ever["symbol"].values, ever.values)
            if bool(flag)
        )

    @property
    def missing_members(self) -> list[str]:
        """In-window members the market panel does not carry at all."""
        market = {str(symbol) for symbol in self.market["symbol"].values}
        return sorted(set(self.in_window_members) - market)

    def report(self) -> dict:
        """Return AND log the coverage report (D-06).

        A `logger.warning` carrying the count and the COMPLETE sorted list
        when non-empty; a `logger.info` when empty. The list is never
        truncated and never sampled -- a truncated list is worse than none,
        because it looks like a complete answer.
        """
        members = self.in_window_members
        missing = self.missing_members
        report = {
            "in_window_members": len(members),
            "missing_count": len(missing),
            "missing_symbols": missing,
        }

        if missing:
            logger.warning(
                f"UniverseMask: {len(missing)} of {len(members)} in-window "
                f"index member(s) are absent from the market panel entirely "
                f"and are dropped by the alignment. Every dropped name is a "
                f"survivorship-bias hole, so the COMPLETE list follows: "
                f"{missing}"
            )
        else:
            logger.info(
                f"UniverseMask: 0 missing members -- the market panel covers "
                f"all {len(members)} in-window index member(s)."
            )
        return report

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
