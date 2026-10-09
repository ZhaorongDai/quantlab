"""A forward return kept only where the symbol is an index member at the signal bar.

``MemberReturn`` is ``Return`` computed on every symbol's prices (members or
not), then blanked where the symbol is not a member at t. Membership is
judged at t only: a stock that leaves the index within the horizon keeps its
return. Computing the return on members-only prices instead would drop those
returns, a survivorship bias.

The membership comes from a *members store*: a price panel of the index,
NaN outside membership (the Sharadar research's ``members.zarr``). A cell
is a member where its ``adjOpen`` is present.

Examples
--------
>>> label = MemberReturn(FactorConfig(
...     warmup_bars=7, dataset=dataset, mode="batch", data_columns=["adjOpen"],
...     kwargs={"n_forward_periods": 1, "members_store": "us3000/members.zarr"},
...     file_path="us3000/label/ret_1.zarr",
... ))
>>> label.get_config()["name"]
'quantlab.label.predefined.member_return.MemberReturn'
"""

from __future__ import annotations

import xarray as xr

from quantlab.label.predefined.fret import Return

__all__ = ["MemberReturn"]


class MemberReturn(Return):
    """``Return`` blanked where the symbol is not a member at t.

    Parameters
    ----------
    factor_config : FactorConfig
        As for ``Return``, with ``kwargs["members_store"]``: the members
        store (a panel NaN outside membership; a cell is a member where its
        ``adjOpen`` is present).

    Examples
    --------
    >>> label.read("2026-09-01", "2026-09-30")["ret_1"].notnull().sum().item() <= 3000 * 21
    True
    """

    def _members(self, panel: xr.Dataset) -> xr.DataArray:
        """Return the membership of ``panel``'s cells, False where the store has none."""
        member = xr.open_zarr(self.config.factor.config.kwargs["members_store"])["adjOpen"].notnull()
        return member.reindex(timestamp=panel["timestamp"], symbol=panel["symbol"], fill_value=False)

    def read(self, start, end) -> xr.Dataset:
        """Return ``Return.read`` kept where the symbol is a member at t."""
        panel = super().read(start, end)
        return panel.where(self._members(panel))

    def compute(self, start, end) -> xr.Dataset:
        """Return ``Return.compute`` kept where the symbol is a member at t."""
        panel = super().compute(start, end)
        return panel.where(self._members(panel))
