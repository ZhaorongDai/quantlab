"""Several datasets merged into one panel, with their bad prints masked (#223).

A *bad print* is a vendor's wrong price for one bar: Sharadar SEP holds
closes of $0.01 between two of $7 and $9, each a one-bar return of about
-99.9% and then +10,000% that flows into every return-based statistic. It
can only be told from a real crash or jump by what is known at the bar,
since the price returning the next bar is not known live:
``quantlab.dataset._support.cleaning.bad_print_mask`` flags a move of more
than ``jump`` times on ordinary volume.

``BadPrintMaskedDataset`` is a ``MergedDataset`` whose panel has the price
variables of a flagged bar set to NaN, so the returns into and out of it are
missing. Give it to a consumer in place of the merged dataset (the factor
risk model and the Barra exposures read it in the Sharadar examples); the
stores are never changed, and a consumer given the plain dataset sees every
price.
"""

from typing import Self, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from quantlab.dataset._support.cleaning import bad_print_mask
from quantlab.dataset.base import BaseDataset, InsufficientHistoryError
from quantlab.dataset.config import BadPrintMaskedDatasetConfig
from quantlab.dataset.merged import MergedDataset
from quantlab.utils.date_range import check_range


class BadPrintMaskedDataset(MergedDataset):
    """A merged view of datasets whose bad prints have no price.

    Parameters
    ----------
    config : BadPrintMaskedDatasetConfig or sequence of BaseDataset
        The datasets to merge, in order, or a config holding them and the
        rule.
    **params
        With a sequence of datasets, the rule's fields of
        ``BadPrintMaskedDatasetConfig`` (``jump``, ``volume_ratio``,
        ``lookback``, ``price_variable``, ``volume_variable``,
        ``masked_variables``).

    Raises
    ------
    ValueError
        As ``MergedDataset``, or if ``jump`` is not above 1,
        ``volume_ratio`` not positive or ``lookback`` below 1.

    Examples
    --------
    ``sep`` holds Sharadar SEP prices and ``daily`` its market caps; CNYD
    (permaticker 120127) printed $0.01 on 2009-06-19, between $7.20 and
    $9.00:

    >>> prices = BadPrintMaskedDataset([sep, daily])
    >>> panel = prices.panel("2009-06-18", "2009-06-23", symbols=[120127])
    >>> panel["close"].values.ravel().tolist()
    [7.2, nan, nan, 9.0]
    >>> MergedDataset([sep, daily]).panel("2009-06-18", "2009-06-23", symbols=[120127])["close"].values.ravel().tolist()
    [7.2, 0.01, 9.0, 9.0]
    """

    # Narrower type annotation for readers and type checkers only.
    config: BadPrintMaskedDatasetConfig

    #: The config class used to rebuild this dataset from a saved config.
    config_cls = BadPrintMaskedDatasetConfig

    def __init__(
        self, config: "BadPrintMaskedDatasetConfig | Sequence[BaseDataset]", **params
    ):
        """Initialize the view; see the class docstring for parameters."""
        if not isinstance(config, BadPrintMaskedDatasetConfig):
            config = BadPrintMaskedDatasetConfig(datasets=tuple(config), **params)
        elif params:
            raise ValueError(
                f"{type(self).__name__}: give the rule's parameters in the config, not "
                f"beside it."
            )
        super().__init__(config)

    def _normalize_config(
        self, config: BadPrintMaskedDatasetConfig
    ) -> BadPrintMaskedDatasetConfig:
        """Check the inputs as ``MergedDataset`` does, then the rule's parameters."""
        merged = super()._normalize_config(config)
        # Raises for a parameter the rule refuses.
        bad_print_mask(
            np.zeros((1, 1)), np.zeros((1, 1)),
            jump=config.jump, volume_ratio=config.volume_ratio, lookback=config.lookback,
        )
        fields = {
            name: getattr(config, name)
            for name in BadPrintMaskedDatasetConfig.__dataclass_fields__
            if name not in ("datasets", "name")
        }
        fields["masked_variables"] = tuple(config.masked_variables)
        return BadPrintMaskedDatasetConfig(datasets=merged.datasets, name=merged.name, **fields)

    def panel(
        self,
        start,
        end,
        symbols: "Sequence | None" = None,
        variables: "Sequence[str] | None" = None,
    ) -> xr.Dataset:
        """Return the merged panel from ``start`` to ``end`` with its bad prints masked.

        The bars are read from ``lookback`` bars before ``start`` (or the
        first bar), so the flags of a bar do not depend on where the window
        starts. Of a flagged bar, every variable of ``masked_variables`` the
        panel holds is NaN; the others are the inputs'.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.
        symbols : sequence, optional
            Symbol labels to keep, in the order given; ``None`` keeps every one.
        variables : sequence of str, optional
            Shared variable names to keep, in the order given; ``None`` keeps
            every one. The rule's price and volume are read either way.

        Returns
        -------
        xr.Dataset
            The panel on ``(timestamp, symbol)``.

        Raises
        ------
        ValueError, KeyError
            As ``MergedDataset.panel``, or a ``KeyError`` if no input holds
            the rule's price or volume.

        Examples
        --------
        >>> prices.panel("2009-06-18", "2009-06-23", symbols=[120127], variables=["adjClose"])["adjClose"].isnull().values.ravel().tolist()
        [False, True, True, False]
        """
        check_range(start, end, f"{self.class_name}.panel()")
        flags, data = self._flagged(start, end, symbols, variables)
        masked = [name for name in self.config.masked_variables if name in data.data_vars]
        data = data.assign(**{name: data[name].where(~flags) for name in masked})
        if variables is not None:
            data = data[list(variables)]
        return data

    def bad_prints(self, start, end, symbols: "Sequence | None" = None) -> xr.DataArray:
        """Return which bars from ``start`` to ``end`` are bad prints.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.
        symbols : sequence, optional
            Symbol labels to keep; ``None`` keeps every one.

        Returns
        -------
        xr.DataArray
            Booleans on ``(timestamp, symbol)``.

        Examples
        --------
        >>> flags = prices.bad_prints("2009-06-18", "2009-06-23", symbols=[120127])
        >>> flags.values.ravel().tolist()
        [False, True, True, False]
        """
        check_range(start, end, f"{self.class_name}.bad_prints()")
        return self._flagged(start, end, symbols, [])[0]

    def _flagged(self, start, end, symbols, variables) -> tuple[xr.DataArray, xr.Dataset]:
        """Return the flags and the merged panel of the range, read with the lookback before it."""
        config = self.config
        rule = [config.price_variable, config.volume_variable]
        wanted = None if variables is None else list(dict.fromkeys([*variables, *rule]))
        first = self._first_read(start)
        data = super().panel(first, end, symbols=symbols, variables=wanted)
        for name in rule:
            if name not in data.data_vars:
                raise KeyError(
                    f"{self.class_name}: no input holds {name!r}, which the bad-print "
                    f"rule reads."
                )
        price = data[config.price_variable].transpose("timestamp", "symbol")
        volume = data[config.volume_variable].transpose("timestamp", "symbol")
        flags = xr.DataArray(
            bad_print_mask(
                price.values, volume.values,
                jump=config.jump, volume_ratio=config.volume_ratio, lookback=config.lookback,
            ),
            coords=price.coords,
            dims=price.dims,
        )
        keep = slice(pd.Timestamp(start), None)
        return flags.sel(timestamp=keep), data.sel(timestamp=keep)

    def _first_read(self, start) -> pd.Timestamp:
        """Return the bar ``lookback`` bars before ``start``, or the first bar."""
        try:
            return self.bar_before(start, self.config.lookback)
        except InsufficientHistoryError:
            calendar = self._calendar()
            return calendar[0] if len(calendar) else pd.Timestamp(start)

    def copy(self) -> Self:
        """Return the view over copies of the inputs, with the same rule.

        Examples
        --------
        >>> other = prices.copy()
        >>> other == prices, other.datasets[0] is prices.datasets[0]
        (True, False)
        """
        fields = {
            name: getattr(self.config, name)
            for name in BadPrintMaskedDatasetConfig.__dataclass_fields__
            if name not in ("datasets", "name")
        }
        return type(self)([dataset.copy() for dataset in self.datasets], **fields)


__all__ = ["BadPrintMaskedDataset"]
