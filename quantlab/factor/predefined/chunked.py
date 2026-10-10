"""Chunked factor: another factor computed one chunk of time at a time.

A factor over a long range of many symbols may not fit in memory at once:
an Alpha158 store of every listed stock over twenty years is larger than
the machine. ``ChunkedFactor`` wraps any factor and splits the range into
**chunks** of one calendar period each, each computed with its own warm-up.
Joined in time order they equal the whole range computed at once, to within
floating-point tolerance, provided the wrapped factor's ``warmup_bars`` is
enough for its rolling operators.

``build`` and ``extend`` write the store chunk by chunk through the wrapped
factor's own ``build`` and ``extend``, so only one chunk is in memory and
only the store's owner writes it (ADR 0029). A build that fails part way
leaves the finished chunks and their recorded range, and ``extend`` resumes
from there. ``compute`` returns one panel in memory, so it fills it chunk
by chunk when only the computation would not fit, and refuses when the
output alone would not.

A factor is chunked along time only: a cross-sectional operator needs
every symbol of a bar at once.
"""

import ctypes
import dataclasses
import sys
from typing import Self

import numpy as np
import pandas as pd
import psutil
import xarray as xr

from quantlab.dataset._support.ledger import TimeChunkPlanner
from quantlab.factor.base import Factor
from quantlab.factor.config import ChunkedConfig
from quantlab.utils.date_range import last_moment

#: Share of the memory available when a computation starts that its chunks
#: may take; the rest is left for the process and the machine.
MEMORY_FRACTION = 0.5


def release_memory() -> None:
    """Hand the heap memory freed by the last chunk back to the system.

    glibc keeps freed blocks of a few tens of megabytes in the process
    heap, where fragmentation keeps them from being reused for the next
    chunk's arrays, so without this the process grows by about a chunk per
    chunk. Does nothing outside Linux.

    Examples
    --------
    >>> release_memory()
    """
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass


def memory_budget() -> int:
    """Return the bytes a chunk may take: ``MEMORY_FRACTION`` of the memory available.

    Examples
    --------
    >>> memory_budget() <= psutil.virtual_memory().available
    True
    """
    return int(psutil.virtual_memory().available * MEMORY_FRACTION)


class ChunkedFactor(Factor):
    """The outputs of ``config.factor``, computed one chunk of time at a time.

    The chunks are one ``TimeChunkPlanner`` period each, cut on the
    dataset's own bars. With ``config.granularity`` set every range is cut
    that way; with ``None`` the range is computed whole when it fits in
    ``memory_budget()``, and otherwise cut at the coarsest granularity
    whose largest chunk, warm-up included, fits. What a bar of a symbol
    takes is the wrapped factor's ``cell_bytes()``, and the symbols are the
    ones it outputs (``output_symbols()``), not its dataset's.

    There is no store of its own: ``store_path``, ``store_range`` and
    ``read`` are the wrapped factor's, so a model can read either.

    Parameters
    ----------
    config : ChunkedConfig
        The wrapped ``factor`` and an optional ``granularity``.

    Raises
    ------
    ValueError
        If the wrapped factor is resampled: resample its source factor's
        store instead, which is aggregation, not computation.

    Examples
    --------
    >>> chunked = ChunkedFactor(ChunkedConfig(factor=alpha158))
    >>> chunked.plan("2004-01-01", "2024-12-31")[0]
    'year'
    >>> chunked.build("2004-01-01", "2024-12-31").store_range()
    ('2004-01-01', '2024-12-31')
    >>> alpha158.read("2024-01-02", "2024-01-31")   # the same store
    """

    #: The config class ``from_config`` rebuilds this factor with.
    config_cls = ChunkedConfig

    # Narrower type annotation for readers and type checkers only.
    config: ChunkedConfig

    @Factor.config.setter
    def config(self, config: ChunkedConfig):
        """Install ``config`` with ``dataset`` set to the wrapped factor's dataset."""
        Factor.config.fset(
            self, dataclasses.replace(config, dataset=config.factor.config.dataset)
        )

    def copy(self) -> Self:
        """Return a copy with its own wrapped factor.

        Examples
        --------
        >>> other = chunked.copy()
        >>> other == chunked, other.config.factor is chunked.config.factor
        (True, False)
        """
        other = super().copy()
        other.config = dataclasses.replace(other.config, factor=self.config.factor.copy())
        return other

    def _get_factor_names(self) -> tuple[str, ...]:
        """Return the wrapped factor's output names."""
        return tuple(self.config.factor.get_factor_names())

    def _validate_config(self) -> None:
        """Refuse a warm-up, a store, resampling, names or a granularity it cannot honour.

        Raises
        ------
        ValueError
            If ``warmup_bars`` is not 0, ``file_path`` or ``resample_freq``
            is set, the wrapped factor is resampled, ``factor_names`` are
            not the wrapped factor's or ``granularity`` is unknown.
        """
        config = self.config
        if config.warmup_bars != 0:
            raise ValueError(
                f"{self.class_name}: warmup_bars must be 0, got {config.warmup_bars}; "
                f"the wrapped factor warms itself up."
            )
        if config.file_path is not None:
            raise ValueError(
                f"{self.class_name}: file_path must be None; the wrapper writes the "
                f"wrapped factor's store and has none of its own."
            )
        if config.resample_freq is not None or config.factor.config.resample_freq is not None:
            raise ValueError(
                f"{self.class_name}: cannot wrap or be a resampled factor; build the "
                f"source factor in chunks, then resample it, which reads its store."
            )
        if tuple(config.factor_names) != self._get_factor_names():
            raise ValueError(
                f"{self.class_name}: factor_names {list(config.factor_names)} are not "
                f"the wrapped factor's {list(self._get_factor_names())}; pin the "
                f"wrapped factor instead."
            )
        if config.granularity not in (None, *TimeChunkPlanner.GRANULARITIES):
            raise ValueError(
                f"{self.class_name}: unknown granularity {config.granularity!r}; "
                f"accepted values are {list(TimeChunkPlanner.GRANULARITIES)} or None."
            )

    @property
    def store_path(self) -> str | None:
        """Return the wrapped factor's store."""
        return self.config.factor.store_path

    def owns_store(self) -> bool:
        """Whether the wrapped factor owns its store; see ``Factor.owns_store``."""
        return self.config.factor.owns_store()

    def store_range(self) -> tuple[str, str] | None:
        """Return the wrapped factor's recorded store range."""
        return self.config.factor.store_range()

    def stored_symbols(self) -> list:
        """Return the symbol axis of the wrapped factor's store."""
        return self.config.factor.stored_symbols()

    def output_symbols(self) -> list:
        """Return the symbols the wrapped factor outputs; see ``Factor.output_symbols``.

        Examples
        --------
        >>> chunked.output_symbols() == alpha158.output_symbols()
        True
        """
        return self.config.factor.output_symbols()

    def read(self, start, end, symbols=None) -> xr.Dataset:
        """Return the wrapped factor's store from ``start`` to ``end``; see ``Factor.read``."""
        return self.config.factor.read(start, end, symbols=symbols)

    def keep_compiled(self):
        """Keep the wrapped factor's compiled graph inside the block."""
        return self.config.factor.keep_compiled()

    def plan(self, start, end) -> tuple[str | None, list[tuple[pd.Timestamp, pd.Timestamp]]]:
        """Return the granularity and the ``(first, last)`` bar of every chunk of a range.

        The granularity is ``None`` when the range is one chunk because it
        fits whole.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range, both inclusive.

        Raises
        ------
        MemoryError
            If one chunk of the finest granularity would not fit either.

        Examples
        --------
        >>> chunked.plan("2024-01-01", "2024-03-31")
        ('month', [(Timestamp('2024-01-02'), Timestamp('2024-01-31')), ...])
        """
        bars = self.config.dataset.calendar(start, end)
        if not len(bars):
            return self.config.granularity, []
        if self.config.granularity is not None:
            return self.config.granularity, self._chunks(bars, self.config.granularity)
        budget, needed = memory_budget(), self._peak_bytes(len(bars))
        if needed <= budget:
            return None, [(bars[0], bars[-1])]
        for granularity in TimeChunkPlanner.GRANULARITIES:
            chunks = self._chunks(bars, granularity)
            longest = max(
                bars.searchsorted(last, "right") - bars.searchsorted(first)
                for first, last in chunks
            )
            needed = self._peak_bytes(longest)
            if needed <= budget:
                return granularity, chunks
        raise MemoryError(
            f"{self.class_name}: one {granularity!r} chunk of "
            f"{self.config.factor.class_name} needs {needed:,} bytes at peak, more than "
            f"the {budget:,} bytes there are room for; free memory or shrink the "
            f"factor's symbols or outputs."
        )

    def compute(self, start, end) -> xr.Dataset:
        """Compute the wrapped factor from ``start`` to ``end``, a chunk at a time if needed.

        The panel is allocated once and each chunk is written into it, so
        only one chunk's computation is in memory beside it.

        Parameters
        ----------
        start, end : str, datetime.date or pd.Timestamp
            The range to return, both inclusive.

        Raises
        ------
        MemoryError
            If the output alone would not fit in ``memory_budget()``: build
            the store instead and read it a range at a time.

        Examples
        --------
        >>> dict(chunked.compute("2024-01-01", "2024-03-31").sizes)
        {'timestamp': 61, 'symbol': 3000}
        """
        factor = self.config.factor
        bars = self.config.dataset.calendar(start, end)
        output = len(bars) * self._symbol_count() * factor.cell_bytes()[1]
        if output > memory_budget():
            raise MemoryError(
                f"{self.class_name}.compute(): the output of {factor.class_name} from "
                f"{start} to {end} alone takes {output:,} bytes, more than the "
                f"{memory_budget():,} bytes there are room for; write it with "
                f"build(start, end) and read it a range at a time."
            )
        _, chunks = self.plan(start, end)
        if len(chunks) <= 1:
            return factor.compute(start, end)
        # The outer chunks keep the caller's own start and end labels.
        firsts = [start, *(first for first, _ in chunks[1:])]
        lasts = [*(last for _, last in chunks[:-1]), end]
        panel = None
        with self.keep_compiled():
            for first, last in zip(firsts, lasts):
                piece = factor.compute(first, last)
                if panel is None:
                    panel = self._allocate(bars, piece)
                rows = bars.get_indexer(pd.DatetimeIndex(piece["timestamp"].values))
                if (rows < 0).any() or not piece["symbol"].equals(panel["symbol"]):
                    raise ValueError(
                        f"{self.class_name}.compute(): the chunk from {first} to {last} "
                        f"is not on the dataset's bars and symbols; chunks cannot be joined."
                    )
                for name in panel.data_vars:
                    panel[name].values[rows] = piece[name].values
                del piece
                release_memory()
        return panel

    def build(self, start, end) -> Self:
        """Build the wrapped factor's store from ``start`` to ``end``, a chunk at a time.

        The first chunk is the wrapped factor's ``build``, every later one
        its ``extend``; see ``Factor.build``.

        Examples
        --------
        >>> chunked.build("2024-01-01", "2024-03-31").store_range()
        ('2024-01-01', '2024-03-31')
        """
        factor = self.config.factor
        _, chunks = self.plan(start, end)
        with self.keep_compiled():
            if len(chunks) <= 1:
                factor.build(start, end)
                return self
            factor.build(start, chunks[0][1])
            release_memory()
            for _, last in chunks[1:-1]:
                factor.extend(last)
                release_memory()
            factor.extend(end)
        return self

    def extend(self, end) -> Self:
        """Append the bars after the wrapped store's recorded range, a chunk at a time.

        Every chunk is the wrapped factor's ``extend``; see ``Factor.extend``.
        After a ``build`` that failed part way, this resumes it.

        Examples
        --------
        >>> chunked.extend("2024-06-30").store_range()
        ('2024-01-01', '2024-06-30')
        """
        factor = self.config.factor
        recorded = factor.store_range()
        if recorded is None or last_moment(end) <= last_moment(recorded[1]):
            factor.extend(end)  # it raises with the reason
            return self
        after = last_moment(recorded[1]) + pd.Timedelta(1, "ns")
        _, chunks = self.plan(after, end)
        with self.keep_compiled():
            for _, last in chunks[:-1]:
                factor.extend(last)
                release_memory()
            factor.extend(end)
        return self

    def _chunks(self, bars: pd.DatetimeIndex, granularity: str) -> list:
        """Return the ``(first, last)`` bars of the ``granularity`` chunks of ``bars``."""
        return TimeChunkPlanner(granularity).plan_from_timestamps(bars)

    def _symbol_count(self) -> int:
        """Return the number of symbols the wrapped factor outputs."""
        return len(self.output_symbols())

    def _peak_bytes(self, bars: int) -> int:
        """Return the peak bytes a chunk of ``bars`` bars takes, its warm-up included."""
        factor = self.config.factor
        return (bars + factor.warmup_bars) * self._symbol_count() * factor.cell_bytes()[0]

    @staticmethod
    def _allocate(bars: pd.DatetimeIndex, piece: xr.Dataset) -> xr.Dataset:
        """Return an empty panel on ``bars`` with ``piece``'s symbols and variables."""
        shape = (len(bars), piece.sizes["symbol"])
        return xr.Dataset(
            {
                name: (("timestamp", "symbol"), np.empty(shape, dtype=piece[name].dtype))
                for name in piece.data_vars
            },
            coords={"timestamp": bars.values, "symbol": piece["symbol"].values},
            attrs=piece.attrs,
        )
