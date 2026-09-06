from pathlib import Path
from typing import Optional, Self

import pandas as pd
import polars as pl
import xarray as xr

from base.backend import DataBackend


class XrBackend(DataBackend):
    def __init__(self) -> None:
        super().__init__()

    def read(self, path: str, overwrite: bool = False, **kwargs) -> Self:
        if not overwrite and hasattr(self, "data"):
            return self

        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = xr.open_dataset(path, **kwargs)
        return self

    #: Chunk length pinned along the append dimension by the FIRST write of
    #: an appended store. Without an explicit `encoding`, Zarr adopts the
    #: first window's own length as the chunk size, and every later append of
    #: a different length (a short trading year, a partial final month) is
    #: then misaligned with the on-disk chunk grid. A fixed value makes the
    #: layout a property of the store rather than of whichever window
    #: happened to be written first.
    APPEND_DIM_CHUNK = 512

    def write(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("mode", "w")
        self.data.to_zarr(path, **kwargs)
        return self

    def append(self, path: str, append_dim: str = "timestamp", **kwargs) -> Self:
        """Create the store, or extend it along `append_dim`.

        The storage-medium half of the chunked ingestion path: "how the
        in-memory object reaches this medium INCREMENTALLY" is the same
        concern, at the same altitude, as `write()`'s "how it reaches this
        medium". Nothing here reads a config -- the backend has no
        `raw_data_dir_path`, no `start_date` and no knowledge of where its
        data came from, and giving it any of those would break the
        medium-agnostic contract that lets `PlBackend` exist.

        **Contract, enforced rather than merely documented.** Every non-append
        dimension and its coordinate values must match the store EXACTLY
        across calls, and every shared data variable must keep its dtype.
        Raw `to_zarr(mode="a", append_dim=...)` enforces neither: a mismatched
        symbol coordinate is silently OVERWRITTEN with the new window's
        labels, leaving previously-written rows attributed to the wrong
        symbols, and an appended float64 NaN written into an int64 variable is
        silently cast to 0 -- a fabricated observation where data was missing.
        Both corruptions are invisible afterwards from the store alone, which
        is why they are checked here, before the irreversible append.

        `from_raw_data_chunked()` satisfies the coordinate half by pinning the
        symbol axis once over the whole range (D-02); this check is what turns
        that guarantee into an assertion.
        """
        target = Path(path)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            kwargs.setdefault("encoding", self._append_encoding(append_dim))
            self.data.to_zarr(path, mode="w", **kwargs)
            return self

        self._assert_append_compatible(path, append_dim)
        # No `encoding` on an append -- xarray rejects it outright, and the
        # chunk grid was already pinned by the creating write above.
        kwargs.pop("encoding", None)
        self.data.to_zarr(path, mode="a", append_dim=append_dim, **kwargs)
        return self

    def _append_encoding(self, append_dim: str) -> dict:
        """Pin each data variable's chunk shape: `APPEND_DIM_CHUNK` along the
        append dimension, the full length across every other one.
        """
        encoding = {}
        for name, variable in self.data.data_vars.items():
            if append_dim not in variable.dims:
                continue
            chunks = []
            for dim in variable.dims:
                size = int(self.data.sizes[dim])
                if dim == append_dim:
                    chunks.append(max(min(self.APPEND_DIM_CHUNK, size), 1))
                else:
                    chunks.append(max(size, 1))
            encoding[name] = {"chunks": tuple(chunks)}
        return encoding

    def _assert_append_compatible(self, path: str, append_dim: str) -> None:
        """Raise before an append that would silently corrupt the store."""
        existing = xr.open_zarr(path)
        try:
            for dim in self.data.dims:
                if dim == append_dim or dim not in existing.dims:
                    continue
                if dim not in self.data.coords or dim not in existing.coords:
                    continue
                incoming = self.data[dim].values
                stored = existing[dim].values
                if len(incoming) != len(stored) or not (incoming == stored).all():
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"the '{dim}' coordinate does not match the store "
                        f"({len(incoming)} incoming label(s) vs "
                        f"{len(stored)} stored). Zarr would OVERWRITE the "
                        f"stored labels without complaint, silently "
                        f"re-attributing every previously written row. Pin "
                        f"the '{dim}' axis over the whole range before the "
                        f"first window, the way "
                        f"BaseDataset.from_raw_data_chunked() does."
                    )
            for name, variable in self.data.data_vars.items():
                if name not in existing.data_vars:
                    continue
                if variable.dtype != existing[name].dtype:
                    raise ValueError(
                        f"XrBackend.append: refusing to append to {path} -- "
                        f"variable '{name}' has dtype {variable.dtype} but "
                        f"the store holds {existing[name].dtype}. Zarr would "
                        f"cast silently, and a float NaN cast into an "
                        f"integer store becomes 0: a fabricated observation "
                        f"where the data was missing."
                    )
        finally:
            existing.close()

    def to_internal(self, data: xr.Dataset) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.sel({col: slice(start_date, end_date)})
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.sel({col: list(symbols)})
        return self

    def get_xarray_dataset(
        self, indexes: Optional[list[str]] = None
    ) -> xr.Dataset:
        return self.data

    def get_lazyframe(self) -> pl.LazyFrame:
        data = self.data.to_dataframe().reset_index()
        return pl.from_pandas(data).lazy()


class PlBackend(DataBackend):
    def read(self, path: str, **kwargs) -> Self:
        if not Path(path).exists():
            raise FileNotFoundError(f"File {path} does not exist.")
        self.data = pl.scan_parquet(path)
        return self

    def write(self, path: str, **kwargs) -> Self:
        self.data.collect().write_parquet(path, **kwargs)
        return self

    def to_internal(self, data: pl.LazyFrame) -> Self:
        self.data = data
        return self

    def filter_by_date(self, col: str, start_date: str, end_date: str) -> Self:
        self.data = self.data.filter(
            pl.col(col).is_between(
                pl.lit(pd.to_datetime(start_date)),
                pl.lit(pd.to_datetime(end_date)),
            )
        )
        return self

    def filter_by_symbol(self, col: str, symbols: tuple[str, ...]) -> Self:
        self.data = self.data.filter(pl.col(col).is_in(symbols))
        return self

    def get_lazyframe(self) -> pl.LazyFrame:
        return self.data

    def get_xarray_dataset(self, indexes: list[str]) -> xr.Dataset:
        data = self.data.collect().to_pandas()
        data = data.set_index(indexes)
        return xr.Dataset.from_dataframe(data)
