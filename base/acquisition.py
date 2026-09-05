import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from base.config import AcquisitionConfig
from enums.constant import Date


class Acquisition(ABC):
    """Abstract base for network-fetching, config-driven data acquisition.

    Mirrors `base/data.py:Dataset`'s config lifecycle idiom (config
    property/setter, `import_path`, `class_name`), but is decoupled from the
    Dataset/Zarr layer entirely -- `Acquisition` subclasses only fetch raw
    vendor data over the network and write it to local raw files under
    `config.raw_data_dir_path`; they never touch xarray/Zarr storage. The
    matching `Dataset` subclass is responsible for converting those raw local
    files into the canonical xarray representation via `_raw_data_to_xr()`.

    `download()` performs a one-shot full backfill over
    `[config.start_date, config.end_date]`. `refresh()` performs an
    incremental fetch per-symbol, starting from that symbol's last recorded
    watermark date (falling back to `config.start_date` if no watermark
    exists yet), so repeated calls do not re-fetch the entire history.
    """

    def __init__(self, config: AcquisitionConfig):
        self.config = config

    def __repr__(self):
        return f"{self.__class__.__name__}(config={self.config})"

    @property
    def config(self) -> AcquisitionConfig:
        return self._config

    @config.setter
    def config(self, config: AcquisitionConfig):
        self._config = config
        self._config.name = self.import_path

        if self._config.start_date is None:
            self._config.start_date = Date.START_DATE
        if self._config.end_date is None:
            self._config.end_date = Date.END_DATE

    @property
    def import_path(self) -> str:
        return f"{self.__class__.__module__}.{self.__class__.__qualname__}"

    @property
    def class_name(self) -> str:
        return self.__class__.__name__

    def _watermark_path(self, symbol: str) -> Path:
        return Path(self.config.watermark_path) / f"{symbol}.json"

    def _read_watermark(self, symbol: str) -> str | None:
        path = self._watermark_path(symbol)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f).get("last_date")
        except (json.JSONDecodeError, OSError):
            # A missing or corrupt watermark sidecar must never crash the
            # refresh workflow -- fall back to None (config.start_date),
            # matching Dataset._reset_symbols' FileNotFoundError-fallback
            # pattern. Worst case is a wider-than-necessary re-fetch.
            return None

    def _write_watermark(self, symbol: str, last_date: str) -> None:
        path = self._watermark_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"last_date": last_date}, f)

    def download(self, symbols: list[str] | None = None) -> Self:
        for symbol in symbols or list(self.config.symbols):
            self._fetch_and_write(
                symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            self._write_watermark(symbol, self.config.end_date)
        return self

    def refresh(self, symbols: list[str] | None = None) -> Self:
        for symbol in symbols or list(self.config.symbols):
            start = self._read_watermark(symbol) or self.config.start_date
            self._fetch_and_write(
                symbol, start_date=start, end_date=self.config.end_date
            )
            self._write_watermark(symbol, self.config.end_date)
        return self

    @abstractmethod
    def _fetch_and_write(
        self, symbol: str, start_date: str, end_date: str
    ) -> None:
        """Fetch raw vendor data for `symbol` over `[start_date, end_date]`
        and write it to local raw files under `self.config.raw_data_dir_path`,
        in the schema the matching `Dataset` subclass's `_raw_data_to_xr()`
        expects. Must not touch xarray/Zarr storage directly.
        """
        ...
