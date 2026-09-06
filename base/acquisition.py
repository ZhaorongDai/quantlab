import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from loguru import logger

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

    **Watermarks record the covered RANGE, not just its end** (260906-26o
    D-03). Recording only the end date made a widened `start_date` silently
    skip every already-fetched symbol, shipping a dataset whose per-symbol
    history depth was inconsistent with no warning at all. The schema is
    additive -- `{"start_date": ..., "last_date": ...}`, with `last_date`
    keeping its original name -- so old and new readers each tolerate the
    other's files. An unknown covered start is represented by the key being
    ABSENT and is never guessed; see `_read_coverage` and `stamp_watermarks`.
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

    def _read_sidecar(self, symbol: str) -> dict | None:
        """Load a watermark sidecar's raw JSON, or None if it is absent or
        unparseable.

        The single tolerant read both `_read_watermark` and `_read_coverage`
        share, so there is exactly ONE failure policy for a corrupt sidecar
        rather than two that could drift apart.
        """
        path = self._watermark_path(symbol)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A missing or corrupt watermark sidecar must never crash the
            # refresh workflow -- fall back to None (config.start_date),
            # matching Dataset._reset_symbols' FileNotFoundError-fallback
            # pattern. Worst case is a wider-than-necessary re-fetch.
            return None
        return payload if isinstance(payload, dict) else None

    def _read_watermark(self, symbol: str) -> str | None:
        """The LAST covered date for `symbol`, or None.

        Signature and meaning are deliberately unchanged by the range-aware
        schema: both loops below and
        `ConcurrentTiingoAcquisition._attempt` use this to compute an
        incremental start, and none of them wants the covered start.
        """
        payload = self._read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def _read_coverage(self, symbol: str) -> dict | None:
        """The covered RANGE for `symbol` as `{"start_date", "last_date"}`,
        or None when no readable sidecar exists.

        Either component may be None. In particular a LEGACY sidecar --
        `{"last_date": ...}`, the only format written before 260906-26o --
        reads back with `start_date=None`, and nothing anywhere fills that in
        from `config.start_date` or any other fallback.

        That absence is the whole point (D-04). Only the user knows what
        window those files were actually fetched over; an invented start that
        happens to be wrong reproduces exactly the silent per-symbol history
        gap this schema exists to eliminate, and reproduces it invisibly.
        Stamping is therefore an explicit, user-supplied step --
        `stamp_watermarks()` below.
        """
        payload = self._read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
        }

    def _write_watermark(
        self, symbol: str, last_date: str, start_date: str | None = None
    ) -> None:
        """Record coverage for `symbol`.

        The schema is purely ADDITIVE: `last_date` keeps its name and meaning,
        so new code reads pre-26o files and pre-26o code reads new files, and
        no reader anywhere crashes on either.

        `start_date=None` omits the key ENTIRELY rather than writing a null --
        an unknown covered start is represented by absence, so it cannot be
        mistaken at read time for a recorded value.
        """
        path = self._watermark_path(symbol)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, str] = {"last_date": last_date}
        if start_date is not None:
            payload["start_date"] = start_date
        with open(path, "w") as f:
            json.dump(payload, f)

    def stamp_watermarks(self, start_date: str) -> int:
        """Fill the covered start into every sidecar that lacks one, and
        return how many files changed.

        The explicit, user-supplied migration D-04 requires. It takes the
        start from its CALLER and derives it from nothing -- not from
        `config.start_date`, not from a default. A sidecar that already
        records a start is left untouched, because overwriting a recorded
        range with a guessed one is the same silent-wrong-data failure in a
        different costume (T-26o-04).

        Issues zero network requests.
        """
        directory = Path(self.config.watermark_path)
        if not directory.exists():
            return 0

        changed = 0
        for path in sorted(directory.glob("*.json")):
            symbol = path.stem
            coverage = self._read_coverage(symbol)
            # Skips the failure manifest and anything unparseable: the
            # manifest has no `last_date`, and a corrupt file reads as None.
            if coverage is None or coverage["last_date"] is None:
                continue
            if coverage["start_date"] is not None:
                continue
            self._write_watermark(
                symbol, coverage["last_date"], start_date=start_date
            )
            changed += 1

        logger.info(
            f"Stamped covered start {start_date} onto {changed} watermark(s) "
            f"under {directory}; sidecars already recording a start were left "
            f"untouched."
        )
        return changed

    def download(self, symbols: list[str] | None = None) -> Self:
        for symbol in symbols or list(self.config.symbols):
            self._fetch_and_write(
                symbol,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
            # `_fetch_and_write` overwrites the raw file wholesale rather than
            # appending, so after a successful fetch it holds exactly the
            # requested range. Recording `config.start_date` as the covered
            # start is therefore a TRUE statement about the file on disk,
            # including when the window was narrowed.
            self._write_watermark(
                symbol, self.config.end_date, start_date=self.config.start_date
            )
        return self

    def refresh(self, symbols: list[str] | None = None) -> Self:
        for symbol in symbols or list(self.config.symbols):
            coverage = self._read_coverage(symbol) or {}
            start = coverage.get("last_date") or self.config.start_date
            self._fetch_and_write(
                symbol, start_date=start, end_date=self.config.end_date
            )
            # Refresh fetches from the symbol's own last covered date FORWARD,
            # so the covered start is whatever it already was. If it was
            # unknown it stays unknown -- refresh does not invent coverage it
            # did not fetch (D-04).
            self._write_watermark(
                symbol,
                self.config.end_date,
                start_date=coverage.get("start_date"),
            )
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
