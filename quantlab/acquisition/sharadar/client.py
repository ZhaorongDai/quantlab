"""Sharadar API client: pull a table as a bulk zip and keep it as raw parquet.

Sharadar (``api.sharadar.com/v1.0``) is the primary US-equity vendor (ADR
0023). It is reached through its own API, not Nasdaq Data Link: the
purchased key answers only there. The key is read from the
``SHARADAR_API_KEY`` environment variable and nowhere else, and it travels in
the ``x-api-key`` header, so it never appears in a URL a log could print.

A *bulk pull* asks ``GET /data/<table>?years=full``; Sharadar answers with an
HTTP 302 to a short-lived pre-signed URL of a zipped CSV. The client follows
the redirect itself, without the key, downloads the zip to disk in parallel
byte ranges (one stream when the storage ignores ``Range``), checks that
the CSV header is the table's schema verbatim, and writes the rows as one
parquet file under ``<download-dir>/sharadar/<code>/`` (see
``quantlab.dataset.sharadar.tables``). The raw tier is the vendor's exact
rows, so every Zarr store can be rebuilt from it without the network.

A *date-window pull* (``window_table``) refreshes a date-keyed table every
morning: it asks REST for every calendar day from a trailing number of
trading days before the table's watermark through today, one request per
day and page, and writes the rows as one window file, a complete copy of
the table over those dates that supersedes earlier rows of them. Each pull,
bulk or window, then records the table's watermark, so an interrupted
refresh restarts from the last one that finished.

Every HTTP request goes through one *transport* function, the client's only
seam: tests replace it and run offline. A 401 or 403, or a missing key,
raises ``SharadarEntitlementError`` naming the table. A 429 (rate limited) or
a 5xx is retried after a back-off: ``Retry-After`` when the response gives
one, otherwise a wait that doubles on each attempt.

Examples
--------
Pull SEP and TICKERS (needs ``SHARADAR_API_KEY`` and the network); each lands
at ``/data/quantlab/downloads/sharadar/<code>/<code>.parquet``::

    client = SharadarClient()
    client.bulk_table("sep", "/data/quantlab/downloads")
    client.bulk_table("tickers", "/data/quantlab/downloads")
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import requests
from loguru import logger

from quantlab.dataset.sharadar.tables import (
    VENDOR_DIR,
    WINDOW_PREFIX,
    SharadarTable,
    raw_table_dir,
    read_watermark,
    scan_raw_table,
    table,
    vendor_today,
    window_file,
    write_watermark,
)

#: The environment variable holding the API key; the only place it is read from.
API_KEY_ENV = "SHARADAR_API_KEY"

#: Root of the data endpoints.
DATA_URL = "https://api.sharadar.com/v1.0/data"


class SharadarEntitlementError(RuntimeError):
    """The key is missing, or Sharadar refused it (HTTP 401 or 403) for a table."""


class SharadarHttpError(RuntimeError):
    """Sharadar answered with an error status, or kept failing past the retries."""


class SharadarTransportError(ConnectionError):
    """The connection failed to open, or broke while a body was read.

    The transport raises it, so the client retries a network failure
    without knowing the HTTP library underneath.
    """


@dataclass(frozen=True)
class Response:
    """One HTTP response as the transport returns it.

    Attributes
    ----------
    status : int
        The HTTP status code.
    headers : Mapping[str, str]
        The response headers.
    body : Iterable[bytes]
        The body in chunks, so a gigabyte zip is never held in memory.
    close : callable
        Releases the connection; called on a response the client discards.
    """

    status: int
    headers: Mapping[str, str]
    body: Iterable[bytes]
    close: Callable[[], None] = lambda: None


#: The transport's signature: ``transport(url, params=..., headers=...)``.
Transport = Callable[..., Response]


def http_get(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> Response:
    """Send one GET without following redirects; the default transport.

    Parameters
    ----------
    url : str
        The URL, which may carry its own query string (a pre-signed URL).
    params : Mapping[str, str]
        Query parameters to add.
    headers : Mapping[str, str]
        Request headers.

    Returns
    -------
    Response
        The status, headers and the streamed body.

    Raises
    ------
    SharadarTransportError
        If the connection fails to open, or breaks while the body is read.
    """
    try:
        response = requests.get(
            url,
            params=dict(params),
            headers=dict(headers),
            stream=True,
            allow_redirects=False,
            timeout=(30, 300),
        )
    except requests.RequestException as exc:
        raise SharadarTransportError(str(exc)) from exc

    def body() -> Iterable[bytes]:
        try:
            yield from response.iter_content(chunk_size=1 << 20)
        except requests.RequestException as exc:
            raise SharadarTransportError(str(exc)) from exc

    return Response(
        status=response.status_code,
        headers=response.headers,
        body=body(),
        close=response.close,
    )


class SharadarClient:
    """Pull Sharadar tables into the raw tier through one transport function.

    Parameters
    ----------
    transport : callable, default ``http_get``
        Sends one GET: ``transport(url, params=..., headers=...)`` returns a
        ``Response`` and must not follow redirects.
    max_retries : int, default 5
        Retries after a 429 or 5xx before ``SharadarHttpError`` is raised.
    backoff_seconds : float, default 2.0
        First wait when the response gives no ``Retry-After``; it doubles on
        each retry.
    sleep : callable, default ``time.sleep``
        Called with the seconds to wait between retries.
    download_workers : int, default 8
        Threads downloading byte ranges of a bulk zip at once.
    part_bytes : int, default 64 MiB
        Size of one byte range.

    Examples
    --------
    Tests pass a fake transport, which keeps every request offline::

        client = SharadarClient(transport=fake_transport, sleep=lambda s: None)
        client.bulk_table("indicators", download_dir)
    """

    #: Statuses that are retried after a back-off.
    RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        transport: Transport = http_get,
        *,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        download_workers: int = 8,
        part_bytes: int = 64 << 20,
    ):
        """Initialize the client; see the class docstring for parameters."""
        if download_workers < 1 or part_bytes < 1:
            raise ValueError("download_workers and part_bytes must be at least 1")
        self._transport = transport
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep
        self._download_workers = download_workers
        self._part_bytes = part_bytes

    def bulk_table(self, code: str, download_dir: str | Path, *, years: str = "full") -> Path:
        """Pull a whole table as a bulk zip and write it as raw parquet.

        The parquet replaces any earlier bulk pull of the table and is
        written only once the whole zip has been read and checked, so an
        interrupted pull leaves the previous file in place.

        Parameters
        ----------
        code : str
            The table's code (``"sep"``, ``"tickers"``, ``"indicators"``).
        download_dir : str or Path
            The download root; the table goes under ``sharadar/<code>/``.
        years : str, default "full"
            The history tier of the bulk file: ``"5"``, ``"10"`` or
            ``"full"``. It must not exceed the subscription.

        Returns
        -------
        Path
            The parquet file written.

        Raises
        ------
        KeyError
            If the code is not a known table; no request is sent.
        SharadarEntitlementError
            If the key is missing or refused for this table.
        SharadarHttpError
            On any other error status, or a rate limit past the retries.
        ValueError
            If the CSV header is not the table's schema.

        Examples
        --------
        Pull the five-year tier (needs the key and the network)::

            SharadarClient().bulk_table("sep", "/data/quantlab/downloads", years="5")
        """
        spec = table(code)
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise SharadarEntitlementError(
                f"cannot pull Sharadar table {code!r}: {API_KEY_ENV} is not set."
            )
        response = self._get(
            spec,
            f"{DATA_URL}/{spec.api_name}",
            params={"years": years},
            headers={"x-api-key": key},
        )
        response.close()
        if response.status != 302 or "Location" not in response.headers:
            raise SharadarHttpError(
                f"Sharadar table {code!r}: a bulk pull answered HTTP "
                f"{response.status} without a redirect to the zip."
            )
        directory = raw_table_dir(Path(download_dir) / VENDOR_DIR, code)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{code}.parquet"
        with tempfile.TemporaryDirectory(dir=directory, prefix=".pull-") as scratch:
            archive = Path(scratch) / f"{code}.zip"
            self._download_zip(spec, response.headers["Location"], archive)
            csv_path = self._extract_csv(spec, archive, Path(scratch))
            partial = Path(scratch) / target.name
            pl.scan_csv(csv_path, schema=spec.schema).sink_parquet(partial)
            os.replace(partial, target)
        # The bulk file is the newest copy of every date, so it supersedes the
        # windows; the watermark (the day of the pull) is written last.
        vendor_root = Path(download_dir) / VENDOR_DIR
        for window in directory.glob(f"{WINDOW_PREFIX}*.parquet"):
            window.unlink()
        if "date" in spec.schema:
            write_watermark(vendor_root, code, vendor_today())
        logger.info(f"Sharadar {code}: bulk pull (years={years}) written to {target}")
        return target

    def window_table(
        self,
        code: str,
        download_dir: str | Path,
        *,
        through: str | date | None = None,
        trading_days: int = 10,
        page_rows: int = 10_000,
    ) -> Path:
        """Pull a table's recent dates over REST and write them as a window file.

        The window runs from the ``trading_days``-th most recent date of the
        raw table on or before its watermark through ``through``, every
        calendar day (a weekend answers no row). Each day is asked for
        separately and paged by ``page_rows`` in ticker order, so paging is
        stable. The rows are checked against the schema, written as one
        window file (see ``quantlab.dataset.sharadar.tables.window_file``),
        and only then is the watermark moved to ``through``: a failed pull
        leaves the raw tier and the watermark as they were.

        Parameters
        ----------
        code : str
            A date-keyed table's code (``"sep"``, ``"actions"``).
        download_dir : str or Path
            The download root holding ``sharadar/<code>/``, with a bulk pull
            of the table already in it.
        through : str, date or None, default None
            Last day of the window; ``None`` is today in Sharadar's time
            zone (US/Eastern).
        trading_days : int, default 10
            Raw dates re-pulled before the watermark, so a late vendor
            correction to a recent day is seen.
        page_rows : int, default 10000
            Rows per request (Sharadar's ``limit``).

        Returns
        -------
        Path
            The window file written.

        Raises
        ------
        KeyError
            If the code is not a known table.
        ValueError
            If the table has no ``date`` column, or no raw pull yet.
        SharadarEntitlementError, SharadarHttpError
            As for ``bulk_table``.

        Examples
        --------
        Refresh SEP and ACTIONS each morning (needs the key and the network)::

            client = SharadarClient()
            for code in ("sep", "actions"):
                client.window_table(code, "/data/quantlab/downloads")
        """
        spec = table(code)
        if "date" not in spec.schema:
            raise ValueError(f"Sharadar table {code!r} has no date column to window.")
        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise SharadarEntitlementError(
                f"cannot pull Sharadar table {code!r}: {API_KEY_ENV} is not set."
            )
        vendor_root = Path(download_dir) / VENDOR_DIR
        end = date.fromisoformat(str(through)) if through is not None else vendor_today()
        start = self._window_start(vendor_root, code, end, trading_days)
        frames = []
        day = start
        while day <= end:
            frames.extend(self._pull_day(spec, key, day, page_rows))
            day += timedelta(days=1)
        rows = pl.concat(frames) if frames else pl.DataFrame(schema=spec.schema)
        target = window_file(vendor_root, code, datetime.now(UTC), start, end)
        partial = target.with_name(f".{target.name}.partial")
        rows.write_parquet(partial)
        os.replace(partial, target)
        write_watermark(vendor_root, code, end)
        logger.info(
            f"Sharadar {code}: {rows.height} row(s) over {start}..{end} written to {target}"
        )
        return target

    @staticmethod
    def _window_start(vendor_root: Path, code: str, end: date, trading_days: int) -> date:
        """Return the first day of a window: ``trading_days`` raw dates before the watermark."""
        watermark = read_watermark(vendor_root, code)
        if watermark is None:
            raise ValueError(
                f"Sharadar table {code!r} has no watermark under {vendor_root}; "
                f"pull it in bulk first (SharadarClient.bulk_table)."
            )
        through = min(watermark, end)
        dates = (
            scan_raw_table(vendor_root, code)
            .filter(pl.col("date") <= pl.lit(through))
            .select(pl.col("date").unique().sort(descending=True).head(trading_days))
            .collect()
            .get_column("date")
        )
        return dates.min() if dates.len() else through

    def _pull_day(
        self, spec: SharadarTable, key: str, day: date, page_rows: int
    ) -> list[pl.DataFrame]:
        """Return one day's rows of a table, page by page."""
        pages = []
        offset = 0
        while True:
            response = self._get(
                spec,
                f"{DATA_URL}/{spec.api_name}",
                params={
                    "from": day.isoformat(),
                    "to": day.isoformat(),
                    "format": "csv",
                    "sort": "ticker.asc",
                    "limit": str(page_rows),
                    "offset": str(offset),
                },
                headers={"x-api-key": key},
            )
            text = _text(response, limit=None)
            page = self._parse_page(spec, text)
            pages.append(page)
            if page.height < page_rows:
                return pages
            offset += page_rows

    @staticmethod
    def _parse_page(spec: SharadarTable, text: str) -> pl.DataFrame:
        """Parse one CSV page, checking its header is the table's schema."""
        _check_header(spec, next(csv.reader(io.StringIO(text)), []))
        return pl.read_csv(io.StringIO(text), schema=spec.schema)

    def _download_zip(self, spec: SharadarTable, url: str, archive: Path) -> None:
        """Download the pre-signed zip to ``archive`` in parallel byte ranges.

        A one-byte ``Range`` probe reads the size; the parts are then
        fetched by ``download_workers`` threads, each retried like any
        request, and written at their offsets. Storage that ignores
        ``Range`` (answers 200) is read as one stream instead. The
        pre-signed URL is the vendor's file storage: it carries its own
        signature and never gets the key.
        """
        probe = self._get(spec, url, params={}, headers={"Range": "bytes=0-0"}, keyed=False)
        if probe.status == 200:
            self._write_body(spec, probe, archive, offset=0, length=None)
            return
        probe.close()
        total = _range_total(probe) if probe.status == 206 else None
        if total is None:
            raise SharadarHttpError(
                f"Sharadar table {spec.code!r}: the bulk zip answered HTTP "
                f"{probe.status} to a range request."
            )
        with archive.open("wb") as handle:
            handle.truncate(total)
        parts = [
            (start, min(start + self._part_bytes, total) - 1)
            for start in range(0, total, self._part_bytes)
        ]

        def fetch(part: tuple[int, int]) -> None:
            first, last = part
            for attempt in range(self._max_retries + 1):
                response = self._get(
                    spec, url, params={}, headers={"Range": f"bytes={first}-{last}"}, keyed=False
                )
                if response.status != 206:
                    response.close()
                    raise SharadarHttpError(
                        f"Sharadar table {spec.code!r}: bytes {first}-{last} of the "
                        f"bulk zip answered HTTP {response.status}, not 206."
                    )
                try:
                    self._write_body(
                        spec, response, archive, offset=first, length=last - first + 1
                    )
                    return
                except SharadarTransportError as exc:
                    error = exc
                if attempt < self._max_retries:
                    self._back_off(spec, attempt, f"bytes {first}-{last}: {error}")
            raise SharadarHttpError(
                f"Sharadar table {spec.code!r}: bytes {first}-{last} of the bulk "
                f"zip still broken after {self._max_retries} retries: {error}"
            )

        workers = min(self._download_workers, len(parts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() re-raises the first failed part.
            list(pool.map(fetch, parts))
        logger.info(
            f"Sharadar {spec.code}: {total / 2**20:.1f} MiB downloaded in "
            f"{len(parts)} part(s) by {workers} thread(s)"
        )

    @staticmethod
    def _write_body(
        spec: SharadarTable, response: Response, archive: Path, *, offset: int, length: int | None
    ) -> None:
        """Write a response body into ``archive`` at ``offset``, checking its length.

        Raises
        ------
        SharadarTransportError
            If the stream breaks or ends short of ``length``.
        """
        written = 0
        try:
            with archive.open("r+b" if length is not None else "wb") as handle:
                handle.seek(offset)
                for chunk in response.body:
                    handle.write(chunk)
                    written += len(chunk)
        finally:
            response.close()
        if length is not None and written != length:
            raise SharadarTransportError(
                f"Sharadar table {spec.code!r}: bytes {offset}-{offset + length - 1} "
                f"of the bulk zip arrived as {written} bytes."
            )

    def _get(
        self,
        spec: SharadarTable,
        url: str,
        *,
        params: Mapping[str, str],
        headers: Mapping[str, str],
        keyed: bool = True,
    ) -> Response:
        """Send one GET, retrying rate limits and server errors.

        ``keyed`` says whether the request carries the key. A 401 or 403 on
        a request without it (the pre-signed zip URL, whose signature may
        have expired) says nothing about the key and is a plain HTTP error.

        Raises
        ------
        SharadarEntitlementError
            On a 401 or 403 to a keyed request.
        SharadarHttpError
            On another 4xx, or a retried status or connection failure past
            ``max_retries``.
        """
        for attempt in range(self._max_retries + 1):
            try:
                response = self._transport(url, params=params, headers=headers)
            except SharadarTransportError as exc:
                failure = str(exc)
                if attempt < self._max_retries:
                    self._back_off(spec, attempt, failure)
                continue
            if response.status in (401, 403) and keyed:
                raise SharadarEntitlementError(
                    f"Sharadar refused table {spec.code!r} (API name "
                    f"{spec.api_name!r}) with HTTP {response.status}: "
                    f"{_text(response)!r}. Check that {API_KEY_ENV} is a paid "
                    f"sharadar.com key whose plan covers this table."
                )
            if response.status not in self.RETRY_STATUSES:
                if response.status >= 400:
                    raise SharadarHttpError(
                        f"Sharadar table {spec.code!r}: HTTP {response.status}: "
                        f"{_text(response)!r}"
                    )
                return response
            response.close()
            failure = f"HTTP {response.status}"
            if attempt < self._max_retries:
                self._back_off(spec, attempt, failure, _retry_after(response))
        raise SharadarHttpError(
            f"Sharadar table {spec.code!r}: still {failure} after "
            f"{self._max_retries} retries."
        )

    def _back_off(
        self, spec: SharadarTable, attempt: int, failure: str, wait: float | None = None
    ) -> None:
        """Log a retryable failure and sleep before retry ``attempt + 1``.

        ``wait`` is the server's ``Retry-After``; without it the wait is
        ``backoff_seconds`` doubled on each attempt.
        """
        wait = wait or self._backoff_seconds * 2**attempt
        logger.warning(
            f"Sharadar table {spec.code!r}: {failure}; retry "
            f"{attempt + 1}/{self._max_retries} in {wait:.1f}s"
        )
        self._sleep(wait)

    @staticmethod
    def _extract_csv(spec: SharadarTable, archive: Path, scratch: Path) -> Path:
        """Extract the zip's one CSV member and check its header is the schema."""
        with zipfile.ZipFile(archive) as bundle:
            members = [name for name in bundle.namelist() if name.endswith(".csv")]
            if len(members) != 1:
                raise ValueError(
                    f"Sharadar table {spec.code!r}: the bulk zip holds "
                    f"{len(members)} CSV files ({members}), expected one."
                )
            target = scratch / f"{spec.code}.csv"
            with bundle.open(members[0]) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1 << 20)
        with target.open(newline="", encoding="utf-8") as handle:
            _check_header(spec, next(csv.reader(handle), []))
        return target


def _range_total(response: Response) -> int | None:
    """Return the full size from a 206's ``Content-Range: bytes a-b/total``."""
    value = response.headers.get("Content-Range", "")
    _, _, total = value.rpartition("/")
    return int(total) if total.isdigit() else None


def _retry_after(response: Response) -> float | None:
    """Return the ``Retry-After`` seconds of a response, or ``None``."""
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _text(response: Response, limit: int | None = 300) -> str:
    """Return a body as text (its first ``limit`` bytes, or all of it), then close the response."""
    body = io.BytesIO()
    try:
        for chunk in response.body:
            body.write(chunk)
            if limit is not None and body.tell() >= limit:
                break
    finally:
        response.close()
    return body.getvalue()[:limit].decode("utf-8", errors="replace")


def _check_header(spec: SharadarTable, header: list[str]) -> None:
    """Raise if a CSV header is not the table's schema, verbatim."""
    if tuple(header) != tuple(spec.schema):
        raise ValueError(
            f"Sharadar table {spec.code!r}: the CSV columns {header} are "
            f"not the declared schema {list(spec.schema)}. Sharadar changed "
            f"the table; update quantlab.dataset.sharadar.tables."
        )
