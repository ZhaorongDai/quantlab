"""Sharadar API client: pull a table as a bulk zip and keep it as raw parquet.

Sharadar (``api.sharadar.com/v1.0``) is the primary US-equity vendor (ADR
0023). It is reached through its own API, not Nasdaq Data Link: the
purchased key answers only there. The key is read from the
``SHARADAR_API_KEY`` environment variable and nowhere else, and it travels in
the ``x-api-key`` header, so it never appears in a URL a log could print.

A *bulk pull* asks ``GET /data/<table>?years=full``; Sharadar answers with an
HTTP 302 to a short-lived pre-signed URL of a zipped CSV. The client follows
the redirect itself, without the key, streams the zip to disk, checks that
the CSV header is the table's schema verbatim, and writes the rows as one
parquet file under ``<download-dir>/sharadar/<code>/`` (see
``quantlab.dataset.sharadar.tables``). The raw tier is the vendor's exact
rows, so every Zarr store can be rebuilt from it without the network.

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
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import requests
from loguru import logger

from quantlab.dataset.sharadar.tables import (
    VENDOR_DIR,
    SharadarTable,
    raw_table_dir,
    table,
)

#: The environment variable holding the API key; the only place it is read from.
API_KEY_ENV = "SHARADAR_API_KEY"

#: Root of the data endpoints.
DATA_URL = "https://api.sharadar.com/v1.0/data"


class SharadarEntitlementError(RuntimeError):
    """The key is missing, or Sharadar refused it (HTTP 401 or 403) for a table."""


class SharadarHttpError(RuntimeError):
    """Sharadar answered with an error status, or kept rate-limiting past the retries."""


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
    """
    response = requests.get(
        url,
        params=dict(params),
        headers=dict(headers),
        stream=True,
        allow_redirects=False,
        timeout=(30, 300),
    )
    return Response(
        status=response.status_code,
        headers=response.headers,
        body=response.iter_content(chunk_size=1 << 20),
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
    ):
        """Initialize the client; see the class docstring for parameters."""
        self._transport = transport
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds
        self._sleep = sleep

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
        # The pre-signed URL is the vendor's file storage: it carries its own
        # signature and never gets the key.
        signed = self._get(
            spec, response.headers["Location"], params={}, headers={}, keyed=False
        )
        if signed.status != 200:
            signed.close()
            raise SharadarHttpError(
                f"Sharadar table {code!r}: the bulk zip answered HTTP {signed.status}."
            )
        directory = raw_table_dir(Path(download_dir) / VENDOR_DIR, code)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{code}.parquet"
        with tempfile.TemporaryDirectory(dir=directory, prefix=".pull-") as scratch:
            archive = Path(scratch) / f"{code}.zip"
            try:
                with archive.open("wb") as handle:
                    for chunk in signed.body:
                        handle.write(chunk)
            finally:
                signed.close()
            csv_path = self._extract_csv(spec, archive, Path(scratch))
            partial = Path(scratch) / target.name
            pl.scan_csv(csv_path, schema=spec.schema).sink_parquet(partial)
            os.replace(partial, target)
        logger.info(f"Sharadar {code}: bulk pull (years={years}) written to {target}")
        return target

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
            On another 4xx, or a retried status past ``max_retries``.
        """
        for attempt in range(self._max_retries + 1):
            response = self._transport(url, params=params, headers=headers)
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
            if attempt == self._max_retries:
                break
            wait = _retry_after(response) or self._backoff_seconds * 2**attempt
            logger.warning(
                f"Sharadar table {spec.code!r}: HTTP {response.status}; retry "
                f"{attempt + 1}/{self._max_retries} in {wait:.1f}s"
            )
            self._sleep(wait)
        raise SharadarHttpError(
            f"Sharadar table {spec.code!r}: still HTTP {response.status} after "
            f"{self._max_retries} retries."
        )

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
            header = next(csv.reader(handle), [])
        if tuple(header) != tuple(spec.schema):
            raise ValueError(
                f"Sharadar table {spec.code!r}: the CSV columns {header} are "
                f"not the declared schema {list(spec.schema)}. Sharadar changed "
                f"the table; update quantlab.dataset.sharadar.tables."
            )
        return target


def _retry_after(response: Response) -> float | None:
    """Return the ``Retry-After`` seconds of a response, or ``None``."""
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _text(response: Response, limit: int = 300) -> str:
    """Return the start of a small error body as text, then close the response."""
    body = io.BytesIO()
    try:
        for chunk in response.body:
            body.write(chunk)
            if body.tell() >= limit:
                break
    finally:
        response.close()
    return body.getvalue()[:limit].decode("utf-8", errors="replace")
