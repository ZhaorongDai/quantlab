"""Massive client: one trading day's file from S3, and the trade-condition table over REST.

Massive (Stocks Developer, personal use) serves every SIP trade and its own
minute and day aggregates as one gzipped CSV per data type and trading day
in the S3 bucket ``flatfiles`` at ``files.massive.com`` (ADR 0030), and
reference data over REST at ``api.massive.com``. ``MassiveClient`` makes a
day's file present and verified in the raw tier
(``quantlab.dataset.massive.raw``) and stores the condition table beside
it, so a conversion needs no network.

Credentials come from the environment only: ``MASSIVE_API_KEY`` (REST, and
the S3 secret unless ``MASSIVE_S3_SECRET_ACCESS_KEY`` is set) and
``MASSIVE_S3_ACCESS_KEY_ID``. A missing one fails when the client is made,
before any transfer, naming the variables. The API key travels in an
``Authorization`` header, never in a URL a log could print.

Two seams carry every transfer, and tests replace both: the *object store*
(``size`` and ``read`` of a key; the default signs S3 requests with boto3)
and ``http_get`` (one REST GET; the default is ``rest_get``). The client
tells three failures apart: ``MassiveNotPublishedError`` (no file for that
day: a holiday, or today before the vendor publishes), ``MassiveEntitlementError`` (the key was
refused, or the day is outside the plan's window) and
``MassiveTransportError`` (the network), which alone is retried after a
back-off that doubles on each attempt.

A file is downloaded in one stream to ``<name>.part``, checked against the
vendor's size and decoded to the end of its gzip, and only then renamed
into place.

Examples
--------
Needs the credentials and the network::

    client = MassiveClient()
    client.condition_table("/data/quantlab/downloads")
    client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
"""

from __future__ import annotations

import gzip
import json
import os
import time
import zlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import requests
from loguru import logger

from quantlab.dataset.massive.raw import (
    BUCKET,
    VENDOR_DIR,
    conditions_file,
    raw_file,
    s3_key,
)
from quantlab.utils.atomic import write_json_atomically

#: The environment variable holding the API key (REST, and the S3 secret by default).
API_KEY_ENV = "MASSIVE_API_KEY"

#: The environment variable holding the S3 access key id.
S3_ACCESS_KEY_ENV = "MASSIVE_S3_ACCESS_KEY_ID"

#: The environment variable holding the S3 secret, when it is not the API key.
S3_SECRET_ENV = "MASSIVE_S3_SECRET_ACCESS_KEY"

#: The S3 endpoint of the daily files.
S3_ENDPOINT = "https://files.massive.com"

#: The REST endpoint of the condition table.
CONDITIONS_URL = "https://api.massive.com/v3/reference/conditions"

#: Bytes read from the store per chunk.
_CHUNK = 1 << 20


class MassiveCredentialError(RuntimeError):
    """A credential variable is not set."""


class MassiveEntitlementError(RuntimeError):
    """Massive refused the key (HTTP 401 or 403), or the day is outside the plan's window."""


class MassiveNotPublishedError(LookupError):
    """Massive has no file for that day: not a trading day, or not published yet."""


class MassiveTransportError(ConnectionError):
    """The connection failed or broke, or a downloaded file failed its checks; retried."""


@dataclass(frozen=True)
class MassiveCredentials:
    """The three secrets, read from the environment; ``repr`` shows none of them.

    Examples
    --------
    >>> MassiveCredentials("k", "i", "s")
    MassiveCredentials(<hidden>)
    """

    api_key: str = field(repr=False)
    s3_access_key_id: str = field(repr=False)
    s3_secret_access_key: str = field(repr=False)

    def __repr__(self) -> str:
        """Return a repr without the secrets."""
        return "MassiveCredentials(<hidden>)"

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> MassiveCredentials:
        """Read the credentials; the S3 secret defaults to the API key.

        Raises
        ------
        MassiveCredentialError
            If ``MASSIVE_API_KEY`` or ``MASSIVE_S3_ACCESS_KEY_ID`` is unset,
            naming every missing one.

        Examples
        --------
        >>> MassiveCredentials.from_env({"MASSIVE_API_KEY": "k"})
        Traceback (most recent call last):
        ...
        MassiveCredentialError: MASSIVE_S3_ACCESS_KEY_ID not set; ...
        """
        environ = os.environ if environ is None else environ
        missing = [name for name in (API_KEY_ENV, S3_ACCESS_KEY_ENV) if not environ.get(name)]
        if missing:
            raise MassiveCredentialError(
                f"{' and '.join(missing)} not set; set them in the environment (the S3 secret "
                f"{S3_SECRET_ENV} defaults to {API_KEY_ENV}). Nothing was downloaded."
            )
        return cls(
            api_key=environ[API_KEY_ENV],
            s3_access_key_id=environ[S3_ACCESS_KEY_ENV],
            s3_secret_access_key=environ.get(S3_SECRET_ENV) or environ[API_KEY_ENV],
        )


class ObjectStore(Protocol):
    """The S3 seam: the size of a key, and its bytes in chunks.

    Both raise ``MassiveNotPublishedError``, ``MassiveEntitlementError`` or
    ``MassiveTransportError``.
    """

    def size(self, key: str) -> int:
        """Return the object's size in bytes."""

    def read(self, key: str) -> Iterable[bytes]:
        """Return the object's bytes in chunks."""


class S3ObjectStore:
    """``ObjectStore`` over Massive's S3 endpoint, signed by boto3.

    Parameters
    ----------
    credentials : MassiveCredentials
        The S3 key id and secret.
    """

    def __init__(self, credentials: MassiveCredentials) -> None:
        """Create the boto3 client; see the class docstring."""
        import boto3
        from botocore.config import Config

        self._client = boto3.session.Session().client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=credentials.s3_access_key_id,
            aws_secret_access_key=credentials.s3_secret_access_key,
            config=Config(signature_version="s3v4", retries={"max_attempts": 1}, read_timeout=300),
        )

    @staticmethod
    def _translate(error: Exception, key: str) -> Exception:
        """Return the client's exception for a boto3 one."""
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(error, ClientError):
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return MassiveNotPublishedError(f"Massive has no {key!r} (not a trading day, or not published yet).")
            if code in ("401", "403", "AccessDenied", "Forbidden", "InvalidAccessKeyId", "SignatureDoesNotMatch"):
                return MassiveEntitlementError(
                    f"Massive refused {key!r} ({code}): the S3 credentials were rejected, or the day "
                    f"is outside the plan's window."
                )
            return MassiveTransportError(f"S3 error {code} on {key!r}: {error}")
        if isinstance(error, (BotoCoreError, OSError)):
            return MassiveTransportError(f"{type(error).__name__} on {key!r}: {error}")
        return error

    def size(self, key: str) -> int:
        """Return the object's size from a HEAD request.

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            store.size(s3_key("trades", date(2016, 11, 25)))  # 217911421
        """
        try:
            return int(self._client.head_object(Bucket=BUCKET, Key=key)["ContentLength"])
        except Exception as error:
            raise self._translate(error, key) from error

    def read(self, key: str) -> Iterable[bytes]:
        """Yield the object's bytes in chunks from one GET.

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            first = next(iter(store.read(s3_key("trades", date(2016, 11, 25)))))
        """
        try:
            body = self._client.get_object(Bucket=BUCKET, Key=key)["Body"]
            yield from body.iter_chunks(_CHUNK)
        except Exception as error:
            raise self._translate(error, key) from error


@dataclass(frozen=True)
class HttpResponse:
    """One REST response: its status and body."""

    status: int
    body: bytes


#: The REST seam: ``http_get(url, params=..., headers=...)``.
HttpGet = Callable[..., HttpResponse]


def rest_get(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> HttpResponse:
    """Send one GET; the default REST transport.

    Raises
    ------
    MassiveTransportError
        If the connection fails.

    Examples
    --------
    Needs the network; without a key Massive answers 401::

        rest_get(CONDITIONS_URL, params={"asset_class": "stocks"}, headers={}).status
    """
    try:
        response = requests.get(url, params=dict(params), headers=dict(headers), timeout=(30, 120))
    except requests.RequestException as error:
        raise MassiveTransportError(f"{type(error).__name__} on {url.split('?')[0]}") from error
    return HttpResponse(response.status_code, response.content)


class MassiveClient:
    """Download Massive's daily files and condition table into the raw tier.

    Parameters
    ----------
    credentials : MassiveCredentials, optional
        Read from the environment when omitted, so a missing variable fails
        here, before any transfer.
    store : ObjectStore, optional
        The S3 seam; an ``S3ObjectStore`` when omitted.
    http_get : callable, optional
        The REST seam; ``rest_get`` when omitted.
    retries : int, default 5
        Attempts after the first for a transport error.
    backoff : float, default 2.0
        Seconds before the first retry, doubled on each later one.
    sleep : callable, default time.sleep
        How the client waits.

    Raises
    ------
    MassiveCredentialError
        If a credential variable is unset.

    Examples
    --------
    Needs the credentials and the network::

        client = MassiveClient()
        client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
    """

    def __init__(
        self,
        *,
        credentials: MassiveCredentials | None = None,
        store: ObjectStore | None = None,
        http_get: HttpGet | None = None,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize; see the class docstring."""
        self.credentials = credentials if credentials is not None else MassiveCredentials.from_env()
        self.store = store if store is not None else S3ObjectStore(self.credentials)
        self.http_get = http_get if http_get is not None else rest_get
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep

    def _retrying(self, what: str, attempt: Callable[[], object]):
        """Run ``attempt``, retrying a ``MassiveTransportError`` after a doubling back-off."""
        wait = self.backoff
        for tries in range(self.retries + 1):
            try:
                return attempt()
            except MassiveTransportError as error:
                if tries == self.retries:
                    raise
                logger.warning(f"Massive: {what} failed ({error}); retrying in {wait:g}s.")
                self.sleep(wait)
                wait *= 2

    def download_day(self, data_type: str, day: date, download_root: str | Path) -> Path:
        """Make one data type's file for one trading day present and verified in the raw tier.

        A file already there with the vendor's size is kept as is.

        Parameters
        ----------
        data_type : str
            ``"trades"``, ``"minute_aggs"`` or ``"day_aggs"``.
        day : date
            The trading day.
        download_root : str or Path
            The download directory; the file lands under its ``massive/``.

        Returns
        -------
        Path
            The file in the raw tier.

        Raises
        ------
        MassiveNotPublishedError
            If Massive has no file for the day.
        MassiveEntitlementError
            If Massive refused the credentials or the day.
        MassiveTransportError
            If the transfer kept failing, or the file kept failing its checks.

        Examples
        --------
        Needs the credentials and the network; the file lands at
        ``/data/quantlab/downloads/massive/trades/2016/2016-11-25.csv.gz``,
        and Thanksgiving (2016-11-24) raises ``MassiveNotPublishedError``::

            client.download_day("trades", date(2016, 11, 25), "/data/quantlab/downloads")
            client.download_day("trades", date(2016, 11, 24), "/data/quantlab/downloads")
        """
        key = s3_key(data_type, day)
        target = raw_file(Path(download_root) / VENDOR_DIR, data_type, day)
        expected = self._retrying(f"HEAD {key}", lambda: self.store.size(key))
        if target.exists() and target.stat().st_size == expected:
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")

        def fetch() -> None:
            try:
                with partial.open("wb") as handle:
                    for chunk in self.store.read(key):
                        handle.write(chunk)
                written = partial.stat().st_size
                if written != expected:
                    raise MassiveTransportError(f"{key}: got {written} bytes, Massive lists {expected}")
                _check_gzip(partial, key)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise

        started = time.monotonic()
        self._retrying(f"GET {key}", fetch)
        os.replace(partial, target)
        elapsed = max(time.monotonic() - started, 1e-9)
        logger.info(f"Massive: {key} ({expected / 1e6:.1f} MB, {expected / 1e6 / elapsed:.1f} MB/s).")
        return target

    def _get_json(self, url: str, params: Mapping[str, str]) -> dict:
        """GET ``url`` with the key in a header; retry throttling, 5xx and transport errors."""
        headers = {"Authorization": f"Bearer {self.credentials.api_key}"}
        # Never the query: a next_url carries a cursor, and logs print this.
        where = url.split("?")[0]

        def attempt() -> dict:
            response = self.http_get(url, params=params, headers=headers)
            if response.status in (401, 403):
                raise MassiveEntitlementError(
                    f"Massive refused the API key on {where} ({response.status}: "
                    f"{_vendor_error(response.body)})."
                )
            if response.status == 429 or response.status >= 500:
                raise MassiveTransportError(f"HTTP {response.status} on {where}")
            if response.status != 200:
                raise MassiveTransportError(
                    f"HTTP {response.status} on {where}: {_vendor_error(response.body)}"
                )
            return json.loads(response.body)

        return self._retrying(f"GET {where}", attempt)

    def condition_table(self, download_root: str | Path) -> Path:
        """Pull the stock condition table over REST and store it as a snapshot.

        Every page is followed; the snapshot is the vendor's records
        verbatim, named by the pull's UTC time
        (``quantlab.dataset.massive.raw.conditions_file``).

        Parameters
        ----------
        download_root : str or Path
            The download directory; the snapshot lands under its ``massive/``.

        Returns
        -------
        Path
            The snapshot.

        Raises
        ------
        MassiveEntitlementError
            If Massive refused the API key.

        Examples
        --------
        Needs the credentials and the network (94 records on 2026-10-09)::

            path = client.condition_table("/data/quantlab/downloads")
        """
        pulled_at = datetime.now(UTC)
        records: list[dict] = []
        url, params = CONDITIONS_URL, {"asset_class": "stocks", "limit": "1000"}
        while url:
            page = self._get_json(url, params)
            records.extend(page.get("results", []))
            url, params = page.get("next_url"), {}
        path = conditions_file(Path(download_root) / VENDOR_DIR, pulled_at)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomically(path, records)
        logger.info(f"Massive: {len(records)} condition(s) -> {path}.")
        return path


def _check_gzip(path: Path, key: str) -> None:
    """Decode ``path`` to its end; raise ``MassiveTransportError`` if it is not a whole gzip."""
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(_CHUNK * 16):
                pass
    except (OSError, EOFError, zlib.error) as error:
        raise MassiveTransportError(f"{key}: not a whole gzip ({type(error).__name__}: {error})") from error


def _vendor_error(body: bytes) -> str:
    """Return the vendor's ``error`` text from a JSON body, else the body's start."""
    try:
        payload = json.loads(body)
        return str(payload.get("error") or payload.get("message") or "")
    except (ValueError, AttributeError):
        return body[:200].decode("utf-8", "replace")
