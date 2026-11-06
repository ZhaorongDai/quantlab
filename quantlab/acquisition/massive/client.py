"""Massive client: one trading day's file from S3, and REST endpoints into the raw tier.

Massive (Stocks Developer, personal use) serves every SIP trade and its own
minute and day aggregates as one gzipped CSV per data type and trading day
in the S3 bucket ``flatfiles`` at ``files.massive.com``, and
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
(``size`` of a key and ``read`` of a byte range; the default signs S3
requests with boto3) and ``http_get`` (one REST GET returning an
``HttpResponse`` of ``quantlab.acquisition._support.http``; the default is
``rest_transport``). The client tells three failures apart:
``MassiveNotPublishedError`` (no file for that day: a holiday, or today
before the vendor publishes), ``MassiveEntitlementError`` (the key was
refused, or the day is outside the plan's window) and
``MassiveTransportError`` (the network, throttling included), which alone is
retried after a back-off that doubles on each attempt. Every ranged GET is
conditioned on the ETag its HEAD read: an object replaced mid-download
answers 412 (``MassiveObjectChangedError``), and the day starts over at once
from a fresh HEAD.

``pull`` keeps one REST endpoint (``quantlab.acquisition.massive.rest``) in
the raw tier, in the endpoint's pull shape
(``quantlab.acquisition.massive.shapes``): the client carries each GET, the
shape says what to ask and what to keep, so a rerun asks only for what is new
and an interrupted backfill resumes. HTTP 429 and 5xx are retried after the same back-off, or after the ``Retry-After`` the
response gives; 401 and 403 raise
``MassiveEntitlementError`` naming the endpoint.

A file is downloaded in ``streams`` byte ranges of ``part_bytes`` at once, each
written at its offset of ``<name>.part``; a range whose stream breaks is
asked again for only the bytes it still lacks
(``quantlab.acquisition._support.ranged_download``). The whole file is then
checked against the vendor's size and decoded to the end of its gzip, and
only then renamed into place. ``download_days`` runs over the XNYS sessions
of a date range, oldest first, with several files in flight, and records a
watermark per data type, so an interrupted run resumes where it stopped.
One file is fetched at a time, with all ``streams``, while up to
``files - 1`` earlier ones are being verified: decoding a day of trades
takes a core about twice as long as fetching it (``docs/user-guide/massive.md`` lists
the measured rates).

Examples
--------
Needs the credentials and the network::

    client = MassiveClient()
    client.condition_table("/data/quantlab/downloads")
    client.pull("splits", "/data/quantlab/downloads")
    client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
    for day in client.download_days("minute_aggs", date(2016, 10, 11), date(2016, 12, 30),
                                     "/data/quantlab/downloads"):
        print(day.day, day.published)
"""

from __future__ import annotations

import gzip
import json
import os
import threading
import time
import zlib
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

from joblib import Parallel, delayed
from loguru import logger

from quantlab.acquisition._support.http import HttpResponse, Transport, http_get
from quantlab.acquisition._support.ranged_download import DownloadAborted, download_ranges
from quantlab.acquisition._support.request_policy import (
    REFUSAL_STATUSES,
    RequestPolicy,
    TransientError,
    require_env,
    retry_after,
)
from quantlab.acquisition._support.sessions import xnys_sessions
from quantlab.acquisition.massive.raw_tier import write_day_file_watermark
from quantlab.acquisition.massive.rest import CONDITIONS, ENDPOINTS
from quantlab.acquisition.massive.shapes import MassiveResponseError, PullRun, RestEndpoint, RestPull
from quantlab.dataset.massive.raw import BUCKET, VENDOR_DIR, raw_file, read_watermark, s3_key
from quantlab.utils.watermark import holds, resume_from

#: The environment variable holding the API key (REST, and the S3 secret by default).
API_KEY_ENV = "MASSIVE_API_KEY"

#: The environment variable holding the S3 access key id.
S3_ACCESS_KEY_ENV = "MASSIVE_S3_ACCESS_KEY_ID"

#: The environment variable holding the S3 secret, when it is not the API key.
S3_SECRET_ENV = "MASSIVE_S3_SECRET_ACCESS_KEY"

#: The S3 endpoint of the daily files.
S3_ENDPOINT = "https://files.massive.com"

#: Bytes read from the store per chunk.
_CHUNK = 1 << 20

#: S3 error codes of a request Massive refused.
_FORBIDDEN = ("401", "403", "AccessDenied", "Forbidden")

#: S3 error codes of credentials Massive does not know.
_BAD_CREDENTIALS = ("InvalidAccessKeyId", "SignatureDoesNotMatch")

#: Days before today within which a 403 on a month with nothing listed yet
#: is taken as not published yet, rather than outside the plan's window.
_RECENT_DAYS = 7

#: S3 error codes of throttling; retried like any transport error.
_THROTTLING = ("SlowDown", "503", "429", "Throttling", "ThrottlingException", "RequestLimitExceeded")


class MassiveCredentialError(RuntimeError):
    """A credential variable is not set."""


class MassiveEntitlementError(RuntimeError):
    """Massive refused the key (HTTP 401 or 403), or the day is outside the plan's window."""


class MassiveNotPublishedError(LookupError):
    """Massive has no file for that day: not a trading day, or not published yet."""


class MassiveObjectChangedError(RuntimeError):
    """The object was replaced while its ranges were fetched (S3 answered 412 to ``If-Match``).

    Not transient for the range: the download of the day starts over from a
    fresh HEAD, ``retries`` times.
    """


class MassiveTransportError(TransientError):
    """The connection failed or broke, Massive throttled, or a downloaded file failed its checks; retried.

    A ``TransientError`` (a ``ConnectionError``) of the request policy
    (``quantlab.acquisition._support.request_policy``).
    """


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
        api_key, access_key_id = require_env(
            (API_KEY_ENV, S3_ACCESS_KEY_ENV),
            hint=(
                f"set them in the environment (the S3 secret {S3_SECRET_ENV} defaults to "
                f"{API_KEY_ENV}). Nothing was downloaded."
            ),
            error=MassiveCredentialError,
            environ=environ,
        )
        return cls(
            api_key=api_key,
            s3_access_key_id=access_key_id,
            s3_secret_access_key=environ.get(S3_SECRET_ENV) or api_key,
        )


class ObjectStore(Protocol):
    """The S3 seam: the size of a key, and the bytes of a range of it in chunks.

    Both raise ``MassiveNotPublishedError``, ``MassiveEntitlementError`` or
    ``MassiveTransportError``; ``read`` may raise it after yielding some
    chunks, when its stream breaks. Both are called from several threads.
    ``etag`` names the object ``size`` last read (``None`` when the store
    cannot name it: its download then never resumes), and ``read`` given
    ``if_match`` raises ``MassiveObjectChangedError`` when the object no
    longer has that ETag.
    """

    def size(self, key: str) -> int:
        """Return the object's size in bytes."""

    def etag(self, key: str) -> str | None:
        """Return the ETag ``size`` last read for ``key``, else None."""

    def read(self, key: str, start: int, stop: int, if_match: str | None = None) -> Iterable[bytes]:
        """Return the object's bytes ``start`` up to (not including) ``stop``, in chunks."""


class S3ObjectStore:
    """``ObjectStore`` over Massive's S3 endpoint, signed by boto3.

    Massive answers 404 for a missing day inside the plan's window (a
    holiday) and 403 for every day outside it, at both ends: before the
    oldest day, and after the newest published one (today before it is
    published, the future). On a 403, ``size`` lists the key's month: a day
    after the newest listed day, or a recent day of a month with nothing
    listed yet, is not published yet; any other is outside the plan's
    window; a refused listing means the credentials were refused.

    Parameters
    ----------
    credentials : MassiveCredentials
        The S3 key id and secret.
    max_connections : int, default 16
        The connection pool's size; at least the streams in flight.
    read_timeout : float, default 30
        Seconds a stream may send nothing before it is given up as stalled;
        the range then resumes from the bytes it wrote.
    client : botocore S3 client, optional
        A client to use instead of one made from ``credentials``.
    today : callable, optional
        Today's date in New York; the system clock's when omitted.

    Examples
    --------
    Needs the credentials and the network::

        store = S3ObjectStore(MassiveCredentials.from_env(), max_connections=32)
    """

    def __init__(
        self,
        credentials: MassiveCredentials,
        *,
        max_connections: int = 16,
        read_timeout: float = 30.0,
        client: Any | None = None,
        today: Callable[[], date] | None = None,
    ) -> None:
        """Create the boto3 client; see the class docstring."""
        self._today = today if today is not None else _new_york_today
        # The ETag of each key's last HEAD, for etag().
        self._etags: dict[str, str | None] = {}
        if client is not None:
            self._client = client
            return
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise ImportError(
                "S3ObjectStore needs boto3, an optional extra of quantlab: install it "
                "with `uv sync --extra massive` or `pip install 'quantlab-py[massive]'`."
            ) from exc

        self._client = boto3.session.Session().client(
            "s3",
            endpoint_url=S3_ENDPOINT,
            aws_access_key_id=credentials.s3_access_key_id,
            aws_secret_access_key=credentials.s3_secret_access_key,
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 1},
                read_timeout=read_timeout,
                max_pool_connections=max_connections,
            ),
        )

    @staticmethod
    def _translate(error: Exception, key: str) -> Exception:
        """Return the client's exception for a boto3 one."""
        from botocore.exceptions import BotoCoreError, ClientError

        if isinstance(error, ClientError):
            code = str(error.response.get("Error", {}).get("Code", ""))
            if code in ("412", "PreconditionFailed"):
                return MassiveObjectChangedError(f"Massive replaced {key!r} while it was being downloaded.")
            if code in ("404", "NoSuchKey", "NotFound"):
                return MassiveNotPublishedError(f"Massive has no {key!r} (not a trading day, or not published yet).")
            if code in _FORBIDDEN or code in _BAD_CREDENTIALS:
                return MassiveEntitlementError(
                    f"Massive refused {key!r} ({code}): the S3 credentials were rejected, or the day "
                    f"is outside the plan's window."
                )
            if code in _THROTTLING:
                return MassiveTransportError(f"Massive throttled {key!r} ({code}).")
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
        from botocore.exceptions import ClientError

        try:
            head = self._client.head_object(Bucket=BUCKET, Key=key)
            # Kept for etag(): the same HEAD names the object, no second request.
            self._etags[key] = head.get("ETag")
            return int(head["ContentLength"])
        except ClientError as error:
            code = str(error.response.get("Error", {}).get("Code", ""))
            raise (self._forbidden(key, code) if code in _FORBIDDEN else self._translate(error, key)) from error
        except Exception as error:
            raise self._translate(error, key) from error

    def etag(self, key: str) -> str | None:
        """The ETag ``size`` last read for ``key`` (it changes when the object is replaced), else None.

        The ranged download's identity: a ``.part`` left for another ETag starts over.

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            key = s3_key("trades", date(2016, 11, 25))
            store.size(key), store.etag(key)
        """
        return self._etags.get(key)

    def _forbidden(self, key: str, code: str) -> Exception:
        """Tell a 403 on ``key`` apart by listing its month; see the class docstring."""
        from botocore.exceptions import BotoCoreError, ClientError

        prefix, name = key.rsplit("/", 1)
        day = date.fromisoformat(name[:10])
        try:
            listed = self._client.list_objects_v2(Bucket=BUCKET, Prefix=f"{prefix}/").get("Contents", [])
        except ClientError as error:
            return MassiveEntitlementError(
                f"Massive refused {key!r} ({code}) and the listing of its month "
                f"({error.response.get('Error', {}).get('Code', '')}): the S3 credentials were rejected."
            )
        except (BotoCoreError, OSError) as error:
            return MassiveTransportError(f"{type(error).__name__} listing {prefix!r}: {error}")
        days = [date.fromisoformat(item["Key"].rsplit("/", 1)[1][:10]) for item in listed]
        if (days and day > max(days)) or (not days and day >= self._today() - timedelta(days=_RECENT_DAYS)):
            return MassiveNotPublishedError(f"Massive has not published {key!r} yet.")
        return MassiveEntitlementError(
            f"Massive refused {key!r} ({code}): the day is outside the plan's window "
            f"(before the oldest day it serves)."
        )

    def read(self, key: str, start: int, stop: int, if_match: str | None = None) -> Iterable[bytes]:
        """Yield the bytes ``start`` up to ``stop`` of the object in chunks from one ranged GET.

        With ``if_match`` (an ETag) the GET is conditioned on it: an object
        replaced since raises ``MassiveObjectChangedError`` (S3's 412).

        Examples
        --------
        Needs the credentials and the network::

            store = S3ObjectStore(MassiveCredentials.from_env())
            head = b"".join(store.read(s3_key("trades", date(2016, 11, 25)), 0, 1024))
        """
        try:
            request = {"Bucket": BUCKET, "Key": key, "Range": f"bytes={start}-{stop - 1}"}
            if if_match is not None:
                request["IfMatch"] = if_match
            body = self._client.get_object(**request)["Body"]
            yield from body.iter_chunks(_CHUNK)
        except Exception as error:
            raise self._translate(error, key) from error


#: Seconds to connect to the REST API, and to wait for each read.
REST_TIMEOUT = (30, 120)


def rest_transport(url: str, *, params: Mapping[str, str], headers: Mapping[str, str]) -> HttpResponse:
    """Send one REST GET; the client's default ``http_get``.

    ``quantlab.acquisition._support.http.http_get`` with ``REST_TIMEOUT``,
    following redirects (``requests`` drops ``Authorization`` on a redirect to
    another host), and raising ``MassiveTransportError`` for a connection that
    fails or breaks.

    Examples
    --------
    Needs the network; without a key Massive answers 401::

        rest_transport(CONDITIONS.url, params=CONDITIONS.params, headers={}).status
    """
    return http_get(
        url,
        params=params,
        headers=headers,
        timeout=REST_TIMEOUT,
        error=MassiveTransportError,
        follow_redirects=True,
    )


@dataclass(frozen=True)
class DayDownload:
    """What ``MassiveClient.download_days`` did for one trading day.

    Attributes
    ----------
    day : date
        The trading day.
    path : Path or None
        The verified file in the raw tier; ``None`` when Massive has no file
        for the day (a holiday, or not published yet).
    fetched_bytes : int
        Bytes downloaded for it; 0 when the file was already there whole.
    seconds : float
        Wall-clock seconds its download took.

    Examples
    --------
    >>> DayDownload(date(2024, 11, 29), Path("2024-11-29.csv.gz"), 2_300_000_000, 40.0).published
    True
    >>> DayDownload(date(2024, 11, 29), None).published
    False
    """

    day: date
    path: Path | None
    fetched_bytes: int = 0
    seconds: float = 0.0

    @property
    def published(self) -> bool:
        """Whether Massive has a file for the day.

        Examples
        --------
        >>> DayDownload(date(2024, 12, 2), None).published
        False
        """
        return self.path is not None


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
        The REST seam: ``http_get(url, params=..., headers=...)`` returns an
        ``HttpResponse``; ``rest_transport`` when omitted.
    streams : int, default 16
        Byte ranges in flight at once, all of one file; also the S3
        connection pool's size.
    files : int, default 4
        Files ``download_days`` has in flight at once: one being fetched,
        the others being verified (one core each).
    part_bytes : int, default 32 MiB
        Size of one byte range.
    read_timeout : float, default 30
        Seconds a stream may send nothing before it is given up and its
        range resumed; passed to the ``S3ObjectStore`` made when ``store``
        is omitted.
    retries : int, default 5
        Attempts after the first for a transport error; a range whose
        stream breaks after delivering bytes resumes at once and does not
        use one up.
    backoff : float, default 2.0
        Seconds before the first retry, doubled on each later one.
    sleep : callable, default time.sleep
        How the client waits.

    Raises
    ------
    MassiveCredentialError
        If a credential variable is unset.
    ValueError
        If ``streams``, ``files`` or ``part_bytes`` is below 1.

    Examples
    --------
    Needs the credentials and the network::

        client = MassiveClient(streams=32, files=4)
        client.download_day("trades", date(2024, 11, 29), "/data/quantlab/downloads")
    """

    def __init__(
        self,
        *,
        credentials: MassiveCredentials | None = None,
        store: ObjectStore | None = None,
        http_get: Transport | None = None,
        streams: int = 16,
        files: int = 4,
        part_bytes: int = 32 << 20,
        read_timeout: float = 30.0,
        retries: int = 5,
        backoff: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize; see the class docstring."""
        if min(streams, files, part_bytes) < 1:
            raise ValueError(
                f"Massive client needs streams, files and part_bytes of at least 1; "
                f"got streams={streams}, files={files}, part_bytes={part_bytes}."
            )
        self.streams = streams
        self.files = files
        self.part_bytes = part_bytes
        self.credentials = credentials if credentials is not None else MassiveCredentials.from_env()
        self.store = store if store is not None else S3ObjectStore(
            self.credentials, max_connections=streams, read_timeout=read_timeout
        )
        self.http_get = http_get if http_get is not None else rest_transport
        self.retries = retries
        self.backoff = backoff
        self.sleep = sleep

    @property
    def _policy(self) -> RequestPolicy:
        """The request policy: retries, back-off, and the credentials scrubbed and kept out of URLs."""
        return RequestPolicy(
            {
                API_KEY_ENV: self.credentials.api_key,
                S3_ACCESS_KEY_ENV: self.credentials.s3_access_key_id,
                S3_SECRET_ENV: self.credentials.s3_secret_access_key,
            },
            retries=self.retries,
            backoff=self.backoff,
            sleep=self.sleep,
            vendor="Massive",
            credential=API_KEY_ENV,
            refusal_error=MassiveEntitlementError,
        )

    def _retrying(self, what: str, attempt: Callable[[], object]):
        """Run ``attempt``, retrying a ``MassiveTransportError`` after a doubling back-off."""
        return self._policy.call(
            attempt, lambda tries, wait, error: f"Massive: {what} failed ({error}); retrying in {wait:g}s."
        )

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
        return self._download(data_type, day, download_root, threading.Lock(), threading.Event())[0]

    def published_size(self, data_type: str, day: date) -> int:
        """Return the size of one data type's file for one trading day, downloading nothing.

        One HEAD request, not retried: a caller probing which days the plan
        serves (the oldest day of its rolling window) tells the answers apart
        by the exception.

        Parameters
        ----------
        data_type : str
            ``"trades"``, ``"minute_aggs"`` or ``"day_aggs"``.
        day : date
            The trading day.

        Returns
        -------
        int
            The file's size in bytes.

        Raises
        ------
        MassiveNotPublishedError
            If Massive has no file for the day.
        MassiveEntitlementError
            If Massive refused the credentials or the day (outside the plan's window).
        MassiveTransportError
            If the request failed.

        Examples
        --------
        Needs the credentials and the network::

            client.published_size("trades", date(2016, 11, 25))  # 217911421
        """
        return self.store.size(s3_key(data_type, day))

    def download_days(
        self, data_type: str, start: date, end: date, download_root: str | Path
    ) -> Iterator[DayDownload]:
        """Download one data type's files over the XNYS sessions of a date range, oldest first.

        ``files`` files are in flight at once, one being fetched with all
        ``streams`` byte ranges while the others are verified, and the days are yielded in date order as they finish, so a
        caller converting day d while d+1 downloads keeps only a few files
        ahead: a day is started only once the caller has moved past the day
        ``files`` before it, so however slow the caller, the day it holds and
        at most ``files - 1`` after it are on disk. Only XNYS sessions are asked for: weekends and holidays are
        not. A session Massive has no file for (not published yet) is
        yielded unpublished and skipped. Each published day moves the data
        type's watermark to it (``raw.read_watermark``) while no earlier day
        of the run is missing, and a later run starts after the watermark:
        a day not published yet holds it, so a later run asks for that day
        again. The watermark says what was downloaded, not what was
        converted; a caller converting the files keeps its own record.

        Parameters
        ----------
        data_type : str
            ``"trades"``, ``"minute_aggs"`` or ``"day_aggs"``.
        start, end : date
            The first and last day, inclusive; days on or before the
            watermark are left out.
        download_root : str or Path
            The download directory; the files land under its ``massive/``.

        Yields
        ------
        DayDownload
            One per XNYS session after the watermark, in date order.

        Raises
        ------
        MassiveEntitlementError
            If Massive refused the credentials or a day; the run stops, and
            the watermark stays on the last day before it.
        MassiveTransportError
            If a transfer kept failing; likewise.

        Examples
        --------
        Needs the credentials and the network; 2016-11-24 (Thanksgiving) is
        not asked for::

            client = MassiveClient()
            for done in client.download_days("trades", date(2016, 11, 21), date(2016, 11, 25),
                                             "/data/quantlab/downloads"):
                print(done.day, done.published, done.fetched_bytes)
        """
        vendor_root = Path(download_root) / VENDOR_DIR
        watermark = read_watermark(vendor_root, data_type)
        if holds(watermark, start):
            logger.info(f"Massive: {data_type} downloaded through {watermark}; resuming after it.")
        start = resume_from(watermark, start)
        days = xnys_sessions(start, end)
        # One file is fetched at a time, with every stream; verifying is not
        # serialized, so the network stays busy while earlier files decode.
        fetching = threading.Lock()
        # Set when the run ends early (a failure, or the caller stopping), so
        # every range in flight, of every file, stops at its next chunk.
        abort = threading.Event()

        # joblib dispatches the next day when a worker finishes, not when the
        # caller takes a result, so a day waits here until the caller has
        # moved past every day ``files`` or more before it: the day being
        # converted and at most ``files - 1`` after it are on disk.
        handed = threading.Condition()
        consumed = 0

        def moved_past() -> None:
            nonlocal consumed
            with handed:
                consumed += 1
                handed.notify_all()

        def attempt(index: int, day: date) -> tuple[date, tuple[Path, int, float] | None]:
            with handed:
                while index >= consumed + self.files and not abort.is_set():
                    handed.wait(timeout=1.0)
            if abort.is_set():
                raise DownloadAborted(f"Massive: {data_type} {day}")
            try:
                return day, self._timed(data_type, day, download_root, fetching, abort)
            except MassiveNotPublishedError:
                return day, None

        # Threads, as every fan-out in quantlab: the files wait on the network.
        # Results come back in date order.
        outcomes = Parallel(
            n_jobs=self.files, backend="threading", return_as="generator", pre_dispatch="n_jobs", batch_size=1
        )(delayed(attempt)(index, day) for index, day in enumerate(days))
        complete = True
        try:
            for day, outcome in outcomes:
                if outcome is None:
                    complete = False
                    logger.info(f"Massive: no {data_type} file for {day} yet (not published).")
                    yield DayDownload(day, None)
                    moved_past()
                    continue
                path, fetched, seconds = outcome
                if complete:
                    write_day_file_watermark(vendor_root, data_type, day)
                yield DayDownload(day, path, fetched, seconds)
                moved_past()
        finally:
            abort.set()
            with handed:
                handed.notify_all()

    def _timed(
        self, data_type: str, day: date, download_root: str | Path, fetching: threading.Lock, abort: threading.Event
    ) -> tuple[Path, int, float]:
        """``_download`` and the wall-clock seconds it took."""
        started = time.monotonic()
        path, fetched = self._download(data_type, day, download_root, fetching, abort)
        return path, fetched, time.monotonic() - started

    def _download(
        self, data_type: str, day: date, download_root: str | Path, fetching: threading.Lock, abort: threading.Event
    ) -> tuple[Path, int]:
        """Make one day's file present and verified; return it and the bytes fetched.

        Its ranges are fetched by ``download_ranges`` while holding
        ``fetching``; setting ``abort`` stops them at their next chunk. A
        file that is not a whole gzip is downloaded again, ``retries`` times,
        after a back-off; an object replaced mid-download (each ranged GET is
        conditioned on the ETag its HEAD read) starts over at once from a
        fresh HEAD, within the same ``retries``.
        """
        key = s3_key(data_type, day)
        target = raw_file(Path(download_root) / VENDOR_DIR, data_type, day)
        expected = self._retrying(f"HEAD {key}", lambda: self.store.size(key))
        if target.exists() and target.stat().st_size == expected:
            return target, 0
        target.parent.mkdir(parents=True, exist_ok=True)
        # The object's name (S3's ETag) lets a .part left by an earlier run resume.
        identity = self.store.etag(key)
        started = time.monotonic()
        policy = self._policy
        for tries in range(self.retries + 1):
            try:
                download_ranges(
                    target,
                    expected,
                    lambda start, stop, identity=identity: self.store.read(key, start, stop, if_match=identity),
                    workers=self.streams,
                    part_bytes=self.part_bytes,
                    policy=policy,
                    what=f"Massive: {key}",
                    error=MassiveTransportError,
                    verify=lambda partial: _check_gzip(partial, key),
                    fetch_lock=fetching,
                    abort=abort,
                    identity=identity,
                )
                break
            except MassiveObjectChangedError as error:
                if tries == self.retries:
                    raise
                logger.warning(f"Massive: {error} Starting it over from a fresh HEAD.")
                expected = self._retrying(f"HEAD {key}", lambda: self.store.size(key))
                identity = self.store.etag(key)
            except _NotWholeGzip as error:
                if tries == self.retries:
                    raise
                policy.back_off(tries, lambda wait, error=error: f"Massive: {error}; downloading it again in {wait:g}s.")
        elapsed = max(time.monotonic() - started, 1e-9)
        logger.info(f"Massive: {key} ({expected / 1e6:.1f} MB) fetched and verified in {elapsed:.0f}s.")
        return target, expected

    def _get_json(self, url: str, params: Mapping[str, str], *, missing_ok: bool = False) -> dict | None:
        """GET ``url`` with the key in a header; retry throttling, 5xx and transport errors.

        With ``missing_ok`` a 404 returns ``None`` instead of raising.
        """
        policy = self._policy
        headers = {"Authorization": f"Bearer {self.credentials.api_key}"}
        # Never the query: a next_url carries a cursor, and logs print this.
        where = url.split("?")[0]

        def attempt() -> dict | None:
            response = self.http_get(url, params=params, headers=headers)
            body = response.read()
            if missing_ok and response.status == 404:
                return None
            if response.status in REFUSAL_STATUSES:
                raise policy.refusal(response.status, where=where, detail=_vendor_error(body))
            # Every other failure is retried, a 4xx too: Massive's own words go in the message.
            if response.status == 429 or response.status >= 500:
                raise MassiveTransportError(
                    f"HTTP {response.status} on {where}", retry_after=retry_after(response.headers)
                )
            if response.status != 200:
                raise MassiveTransportError(
                    policy.scrub(f"HTTP {response.status} on {where}: {_vendor_error(body)}")
                )
            try:
                return json.loads(body)
            except ValueError as error:
                raise MassiveResponseError(f"{where} answered 200 with a body that is not JSON: {error}") from error

        return self._retrying(f"GET {where}", attempt)

    def pull(
        self,
        endpoint: RestEndpoint | str,
        download_root: str | Path,
        *,
        end: date | None = None,
        today: date | None = None,
    ) -> RestPull:
        """Pull one REST endpoint's new rows into the raw tier.

        The endpoint's pull shape says what is asked for and kept
        (``quantlab.acquisition.massive.shapes``). Every ``next_url`` is
        followed after any key in it is dropped, and a watermark never passes
        yesterday in New York and never moves back.

        Parameters
        ----------
        endpoint : RestEndpoint or str
            The endpoint, or its name in ``rest.ENDPOINTS``.
        download_root : str or Path
            The download directory; pages land under its ``massive/rest/``.
        end : date, optional
            The last day asked for; no upper bound when omitted.
        today : date, optional
            Today in New York; the system clock's when omitted.

        Returns
        -------
        RestPull
            The pages and rows written and the watermark after the pull.

        Raises
        ------
        ValueError
            If ``endpoint`` is a name not in ``rest.ENDPOINTS``.
        MassiveEntitlementError
            If Massive refused the key (401) or the endpoint (403), naming
            it; the pages written before stay, and so does the watermark.
        MassiveTransportError
            If a page kept failing after ``retries`` back-offs; likewise, so
            the next pull resumes from the watermark.
        MassiveResponseError
            If a page is not JSON, a record lacks the field the watermark or
            the id it was asked by needs, or a condition table page holds no
            ``results``; likewise.

        Examples
        --------
        Needs the credentials and the network::

            client = MassiveClient()
            done = client.pull("dividends", "/data/quantlab/downloads")
            print(done.rows, done.through)
        """
        if isinstance(endpoint, str):
            if endpoint not in ENDPOINTS:
                raise ValueError(f"Unknown Massive REST endpoint {endpoint!r}; expected one of {list(ENDPOINTS)}.")
            endpoint = ENDPOINTS[endpoint]
        return endpoint.pull(PullRun(
            self._get_json,
            self._policy.keyless,
            Path(download_root) / VENDOR_DIR,
            today if today is not None else _new_york_today(),
            end,
        ))

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
        MassiveResponseError
            If a page is not JSON or holds no ``results``; no file is written.

        Examples
        --------
        Needs the credentials and the network (94 records on 2026-10-09)::

            path = client.condition_table("/data/quantlab/downloads")
        """
        return self.pull(CONDITIONS, download_root).files[0]


def _new_york_today() -> date:
    """Return today's date in New York."""
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date()


class _NotWholeGzip(MassiveTransportError):
    """A downloaded file that does not decode to the end of its gzip; it is downloaded again."""


def _check_gzip(path: Path, key: str) -> None:
    """Decode ``path`` to its end; raise ``_NotWholeGzip`` if it is not a whole gzip."""
    try:
        with gzip.open(path, "rb") as handle:
            while handle.read(_CHUNK * 16):
                pass
    except (OSError, EOFError, zlib.error) as error:
        raise _NotWholeGzip(f"{key}: not a whole gzip ({type(error).__name__}: {error})") from error


def _vendor_error(body: bytes) -> str:
    """Return the vendor's ``error`` text from a JSON body, else the body's start."""
    try:
        payload = json.loads(body)
        return str(payload.get("error") or payload.get("message") or "")
    except (ValueError, AttributeError):
        return body[:200].decode("utf-8", "replace")
