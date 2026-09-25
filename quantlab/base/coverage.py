"""Coverage bookkeeping over the per-symbol watermark sidecars.

Every acquisition run leaves a small JSON sidecar per symbol under the
config's ``watermark_path`` recording the date range already on disk, plus a
``_failures.json`` manifest of the symbols that did not land.
``CoverageLedger`` reads those files and answers the one question a refresh
needs: which of the requested symbols still have to be fetched for the
requested window. It is the single implementation of that rule. The
acquisition engine in ``quantlab/base/acquisition.py`` composes a ledger and
delegates to it, and the credential-free source inspector builds one directly
through ``CoverageLedger.for_config``, so both report the same answer by
construction.

This module reads local files only, issues no vendor requests and needs no
credentials. It imports nothing from the acquisition side, which is what lets
a machine with no API key browse what it already has on disk.
"""

import json
from pathlib import Path
from typing import Iterator, Sequence

from quantlab.base.config import AcquisitionConfig
from quantlab.enums.data import RAW_HIVE_KEYS, TRADEABLE_TICKER_PATTERN

#: Filename of the per-run failure manifest, written under ``watermark_root``
#: next to the per-symbol sidecars. Declared here and bound onto the
#: acquisition engine as a class constant, so the writer and the
#: credential-free reader name the same file.
FAILURE_MANIFEST_NAME = "_failures.json"

#: Subdirectory under ``watermark_root`` that holds the per-batch page
#: ledgers. Named here so ``iter_watermark_symbols`` can skip it explicitly
#: instead of reporting it as a symbol.
PAGE_LEDGER_DIR_NAME = "_pages"

#: Accepted values of ``config.kwargs["legacy_watermarks"]``. ``"warn"`` skips
#: a sidecar with no recorded covered start but reports it on every run;
#: ``"refetch"`` treats unknown coverage as uncovered. See
#: ``CoverageLedger.coverage_status`` for why ``"warn"`` is the default.
#: Declared here rather than on the acquisition engine so that
#: ``CoverageLedger.for_config`` can use the same policy set without
#: importing it; the engine re-exposes both as class constants.
LEGACY_WATERMARK_POLICIES = ("warn", "refetch")
DEFAULT_LEGACY_WATERMARK_POLICY = "warn"

#: The well-formedness rule a symbol must satisfy before it becomes a
#: filesystem path segment or a query-string value. This is the same compiled
#: pattern the roster builder filters on, bound rather than re-declared so the
#: two cannot drift.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


class CoverageLedger:
    """Read-only view of what is already on disk for one acquisition config.

    The ledger reads the per-symbol watermark sidecars and the failure
    manifest under ``watermark_root`` and classifies each requested symbol
    against the config's ``[start_date, end_date]`` window. It is the one
    place that comparison is made; every other component asks the ledger
    rather than re-implementing the rule, because the four-way classification
    (``uncovered``, ``covered``, ``widened``, ``legacy``) is easy to get
    subtly wrong.

    ``data_type`` is passed in rather than resolved from a vendor class, so a
    ledger can be built without constructing a vendor client and therefore
    without credentials; ``for_config`` does exactly that.

    Parameters
    ----------
    config : AcquisitionConfig
        The acquisition config whose watermark tree to read.
    data_type : str | None
        Raw-tier data type (for example ``"quotes"`` or
        ``"trades"``) for frequencies whose hive layout partitions on
        one; ``None`` otherwise.
    legacy_policies : Sequence[str]
        Accepted values of the ``legacy_watermarks`` knob.
    default_legacy_policy : str
        Policy used when the knob is unset.
    owner_label : str
        Name reported in ``validate_symbols`` error messages.

    Examples
    --------
    With sidecars on disk for ``AAPL`` (fully covered), ``MSFT`` (no
    recorded start) and ``NVDA`` (recorded start later than requested,
    carrying the ``no_data`` marker), and none for ``TSLA``:

    >>> ledger = CoverageLedger.for_config(config)
    >>> pending, counts = ledger.partition_by_coverage(
    ...     ["AAPL", "MSFT", "NVDA", "TSLA"], from_watermark=False
    ... )
    >>> pending
    ['NVDA', 'TSLA']
    >>> counts
    {'covered': 1, 'widened': 1, 'legacy': 1, 'no_data': 1}

    The method examples below continue from this ledger.
    """

    def __init__(
        self,
        config: AcquisitionConfig,
        *,
        data_type: str | None = None,
        legacy_policies: Sequence[str] = LEGACY_WATERMARK_POLICIES,
        default_legacy_policy: str = DEFAULT_LEGACY_WATERMARK_POLICY,
        owner_label: str = "CoverageLedger",
    ) -> None:
        """Store the config and policy values; nothing is read from disk yet."""
        self.config = config
        self.data_type = data_type
        self.legacy_policies = tuple(legacy_policies)
        self.default_legacy_policy = default_legacy_policy
        self.owner_label = owner_label

    def __repr__(self) -> str:
        """Return the vendor, frequency and data type."""
        return (
            f"{self.__class__.__name__}(vendor={self.config.vendor!r}, "
            f"frequency={self.config.frequency!r}, "
            f"data_type={self.data_type!r})"
        )

    # -- construction from a bare config (the inspector's entry point) ------

    @classmethod
    def for_config(
        cls,
        config: AcquisitionConfig,
        *,
        owner_label: str = "SourceInspector",
    ) -> "CoverageLedger":
        """Build a ledger from a config alone, without any vendor class.

        ``data_type`` is read from ``config.kwargs`` only when the frequency's
        hive layout partitions on it (tick data), mirroring the condition in
        ``watermark_root``, so bar-frequency ledgers built this way carry
        ``data_type=None`` and their sidecar paths match those of a ledger
        the acquisition engine composes. The value is not validated against a
        vendor's supported data types, because the ledger has no vendor; an
        invalid value simply names a directory that does not exist, so every
        symbol reads back as uncovered.

        Parameters
        ----------
        config : AcquisitionConfig
            The acquisition config whose watermark tree to read.
        owner_label : str
            Name reported in ``validate_symbols`` error messages.

        Raises
        ------
        ValueError
            If the frequency partitions on ``data_type`` and the
            config does not set it.

        Examples
        --------
        >>> CoverageLedger.for_config(config)
        CoverageLedger(vendor='tiingo', frequency='1d', data_type=None)
        """
        return cls(
            config,
            data_type=cls._resolve_data_type(config),
            owner_label=owner_label,
        )

    @staticmethod
    def _resolve_data_type(config: AcquisitionConfig) -> str | None:
        """Return ``config.kwargs["data_type"]`` where the layout uses it.

        Returns None for frequencies whose hive layout has no ``data_type``
        key. There is deliberately no default when the key is required: the
        data types share one vendor root and their sidecars are namespaced by
        data type, so guessing would let a completed backfill of one type
        tell a run for another that every symbol is already covered.

        Raises
        ------
        ValueError
            If the layout uses ``data_type`` and it is unset.
        """
        if "data_type" not in RAW_HIVE_KEYS[config.frequency]:
            return None
        data_type = (config.kwargs or {}).get("data_type")
        if not data_type:
            raise ValueError(
                f"CoverageLedger: frequency {config.frequency!r} partitions on "
                f"`data_type`, so kwargs['data_type'] must be set (e.g. "
                f"'quotes' or 'trades'); got {data_type!r}. There is "
                f"deliberately NO default -- the two land under one vendor "
                f"root and their watermark sidecars are namespaced by it, so "
                f"guessing here would let a completed quotes backfill tell a "
                f"trades run that every symbol is already covered."
            )
        return str(data_type)

    # -- knobs --------------------------------------------------------------

    def _knob(self, name: str, default=None):
        """Read a per-run tuning parameter from ``config.kwargs``."""
        return (self.config.kwargs or {}).get(name, default)

    # -- sidecar paths ------------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """Return the hive partition keys for this config's frequency."""
        return RAW_HIVE_KEYS[self.config.frequency]

    @property
    def watermark_root(self) -> Path:
        """Return ``config.watermark_path``, namespaced by data type if needed.

        Tick-frequency quotes and trades share one vendor raw root and are
        separated only by the leading ``data_type=`` hive key, but their
        sidecars carry no such key. Without this namespacing a completed
        quotes backfill would tell a later trades run that every symbol is
        covered, and that run would skip the whole roster. Frequencies whose
        layout has no ``data_type`` key are unaffected.

        Examples
        --------
        >>> ledger.watermark_root == Path(config.watermark_path)
        True
        >>> CoverageLedger.for_config(tick_config).watermark_root.name
        quotes
        """
        root = Path(self.config.watermark_path)
        if "data_type" in self._hive_keys:
            root = root / str(self.data_type)
        return root

    def watermark_path(self, symbol: str) -> Path:
        """Return the watermark sidecar path for ``symbol``.

        Examples
        --------
        >>> ledger.watermark_path("AAPL").name
        AAPL.json
        """
        return self.watermark_root / f"{symbol}.json"

    @property
    def failure_manifest_path(self) -> Path:
        """Return the path of ``_failures.json`` for this config.

        Defined on the ledger so the reader and the writer share one path
        expression, including the data-type namespacing.

        Examples
        --------
        >>> ledger.failure_manifest_path.name
        _failures.json
        """
        return self.watermark_root / FAILURE_MANIFEST_NAME

    def read_failure_manifest(self) -> dict[str, str]:
        """Return every symbol the failure manifest records as failing.

        The manifest accumulates across runs: before overwriting it, the
        writer folds forward the entries of symbols the run never attempted,
        so it can name symbols no recent run requested. A missing or
        unreadable file reads back as ``{}``, the same tolerance
        ``read_sidecar`` applies. The reasons were scrubbed of credentials
        when they were written, so this method adds no new path for raw
        vendor exception text.

        Returns
        -------
        dict[str, str]
            A ``{symbol: reason}`` mapping, empty when nothing is recorded.

        Examples
        --------
        >>> ledger.read_failure_manifest()
        {'GOOG': 'HTTP 500'}
        """
        path = self.failure_manifest_path
        if not path.exists():
            return {}
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(symbol): str(reason) for symbol, reason in payload.items()}

    def iter_watermark_symbols(self) -> Iterator[str]:
        """Yield, in sorted order, every symbol with a watermark sidecar.

        Only the top level of ``watermark_root`` is scanned, and the failure
        manifest and the page-ledger directory are skipped explicitly so that
        neither is reported as a symbol.

        Examples
        --------
        >>> list(ledger.iter_watermark_symbols())
        ['AAPL', 'MSFT', 'NVDA']
        """
        root = self.watermark_root
        if not root.exists():
            return
        skip = {FAILURE_MANIFEST_NAME, PAGE_LEDGER_DIR_NAME}
        for path in sorted(root.glob("*.json")):
            if path.name in skip or path.stem in skip:
                continue
            if not path.is_file():
                continue
            yield path.stem

    # -- sidecar reads ------------------------------------------------------

    def read_sidecar(self, symbol: str) -> dict | None:
        """Return a sidecar's raw JSON, or None if it is absent or unreadable.

        This is the single tolerant read that ``read_watermark`` and
        ``read_coverage`` share, so there is exactly one failure policy for a
        corrupt sidecar.

        Examples
        --------
        >>> ledger.read_sidecar("AAPL")
        {'start_date': '2024-01-01', 'last_date': '2024-01-31'}
        >>> ledger.read_sidecar("TSLA") is None
        True
        """
        path = self.watermark_path(symbol)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            # A corrupt or half-written sidecar must never crash a refresh;
            # treating it as absent costs at most a wider re-fetch. Files
            # written before sidecar writes became atomic may still be on
            # disk, so do not turn this into a raise.
            return None
        return payload if isinstance(payload, dict) else None

    def read_watermark(self, symbol: str) -> str | None:
        """Return the last covered date for ``symbol``, or None.

        This is what an incremental refresh uses to compute its start date;
        the covered start is not needed there.

        Examples
        --------
        >>> ledger.read_watermark("AAPL")
        '2024-01-31'
        >>> ledger.read_watermark("TSLA") is None
        True
        """
        payload = self.read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def read_coverage(self, symbol: str) -> dict | None:
        """Return the covered range for ``symbol``, or None without a sidecar.

        The result is ``{"start_date", "last_date", "no_data"}``. Either date
        may be None. A sidecar written by an older layout records only
        ``last_date`` and reads back with ``start_date=None``; nothing fills
        that in from ``config.start_date``, because only the user knows what
        window those files were fetched over and an invented start would
        silently hide a per-symbol history gap. Stamping the covered start is
        an explicit user step on the acquisition engine. ``no_data`` defaults
        to False when the key is absent, which is correct for sidecars written
        before the marker existed, since those were only written after a
        successful fetch.

        Examples
        --------
        >>> ledger.read_coverage("AAPL")
        {'start_date': '2024-01-01', 'last_date': '2024-01-31', 'no_data': False}
        >>> ledger.read_coverage("MSFT")
        {'start_date': None, 'last_date': '2024-01-31', 'no_data': False}
        """
        payload = self.read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
            "no_data": bool(payload.get("no_data", False)),
        }

    # -- the coverage rule (exactly one implementation) ---------------------

    def legacy_policy(self) -> str:
        """Return the ``legacy_watermarks`` policy in force.

        Raises
        ------
        ValueError
            If the knob is set to a value outside
            ``legacy_policies``.

        Examples
        --------
        >>> ledger.legacy_policy()
        warn
        """
        policy = self._knob("legacy_watermarks", self.default_legacy_policy)
        if policy not in self.legacy_policies:
            raise ValueError(
                f"legacy_watermarks={policy!r} is not one of "
                f"{list(self.legacy_policies)}."
            )
        return policy

    def coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Classify ``symbol`` against the requested window.

        Returns one of ``"uncovered"``, ``"covered"``, ``"widened"`` or
        ``"legacy"``; see ``classify_coverage`` for the rule. Dates are ISO
        ``YYYY-MM-DD`` strings compared lexically, so no parsing or time zone
        is involved.

        Parameters
        ----------
        symbol : str
            The symbol whose sidecar to read.
        from_watermark : bool
            True for an incremental refresh, which requests
            ``[watermark, end_date]`` per symbol rather than
            ``config.start_date``. Only the end date is then checked:
            judging a refresh against a widened start would mark every
            symbol pending on every run while the refresh could never
            close the gap. Widening the covered range is the job of a
            full download.

        Examples
        --------
        >>> ledger.coverage_status("AAPL")
        covered
        >>> ledger.coverage_status("NVDA")
        widened
        >>> ledger.coverage_status("MSFT")
        legacy
        >>> ledger.coverage_status("MSFT", from_watermark=True)
        covered
        """
        return self.classify_coverage(self.read_coverage(symbol), from_watermark)

    def classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """Apply the coverage rule to an already-read coverage dict.

        This is the only place ``last_date`` and ``start_date`` are compared
        against the config's window. A symbol is ``"covered"`` when its
        recorded ``last_date`` equals ``config.end_date`` and its recorded
        start is known and no later than ``config.start_date``;
        ``"uncovered"`` when there is no sidecar or the end date differs;
        ``"widened"`` when the end date matches but the recorded start is
        later than requested, so the symbol's history is shallower than asked
        for and must be re-fetched; and ``"legacy"`` when the end date
        matches but the covered start is unknown.

        Legacy sidecars are neither assumed to cover the request nor
        re-fetched outright by default: an assumed start would hide a real
        gap, and re-fetching every already-downloaded symbol burns a whole
        quota window. Instead they count as covered for skipping purposes and
        are reported on every run until stamped; ``legacy_watermarks="refetch"``
        opts into the re-fetch.

        The rule ignores ``coverage["no_data"]`` on purpose. The marker
        records what the vendor said about one window, not a verdict about
        the symbol, so a later request for a deeper window can still reach
        the vendor.

        Parameters
        ----------
        coverage : dict | None
            A dict from ``read_coverage``, or None.
        from_watermark : bool
            See ``coverage_status``.

        Examples
        --------
        >>> ledger.classify_coverage(
        ...     {"start_date": "2024-01-15", "last_date": "2024-01-31",
        ...      "no_data": False}
        ... )
        widened
        >>> ledger.classify_coverage(None)
        uncovered
        """
        if coverage is None or coverage["last_date"] != self.config.end_date:
            return "uncovered"
        if from_watermark:
            return "covered"
        if coverage["start_date"] is None:
            return "legacy"
        if coverage["start_date"] <= self.config.start_date:
            return "covered"
        return "widened"

    def covers(self, symbol: str, from_watermark: bool = False) -> bool:
        """Return whether ``symbol`` may be skipped for the requested window.

        A ``"legacy"`` symbol is skipped only under the ``"warn"`` policy.

        Examples
        --------
        >>> ledger.covers("AAPL")
        True
        >>> ledger.covers("NVDA")
        False
        """
        status = self.coverage_status(symbol, from_watermark)
        if status == "legacy":
            return self.legacy_policy() == "warn"
        return status == "covered"

    def partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split ``requested`` into the symbols still to fetch, plus counts.

        Each sidecar is read exactly once. ``no_data`` is counted alongside
        the status rather than as a status of its own, so a run can report
        how many symbols the vendor had nothing for separately from how many
        failed.

        Parameters
        ----------
        requested : list[str]
            Symbols in the order they were requested.
        from_watermark : bool
            See ``coverage_status``.

        Returns
        -------
        tuple[list[str], dict[str, int]]
            ``(pending, counts)`` where ``pending`` preserves request order
            and ``counts`` has the keys ``covered``, ``widened``, ``legacy``
            and ``no_data``. Legacy symbols are included in ``pending`` only
            under the ``"refetch"`` policy.

        Examples
        --------
        >>> pending, counts = ledger.partition_by_coverage(
        ...     ["AAPL", "MSFT", "NVDA", "TSLA"], from_watermark=False
        ... )
        >>> pending
        ['NVDA', 'TSLA']
        >>> counts
        {'covered': 1, 'widened': 1, 'legacy': 1, 'no_data': 1}
        """
        legacy_is_skipped = self.legacy_policy() == "warn"
        pending: list[str] = []
        counts = {"covered": 0, "widened": 0, "legacy": 0, "no_data": 0}

        for symbol in requested:
            coverage = self.read_coverage(symbol)
            status = self.classify_coverage(coverage, from_watermark)
            # Counted alongside the status, not as a status: the marker
            # describes one window, not the symbol.
            if coverage is not None and coverage["no_data"]:
                counts["no_data"] += 1
            if status == "covered":
                counts["covered"] += 1
                continue
            if status == "legacy":
                counts["legacy"] += 1
                if legacy_is_skipped:
                    continue
            elif status == "widened":
                counts["widened"] += 1
            pending.append(symbol)

        return pending, counts

    # -- symbol validation --------------------------------------------------

    def validate_symbols(self, symbols: Sequence[str]) -> list[str]:
        """Reject any symbol that is not a well-formed ticker.

        A thin wrapper over the module-level ``validate_symbols`` for callers
        that already hold a ledger; the free function exists for readers that
        hold only a dataset config.

        Returns
        -------
        list[str]
            The validated symbols as strings, in the given order.

        Raises
        ------
        ValueError
            On the first symbol that does not match the pattern.

        Examples
        --------
        >>> ledger.validate_symbols(["AAPL", "BRK-B"])
        ['AAPL', 'BRK-B']
        >>> ledger.validate_symbols(["../etc"])
        Traceback (most recent call last):
            ...
        ValueError: SourceInspector: refusing to fetch '../etc' -- ...
        """
        return validate_symbols(
            symbols,
            owner_label=self.owner_label,
            raw_root=self.config.raw_data_dir_path,
        )


def validate_symbols(
    symbols: Sequence[str], *, owner_label: str, raw_root: str
) -> list[str]:
    """Reject any symbol that is not a well-formed ticker.

    Called before any filesystem path or vendor query is built from a
    caller-supplied symbol, because a symbol crosses two trust boundaries at
    once: it becomes a path component under the raw root, where ``/`` or
    ``..`` would escape the root, and it becomes one element of a
    comma-joined ``symbols=`` query parameter, where an embedded comma would
    silently change which symbols were requested. The pattern is the same
    compiled object the roster builder filters on, so any symbol the builder
    persists is admitted here. It allows digits and up to two hyphenated
    suffix segments, both of which real tickers use.

    Parameters
    ----------
    symbols : Sequence[str]
        The symbols to check.
    owner_label : str
        Name reported at the front of the error message.
    raw_root : str
        The raw data directory named in the error message.

    Returns
    -------
    list[str]
        The symbols as strings, in the given order.

    Raises
    ------
    ValueError
        On the first symbol that does not match the pattern.

    Examples
    --------
    >>> validate_symbols(["AAPL", "BRK-B"], owner_label="Inspector",
    ...                  raw_root="/data/raw")
    ['AAPL', 'BRK-B']
    >>> validate_symbols(["a,b"], owner_label="Inspector", raw_root="/data/raw")
    Traceback (most recent call last):
        ...
    ValueError: Inspector: refusing to fetch 'a,b' -- ...
    """
    validated = []
    for symbol in symbols:
        text = str(symbol)
        if not _TICKER_PATTERN.match(text):
            raise ValueError(
                f"{owner_label}: refusing to fetch {text!r} -- it does "
                f"not match the well-formed ticker pattern "
                f"{_TICKER_PATTERN.pattern}. A symbol becomes both a "
                f"filesystem path segment under {raw_root} "
                f"and a comma-joined query-string value, so a separator, a "
                f"parent reference or an embedded comma would escape the "
                f"raw root or silently change which symbols were requested. "
                f"Fix the roster rather than relaxing this pattern -- a "
                f"malformed symbol should have been dropped by the "
                f"build-time well-formedness filter in "
                f"quantlab/universe.py:TiingoRosterFetcher.fetch(), so "
                f"reaching here means the reference table predates that "
                f"filter and needs rebuilding."
            )
        validated.append(text)
    return validated
