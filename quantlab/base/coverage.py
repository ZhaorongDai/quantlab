"""Coverage judgement over the local watermark sidecars -- the ONE place the
question "is this symbol already on disk for this window?" is answered.

Extracted from `quantlab/base/acquisition.py` by 03.4-04 under D-09, which
makes it BINDING that the read-only inspector does not re-implement coverage
judgement. `Acquisition` composes a `CoverageLedger` and delegates every
coverage member to it; `quantlab/acquisition/inspector.py` builds one directly
through `CoverageLedger.for_config`. Both therefore reach the SAME
`partition_by_coverage` function object, which is what makes "shared" provable
by identity rather than by two results that happen to agree today.

**This module is a LEAF on the acquisition side.** It imports only stdlib,
`quantlab/base/config.py` and `quantlab/enums/data.py` -- never
`quantlab/base/acquisition.py` and never anything under `quantlab/acquisition/`.
That constraint is not tidiness: it is the whole reason a credential-free
reader can reach coverage judgement at all. `TiingoAcquisition.__init__` raises
`RuntimeError` without `TIINGO_API_KEY`, so any path that drags an
`Acquisition` subclass in makes browsing local data impossible on an
unconfigured machine (D-08).

Every module name in this file is written in PATH form (`a/b.py`) rather than
dotted form (`a.b`) on purpose: the acceptance check for the rule above is a
literal scan of this source for the dotted spellings, and prose that names the
forbidden import trips it -- the same false positive 03.4-01 and 03.4-03 each
had to undo in a docstring. The structural proof that actually binds is the AST
import resolver in `tests/test_source_inspector.py`, which resolves this file's
imports (relative spellings included, which no substring scan can see) and
asserts the forbidden set is untouched.

Everything here is a pure local file read. It issues zero vendor requests and
reads no credential.
"""

import json
from pathlib import Path
from typing import Iterator, Sequence

from quantlab.base.config import AcquisitionConfig
from quantlab.enums.data import RAW_HIVE_KEYS, TRADEABLE_TICKER_PATTERN

#: The per-run crash-durable record of what did NOT land, written under
#: `watermark_root` alongside the per-symbol watermark sidecars because "what
#: did and did not land" is exactly the same question those sidecars answer
#: (T-0iy-07).
#:
#: Declared HERE rather than on `Acquisition`, and BOUND there
#: (`Acquisition.FAILURE_MANIFEST_NAME = FAILURE_MANIFEST_NAME`), so the writer
#: and the credential-free reader name the same file by construction. A reader
#: that could not import `Acquisition` would otherwise have to re-declare the
#: literal, which is the two-free-to-diverge-copies shape quick task 260907-10t
#: already had to undo once for the ticker pattern.
FAILURE_MANIFEST_NAME = "_failures.json"

#: The subdirectory `PageLedger` keeps its per-batch resume ledgers in, also
#: under `watermark_root`.
#:
#: Named here for exactly one reason: `iter_watermark_symbols` must skip it
#: EXPLICITLY. `Acquisition.stamp_watermarks` gets away with globbing `*.json`
#: because a page ledger and the failure manifest both read back with
#: `last_date is None` and fall out of its loop implicitly -- an inspector that
#: reports "how many symbols have a watermark" has no such filter and would
#: count the manifest as a symbol called `_failures`.
PAGE_LEDGER_DIR_NAME = "_pages"

#: What `config.kwargs["legacy_watermarks"]` may be set to (260906-26o D-04).
#: `"warn"` skips a sidecar with no recorded covered start but reports it on
#: every run; `"refetch"` treats unknown coverage as uncovered. See
#: `CoverageLedger.coverage_status` for why `"warn"` is the default.
#:
#: Declared here and BOUND onto `Acquisition` as class constants, for the same
#: single-definition reason as `FAILURE_MANIFEST_NAME`: `for_config` -- the
#: constructor the inspector uses -- needs the same policy set the real run
#: uses, and it cannot import `Acquisition` to get it. They stay reachable as
#: `Acquisition.LEGACY_WATERMARK_POLICIES` because `ingest_us_equity.py` reads
#: that attribute for its argparse `choices`.
LEGACY_WATERMARK_POLICIES = ("warn", "refetch")
DEFAULT_LEGACY_WATERMARK_POLICY = "warn"

#: The one well-formedness rule a symbol must satisfy before it becomes a
#: filesystem path segment or a query-string value.
#:
#: BOUND, not re-declared -- the SAME compiled object the roster builder
#: (`acquisition/universe.py:TiingoRosterFetcher.fetch`) filters on and the
#: same one `base/acquisition.py` binds. See that module's comment on
#: `_TICKER_PATTERN` for why identity rather than equality is the requirement:
#: two copies of the literal HAD diverged once, and that is the bug quick task
#: 260907-10t fixed.
_TICKER_PATTERN = TRADEABLE_TICKER_PATTERN


class CoverageLedger:
    """Every question about what is already on disk, answerable with NO
    credentials.

    **D-09: this is the one place coverage is judged.** `Acquisition` composes
    a ledger and delegates; `SourceInspector` composes one directly. Any
    `last_date == end_date` or `start_date <=` comparison written OUTSIDE
    `classify_coverage` is the D-09 violation -- the four-state rule
    (`uncovered` / `covered` / `widened` / `legacy`) plus the ORTHOGONAL
    `no_data` count is subtle enough that a "simple" reimplementation gets
    `legacy` wrong, and two answers that agree today are exactly how an
    operator ends up trusting the wrong one.

    `data_type` is passed IN rather than read off a class. That is the whole
    extraction blocker: `Acquisition._data_type` is base-`None` but
    `AlpacaAcquisition` overrides it as an INSTANCE property reading
    `_knob("data_type")` with no default, so a ledger that wanted to resolve it
    itself from a vendor would have to construct that vendor -- which is the
    credential-requiring step D-08 exists to avoid. `for_config` resolves the
    same knob from `config.kwargs` directly, and a test pins the two
    constructions to the same `watermark_root`.

    `legacy_policies` / `default_legacy_policy` are plain values rather than
    class constants read off `Acquisition`, for the same layering reason.

    `owner_label` is the name `validate_symbols`' error message reports.
    `Acquisition` passes `self.class_name`, so that message renders
    byte-identically to its pre-extraction text.
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
        self.config = config
        self.data_type = data_type
        self.legacy_policies = tuple(legacy_policies)
        self.default_legacy_policy = default_legacy_policy
        self.owner_label = owner_label

    def __repr__(self) -> str:
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
        """Build a ledger from a config alone, with NO vendor class involved.

        `data_type` is resolved from `config.kwargs` when -- and ONLY when --
        the frequency's `RAW_HIVE_KEYS` actually include `data_type`, i.e. for
        `tick`. That mirrors `watermark_root`'s own condition exactly, so `1d`
        and `1m` ledgers built this way carry `data_type=None` just as an
        `Acquisition`-composed one does for Tiingo, and their sidecar paths are
        byte-identical.

        **The one deliberate asymmetry, recorded rather than hidden:** this
        resolution does NOT validate the value against a vendor's own
        `TICK_DATA_TYPES`, because the ledger has no vendor. It cannot: knowing
        which tick shapes exist is vendor knowledge, and reaching for it would
        reintroduce the import D-08 forbids. The consequence is bounded and
        benign -- for every VALID configuration this produces the same
        `watermark_root` as the ledger `Acquisition` composes (asserted by
        `tests/test_source_inspector.py::
        test_tick_watermark_roots_agree_between_the_two_ledger_constructors`),
        and an invalid `data_type` simply names a directory that does not
        exist, so every symbol reads back as uncovered rather than as another
        data type's coverage. The dangerous direction -- quotes' watermarks
        answering a trades query -- is closed by applying the namespacing at
        all, which is what this does.
        """
        return cls(
            config,
            data_type=cls._resolve_data_type(config),
            owner_label=owner_label,
        )

    @staticmethod
    def _resolve_data_type(config: AcquisitionConfig) -> str | None:
        """`config.kwargs["data_type"]`, but only where the layout uses it."""
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
        """Read a per-run tuning parameter from `config.kwargs`.

        The same escape hatch `Acquisition._knob` reads, spelled out here
        rather than imported, because importing `Acquisition` for a two-line
        dict lookup is precisely the dependency this module must not have.
        """
        return (self.config.kwargs or {}).get(name, default)

    # -- sidecar paths ------------------------------------------------------

    @property
    def _hive_keys(self) -> tuple[str, ...]:
        """The hive partition key(s) for this config's frequency.

        Read from `enums.data.RAW_HIVE_KEYS`, the SAME mapping
        `dataset/stock.py` and `base/acquisition.py` read, so the writer and
        the reader cannot drift.
        """
        return RAW_HIVE_KEYS[self.config.frequency]

    @property
    def watermark_root(self) -> Path:
        """`config.watermark_path`, namespaced by data type where the raw tier
        partitions on one.

        Tick's quotes and trades share ONE vendor raw root, separated on disk
        only by the leading `data_type=` hive key. Their bookkeeping sidecars
        have no such key -- they are `{symbol}.json`, `_failures.json` and
        `{batch_key}.pages.json` -- so without this namespacing a completed
        quotes backfill's watermarks would tell a subsequent TRADES run that
        every symbol is already covered. That run would skip the entire roster
        and report success having fetched nothing.

        Only a frequency whose `RAW_HIVE_KEYS` actually include `data_type` is
        affected, so `1d` and `1m` sidecar paths are byte-identical to what
        they were, and no existing watermark tree moves.
        """
        root = Path(self.config.watermark_path)
        if "data_type" in self._hive_keys:
            root = root / str(self.data_type)
        return root

    def watermark_path(self, symbol: str) -> Path:
        return self.watermark_root / f"{symbol}.json"

    @property
    def failure_manifest_path(self) -> Path:
        """Where `_failures.json` lives for this config.

        On the ledger rather than on the writer because the credential-free
        reader needs it too and must not reconstruct the path itself -- a
        second path expression is how the tick namespacing gets forgotten on
        one of the two sides.
        """
        return self.watermark_root / FAILURE_MANIFEST_NAME

    def read_failure_manifest(self) -> dict[str, str]:
        """Every symbol `_failures.json` records as failing, accumulated across runs.

        Returned as `{symbol: reason}`, or `{}`. A CROSS-RUN record rather than
        a report on the most recent run: the manifest can name symbols no
        recent run requested at all, because the writer folds unattempted
        entries forward before every overwrite. This summary line is kept
        verbatim in sync with `SourceInspector.failures`; the two return the
        same value, and a drifting pair of summaries is where the next
        divergence starts.

        On the ledger for exactly the reason `failure_manifest_path` is, one
        step further: the manifest now has TWO readers -- the credential-free
        `SourceInspector.failures`, and `Acquisition._run`'s pre-write merge
        (03.4 D-17/D-18), which folds the previous run's entries for symbols
        THIS run never attempted back in before the overwrite. That merge runs
        on EVERY exit path of the resume loop since 03.4-08; it was wired into
        the cancel branch alone before that, which is what let the default
        quota abort empty the manifest (`03.4-VERIFICATION.md` gap 1). Two
        tolerant readers with two copies of the failure policy is precisely the
        duplication this phase's research names as the failure mode, so there
        is one.

        Absent means `{}`, not an error: a source that has never failed and a
        source that has never run look the same from here, and both answers are
        "nothing to report". A corrupt file is tolerated the same way
        `read_sidecar` tolerates a corrupt sidecar -- one failure policy for
        unreadable JSON, not two that can drift.

        The values are already scrubbed on the way IN: they are the strings
        `Acquisition._attempt_batch` produced through `_scrub`. This method
        adds no new egress path for raw vendor exception text.
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
        """Every symbol that has a watermark sidecar under `watermark_root`.

        Sorted, and EXPLICITLY skipping `PAGE_LEDGER_DIR_NAME` and
        `FAILURE_MANIFEST_NAME`. `Acquisition.stamp_watermarks` globs the same
        directory and skips both only IMPLICITLY, via its `last_date is None`
        check -- a filter this method does not have, because "does a watermark
        exist for this symbol" is a question about file presence. Globbing
        blindly here would report the failure manifest as a symbol named
        `_failures`, and the top-level glob would additionally be wrong the day
        a page ledger is written with a name that parses.

        A non-recursive glob, so `_pages/` is out of reach by construction; it
        is named in the skip set anyway, because a rule the code merely happens
        to satisfy is not a rule.
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
        """Load a watermark sidecar's raw JSON, or None if it is absent or
        unparseable.

        The single tolerant read both `read_watermark` and `read_coverage`
        share, so there is exactly ONE failure policy for a corrupt sidecar
        rather than two that could drift apart.
        """
        path = self.watermark_path(symbol)
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
            #
            # Do NOT tighten this into a raise on the grounds that writes are
            # now atomic (03.4-03 D-20): files written before that phase are
            # still on disk. Pinned by
            # `tests/test_atomic_sidecars.py::
            # test_a_half_written_watermark_sidecar_reads_back_as_absent`.
            return None
        return payload if isinstance(payload, dict) else None

    def read_watermark(self, symbol: str) -> str | None:
        """The LAST covered date for `symbol`, or None.

        Signature and meaning are deliberately unchanged by the range-aware
        schema: both `_refresh_batches` and `_attempt_batch` use this to
        compute an incremental start, and neither wants the covered start.
        """
        payload = self.read_sidecar(symbol)
        return None if payload is None else payload.get("last_date")

    def read_coverage(self, symbol: str) -> dict | None:
        """The covered RANGE for `symbol` as
        `{"start_date", "last_date", "no_data"}`, or None when no readable
        sidecar exists.

        Either date component may be None. In particular a LEGACY sidecar --
        `{"last_date": ...}`, the only format written before 260906-26o --
        reads back with `start_date=None`, and nothing anywhere fills that in
        from `config.start_date` or any other fallback.

        That absence is the whole point (D-04). Only the user knows what
        window those files were actually fetched over; an invented start that
        happens to be wrong reproduces exactly the silent per-symbol history
        gap this schema exists to eliminate, and reproduces it invisibly.
        Stamping is therefore an explicit, user-supplied step --
        `Acquisition.stamp_watermarks()`.

        `no_data` follows the SAME discipline from the other side: it defaults
        to `False` when the key is absent, which is the correct reading of
        every sidecar written before 03.2 because the old code only wrote a
        watermark after a successful fetch. It is read through `read_sidecar`
        like everything else -- adding a second tolerant read for the marker
        would give a corrupt sidecar two failure policies that could drift.
        """
        payload = self.read_sidecar(symbol)
        if payload is None:
            return None
        return {
            "start_date": payload.get("start_date"),
            "last_date": payload.get("last_date"),
            "no_data": bool(payload.get("no_data", False)),
        }

    # -- the coverage rule (D-09: exactly one implementation) ---------------

    def legacy_policy(self) -> str:
        policy = self._knob("legacy_watermarks", self.default_legacy_policy)
        if policy not in self.legacy_policies:
            raise ValueError(
                f"legacy_watermarks={policy!r} is not one of "
                f"{list(self.legacy_policies)}."
            )
        return policy

    def coverage_status(self, symbol: str, from_watermark: bool = False) -> str:
        """Classify `symbol` against the REQUESTED window, returning one of
        `"uncovered"`, `"covered"`, `"widened"` or `"legacy"`.

        A symbol is `"covered"` iff its recorded `last_date` equals
        `config.end_date` AND its recorded covered start is known and is
        `<=` `config.start_date`. ISO-8601 `YYYY-MM-DD` orders correctly under
        plain string comparison, so no date parsing happens here and no time
        zone can creep in.

        `"widened"` is the 260906-26o defect (D-03): the end date matches but
        the recorded coverage starts LATER than what is being asked for, so
        the symbol's history is shallower than the request and it must be
        re-fetched. Before this predicate existed it was skipped in silence,
        and the dataset shipped with inconsistent per-symbol history depth.

        `"legacy"` is a sidecar written before this schema: the end date
        matches but the covered start is UNKNOWN. Three responses exist and
        two are wrong. Assuming a start is forbidden outright (D-04) -- an
        assumed range that is wrong reproduces the silent gap invisibly.
        Treating unknown as uncovered is correct for integrity but re-fetches
        every already-downloaded symbol and burns a whole quota window (D-01).
        So the default is the third: treat it as covered for SKIP purposes and
        say so LOUDLY on every run until stamped. What made the D-03 failure
        dangerous was the silence, not the skip -- a run that skips these
        while printing their count and the exact command that fixes them is a
        REPORTED gap with a named cure, and only the user knows what window
        those files were fetched over. `legacy_watermarks="refetch"` is the
        opt-in escape hatch that makes this a choice rather than an accident.

        `from_watermark` (i.e. `refresh()`) short-circuits to the END-DATE
        rule alone, deliberately. Refresh requests `[watermark, end_date]` per
        symbol and never `config.start_date`, so judging it against a widened
        `config.start_date` would mark every symbol pending on every run while
        the re-fetch it triggers could not close the gap -- an endless, silent
        quota burn. Widening the covered range is `download()`'s job.

        The rule itself lives in `classify_coverage`, over an ALREADY-READ
        coverage dict, so `partition_by_coverage` can classify and count the
        `no_data` marker from a single read per sidecar. That is a split of
        read from rule, not a second read path -- `read_sidecar` remains the
        only place a sidecar is opened.
        """
        return self.classify_coverage(self.read_coverage(symbol), from_watermark)

    def classify_coverage(
        self, coverage: dict | None, from_watermark: bool = False
    ) -> str:
        """`coverage_status`'s rule, applied to an already-read coverage dict.

        **THE D-09 CHOKE POINT.** Every `last_date == end_date` and every
        `start_date <=` comparison in this repository belongs in this method
        and nowhere else. A caller that "simplifies" the rule locally will get
        `legacy` wrong, and its answer will agree with this one on every case
        it was tested against.

        Deliberately blind to `coverage["no_data"]`. A marked symbol whose
        recorded window still covers the request is `covered` by the ordinary
        rule and is skipped; a marked symbol whose recorded window is narrower
        is `widened` and is re-fetched. The ABSENCE of a special case here is
        the design (D-04): the marker records what the vendor said about a
        WINDOW, and a branch that turned it into a permanent verdict about the
        symbol would make a later, deeper request unable to reach the vendor
        at all. Tests pin both directions so the branch cannot be added later
        as a plausible-looking "optimisation".
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
        """Whether `symbol` may be skipped for the requested window.

        The skip predicate `_run` filters on. See `coverage_status` for the
        rule and for the D-04 argument behind the `"legacy"` branch.
        """
        status = self.coverage_status(symbol, from_watermark)
        if status == "legacy":
            return self.legacy_policy() == "warn"
        return status == "covered"

    def partition_by_coverage(
        self, requested: list[str], from_watermark: bool
    ) -> tuple[list[str], dict[str, int]]:
        """Split `requested` into what still needs fetching, plus the counts
        the run reports. One pass, so each sidecar is read exactly once.

        **This is the function object D-09 requires both callers to reach.**
        `Acquisition._partition_by_coverage` delegates here and
        `SourceInspector.coverage` calls it directly, so mutating this one
        body changes BOTH answers -- which is how the sharing is proved, since
        two results that merely agree would pass for two implementations.
        """
        legacy_is_skipped = self.legacy_policy() == "warn"
        pending: list[str] = []
        counts = {"covered": 0, "widened": 0, "legacy": 0, "no_data": 0}

        for symbol in requested:
            coverage = self.read_coverage(symbol)
            status = self.classify_coverage(coverage, from_watermark)
            # Counted ALONGSIDE the status rather than as a fourth status: a
            # marked symbol is `covered`/`widened`/`legacy` by exactly the same
            # rule as an unmarked one (the marker records what the vendor said
            # about a window, not a verdict about the symbol), and the count
            # exists so a run can REPORT how many symbols the vendor had
            # nothing for -- distinguishably from how many failed. What made
            # the 260906-26o defect dangerous was the silence, not the skip.
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
        """Reject any symbol that is not a well-formed ticker, and return the
        validated list.

        Thin wrapper over the module-level `validate_symbols`, which is the one
        implementation. The wrapper exists because most callers already hold a
        ledger and should not have to unpack two of its fields to ask this
        question; the free function exists because `SourceInspector.browse_raw`
        holds a `DatasetConfig` and no ledger, and passing a `DatasetConfig`
        where an `AcquisitionConfig` is declared -- which would work today,
        since this check reads only `raw_data_dir_path` -- is a false type
        claim waiting to break silently.
        """
        return validate_symbols(
            symbols,
            owner_label=self.owner_label,
            raw_root=self.config.raw_data_dir_path,
        )


def validate_symbols(
    symbols: Sequence[str], *, owner_label: str, raw_root: str
) -> list[str]:
    """Reject any symbol that is not a well-formed ticker, and return the
    validated list.

    Called BEFORE path construction by everything that builds a filesystem
    path from a caller-supplied symbol -- `Acquisition._run`,
    `Acquisition.coverage_report`, `SourceInspector.coverage` and
    `SourceInspector.browse_raw` -- because a symbol crosses two trust
    boundaries at once:

    - it becomes a filesystem path component under the raw root, where a
      value containing `/` or `..` would escape that root entirely
      (T-03.2-03);
    - it becomes one element of a comma-joined `symbols=` query parameter,
      where an embedded comma would silently change WHICH symbols were
      requested -- the response would look fine and the data would be for
      something else (T-03.2-04).

    One control covers both, which is why it lives on the shared ledger
    rather than in each vendor's `_fetch_page` or in each reader.

    The pattern is `enums.data.TRADEABLE_TICKER_PATTERN` -- the SAME
    compiled object the roster builder filters its output on, bound here
    rather than re-declared. That shared identity is the point: a symbol
    the builder persists is admitted here by construction, which is exactly
    what was NOT true before quick task 260907-10t, when a local copy of a
    narrower literal made `download()`'s whole-roster pre-flight abort a
    multi-hour full-market job on `NXG-R-W`.

    It admits digits deliberately (260906-eme: digit-bearing tickers are
    real) and up to TWO suffix segments (260907-10t: 77 `us_all` and 4
    `nasdaq_all` symbols are three-segment `ROOT-X-Y`). It is NOT
    `acquisition/universe.py`'s `_WELL_FORMED_TICKER`, which is
    deliberately narrower because it guards Wikipedia change-log cells --
    see that constant's own comment before considering aligning them.
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
                f"acquisition/universe.py:TiingoRosterFetcher.fetch(), so "
                f"reaching here means the reference table predates that "
                f"filter and needs rebuilding."
            )
        validated.append(text)
    return validated
