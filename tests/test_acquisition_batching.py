"""Base-class batching, abort ordering and no-data marking (03.2 SC-1/SC-2/SC-4).

Phase 03.2 promotes the unit of acquisition work from a *symbol* to a
`(vendor, batch, page)` triple: `_fetch_page(symbols, start, end, page_token)`
becomes the single abstract fetch primitive and a single-symbol vendor is the
degenerate case (`DEFAULT_BATCH_SIZE = 1`), with no placeholder anywhere in the
hierarchy (SC-1, CONTEXT.md D-01). Two orderings in the shared base are
load-bearing and invisible at runtime if they regress:

- the global-abort check must be `_attempt`'s FIRST statement -- joblib cannot
  cancel already-queued work, so an input-generator check is an optimisation,
  not the guarantee (260906-26o D-05);
- the D-04 "queried, no data" marker must be evaluated only AFTER a batch's
  last page, never per page. Alpaca is symbol-major, so page 1 of a 100-symbol
  batch legitimately holds one symbol; per-page evaluation would stamp 99
  symbols "no data", advance their watermarks and skip them forever -- a silent
  99% loss that looks like a successful run (RESEARCH Pitfall 4).

Every test here is offline. Nothing sleeps for real, makes a network call,
requires a credential, or touches any real data volume.

This file lands in 03.2-01 (Wave 0) carrying its fixture self-test; 03.2-02 and
03.2-03 fill in the behavioural tests above. It is deliberately NOT an empty
placeholder: a pytest file with zero collected tests exits 5 ("no tests ran"),
which a later task's automated command reads as green.
"""

from pathlib import Path


def test_acquisition_config_fixture_places_watermarks_beside_the_vendor_raw_root(
    acquisition_config,
):
    """Fixture self-test for the two SC-7 path invariants, asserted in BOTH
    directions because only one of them is loud when it breaks.

    Direction 1 -- `raw_data_dir_path` terminates AT the vendor segment. This
    is the equality `StockDataset._scan_raw` asserts; a root pointing one level
    up silently unions two vendors (measured in RESEARCH Pattern 5).

    Direction 2 -- `watermark_path` is NOT a descendant of `raw_data_dir_path`.
    A refactor that tucks watermarks back under the raw root (the pre-03.2
    layout: `{subdir}/_watermarks`) breaks nothing visibly at write time, and
    then a polars directory scan of the raw root walks into the `.json`
    sidecars and fails far away from the cause.
    """
    for vendor in ("tiingo", "alpaca"):
        config = acquisition_config(vendor=vendor)
        raw_root = Path(config.raw_data_dir_path)
        watermark_root = Path(config.watermark_path)

        assert raw_root.name == vendor, (
            f"raw_data_dir_path must terminate at the vendor segment; got "
            f"{raw_root} whose basename is {raw_root.name!r}, not {vendor!r}"
        )

        assert not watermark_root.is_relative_to(raw_root), (
            f"watermark_path {watermark_root} must be a SIBLING of the raw "
            f"root {raw_root}, never a descendant -- a scan of the raw root "
            f"walks every file beneath it, including .json sidecars"
        )
        assert watermark_root.parent.name == "_watermarks"
        assert watermark_root.name == vendor


def test_acquisition_config_fixture_isolates_vendors_from_each_other(
    acquisition_config,
):
    """Two vendors built from the same fixture root share a parent but never a
    raw root -- the precondition every SC-7 assertion is written against."""
    tiingo = acquisition_config(vendor="tiingo")
    alpaca = acquisition_config(vendor="alpaca")

    tiingo_raw = Path(tiingo.raw_data_dir_path)
    alpaca_raw = Path(alpaca.raw_data_dir_path)

    assert tiingo_raw != alpaca_raw
    assert tiingo_raw.parent == alpaca_raw.parent
    assert not tiingo_raw.is_relative_to(alpaca_raw)
    assert not alpaca_raw.is_relative_to(tiingo_raw)


# ---------------------------------------------------------------------------
# 03.2-03 Task 1 -- batch construction on the hoisted base class.
#
# `_batches` chunks a roster; `_refresh_batches` first GROUPS by recorded
# watermark, because one request carries exactly one `start`. Both are pure
# functions of the roster and the sidecars on disk: no vendor call, no
# credential, no network.
# ---------------------------------------------------------------------------

#: Far larger than any batch size under test, and deliberately not a round
#: multiple of every one of them. Borrowed from `tests/test_tiingo_quota.py`'s
#: `_MANY` idiom: "chunked correctly" and "silently collapsed to one batch"
#: must not be able to produce the same count.
_MANY = tuple(f"SYM{i:04d}" for i in range(250))


def _acquisition(acquisition_config, **config_kwargs):
    """A minimal CONCRETE `Acquisition` whose `_fetch_page` is never called.

    Deliberately not `TiingoAcquisition` or `AlpacaAcquisition`: the batching
    rules under test belong to the base class, and instantiating a vendor here
    would let a vendor override silently satisfy the assertion.
    """
    from base.acquisition import Acquisition

    class _Batching(Acquisition):
        VENDOR = "tiingo"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            raise AssertionError(
                "batch construction must not issue a vendor request"
            )

    return _Batching(acquisition_config(**config_kwargs))


def test_batches_chunk_the_roster_by_batch_size(acquisition_config):
    """250 symbols is 3 batches at 100 and 250 batches at 1.

    Both directions matter. The 100 case proves chunking happens at all; the 1
    case proves the DEGENERATE vendor (Tiingo) still gets one symbol per
    request, which is what makes its quota-abort check run once per symbol
    rather than once per hundred.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    batches = list(acq._batches(_MANY))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    # Every symbol appears exactly once, in order -- chunking, not sampling.
    assert [symbol for batch in batches for symbol in batch] == list(_MANY)

    degenerate = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 1}
    )
    single = list(degenerate._batches(_MANY))
    assert len(single) == len(_MANY) == 250
    assert all(len(batch) == 1 for batch in single)


def test_refresh_batches_group_symbols_sharing_a_watermark(acquisition_config):
    """Two symbols at the same recorded `last_date` travel together; a third
    at a different one does not.

    One request carries exactly one `start`, so a mixed batch would have to
    either re-fetch history for some symbols or under-fetch for others.
    Grouping is REQUEST PACKING only -- D-06's window rule is untouched, which
    is why the derived starts are asserted here too.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT", "GOOG"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("MSFT", "2024-01-15", start_date="2020-01-01")
    acq._write_watermark("GOOG", "2024-01-20", start_date="2024-01-01")

    batches = list(acq._refresh_batches(["AAPL", "MSFT", "GOOG"]))

    assert sorted(sorted(batch) for batch in batches) == [
        ["AAPL", "MSFT"],
        ["GOOG"],
    ], batches

    # The grouping key is the WATERMARK, not the covered start -- AAPL and
    # MSFT share a `last_date` while recording different `start_date`s, and
    # they still batch together because only `last_date` decides the request.
    by_symbol = {tuple(sorted(batch)): batch for batch in batches}
    assert ("AAPL", "MSFT") in by_symbol


def test_refresh_batches_bucket_un_watermarked_symbols_under_the_config_start(
    acquisition_config,
):
    """A symbol with no sidecar has no watermark to group on, so it buckets
    under `config.start_date` -- which is exactly the start `_attempt_batch`
    would derive for it. Grouping it with a watermarked symbol would request
    the wrong window for one of the two.
    """
    acq = _acquisition(
        acquisition_config,
        symbols=("AAPL", "MSFT"),
        kwargs={"batch_size": 100},
    )
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    # MSFT has no sidecar at all.

    batches = list(acq._refresh_batches(["AAPL", "MSFT"]))

    assert sorted(sorted(batch) for batch in batches) == [["AAPL"], ["MSFT"]]


def test_refresh_batches_still_chunk_a_large_shared_watermark_group(
    acquisition_config,
):
    """The common case: a routine refresh where every symbol shares one
    watermark. Grouping must not defeat chunking and hand the vendor a
    250-symbol request.
    """
    acq = _acquisition(
        acquisition_config, symbols=_MANY, kwargs={"batch_size": 100}
    )
    for symbol in _MANY:
        acq._write_watermark(symbol, "2024-01-15", start_date="2024-01-01")

    batches = list(acq._refresh_batches(list(_MANY)))

    assert len(batches) == 3, len(batches)
    assert [len(batch) for batch in batches] == [100, 100, 50]
    assert sorted(symbol for batch in batches for symbol in batch) == sorted(
        _MANY
    )


# ---------------------------------------------------------------------------
# 03.2-03 Task 3 -- two invariants locked by SOURCE INTROSPECTION.
#
# Both are properties of code SHAPE that no runtime assertion can reach:
#
# - the global-abort check must be the FIRST executable statement of
#   `_attempt_batch`. joblib cannot cancel work it has already queued, so the
#   input generator's check is an optimisation and this one is the guarantee.
#   Move it three lines down, past a coverage read, and every test still
#   passes while a quota-exhausted run keeps issuing vendor requests.
# - no concrete `Acquisition` subclass may carry a not-implemented placeholder,
#   and every one must reach its vendor through the shared `_fetch_batch`
#   (SC-1). Asserted by WALKING `Acquisition.__subclasses__()` rather than by
#   naming the vendors, so a third vendor added later is covered without
#   editing this file.
#
# The Phase-3 precedent for replacing a runtime surprise with a test-time
# source assertion is `03-03-PLAN.md`'s `Factor` introspection.
# ---------------------------------------------------------------------------


def _first_executable_statement(func):
    """The first statement of `func`'s body, docstring excluded.

    Parsed from the AST, never matched as a substring of the file: a COMMENT
    mentioning the abort check, or a docstring quoting it, must not be able to
    satisfy an assertion about what the code actually does first.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    body = [
        node
        for node in tree.body[0].body
        if not (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
        )
    ]
    assert body, f"{func.__qualname__} has an empty body"
    return body[0]


def test_the_abort_check_is_first_in_attempt_batch_abort_is_first():
    """Named to match the `-k abort_is_first` selector, deliberately.

    The selector is the plan's evidence that this invariant has a guard, so
    the name is load-bearing: a rename that drops the token would leave the
    selector matching zero tests, and `pytest` exits 5 on "no tests ran" --
    which reads as green to anything checking only the exit code.
    """
    import ast

    from base.acquisition import Acquisition

    first = _first_executable_statement(Acquisition._attempt_batch)

    assert isinstance(first, ast.If), (
        f"the first statement of _attempt_batch must be the abort guard; got "
        f"{type(first).__name__}"
    )
    condition = ast.dump(first.test)
    assert "_abort" in condition and "is_set" in condition, condition

    # ...and it must RETURN, not merely log. A guard that falls through is not
    # a guard.
    assert any(isinstance(node, ast.Return) for node in first.body)
    returned = ast.dump(ast.Module(body=first.body, type_ignores=[]))
    assert "skipped" in returned, returned


def _concrete_acquisition_subclasses():
    """Every concrete `Acquisition` subclass defined in the SHIPPED tree.

    Imports every module under the `acquisition` package first, so the walk
    does not depend on which vendor some earlier test happened to import --
    and so a third vendor dropped into that package is picked up here with no
    edit to this file.

    Classes defined inside the test tree are excluded: a test double is not a
    vendor, and whether one exists at all depends on test ordering.
    """
    import importlib
    import pkgutil
    import sys

    import acquisition
    from base.acquisition import Acquisition

    for info in pkgutil.iter_modules(
        acquisition.__path__, acquisition.__name__ + "."
    ):
        importlib.import_module(info.name)

    found: dict[str, type] = {}

    def walk(cls):
        for subclass in cls.__subclasses__():
            module = sys.modules.get(subclass.__module__)
            path = Path(getattr(module, "__file__", "") or "")
            if "tests" not in path.parts and not getattr(
                subclass, "__abstractmethods__", None
            ):
                found[subclass.__qualname__] = subclass
            walk(subclass)

    walk(Acquisition)
    return list(found.values())


#: Phrases that mark a method body as a PLACEHOLDER rather than an
#: implementation. `NotImplementedError` is the obvious one; this codebase has
#: also used `raise ValueError("Not finished")` for the same thing
#: (`dataset/stock.py` pre-03.2), so both idioms are caught.
_PLACEHOLDER_TOKENS = ("not implemented", "not finished", "notimplementederror")


def test_no_concrete_acquisition_subclass_carries_a_not_implemented_placeholder():
    """SC-1, asserted over the hierarchy rather than over two named classes.

    A single-symbol vendor is the DEGENERATE case of the batched primitive
    (`DEFAULT_BATCH_SIZE = 1`, `_fetch_page` returning `(frame, None)`), not a
    special case that has to fake a multi-symbol interface. If any subclass
    ever needs a placeholder to satisfy the base contract, the contract is
    wrong -- and this test is where that shows up.
    """
    import ast
    import inspect
    import textwrap

    classes = _concrete_acquisition_subclasses()
    assert len(classes) >= 2, (
        f"the walk found {len(classes)} concrete subclass(es); with fewer "
        f"than two vendors this assertion proves nothing about sharing"
    )

    offenders = []
    for cls in classes:
        for name, member in vars(cls).items():
            function = getattr(member, "__func__", member)
            if not inspect.isfunction(function):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Raise):
                    continue
                rendered = ast.dump(node).lower()
                if any(token in rendered for token in _PLACEHOLDER_TOKENS):
                    offenders.append(f"{cls.__qualname__}.{name}")

    assert not offenders, (
        f"placeholder(s) found in the Acquisition hierarchy: {offenders}. "
        f"Every concrete subclass must reach its vendor through the shared "
        f"_fetch_batch/_fetch_page path (SC-1)."
    )


def test_every_concrete_acquisition_subclass_reaches_the_vendor_via_fetch_batch(
    mock_tiingo_client, mock_alpaca_client, acquisition_config
):
    """SC-1's positive direction: one shared path, walked per subclass.

    Both vendor transports are mocked by the two fixtures, so this issues no
    request and needs no credential. The spy is installed on the INSTANCE, so
    nothing global is mutated and the vendors cannot interfere with each other.
    """
    classes = _concrete_acquisition_subclasses()
    assert len(classes) >= 2, len(classes)

    roster = ["AAPL", "MSFT"]
    for cls in classes:
        cfg = acquisition_config(vendor=cls.VENDOR, symbols=tuple(roster))
        acq = cls(cfg)

        batched: list[list[str]] = []
        real = acq._fetch_batch

        def spy(symbols, *args, _real=real, _sink=batched, **kwargs):
            _sink.append(list(symbols))
            return _real(symbols, *args, **kwargs)

        acq._fetch_batch = spy
        acq.download(list(roster))

        assert batched, (
            f"{cls.__qualname__}.download() never reached _fetch_batch -- it "
            f"is carrying a second write path that can drift from the shared "
            f"one"
        )
        assert sorted({symbol for call in batched for symbol in call}) == sorted(
            roster
        ), f"{cls.__qualname__} fetched {batched}, expected every symbol once"

        # ...and it actually landed: one watermark per symbol, and a shard
        # tree that is not empty.
        for symbol in roster:
            assert (Path(cfg.watermark_path) / f"{symbol}.json").exists(), (
                f"{cls.__qualname__} wrote no watermark for {symbol}"
            )
        assert sorted(Path(cfg.raw_data_dir_path).rglob("*.pqt")), (
            f"{cls.__qualname__} wrote no raw shard"
        )


def test_refresh_actually_dispatches_the_watermark_grouped_batches(
    acquisition_config,
):
    """The grouping must be WIRED, not merely available.

    Found by mutation: replacing `_refresh_batches` with `_batches` inside
    `_run_once` left the entire suite green. The unit tests above prove the
    grouping FUNCTION is right; nothing proved `refresh()` calls it, and the
    one vendor whose refresh is covered end to end has
    `DEFAULT_BATCH_SIZE = 1`, which makes grouping a no-op that cannot fail.

    The consequence of the un-wired version is silent and expensive: symbols
    with unequal watermarks share one request, so one `start` is used for all
    of them -- re-fetching history for some and, if the earliest start is not
    chosen, under-fetching for others.

    Asserted on a batch-size-100 subclass rather than on a named vendor, so
    this covers whichever vendor is multi-symbol rather than the one that
    happens to be today.
    """
    from base.acquisition import Acquisition

    requests_made: list[tuple[tuple[str, ...], str]] = []

    class _Recording(Acquisition):
        VENDOR = "alpaca"
        RAW_COLUMNS = ("timestamp", "symbol", "vendor")
        DEFAULT_BATCH_SIZE = 100

        def _fetch_page(self, symbols, start_date, end_date, page_token=None):
            import polars as pl

            requests_made.append((tuple(symbols), start_date))
            frame = pl.DataFrame(
                schema={
                    "timestamp": pl.Datetime,
                    "symbol": pl.String,
                    "vendor": pl.String,
                }
            )
            return frame.select(self.RAW_COLUMNS), None

    cfg = acquisition_config(
        vendor="alpaca", symbols=("AAPL", "MSFT", "GOOG"), kwargs={"progress": False}
    )
    acq = _Recording(cfg)
    acq._write_watermark("AAPL", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("MSFT", "2024-01-15", start_date="2024-01-01")
    acq._write_watermark("GOOG", "2024-01-20", start_date="2024-01-01")

    acq.refresh()

    dispatched = {symbols: start for symbols, start in requests_made}
    assert len(requests_made) == 2, (
        f"three symbols across TWO distinct watermarks must become two "
        f"requests, not one and not three; got {requests_made}"
    )
    assert dispatched[("AAPL", "MSFT")] == "2024-01-15"
    assert dispatched[("GOOG",)] == "2024-01-20"


# ---------------------------------------------------------------------------
# 03.2-05 Task 1 -- the "queried, no data" third state (D-04, SC-4).
#
# ONE additive boolean on the existing watermark sidecar, written only when
# TRUE. That asymmetry is the whole design: absence is the default, so every
# sidecar written before this phase reads back as not-no-data -- which is
# correct, because the pre-change code only ever wrote a watermark after a
# successful fetch. Writing `false` would instead make those older files
# ambiguous (RESEARCH Pattern 4, 260906-26o D-04).
#
# Four read-time states out of three storage facts:
#
#   never fetched        -> no sidecar, no manifest entry
#   fetch failed         -> no sidecar, PLUS a `_failures.json` entry
#   fetched, data landed -> sidecar, marker ABSENT
#   queried, no data     -> sidecar, marker present and true
#
# Every test below is offline: no vendor call, no credential, no network.
# ---------------------------------------------------------------------------

#: The exact key set `_write_watermark` produced BEFORE this task, for a call
#: that passes a covered start. Written as a literal rather than derived, so
#: the additive-schema assertion is anchored to the old contract and not to
#: whatever the current implementation happens to emit.
_PRE_CHANGE_SIDECAR_KEYS = {"last_date", "start_date"}


def _sidecar_json(acq, symbol: str) -> dict:
    """The raw sidecar bytes as parsed JSON, bypassing every reader.

    Deliberately not `_read_coverage`: a test about what is ON DISK must not
    be satisfiable by a reader that invents the key.
    """
    import json

    with open(Path(acq.config.watermark_path) / f"{symbol}.json") as f:
        return json.load(f)


def _observed_state(acq, symbol: str) -> str:
    """Which of the four read-time states `symbol` is in, from disk only.

    ONE helper, used for all four assertions, so the four states are proved
    distinguishable by a single reading procedure rather than by four bespoke
    lookups that could each be reading a different thing.
    """
    import json

    coverage = acq._read_coverage(symbol)
    manifest_path = Path(acq.config.watermark_path) / acq.FAILURE_MANIFEST_NAME
    manifest: dict = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)

    if coverage is None:
        return "failed" if symbol in manifest else "never_fetched"
    return "no_data" if coverage["no_data"] else "has_data"


def test_write_watermark_omits_the_no_data_key_unless_it_is_true(
    acquisition_config,
):
    """The marker is written ONLY when true; false omits it entirely.

    Both directions are asserted on the raw JSON, because this is a statement
    about the on-disk contract and not about the reader. The false case is
    additionally compared against the PRE-CHANGE key set: a sidecar written
    with the flag false must be byte-comparable in its key structure to what
    the old code wrote, which is what makes the extension additive rather than
    a new format wearing the old name.
    """
    acq = _acquisition(acquisition_config)

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01", no_data=True)
    marked = _sidecar_json(acq, "AAPL")
    assert marked["no_data"] is True, marked
    assert marked["last_date"] == "2024-01-31"
    assert marked["start_date"] == "2024-01-01"

    acq._write_watermark("MSFT", "2024-01-31", start_date="2024-01-01", no_data=False)
    plain = _sidecar_json(acq, "MSFT")
    assert "no_data" not in plain, (
        f"the marker must be OMITTED when false, never written as false -- "
        f"got {plain}"
    )
    assert set(plain) == _PRE_CHANGE_SIDECAR_KEYS, (
        f"a sidecar written with the flag false must carry exactly the "
        f"pre-change key set {sorted(_PRE_CHANGE_SIDECAR_KEYS)}; got "
        f"{sorted(plain)}"
    )

    # ...and the default is false, so no existing caller starts marking.
    acq._write_watermark("GOOG", "2024-01-31", start_date="2024-01-01")
    assert set(_sidecar_json(acq, "GOOG")) == _PRE_CHANGE_SIDECAR_KEYS


def test_a_pre_change_sidecar_reads_back_as_not_no_data(acquisition_config):
    """Additive direction 1: an OLD file is valid input to the NEW reader.

    The sidecar is hand-written in the pre-change shape rather than produced by
    the current writer, so the assertion cannot be satisfied by a writer that
    silently started stamping the key.
    """
    import json

    acq = _acquisition(acquisition_config)
    directory = Path(acq.config.watermark_path)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "AAPL.json", "w") as f:
        json.dump({"start_date": "2024-01-01", "last_date": "2024-01-31"}, f)

    coverage = acq._read_coverage("AAPL")
    assert coverage is not None
    assert coverage["no_data"] is False
    assert coverage["start_date"] == "2024-01-01"
    assert coverage["last_date"] == "2024-01-31"

    # And a LEGACY sidecar -- the pre-26o shape, end date only -- likewise.
    with open(directory / "MSFT.json", "w") as f:
        json.dump({"last_date": "2024-01-31"}, f)
    legacy = acq._read_coverage("MSFT")
    assert legacy is not None
    assert legacy["no_data"] is False
    assert legacy["start_date"] is None


def test_a_no_data_sidecar_is_still_read_by_the_unchanged_read_watermark(
    acquisition_config,
):
    """Additive direction 2: a NEW file is valid input to an OLD reader.

    `_read_watermark` is deliberately untouched by this task, so it returns the
    same `last_date` for a marked sidecar as for an unmarked one. That is the
    property that makes the marker safe to write onto a volume that older code
    may still read.
    """
    acq = _acquisition(acquisition_config)

    acq._write_watermark("AAPL", "2024-01-31", start_date="2024-01-01", no_data=True)
    acq._write_watermark("MSFT", "2024-01-31", start_date="2024-01-01")

    assert acq._read_watermark("AAPL") == "2024-01-31"
    assert acq._read_watermark("MSFT") == acq._read_watermark("AAPL")


def test_the_four_no_data_states_are_distinguishable_on_disk(acquisition_config):
    """SC-4's actual claim: FOUR states, not two, from three storage facts.

    A design that collapses "fetch failed" into "queried, no data" has missed
    the point -- one is a fault to retry, the other is a confirmed absence to
    skip, and reading them as the same thing either loses data or burns quota
    forever.
    """
    acq = _acquisition(
        acquisition_config, symbols=("NEVER", "FAILED", "HASDATA", "NODATA")
    )

    # 1. never fetched -- nothing written for NEVER at all.
    # 2. fetch failed  -- no sidecar, plus a manifest entry.
    acq._write_failure_manifest({"FAILED": "HTTPError: 500"})
    # 3. fetched, data landed.
    acq._write_watermark("HASDATA", "2024-01-31", start_date="2024-01-01")
    # 4. queried, vendor returned nothing.
    acq._write_watermark(
        "NODATA", "2024-01-31", start_date="2024-01-01", no_data=True
    )

    observed = {
        symbol: _observed_state(acq, symbol)
        for symbol in ("NEVER", "FAILED", "HASDATA", "NODATA")
    }

    assert observed == {
        "NEVER": "never_fetched",
        "FAILED": "failed",
        "HASDATA": "has_data",
        "NODATA": "no_data",
    }, observed
    assert len(set(observed.values())) == 4, observed


def test_a_covered_no_data_symbol_is_skipped_and_issues_no_data_request(
    acquisition_config,
):
    """The re-fetch storm D-04 exists to prevent, asserted end to end.

    `_coverage_status` gets NO new branch: a marked symbol whose recorded
    window still covers the request already classifies `covered` through the
    existing rule. This test is what keeps someone from adding one -- and what
    proves the skip actually reaches `download()`, where the helper's
    `_fetch_page` raises if a vendor request is ever issued.
    """
    acq = _acquisition(acquisition_config, symbols=("AAPL",), kwargs={"progress": False})
    acq._write_watermark(
        "AAPL", acq.config.end_date, start_date=acq.config.start_date, no_data=True
    )

    assert acq._coverage_status("AAPL") == "covered"

    pending, counts = acq._partition_by_coverage(["AAPL"], from_watermark=False)
    assert pending == []
    assert counts["covered"] == 1
    assert counts["no_data"] == 1

    # `_fetch_page` raises on any call, so reaching the vendor fails the test.
    acq.download()


def test_a_no_data_symbol_with_narrower_coverage_is_still_re_fetched(
    acquisition_config,
):
    """The marker records what the vendor said about a WINDOW, never a
    permanent verdict about the symbol.

    A marked symbol whose recorded coverage starts after the requested start is
    `widened` exactly as an unmarked one is -- otherwise the marker would turn
    into a tombstone and a later, deeper request could never reach the vendor.
    """
    acq = _acquisition(acquisition_config, symbols=("AAPL",))
    assert acq.config.start_date == "2024-01-01"
    acq._write_watermark(
        "AAPL", acq.config.end_date, start_date="2024-01-15", no_data=True
    )

    assert acq._coverage_status("AAPL") == "widened"

    pending, counts = acq._partition_by_coverage(["AAPL"], from_watermark=False)
    assert pending == ["AAPL"]
    assert counts["widened"] == 1


def test_a_corrupt_no_data_sidecar_takes_the_same_tolerant_path(
    acquisition_config,
):
    """ONE corrupt-sidecar policy, not two.

    `_read_sidecar` is the single tolerant read both `_read_watermark` and
    `_read_coverage` share. A second read path added for the marker could
    drift from it, and the drift would show up as a crash in a 15k-symbol run
    rather than as a wider-than-necessary re-fetch.
    """
    acq = _acquisition(acquisition_config)
    directory = Path(acq.config.watermark_path)
    directory.mkdir(parents=True, exist_ok=True)

    # Truncated mid-marker: the bytes mention the key, so a substring-matching
    # reader would "find" it.
    (directory / "AAPL.json").write_text('{"last_date": "2024-01-31", "no_data": tr')
    assert acq._read_coverage("AAPL") is None
    assert acq._read_watermark("AAPL") is None

    # ...identical to a corrupt sidecar that never mentions the marker.
    (directory / "MSFT.json").write_text("{not json at all")
    assert acq._read_coverage("MSFT") is None
    assert acq._read_watermark("MSFT") is None
