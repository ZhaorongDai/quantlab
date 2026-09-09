"""Home for the progress-reporting and cancellation proofs: ROADMAP success
criterion SC-5, requirements D-16, D-17 and D-18.

SC-5 — a caller can start an acquisition programmatically, observe its
progress, cancel it at a batch boundary leaving the store resumable, and
receive a result object describing what happened.

Scaffolded by plan 03.4-01 (Wave 0). `ProgressEvent`, `ProgressReporter`,
`CancelToken` and `AcquisitionResult` do not exist yet; plans 03.4-02 and
03.4-05 build them and fill this file in.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents recorded in
`.planning/STATE.md`:

1. EVERY test here must be a real assertion. A pytest file with zero tests
   exits **5** ("no tests ran"), which a per-file command reads as green, so a
   placeholder, a body that is only a no-op statement, or a skip/xfail marker
   is indistinguishable from a passing file. On pytest 9.1.1 that same exit 5
   is what a `-k` selector matching nothing produces; which of the two is a
   bug depends on which was expected.

2. No test here is named for a selector `03.4-VALIDATION.md` assigns to a
   later plan (`run_returns_a_result`, `emits_one_event_per_batch`,
   `a_raising_reporter_does_not_abort`, `tqdm_default`,
   `cancel_leaves_a_resumable_store`, `cancel_is_not_a_quota_abort`,
   `cancel_check_is_first`, `result_and_manifest_agree`, `result_is_scrubbed`,
   `logging_is_unchanged`). Those must match ZERO tests until the behaviour
   exists -- in 03.2 a deleted mechanism left `-k fingerprint` green because
   its only covering test was named outside the selector.
"""

from quantlab.acquisition.tiingo import TiingoAcquisition


def test_attempt_batch_returns_exactly_three_elements(
    mock_tiingo_client, acquisition_config
) -> None:
    """The L-6 arity constraint, locked BEFORE the cancel work can widen it.

    `Acquisition._attempt_batch` returns `(symbols, status, message_or_None)`
    and two tests in `tests/test_acquisition_batching.py` (`:1097`, and the
    positional call at `:966`) already destructure it. So the cancel token
    (D-17) must reuse the EXISTING first-statement abort seam and the existing
    `"skipped"` status rather than widening this signature to carry a fourth
    "why it stopped" element -- widening it silently breaks both unpackers with
    a ValueError far from the cause. `_run_once`'s arity, by contrast, is
    pinned by nothing and is the one that may widen.

    Asserted BY LENGTH rather than by unpacking on purpose. `a, b, c = result`
    would fail identically for a 2-tuple and a 4-tuple with an opaque
    "too many values to unpack"; `len(result) == 3` fails saying what it got.

    The abort is set first so the call returns on its FIRST statement, having
    issued zero vendor requests -- this test asserts a signature, and a
    signature test that also performed a fetch would be slow for no reason and
    would fail for reasons unrelated to arity.
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)

    acq._abort.set()
    result = acq._attempt_batch(["AAPL", "MSFT"], from_watermark=False)

    assert isinstance(result, tuple)
    assert len(result) == 3, (
        f"_attempt_batch must return exactly (symbols, status, message); got "
        f"{len(result)} elements: {result!r}"
    )
    assert result[1] == "skipped"
    assert not mock_tiingo_client.calls, (
        "the abort guard is the FIRST statement, so an already-aborted batch "
        "must issue zero vendor requests"
    )


def test_the_progress_knob_is_read_through_kwargs_and_defaults_on(
    mock_tiingo_client, acquisition_config
) -> None:
    """`progress` reaches the run through `config.kwargs`, and is on by default.

    `TqdmProgressReporter` (D-16) replaces today's inline tqdm block in
    `_run_once`, and the ONE thing it must not change is how an operator turns
    the bar off: `AcquisitionConfig.kwargs["progress"]`. A knob read through
    `_knob` never becomes a constructor argument no config file could reach,
    which is what keeps the pipeline config-driven per CLAUDE.md -- so a
    reporter that took `progress` as a constructor argument instead would be a
    regression this test catches.

    Both directions are asserted in one test: the default (no `kwargs` at all,
    the common config) and the explicit opt-out. Asserting only the default
    would pass for a `_knob` that ignored `kwargs` entirely.
    """
    default_acq = TiingoAcquisition(acquisition_config(vendor="tiingo"))
    assert default_acq._knob("progress", True) is True

    disabled_acq = TiingoAcquisition(
        acquisition_config(vendor="tiingo", kwargs={"progress": False})
    )
    assert disabled_acq._knob("progress", True) is False


# ---------------------------------------------------------------------------
# D-16 -- progress as event objects through a pluggable reporter
#
# Every test below imports `quantlab.base.progress` INSIDE its own body rather
# than at module scope. That is deliberate and follows 03.4-03's Task-1 RED
# pattern: a module-scope import of a not-yet-existing module fails the whole
# FILE with a collection error, which the TDD RED gate classifies as a
# fixture/load failure rather than as the named target test failing. A local
# import fails each NAMED test individually, which is what RED evidence has to
# look like.
# ---------------------------------------------------------------------------

#: Copied verbatim from `tests/test_tiingo_quota.py:_captured` (attribution
#: comment kept per this phase's copied-helper convention -- `tests/` is not a
#: package, so a cross-test import would couple one file's collection to
#: another's import-time state). loguru does not propagate to stdlib `logging`,
#: so pytest's `caplog` sees nothing and an explicit in-memory sink is the only
#: way to assert on a log record.
def _captured(level: str = "WARNING"):
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level=level, format="{message}")
    return messages, sink_id


class _RecordingTqdm:
    """A stand-in for `tqdm` that records its constructor kwargs.

    The default reporter's rendering is asserted through the CONSTRUCTOR
    arguments, never by capturing stderr: a bar's rendered width depends on the
    terminal, so a stderr snapshot would be a test of `$COLUMNS`. `total`,
    `desc`, `unit` and `disable` are the four arguments the incumbent block
    passes, and they are what "byte-compatible default" actually means here.
    """

    constructions: list[dict] = []

    def __init__(self, **kwargs):
        _RecordingTqdm.constructions.append(dict(kwargs))
        self.descriptions: list[str] = []
        self.updates = 0
        self.closed = False

    def update(self, n: int = 1) -> None:
        self.updates += n

    def set_description(self, desc: str, refresh: bool = True) -> None:
        self.descriptions.append(desc)

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_emits_one_event_per_batch(mock_tiingo_client, acquisition_config) -> None:
    """One `batch_completed` per batch, `completed` rising 1..N, `total` == N.

    Tiingo's `DEFAULT_BATCH_SIZE` is 1, so five symbols is five batches and the
    batch count is directly checkable against the roster the test wrote.

    `run_started` is asserted to arrive BEFORE the first `batch_completed` and
    `run_finished` AFTER the last, by index rather than by mere presence: a
    reporter that received them in the wrong order would still see the right
    multiset, and a console rendering a bar from that stream would show a bar
    that finishes before it starts.
    """
    from quantlab.base.progress import CallbackProgressReporter

    symbols = ("AAPL", "MSFT", "GOOG", "AMZN", "META")
    config = acquisition_config(vendor="tiingo", symbols=symbols)

    events = []
    acq = TiingoAcquisition(config)
    acq.attach(reporter=CallbackProgressReporter(events.append))
    acq.download()

    kinds = [event.kind for event in events]
    assert kinds.count("run_started") == 1, kinds
    assert kinds.count("run_finished") == 1, kinds

    started = kinds.index("run_started")
    finished = kinds.index("run_finished")
    batch_positions = [i for i, kind in enumerate(kinds) if kind == "batch_completed"]
    assert batch_positions, kinds
    assert started < min(batch_positions)
    assert finished > max(batch_positions)

    batches = [event for event in events if event.kind == "batch_completed"]
    assert len(batches) == len(symbols), (
        f"one event per batch: {len(symbols)} symbols at batch_size=1 must "
        f"produce {len(symbols)} events, got {len(batches)}"
    )
    assert [event.completed for event in batches] == list(
        range(1, len(symbols) + 1)
    )
    assert {event.total for event in batches} == {len(symbols)}
    assert events[started].total == len(symbols)

    # Every event names the vendor, so a console multiplexing two concurrent
    # acquisitions can tell their streams apart.
    assert {event.vendor for event in events} == {"tiingo"}


def test_a_raising_reporter_does_not_abort(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """RESEARCH Pitfall 9: a console callback that throws must not tear down
    the fan-out and end a multi-hour backfill over a UI bug.

    Asserted by DIFFERENCE against a silent run rather than by "the call
    returned": the run has to produce the same successes, the same failures and
    the same watermarks, which is the property that makes the reporter
    genuinely inert. A warning record is asserted too, so the exception is
    swallowed but not hidden (RESEARCH A6).
    """
    from quantlab.base.progress import CallbackProgressReporter

    class _Exploding(CallbackProgressReporter):
        def __init__(self):
            super().__init__(self._boom)

        @staticmethod
        def _boom(event):
            raise RuntimeError("reporter is broken")

    symbols = ("AAPL", "MSFT", "GOOG")

    quiet = TiingoAcquisition(
        acquisition_config(
            vendor="tiingo", symbols=symbols, root=tmp_path / "quiet"
        )
    )
    quiet.download()

    loud_config = acquisition_config(
        vendor="tiingo", symbols=symbols, root=tmp_path / "loud"
    )
    loud = TiingoAcquisition(loud_config)
    loud.attach(reporter=_Exploding())

    from loguru import logger

    messages, sink_id = _captured()
    try:
        loud.download()
    finally:
        logger.remove(sink_id)

    assert loud.last_result.succeeded == quiet.last_result.succeeded
    assert loud.last_result.failures == quiet.last_result.failures
    assert loud.last_result.cancelled == quiet.last_result.cancelled

    quiet_marks = sorted(
        p.name for p in quiet._watermark_root.glob("*.json")
    )
    loud_marks = sorted(p.name for p in loud._watermark_root.glob("*.json"))
    assert loud_marks == quiet_marks
    assert len(loud_marks) > 1, (
        "the comparison is only meaningful if the silent run actually wrote "
        "watermarks for several symbols"
    )

    assert any("reporter" in message.lower() for message in messages), (
        f"a swallowed reporter exception must still be logged at warning "
        f"level; captured: {messages}"
    )


def test_tqdm_default_bar_arguments_are_unchanged(
    monkeypatch, mock_tiingo_client, acquisition_config
) -> None:
    """The incumbent stderr rendering survives the refactor (D-16).

    With NO reporter attached the run must still construct exactly one tqdm bar
    per pass with the same four arguments the replaced inline block passed:
    `total=len(batches)`, `desc=f"{VENDOR} {start}..{end}"`, `unit="batch"` and
    `disable=not progress_knob`. Replacing a visible bar with a silent default
    is a regression dressed as a refactor -- every existing shell run would
    look hung for hours.
    """
    import quantlab.base.progress as progress

    _RecordingTqdm.constructions = []
    monkeypatch.setattr(progress, "tqdm", _RecordingTqdm)

    symbols = ("AAPL", "MSFT", "GOOG")
    config = acquisition_config(
        vendor="tiingo",
        symbols=symbols,
        start_date="2024-01-01",
        end_date="2024-01-31",
    )
    TiingoAcquisition(config).download()

    assert len(_RecordingTqdm.constructions) == 1, (
        f"exactly one bar per pass, as today; got "
        f"{_RecordingTqdm.constructions}"
    )
    built = _RecordingTqdm.constructions[0]
    assert built["total"] == len(symbols)
    assert built["desc"] == "tiingo 2024-01-01..2024-01-31"
    assert built["unit"] == "batch"
    assert built["disable"] is False


def test_the_progress_knob_off_constructs_no_bar(
    monkeypatch, mock_tiingo_client, acquisition_config
) -> None:
    """`config.kwargs["progress"] = False` still turns the rendering off.

    The knob keeps its meaning through `_knob`, so an operator's existing
    opt-out is unchanged -- and the run still completes, which is asserted so
    the test cannot pass because nothing ran at all.
    """
    import quantlab.base.progress as progress

    _RecordingTqdm.constructions = []
    monkeypatch.setattr(progress, "tqdm", _RecordingTqdm)

    config = acquisition_config(
        vendor="tiingo",
        symbols=("AAPL", "MSFT"),
        kwargs={"progress": False},
    )
    acq = TiingoAcquisition(config)
    acq.download()

    assert _RecordingTqdm.constructions == []
    assert set(acq.last_result.succeeded) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# D-17 -- cancellation as a separate token, checked at the batch boundary
# ---------------------------------------------------------------------------

#: More batches than workers, and -- with Tiingo's `DEFAULT_BATCH_SIZE = 1` --
#: more batches than joblib's default `pre_dispatch` of `2 * n_jobs` allows to
#: be withheld. At `max_workers = 8` every one of these 12 batches is QUEUED
#: before the first result is drained, so the input generator's `break` can
#: never fire and the FIRST-STATEMENT check in `_attempt_batch` is the only
#: thing that can stop the vendor requests. That is RESEARCH Pitfall 2 built
#: into the fixture rather than asserted in prose: an implementation that
#: checked only the generator would issue all 12 calls.
_CANCEL_SYMBOLS = tuple(f"SYM{i:02d}" for i in range(12))
_CANCEL_WORKERS = 8


def _cancel_on_first_call(mock_client, token) -> None:
    """Make the fake vendor cancel the run from inside its FIRST request.

    Cancelling from the client rather than from the drain loop is what makes
    the test deterministic: at `n_jobs` concurrent workers at most `n_jobs`
    batches can already be past the first-statement guard when the token is
    set, so the vendor call count is bounded by `max_workers` -- a real bound,
    not a timing hope.
    """
    original = mock_client.get_ticker_price

    def cancelling(self, ticker, **kwargs):
        token.cancel()
        return original(self, ticker, **kwargs)

    mock_client.get_ticker_price = cancelling


def _sleep_recording_acquisition():
    """A `TiingoAcquisition` whose `_sleep` seam records instead of sleeping.

    Adapted from `tests/test_tiingo_quota.py:_acquisition_class` (attribution
    kept per this phase's copied-helper convention). A cancel that took the
    quota branch would sit here for an hour; recording turns "did not wait"
    into an assertion.
    """
    from quantlab.acquisition.tiingo import TiingoAcquisition as _Tiingo

    class _Recording(_Tiingo):
        def __init__(self, config):
            super().__init__(config)
            self.sleeps: list[float] = []

        def _sleep(self, seconds: float) -> None:
            self.sleeps.append(seconds)

    return _Recording


def test_cancel_check_is_first(mock_tiingo_client, acquisition_config) -> None:
    """With ONLY the cancel token set, `_attempt_batch` returns on its first
    statement having issued zero vendor requests.

    The return shape is asserted to be the unchanged 3-element
    `(symbols, "skipped", None)`: `tests/test_acquisition_batching.py` unpacks
    that tuple in two places, so the cancel had to REUSE the existing seam
    rather than widen the signature with a fourth "why it stopped" element
    (L-6). The abort is deliberately NOT set, so this proves the cancel token
    reaches the guard on its own rather than riding the quota flag.
    """
    from quantlab.base.progress import CancelToken

    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)
    token = CancelToken()
    acq.attach(cancel=token)

    assert not acq._abort.is_set(), (
        "the quota abort must be clear, or this test would pass on the "
        "pre-existing abort check alone"
    )
    assert acq._should_stop() is False
    token.cancel()
    assert acq._should_stop() is True

    result = acq._attempt_batch(["AAPL", "MSFT"], from_watermark=False)

    assert isinstance(result, tuple)
    assert len(result) == 3, (
        f"the cancel must reuse the 3-element shape; got {result!r}"
    )
    assert result == (["AAPL", "MSFT"], "skipped", None)
    assert not mock_tiingo_client.calls, (
        "a cancelled batch must issue zero vendor requests"
    )


def test_cancel_is_not_a_quota_abort(
    mock_tiingo_client, acquisition_config
) -> None:
    """RESEARCH Pitfall 1: a cancel must not be reported, or handled, as the
    vendor's allocation running out.

    `wait_for_quota=True` is set deliberately -- with waiting OFF the "no
    `_sleep`" assertion would pass for an implementation that had folded the
    cancel into the quota branch, because that branch does not sleep either
    when waiting is disabled. With it ON, a cancel that took the quota path
    would sleep, log "allocation", and then RESUME the run the operator just
    cancelled.
    """
    from loguru import logger

    from quantlab.base.progress import CancelToken

    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        kwargs={
            "max_workers": _CANCEL_WORKERS,
            "wait_for_quota": True,
            "quota_wait_seconds": 3600,
            "quota_max_waits": 3,
        },
    )
    acq = _sleep_recording_acquisition()(config)
    token = CancelToken()
    acq.attach(cancel=token)
    _cancel_on_first_call(mock_tiingo_client, token)

    messages, sink_id = _captured()
    try:
        acq.download()
    finally:
        logger.remove(sink_id)

    text = "\n".join(messages).lower()
    assert "cancel" in text, (
        f"a cancelled run must say so; captured warnings: {messages}"
    )
    assert "allocation" not in text, (
        f"a cancel is not a quota abort and must not be reported as one; "
        f"captured warnings: {messages}"
    )
    assert acq.sleeps == [], (
        f"a cancel must never take the quota wait; slept {acq.sleeps}"
    )
    assert acq.last_result.cancelled is True
    assert acq.last_result.quota_aborted is False


def test_cancel_leaves_a_resumable_store(
    mock_tiingo_client, acquisition_config, tmp_path
) -> None:
    """SC-5's second half: completed batches land with their watermarks, the
    remainder is never attempted, and a re-run resumes exactly there.

    The fixture is chosen so the INPUT GENERATOR cannot be what stops the burn
    (see `_CANCEL_SYMBOLS`): every batch is queued before the first result is
    drained, so an implementation checking only the generator would call the
    vendor 12 times. The bound asserted here is `<= max_workers`, which is the
    real guarantee of a first-statement check -- only batches already past the
    guard can still reach the vendor.
    """
    from quantlab.base.progress import CancelToken

    root = tmp_path / "run"
    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        root=root,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    acq = TiingoAcquisition(config)
    token = CancelToken()
    acq.attach(cancel=token)
    _cancel_on_first_call(mock_tiingo_client, token)

    acq.download()

    calls = len(mock_tiingo_client.calls)
    assert 1 <= calls <= _CANCEL_WORKERS, (
        f"only batches already past the first-statement guard may reach the "
        f"vendor: at most {_CANCEL_WORKERS} concurrent workers, got {calls} "
        f"calls out of {len(_CANCEL_SYMBOLS)} batches"
    )

    stamped = {p.stem for p in acq._watermark_root.glob("SYM*.json")}
    assert stamped, "completed batches must keep their watermarks"
    assert stamped < set(_CANCEL_SYMBOLS), (
        "a cancelled run must leave the remainder un-stamped, or a re-run "
        "would skip symbols it never fetched"
    )
    assert stamped == set(acq.last_result.succeeded)

    remainder = set(_CANCEL_SYMBOLS) - stamped

    # The re-run: a FRESH acquisition with no cancel token, over the same
    # store. Resume is driven by watermark presence, so this is the property
    # that matters -- the operator loses no work and re-downloads nothing.
    mock_tiingo_client.calls = []
    resumed_config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        root=root,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    TiingoAcquisition(resumed_config).download()

    asked = {call["ticker"] for call in mock_tiingo_client.calls}
    assert asked == remainder, (
        f"the re-run must ask for exactly the un-stamped remainder; asked "
        f"{sorted(asked)}, expected {sorted(remainder)}"
    )


def test_cancelling_twice_or_after_the_run_is_a_no_op(
    mock_tiingo_client, acquisition_config
) -> None:
    """D-17 idempotency: a second `cancel()`, or a cancel arriving after the
    run has already finished, changes nothing -- no exception, no second
    manifest write, no second result.

    Both shapes are covered: a double cancel DURING the run (the console's
    user double-clicking) and a cancel AFTER it (the console's stop signal
    racing a run that just completed).
    """
    from quantlab.base.progress import CallbackProgressReporter, CancelToken

    config = acquisition_config(
        vendor="tiingo",
        symbols=_CANCEL_SYMBOLS,
        kwargs={"max_workers": _CANCEL_WORKERS},
    )
    acq = TiingoAcquisition(config)
    token = CancelToken()

    cancels = {"n": 0}

    def cancel_twice(event):
        if event.kind == "batch_completed" and cancels["n"] < 2:
            cancels["n"] += 1
            token.cancel()

    acq.attach(reporter=CallbackProgressReporter(cancel_twice), cancel=token)
    acq.download()

    assert cancels["n"] == 2, "the double cancel must actually have happened"
    first_result = acq.last_result
    assert first_result.cancelled is True

    manifest = acq._coverage.failure_manifest_path
    before = manifest.read_bytes()
    mtime = manifest.stat().st_mtime_ns

    # ...and a cancel arriving after the run is over.
    token.cancel()
    token.cancel()

    assert acq.last_result is first_result
    assert manifest.read_bytes() == before
    assert manifest.stat().st_mtime_ns == mtime
    assert token.is_cancelled() is True


def test_run_forwards_the_reporter_and_the_cancel_token(
    mock_tiingo_client, acquisition_config
) -> None:
    """`registry.run()` is the console's entry point, so both arguments have to
    survive the trip through it (D-14 / D-16 / D-17).

    Both are KEYWORD-ONLY with `None` defaults, which is what keeps every
    existing call site of `run(descriptor, config)` unchanged -- asserted here
    through `inspect.signature` rather than by a call that happens to work.
    """
    import inspect

    from quantlab.acquisition.registry import run
    from quantlab.acquisition.tiingo import TIINGO_SOURCE
    from quantlab.base.progress import CallbackProgressReporter, CancelToken

    parameters = inspect.signature(run).parameters
    for name in ("reporter", "cancel"):
        assert name in parameters, f"run() must accept {name}"
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None

    events = []
    token = CancelToken()
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))

    result = run(
        TIINGO_SOURCE,
        config,
        reporter=CallbackProgressReporter(events.append),
        cancel=token,
    )

    assert result is not None
    assert set(result.succeeded) == {"AAPL", "MSFT"}
    assert result.cancelled is False
    assert [event.kind for event in events].count("batch_completed") == 2, (
        f"the reporter must have been forwarded through attach(); saw "
        f"{[event.kind for event in events]}"
    )

    # And the token: a run started with an already-cancelled token fetches
    # nothing at all, which is only true if `run()` forwarded it.
    token.cancel()
    mock_tiingo_client.calls = []
    cancelled = run(TIINGO_SOURCE, acquisition_config(vendor="tiingo"), cancel=token)
    assert cancelled.cancelled is True
    assert mock_tiingo_client.calls == []
