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
