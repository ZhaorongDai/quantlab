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
