"""Acquisition progress as EVENT OBJECTS, plus the cancel token that stops a
run at a batch boundary (03.4 D-16 / D-17).

**Why this module exists.** Until 03.4 the acquisition loop rendered a `tqdm`
bar straight to stderr and logged through loguru. Neither is consumable by an
in-process caller, and D-12 made the caller in-process: the out-of-repo
`quantlab-console` imports quantlab directly rather than spawning a CLI, so
"progress" has to become something a Textual screen can subscribe to.
`tqdm`'s cursor control actively fights such a screen. The stderr bar is
therefore not removed -- it is DEMOTED to one reporter among several, and it
stays the default, because a shell run that suddenly renders nothing for four
hours is a regression dressed as a refactor.

**Why it lives in `base/` and not in `quantlab/acquisition/progress.py`,**
which is where 03.4-RESEARCH.md suggested it: `base/acquisition.py` imports the
default reporter, and this repository's layering runs `base` -> concrete
packages (`acquisition`, `dataset`, `factor`, ...), never the reverse. A
reporter beside the vendor classes would have to be imported backwards. No
D-decision names a path, so the divergence is recorded here rather than made
silently.

**This module is a LEAF.** It imports stdlib plus `tqdm` and NOTHING from
quantlab, so it cannot participate in an import cycle and any module may import
it. `tests/test_acquisition_progress.py` asserts that structurally by walking
this file's `ast`.

It does not import `loguru` either, and that is not an oversight: the
never-raises wrapper around `emit` lives in `Acquisition._emit`, on the caller's
side, because the exception has to be SCRUBBED with the vendor's own
`CREDENTIAL_ENV_VARS` before it is logged and only the acquisition object knows
those. Putting the try/except here would either log unscrubbed text or force
this leaf to learn about credentials.

**Cancellation is a token, never a reporter return value** (D-17). A reporter
that only wants to log must not be able to halt a multi-hour backfill by
forgetting to return the right value, and a buggy reporter must not be able to
end the run either -- `ProgressReporter.emit` returns `None` and the caller
ignores whatever it gets back.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

#: The description the incumbent bar switches to when the vendor's allocation
#: is gone and the result generator is being DRAINED rather than fetched.
#:
#: Defined here, once, and read by BOTH the emitter (which puts it on the
#: event's `message`) and `TqdmProgressReporter` (which falls back to it when a
#: caller emits the event with no message). Two copies of this string is how
#: the bar and the event drift into saying different things about the same
#: moment.
QUOTA_EXHAUSTED_DESCRIPTION = "QUOTA EXHAUSTED -- draining, not fetching"

#: Every `ProgressEvent.kind` quantlab's long-running loops emit -- the
#: acquisition fan-out first, and as of 03.5-06 the raw->Zarr chunk loop too.
#: Advisory rather than enforced -- a frozen dataclass validating an enum would
#: make an out-of-repo consumer's forward-compatibility depend on quantlab
#: shipping first -- but exhaustive, and the tuple a reporter can match against
#: instead of hard-coding literals.
#:
#: The two groups are listed separately because they describe two different
#: operations a console renders differently: a `run_*` bar counts BATCHES
#: downloaded from a vendor, a `conversion_*` bar counts WINDOWS densified and
#: appended. `cancelled` is shared, deliberately: it means the same thing in
#: both loops -- a caller's token was observed at a safe boundary -- and a
#: second spelling of it would make a console match two literals to render one
#: state.
EVENT_KINDS = (
    "run_started",
    "coverage",
    "batch_completed",
    "batch_failed",
    "quota_exhausted",
    "cancelled",
    "run_finished",
    "conversion_started",
    "window_written",
    "window_skipped",
    "conversion_finished",
)


@dataclass(frozen=True)
class ProgressEvent:
    """ONE thing that happened during an acquisition or a conversion.

    `kind` is one of `EVENT_KINDS`:

    - ``run_started``     -- a pass is about to fan out. `total` is the batch
      count for the pass and `message` is the bar description.
    - ``coverage``        -- the resume partition for the pass. `detail`
      carries the same counts `_report_coverage` logs.
    - ``batch_completed`` -- one batch's result landed. `completed` rises 1..N
      and `total` is that same batch count.
    - ``batch_failed``    -- reserved for a per-batch failure notification.
    - ``quota_exhausted`` -- the vendor's allocation is gone and the remaining
      batches are being drained, not fetched.
    - ``cancelled``       -- a cancel token was observed at a safe boundary:
      mid-pass in the acquisition loop, between two windows in the chunk loop.
    - ``run_finished``    -- the result generator was drained to completion.

    The raw->Zarr chunk loop (`BaseDataset.from_raw_data_chunked`, 03.5 D-05)
    emits these four:

    - ``conversion_started``  -- the window plan is fixed and the loop is about
      to start. `total` is the PLANNED window count and `detail` carries the
      pinned symbol count, the granularity and the target store path.
    - ``window_written``      -- one window was densified, appended and
      recorded in the ledger. `completed` rises 1..N over the windows this run
      resolved and `total` is that same planned count.
    - ``window_skipped``      -- one window was already in the ledger and was
      not re-appended. A resume is N of these, not silence.
    - ``conversion_finished`` -- the window loop drained, whether it ran to the
      end or stopped on a cancel. `cancelled` is what distinguishes the two,
      and it arrives BEFORE this one.

    `vendor` carries the dataset config's own vendor token on the conversion
    events, falling back to the dataset's class name. A conversion has no
    vendor in the acquisition sense, but the field is REQUIRED and this
    dataclass is frozen and consumed by an out-of-repo reporter -- widening it
    is a contract change where reusing a field is not.

    **`message` is ALWAYS pre-scrubbed by the emitter and is never raw vendor
    exception text.** `Acquisition._scrub` is the single choke point that made
    the failure manifest safe to paste into an issue; Tiingo's error text
    echoes back the full request URL, which carries the API token as a query
    parameter. A progress event is a NEW egress path for exception strings, and
    an egress path that skips the choke point is exactly how the next leak
    happens (03.4-RESEARCH Pitfall 10). Frozen, so a reporter cannot mutate an
    event another reporter is about to receive.
    """

    kind: str
    vendor: str
    completed: int = 0
    total: int = 0
    symbols: tuple[str, ...] = ()
    message: str | None = None
    detail: dict = field(default_factory=dict)


class ProgressReporter(ABC):
    """Where progress events GO.

    ABC + concrete subclasses in the same module, mirroring
    `quantlab/base/backend.py:DataBackend` -- the repo's established shape for
    "one contract, several transports". A module of free functions would not
    satisfy it (CLAUDE.md and MEMORY.md both require class-based, layered
    components here).

    `emit` returns `None` and its return value is IGNORED by the caller. That
    is D-17 expressed in the signature: a reporter cannot stop an acquisition,
    because there is no value it could return that would mean "stop".
    """

    @abstractmethod
    def emit(self, event: ProgressEvent) -> None:
        """Receive one event. Must not raise; if it does, the caller catches
        and logs rather than letting a UI bug end a backfill.
        """
        ...

    def close(self) -> None:
        """Release whatever `emit` acquired. Concrete no-op: most reporters own
        nothing, and forcing every one of them to write an empty override is
        how an ABC accumulates ceremony.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class NullProgressReporter(ProgressReporter):
    """Emits nothing.

    What `config.kwargs["progress"] = False` resolves to, and what any test not
    testing progress should attach so its assertions are not entangled with a
    bar.
    """

    def emit(self, event: ProgressEvent) -> None:
        return None


class TqdmProgressReporter(ProgressReporter):
    """The DEFAULT reporter: the incumbent stderr bar, unchanged.

    Byte-compatible with the block it replaced
    (`Acquisition._run_once`, pre-03.4-05). The four constructor arguments it
    must keep passing to `tqdm` are `total`, `desc`, `unit` and `disable`, with
    `desc` the `f"{VENDOR} {start}..{end}"` string and `unit="batch"`. They are
    asserted directly by `tests/test_acquisition_progress.py`, by patching
    `tqdm` and capturing the kwargs -- never by capturing stderr, whose
    rendered width depends on the terminal.

    The bar is built LAZILY on `run_started`, because that is the first event
    carrying the batch total, and a bar with no total renders a spinner rather
    than a progress bar. A second `run_started` (the resume loop runs a pass
    per attempt) closes the previous bar and opens a fresh one, which is
    exactly what the incumbent per-pass `with tqdm(...)` did.
    """

    def __init__(self, *, disable: bool = False) -> None:
        self._disable = disable
        self._bar = None
        self._switched = False

    def emit(self, event: ProgressEvent) -> None:
        if event.kind == "run_started":
            self.close()
            self._switched = False
            self._bar = tqdm(
                total=event.total,
                desc=event.message,
                unit="batch",
                disable=self._disable,
            )
        elif event.kind == "batch_completed":
            if self._bar is not None:
                self._bar.update(1)
        elif event.kind == "quota_exhausted":
            # Switched ONCE per pass, as before: the description is a state,
            # not a stream, and re-setting it per drained batch is churn.
            if self._bar is not None and not self._switched:
                self._switched = True
                self._bar.set_description(
                    event.message or QUOTA_EXHAUSTED_DESCRIPTION
                )
        elif event.kind == "run_finished":
            self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class CallbackProgressReporter(ProgressReporter):
    """Forwards every event to a caller-supplied callable.

    This is what the console passes: it hands in a bound method that pushes
    onto a Textual message queue, and receives structured counts instead of
    ANSI cursor movements.

    The callable's return value is discarded (D-17). It also may raise without
    consequence -- `Acquisition._emit` wraps every `emit` -- but a reporter
    that raises per batch will log a warning per batch, which is deliberate:
    swallowed silently, a broken console callback would be invisible.
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        self._callback = callback

    def emit(self, event: ProgressEvent) -> None:
        self._callback(event)


class CancelToken:
    """A cooperative stop signal, checked at BATCH BOUNDARIES.

    A named class rather than a bare `threading.Event`, and the reason is the
    consumer: the console imports a TYPE with `cancel()` / `is_cancelled()`
    rather than reimplementing a protocol against a primitive, while the
    primitive underneath stays the thread-safe stdlib one this repository
    already uses for the vendor quota abort (`Acquisition._abort`). A named
    type is also what lets the cancel path be told apart from the quota path at
    a glance, which is 03.4-RESEARCH Pitfall 1 in one line.

    **Separate from the quota abort on purpose.** Reusing `_abort` would make a
    cancel take the quota branch: the run would log "Vendor request allocation
    exhausted" and, with `wait_for_quota=True`, sleep an hour and then resume
    the run the operator just cancelled.

    `cancel()` is IDEMPOTENT by construction, because `Event.set()` is:
    cancelling twice, or cancelling a run that has already finished, does
    nothing the first cancel did not already do.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request a stop. Idempotent; safe from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the signal so the token can be reused for a new run.

        Deliberately NOT called by the acquisition loop: a token is the
        CALLER's object, and a run that silently un-cancelled it would make
        "cancel then start" a race the caller cannot win.
        """
        self._event.clear()

    def __repr__(self) -> str:
        return f"CancelToken(cancelled={self.is_cancelled()})"
