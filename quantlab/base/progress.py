"""Progress events and cooperative cancellation for long-running loops.

The acquisition engine in ``quantlab/base/acquisition.py`` and the raw-to-Zarr
conversion loop in ``quantlab/base/data.py`` report progress by handing
``ProgressEvent`` objects to a ``ProgressReporter``. Three reporters ship
here: ``TqdmProgressReporter`` renders the familiar stderr bar and is the
default, ``NullProgressReporter`` discards everything, and
``CallbackProgressReporter`` forwards each event to a caller-supplied callable
so an in-process UI can subscribe without parsing terminal output.
``CancelToken`` is the matching stop signal: a caller sets it and the loop
stops at its next safe boundary.

This module imports only the standard library and ``tqdm``, so any part of
quantlab may import it without creating a cycle. It deliberately does not
catch exceptions raised by reporters: the caller wraps every ``emit`` call and
scrubs credentials from the message before logging it, which only the caller
can do.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

#: Bar description used once a vendor's request allocation is exhausted and
#: the remaining batches are being drained rather than fetched. The emitter
#: puts it on the event's ``message`` and ``TqdmProgressReporter`` falls back
#: to it when the event carries no message, so the bar and the event agree.
QUOTA_EXHAUSTED_DESCRIPTION = "QUOTA EXHAUSTED -- draining, not fetching"

#: Every ``ProgressEvent.kind`` the acquisition loop and the conversion loop
#: emit. Advisory rather than enforced, so an out-of-repo reporter keeps
#: working when a kind is added; match against it instead of hard-coding
#: literals. The ``run_*`` and ``batch_*`` kinds count batches downloaded from
#: a vendor; the ``conversion_*`` and ``window_*`` kinds count windows
#: densified and appended to a store. ``cancelled`` is shared because it means
#: the same thing in both loops.
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
    """One thing that happened during an acquisition or a conversion.

    ``kind`` is one of ``EVENT_KINDS``. The acquisition loop emits
    ``run_started`` (``total`` is the batch count and ``message`` the bar
    description), ``coverage`` (``detail`` carries the resume partition
    counts), ``batch_completed`` (``completed`` rises from 1 to ``total``),
    ``quota_exhausted`` (the remaining batches are drained, not fetched),
    ``cancelled`` and ``run_finished``; ``batch_failed`` is reserved for a
    per-batch failure notification. The conversion loop emits
    ``conversion_started`` (``total`` is the planned window count and
    ``detail`` carries the pinned symbol count, the granularity and the store
    path), ``window_written``, ``window_skipped`` (the window was already in
    the ledger) and ``conversion_finished``. A cancelled conversion emits
    ``cancelled`` before ``conversion_finished``.

    ``vendor`` is the vendor token on acquisition events; conversion events
    reuse the field for the dataset config's vendor, falling back to the
    dataset class name. ``message`` never contains raw vendor exception text:
    the emitter scrubs credentials before building the event, because some
    vendors echo the full request URL, API token included, in their error
    text. The dataclass is frozen so one reporter cannot mutate an event that
    another reporter is about to receive.

    Attributes:
        kind: The event kind, one of ``EVENT_KINDS``.
        vendor: The vendor token, or the dataset's stand-in for one.
        completed: How many units have finished so far.
        total: How many units the pass will process.
        symbols: The symbols the event concerns, when it concerns any.
        message: A human-readable description, already scrubbed.
        detail: Structured extras specific to the kind.
    """

    kind: str
    vendor: str
    completed: int = 0
    total: int = 0
    symbols: tuple[str, ...] = ()
    message: str | None = None
    detail: dict = field(default_factory=dict)


class ProgressReporter(ABC):
    """Destination for progress events.

    Subclasses implement ``emit``; the loop calls it once per event and
    ignores its return value. A reporter therefore cannot stop a run, by
    design: cancellation goes through ``CancelToken``, so a reporter that only
    wants to log cannot halt a multi-hour backfill by accident.

    Example:
        >>> class PrintReporter(ProgressReporter):
        ...     def emit(self, event: ProgressEvent) -> None:
        ...         print(event.kind, event.completed, event.total)
    """

    @abstractmethod
    def emit(self, event: ProgressEvent) -> None:
        """Receive one event.

        Should not raise. If it does, the caller catches and logs the
        exception rather than letting a reporting bug end a backfill.
        """
        ...

    def close(self) -> None:
        """Release anything ``emit`` acquired. The default is a no-op."""
        return None

    def __repr__(self) -> str:
        """Return the class name followed by ``()``."""
        return f"{self.__class__.__name__}()"


class NullProgressReporter(ProgressReporter):
    """Reporter that discards every event.

    This is what ``config.kwargs["progress"] = False`` resolves to, and the
    reporter to attach in tests that are not about progress.
    """

    def emit(self, event: ProgressEvent) -> None:
        """Discard ``event``."""
        return None


class TqdmProgressReporter(ProgressReporter):
    """Default reporter: a ``tqdm`` bar on stderr.

    The bar is built lazily on ``run_started``, the first event that carries
    the batch total, with ``unit="batch"`` and the event's ``message`` as its
    description. Each ``batch_completed`` advances it by one. The first
    ``quota_exhausted`` of a pass switches the description to the event's
    message (or ``QUOTA_EXHAUSTED_DESCRIPTION``); later ones are ignored,
    because the description is a state rather than a stream. ``run_finished``
    closes the bar, and a second ``run_started`` closes the previous bar and
    opens a fresh one, so a resume loop that runs one pass per attempt renders
    one bar per pass.

    Args:
        disable: Passed through to ``tqdm``; when true nothing is rendered.

    Example:
        >>> reporter = TqdmProgressReporter()
        >>> acquisition.attach(reporter=reporter).download()  # doctest: +SKIP
    """

    def __init__(self, *, disable: bool = False) -> None:
        """Create a reporter with no bar open yet."""
        self._disable = disable
        self._bar = None
        self._switched = False

    def emit(self, event: ProgressEvent) -> None:
        """Open, advance, relabel or close the bar according to ``event``."""
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
            # Switch the description once per pass; it is a state, not a stream.
            if self._bar is not None and not self._switched:
                self._switched = True
                self._bar.set_description(
                    event.message or QUOTA_EXHAUSTED_DESCRIPTION
                )
        elif event.kind == "run_finished":
            self.close()

    def close(self) -> None:
        """Close the open bar, if any."""
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class CallbackProgressReporter(ProgressReporter):
    """Reporter that forwards every event to a caller-supplied callable.

    Intended for in-process consumers such as a terminal UI: hand in a bound
    method that pushes onto a message queue and receive structured counts
    instead of terminal cursor movements. The callable's return value is
    discarded. If it raises, the caller logs a warning for that event and the
    run continues; a callback that raises on every batch therefore logs on
    every batch, which keeps a broken consumer visible.

    Args:
        callback: Called once per event with the ``ProgressEvent``.

    Example:
        >>> events = []
        >>> reporter = CallbackProgressReporter(events.append)
        >>> reporter.emit(ProgressEvent(kind="run_started", vendor="x", total=3))
        >>> events[0].total
        3
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Store ``callback``."""
        self._callback = callback

    def emit(self, event: ProgressEvent) -> None:
        """Pass ``event`` to the callback and discard its result."""
        self._callback(event)


class CancelToken:
    """Cooperative stop signal checked at batch boundaries.

    A caller creates a token, attaches it to a run and calls ``cancel()`` from
    any thread. The run observes the token at its next safe boundary, emits a
    ``cancelled`` event and stops; work already in flight completes. The token
    wraps a ``threading.Event``, so ``cancel()`` is idempotent and thread-safe.
    It is kept separate from the vendor-quota abort on purpose: a cancel must
    never take the quota path, which may sleep and then resume the run the
    operator just cancelled.

    Example:
        >>> token = CancelToken()
        >>> token.is_cancelled()
        False
        >>> token.cancel()
        >>> token.is_cancelled()
        True
    """

    def __init__(self) -> None:
        """Create a token in the not-cancelled state."""
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request a stop. Idempotent and safe to call from any thread."""
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether ``cancel()`` has been called since the last reset."""
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the signal so the token can be reused for a new run.

        The acquisition loop never calls this: the token belongs to the
        caller, and a run that silently un-cancelled it would make "cancel
        then start" a race the caller cannot win.
        """
        self._event.clear()

    def __repr__(self) -> str:
        """Return ``CancelToken(cancelled=...)``."""
        return f"CancelToken(cancelled={self.is_cancelled()})"
