"""Progress events and cooperative cancellation for long-running loops.

Two loops in quantlab can run for hours: the acquisition engine, which
downloads data from a vendor in batches, and the conversion loop, which turns
raw vendor files into a Zarr store one time window at a time. Both report
progress by handing ``ProgressEvent`` objects to a ``ProgressReporter``.

Three reporters are provided. ``TqdmProgressReporter`` draws a progress bar
on stderr and is the default. ``NullProgressReporter`` discards everything.
``CallbackProgressReporter`` forwards each event to a function you supply, so
an in-process user interface can follow a run without parsing terminal
output. ``CancelToken`` is the matching stop signal: the caller sets it, and
the loop stops at its next safe point.

This module imports only the standard library and ``tqdm``, so any part of
quantlab can import it without creating an import cycle. Reporters' exceptions
are not caught here. The loop that calls ``emit`` catches them and removes
credentials from the message before logging it, because only that loop knows
which credentials are in play.
"""

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

from tqdm import tqdm

#: Progress-bar label shown once the vendor's request quota is used up. From
#: then on the remaining batches are skipped ("drained") instead of fetched.
#: The loop puts this text on the event's ``message``, and
#: ``TqdmProgressReporter`` falls back to it when the event has no message, so
#: the bar and the event always say the same thing.
QUOTA_EXHAUSTED_DESCRIPTION = "QUOTA EXHAUSTED -- draining, not fetching"

#: Every ``ProgressEvent.kind`` the two loops emit. The list is informative,
#: not enforced, so a reporter written outside this repository keeps working
#: when a new kind is added; compare against this tuple rather than hard-coding
#: strings. The ``run_*`` and ``batch_*`` kinds count batches downloaded from a
#: vendor. The ``conversion_*`` and ``window_*`` kinds count time windows
#: converted and appended to a store. ``cancelled`` means the same in both.
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

    The acquisition loop emits ``run_started`` (``total`` is the number of
    batches, ``message`` the bar label), ``coverage`` (``detail`` holds counts
    of symbols already on disk versus still to fetch), ``batch_completed``
    (``completed`` rises from 1 to ``total``), ``quota_exhausted`` (the rest of
    the batches are skipped, not fetched), ``cancelled`` and ``run_finished``.
    ``batch_failed`` is reserved for reporting a single failed batch.

    The conversion loop emits ``conversion_started`` (``total`` is the number
    of planned time windows; ``detail`` holds the symbol count, the window
    size and the store path), then ``window_written`` or ``window_skipped``
    for each window (skipped means an earlier run already wrote it), and
    finally ``conversion_finished``. A cancelled conversion emits
    ``cancelled`` just before ``conversion_finished``.

    ``message`` never contains raw vendor error text. Some vendors echo the
    full request URL, API key included, in their errors, so the loop removes
    credentials before building the event. The class is frozen so that one
    reporter cannot change an event another reporter is about to receive.

    Attributes
    ----------
    kind : str
        The event kind, one of ``EVENT_KINDS``.
    vendor : str
        The vendor name (for example ``"tiingo"``). Conversion events use the
        dataset config's vendor, or the dataset class name if it has none.
    completed : int
        How many batches or windows have finished so far.
    total : int
        How many batches or windows the run will process.
    symbols : tuple[str, ...]
        The symbols the event concerns, when it concerns any.
    message : str | None
        A human-readable description with credentials already removed.
    detail : dict
        Extra structured data whose keys depend on ``kind``.

    Examples
    --------
    >>> event = ProgressEvent(kind="batch_completed", vendor="tiingo",
    ...                       completed=3, total=10, symbols=("AAPL",))
    >>> event.kind, event.completed, event.total
    ('batch_completed', 3, 10)
    """

    kind: str
    vendor: str
    completed: int = 0
    total: int = 0
    symbols: tuple[str, ...] = ()
    message: str | None = None
    detail: dict = field(default_factory=dict)


class ProgressReporter(ABC):
    """Abstract destination for progress events.

    Subclasses implement ``emit``. The loop calls it once per event and
    ignores the return value, so a reporter cannot stop a run. That is
    intentional: stopping goes through ``CancelToken``, so a reporter that
    only wants to log can never halt a multi-hour download by accident.

    Examples
    --------
    >>> class PrintReporter(ProgressReporter):
    ...     def emit(self, event: ProgressEvent) -> None:
    ...         print(event.kind, event.completed, event.total)
    """

    @abstractmethod
    def emit(self, event: ProgressEvent) -> None:
        """Receive one event.

        Implementations should not raise. If one does, the calling loop logs
        the exception and carries on, so a bug in reporting cannot end a
        long download.

        Parameters
        ----------
        event : ProgressEvent
            The event to handle.

        Examples
        --------
        >>> class ListReporter(ProgressReporter):
        ...     def __init__(self):
        ...         self.events = []
        ...     def emit(self, event: ProgressEvent) -> None:
        ...         self.events.append(event)
        """
        ...

    def close(self) -> None:
        """Release any resource ``emit`` opened. The default does nothing.

        Examples
        --------
        >>> class FileReporter(ProgressReporter):
        ...     def __init__(self, path):
        ...         self._handle = open(path, "a")
        ...     def emit(self, event: ProgressEvent) -> None:
        ...         print(event.kind, file=self._handle)
        ...     def close(self) -> None:
        ...         self._handle.close()
        """
        return None

    def __repr__(self) -> str:
        """Return the class name followed by ``()``."""
        return f"{self.__class__.__name__}()"


class NullProgressReporter(ProgressReporter):
    """Reporter that discards every event.

    Setting ``config.kwargs["progress"] = False`` selects this reporter. It is
    also the one to use in tests that are not about progress.

    Examples
    --------
    >>> reporter = NullProgressReporter()
    >>> reporter.emit(ProgressEvent(kind="run_started", vendor="tiingo",
    ...                             total=3))
    >>> reporter.close()
    """

    def emit(self, event: ProgressEvent) -> None:
        """Discard ``event``.

        Parameters
        ----------
        event : ProgressEvent
            The event, which is ignored.

        Examples
        --------
        >>> NullProgressReporter().emit(
        ...     ProgressEvent(kind="cancelled", vendor="tiingo")
        ... )
        """
        return None


class TqdmProgressReporter(ProgressReporter):
    """Default reporter: a ``tqdm`` progress bar on stderr.

    The bar is created on ``run_started``, the first event that knows the
    number of batches, with the event's ``message`` as its label. Each
    ``batch_completed`` advances it by one. The first ``quota_exhausted`` in a
    run changes the label to the event's message (or
    ``QUOTA_EXHAUSTED_DESCRIPTION``); repeats are ignored because the label
    describes a state that has already been shown. ``run_finished`` closes
    the bar. A new ``run_started`` closes any open bar and starts a fresh one,
    so a retry loop that makes several passes shows one bar per pass.

    This reporter only reacts to acquisition events; conversion events are
    ignored.

    Parameters
    ----------
    disable : bool, default False
        Passed to ``tqdm``. When true, nothing is drawn.

    Examples
    --------
    >>> reporter = TqdmProgressReporter()
    >>> acquisition.attach(reporter=reporter).download()  # doctest: +SKIP
    """

    def __init__(self, *, disable: bool = False) -> None:
        """Initialize the reporter; see the class docstring for parameters."""
        self._disable = disable
        self._bar = None
        self._switched = False

    def emit(self, event: ProgressEvent) -> None:
        """Open, advance, relabel or close the bar according to ``event``.

        Parameters
        ----------
        event : ProgressEvent
            The event to render. Kinds other than ``run_started``,
            ``batch_completed``, ``quota_exhausted`` and ``run_finished`` are
            ignored.

        Examples
        --------
        >>> reporter = TqdmProgressReporter(disable=True)
        >>> reporter.emit(ProgressEvent(kind="run_started", vendor="tiingo",
        ...                             total=2, message="tiingo 1d"))
        >>> reporter.emit(ProgressEvent(kind="batch_completed",
        ...                             vendor="tiingo", completed=1, total=2))
        >>> reporter.emit(ProgressEvent(kind="run_finished", vendor="tiingo",
        ...                             completed=2, total=2))
        """
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
            # Relabel once per run: the label shows a state, not a stream of events.
            if self._bar is not None and not self._switched:
                self._switched = True
                self._bar.set_description(
                    event.message or QUOTA_EXHAUSTED_DESCRIPTION
                )
        elif event.kind == "run_finished":
            self.close()

    def close(self) -> None:
        """Close the open bar, if any.

        Examples
        --------
        >>> reporter = TqdmProgressReporter(disable=True)
        >>> reporter.close()  # a no-op while no bar is open
        """
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class CallbackProgressReporter(ProgressReporter):
    """Reporter that forwards every event to a function you supply.

    Use it for consumers in the same process, such as a terminal user
    interface: pass a function that pushes each event onto a queue, and you
    receive structured counts instead of a drawn progress bar. The function's
    return value is ignored. If it raises, the loop logs a warning for that
    event and continues, so a callback that fails on every batch logs on
    every batch and the problem stays visible.

    Parameters
    ----------
    callback : Callable[[ProgressEvent], None]
        Called once per event with the ``ProgressEvent``.

    Examples
    --------
    >>> events = []
    >>> reporter = CallbackProgressReporter(events.append)
    >>> reporter.emit(ProgressEvent(kind="run_started", vendor="x", total=3))
    >>> events[0].total
    3
    """

    def __init__(self, callback: Callable[[ProgressEvent], None]) -> None:
        """Initialize the reporter; see the class docstring for parameters."""
        self._callback = callback

    def emit(self, event: ProgressEvent) -> None:
        """Pass ``event`` to the callback and discard its result.

        Parameters
        ----------
        event : ProgressEvent
            The event to forward.

        Examples
        --------
        >>> events = []
        >>> reporter = CallbackProgressReporter(events.append)
        >>> reporter.emit(ProgressEvent(kind="cancelled", vendor="tiingo"))
        >>> events[-1].kind
        'cancelled'
        """
        self._callback(event)


class CancelToken:
    """Cooperative stop signal that a long loop checks between steps.

    "Cooperative" means the loop is not interrupted; it checks the token at
    safe points (between batches or between time windows) and stops there.
    Create a token, pass it to a run, and call ``cancel()`` from any thread.
    At its next check the run emits a ``cancelled`` event and stops; work
    already in progress is allowed to finish. The token wraps a
    ``threading.Event``, so ``cancel()`` is thread-safe and calling it twice
    is harmless.

    Cancellation is deliberately separate from the handling of an exhausted
    vendor quota. That handling may wait and then resume the run, which is
    the opposite of what a user who pressed cancel wants.

    Examples
    --------
    >>> token = CancelToken()
    >>> token.is_cancelled()
    False
    >>> token.cancel()
    >>> token.is_cancelled()
    True
    """

    def __init__(self) -> None:
        """Initialize the token in the not-cancelled state."""
        self._event = threading.Event()

    def cancel(self) -> None:
        """Request a stop. Safe to call more than once and from any thread.

        Examples
        --------
        >>> token = CancelToken()
        >>> token.cancel()
        >>> token
        CancelToken(cancelled=True)
        """
        self._event.set()

    def is_cancelled(self) -> bool:
        """Return whether ``cancel()`` has been called since the last reset.

        Examples
        --------
        >>> token = CancelToken()
        >>> token.is_cancelled()
        False
        """
        return self._event.is_set()

    def reset(self) -> None:
        """Clear the signal so the token can be reused for a new run.

        The loops never call this themselves. The token belongs to the
        caller; if a run cleared it on start, a cancel issued just before the
        run began would be silently lost.

        Examples
        --------
        >>> token = CancelToken()
        >>> token.cancel()
        >>> token.reset()
        >>> token.is_cancelled()
        False
        """
        self._event.clear()

    def __repr__(self) -> str:
        """Return ``CancelToken(cancelled=...)``."""
        return f"CancelToken(cancelled={self.is_cancelled()})"
