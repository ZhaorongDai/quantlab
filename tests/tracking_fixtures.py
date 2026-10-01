"""A tracker that records every call, for tests that observe tracking.

Put a ``RecordingTracker`` in a model or backtest config and read what an
outsider would see: which runs were opened (project, group, name, config),
their step metrics, summaries, config updates, tables and files, and how
each run ended. It lives in the tests, not the library (spec #99).
"""

from dataclasses import dataclass, field
from pathlib import Path

from quantlab.base.tracking import Tracker, TrackingRun


class RecordedRun(TrackingRun):
    """One run a ``RecordingTracker`` opened, with everything written to it."""

    def __init__(self, *, project, group, name, config):
        self.project = project
        self.group = group
        self.name = name
        self.config = config
        self.steps: list[tuple[int, dict]] = []
        self.summary: dict = {}
        self.config_updates: list[dict] = []
        self.tables: dict[str, tuple[list, list, int | None]] = {}
        self.files: list[Path] = []
        self.finished = False
        self.failed: bool | None = None

    def _log(self, metrics, step):
        self.steps.append((step, metrics))

    def _summarize(self, metrics):
        self.summary.update(metrics)

    def _update_config(self, params):
        self.config_updates.append(params)

    def _log_table(self, name, columns, rows, top_bars):
        self.tables[name] = (columns, rows, top_bars)

    def _log_file(self, path):
        self.files.append(path)

    def _finish(self, *, failed):
        assert not self.finished, "a run was finished twice"
        self.finished = True
        self.failed = failed


@dataclass(frozen=True, kw_only=True)
class RecordingTracker(Tracker):
    """Record every run opened through it, in ``runs``, in opening order.

    ``runs`` is shared by every copy of the tracker (``dataclasses.replace``
    or a deep-copied config keep pointing at the same list), so a test reads
    it off the tracker it put in the config. It is not part of the config.
    """

    runs: list = field(default_factory=list, compare=False, repr=False)

    def get_config(self) -> dict:
        """Return the config without ``runs``, which is not a setting."""
        return {"project": self.project, "name": self.import_path}

    def __deepcopy__(self, memo):
        return self

    def _open(self, *, project, group, name, config):
        run = RecordedRun(project=project, group=group, name=name, config=config)
        self.runs.append(run)
        return run
