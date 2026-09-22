"""Offline store REBUILD -- the framework-agnostic skeleton (phase 03.11 W0).

**Why this layer exists at all.** A Zarr store on disk is a *derived* artefact:
it is whatever the conversion code produced on the day it ran. When that code
is fixed, every store written before the fix keeps describing a panel the
current tree can no longer produce -- and every number measured on it is a
measurement of deleted code. Phase 03.11's W0 hit exactly that: the on-disk
CRSP sidecars are timestamped 13:46/13:47 while the commit that fixed the
no-price-sentinel adjustment anchor landed at 16:03, so the audit's headline
counts were all pre-fix. Re-running the conversion is therefore not a
convenience task; it is the precondition for every later acceptance criterion.

**The shape.** This module holds only what is true of ANY store: the four-step
order, the refusals, the measurement carrier. Everything vendor-specific --
which sidecars exist, which converter to call, what to count -- is a subclass's
job, mirroring the `base/data.py` -> `dataset/stock.py` split the rest of the
project already uses. See `quantlab/dataset/crsp/rebuild.py` for the CRSP
implementation.

**The order is the safety property**, not an implementation detail::

    assert_inputs_present() -> backup() -> clear() -> _convert() -> _measure()

`clear()` deletes a real store. Putting `assert_inputs_present()` first means a
missing raw tier costs nothing: the refusal happens while the old store is
still on disk. Putting `backup()` before `clear()` means even a conversion that
fails halfway leaves the operator with the previous panel. `_measure()` runs
last and only on success, because measuring a store that `_convert()` never
finished writing is how a half-written panel becomes a quoted number.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RebuildMeasurement:
    """Everything one rebuild produced, in the form a SUMMARY quotes verbatim.

    Frozen because it is evidence. A caller that could mutate a field after the
    fact turns "what the rebuild measured" into "what somebody last wrote
    here", and the whole reason this object exists is that phase 03.11 could no
    longer trust numbers whose provenance had gone quiet.

    `data_root` is an ABSOLUTE path STRING rather than a `Path`, and it is
    first on purpose. Executions run inside a git worktree while the data lives
    in the main repository (`data/` is gitignored, so it is simply absent from
    a worktree); the recorded accident is an executor reading the wrong tree
    and reporting a false green. Printing this field is the only cheap way an
    operator can see WHICH tree was read.
    """

    #: The absolute filesystem root the rebuild read and wrote under.
    data_root: str

    #: The absolute path of the Zarr store that was rebuilt.
    store_path: str

    #: `dict(ds.sizes)` of the rebuilt panel -- axis name to length.
    dims: dict[str, int]

    #: `len(ds.data_vars)` of the rebuilt panel.
    data_var_count: int

    #: The subclass's own measurements, keyed by metric name.
    metrics: dict[str, int]

    #: Every path `clear()` actually deleted, sorted.
    removed: tuple[str, ...]

    #: Where the pre-rebuild store was copied, or `None` when there was no
    #: store to copy (or no `backup_dir` was asked for).
    backup_path: str | None


class BaseStoreRebuilder(ABC):
    """Rebuild one on-disk store from its raw tier, safely and measurably.

    Subclasses declare `SIDECAR_SUFFIXES` and implement four hooks:
    `_required_inputs`, `_convert`, `_measure` and `_measure_dims`.
    """

    #: The suffixes appended to `store_path` to name every sidecar file that
    #: belongs to this store. Declared by the subclass; `()` means the store
    #: has no sidecars. These are the files `clear()` deletes alongside the
    #: store directory -- see `clear()` for why that is not optional.
    SIDECAR_SUFFIXES: tuple[str, ...] = ()

    def __init__(self, config, *, data_root: Path | str) -> None:
        """Bind a config and the filesystem root every path resolves under.

        `data_root` is KEYWORD-ONLY, has NO DEFAULT and gets NO cwd fallback,
        and all three are deliberate. The caller always knows which tree it
        means; this class never does. A default would silently make "the
        current directory" the answer, which inside a git worktree is the
        worktree -- the exact tree that has no `data/` at all, and the exact
        way a rebuild reports success against data it never touched.

        In a worktree the right value is::

            dirname( git rev-parse --path-format=absolute --git-common-dir )

        which resolves to the MAIN repository root from inside a worktree and
        to the repository root from the main tree, so one expression serves
        both.
        """
        self.config = config
        resolved = Path(data_root).resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"{type(self).__name__}: data_root {str(resolved)!r} is not an "
                f"existing directory, so there is no tree to rebuild under. "
                f"This class has no default and no cwd fallback on purpose -- "
                f"guessing here is how a rebuild reports success against a "
                f"tree it never read. Pass the repository root explicitly; "
                f"inside a git worktree (where `data/` is gitignored and "
                f"therefore absent) the correct value is the parent of "
                f"`git rev-parse --path-format=absolute --git-common-dir`, "
                f"which points at the MAIN repository."
            )
        self._data_root = resolved

    @property
    def data_root(self) -> Path:
        """The resolved absolute root every input and output path sits under."""
        return self._data_root

    @property
    def store_path(self) -> Path:
        """The Zarr store this rebuilder replaces, from the config."""
        return Path(str(self.config.zarr_file_path))

    def sidecar_paths(self) -> tuple[Path, ...]:
        """`store_path` + each entry of `SIDECAR_SUFFIXES`, existing or not.

        Sidecars are SIBLINGS of the store directory, never files inside it:
        a path inside the `.zarr` directory would be read as an array by any
        Zarr reader that walked it.
        """
        return tuple(
            Path(str(self.store_path) + suffix)
            for suffix in self.SIDECAR_SUFFIXES
        )

    def assert_inputs_present(self) -> None:
        """Refuse, naming names, when any required input is missing.

        Runs FIRST in `rebuild()`, before anything destructive, so a missing
        raw tier costs an error message rather than a deleted store.

        The message follows the house style for a refusal
        (`quantlab/backend.py:XrBackend.widen_symbol_axis`): collect the
        offenders, describe them individually, say what cannot be done without
        them, and end with the remedy. A rebuild that quietly converted an
        absent raw tier would write an EMPTY panel over a real one, and an
        empty panel is not distinguishable afterwards from a market in which
        nothing traded.
        """
        missing = [
            path for path in self._required_inputs() if not Path(path).exists()
        ]
        if not missing:
            return
        described = "\n".join(f"  - {str(Path(path))}" for path in missing)
        raise FileNotFoundError(
            f"{type(self).__name__}: refusing to rebuild "
            f"{str(self.store_path)!r} -- the following required input(s) do "
            f"not exist under data_root {str(self.data_root)!r}:\n{described}\n"
            f"Without them the conversion reads nothing and would write an "
            f"EMPTY panel over the existing store, which afterwards is "
            f"indistinguishable from a period in which nothing traded. Pull "
            f"the raw tier (and its sibling reference tier) for this vendor "
            f"before re-running, or point data_root at the tree that already "
            f"holds them."
        )

    def backup(self, dest: Path) -> str | None:
        """Copy the store and every existing sidecar into `dest`.

        Returns `str(dest)`, or `None` when there is no store to copy -- a
        first-ever conversion has nothing to preserve and must not be turned
        into an error by the safety net that exists for the other case.

        This is what makes `clear()` a reversible decision. The pre-fix store
        it deletes can no longer be REPRODUCED (the code path that wrote it is
        gone, which is the whole reason for the rebuild), so the copy is the
        only remaining record of what the old numbers were measured on.
        """
        store = self.store_path
        if not store.exists():
            return None
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(store, dest / store.name, dirs_exist_ok=True)
        for sidecar in self.sidecar_paths():
            if sidecar.exists():
                shutil.copy2(sidecar, dest / sidecar.name)
        return str(dest)

    def clear(self) -> tuple[str, ...]:
        """Delete the store directory AND every sidecar; return what went.

        **Deleting the sidecars is not tidiness -- it is correctness.**
        `dataset/crsp/__init__.py:_write_identity_reports` opens with a store-exists
        guard (crsp/__init__.py:1414): if the store is already on disk, the audit
        sidecars are left untouched. That guard is right for an APPEND, whose
        reports would otherwise be replaced by numbers for a panel that was
        refused and never written. But it means a rebuild that removed only
        the store directory would finish with sidecars describing the PREVIOUS
        panel sitting beside the new one -- an audit trail that is confidently
        wrong, which is worse than none. So the store and its sidecars are
        cleared together, always.

        Idempotent by design: `rebuild()` calls this unconditionally, and a
        first-ever conversion has nothing to remove. Returns the paths that
        actually existed, sorted, so the caller can record what it destroyed.
        """
        removed: list[str] = []
        store = self.store_path
        if store.exists():
            removed.append(str(store))
            shutil.rmtree(store, ignore_errors=True)
        for sidecar in self.sidecar_paths():
            if sidecar.exists():
                removed.append(str(sidecar))
                sidecar.unlink(missing_ok=True)
        return tuple(sorted(removed))

    def rebuild(self, *, backup_dir: Path | None = None) -> RebuildMeasurement:
        """Run the full four-step rebuild and return its measurement.

        Order: `assert_inputs_present` -> `backup` -> `clear` -> `_convert` ->
        `_measure`. See the module docstring for why each position is load
        bearing. `_measure` is reached only when `_convert` returned, so an
        exception mid-conversion propagates with no measurement attached
        rather than producing numbers for a half-written store.

        `backup_dir=None` skips the copy. Every caller that is about to
        destroy a real panel should pass one.
        """
        self.assert_inputs_present()
        backup_path = self.backup(backup_dir) if backup_dir is not None else None
        removed = self.clear()
        self._convert()
        metrics = self._measure()
        dims, data_var_count = self._measure_dims()
        return RebuildMeasurement(
            data_root=str(self.data_root),
            store_path=str(self.store_path),
            dims=dims,
            data_var_count=data_var_count,
            metrics=metrics,
            removed=removed,
            backup_path=backup_path,
        )

    @abstractmethod
    def _required_inputs(self) -> tuple[Path, ...]:
        """Every path that must exist before the conversion may start."""

    @abstractmethod
    def _convert(self) -> object:
        """Run the conversion that writes the store. Return whatever it gives."""

    @abstractmethod
    def _measure(self) -> dict[str, int]:
        """Measure the freshly-written store. Keys are the subclass's contract."""

    @abstractmethod
    def _measure_dims(self) -> tuple[dict[str, int], int]:
        """`(dict(ds.sizes), len(ds.data_vars))` of the freshly-written store."""
