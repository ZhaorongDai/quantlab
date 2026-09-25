"""Rebuilding an on-disk store from its raw tier.

A Zarr store is a derived artefact: it holds whatever the conversion code
produced on the day it ran, and once that code changes the store describes a
panel the current code can no longer produce. ``BaseStoreRebuilder`` is the
vendor-agnostic skeleton for re-running a conversion safely. It fixes the
order of operations, the refusals, and the ``RebuildMeasurement`` record a
rebuild returns; which raw files must exist, which converter to call and what
to count are a subclass's job (see ``quantlab/dataset/crsp/rebuild.py``).

The order is the safety property::

    assert_inputs_present() -> backup() -> clear() -> _convert() -> _measure()

Checking inputs first means a missing raw tier is refused while the old store
is still on disk. Backing up before clearing means a conversion that fails
halfway still leaves the previous panel available. Measuring last, and only
on success, means a half-written store is never quoted as a number.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RebuildMeasurement:
    """The record one rebuild produced, quoted verbatim by a summary.

    Frozen because it is evidence: a mutable field would turn "what the
    rebuild measured" into "what somebody last wrote here". ``data_root`` is
    an absolute path string and comes first so that a printed measurement
    shows which tree was read; a rebuild run from a git worktree could
    otherwise report success against a ``data/`` directory that is absent
    there.

    Example:
        >>> measurement = rebuilder.rebuild()
        >>> measurement.dims, measurement.data_var_count
        ({'timestamp': 3, 'symbol': 2}, 1)
        >>> measurement.metrics
        {'rows': 6}
    """

    #: The absolute filesystem root the rebuild read and wrote under.
    data_root: str

    #: The absolute path of the Zarr store that was rebuilt.
    store_path: str

    #: ``dict(ds.sizes)`` of the rebuilt panel: axis name to length.
    dims: dict[str, int]

    #: ``len(ds.data_vars)`` of the rebuilt panel.
    data_var_count: int

    #: The subclass's own measurements, keyed by metric name.
    metrics: dict[str, int]

    #: Every path ``clear()`` actually deleted, sorted.
    removed: tuple[str, ...]

    #: Where the pre-rebuild store was copied, or ``None`` when there was no
    #: store to copy or no ``backup_dir`` was given.
    backup_path: str | None


class BaseStoreRebuilder(ABC):
    """Rebuild one on-disk store from its raw tier, safely and measurably.

    Subclasses declare ``SIDECAR_SUFFIXES`` and implement the four hooks
    ``_required_inputs``, ``_convert``, ``_measure`` and ``_measure_dims``.
    ``rebuild()`` runs the whole sequence and returns a ``RebuildMeasurement``.

    Example:
        A subclass names its sidecars and fills in the hooks::

            class MyRebuilder(BaseStoreRebuilder):
                SIDECAR_SUFFIXES = (".chunks.json",)

                def _required_inputs(self):
                    return (self.data_root / "raw" / "prices.parquet",)

                ...

            measurement = MyRebuilder(config, data_root=repo_root).rebuild(
                backup_dir=repo_root / "backup"
            )

        The method examples below use ``rebuilder = MyRebuilder(config,
        data_root=repo_root)`` with ``SIDECAR_SUFFIXES = (".chunks.json",)``
        and a store already on disk.
    """

    #: The suffixes appended to ``store_path`` to name every sidecar file that
    #: belongs to this store. ``()`` means the store has no sidecars. These
    #: are deleted together with the store directory by ``clear()``.
    SIDECAR_SUFFIXES: tuple[str, ...] = ()

    def __init__(self, config, *, data_root: Path | str) -> None:
        """Bind a config and the filesystem root every path resolves under.

        ``data_root`` is keyword-only with no default and no fallback to the
        current directory. Inside a git worktree the current directory has no
        ``data/`` at all, and a default would let a rebuild report success
        against a tree it never read. From a worktree the right value is the
        parent of ``git rev-parse --path-format=absolute --git-common-dir``,
        which is the main repository root.

        Args:
            config: The dataset config naming ``zarr_file_path``.
            data_root: An existing directory the raw tier and store sit under.

        Raises:
            ValueError: If ``data_root`` is not an existing directory.
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
        """The resolved absolute root every input and output path sits under.

        Example:
            >>> rebuilder.data_root == Path(repo_root).resolve()
            True
        """
        return self._data_root

    @property
    def store_path(self) -> Path:
        """The Zarr store this rebuilder replaces, taken from the config.

        Example:
            >>> rebuilder.store_path.name
            'panel.zarr'
        """
        return Path(str(self.config.zarr_file_path))

    def sidecar_paths(self) -> tuple[Path, ...]:
        """Return ``store_path`` joined with each of ``SIDECAR_SUFFIXES``.

        Paths are returned whether or not they exist. Sidecars are siblings of
        the store directory, never files inside it, because a Zarr reader
        walking the directory would try to read them as arrays.

        Example:
            >>> [path.name for path in rebuilder.sidecar_paths()]
            ['panel.zarr.chunks.json']
        """
        return tuple(
            Path(str(self.store_path) + suffix)
            for suffix in self.SIDECAR_SUFFIXES
        )

    def assert_inputs_present(self) -> None:
        """Refuse, naming every missing path, when a required input is absent.

        Runs first in ``rebuild()``, before anything destructive, so a missing
        raw tier costs an error message rather than a deleted store. A rebuild
        that converted an absent raw tier would write an empty panel over a
        real one, and afterwards that is indistinguishable from a period in
        which nothing traded.

        Raises:
            FileNotFoundError: If any path from ``_required_inputs()`` does
                not exist.

        Example:
            >>> rebuilder.assert_inputs_present()  # every input exists
            >>> Path(repo_root, "raw", "prices.parquet").unlink()
            >>> rebuilder.assert_inputs_present()
            Traceback (most recent call last):
                ...
            FileNotFoundError: MyRebuilder: refusing to rebuild ...
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
        """Copy the store and every existing sidecar into ``dest``.

        This is what makes ``clear()`` reversible: the store it deletes was
        written by code that no longer exists, so the copy is the only record
        of what earlier numbers were measured on.

        Args:
            dest: The directory to copy into; created if needed.

        Returns:
            ``str(dest)``, or ``None`` when there is no store to copy, since a
            first-ever conversion has nothing to preserve.

        Example:
            >>> backup_dir = Path(repo_root, "backup")
            >>> rebuilder.backup(backup_dir) == str(backup_dir)
            True
            >>> sorted(path.name for path in backup_dir.iterdir())
            ['panel.zarr', 'panel.zarr.chunks.json']
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
        """Delete the store directory and every sidecar; return what went.

        The sidecars must go with the store. A converter that writes audit
        sidecars may skip them when a store already exists, which is right
        for an append but would leave a rebuild with sidecars describing the
        previous panel beside the new one. Idempotent: ``rebuild()`` calls it
        unconditionally, and a first conversion has nothing to remove.

        Returns:
            The paths that existed and were deleted, sorted.

        Example:
            >>> [Path(path).name for path in rebuilder.clear()]
            ['panel.zarr', 'panel.zarr.chunks.json']
            >>> rebuilder.clear()
            ()
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
        """Run the full sequence and return its measurement.

        The order is ``assert_inputs_present``, ``backup``, ``clear``,
        ``_convert``, ``_measure``; see the module docstring for why each
        position matters. An exception inside ``_convert`` propagates with no
        measurement attached rather than producing numbers for a half-written
        store.

        Args:
            backup_dir: Where to copy the existing store first. ``None`` skips
                the copy; every caller about to destroy a real panel should
                pass one.

        Returns:
            The ``RebuildMeasurement`` for the freshly written store.

        Example:
            >>> measurement = rebuilder.rebuild(backup_dir=Path(repo_root, "backup"))
            >>> measurement.dims, measurement.data_var_count
            ({'timestamp': 3, 'symbol': 2}, 1)
            >>> [Path(path).name for path in measurement.removed]
            ['panel.zarr', 'panel.zarr.chunks.json']
            >>> measurement.backup_path == str(Path(repo_root, "backup"))
            True
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
        """Return every path that must exist before the conversion may start."""

    @abstractmethod
    def _convert(self) -> object:
        """Run the conversion that writes the store and return its result."""

    @abstractmethod
    def _measure(self) -> dict[str, int]:
        """Measure the freshly written store; the keys are the subclass's."""

    @abstractmethod
    def _measure_dims(self) -> tuple[dict[str, int], int]:
        """Return ``(dict(ds.sizes), len(ds.data_vars))`` of the new store."""
