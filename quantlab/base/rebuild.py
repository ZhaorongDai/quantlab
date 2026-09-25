"""Rebuild an on-disk Zarr store from the raw files it was made from.

quantlab keeps data in two tiers. The *raw tier* is the files downloaded from
a vendor, kept as delivered. A *store* is a Zarr directory holding a *panel*,
an ``xarray.Dataset`` indexed by ``timestamp`` and ``symbol``, produced from
the raw tier by conversion code. A store is therefore derived data: once the
conversion code changes, the store on disk no longer matches what the
current code would produce, and it has to be rebuilt.

``BaseStoreRebuilder`` is the vendor-independent skeleton for doing that
safely. It fixes the order of the steps, the checks that refuse to proceed,
and the ``RebuildMeasurement`` that a rebuild returns. A subclass decides
which raw files must exist, which converter to call and what to measure (the
CRSP subclass in ``quantlab/dataset/crsp/rebuild.py`` is an example).

The steps run in this order: ``assert_inputs_present``, ``backup``,
``clear``, ``_convert``, ``_measure``. Checking inputs first means a missing
raw tier is reported while the old store is still on disk. Backing up before
clearing means a conversion that fails halfway leaves a copy of the previous
panel. Measuring last, and only after success, means a half-written store is
never reported as a result.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RebuildMeasurement:
    """Summary of what one rebuild wrote, returned by ``rebuild()``.

    The class is frozen so that the record always shows what the rebuild
    measured and cannot be edited afterwards. ``data_root`` is an absolute
    path and comes first, so a printed measurement shows which directory tree
    was read. That matters when working in a git worktree, which has no
    ``data/`` directory of its own; pointing at the wrong tree would otherwise
    go unnoticed.

    Attributes are documented inline below.

    Examples
    --------
    >>> measurement = rebuilder.rebuild()
    >>> measurement.dims, measurement.data_var_count
    ({'timestamp': 3, 'symbol': 2}, 1)
    >>> measurement.metrics
    {'rows': 6}
    """

    #: The absolute directory the rebuild read from and wrote under.
    data_root: str

    #: The absolute path of the Zarr store that was rebuilt.
    store_path: str

    #: Length of each axis of the rebuilt panel, as ``dict(ds.sizes)``.
    dims: dict[str, int]

    #: Number of data variables in the rebuilt panel.
    data_var_count: int

    #: Measurements defined by the subclass, keyed by metric name.
    metrics: dict[str, int]

    #: Every path ``clear()`` deleted, sorted.
    removed: tuple[str, ...]

    #: Directory the old store was copied to, or ``None`` when there was no
    #: store to copy or no ``backup_dir`` was given.
    backup_path: str | None


class BaseStoreRebuilder(ABC):
    """Abstract base for rebuilding one Zarr store from its raw tier.

    Subclasses set ``SIDECAR_SUFFIXES`` and implement four hooks:
    ``_required_inputs``, ``_convert``, ``_measure`` and ``_measure_dims``.
    ``rebuild()`` runs the whole sequence and returns a
    ``RebuildMeasurement``.

    Parameters
    ----------
    config : object
        The dataset config. Only its ``zarr_file_path`` attribute is read
        here; subclasses may use more.
    data_root : Path or str
        An existing directory under which the raw tier and the store live.
        Keyword-only and required: there is no default and no fallback to
        the current directory. Inside a git worktree the current directory
        has no ``data/`` at all, and a default would let a rebuild report
        success against a tree it never read. From a worktree, pass the
        parent of ``git rev-parse --path-format=absolute --git-common-dir``,
        which is the main repository root.

    Attributes
    ----------
    config : object
        The config passed in.
    data_root : Path
        ``data_root`` resolved to an absolute path.

    Raises
    ------
    ValueError
        If ``data_root`` is not an existing directory.

    Examples
    --------
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

    #: Suffixes appended to ``store_path`` to name the store's *sidecars*, the
    #: small bookkeeping files kept next to the store directory. ``()`` means
    #: the store has none. ``clear()`` deletes them together with the store.
    SIDECAR_SUFFIXES: tuple[str, ...] = ()

    def __init__(self, config, *, data_root: Path | str) -> None:
        """Initialize the rebuilder; see the class docstring for parameters."""
        self.config = config
        resolved = Path(data_root).resolve()
        if not resolved.is_dir():
            raise ValueError(
                f"{type(self).__name__}: data_root {str(resolved)!r} is not an "
                f"existing directory, so there is no tree to rebuild under. "
                f"There is no default and no fallback to the current "
                f"directory, because a guess here could let a rebuild report "
                f"success against a tree it never read. Pass the repository "
                f"root explicitly. Inside a git worktree (where `data/` is "
                f"gitignored and therefore absent) use the parent of "
                f"`git rev-parse --path-format=absolute --git-common-dir`, "
                f"which points at the main repository."
            )
        self._data_root = resolved

    @property
    def data_root(self) -> Path:
        """Absolute directory under which every input and output path lies.

        Examples
        --------
        >>> rebuilder.data_root == Path(repo_root).resolve()
        True
        """
        return self._data_root

    @property
    def store_path(self) -> Path:
        """Path of the Zarr store this rebuilder replaces, from the config.

        Examples
        --------
        >>> rebuilder.store_path.name
        'panel.zarr'
        """
        return Path(str(self.config.zarr_file_path))

    def sidecar_paths(self) -> tuple[Path, ...]:
        """Return ``store_path`` joined with each of ``SIDECAR_SUFFIXES``.

        Paths are returned whether or not they exist. Sidecars sit next to the
        store directory, never inside it, because a Zarr reader walking the
        directory would try to read them as arrays.

        Examples
        --------
        >>> [path.name for path in rebuilder.sidecar_paths()]
        ['panel.zarr.chunks.json']
        """
        return tuple(
            Path(str(self.store_path) + suffix)
            for suffix in self.SIDECAR_SUFFIXES
        )

    def assert_inputs_present(self) -> None:
        """Raise if any required input file is missing, naming every one.

        This runs first in ``rebuild()``, before anything is deleted, so a
        missing raw tier costs an error message rather than a lost store.
        Converting an absent raw tier would write an empty panel over a real
        one, and an empty panel looks the same as a period in which nothing
        traded.

        Raises
        ------
        FileNotFoundError
            If any path returned by ``_required_inputs()`` does not exist.

        Examples
        --------
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
            f"empty panel over the existing store, which afterwards looks the "
            f"same as a period in which nothing traded. Download the raw "
            f"files (and the reference files stored beside them) for this "
            f"vendor before re-running, or point data_root at the tree that "
            f"already holds them."
        )

    def backup(self, dest: Path) -> str | None:
        """Copy the store and every existing sidecar into ``dest``.

        The copy is what makes ``clear()`` reversible. The store about to be
        deleted was written by older conversion code, so the copy is the only
        record of the data earlier results were computed on.

        Parameters
        ----------
        dest : Path
            The directory to copy into. It is created if needed.

        Returns
        -------
        str or None
            ``str(dest)``, or ``None`` when there is no store to copy (the
            first conversion has nothing to preserve).

        Examples
        --------
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
        """Delete the store directory and its sidecars, returning what was removed.

        The sidecars must be deleted with the store. A converter may skip
        writing its sidecars when a store already exists, which is right when
        appending, but after a rebuild it would leave sidecars that describe
        the old panel next to the new one. Calling this twice is harmless:
        ``rebuild()`` always calls it, and on a first conversion there is
        nothing to remove.

        Returns
        -------
        tuple[str, ...]
            The paths that existed and were deleted, sorted.

        Examples
        --------
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
        """Check inputs, back up, clear, convert and measure, in that order.

        See the module docstring for why the order matters. If ``_convert``
        raises, the exception propagates and no measurement is returned, so
        a half-written store is never reported as a result.

        Parameters
        ----------
        backup_dir : Path or None, default None
            Directory to copy the existing store into before it is deleted.
            ``None`` skips the copy; pass a directory whenever the store
            holds real data.

        Returns
        -------
        RebuildMeasurement
            The measurement of the newly written store.

        Examples
        --------
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
        """Return subclass-defined measurements of the newly written store."""

    @abstractmethod
    def _measure_dims(self) -> tuple[dict[str, int], int]:
        """Return the new store's axis lengths and its number of data variables.

        The expected value is ``(dict(ds.sizes), len(ds.data_vars))``.
        """
