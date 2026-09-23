"""Atomic JSON file writes.

Every JSON sidecar quantlab persists (resume ledgers, acquisition watermarks,
failure manifests) goes through ``write_json_atomically``. The payload is
written to a temporary file in the destination's own directory and then
renamed over the destination, so an interrupted write leaves either the
previous complete file or the new one, never a truncated file that a resumed
run would fail to parse.

No lock is taken: two concurrent writers still race, but the loser sees a
complete file rather than a fragment. This module imports only the standard
library so any layer can use it without creating an import cycle.
"""

import json
import os
import tempfile
from pathlib import Path

__all__ = ["write_json_atomically"]


def write_json_atomically(path: str | Path, payload: object, **json_kwargs) -> None:
    """Serialise ``payload`` as JSON to ``path`` with an atomic replace.

    Missing parent directories are created. ``json_kwargs`` is forwarded
    unchanged to ``json.dump``, so each caller keeps its own formatting
    (compact, ``indent=2``, ``sort_keys=True`` and so on).

    On any exception, including ``KeyboardInterrupt``, the temporary file is
    removed and the exception propagates, leaving the previous file (if any)
    untouched.

    Args:
        path: Destination file.
        payload: Any object ``json.dump`` accepts.
        **json_kwargs: Formatting options passed to ``json.dump``.

    Example:
        >>> write_json_atomically("run/watermark.json", {"last": "2024-01-31"})
        >>> write_json_atomically("run/manifest.json", data, indent=2, sort_keys=True)
    """
    destination = str(path)
    directory = Path(destination).parent
    directory.mkdir(parents=True, exist_ok=True)
    # The temp file must live in the destination's directory: `os.replace` is
    # atomic only as a same-filesystem rename. Staged in the system temp dir it
    # could become a cross-device copy, which is interruptible.
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=str(directory),
        prefix=Path(destination).name + ".",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            json.dump(payload, handle, **json_kwargs)
            handle.flush()
            # fsync before the rename: `flush()` only reaches the OS page
            # cache, and a power loss after the rename could otherwise publish
            # an empty file over a good one.
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
