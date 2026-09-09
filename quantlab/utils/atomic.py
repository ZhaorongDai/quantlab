"""The single atomic JSON-sidecar writer this repository has (D-20).

Every JSON sidecar quantlab writes -- the two resume ledgers
(`base/pageledger.py:PageLedger._flush`, `base/chunking.py:ChunkLedger._flush`)
and the two acquisition sidecars (`base/acquisition.py:_write_watermark`,
`_write_failure_manifest`) -- goes through `write_json_atomically` below.

**Why atomically.** The payload is written to a temp file in the SAME directory
and then `os.replace`d over the destination, so a crash mid-write leaves either
the previous valid file or the new one -- never a half-written file that cannot
be parsed, which is precisely the file a resumed run would read. D-17's
cancellation interrupts at a batch boundary, which is exactly when a watermark
is being written, so SC-5's promise that a cancelled run leaves the store and
watermarks resumable is not true against a plain `open(path, "w")`.

**Why here, once.** This body existed twice already -- `ChunkLedger._flush`
wrote it and `PageLedger._flush` copied it verbatim. Rather than let a third
and fourth copy appear for the two acquisition sidecars, it was EXTRACTED
(03.4-03): the two ledgers now delegate to this function and produce
byte-identical output, and the two acquisition writers call the same code. One
implementation cannot drift from itself.

**No lock is taken here, deliberately.** D-20 leaves concurrency control to the
console's task queue; quantlab adds none. An atomic write is not a lock -- two
concurrent writers still race for last-writer-wins -- but it does mean the loser
of that race sees a complete file rather than a truncated one, which is what
makes the accepted residual risk (a thin-shell script and the console
overlapping on one source) materially safer rather than merely tolerated.

A LEAF module: stdlib only, zero project-internal imports, so anything may
import it without any possibility of an import cycle.
"""

import json
import os
import tempfile
from pathlib import Path

__all__ = ["write_json_atomically"]


def write_json_atomically(path: str | Path, payload: object, **json_kwargs) -> None:
    """Serialise `payload` as JSON to `path`, atomically.

    `**json_kwargs` is forwarded verbatim to `json.dump`, so every caller keeps
    its OWN formatting: the watermark sidecar stays compact, the failure
    manifest stays `indent=2, sort_keys=True`, and both ledgers stay `indent=2`.
    That is not a convenience -- normalising all four onto one format would make
    every sidecar already on disk differ from every newly written one, a
    repudiation-shaped change (T-03.4-03-04) hidden inside a durability fix.

    Missing parent directories are created, matching the
    `mkdir(parents=True, exist_ok=True)` every caller performed for itself
    before this helper existed.

    On ANY exception -- including `KeyboardInterrupt`, which is how a cancelled
    run arrives -- the temp file is unlinked and the original exception
    propagates, so a failed write leaves neither a stale destination nor a
    `*.tmp` holding the fragment it managed to serialise.
    """
    destination = str(path)
    directory = Path(destination).parent
    directory.mkdir(parents=True, exist_ok=True)
    # `dir=` is load-bearing, not tidiness: a temp file in the DESTINATION's own
    # directory makes `os.replace` a same-filesystem rename, which is the
    # operation that is atomic. Staged in the system temp dir instead, the
    # rename silently degrades into a cross-device copy -- interruptible, and
    # therefore capable of leaving exactly the half-written file this function
    # exists to make impossible.
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
            # fsync BEFORE the rename: `flush()` only reaches the OS page
            # cache, so without this the rename could be ordered ahead of the
            # data reaching the device and a power loss would publish an empty
            # file over a good one.
            os.fsync(handle.fileno())
        os.replace(handle.name, destination)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
