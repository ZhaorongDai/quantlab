"""Package-derived locations for files that ship *inside* the distribution.

This module deliberately imports nothing but the standard library. The
configuration package imports the dataset layer, which imports the nautilus
helper, so a constant read from the configuration package at module scope
would close an import cycle. A stdlib-only leaf under ``utils`` is importable
from both ends of that chain and from a bare CLI that has no business loading
the dataset layer.

DECISION (quick task 260907-sm2): the instrument metadata file is *package
data*. Its location is derived from this module's own file, never from the
process's current working directory. Now that the project is installable, a
consumer imports ``quantlab`` from wherever it happens to be standing; a
working-directory-relative default resolves correctly exactly once -- when the
process starts at the repository root -- and silently points at nothing
everywhere else. For this repository's own editable install the
package-derived path lands on the tracked file, which is what the Binance CLI
is supposed to read and rewrite.

Both the instrument loader and the Binance CLI read this single constant, so
the two cannot drift apart again.
"""

from pathlib import Path

__all__ = ["PACKAGE_ROOT", "INSTRUMENTS_CONFIG_PATH"]

#: Root of the installed ``quantlab`` package (the parent of ``utils``).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent

#: Instrument/venue metadata shipped as package data alongside the
#: configuration package. Declared in ``[tool.setuptools.package-data]`` so it
#: is present in a built wheel as well as in the source tree.
INSTRUMENTS_CONFIG_PATH: Path = PACKAGE_ROOT / "config" / "instruments.yaml"
