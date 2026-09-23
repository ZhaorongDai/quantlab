"""Locations of files that ship inside the ``quantlab`` distribution.

Paths here are derived from this module's own location, never from the
process's working directory, so they resolve correctly wherever ``quantlab``
is imported from. The module imports only the standard library so both the
dataset layer and the command-line tools can read these constants without
creating an import cycle.
"""

from pathlib import Path

__all__ = ["PACKAGE_ROOT", "INSTRUMENTS_CONFIG_PATH"]

#: Root of the installed ``quantlab`` package (the parent of ``utils``).
PACKAGE_ROOT: Path = Path(__file__).resolve().parent.parent

#: Instrument and venue metadata shipped as package data next to the
#: configuration package. It is declared in ``[tool.setuptools.package-data]``
#: so it is present in a built wheel as well as in the source tree. Both the
#: instrument loader and the Binance refresh CLI use this one constant.
INSTRUMENTS_CONFIG_PATH: Path = PACKAGE_ROOT / "config" / "instruments.yaml"
