"""A test stand-in in the label role.

Plain helpers, not pytest fixtures, imported as ``from tests.label_stubs
import StubLabel``. A model accepts as labels only objects with ``lookahead_bars()``
(see ``BaseModel._check_roles``), so a lightweight panel stand-in that serves
as a factor in one place and as a label in another is wrapped here when it is
a label.
"""


class StubLabel:
    """Wrap a stand-in panel as a label that reads no bars after t.

    Every attribute but the two label methods is the wrapped panel's.
    ``lookahead_bars()`` is 0, so no split purges a bar for it, and
    ``span_bars()`` is 0, so it adds nothing to a backtest's in-sample window.
    """

    def __init__(self, panel):
        self._panel = panel

    def __getattr__(self, name):
        # ``copy.deepcopy`` (``dataclasses.asdict`` of a config) builds the
        # copy without calling ``__init__``, so ``_panel`` is not set yet.
        if name == "_panel":
            raise AttributeError(name)
        return getattr(self._panel, name)

    def lookahead_bars(self) -> int:
        return 0

    def span_bars(self) -> int:
        return 0
