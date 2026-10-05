"""A test stand-in in the label role.

Plain helpers, not pytest fixtures, imported as ``from tests.label_stubs
import StubLabel``. A model accepts as labels only objects with ``lookahead_bars()``
(see ``BaseModel._check_roles``), so a lightweight panel stand-in that serves
as a factor in one place and as a label in another is wrapped here when it is
a label.
"""


import dataclasses

from quantlab.core.component import Component, component


@dataclasses.dataclass(frozen=True)
class StubLabelConfig:
    panel: object = component()
    lookahead: int = 0


class StubLabel(Component):
    """Wrap a stand-in panel as a label that reads ``lookahead`` bars after t.

    Every attribute but the two label methods is the wrapped panel's.
    ``lookahead_bars()`` defaults to 0, so no split purges a bar for it and
    it adds nothing to a backtest's in-sample window; ``span_bars()`` is 0.
    """

    config_cls = StubLabelConfig

    def __init__(self, panel, lookahead: int = 0):
        self._panel = panel
        self._lookahead = lookahead

    @property
    def config(self) -> StubLabelConfig:
        return StubLabelConfig(panel=self._panel, lookahead=self._lookahead)

    @classmethod
    def from_config(cls, config, run_dir=None):
        fields = cls._rebuilt_fields(config, run_dir)
        return cls(fields["panel"], fields["lookahead"])

    def __getattr__(self, name):
        # ``copy.deepcopy`` (``dataclasses.asdict`` of a config) builds the
        # copy without calling ``__init__``, so ``_panel`` is not set yet.
        if name in ("_panel", "_lookahead"):
            raise AttributeError(name)
        return getattr(self._panel, name)

    def lookahead_bars(self) -> int:
        return self._lookahead

    def span_bars(self) -> int:
        return 0
