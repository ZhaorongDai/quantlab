"""The config of the label layer.

A label (a forward-shifted factor, ADR 0005) is constructed from a ``ForwardConfig``;
its ``factor`` is declared with ``component()`` and written as its own config.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from quantlab.core.component import component
from quantlab.core.config import FrozenConfig

if TYPE_CHECKING:
    from quantlab.factor.base import Factor


@dataclass(kw_only=True, frozen=True)
class ForwardConfig(FrozenConfig):
    """Config of a ``Forward`` label: the factor it shifts and by how much.

    The label at bar t is ``factor`` at bar t + ``delay`` + ``span``. A
    ``Forward`` owns no store, so the config has no path or warm-up of its
    own; the wrapped factor keeps its config. ``factor`` is a component
    field, so ``to_dict()`` nests the factor's config dict under it and
    ``Forward.from_config`` rebuilds it.

    Examples
    --------
    With ``factor`` a factor built earlier:

    >>> cfg = ForwardConfig(factor=factor, span=5)
    >>> cfg.span, cfg.delay
    (5, 1)
    """

    #: The factor shifted forward to make the label.
    factor: "Factor" = component()
    #: Bars the label accumulates over, such as the n bars of an n-bar
    #: forward return.
    span: int
    #: Bars between the bar a signal forms on and the first bar the label
    #: counts; 1 because a signal at t fills at t+1's open.
    delay: int = 1
    #: Dotted import path of the label class; filled by the config setter.
    name: str | None = None
