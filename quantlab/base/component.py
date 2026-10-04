"""The component rule: how a configurable object is serialised and rebuilt.

A *component* is an object built from one config dataclass: a dataset, a
factor, a label. Some config fields hold other components (a factor's
dataset, a label's factor, a merged dataset's inputs). Those fields are
declared on the config dataclass with ``component()``, next to the fields
themselves, and that one declaration drives both directions:

- ``config_to_dict`` writes a config as a plain dict, each declared field
  replaced by its component's own ``get_config()``, whose ``"name"`` is the
  component class's import path;
- ``Component.from_config(d, run_dir=None)`` rebuilds the declared fields
  first, recursing only along them, then constructs the object from the
  config class its class declares in ``config_cls``. ``run_dir``, the run
  directory the dict was read from, reaches every level.

Nothing is recognised by key name or by shape: a free-form dict field (a
factor's ``kwargs``) is never entered, even when it holds a ``"name"`` key.
Only a class that declares a config class is ever rebuilt, so a saved dict
cannot instantiate an arbitrary importable callable.

Examples
--------
With ``momentum`` a ``Momentum`` factor over a ``SpotKlineDataset``:

>>> from quantlab.base.config import ForwardConfig
>>> from quantlab.label.forward import Forward
>>> label = Forward(ForwardConfig(factor=momentum, span=5))
>>> saved = label.get_config()
>>> saved["factor"]["name"], saved["factor"]["dataset"]["name"]
('quantlab.factor.predefined.momentum.Momentum', 'quantlab.dataset.spot.SpotKlineDataset')
>>> rebuild(saved) == label
True
"""

from __future__ import annotations

import copy
import dataclasses
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

#: The key under a field's ``metadata`` that marks it as holding components;
#: its value is whether the field holds many.
_METADATA_KEY = "quantlab.component"


def component(*, many: bool = False, **field_kwargs: Any) -> Any:
    """Declare a config dataclass field as holding a component, or several.

    Use it in place of ``dataclasses.field`` on a config dataclass. With
    ``many=False`` the field holds one component (or ``None``); with
    ``many=True`` it holds a list or tuple of components, or a dict mapping
    names to components. ``config_to_dict`` and ``Component.from_config``
    walk exactly the fields declared this way.

    Parameters
    ----------
    many : bool, default False
        Whether the field holds a collection of components rather than one.
    **field_kwargs
        Passed to ``dataclasses.field`` (``default``, ``default_factory``,
        ``kw_only`` ...).

    Returns
    -------
    dataclasses.Field
        The field, its metadata marking it as a component field.

    Examples
    --------
    >>> import dataclasses
    >>> @dataclasses.dataclass(frozen=True)
    ... class PairConfig:
    ...     first: object = component()
    ...     rest: tuple = component(many=True, default=())
    >>> component_fields(PairConfig)
    {'first': False, 'rest': True}
    """
    metadata = {**field_kwargs.pop("metadata", {}), _METADATA_KEY: many}
    return dataclasses.field(metadata=metadata, **field_kwargs)


def component_fields(config_cls: type) -> dict[str, bool]:
    """Return the declared component fields of a config dataclass.

    Parameters
    ----------
    config_cls : type
        A dataclass, or an instance of one.

    Returns
    -------
    dict[str, bool]
        Field name to whether it holds many components, in field order.

    Examples
    --------
    >>> from quantlab.base.config import ForwardConfig, MergedDatasetConfig
    >>> component_fields(ForwardConfig), component_fields(MergedDatasetConfig)
    ({'factor': False}, {'datasets': True})
    """
    return {
        spec.name: spec.metadata[_METADATA_KEY]
        for spec in dataclasses.fields(config_cls)
        if _METADATA_KEY in spec.metadata
    }


def config_to_dict(config: Any) -> dict[str, Any]:
    """Return a config dataclass as a plain dict, components as their own configs.

    Every declared component field (see ``component``) is written as the
    ``get_config()`` of the component it holds: one dict, a list of them, or
    a dict of them keyed as the field is. Every other field is deep-copied,
    a nested dataclass written as ``dataclasses.asdict``; a free-form dict is
    copied as data whatever keys it holds.

    Parameters
    ----------
    config : dataclass instance
        The config to write.

    Returns
    -------
    dict
        One entry per field, in field order.

    Examples
    --------
    With ``factor`` a factor over a ``StockDataset``:

    >>> saved = config_to_dict(factor.config)
    >>> saved["dataset"]["name"]
    'quantlab.dataset.stock.StockDataset'
    """
    declared = component_fields(config)
    out = {}
    for spec in dataclasses.fields(config):
        value = getattr(config, spec.name)
        out[spec.name] = (
            _write_components(value)
            if spec.name in declared
            else _plain(value)
        )
    return out


def _plain(value: Any) -> Any:
    """Return a non-component field value as data, copied."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return copy.deepcopy(value)


def _write_components(value: Any) -> Any:
    """Return one component, a sequence or a mapping of them, as configs."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {key: item.get_config() for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [item.get_config() for item in value]
    return value.get_config()


def config_cls_of(cls: type) -> type:
    """Return the config class ``cls`` declares in ``config_cls``.

    The config class is never guessed. A guess could rebuild one factor
    backend's config dict as another backend's config, and it would let any
    importable callable be instantiated with a config dict.

    Parameters
    ----------
    cls : type
        The class to rebuild.

    Returns
    -------
    type
        Its ``config_cls``.

    Raises
    ------
    TypeError
        If ``cls`` has no ``config_cls`` attribute, or it is not a type.

    Examples
    --------
    >>> from quantlab.label.forward import Forward
    >>> config_cls_of(Forward).__name__
    'ForwardConfig'
    """
    config_cls = getattr(cls, "config_cls", None)
    if not isinstance(config_cls, type):
        raise TypeError(
            f"{cls.__qualname__} declares no config_cls, so it cannot be "
            f"rebuilt from a config dict."
        )
    return config_cls


def rebuild(config: Mapping[str, Any], run_dir: str | os.PathLike | None = None) -> Any:
    """Rebuild the component a saved config names, and everything nested in it.

    The class named by ``config["name"]`` is imported, checked to declare a
    config class, and asked to rebuild itself with ``from_config``.
    ``config`` is not modified.

    Parameters
    ----------
    config : Mapping
        A dict ``get_config()`` returned, for example read back from JSON.
    run_dir : str or os.PathLike, optional
        The run directory the config was read from; a dataset recorded
        relative to it (``FrameDataset``) is resolved against it at any
        depth.

    Returns
    -------
    object
        The rebuilt component.

    Raises
    ------
    TypeError
        If the named class declares no config class.
    ValueError
        If the config, or a config nested in it, holds a key its config
        class does not have, or a dataset relative to a run directory is
        found without ``run_dir``.

    Examples
    --------
    With ``label`` a ``Forward`` label over a factor over a dataset:

    >>> rebuild(label.get_config()) == label
    True
    """
    from quantlab.utils.module import get_cls_from_path

    cls = get_cls_from_path(config["name"])
    config_cls_of(cls)
    return cls.from_config(config, run_dir=run_dir)


def _rebuild_components(value: Any, many: bool, run_dir: Path | None) -> Any:
    """Rebuild one declared field's saved value."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [rebuild(item, run_dir) for item in value]
    if many and isinstance(value, Mapping):
        return {key: rebuild(item, run_dir) for key, item in value.items()}
    if isinstance(value, Mapping):
        return rebuild(value, run_dir)
    # Already an object (a caller filled the field in), kept as given.
    return value


class Component:
    """Root of every configurable object: serialised and rebuilt by declaration.

    A subclass declares ``config_cls``, the config dataclass it is built
    from, and holds its config as ``self.config``; the component fields of
    that dataclass are declared with ``component()``. ``get_config`` and
    ``from_config`` then need no code of their own. A component with a
    shape of its own may override either.

    Examples
    --------
    With ``factor`` any factor built earlier:

    >>> saved = factor.get_config()
    >>> type(factor).from_config(saved) == factor
    True
    """

    #: The config dataclass the component is built from and rebuilt with.
    config_cls: type

    #: The component's config, an instance of ``config_cls``.
    config: Any

    @property
    def import_path(self) -> str:
        """The class as a dotted import path, the ``name`` of its config.

        Examples
        --------
        >>> label.import_path
        'quantlab.label.forward.Forward'
        """
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def get_config(self) -> dict[str, Any]:
        """Return the config as a plain dict with the class's import path as ``"name"``.

        Declared component fields hold their components' own configs (see
        ``config_to_dict``).

        Examples
        --------
        >>> cfg = label.get_config()
        >>> cfg["name"], cfg["factor"]["name"]
        ('quantlab.label.forward.Forward', 'quantlab.factor.predefined.momentum.Momentum')
        """
        return {**config_to_dict(self.config), "name": self.import_path}

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any], run_dir: str | os.PathLike | None = None
    ) -> Self:
        """Rebuild the component from the dict ``get_config()`` returned.

        The declared component fields are rebuilt first, each by the class
        its own ``"name"`` names, with ``run_dir`` passed down; the object is
        then constructed from ``config_cls``. The top-level ``"name"`` is
        kept only when the config class has such a field. ``config`` is not
        modified.

        Parameters
        ----------
        config : Mapping
            The saved config of a component of this class.
        run_dir : str or os.PathLike, optional
            The run directory the config was read from.

        Returns
        -------
        Component
            The rebuilt component.

        Raises
        ------
        TypeError
            If the class declares no config class.
        ValueError
            If ``config`` holds a key the config class does not have.

        Examples
        --------
        >>> from quantlab.label.forward import Forward
        >>> Forward.from_config(label.get_config()) == label
        True
        """
        config_cls = config_cls_of(cls)
        known = {spec.name for spec in dataclasses.fields(config_cls)}
        unknown = sorted(set(config) - known - {"name"})
        if unknown:
            raise ValueError(
                f"{cls.__qualname__}: the saved config holds unknown key(s) "
                f"{', '.join(repr(key) for key in unknown)}, which "
                f"{config_cls.__name__} does not have; refusing to rebuild it."
            )
        run_dir = None if run_dir is None else Path(run_dir)
        params = {
            key: copy.deepcopy(value)
            for key, value in config.items()
            if key in known
        }
        for name, many in component_fields(config_cls).items():
            if name in params:
                params[name] = _rebuild_components(params[name], many, run_dir)
        return cls(config_cls(**params))
