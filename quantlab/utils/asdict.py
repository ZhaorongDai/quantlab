"""A recursive ``asdict`` that tolerates values ``copy.deepcopy`` cannot handle.

``dataclasses.asdict`` raises as soon as a field holds something that cannot be
deep-copied, such as an open file handle or a compiled model. ``asdict_customized``
walks dataclasses, namedtuples, lists, tuples and dicts the same way the standard
version does, but any leaf that fails to deep-copy becomes ``None`` instead of
raising, so a config object can always be flattened for logging or JSON output.
"""

from dataclasses import fields
import copy


_FIELDS = "__dataclass_fields__"


def _is_dataclass_instance(obj):
    """Return True if ``obj`` is an instance (not the class) of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def asdict_customized(obj, dict_factory=dict):
    """Convert a dataclass instance to a dict, replacing uncopyable leaves with None.

    Dataclass fields, namedtuples, lists, tuples and dicts (including
    ``defaultdict``) are recursed into and rebuilt with the same container type.
    Every other value is deep-copied; if the copy raises ``TypeError`` the value
    is replaced by ``None`` rather than aborting the whole conversion.

    Parameters
    ----------
    obj
        The dataclass instance, container or leaf value to convert.
    dict_factory
        Callable that builds the mapping for each dataclass
        level, as in ``dataclasses.asdict``.

    Returns
    -------
    dict
        A plain-data mirror of ``obj``.

    Examples
    --------
    >>> import threading
    >>> from dataclasses import dataclass, field
    >>> @dataclass
    ... class Job:
    ...     name: str
    ...     lock: object = field(default_factory=threading.Lock)
    >>> asdict_customized(Job("nightly"))
    {'name': 'nightly', 'lock': None}
    """
    if _is_dataclass_instance(obj):
        # fast path for the common case
        if dict_factory is dict:
            return {
                f.name: asdict_customized(getattr(obj, f.name), dict)
                for f in fields(obj)
            }
        else:
            result = []
            for f in fields(obj):
                value = asdict_customized(getattr(obj, f.name), dict_factory)
                result.append((f.name, value))
            return dict_factory(result)
    elif isinstance(obj, tuple) and hasattr(obj, "_fields"):
        # A namedtuple is rebuilt as the same namedtuple type (positional
        # construction) rather than through its own `_asdict`, which neither
        # recurses into nested fields nor returns the tuple type json expects.
        return type(obj)(*[asdict_customized(v, dict_factory) for v in obj])  # type: ignore
    elif isinstance(obj, (list, tuple)):
        # Assume the container type accepts a generator (namedtuples, which do
        # not, were handled above).
        return type(obj)(asdict_customized(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        if hasattr(type(obj), "default_factory"):
            # A defaultdict takes its default_factory as the first constructor
            # argument, so it cannot be rebuilt from pairs like a plain dict.
            result = type(obj)(getattr(obj, "default_factory"))
            for k, v in obj.items():
                result[asdict_customized(k, dict_factory)] = asdict_customized(
                    v, dict_factory
                )
            return result
        return type(obj)(
            (asdict_customized(k, dict_factory), asdict_customized(v, dict_factory))
            for k, v in obj.items()
        )
    else:
        try:
            return copy.deepcopy(obj)
        except TypeError:
            return None
