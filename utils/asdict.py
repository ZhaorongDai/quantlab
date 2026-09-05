from dataclasses import fields
import copy


# customized asdict
_FIELDS = "__dataclass_fields__"


def _is_dataclass_instance(obj):
    """Returns True if obj is an instance of a dataclass."""
    return hasattr(type(obj), _FIELDS)


def asdict_customized(obj, dict_factory=dict):
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
        # obj is a namedtuple.  Recurse into it, but the returned
        # object is another namedtuple of the same type.  This is
        # similar to how other list- or tuple-derived classes are
        # treated (see below), but we just need to create them
        # differently because a namedtuple's __init__ needs to be
        # called differently (see bpo-34363).

        # I'm not using namedtuple's _asdict()
        # method, because:
        # - it does not recurse in to the namedtuple fields and
        #   convert them to dicts (using dict_factory).
        # - I don't actually want to return a dict here.  The main
        #   use case here is json.dumps, and it handles converting
        #   namedtuples to lists.  Admittedly we're losing some
        #   information here when we produce a json list instead of a
        #   dict.  Note that if we returned dicts here instead of
        #   namedtuples, we could no longer call asdict() on a data
        #   structure where a namedtuple was used as a dict key.

        return type(obj)(*[asdict_customized(v, dict_factory) for v in obj])  # type: ignore
    elif isinstance(obj, (list, tuple)):
        # Assume we can create an object of this type by passing in a
        # generator (which is not true for namedtuples, handled
        # above).
        return type(obj)(asdict_customized(v, dict_factory) for v in obj)
    elif isinstance(obj, dict):
        if hasattr(type(obj), "default_factory"):
            # obj is a defaultdict, which has a different constructor from
            # dict as it requires the default_factory as its first arg.
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
