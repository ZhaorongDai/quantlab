"""`MlBackend`'s three methods must return `Self` so chaining works.

Quick task 260907-fl6, batch 2. `base/backend.py:ModelBackend` declares
`read`/`write`/`to_internal` as `-> Self`, mirroring `DataBackend`, whose two
implementations (`XrBackend`, `PlBackend`) all return `self` and are used in
chains throughout the codebase. `ml_model/backend.py:MlBackend`'s three
implementations returned `None` implicitly, so any chained call died with
`AttributeError: 'NoneType' object has no attribute ...` -- evidence the class
had never actually been run.

`MlBackend` is zero-call-site but deliberately KEPT: `BaseModel.predict()`'s
`np.ndarray` branch exists for non-torch `MLConfig` models (xgboost and the
like) and this is their joblib persistence. Scaffolding for a path that is
intended and not yet built is not an abandoned entrance -- but scaffolding
that breaks the moment someone uses it as documented is worth fixing before
the first caller arrives, which is what this file locks.
"""

from pathlib import Path

import joblib
import pytest

from quantlab.base.backend import ModelBackend
from quantlab.ml_model.backend import MlBackend


class _Model:
    """A tiny picklable stand-in for a non-torch model."""

    def __init__(self, coef: float):
        self.coef = coef

    def predict(self, x):
        return [v * self.coef for v in x]

    def __eq__(self, other):
        return isinstance(other, _Model) and other.coef == self.coef


def test_the_full_chain_round_trips_a_model(tmp_path: Path):
    """The single call the old code could not survive.

    Every link is a method that used to return `None`: `to_internal(...)`
    would have made `.write(...)` an `AttributeError`, and `read(...)` would
    have made `.get_model()` one.
    """
    store = tmp_path / "nested" / "model.joblib"
    model = _Model(2.5)

    MlBackend().to_internal(model).write(str(store))
    assert store.exists()

    loaded = MlBackend().read(str(store)).get_model()
    assert loaded == model
    assert loaded.predict([1.0, 2.0]) == [2.5, 5.0]


@pytest.mark.parametrize("method", ["read", "write", "to_internal"])
def test_each_method_returns_the_backend_itself(tmp_path: Path, method):
    """Asserted per method, so a regression names the one that broke rather
    than failing somewhere down a chain."""
    store = tmp_path / "model.joblib"
    backend = MlBackend()

    if method == "to_internal":
        assert backend.to_internal(_Model(1.0)) is backend
    elif method == "write":
        backend.to_internal(_Model(1.0))
        assert backend.write(str(store)) is backend
    else:
        joblib.dump(_Model(1.0), store)
        assert backend.read(str(store)) is backend


def test_write_creates_the_parent_directory(tmp_path: Path):
    """`XrBackend.write` does the same; a persistence backend that requires
    its caller to pre-create directories is a papercut, not a contract."""
    store = tmp_path / "a" / "b" / "c" / "model.joblib"
    MlBackend().to_internal(_Model(3.0)).write(str(store))
    assert store.exists()


def test_signatures_match_the_declared_ModelBackend_contract():
    """The `Self` half is behavioural above; this pins the `**kwargs` half.

    The ABC declares `read(path, **kwargs)` and `write(path, **kwargs)` --
    `to_internal(model)` deliberately takes none, since there is no library
    call underneath it to forward to. The implementations were dropping both
    of the two that exist, so a caller passing e.g. `compress=` to `write`
    got a `TypeError`.
    """
    import inspect

    for name in ("read", "write"):
        params = inspect.signature(getattr(MlBackend, name)).parameters
        assert any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ), f"MlBackend.{name} drops the **kwargs ModelBackend declares"

    assert (
        inspect.signature(MlBackend.to_internal).parameters.keys()
        == inspect.signature(ModelBackend.to_internal).parameters.keys()
    )
    assert issubclass(MlBackend, ModelBackend)


def test_write_forwards_kwargs_to_joblib(tmp_path: Path):
    """`**kwargs` must be a real passthrough, not decoration."""
    store = tmp_path / "compressed.joblib"
    MlBackend().to_internal(_Model(4.0)).write(str(store), compress=3)
    assert MlBackend().read(str(store)).get_model() == _Model(4.0)


def test_reading_a_missing_store_raises(tmp_path: Path):
    """Not a new guard -- just pinning that the failure is still the obvious
    one and was not masked by the `return self` change."""
    with pytest.raises(FileNotFoundError):
        MlBackend().read(str(tmp_path / "absent.joblib"))
