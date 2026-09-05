"""Factor-hierarchy boundary-contract tests (FACTOR-04 + D-03).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0); the `Factor` ABC extraction
assertions were added by **03-02**. The KunQuant/Polars interchangeability
integration test lands in **03-05**.

`test_base_model_does_not_dispatch_on_concrete_factor_types` is the cheap,
permanent regression lock on D-03's "seamless interchangeability" claim. It
passed before 03-02 split `FactorKunQuant` into `Factor` + backend subclasses
and MUST keep passing after 03-04 adds `FactorPolars`. The moment the model
layer has to know which factor backend it was handed, the interchangeability
claim is dead -- and that test is what catches it.

The rest of this file locks the shape of the 03-02 refactor itself: D-03's
hierarchy, D-07's streaming isolation, the two hazards a naive hoist gets
wrong (a `mode` read on the shared base, a reordered `__init__`), and
FACTOR-04's xarray-only public boundary. Several are source-introspection
tests rather than behavioural ones, because nothing at runtime fails today
if those invariants are broken -- the breakage only surfaces later, in the
Polars backend that does not exist yet.

Written in the established grep-style purity idiom of
`tests/test_extensibility_contract.py:test_core_layer_purity_no_market_
specific_logic`, including its comment-line exclusion so an architecture note
mentioning these names in prose is not a false positive.

Import-safety rule (tests/conftest.py module docstring): nothing here may
import `base.factor_polars` or `factor.momentum` at module level -- neither
exists yet.
"""

import dataclasses
import inspect
from pathlib import Path
from typing import Callable

import xarray as xr

from base.config import DatasetConfig, DLConfig, FactorConfig, MLConfig
from base.factor import Factor, FactorKunQuant
from dataset.spot import SpotKlineDataset
from label.spot import SpotBinaryReturn, SpotReturn

CONSUMER_FILE = "base/model.py"

# KunQuant streaming / compiled-graph members. D-07 makes the Polars backend
# batch-only, so none of these may live on the shared base -- a batch-only
# backend must not inherit an obligation to implement streaming.
KUNQUANT_ONLY_MEMBERS = (
    "init_stream",
    "cal_stream",
    "_make",
    "_make_stream",
    "_to_xarray_dataset",
    "_get_factor_func",
)

# The KunQuant-only config field that does NOT exist on `PolarsFactorConfig`.
# A read of it from any shared-base method is an AttributeError waiting to
# happen in a sibling backend.
KUNQUANT_ONLY_CONFIG_ATTRIBUTE = ".mode"

# The attribute name of the storage backend, assigned AFTER `self.config` in
# `Factor.__init__` and therefore absent while the config setter runs.
STORAGE_BACKEND_ATTRIBUTE = "data_backend"

# The public methods that make up the factor layer's module boundary.
# Private helpers are excluded on purpose (see the boundary test's docstring).
PUBLIC_FACTOR_API = (
    "cal",
    "read",
    "save",
    "get_features",
    "get_labels",
    "get_factor_names",
    "get_config",
)

# The exhaustive set of members `base/model.py:115-201` invokes on a factor or
# a label object (03-PATTERNS.md section 7). Every one must resolve on the
# shared `Factor` base, or a non-KunQuant backend could not be dropped into
# `DLConfig.factors` unchanged.
BASE_MODEL_CALL_SURFACE = (
    "config",
    "_reset_dataset_config",
    "cal",
    "read",
    "get_features",
    "get_labels",
    "_get_factor_names",
    "get_config",
)

# A live reference to either concrete factor backend in the model layer would
# mean the model knows which backend computed its features -- exactly the
# coupling D-03 forbids.
FORBIDDEN_SUBSTRINGS = (
    "FactorKunQuant",
    "FactorPolars",
)


def test_base_model_does_not_dispatch_on_concrete_factor_types() -> None:
    """`base/model.py` must never name a concrete factor backend, nor branch
    on a factor's runtime type via `isinstance`.

    `DLConfig.factors` / `MLConfig.factors` are typed as a plain list of
    factors; the model layer only ever calls the shared contract
    (`get_features()` / `get_factor_names()` / `read()` / `cal()`). Any
    `isinstance(..., Factor...)` dispatch would make adding a third backend a
    model-layer change, breaking FACTOR-04 and D-03.

    Lines whose stripped content starts with `#` are excluded before checking,
    so a comment mentioning these names in passing is not a violation -- only
    a live, non-comment reference counts.
    """
    violations: list[str] = []

    source_lines = Path(CONSUMER_FILE).read_text(encoding="utf-8").splitlines()

    for line_number, line in enumerate(source_lines, start=1):
        if line.strip().startswith("#"):
            continue

        for forbidden in FORBIDDEN_SUBSTRINGS:
            if forbidden in line:
                violations.append(
                    f"{CONSUMER_FILE}:{line_number} references "
                    f"'{forbidden}': {line.strip()}"
                )

        if "isinstance" in line and "Factor" in line:
            violations.append(
                f"{CONSUMER_FILE}:{line_number} dispatches on a factor type: "
                f"{line.strip()}"
            )

    assert not violations, (
        "base/model.py must stay agnostic of which factor backend computed "
        "its features (FACTOR-04 / D-03):\n" + "\n".join(violations)
    )


def _label_factor_config(
    dataset_config: DatasetConfig, tmp_path: Path
) -> FactorConfig:
    """A `FactorConfig` for a label class with `factor_names` left as `None`.

    Leaving it `None` is the whole point: it forces the `Factor.config` setter
    to go through `_maybe_resolve_factor_names()`, which is the code path
    03-02 rewrote.
    """
    return FactorConfig(
        window=10,
        dataset=SpotKlineDataset(dataset_config),
        mode="batch",
        data_columns=["close"],
        factor_names=None,
        file_path=str(tmp_path / "labels" / "out.zarr"),
        njobs=4,
        kwargs={"n_forward_periods": 1},
    )


def test_label_classes_construct_through_the_refactored_hierarchy(
    spot_kline_zarr: Callable[..., DatasetConfig], tmp_path: Path
) -> None:
    """`SpotReturn` / `SpotBinaryReturn` still construct with their factor
    names resolved eagerly (03-02 Hazard 3).

    This is the ONLY coverage `label/spot.py` has anywhere in the repository.
    It exists because 03-02 rewrote exactly the code path label construction
    runs: both classes are direct `FactorKunQuant` subclasses, so building one
    fires the hoisted `Factor.config` setter and the brand-new
    `_maybe_resolve_factor_names()` hook. If that hook's default stopped
    resolving names eagerly, every label class would silently start carrying
    `factor_names=None` into `cal()`.

    Construction only -- `.cal()` is deliberately not called: label
    computation is out of 03-02's scope and would add a KunQuant compile to
    the test run.
    """
    dataset_config = spot_kline_zarr(periods=30, seed=0)

    label = SpotReturn(_label_factor_config(dataset_config, tmp_path))

    resolved = label.config.factor_names
    assert isinstance(resolved, (tuple, list))
    assert len(resolved) > 0
    assert resolved[0] == "ret_1"

    for label_cls in (SpotReturn, SpotBinaryReturn):
        assert issubclass(label_cls, FactorKunQuant)
        assert issubclass(label_cls, Factor)
        for member in BASE_MODEL_CALL_SURFACE:
            assert member in dir(label_cls), (
                f"{label_cls.__name__} is missing '{member}' from the "
                "base/model.py call surface"
            )


def _own_function_sources(cls: type) -> str:
    """Concatenated source of every callable defined DIRECTLY on `cls`.

    Plain functions plus property getters/setters -- inherited members are
    excluded, which is the whole point: `FactorKunQuant`'s overrides read the
    KunQuant-only config fields legitimately, so a grep over the whole module
    would be meaningless.
    """
    chunks: list[str] = []

    for member in vars(cls).values():
        if inspect.isfunction(member):
            chunks.append(inspect.getsource(member))
        elif isinstance(member, property):
            for accessor in (member.fget, member.fset, member.fdel):
                if accessor is not None:
                    chunks.append(inspect.getsource(accessor))

    return "".join(chunks)


def test_factor_kunquant_subclasses_shared_factor_base() -> None:
    """D-03: a shared abstract `Factor` base exists and `FactorKunQuant` is one
    implementation of it, not the root of the hierarchy.

    `cal` and `_get_factor_names` are abstract ON THE SHARED BASE, which is the
    mechanism that makes `base/model.py`'s `factor.cal()` polymorphic across
    backends rather than a KunQuant call in disguise.
    """
    assert inspect.isabstract(Factor)
    assert issubclass(FactorKunQuant, Factor)
    assert "cal" in Factor.__abstractmethods__
    assert "_get_factor_names" in Factor.__abstractmethods__


def test_streaming_members_stay_on_the_kunquant_subclass() -> None:
    """D-07: every KunQuant streaming / compiled-graph member stays on
    `FactorKunQuant` and is absent from `Factor`.

    The Polars backend (03-04) is batch-only. If any of these leaked onto the
    shared base -- especially as an `@abstractmethod` -- every future backend
    would be forced to implement streaming it has no use for.
    """
    own = set(FactorKunQuant.__dict__)
    shared = set(Factor.__dict__)

    missing = [name for name in KUNQUANT_ONLY_MEMBERS if name not in own]
    leaked = [name for name in KUNQUANT_ONLY_MEMBERS if name in shared]

    assert not missing, f"no longer defined on FactorKunQuant: {missing}"
    assert not leaked, f"leaked onto the shared Factor base (D-07): {leaked}"


def test_shared_factor_base_never_reads_config_mode() -> None:
    """Hazard 2: no method defined on `Factor` may read the KunQuant-only
    `mode` config field.

    `mode` lives on `FactorConfig`, not on `BaseFactorConfig`, so
    `PolarsFactorConfig` has no such attribute. A hoisted `mode` read would
    therefore raise `AttributeError` at RUNTIME, only for the Polars backend,
    and only once someone actually calls the method -- never at type-check
    time. This test moves that failure to test time.

    Only functions defined directly on `Factor` are inspected:
    `FactorKunQuant._auto_filter` / `.num_symbols` / `.symbols` read `mode`
    legitimately, so a file-wide grep could not express this invariant.
    """
    source = _own_function_sources(Factor)

    assert KUNQUANT_ONLY_CONFIG_ATTRIBUTE not in source, (
        "a method on the shared Factor base reads the KunQuant-only 'mode' "
        "config field; PolarsFactorConfig does not have it (Hazard 2)"
    )


def test_factor_base_carries_the_full_base_model_call_surface() -> None:
    """D-03 interchangeability: every member `base/model.py` invokes on a
    factor or a label object resolves on the shared `Factor` base.

    `BASE_MODEL_CALL_SURFACE` is the exhaustive set read out of
    `base/model.py:115-201` (03-PATTERNS.md section 7) -- note it includes the
    PRIVATE `_get_factor_names`, which the model layer calls directly rather
    than through the public `get_factor_names()` wrapper. Because the whole
    set resolves on `Factor`, any backend satisfying `Factor` can be dropped
    into `DLConfig.factors` with zero edits to `base/model.py`.
    """
    missing = [
        name for name in BASE_MODEL_CALL_SURFACE if not hasattr(Factor, name)
    ]

    assert not missing, (
        "the shared Factor base is missing members base/model.py calls on "
        f"every factor: {missing}"
    )


def test_model_configs_are_typed_against_the_shared_factor_base() -> None:
    """D-03 blast-radius fix: `DLConfig`/`MLConfig` factor and label fields are
    typed against the shared `Factor`, never against a concrete backend.

    The type hint is not itself the proof of interchangeability (nothing here
    is type-checked at runtime) -- `test_base_model_does_not_dispatch_on_
    concrete_factor_types` is. But a `FactorKunQuant` annotation would tell
    every future reader the model layer expects one specific backend, which is
    exactly the coupling D-03 removes.
    """
    for config_cls in (DLConfig, MLConfig):
        for field in dataclasses.fields(config_cls):
            if field.name not in ("factors", "labels"):
                continue

            annotation = str(field.type)
            assert "Factor" in annotation, (
                f"{config_cls.__name__}.{field.name} is not typed against a "
                f"factor type at all: {annotation}"
            )
            assert "FactorKunQuant" not in annotation, (
                f"{config_cls.__name__}.{field.name} names a concrete factor "
                f"backend (D-03): {annotation}"
            )


def test_factor_init_assigns_config_before_the_storage_backend() -> None:
    """Hazard 1: `Factor.__init__` must assign `self.config` BEFORE the storage
    backend, and no setter-reachable method may read the storage backend.

    Assigning `self.config` fires the `Factor.config` property setter, which
    calls `_maybe_resolve_factor_names()` and `_reset_dataset_config()` --
    all while `self.data_backend` does not yet exist. Reversing the two
    `__init__` lines, or making any setter-reachable method read the storage
    backend, raises `AttributeError` on EVERY factor construction. Today no
    such path exists; this test is what keeps it that way, because nothing
    else in the suite would fail if the invariant were quietly broken.

    `Factor.__init__` and `FactorKunQuant.__init__` are excluded from the
    second assertion: they are the assignment site, not a setter-reachable
    path.
    """
    init_source = inspect.getsource(Factor.__init__)
    config_index = init_source.index("self.config = config")
    backend_index = init_source.index(f"self.{STORAGE_BACKEND_ATTRIBUTE} =")

    assert config_index < backend_index, (
        "Factor.__init__ assigns the storage backend before self.config; the "
        "config setter would then run against a half-built instance"
    )

    # Everything the config setter can reach. `import_path` is included
    # alongside the three members 03-02 names because the setter calls it too.
    reachable = [
        vars(Factor)["config"].fset,
        Factor._maybe_resolve_factor_names,
        Factor._reset_dataset_config,
        vars(Factor)["import_path"].fget,
    ]
    setter_reachable_names = (
        "config",
        "_maybe_resolve_factor_names",
        "_reset_dataset_config",
    )
    for name in setter_reachable_names:
        override = vars(FactorKunQuant).get(name)
        if isinstance(override, property):
            reachable.append(override.fset or override.fget)
        elif override is not None:
            reachable.append(override)

    reachable_source = "".join(inspect.getsource(fn) for fn in reachable)

    assert STORAGE_BACKEND_ATTRIBUTE not in reachable_source, (
        "a method reachable from the Factor.config setter reads "
        f"self.{STORAGE_BACKEND_ATTRIBUTE}, which does not exist yet during "
        "__init__ (Hazard 1)"
    )


def test_public_factor_api_exchanges_only_xarray_datasets() -> None:
    """FACTOR-04: no public method of `Factor` or `FactorKunQuant` accepts or
    returns a pandas/polars DataFrame -- `xr.Dataset` is the only exchange
    type at the module boundary.

    Private helpers are deliberately excluded: `Factor._get_lazyframe()`
    legitimately returns a `pl.LazyFrame`, and `_get_features(data)` takes the
    already-unwrapped dataset. FACTOR-04 constrains what crosses the factor
    layer's boundary, not what it uses internally.
    """
    for cls in (Factor, FactorKunQuant):
        for name in PUBLIC_FACTOR_API:
            signature = str(inspect.signature(getattr(cls, name)))
            assert "DataFrame" not in signature, (
                f"{cls.__name__}.{name} exchanges a DataFrame at the public "
                f"boundary (FACTOR-04): {signature}"
            )

        for name in ("get_features", "get_labels"):
            annotation = inspect.signature(
                getattr(cls, name)
            ).return_annotation
            assert annotation is xr.Dataset, (
                f"{cls.__name__}.{name} must return an xr.Dataset, not "
                f"{annotation!r}"
            )
