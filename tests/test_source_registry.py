"""Home for the data-source registry proofs: ROADMAP success criteria SC-1 and
SC-2, requirements D-01..D-07.

SC-1 — every acquirable source is enumerable from ONE registry, with no vendor
class named at the call site; each descriptor carries its capabilities, its
credential env-var NAMES and its acquisition class.
SC-2 — enumeration and credential status never return a credential VALUE and
never require one to be present.

Scaffolded by plan 03.4-01 (Wave 0) and filled in by plan 03.4-02, which built
`quantlab/acquisition/registry.py` and both vendor descriptors.

TWO RULES THIS FILE IS SUBJECT TO, both from incidents this repository has
already had (recorded in `.planning/STATE.md`):

1. EVERY test here must be a real assertion. A pytest file containing zero
   tests exits **5** ("no tests ran"), and a per-file command reads that exit
   as success. A placeholder, a body that is only a no-op statement, or a
   skip/xfail marker is therefore indistinguishable from a passing file.

   On pytest 9.1.1 that same exit 5 is ALSO what a `-k` selector matching
   nothing produces ("no tests collected (N deselected)"). The two are told
   apart by which is expected: for a scaffold file exit 5 is the failure this
   file exists to prevent; for one of the not-yet-implemented selectors listed
   in rule 2 it is the required result. Every test below is a genuine
   assertion against code that exists: the first three are the Wave-0
   infrastructure self-tests (fixture layout, credential-name anchoring), the
   rest pin the registry contract itself.

2. A `-k` selector name must not be attached to a test that does not honestly
   cover that selector's behaviour. In 03.2 a mechanism was deleted and
   `-k fingerprint` stayed green, because the only test covering it was named
   outside its own selector. Every selector `03.4-VALIDATION.md` assigns to
   this file -- `one_descriptor_per_vendor`, `capabilities`,
   `capabilities_match_the_vendor_class`, `direct_class_reference`,
   `env_names_are_exactly_what_gates_construction`,
   `never_returns_a_credential_value`, `registration_tuple_shape`,
   `decorator_registers`, `enumeration_order`,
   `enumeration_is_complete_from_a_cold_import` -- now matches at least one
   test, and each of those tests genuinely covers the behaviour its selector
   names. Renaming one of them without moving its assertions is how the 03.2
   incident repeats.
"""

import json
import os
from pathlib import Path

import pytest

import quantlab.acquisition.alpaca as alpaca
import quantlab.acquisition.tiingo as tiingo


def test_the_acquisition_config_fixture_serves_both_vendors_with_a_terminated_raw_root(
    acquisition_config,
) -> None:
    """The two path invariants every registry/inspector/run test inherits.

    `acquisition_config` is the fixture plan 02's registry tests and plan 04's
    inspector tests both build on, so its layout is load-bearing for work that
    has not been written yet. Two properties in particular:

    - `raw_data_dir_path` TERMINATES at the vendor segment, which is the
      equality `StockDataset._scan_raw` asserts to make a cross-vendor silent
      merge unreachable by accident.
    - the watermark root is a SIBLING of the raw root, never inside it. A
      polars directory scan of the raw root walks every file beneath it, so a
      `.json` sidecar in that tree would break the scan outright -- and
      `browse_raw` (D-10) is exactly such a scan.

    Both are asserted here rather than assumed, because a fixture whose layout
    quietly changed would take the later pruning and isolation tests with it
    while they still looked green.
    """
    tiingo_cfg = acquisition_config(vendor="tiingo")
    alpaca_cfg = acquisition_config(vendor="alpaca")

    for cfg in (tiingo_cfg, alpaca_cfg):
        raw_root = Path(cfg.raw_data_dir_path)
        watermark_root = Path(cfg.watermark_path).parent

        assert raw_root.name == cfg.vendor
        # Sibling, not descendant. Stated both ways: the positive form pins
        # the exact layout, the negative form pins the property that matters
        # (a layout change that kept the property would still be fine).
        assert watermark_root == raw_root.parent / "_watermarks"
        assert raw_root not in watermark_root.parents
        assert raw_root != watermark_root

    assert Path(tiingo_cfg.raw_data_dir_path) != Path(alpaca_cfg.raw_data_dir_path)


def test_vendor_credential_env_names_are_module_level_constants_not_client_attributes() -> (
    None
):
    """Each vendor's `CREDENTIAL_ENV_VARS` is exactly its module-level names.

    This is the pin plan 02's `SourceDescriptor.required_env` will be checked
    against in BOTH directions (D-04): the descriptor must declare exactly the
    names that gate construction in the acquisition class. That comparison is
    only meaningful if the acquisition class's own declaration is itself
    anchored to the module-level constants, so that half is asserted first,
    here, before anything depends on it.

    The constants are module-level rather than client-class attributes on
    purpose (03.2-02 deviation #2): a security control must not be reachable
    through an indirection whose whole purpose is to be replaced by a test
    double. `_AlpacaMarketDataClient` re-exposes both names, but they are
    DEFINED at module scope, and `CREDENTIAL_ENV_VARS` is built from there.

    Only NAMES are compared. No value is read, and none of these names is
    required to be set for this test to run.
    """
    assert tiingo.KEY_ENV == "TIINGO_API_KEY"
    assert alpaca.KEY_ENV == "APCA_API_KEY_ID"
    assert alpaca.SECRET_ENV == "APCA_API_SECRET_KEY"

    assert tiingo.TiingoAcquisition.CREDENTIAL_ENV_VARS == (tiingo.KEY_ENV,)
    assert alpaca.AlpacaAcquisition.CREDENTIAL_ENV_VARS == (
        alpaca.KEY_ENV,
        alpaca.SECRET_ENV,
    )


def test_the_no_credentials_fixture_empties_every_declared_name(
    no_credentials,
) -> None:
    """The `no_credentials` fixture actually clears what it claims to clear.

    Every SC-3 proof in plan 04 -- "the read surface answers with NO
    credentials present" -- rests entirely on this fixture. If it silently
    stopped deleting one of the three names on a machine where that name
    happens to be set, those tests would pass while proving nothing, which is
    the same green-on-nothing failure as the exit-5 trap.

    The fixture's return is asserted to be the NAMES, and only the names: it
    must never carry a credential value out of the environment it just
    emptied.
    """
    assert no_credentials == (
        "TIINGO_API_KEY",
        "APCA_API_KEY_ID",
        "APCA_API_SECRET_KEY",
    )

    for name in no_credentials:
        assert name not in os.environ


# ---------------------------------------------------------------------------
# 03.4-02 Task 1 (TRACER) -- the ONE path, end to end
# ---------------------------------------------------------------------------


def test_tracer_end_to_end_registry_to_raw_shard(
    monkeypatch, acquisition_config, mock_tiingo_client
) -> None:
    """One path through every layer this phase touches, asserted end to end.

    Registry -> descriptor -> vendor module -> `Acquisition` base -> raw shard
    on disk, in one test. This is deliberately NOT a per-layer unit test: the
    per-layer contracts are pinned separately, and what THIS proves is that the
    seams between them actually meet -- an architectural dead end surfaces here,
    after one commit, rather than after the inspector, the reporter, the cancel
    token and three shells have all been built on top of it.

    Four claims, in the order a caller meets them:

    1. the descriptor is reachable by VENDOR TOKEN, with no vendor class named
       anywhere in this test (SC-1);
    2. `is_configured` answers `False` with the key absent and `True` with it
       set -- and answers at all without a credential present (SC-2);
    3. `run(descriptor, config)` reaches the vendor through the descriptor's
       own class reference and returns an `AcquisitionResult` naming the
       requested symbols (D-12 / D-14 / D-18);
    4. `_failures.json` still lands on disk beside the watermarks, because the
       in-process result and the crash-durable record are two outputs on
       purpose, not one replacing the other (D-18).

    Issues zero real requests: `mock_tiingo_client` replaces the transport
    wholesale.
    """
    from quantlab.acquisition.registry import (
        DataSourceRegistry,
        is_configured,
        run,
    )
    from quantlab.base.acquisition import AcquisitionResult

    descriptor = DataSourceRegistry.get("tiingo")
    config = acquisition_config(vendor="tiingo")

    # (2) -- both directions, and neither reads a VALUE.
    monkeypatch.delenv("TIINGO_API_KEY", raising=False)
    assert is_configured(descriptor) is False
    monkeypatch.setenv("TIINGO_API_KEY", "not-a-real-credential")
    assert is_configured(descriptor) is True

    # (3)
    result = run(descriptor, config)

    assert isinstance(result, AcquisitionResult)
    assert result.vendor == "tiingo"
    assert set(result.requested) == set(config.symbols)
    assert set(result.succeeded) == set(config.symbols)
    assert result.failures == {}
    assert result.cancelled is False
    assert result.quota_aborted is False
    assert result.coverage["requested"] == len(config.symbols)

    # (4) -- the manifest is a SIBLING artefact, not a replacement.
    manifest = Path(config.watermark_path) / "_failures.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text()) == result.failures

    # ... and the raw tier actually received shards, under the vendor-
    # terminated root. Asserted on the ROOT the config names rather than on a
    # path rebuilt here, so a layout change fails at the config invariant test
    # above instead of silently passing a reconstructed path.
    raw_root = Path(config.raw_data_dir_path)
    assert raw_root.name == "tiingo"
    assert list(raw_root.rglob("*.pqt"))


# ---------------------------------------------------------------------------
# 03.4-02 Task 2 -- the D-01 / D-02 / D-03 / D-05 / D-06 registry contract
# ---------------------------------------------------------------------------


def _fake_descriptor(vendor: str, **overrides):
    """A descriptor over a NOVEL vendor token, built for one test.

    Novel by construction, in the spirit of
    `tests/test_extensibility_contract.py`'s `FakeDataset`: registering a token
    the production code has never heard of proves registration works by
    CONSTRUCTION rather than by re-asserting what the two shipped descriptors
    happen to do. Anything a caller would notice can be overridden.

    `acquisition_cls` defaults to the real `TiingoAcquisition` because D-03's
    "two descriptors may name the same class" case needs exactly that, and
    nothing here constructs it.
    """
    from quantlab.acquisition.registry import Capability, SourceDescriptor

    fields = {
        "vendor": vendor,
        "display_name": f"Fake {vendor}",
        "acquisition_cls": tiingo.TiingoAcquisition,
        "config_factory": lambda **kw: None,
        "capabilities": (Capability(market="us_equity", frequency="1d"),),
        "required_env": (),
    }
    fields.update(overrides)
    return SourceDescriptor(**fields)  # type: ignore[arg-type]


def test_one_descriptor_per_vendor_rejects_a_duplicate(isolated_registry) -> None:
    """D-01: a second descriptor for a registered vendor RAISES.

    It must not merge the two capability lists and must not replace the first
    -- a registry that quietly took the last definition would let IMPORT ORDER
    decide what a vendor can do, which is the one thing enumeration order was
    sorted to stop mattering about.

    Registered under a NOVEL vendor token so the collision is created here
    rather than borrowed from the shipped descriptors: the mechanism is proved
    by construction, not by re-asserting a fact that already holds.
    """
    from quantlab.acquisition.registry import register_source

    first = _fake_descriptor("fakevendor")
    register_source(first)
    assert isolated_registry.get("fakevendor") is first

    second = _fake_descriptor("fakevendor", display_name="Impostor")
    with pytest.raises(ValueError, match="fakevendor"):
        register_source(second)

    # Neither replaced nor merged.
    assert isolated_registry.get("fakevendor") is first
    assert second not in isolated_registry.SOURCES
    assert [d.vendor for d in isolated_registry.SOURCES].count("fakevendor") == 1


def test_capabilities_are_dataclass_instances_and_reject_the_cross_product() -> None:
    """D-02: capabilities are `Capability` INSTANCES, and the set is not a
    cross-product of two flat tuples.

    Alpaca's four entries and Tiingo's one are asserted exactly. A
    `markets x frequencies` product over Alpaca would yield `(us_equity, 1d)`,
    `(us_equity, 1m)` and `(us_equity, tick)` with no data_type at all -- it
    could express neither the quotes/trades split nor the fact that `tick` is
    the only frequency carrying one. `supports()` is asserted in both
    directions for the same reason: the negative case is the one a product
    would get wrong.
    """
    from quantlab.acquisition.registry import Capability, DataSourceRegistry

    alpaca_source = DataSourceRegistry.get("alpaca")
    tiingo_source = DataSourceRegistry.get("tiingo")

    for descriptor in (alpaca_source, tiingo_source):
        assert isinstance(descriptor.capabilities, tuple)
        assert descriptor.capabilities
        for capability in descriptor.capabilities:
            assert isinstance(capability, Capability)

    def triples(descriptor):
        return {
            (c.market, c.frequency, c.data_type) for c in descriptor.capabilities
        }

    assert triples(alpaca_source) == {
        ("us_equity", "1d", "bars"),
        ("us_equity", "1m", "bars"),
        ("us_equity", "tick", "quotes"),
        ("us_equity", "tick", "trades"),
    }
    assert triples(tiingo_source) == {("us_equity", "1d", None)}

    assert tiingo_source.supports("us_equity", "1d") is True
    assert tiingo_source.supports("us_equity", "tick") is False
    assert alpaca_source.supports("us_equity", "tick", "quotes") is True
    assert alpaca_source.supports("us_equity", "tick", "trades") is True
    assert alpaca_source.supports("us_equity", "1m", "quotes") is False
    assert alpaca_source.supports("crypto_spot", "1d") is False


def test_capabilities_preserve_declaration_order_across_repeated_reads() -> None:
    """D-02: `capabilities` is an ordered tuple, not a set.

    Enumerating one descriptor's capabilities twice must yield the same
    sequence, because an operator surface renders them in that order and a set
    would reorder them per process (string hashing is salted per interpreter
    run). Asserted as a SEQUENCE, so a change to a frozenset fails here rather
    than only on a machine whose hash seed happens to differ.
    """
    from quantlab.acquisition.registry import DataSourceRegistry

    for descriptor in DataSourceRegistry.all():
        first = list(descriptor.capabilities)
        second = list(descriptor.capabilities)
        assert first == second
        assert isinstance(descriptor.capabilities, tuple)

    assert [
        (c.frequency, c.data_type)
        for c in DataSourceRegistry.get("alpaca").capabilities
    ] == [("1d", "bars"), ("1m", "bars"), ("tick", "quotes"), ("tick", "trades")]


def test_capabilities_match_the_vendor_class_constants() -> None:
    """D-02: the capability set is PINNED against the vendor class's own
    constants, not restated independently of them.

    This is what turns "someone added an endpoint to `AlpacaAcquisition` and
    forgot the descriptor" into a red test instead of a source the console
    under-advertises forever. Three separate pins, because the class carries
    three separate facts:

    - bar frequencies <-> `TIMEFRAME_MAP` keys;
    - tick data types <-> `TICK_DATA_TYPES`;
    - every declared data_type is a key of `ENDPOINT_MAP`, i.e. a capability
      cannot name a shape the class has no endpoint for.

    Tiingo is pinned the other way round: it declares `data_type=None`, so the
    assertion is that it has no endpoint map to disagree with and exactly one
    frequency, matching `_FREQUENCY_MAP`.
    """
    from quantlab.acquisition.registry import DataSourceRegistry

    alpaca_source = DataSourceRegistry.get("alpaca")
    alpaca_cls = alpaca.AlpacaAcquisition

    bar_frequencies = {
        c.frequency for c in alpaca_source.capabilities if c.data_type == "bars"
    }
    assert bar_frequencies == set(alpaca_cls.TIMEFRAME_MAP)

    tick_types = {
        c.data_type for c in alpaca_source.capabilities if c.frequency == "tick"
    }
    assert tick_types == set(alpaca_cls.TICK_DATA_TYPES)

    for capability in alpaca_source.capabilities:
        assert capability.data_type in alpaca_cls.ENDPOINT_MAP

    tiingo_source = DataSourceRegistry.get("tiingo")
    assert {c.frequency for c in tiingo_source.capabilities} == set(
        tiingo._FREQUENCY_MAP
    )
    assert all(c.data_type is None for c in tiingo_source.capabilities)
    assert not hasattr(tiingo.TiingoAcquisition, "ENDPOINT_MAP")


def test_direct_class_reference_is_the_class_object(isolated_registry) -> None:
    """D-03: `acquisition_cls` is the CLASS OBJECT, never a dotted string.

    Asserted with `is`, so a dotted path that happened to resolve to the same
    class would still fail -- the point of D-03 is the direct reference itself,
    which is what makes `descriptor.acquisition_cls.DEFAULT_BATCH_SIZE`
    readable at a call site without a resolver.

    The second half is the D-03 corollary: only VENDOR collides. Two
    descriptors may name the SAME class under different vendors (alpaca paper
    vs live is the motivating case), so registering one must not raise.
    """
    from quantlab.acquisition.registry import register_source

    assert isolated_registry.get("tiingo").acquisition_cls is tiingo.TiingoAcquisition
    assert isolated_registry.get("alpaca").acquisition_cls is alpaca.AlpacaAcquisition
    for descriptor in isolated_registry.all():
        assert isinstance(descriptor.acquisition_cls, type)
        assert not isinstance(descriptor.acquisition_cls, str)

    twin = _fake_descriptor(
        "tiingotwin", acquisition_cls=tiingo.TiingoAcquisition
    )
    assert register_source(twin) is twin
    assert isolated_registry.get("tiingotwin").acquisition_cls is (
        isolated_registry.get("tiingo").acquisition_cls
    )


def test_registration_tuple_shape_is_instances_in_a_rebound_tuple() -> None:
    """D-05: `SOURCES` is a `tuple` of `SourceDescriptor` INSTANCES, rebound
    rather than mutated.

    This test is the ONLY thing keeping `tests/conftest.py:isolated_registry`
    honest, and the fixture's own docstring says so. That fixture isolates by
    `monkeypatch.setattr`-ing `SOURCES`, which saves the old OBJECT and
    restores it on teardown -- correct only while registration REBINDS the
    attribute. Switch `register_source` to `list.append` and the fixture would
    mutate the very object it restores: isolation would silently stop working
    and a fake descriptor would leak into every later test in the session, with
    nothing anywhere going red. Hence the source-level arm below.

    The instance-vs-class arm is the other half of D-05: `MEMBERSHIP_FETCHERS`
    holds classes, this holds instances, and a class accidentally registered
    here would still satisfy a `len()` check.
    """
    import ast

    from quantlab.acquisition.registry import DataSourceRegistry, SourceDescriptor

    assert isinstance(DataSourceRegistry.SOURCES, tuple)
    assert DataSourceRegistry.SOURCES
    for element in DataSourceRegistry.SOURCES:
        assert isinstance(element, SourceDescriptor)
        assert not isinstance(element, type)

    source = Path("quantlab/acquisition/registry.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    # No `.append` may reach SOURCES -- neither `SOURCES.append(...)` nor
    # `DataSourceRegistry.SOURCES.append(...)`. Matched on the ATTRIBUTE CHAIN
    # rather than on the rendered text, so a comment or docstring mentioning
    # `.append` (this one does) is not a false positive.
    appends = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        and "SOURCES" in ast.unparse(node.func)
    ]
    assert appends == [], appends

    # ... and the rebinding is really there, as an AugAssign onto SOURCES.
    rebinds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Attribute)
        and node.target.attr == "SOURCES"
    ]
    assert len(rebinds) == 1, ast.dump(tree) and len(rebinds)


def test_decorator_registers_and_returns_the_descriptor(isolated_registry) -> None:
    """D-06: `register_source(d) is d`, and applying it twice raises.

    The identity return is what makes the decorated module-level name the
    descriptor rather than `None` -- the classic decorator bug, which would
    only surface at the first attribute read somewhere far away.

    Applying the decorator to the SAME OBJECT a second time must take the
    duplicate-vendor path rather than being special-cased as idempotent: a
    module imported twice under two names is a real way that happens, and
    registering it twice would double the row in an operator's list.
    """
    from quantlab.acquisition.registry import register_source

    descriptor = _fake_descriptor("decoratedvendor")
    assert register_source(descriptor) is descriptor
    assert descriptor in isolated_registry.SOURCES
    assert isolated_registry.get("decoratedvendor") is descriptor

    with pytest.raises(ValueError, match="already registered"):
        register_source(descriptor)
    assert isolated_registry.SOURCES.count(descriptor) == 1


def test_the_isolated_registry_fixture_restored_the_fake_vendors() -> None:
    """The teardown half of the test above, which it cannot assert itself.

    A fixture that snapshots but never restores looks identical from inside
    the test that used it. This runs AFTER those tests in file order and
    asserts the session-global registry is back to the two shipped
    descriptors, so a leak is caught here rather than surfacing as an
    inexplicable third row in some later plan's enumeration assertion.
    """
    from quantlab.acquisition.registry import DataSourceRegistry

    assert [d.vendor for d in DataSourceRegistry.all()] == ["alpaca", "tiingo"]
    for leaked in ("fakevendor", "decoratedvendor", "tiingotwin"):
        assert leaked not in {d.vendor for d in DataSourceRegistry.SOURCES}


def test_enumeration_order_is_sorted_by_vendor() -> None:
    """D-06: `all()` sorts by vendor rather than returning import order.

    Import order is a function of which module the caller touched first, so
    two installations of the same code would render an operator's source list
    differently. Sorting also makes this assertion a literal comparison.
    """
    from quantlab.acquisition.registry import DataSourceRegistry

    assert [d.vendor for d in DataSourceRegistry.all()] == ["alpaca", "tiingo"]

    vendors = [d.vendor for d in DataSourceRegistry.all()]
    assert vendors == sorted(vendors)


def test_registry_get_is_empty_and_unknown_safe(
    isolated_registry, monkeypatch
) -> None:
    """D-01: `all()` over an EMPTY registry returns `()` rather than raising,
    and `get()` for an unknown vendor names both the request and what exists.

    "Nothing is registered" is a legitimate state a console must be able to
    render -- an exception would make the empty case the caller's problem at
    exactly the moment it has the least information. `get()` is the opposite:
    a miss is a caller error, so it raises, and the message lists the
    registered vendors because a bare `KeyError` says nothing about a registry
    the caller cannot see.
    """
    monkeypatch.setattr(isolated_registry, "SOURCES", ())

    assert isolated_registry.all() == ()

    with pytest.raises(ValueError, match="tiingo") as excinfo:
        isolated_registry.get("tiingo")
    assert "Registered vendors: []" in str(excinfo.value)

    monkeypatch.setattr(isolated_registry, "SOURCES", (_fake_descriptor("only"),))
    with pytest.raises(ValueError) as excinfo:
        isolated_registry.get("nosuchvendor")
    assert "nosuchvendor" in str(excinfo.value)
    assert "only" in str(excinfo.value)


def test_registry_reaches_no_zarr_writer() -> None:
    """D-14's acquisition-only amendment, proved NEGATIVELY.

    `run()` downloads to the raw parquet tier and stops; the raw-to-Zarr
    conversion stays in the shells, because the three entry points convert in
    three different modes with three differently-sized RAM guards. The cheapest
    durable proof of "reaches no Zarr writer" is that this module imports no
    `quantlab.dataset` module at all -- a behavioural test could only show that
    one particular call did not convert.

    An `ast` walk rather than a substring scan: the docstrings in this file
    discuss Zarr conversion by name, and a grep would fail on the prose that
    explains the rule.
    """
    import ast

    tree = ast.parse(
        Path("quantlab/acquisition/registry.py").read_text(encoding="utf-8")
    )
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.add(module)
            imported.update(f"{module}.{alias.name}" for alias in node.names)

    offending = sorted(
        name
        for name in imported
        if name == "quantlab.dataset" or name.startswith("quantlab.dataset.")
    )
    assert offending == [], offending
    assert "quantlab.base.acquisition" in imported
