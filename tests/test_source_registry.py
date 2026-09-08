"""Home for the data-source registry proofs: ROADMAP success criteria SC-1 and
SC-2, requirements D-01..D-07.

SC-1 — every acquirable source is enumerable from ONE registry, with no vendor
class named at the call site; each descriptor carries its capabilities, its
credential env-var NAMES and its acquisition class.
SC-2 — enumeration and credential status never return a credential VALUE and
never require one to be present.

Scaffolded by plan 03.4-01 (Wave 0). The behaviour those criteria describe does
not exist yet; plan 03.4-02 builds it and fills this file in.

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
   in rule 2 it is the required result. The three tests below are
   genuine infrastructure self-tests: each pins a contract that the plan-02
   work depends on, and each fails if that contract drifts.

2. A `-k` selector name must not be attached to a test that does not honestly
   cover that selector's behaviour. In 03.2 a mechanism was deleted and
   `-k fingerprint` stayed green, because the only test covering it was named
   outside its own selector. So none of the tests below is named for a
   selector `03.4-VALIDATION.md` assigns to a later plan
   (`one_descriptor_per_vendor`, `capabilities`,
   `capabilities_match_the_vendor_class`, `direct_class_reference`,
   `env_names_are_exactly_what_gates_construction`,
   `never_returns_a_credential_value`, `registration_tuple_shape`,
   `decorator_registers`, `enumeration_order`,
   `enumeration_is_complete_from_a_cold_import`). Those selectors must match
   ZERO tests until the behaviour they name exists. Each later plan is
   responsible for making its own selector match.
"""

import json
import os
from pathlib import Path

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
