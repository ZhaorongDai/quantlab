"""The WRDS vendor SEAM: one account, several products (phase 03.10, plan 01).

Phase 03.9 registered the `wrds` descriptor inside `wrds_taq.py`, beside the
one provider that existed. Plan 02 adds a SECOND provider (CRSP daily) to the
same vendor, and a registration that lives inside a provider module would then
have to name the other provider's classes -- so `wrds_taq` would import
`wrds_crsp` (or the reverse) purely to be registered, and importing either one
first would cycle through the registry.

This file pins the shape that makes the second provider a one-row change:

- the descriptor MOVED to the neutral `quantlab/acquisition/wrds.py`, which
  imports the provider modules; `wrds_taq.py` imports nothing from the registry;
- the registry's bottom vendor import names `wrds`, so a cold
  `import quantlab.acquisition.registry` still enumerates every source, and so
  does an import that touches a provider module FIRST;
- the one `WrdsSession` offers GENERIC `schema_usable` / `fetch_rows` /
  `copy_csv`, so the CRSP provider reaches the shared connection without adding
  CRSP-shaped methods to a TAQ module -- and TAQ's own methods delegate to them
  without changing a byte of their SQL.

EVERY test here is OFFLINE (D-13). The autouse `_forbid_wrds_network` tripwire
in `tests/conftest.py` makes `psycopg2.connect` raise in every test; the
session tests install their own `FakeConnection` factory with
`monkeypatch.setattr("psycopg2.connect", ...)` inside the test body, which is
the only sanctioned way past it (03.9 D-28). Nothing here can push Duo.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import sql

from tests.wrds_fixtures import fake_connect, render_composed

REPO_ROOT = Path(__file__).resolve().parents[1]
WRDS_TAQ_SOURCE = REPO_ROOT / "quantlab" / "acquisition" / "wrds_taq.py"
REGISTRY_SOURCE = REPO_ROOT / "quantlab" / "acquisition" / "registry.py"

USER = "test-wrds-user-not-real"
SENTINEL_PW = "SENTINEL-PW"
PGPASS_LINE = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:{SENTINEL_PW}\n"

#: The EXACT statement `has_schema_usage` recorded before plan 03.10-01 moved
#: its body into the generic `schema_usable`. Written as a literal rather than
#: rebuilt from the code under test: a pin derived from its own subject proves
#: only that the subject is self-consistent, and what D-03 forbids is a CHANGE
#: to the SQL text, which only a literal can catch.
SCHEMA_USAGE_SQL = (
    "SELECT has_schema_privilege(oid, 'USAGE') "
    "FROM pg_namespace WHERE nspname = %s"
)

#: Likewise for the COPY, rendered from the pre-change `copy_query` for the
#: fixed (day, pairs, columns) triple the test below replays. `copy_query`
#: itself is untouched by this plan; this literal is what proves it.
COPY_DAY = date(2024, 1, 24)
COPY_PAIRS = [("AAPL", None), ("BRK", "B")]
COPY_COLUMNS = (
    "date",
    "time_m",
    "sym_root",
    "sym_suffix",
    "best_bid",
    "best_ask",
)
COPY_SQL = (
    'COPY (SELECT "date", "time_m", "sym_root", "sym_suffix", "best_bid", '
    '"best_ask" FROM "taqm_2024"."complete_nbbo_20240124" WHERE sym_root = '
    "ANY(ARRAY['AAPL', 'BRK']) AND (sym_root, coalesce(sym_suffix, '')) IN "
    "(('AAPL', ''), ('BRK', 'B'))) TO STDOUT WITH (FORMAT csv, HEADER true)"
)


# -- helpers -------------------------------------------------------------------


@pytest.fixture
def pgpass(monkeypatch, tmp_path):
    """A pgpass file libpq would accept, pointed to by `PGPASSFILE`.

    Copied from `tests/test_wrds_taq_acquisition.py:pgpass` with this
    attribution comment: `tests/` carries no shared fixture module for these,
    and a session test without a valid pgpass entry fails in
    `_assert_pgpass_entry` before it ever reaches the helper under test.
    """

    def _write(text: str = PGPASS_LINE, mode: int = 0o600) -> Path:
        path = tmp_path / "pgpass"
        path.write_text(text)
        os.chmod(path, mode)
        monkeypatch.setenv("PGPASSFILE", str(path))
        return path

    return _write


@pytest.fixture
def live_session(monkeypatch, pgpass):
    """A REAL `WrdsSession` for USER, with `psycopg2.connect` replaced by the
    recording `FakeConnection` factory. Returns `(session, connections)`.

    The REAL class deliberately, not `FakeWrdsSession`: what these tests prove
    is that the new helpers run through `_query` -- the username scrub and the
    broken-session rule -- and a session double would satisfy every assertion
    while the real one leaked. Copied from
    `tests/test_wrds_taq_acquisition.py:live_session`.
    """
    from quantlab.acquisition.wrds_taq import WrdsSession

    pgpass()
    monkeypatch.setenv("WRDS_USERNAME", USER)
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
        monkeypatch.delenv(name, raising=False)
    connections: list = []
    monkeypatch.setattr("psycopg2.connect", fake_connect(connections))
    return WrdsSession.shared(), connections


def _run_child(body: str):
    """Run `body` in a FRESH interpreter, rooted at the repo, and return it.

    A subprocess is REQUIRED rather than fastidious: this pytest session has
    already imported the registry and both WRDS modules for other reasons, so
    an in-process assertion about import ORDER would measure the state other
    tests left behind. Same idiom, and same reason, as
    `tests/test_source_registry.py:_run_child`.
    """
    env = dict(os.environ)
    env.pop("WRDS_USERNAME", None)
    return subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env=env,
    )


def _taq_tree() -> ast.Module:
    return ast.parse(WRDS_TAQ_SOURCE.read_text(encoding="utf-8"))


# -- the descriptor moved (D-12) -----------------------------------------------


def test_wrds_descriptor_lives_in_the_neutral_module() -> None:
    """`quantlab.acquisition.wrds.WRDS_SOURCE` IS the registered `wrds`
    descriptor, unchanged in everything an operator can see.

    Asserted by IDENTITY against `DataSourceRegistry.get("wrds")`, so a move
    that left a second descriptor object behind -- registered from one module,
    imported from the other -- fails here rather than as a mystifying duplicate
    row much later.

    The capability carries its own `acquisition_cls` / `config_factory` (plan
    01 Task 1), which is what plan 02's CRSP row differs in. The descriptor
    DEFAULT is still the same pair, so every ingest shell reading
    `SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE` keeps working.

    **This test does NOT pin the capability SET**, and that is deliberate: the
    subject here is the MOVE (one descriptor object, reached from the neutral
    module, defaults intact), and the whole point of the move was to let the
    vendor grow a second capability. The inventory is owned by
    `tests/test_source_registry.py:
    test_wrds_descriptor_serves_nbbo_and_crsp_daily_capabilities`, which
    asserts the exact set -- so nothing is unpinned, it is pinned in the one
    place that is about inventory.
    """
    from quantlab.acquisition.registry import DataSourceRegistry
    from quantlab.acquisition.wrds import WRDS_SOURCE
    from quantlab.acquisition.wrds_taq import WrdsTaqNbboAcquisition
    from quantlab.dataset.nbbo import NbboPanelDataset

    assert WRDS_SOURCE is DataSourceRegistry.get("wrds")
    assert ("us_equity", "tick", "nbbo") in {
        (c.market, c.frequency, c.data_type) for c in WRDS_SOURCE.capabilities
    }

    (capability,) = WRDS_SOURCE.capabilities_for("us_equity", "tick", "nbbo")
    assert capability.dataset_cls is NbboPanelDataset
    assert capability.acquisition_cls is WrdsTaqNbboAcquisition
    assert capability.config_factory == WrdsTaqNbboAcquisition.build_config

    assert WRDS_SOURCE.acquisition_cls is WrdsTaqNbboAcquisition
    assert WRDS_SOURCE.config_factory == WrdsTaqNbboAcquisition.build_config
    assert WRDS_SOURCE.required_env == ("WRDS_USERNAME",)


def test_wrds_taq_registers_nothing_and_imports_no_registry() -> None:
    """The provider module is REGISTRATION-FREE: no `register_source` call, and
    no import from `quantlab.acquisition.registry` or `.wrds`.

    That is the whole anti-cycle property. When plan 02 adds `wrds_crsp.py`,
    only `wrds.py` imports both providers; a provider importing the registry
    (for the decorator) or the neutral module (for the descriptor) is exactly
    the edge that would close the loop.

    An `ast` walk rather than a substring scan: this module's docstrings name
    the registry in prose, and a grep would fail on the explanation of the rule.
    """
    tree = _taq_tree()

    calls = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and "register_source" in ast.unparse(node.func)
    ]
    assert calls == [], calls

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
        if name == "quantlab.acquisition.registry"
        or name.startswith("quantlab.acquisition.registry.")
        or name == "quantlab.acquisition.wrds"
        or name.startswith("quantlab.acquisition.wrds.")
    )
    assert offending == [], offending


def test_registry_bottom_import_names_the_neutral_module() -> None:
    """`registry.py` imports `wrds`, in the MODULE-OBJECT form.

    The form is load-bearing, and the file's own comment says why: when a
    caller imports a vendor module first, the registry runs while that module
    is only partially initialised, and binding the module object is safe where
    reading an attribute off it would raise.
    """
    source = REGISTRY_SOURCE.read_text(encoding="utf-8")

    assert "from quantlab.acquisition import wrds as _wrds" in source

    tree = ast.parse(source)
    from_registry = {
        f"{node.module}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "quantlab.acquisition"
        for alias in node.names
    }
    assert "quantlab.acquisition.wrds" in from_registry
    assert "quantlab.acquisition.wrds_taq" not in from_registry


@pytest.mark.parametrize(
    "first_module",
    ["quantlab.acquisition.wrds_taq", "quantlab.acquisition.wrds"],
)
def test_enumeration_survives_either_wrds_import_order(first_module) -> None:
    """Importing EITHER WRDS module first still enumerates every source.

    The failure this guards against is silent in one direction only: with the
    descriptor inside a provider module, importing that provider first and the
    registry second can leave a half-initialised module in `sys.modules`, and
    the vendor list an operator sees then depends on which module the caller
    happened to touch. Both orders are asserted, in fresh interpreters, with
    `WRDS_USERNAME` stripped -- so this doubles as a proof that enumeration
    needs no credential.
    """
    child = _run_child(
        f"import {first_module}\n"
        "from quantlab.acquisition.registry import DataSourceRegistry\n"
        "print(sorted(d.vendor for d in DataSourceRegistry.all()))\n"
    )

    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "['alpaca', 'tiingo', 'wrds']", child.stdout
    assert "Traceback" not in child.stderr


def test_the_moved_descriptor_is_registered_exactly_once() -> None:
    """One descriptor for the vendor, from a cold import: the move must not
    leave a second registration behind.

    `register_source` raises on a duplicate vendor, so a leftover registration
    in `wrds_taq.py` would make a cold import of the registry FAIL rather than
    double the row -- which is why the assertion is on the child's exit code as
    much as on the count.
    """
    child = _run_child(
        "import json\n"
        "from quantlab.acquisition.registry import DataSourceRegistry\n"
        "vendors = [d.vendor for d in DataSourceRegistry.SOURCES]\n"
        "print(json.dumps({'vendors': vendors}))\n"
    )

    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout)["vendors"].count("wrds") == 1


# -- the generic session helpers (D-03) ----------------------------------------


def test_session_helpers_are_generic_and_provider_neutral(live_session) -> None:
    """`schema_usable` / `fetch_rows` / `copy_csv` work for ANY schema and ANY
    composed query -- nothing in them is TAQ-shaped.

    `schema_usable` is asked with a CRSP schema on purpose: it is the call plan
    02 makes, and a helper that still built a `taqm_` name internally would
    pass a TAQ-only assertion while being useless to the second provider. The
    schema travels as a QUERY PARAMETER, never interpolated into the text.

    Both entitlement answers are asserted, because the `pg_namespace` form
    returns no row (rather than raising) for a schema that does not exist, and
    "not entitled" and "no such schema" must both read as `False`.
    """
    session, connections = live_session

    assert session.schema_usable("crsp_a_stock") is True
    text, params = connections[0].executed[-1]
    assert text == SCHEMA_USAGE_SQL
    assert params == ("crsp_a_stock",)
    assert "crsp_a_stock" not in text

    connections[0].entitled = False
    assert session.schema_usable("crsp_a_stock") is False

    connections[0].tables = ["complete_nbbo_20240124", "complete_nbbo_20240125"]
    rows = session.fetch_rows(
        sql.SQL("SELECT table_name FROM information_schema.tables")
    )
    assert rows == [("complete_nbbo_20240124",), ("complete_nbbo_20240125",)]

    connections[0].copy_payload = b"a,b\n1,2\n"
    payload = session.copy_csv(
        sql.SQL("COPY (SELECT 1) TO STDOUT WITH (FORMAT csv, HEADER true)")
    )
    assert payload == b"a,b\n1,2\n"

    # ONE connection for all of it (D-20): every helper reuses the session.
    assert len(connections) == 1


@pytest.mark.parametrize("helper", ["schema_usable", "fetch_rows", "copy_csv"])
def test_session_helpers_scrub_the_username_and_never_reconnect(
    helper, monkeypatch, pgpass
) -> None:
    """Every new helper runs through `_query` (T-03.10-43).

    Two controls ride on that, and both are asserted per helper rather than
    once for the set: a helper that reached `self._connection()` directly would
    put the driver's text -- which carries the role name -- into an exception
    message and would leave the session reusable after a driver error, so the
    next call reconnects and pushes Duo.

    The planted username is the one in the driver's error text, so "scrubbed"
    is a real substitution here and not a vacuous absence.
    """
    from quantlab.acquisition.wrds_taq import WrdsSession, WrdsSessionError

    pgpass()
    monkeypatch.setenv("WRDS_USERNAME", USER)
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
        monkeypatch.delenv(name, raising=False)

    connections: list = []
    base = fake_connect(connections)
    failure = psycopg2.OperationalError(f"FATAL: role {USER} is denied")

    def _connect(*args, **kwargs):
        connection = base(*args, **kwargs)
        connection.fail_with = failure
        return connection

    monkeypatch.setattr("psycopg2.connect", _connect)
    session = WrdsSession.shared()

    calls = {
        "schema_usable": lambda: session.schema_usable("crsp_a_stock"),
        "fetch_rows": lambda: session.fetch_rows(sql.SQL("SELECT 1")),
        "copy_csv": lambda: session.copy_csv(
            sql.SQL("COPY (SELECT 1) TO STDOUT WITH (FORMAT csv)")
        ),
    }

    with pytest.raises(WrdsSessionError) as excinfo:
        calls[helper]()
    message = str(excinfo.value)
    assert USER not in message
    assert "$WRDS_USERNAME" in message

    # The session is BROKEN, not merely failed: a second call raises without a
    # new connect, so a retry cannot push a second Duo prompt.
    with pytest.raises(WrdsSessionError, match="re-run"):
        calls[helper]()
    assert len(connections) == 1


def test_taq_methods_delegate_without_changing_their_sql(live_session) -> None:
    """`has_schema_usage` and `copy_nbbo_csv` issue EXACTLY the statements they
    issued before the generic helpers existed (D-03).

    Both are compared against literal text captured from the pre-change module,
    not against anything rebuilt from the code under test. The TAQ path is the
    one thing this plan must not disturb: phase 03.9's live-verified NBBO pull
    depends on the COPY having no ORDER BY, no GROUP BY, no DISTINCT and no
    time predicate (D-19), and a "harmless" rewrite during the delegation is
    precisely how that guarantee would be lost.
    """
    session, connections = live_session

    assert session.has_schema_usage(2024) is True
    text, params = connections[0].executed[-1]
    assert text == SCHEMA_USAGE_SQL
    assert params == ("taqm_2024",)

    connections[0].copy_payload = b"header\n"
    payload = session.copy_nbbo_csv(COPY_DAY, COPY_PAIRS, COPY_COLUMNS)
    assert payload == b"header\n"

    copy_text, copy_params = connections[0].executed[-1]
    assert copy_text == COPY_SQL
    assert copy_params is None
    # Restated as the property, so a future re-render that still matched the
    # literal by accident could not drift away from the builder either.
    assert copy_text == render_composed(
        session.copy_query(COPY_DAY, COPY_PAIRS, COPY_COLUMNS)
    )
