"""WRDS NYSE TAQ NBBO provider hardening (phase 03.9, plan 04).

Every test here is OFFLINE. The autouse `_forbid_wrds_network` tripwire in
`tests/conftest.py` makes `psycopg2.connect` raise in every test; a test that
exercises the REAL `WrdsSession` installs its own `FakeConnection` factory
with `monkeypatch.setattr("psycopg2.connect", ...)` inside its body, which is
the only sanctioned way past the tripwire (D-28). Nothing here can push Duo.
"""

from __future__ import annotations

import ast
import builtins
import getpass
import os
from pathlib import Path

import psycopg2
import pytest
from loguru import logger

from tests.wrds_fixtures import fake_connect

WRDS_TAQ_SOURCE = (
    Path(__file__).resolve().parents[1] / "quantlab" / "acquisition" / "wrds_taq.py"
)

USER = "test-wrds-user-not-real"
SENTINEL_PW = "SENTINEL-PW"
PGPASS_LINE = f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:{SENTINEL_PW}\n"


# -- helpers -------------------------------------------------------------------


@pytest.fixture
def no_prompts(monkeypatch):
    """`input()` and `getpass.getpass()` raise if ANY code path reaches them."""

    def _refuse(*args, **kwargs):
        raise AssertionError("an interactive credential prompt was reached")

    monkeypatch.setattr(builtins, "input", _refuse)
    monkeypatch.setattr(getpass, "getpass", _refuse)


@pytest.fixture
def pgpass(monkeypatch, tmp_path):
    """A factory writing a pgpass file with the given text and mode, pointed
    to by `PGPASSFILE`."""

    def _write(text: str = PGPASS_LINE, mode: int = 0o600) -> Path:
        path = tmp_path / "pgpass"
        path.write_text(text)
        os.chmod(path, mode)
        monkeypatch.setenv("PGPASSFILE", str(path))
        return path

    return _write


@pytest.fixture
def live_session(monkeypatch, no_prompts):
    """A REAL `WrdsSession` for USER with `psycopg2.connect` replaced by a
    recording `FakeConnection` factory. Returns `(session, connections)`."""
    from quantlab.acquisition.wrds_taq import WrdsSession

    monkeypatch.setenv("WRDS_USERNAME", USER)
    for name in ("PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"):
        monkeypatch.delenv(name, raising=False)
    connections: list = []
    monkeypatch.setattr("psycopg2.connect", fake_connect(connections))
    return WrdsSession.shared(), connections


@pytest.fixture
def log_records():
    """Every loguru message emitted during the test, as text."""
    records: list[str] = []
    handler = logger.add(lambda message: records.append(str(message)), level=0)
    yield records
    logger.remove(handler)


# -- D-07: credentials ------------------------------------------------------------


def test_credential_missing_username_fails_construction(
    monkeypatch, acquisition_config
):
    from quantlab.acquisition.wrds_taq import WrdsTaqNbboAcquisition

    monkeypatch.delenv("WRDS_USERNAME", raising=False)
    with pytest.raises(RuntimeError, match="WRDS_USERNAME"):
        WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))


def test_credential_planted_username_is_absent_from_config_and_scrubbed(
    mock_wrds_session, monkeypatch, acquisition_config
):
    from quantlab.acquisition.wrds_taq import WrdsTaqNbboAcquisition

    planted = "planted-wrds-user-7f3c"
    monkeypatch.setenv("WRDS_USERNAME", planted)
    acq = WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))
    assert planted not in str(acq.config.to_dict())
    scrubbed = acq._scrub(f"FATAL: password authentication failed for {planted}")
    assert planted not in scrubbed
    assert acq.REDACTION in scrubbed


def test_pgpass_valid_entry_connects_once_with_pinned_parameters(
    live_session, pgpass, log_records
):
    session, connections = live_session
    pgpass()
    session.trading_days(2024)
    assert len(connections) == 1
    kwargs = connections[0].kwargs
    assert kwargs["host"] == "wrds-pgdata.wharton.upenn.edu"
    assert kwargs["port"] == 9737
    assert kwargs["dbname"] == "wrds"
    assert kwargs["user"] == USER
    assert kwargs["sslmode"] == "require"
    assert "password" not in kwargs
    assert connections[0].sessions == [{"readonly": True, "autocommit": True}]

    session.trading_days(2024)
    session.table_columns(__import__("datetime").date(2024, 1, 24))
    session.trading_days(2023)
    assert len(connections) == 1, "one session object must connect at most once"
    assert all(SENTINEL_PW not in record for record in log_records)


@pytest.mark.parametrize(
    "line",
    [
        f"*:*:*:{USER}:x\n",
        f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{USER}:pa\\:ss\\:word\n",
        f"# a comment\n\nother.host:5432:db:someone:pw\n*:9737:*:{USER}:pw\n",
    ],
    ids=["wildcards", "escaped-colon-in-password", "comment-and-second-line"],
)
def test_pgpass_accepts_wildcard_and_escaped_lines(live_session, pgpass, line):
    session, connections = live_session
    pgpass(line)
    session.trading_days(2024)
    assert len(connections) == 1


@pytest.mark.parametrize(
    "setup, expected",
    [
        ("missing", "does not exist"),
        ("mode-0644", "chmod 600"),
        ("no-matching-line", "<password>"),
        ("other-user", "<password>"),
    ],
)
def test_pgpass_problems_fail_before_connect_without_leaking(
    live_session, pgpass, tmp_path, monkeypatch, log_records, setup, expected
):
    session, connections = live_session
    if setup == "missing":
        path = tmp_path / "absent-pgpass"
        monkeypatch.setenv("PGPASSFILE", str(path))
    elif setup == "mode-0644":
        path = pgpass(mode=0o644)
    elif setup == "no-matching-line":
        path = pgpass(f"other.host:9737:wrds:{USER}:{SENTINEL_PW}\n")
    else:
        path = pgpass(
            f"wrds-pgdata.wharton.upenn.edu:9737:wrds:someone-else:{SENTINEL_PW}\n"
        )

    with pytest.raises(RuntimeError) as excinfo:
        session.trading_days(2024)
    message = str(excinfo.value)
    assert connections == [], "the connection must never be attempted"
    assert str(path) in message
    assert expected in message
    assert SENTINEL_PW not in message
    assert USER not in message, "the username is referred to as $WRDS_USERNAME"
    assert all(SENTINEL_PW not in record for record in log_records)


def test_connect_ignores_pghost(live_session, pgpass, monkeypatch):
    session, connections = live_session
    pgpass()
    monkeypatch.setenv("PGHOST", "evil.example")
    session.trading_days(2024)
    assert connections[0].kwargs["host"] == "wrds-pgdata.wharton.upenn.edu"


@pytest.mark.parametrize("name", ["PGHOSTADDR", "PGSERVICE", "PGSERVICEFILE"])
def test_connect_refuses_redirecting_env(live_session, pgpass, monkeypatch, name):
    session, connections = live_session
    pgpass()
    monkeypatch.setenv(name, "somewhere")
    with pytest.raises(RuntimeError, match=name):
        session.trading_days(2024)
    assert connections == []


def test_session_broken_by_driver_error_never_reconnects(
    live_session, pgpass, monkeypatch
):
    from quantlab.acquisition.wrds_taq import WrdsSessionError

    session, connections = live_session
    pgpass()
    session.trading_days(2024)
    connections[0].fail_with = psycopg2.OperationalError("server closed the connection")

    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)

    def _no_second_connect(*args, **kwargs):
        raise AssertionError("a broken WrdsSession reconnected (Duo push)")

    monkeypatch.setattr("psycopg2.connect", _no_second_connect)
    with pytest.raises(WrdsSessionError, match="re-run"):
        session.trading_days(2024)
    assert len(connections) == 1


def test_session_failed_connect_is_not_retried(live_session, pgpass, monkeypatch):
    from quantlab.acquisition.wrds_taq import WrdsSessionError

    session, _ = live_session
    pgpass()
    attempts: list[int] = []

    def _failing_connect(**kwargs):
        attempts.append(1)
        raise psycopg2.OperationalError("Duo push denied")

    monkeypatch.setattr("psycopg2.connect", _failing_connect)
    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)
    with pytest.raises(WrdsSessionError):
        session.trading_days(2024)
    assert attempts == [1]


def test_session_constants_match_wrds_package_and_live_values():
    wrds_sql = pytest.importorskip("wrds.sql")
    from quantlab.acquisition.wrds_taq import WrdsSession

    assert WrdsSession.HOST == "wrds-pgdata.wharton.upenn.edu"
    assert WrdsSession.PORT == 9737
    assert WrdsSession.DBNAME == "wrds"
    assert WrdsSession.HOST == wrds_sql.WRDS_POSTGRES_HOST
    assert WrdsSession.PORT == int(wrds_sql.WRDS_POSTGRES_PORT)
    assert WrdsSession.DBNAME == wrds_sql.WRDS_POSTGRES_DB


# -- D-21: failure classification ----------------------------------------------------


def test_classify_session_and_entitlement_failures_as_global_stop(
    mock_wrds_session, acquisition_config
):
    from quantlab.acquisition.wrds_taq import (
        WrdsEntitlementError,
        WrdsSessionError,
        WrdsTaqNbboAcquisition,
    )

    acq = WrdsTaqNbboAcquisition(acquisition_config(vendor="wrds"))
    for exc in (
        WrdsSessionError("broken"),
        WrdsEntitlementError("taqm_2012"),
        psycopg2.OperationalError("x"),
        psycopg2.InterfaceError("x"),
    ):
        assert acq._classify_error(exc) == "quota", type(exc).__name__
    for exc in (ValueError("x"), RuntimeError("x")):
        assert acq._classify_error(exc) == "failed", type(exc).__name__


# -- D-07/D-20: structural ----------------------------------------------------------


def test_ast_provider_never_imports_wrds_or_names_connection():
    tree = ast.parse(WRDS_TAQ_SOURCE.read_text())
    offenders: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "wrds":
                    offenders.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "wrds":
                offenders.append(f"from {node.module} import ...")
            for alias in node.names:
                if alias.name in ("connect", "Connection"):
                    offenders.append(f"from {node.module} import {alias.name}")
        elif isinstance(node, ast.Name) and node.id == "Connection":
            offenders.append("name Connection")
        elif isinstance(node, ast.Attribute):
            if node.attr == "Connection":
                offenders.append("attribute .Connection")
            if node.attr == "connect" and not (
                isinstance(node.value, ast.Name) and node.value.id == "psycopg2"
            ):
                offenders.append("connect reached other than as psycopg2.connect")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in ("input", "getpass"):
                offenders.append(f"call {node.func.id}()")
    assert not offenders, offenders
    assert "psycopg2.connect(" in WRDS_TAQ_SOURCE.read_text()
