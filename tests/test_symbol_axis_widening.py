"""Direction 1: widening an existing store's symbol axis in place (260906-x2s).

A periodic US-equity refresh HALTS on the first new listing. `XrBackend.append`
-> `_assert_append_compatible` refuses a changed `symbol` coordinate, and it is
RIGHT to: raw `to_zarr(mode="a", append_dim=...)` silently overwrites the stored
labels, so a window carrying `{A, ARM}` written into a store holding `{A, XYZ}`
leaves XYZ's history attributed to ARM with nothing raised and nothing
recoverable afterwards (measured 2026-09-06).

This module pins the opt-in that gets PAST that refusal without weakening it.
`widen_and_append` reindexes BOTH the store and the incoming window onto
`sorted(stored | incoming)` and then calls the UNCHANGED `append()` -- so the
guard still runs and passes on its own terms, by construction rather than by
exemption. A caller who did not opt in still gets the refusal; that half is
pinned next door, in
`tests/test_chunked_ingest.py::test_plain_append_still_refuses_a_labels_differ_axis_of_the_same_length`.

Every test here names, in its docstring, the mutation that reddens it. This
project has ten recorded instances of a test passing for the wrong reason; a
test whose reddening mutation is unstated is a test nobody can check.

WHY EVERY TEST HERE RUNS TWICE, AND WHY THIS SUITE NEEDS ITS OWN LOCK
(260908-dvv). Like its two sibling suites, every test here built its `symbol`
coordinate from a python list literal, which round-trips through zarr to a
FIXED-WIDTH unicode store -- while the store the current chunked ingest writes
is `object`-encoded and decodes to `StringDType()`. So the coordinate now comes
from `conftest.symbol_coord` and every test takes the `symbol_encoding`
fixture, giving each one a `[fixed_width]` and a `[variable_length]` id.

But the defect that motivated all of this -- `widen_data_vars` rebuilding the
store's coordinates from the opened dataset and writing them back -- does NOT
reach `widen_symbol_axis`, which never builds the filler. Measured 2026-09-08
against `quantlab/dataset/backend.py` at `dea1e85`, this suite scores ZERO red
in either arm. That is CORRECT, not a shortfall, and it is recorded here so
nobody later "fixes" the zero by contriving a fixture to reach a defect that
genuinely is not on this path.

This suite's guarantee comes from a different observable instead:
`assert_stored_symbol_encoding(path, symbol_encoding)` after every widen that
is expected to SUCCEED. It is the only thing that can see a widen which raises
nothing, passes every value, dtype, NaN and chunk assertion, and silently
rewrites the store's coordinate encoding underneath. Its reddening mutation is
named in each carrying test's docstring.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from conftest import assert_stored_symbol_encoding, symbol_coord
from quantlab.dataset.backend import XrBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _panel(
    dates: list[str], symbols: list[str], offset: float, *, encoding: str
) -> xr.Dataset:
    """A float `close` panel with a DISTINCT value in every cell.

    Distinct on purpose: an element-for-element history assertion cannot tell
    a correct label-aligned reindex from a positional one if every cell holds
    the same number.

    `encoding` is keyword-only and REQUIRED (260908-dvv): the symbol
    coordinate comes from `conftest.symbol_coord` rather than from `symbols`
    directly, so this builder can produce either of the two encodings that are
    live on real stores. Required so a new test cannot forget it.
    """
    values = (
        np.arange(len(dates) * len(symbols), dtype=float).reshape(
            len(dates), len(symbols)
        )
        + offset
    )
    return xr.Dataset(
        {"close": (["timestamp", "symbol"], values)},
        coords={
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _typed_panel(
    dates: list[str], symbols: list[str], *, encoding: str
) -> xr.Dataset:
    """A panel carrying the two non-float shapes a real cleaned panel has:
    an int64 `volume` and a bool `anomaly_flag`.

    `BaseDataset._pin_append_dtypes` promotes integer variables to float64
    before an append but deliberately leaves `anomaly_flag` bool, so a real
    store has exactly one non-float variable. Both are exercised here because
    the guard is a dtype rule, not an `anomaly_flag` special case.

    Carries the same required keyword-only `encoding` as `_panel`, for the
    same reason.
    """
    shape = (len(dates), len(symbols))
    return xr.Dataset(
        {
            "volume": (
                ["timestamp", "symbol"],
                np.arange(shape[0] * shape[1], dtype="int64").reshape(shape),
            ),
            "anomaly_flag": (
                ["timestamp", "symbol"],
                np.ones(shape, dtype=bool),
            ),
        },
        coords={
            "timestamp": pd.to_datetime(dates),
            "symbol": symbol_coord(symbols, encoding),
        },
    )


def _stored(path: str) -> xr.Dataset:
    return xr.open_zarr(path).load()


# ---------------------------------------------------------------------------
# widen_and_append -- the history-preservation contract
# ---------------------------------------------------------------------------


def test_widening_preserves_history_for_pre_existing_symbols(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The todo's explicit verification requirement: not "the append
    succeeded" but "every pre-existing symbol's history is bit-identical
    afterwards", asserted element for element.

    RED under: reindexing only the incoming window and leaving the store
    alone (the append then refuses), or aligning by POSITION instead of by
    label (B's history lands under C).

    The encoding assertion is RED under M4 -- appending
    `widened = widened.assign_coords({dim: requested})` after the reindex in
    `widen_symbol_axis`. Measured 2026-09-08: that downgrades a
    variable-length store to a fixed-width one, and every other assertion in
    this function, in BOTH arms, stays green. Only `[variable_length]`
    reddens, which is what attributes the red to the encoding.
    """
    path = str(tmp_path / "widen.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    incoming = _panel(["2023-01-04"], ["A", "B", "C"], 500.0, encoding=symbol_encoding)
    XrBackend().to_internal(incoming).widen_and_append(path)

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after.sizes["timestamp"] == 3

    for symbol in ("A", "B"):
        np.testing.assert_array_equal(
            after["close"].sel(symbol=symbol).values[:2],
            before["close"].sel(symbol=symbol).values,
        )
    assert np.isnan(after["close"].sel(symbol="C").values[:2]).all()

    # The new row carries every symbol's REAL value, C included.
    np.testing.assert_array_equal(
        after["close"].isel(timestamp=-1).values,
        incoming["close"].isel(timestamp=-1).values,
    )


def test_widening_does_not_misattribute_when_the_symbol_count_is_unchanged(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """THE measured corruption case, and the reason `_assert_append_compatible`
    may not be relaxed: one delisting plus one new listing leaves the symbol
    COUNT unchanged while the labels differ. Raw
    `to_zarr(mode="a", append_dim=...)` succeeds silently here and produces the
    recorded outcome -- `rows 0-4 were written for XYZ but are now labelled:
    ARM`.

    The widen must union rather than substitute: `['A', 'ARM', 'XYZ']`, XYZ's
    stored values still under XYZ, ARM's historical cells NaN.

    RED under: substituting raw `to_zarr(mode="a", append_dim=...)` for the
    widen, or computing the target axis as the INCOMING labels rather than the
    union.

    The encoding assertion is RED under M4, on `[variable_length]` alone. This
    shape is the one where the symbol COUNT is unchanged, so it is the shape
    where a coordinate rewrite is least visible in any other observable.
    """
    path = str(tmp_path / "misattribute.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"],
            ["A", "XYZ"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    before = _stored(path)

    XrBackend().to_internal(
        _panel(["2023-01-04"], ["A", "ARM"], 900.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "ARM", "XYZ"]

    np.testing.assert_array_equal(
        after["close"].sel(symbol="XYZ").values[:2],
        before["close"].sel(symbol="XYZ").values,
    )
    np.testing.assert_array_equal(
        after["close"].sel(symbol="A").values[:2],
        before["close"].sel(symbol="A").values,
    )
    assert np.isnan(after["close"].sel(symbol="ARM").values[:2]).all()

    # The new row: A and ARM carry real values, the delisted XYZ carries NaN.
    assert not np.isnan(after["close"].sel(symbol="A").values[-1])
    assert not np.isnan(after["close"].sel(symbol="ARM").values[-1])
    assert np.isnan(after["close"].sel(symbol="XYZ").values[-1])


# ---------------------------------------------------------------------------
# widen_symbol_axis -- the guards that fire BEFORE any write
# ---------------------------------------------------------------------------


def test_widening_refuses_a_non_float_variable_without_an_explicit_fill_value(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """`reindex` with no `fill_value` upcasts bool -> float64 and int64 ->
    float64 and writes NaN (measured 2026-09-06). Doing that to a LIVE store is
    a silent schema change -- the same family of invisible corruption
    `_assert_append_compatible` refuses on the append path -- so refuse before
    touching anything.

    RED under: letting `reindex` upcast silently (no `ValueError`, and the
    store comes back with float64 `volume`/`anomaly_flag`).
    """
    path = str(tmp_path / "typed.zarr")
    XrBackend().to_internal(
        _typed_panel(["2022-01-04", "2022-06-15"], ["A", "B"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    message = str(excinfo.value)
    assert "volume" in message or "anomaly_flag" in message
    assert "int64" in message or "bool" in message
    assert "fill_values" in message

    # And nothing was touched: same axis, same values, same dtypes.
    after = _stored(path)
    xr.testing.assert_identical(after, before)
    assert after["volume"].dtype == np.dtype("int64")
    assert after["anomaly_flag"].dtype == np.dtype("bool")


def test_an_explicit_fill_value_preserves_the_stored_dtype(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The opt-in through the refusal above. A per-variable `fill_value` dict
    is measured to preserve `bool` and `int64` EXACTLY -- no `.astype()`
    restoration is needed, and none is done.

    RED under: dropping the per-variable dict and reindexing with a scalar
    NaN, which upcasts both variables to float64.

    The encoding assertion is RED under M4, on `[variable_length]` alone --
    and it sits alongside two DATA-variable dtype assertions on purpose: the
    coordinate's dtype is preserved by a different mechanism than the data
    variables', so a test that checked only `volume` and `anomaly_flag` would
    read as if it covered the coordinate too.
    """
    path = str(tmp_path / "filled.zarr")
    XrBackend().to_internal(
        _typed_panel(["2022-01-04", "2022-06-15"], ["A", "B"], encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    XrBackend().widen_symbol_axis(
        path,
        ["A", "B", "C"],
        fill_values={"anomaly_flag": False, "volume": 0},
    )

    after = _stored(path)
    assert_stored_symbol_encoding(path, symbol_encoding)
    assert after["symbol"].values.tolist() == ["A", "B", "C"]
    assert after["volume"].dtype == np.dtype("int64")
    assert after["anomaly_flag"].dtype == np.dtype("bool")
    assert (after["volume"].sel(symbol="C").values == 0).all()
    assert (~after["anomaly_flag"].sel(symbol="C").values).all()
    # Pre-existing history untouched, values and dtypes both.
    for name in ("volume", "anomaly_flag"):
        for symbol in ("A", "B"):
            np.testing.assert_array_equal(
                after[name].sel(symbol=symbol).values,
                before[name].sel(symbol=symbol).values,
            )


def test_widening_refuses_a_target_axis_that_would_drop_a_stored_symbol(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """A non-superset target silently DELETES a delisted symbol's entire
    history -- `reindex` drops what it is not asked for, and afterwards the
    store is indistinguishable from one that never held it.

    RED under: removing the superset check.
    """
    path = str(tmp_path / "superset.zarr")
    XrBackend().to_internal(
        _panel(
            ["2022-01-04", "2022-06-15"],
            ["A", "XYZ"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "ARM"])

    assert "XYZ" in str(excinfo.value)
    xr.testing.assert_identical(_stored(path), before)


# ---------------------------------------------------------------------------
# The rewrite itself: chunk grid, crash safety, and the cheap path
# ---------------------------------------------------------------------------


def test_the_chunk_grid_survives_a_widen(tmp_path: Path, symbol_encoding: str) -> None:
    """`APPEND_DIM_CHUNK` is what makes the store's layout a property of the
    STORE rather than of whichever window was written first. A rewrite that
    forgets `encoding=` re-pins the append-dim chunk to the whole accumulated
    history, so every subsequent append is misaligned with the grid.

    600 timestamps, deliberately: with fewer than `APPEND_DIM_CHUNK` rows the
    encoded and unencoded append-dim chunk sizes coincide.

    RED under: dropping `encoding=` from the rewrite. Measured 2026-09-07 --
    the rewritten store then inherits the SOURCE store's encoding and comes
    back with chunks `(512, 2)` for a three-symbol array, i.e. a symbol chunk
    still pinned to the pre-widen count. Also RED under restating the chunk
    arithmetic instead of routing through `_append_encoding`.

    The encoding assertion is RED under M4, on `[variable_length]` alone. Note
    that the chunk assertion beside it does NOT move under M4: the chunk grid
    and the coordinate encoding are independent properties of the same
    rewrite, and this is the test that says so.
    """
    path = str(tmp_path / "grid.zarr")
    dates = pd.date_range("2020-01-01", periods=600, freq="D")
    XrBackend().to_internal(
        _panel(
            [d.isoformat() for d in dates],
            ["A", "B"],
            0.0,
            encoding=symbol_encoding,
        )
    ).append(path)
    assert zarr.open_group(path, mode="r")["close"].chunks == (
        XrBackend.APPEND_DIM_CHUNK,
        2,
    )

    XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    assert_stored_symbol_encoding(path, symbol_encoding)
    chunks = zarr.open_group(path, mode="r")["close"].chunks
    assert chunks == (min(XrBackend.APPEND_DIM_CHUNK, 600), 3)


def test_a_failed_widen_leaves_the_original_store_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """The widened panel is written to a SIBLING sidecar and only then swapped
    in. A rewrite that writes over `path` with `mode="w"` has destroyed the
    store by the time it discovers it cannot finish.

    RED under: writing the widened panel over `path` with `mode="w"` (the
    store is then gone or truncated).
    """
    path = str(tmp_path / "crash.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    before = _stored(path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated crash inside the widen rewrite")

    monkeypatch.setattr(xr.Dataset, "to_zarr", _boom)

    with pytest.raises(RuntimeError):
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    monkeypatch.undo()
    xr.testing.assert_identical(_stored(path), before)
    # And no orphan sidecar is left claiming to be a store.
    assert not Path(f"{path}.widening.tmp").exists()
    assert not Path(f"{path}.superseded.tmp").exists()


def test_a_crash_between_the_two_renames_refuses_and_names_the_recovery(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """The one state the rename-aside ordering can leave behind: a
    `.superseded.tmp` holding the real store and NOTHING at `path`. The next
    read must fail loudly and name the manual move, never auto-recover -- which
    of the two directories is authoritative is not this method's call to make.

    RED under: auto-recovering from the sidecar, or ignoring it and rebuilding
    from an absent store.
    """
    path = str(tmp_path / "interrupted.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)
    superseded = f"{path}.superseded.tmp"
    Path(path).rename(superseded)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().widen_symbol_axis(path, ["A", "B", "C"])

    message = str(excinfo.value)
    assert superseded in message
    assert path in message
    # The real store is still there, under the sidecar name.
    assert Path(superseded).exists()


def test_widen_and_append_with_an_unchanged_axis_takes_the_plain_append_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symbol_encoding: str
) -> None:
    """Calling `widen_and_append` unconditionally must stay cheap: with no
    roster change there is nothing to widen, and rewriting the whole store to
    discover that would make the opt-in unusable on a routine refresh.

    RED under: rewriting the store unconditionally.
    """
    path = str(tmp_path / "cheap.zarr")
    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).append(path)

    def _fail(*args, **kwargs):
        raise AssertionError("widen_symbol_axis must not run for an equal axis")

    monkeypatch.setattr(XrBackend, "widen_symbol_axis", _fail)

    XrBackend().to_internal(
        _panel(["2023-01-04"], ["A", "B"], 100.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    store = _stored(path)
    assert store.sizes["timestamp"] == 3
    assert store["symbol"].values.tolist() == ["A", "B"]


def test_widen_and_append_creates_the_store_when_there_is_none(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """One creation path, not two: with no store there is nothing to widen, so
    the call delegates straight to `append()`.

    RED under: giving `widen_and_append` its own creating write, which would
    bypass `_append_encoding` and pin the chunk grid differently from every
    store `append()` creates.
    """
    path = str(tmp_path / "fresh.zarr")

    XrBackend().to_internal(
        _panel(["2022-01-04", "2022-06-15"], ["A", "B"], 0.0, encoding=symbol_encoding)
    ).widen_and_append(path)

    assert _stored(path)["symbol"].values.tolist() == ["A", "B"]
    assert zarr.open_group(path, mode="r")["close"].chunks == (2, 2)


def test_widen_and_append_inherits_the_overlap_refusal_verbatim(
    tmp_path: Path, symbol_encoding: str
) -> None:
    """D-03 (decided 2026-09-07): `widen_and_append` gets NO separate handling
    of the append-dim overlap refusal. It closes by calling the UNCHANGED
    `append()`, which its docstring already declares load-bearing precisely so
    guards keep applying on the widened path -- so the new overlap guard
    covers it for free.

    The proof is MESSAGE EQUALITY, not merely that both paths raise. Two
    stores start identically; one takes a plain `append()` of an overlapping
    window on the existing roster, the other takes a `widen_and_append()` of
    the same overlapping window carrying an ADDED symbol C so the widen
    genuinely runs. With each store's own path normalised out, the two
    messages must be byte-identical. A second, separately-worded check inside
    `widen_and_append` cannot produce that equality -- which is exactly what
    makes this an enforcement of D-03 rather than a restatement of it.

    Also asserted: the ACCEPTED side effect, now recorded in
    `quantlab/dataset/backend.py::XrBackend.widen_and_append`'s docstring. The
    widen COMMITS before the closing `append()` raises, so the store's symbol
    axis MAY have grown to A, B, C while its timestamp axis is still the
    original three labels -- unique, monotonic, and with A's and B's stored
    values on those timestamps unchanged. That is the guarantee worth holding;
    asserting instead that the symbol axis stayed narrow would be asserting
    that `widen_and_append` does NOT delegate, which is the opposite of D-03.

    RED under: adding a second, differently-worded overlap check at the top of
    `widen_and_append` so it raises before delegating (this module's own
    mutation M4, distinct from 260908-dvv's `assign_coords` M4 below).

    The encoding assertion is RED under 260908-dvv's M4, on
    `[variable_length]` alone. It is asserted on the store AFTER the closing
    `append()` raised, which is the interesting moment: the widen COMMITTED,
    so a coordinate rewrite performed by that committed widen is already on
    disk even though the caller saw an exception.
    """
    plain_path = str(tmp_path / "plain.zarr")
    widened_path = str(tmp_path / "widened.zarr")
    dates = ["2022-01-04", "2022-01-05", "2022-01-06"]
    for path in (plain_path, widened_path):
        XrBackend().to_internal(
            _panel(dates, ["A", "B"], 0.0, encoding=symbol_encoding)
        ).append(path)
    before = _stored(widened_path)["close"].values.copy()

    overlapping = ["2022-01-05", "2022-01-06", "2022-01-07"]
    with pytest.raises(ValueError) as plain_error:
        XrBackend().to_internal(
            _panel(overlapping, ["A", "B"], 100.0, encoding=symbol_encoding)
        ).append(plain_path)

    with pytest.raises(ValueError) as widened_error:
        XrBackend().to_internal(
            _panel(overlapping, ["A", "B", "C"], 100.0, encoding=symbol_encoding)
        ).widen_and_append(widened_path)

    placeholder = "<STORE>"
    assert str(plain_error.value).replace(plain_path, placeholder) == str(
        widened_error.value
    ).replace(widened_path, placeholder)

    # The accepted, documented side effect: the widen committed, the window
    # did not. The symbol axis may have grown; the time axis and every
    # pre-existing value on it are untouched.
    assert_stored_symbol_encoding(widened_path, symbol_encoding)
    store = _stored(widened_path)
    index = pd.DatetimeIndex(store["timestamp"].values)
    assert index.tolist() == pd.to_datetime(dates).tolist()
    assert index.is_unique
    assert index.is_monotonic_increasing
    assert store["close"].sel(symbol=["A", "B"]).values.tolist() == (
        before.tolist()
    )
