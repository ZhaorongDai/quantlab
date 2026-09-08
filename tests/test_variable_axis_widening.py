"""Direction 3: the `data_vars` set as a reconciled axis on the append path
(260907-vyr).

Three axes cross `XrBackend.append`. The append dimension is refused on
overlap (260907-uac); the symbol axis has the `widen_symbol_axis` opt-in
(260906-x2s); the data-variable set had NEITHER a rule nor a guard, and is the
most destructive of the three.

`_assert_append_compatible`'s dtype loop opens with a `continue` on any name
not already in the store, so a NEW variable is never examined; and a variable
the store HAS but the incoming panel LACKS is never visited at all, because the
loop iterates the INCOMING panel's variables. Measured 2026-09-07, all three
shapes, `append()` raising nothing in any of them:

    store {alpha}        <- incoming {alpha, beta}  ->  alpha(4,2) beta(2,2)
    store {alpha, beta}  <- incoming {alpha}        ->  alpha(4,2) beta(2,2)
    store {alpha}        <- incoming {beta}         ->  both (2,2), ts=4

and afterwards `xr.open_zarr()` raises `ValueError: conflicting sizes for
dimension 'timestamp'`. The store is not merely wrong, it is UNOPENABLE -- and
in the second shape a store that was VALID before the call is destroyed.

This module pins the refusal (Task 1), the `widen_data_vars` opt-in that gets
past it without weakening it (Task 2), and the three-axis composition inside
`widen_and_append`.

Every test here names, in its docstring, the mutation that reddens it. This
project has recorded instances of a test passing for the wrong reason; a test
whose reddening mutation is unstated is a test nobody can check.
"""

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from quantlab.dataset.backend import XrBackend

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SYMBOLS = ["A", "B"]
EARLY = ["2022-01-04", "2022-01-05"]
LATER = ["2022-01-06", "2022-01-07"]


def _panel(
    dates: Sequence[str],
    symbols: Sequence[str],
    variables: Mapping[str, str] | Sequence[str],
    offset: float = 0.0,
) -> xr.Dataset:
    """A `(timestamp, symbol)` panel carrying the named variables.

    `variables` is either a plain sequence of names (all float64) or a mapping
    of name to dtype string -- the dtype half is what makes the precedence and
    filler-dtype fixtures expressible.

    Every cell holds a DISTINCT value on purpose: an element-for-element
    history assertion cannot tell an intact store from a rewritten one if
    every cell holds the same number.
    """
    if not isinstance(variables, Mapping):
        variables = {str(name): "float64" for name in variables}

    shape = (len(dates), len(symbols))
    size = shape[0] * shape[1]
    data = {}
    for index, (name, dtype) in enumerate(variables.items()):
        values = (
            np.arange(size, dtype="float64").reshape(shape)
            + offset
            + index * 1000.0
        )
        data[name] = (["timestamp", "symbol"], values.astype(dtype))
    return xr.Dataset(
        data,
        coords={"timestamp": pd.to_datetime(list(dates)), "symbol": list(symbols)},
    )


def _typed_panel(
    dates: Sequence[str], symbols: Sequence[str], name: str, dtype: str
) -> xr.Dataset:
    """A single-variable panel at an arbitrary dtype, including non-float.

    Separate from `_panel` because `astype(bool)` on an arange is not a
    meaningful bool panel; this one builds each dtype honestly.
    """
    shape = (len(dates), len(symbols))
    if np.issubdtype(np.dtype(dtype), np.bool_):
        values = np.ones(shape, dtype=bool)
    else:
        values = np.arange(shape[0] * shape[1]).reshape(shape).astype(dtype)
    return xr.Dataset(
        {name: (["timestamp", "symbol"], values)},
        coords={"timestamp": pd.to_datetime(list(dates)), "symbol": list(symbols)},
    )


def _stored(path: str) -> xr.Dataset:
    """Open and materialise the store.

    Load-bearing in every refusal test: a store the append corrupted cannot be
    opened AT ALL, so a successful `open_zarr` is itself the proof that the
    refusal fired ahead of `to_zarr` -- proof no message can fake.
    """
    return xr.open_zarr(path).load()


# ---------------------------------------------------------------------------
# Task 1 -- the refusal
# ---------------------------------------------------------------------------


def test_append_refuses_an_incoming_panel_carrying_an_unstored_variable(
    tmp_path: Path,
) -> None:
    """The tracer, end to end: store `{alpha}`, incoming `{alpha, beta}`.

    The store being OPENABLE afterwards, and still holding exactly `{alpha}`
    at its original extent, is the assertion that matters. Measured on the
    unguarded tree this same call wrote `alpha`(4,2) and `beta`(2,2) and left
    the store unopenable, so "still openable" is not a formality -- it is the
    only observation that distinguishes a refusal fired BEFORE `to_zarr` from
    one fired after.

    RED under: mutation M1 (delete the whole variable-set check) via
    `DID NOT RAISE ValueError`.
    """
    path = str(tmp_path / "new_var.zarr")
    XrBackend().to_internal(_panel(EARLY, SYMBOLS, ["alpha"])).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha", "beta"], offset=100.0)
        ).append(path)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_append_refuses_an_incoming_panel_missing_a_stored_variable(
    tmp_path: Path,
) -> None:
    """The DESTRUCTIVE shape: store `{alpha, beta}`, incoming `{alpha}`.

    This is the only one of the three that destroys data which was VALID
    before the call -- `alpha` grows to (4,2) while `beta` stays STUCK at
    (2,2), and a previously readable two-variable store becomes unopenable.
    That is why this direction is refused UNCONDITIONALLY rather than offered
    a widening path: filling the absent variable with NaN over the incoming
    window would punch holes into recent dates of a variable that was
    complete, and afterwards the store would be indistinguishable from one
    where those values were genuinely missing.

    RED under: M1 (via `DID NOT RAISE`) and M2 (checking only the
    new-variable direction, also `DID NOT RAISE`).
    """
    path = str(tmp_path / "missing_var.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"])
    ).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha"], offset=100.0)
        ).append(path)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after.sizes["timestamp"] == len(EARLY)
    assert after["beta"].shape == before["beta"].shape
    np.testing.assert_array_equal(after["beta"].values, before["beta"].values)


def test_append_refuses_a_fully_disjoint_variable_set(tmp_path: Path) -> None:
    """Store `{alpha}`, incoming `{beta}` -- both a drop AND an addition.

    Deliberately MESSAGE-AGNOSTIC: it asserts only that the append raises and
    that the store survives. Pinning a message here would make mutation M3
    (swapping the two branches' precedence) redden two tests instead of one
    and blur which branch caught it. Its value under M2 is the reverse -- it
    stays GREEN, which is the evidence that the disjoint case is being caught
    by the NEW-variable branch rather than only by the missing one.

    RED under: M1 only.
    """
    path = str(tmp_path / "disjoint.zarr")
    XrBackend().to_internal(_panel(EARLY, SYMBOLS, ["alpha"])).append(path)
    before = _stored(path)

    with pytest.raises(ValueError):
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["beta"], offset=100.0)
        ).append(path)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )


def test_the_two_variable_set_refusals_carry_distinct_messages(
    tmp_path: Path,
) -> None:
    """The two directions are not one refusal with a shared sentence.

    A caller told about the widening opt-in while being refused for DROPPING a
    stored variable would be actively misled -- there is no opt-in for that
    direction, and offering one would be a lie. So the missing-variable
    message names NO opt-in, and the new-variable message names
    `widen_data_vars`.

    One PURE fixture per message, never the mixed one, so a precedence swap
    (M3) cannot move this test -- that is test 5's job, and a test that moves
    under two different mutations tells you less than two tests that each move
    under one.

    RED under: M1 and M2 (the missing half stops raising), and under any
    change that collapses the two messages into one.
    """
    new_path = str(tmp_path / "pure_new.zarr")
    XrBackend().to_internal(_panel(EARLY, SYMBOLS, ["alpha"])).append(new_path)
    with pytest.raises(ValueError) as new_error:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha", "beta"], offset=100.0)
        ).append(new_path)

    missing_path = str(tmp_path / "pure_missing.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"])
    ).append(missing_path)
    with pytest.raises(ValueError) as missing_error:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha"], offset=100.0)
        ).append(missing_path)

    new_message = str(new_error.value)
    missing_message = str(missing_error.value)

    assert new_message != missing_message

    # Each names the store it is refusing and the offending variable.
    assert new_path in new_message, new_message
    assert "beta" in new_message, new_message
    assert missing_path in missing_message, missing_message
    assert "beta" in missing_message, missing_message

    # Only the new-variable direction has an opt-in, so only it names one.
    assert "widen_data_vars" in new_message, new_message
    assert "widen_data_vars" not in missing_message, missing_message


def test_a_mismatch_in_both_directions_raises_the_missing_variable_message(
    tmp_path: Path,
) -> None:
    """Precedence: store `{alpha, beta}` taking `{alpha, gamma}` is BOTH a
    drop and an addition, and the MISSING branch must win.

    The missing direction is the one that destroys valid data, and it has no
    remedy short of recomputing the full variable set. A caller shown the
    widening message first would widen `gamma` in, retry, and be refused all
    over again on `beta`.

    RED under: M3 (swap the precedence -- check NEW before MISSING), on this
    message assertion: it receives the widening message where the
    unconditional one is required.
    """
    path = str(tmp_path / "both_directions.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"])
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha", "gamma"], offset=100.0)
        ).append(path)

    message = str(excinfo.value)
    assert "that the incoming panel does not" in message, message
    assert "beta" in message, message
    assert "widen_data_vars" not in message, message


def test_an_identical_variable_set_still_appends(tmp_path: Path) -> None:
    """The positive control: the new check adds NO refusal to the path every
    existing caller already uses.

    Measured at planning time, the strictest possible variable-set refusal
    injected into `_assert_append_compatible` left the whole suite at 545
    passed -- no currently-green test appends a mismatched variable set at
    all. This test is what stops that fact from silently becoming "the check
    refuses everything and nothing noticed".

    GREEN under every mutation in the table; RED under a check that refuses a
    matching set.
    """
    path = str(tmp_path / "identical.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, ["alpha", "beta"])
    ).append(path)
    XrBackend().to_internal(
        _panel(LATER, SYMBOLS, ["alpha", "beta"], offset=100.0)
    ).append(path)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    assert after["timestamp"].to_index().is_monotonic_increasing
    assert after["timestamp"].to_index().is_unique


def test_a_shared_variable_dtype_mismatch_outranks_the_variable_set_check(
    tmp_path: Path,
) -> None:
    """D-03: the variable-set check sits AFTER the existing shared-variable
    dtype loop, so no pre-existing refusal's precedence changes.

    This COMBINED fixture is the ONLY shape that spans that decision, and that
    is measured rather than argued. With the same symbol axis and a strictly
    later window, store `{alpha: float64, beta}` taking
    `{alpha: float32, gamma}` carries a dtype mismatch on a SHARED variable
    AND a variable-set mismatch at once. Measured in both placements on
    2026-09-07: after the dtype loop it raises the DTYPE message; moved above
    it, the very same fixture raises the missing-variable message instead.

    An identical-variable-set fixture was also measured, and raises the dtype
    message in BOTH placements -- the relocated check has nothing to fire on,
    so it proves nothing about precedence. That is why it is not the fixture
    used here.

    Unlike every other test in this block, this one is GREEN before the
    implementation lands and stays green after. It is a precedence lock, not
    a RED-first test; its entire value is what M4 does to it.

    RED under: M4 (move the variable-set check above the dtype loop).
    """
    path = str(tmp_path / "combined.zarr")
    XrBackend().to_internal(
        _panel(EARLY, SYMBOLS, {"alpha": "float64", "beta": "float64"})
    ).append(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(
                LATER,
                SYMBOLS,
                {"alpha": "float32", "gamma": "float64"},
                offset=100.0,
            )
        ).append(path)

    message = str(excinfo.value)
    assert "has dtype" in message, message
    assert "alpha" in message, message
    assert "float32" in message and "float64" in message, message
    # Neither variable-set message may appear: the dtype guard got there first.
    assert "that the incoming panel does not" not in message, message
    assert "widen_data_vars" not in message, message


def test_a_variable_mismatched_append_carrying_an_unrecognised_kwarg_raises(
    tmp_path: Path,
) -> None:
    """The BEHAVIOURAL half of "there is no opt-out, declared OR smuggled".

    Its STRUCTURAL half is NOT rewritten here: it already exists as
    `tests/test_chunked_ingest.py::test_append_offers_no_overwrite_escape_hatch`
    (`:603`), which asserts `XrBackend.append`'s exact parameter tuple. A
    verbatim copy would give two tests reddening together on the same drift,
    which is noise rather than a split -- and that structural assertion was
    MEASURED (260907-uac) not to span this property at all: a hatch popped
    from `**kwargs` inside the body leaves the signature byte-identical.

    What this adds over its neighbour at `:633`: that one locks the
    append-dim OVERLAP refusal against a smuggled kwarg, this one locks the
    VARIABLE-SET refusal. A bypass added to one branch alone would leave the
    other green, so both are needed.

    The mechanism that makes it pass: `append()` calls
    `_assert_append_compatible` BEFORE it touches `kwargs` at all. Measured on
    the unguarded tree this same call raised `TypeError: Dataset.to_zarr() got
    an unexpected keyword argument` from deep inside the write.

    RED under: M5 (a hatch consumed from `**kwargs` ahead of the guard call),
    under which the structural test at `:603` stays GREEN.
    """
    path = str(tmp_path / "smuggled_var.zarr")
    XrBackend().to_internal(_panel(EARLY, SYMBOLS, ["alpha"])).append(path)
    before = _stored(path)

    with pytest.raises(ValueError) as excinfo:
        XrBackend().to_internal(
            _panel(LATER, SYMBOLS, ["alpha", "beta"], offset=100.0)
        ).append(path, force=True)

    assert "beta" in str(excinfo.value), str(excinfo.value)

    after = _stored(path)
    assert sorted(after.data_vars) == ["alpha"]
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(
        after["alpha"].values, before["alpha"].values
    )
