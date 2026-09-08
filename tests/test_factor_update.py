"""`Factor.update()` -- the AUTOMATIC incremental interface (260907-vyr).

`Factor` had exactly one write path, `save()`, and it is a WHOLESALE one:
zarr's `mode="a"` means "overwrite variables in an existing store", not
"append along time", so a second date range fails outright
(`tests/test_factor_save_mode.py` pins that failure and its message). The
incremental path existed one layer down -- `XrBackend.append` and, since
260906-x2s and this task, `widen_and_append` -- with no route to it from the
factor layer at all. Measured 2026-09-07: `widen_and_append` had NO production
caller anywhere in the repo, only tests.

`update()` is that route, and it is AUTOMATIC: it works out for itself what
changed -- later dates, new symbols, new variables -- and reconciles each axis
without the caller naming which widening to perform. The two interfaces are how
a caller says which it means:

    save()    writes WHOLESALE     (save(mode="w") replaces the store)
    update()  EXTENDS              (no mode, no route to overwrite a stored range)

Every test here names, in its docstring, the mutation that reddens it.
"""

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import BaseFactorConfig
from quantlab.base.factor import Factor

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Only what `Factor`'s config setter touches.

    `_reset_dataset_config()` writes `start_date`/`end_date` onto
    `config.dataset.config`; nothing else on the dataset is reached by the
    `update()` path. Copied in shape from `tests/test_factor_save_mode.py`.
    """

    def __init__(self):
        self.config = SimpleNamespace(start_date=None, end_date=None)


class PanelFactor(Factor):
    """The smallest concrete `Factor`: a `(timestamp, symbol)` panel.

    `cal()` takes the dates, symbols and variables explicitly so one class can
    express every axis the update path reconciles. No KunQuant graph, no
    compiled code, no real dataset.
    """

    SYMBOLS = ["AAA", "BBB"]

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("alpha",)

    def cal(
        self,
        dates: Sequence[str],
        symbols: Sequence[str] | None = None,
        variables: Mapping[str, str] | Sequence[str] = ("alpha",),
        offset: float = 0.0,
    ):
        symbols = list(self.SYMBOLS if symbols is None else symbols)
        if not isinstance(variables, Mapping):
            variables = {str(name): "float64" for name in variables}

        shape = (len(dates), len(symbols))
        data = {}
        for index, (name, dtype) in enumerate(variables.items()):
            if np.issubdtype(np.dtype(dtype), np.bool_):
                values = np.ones(shape, dtype=bool)
            else:
                values = (
                    np.arange(shape[0] * shape[1], dtype="float64").reshape(shape)
                    + offset
                    + index * 1000.0
                ).astype(dtype)
            data[name] = (["timestamp", "symbol"], values)

        self.data_backend.to_internal(
            xr.Dataset(
                data,
                coords={
                    "timestamp": pd.to_datetime(list(dates)),
                    "symbol": symbols,
                },
            )
        )
        return self


class FilledPanelFactor(PanelFactor):
    """A subclass that OVERRIDES the fill seam, the way a classification-label
    factor carrying a boolean variable would have to."""

    def _widen_fill_values(self) -> dict:
        return {"anomaly_flag": False}


def _factor(tmp_path: Path, cls=PanelFactor) -> PanelFactor:
    return cls(
        BaseFactorConfig(
            window=1,
            dataset=_FakeDataset(),  # type: ignore[arg-type]
            file_path=str(tmp_path / "alpha.zarr"),
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )


def _stored(factor: Factor) -> xr.Dataset:
    return xr.open_zarr(factor.config.file_path).load()


EARLY = ["2024-02-01", "2024-02-02", "2024-02-03"]
LATER = ["2024-03-01", "2024-03-02"]


# ---------------------------------------------------------------------------
# Task 3 -- the automatic update interface
# ---------------------------------------------------------------------------


def test_update_extends_the_store_with_a_later_date_range(
    tmp_path: Path,
) -> None:
    """The tracer: the thing `save()` measurably cannot do.

    "Compute February, then compute March" is the call that raises under
    `save()` -- `tests/test_factor_save_mode.py` pins that -- and must simply
    work here. Both ranges present, the axis unique and strictly increasing,
    and February's values bit-identical afterwards.

    RED under: routing `update()` through `data_backend.write` (it would raise
    zarr's dimension-size error), or through anything that does not extend.
    """
    factor = _factor(tmp_path)
    factor.cal(EARLY).update()
    before = _stored(factor)

    factor.cal(LATER, offset=100.0).update()

    after = _stored(factor)
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    index = after["timestamp"].to_index()
    assert index.is_unique and index.is_monotonic_increasing
    np.testing.assert_array_equal(
        after["alpha"].values[: len(EARLY)], before["alpha"].values
    )


def test_update_reconciles_a_new_symbol_without_being_told_to(
    tmp_path: Path,
) -> None:
    """AUTOMATIC is the whole point: the caller names no widening.

    A roster that grew between two runs is the ordinary case, not an
    exceptional one, and a factor-layer caller has no business knowing that
    the storage layer calls this "widening the symbol axis". It passes a
    panel; `update()` works out what changed.

    RED under: M11 (route `update()` through `append()` instead of
    `widen_and_append()`) -- the symbol-coordinate guard refuses.
    """
    factor = _factor(tmp_path)
    factor.cal(EARLY).update()

    factor.cal(LATER, symbols=["AAA", "BBB", "CCC"], offset=100.0).update()

    after = _stored(factor)
    assert after["symbol"].values.tolist() == ["AAA", "BBB", "CCC"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    # The new symbol carries NaN across the historical block and real values
    # on the new one -- a widen backfills, it does not invent history.
    assert np.isnan(after["alpha"].sel(symbol="CCC").values[: len(EARLY)]).all()
    assert not np.isnan(
        after["alpha"].sel(symbol="CCC").values[len(EARLY) :]
    ).any()


def test_update_reconciles_a_new_variable_without_being_told_to(
    tmp_path: Path,
) -> None:
    """The third axis, reconciled just as automatically as the second.

    The SYMBOL axis AGREES on both sides -- the same roster before and after,
    with only the variable set growing. That is load-bearing rather than
    incidental: it is what routes this panel into `widen_and_append`'s fast
    path and therefore what makes M9 observable here at all. A panel that grew
    BOTH axes would be reconciled by the symbol widen regardless and would
    stay green under M9, proving nothing about the variable axis.

    This is also the property that proves `update()` routes through the
    three-axis path rather than a plain append.

    RED under: M11 (via the Task 1 new-variable refusal) and M9 (leaving
    `widen_and_append`'s short-circuit testing the symbol axis alone -- the
    fast path is taken and the closing `append()` refuses `beta`).
    """
    factor = _factor(tmp_path)
    factor.cal(EARLY).update()

    factor.cal(LATER, variables=["alpha", "beta"], offset=100.0).update()

    after = _stored(factor)
    assert sorted(after.data_vars) == ["alpha", "beta"]
    assert after["symbol"].values.tolist() == ["AAA", "BBB"]
    assert after.sizes["timestamp"] == len(EARLY) + len(LATER)
    # The new variable is backfilled across the historical block only.
    assert np.isnan(after["beta"].values[: len(EARLY)]).all()
    assert not np.isnan(after["beta"].values[len(EARLY) :]).any()


def test_update_declares_no_overwrite_parameter(tmp_path: Path) -> None:
    """The DECLARED half of D-10 -- and only that half. Read it with the next
    test; neither is sufficient alone.

    D-10: `update()` EXTENDS, and offers no route to overwrite a range the
    store already holds. Recomputing is `save(mode="w")`'s job. A `mode=` on
    `update()` would blur exactly the boundary the two interfaces exist to
    keep sharp.

    **This assertion is a structural PROXY that measurably does NOT span the
    property it stands for.** Quick task 260907-uac demonstrated live that a
    hatch popped from `**kwargs` INSIDE a method body makes the guard's
    `ValueError` disappear while `inspect.signature` still returns a
    byte-identical parameter tuple. So on its own this test stays green
    through precisely the drift D-10 exists to catch. The behavioural half is
    `test_update_refuses_a_range_the_store_already_holds` directly below.

    RED under: M10 (give `Factor.update` a `mode` parameter), under which the
    behavioural half below stays GREEN -- a declared-but-unpassed `mode` opens
    no route past a guard that runs before `kwargs` is read.
    """
    parameters = tuple(inspect.signature(Factor.update).parameters)
    assert parameters == ("self", "kwargs")


def test_update_refuses_a_range_the_store_already_holds(tmp_path: Path) -> None:
    """The BEHAVIOURAL half of D-10, and the half that spans the realistic
    drift. Read it with `test_update_declares_no_overwrite_parameter` above,
    whose signature assertion measurably does not cover this.

    `update()` inherits the unconditional append-dim overlap refusal shipped
    by 260907-uac, through `widen_and_append` -> the UNCHANGED `append()`. It
    still refuses when the same call carries an unrecognised keyword, because
    the guard runs before `kwargs` is touched anywhere along that chain.

    Tests 4 and 5 are two functions rather than two assertions in one, for the
    measured reason the repo's own pair at `tests/test_chunked_ingest.py:603`
    and `:633` is two: a single pytest function cannot be half-red, so the
    split M10 predicts -- structural red, behavioural green -- would be
    unobservable inside one function.

    RED under: any hatch consumed from `**kwargs` ahead of the guard, or
    giving `update()` a working overwrite route.
    """
    factor = _factor(tmp_path)
    factor.cal(EARLY).update()
    before = _stored(factor)["alpha"].values.copy()

    overlapping = ["2024-02-03", "2024-02-04"]

    with pytest.raises(ValueError) as plain_error:
        factor.cal(overlapping, offset=100.0).update()
    assert "already ends at" in str(plain_error.value), str(plain_error.value)

    with pytest.raises(ValueError) as smuggled_error:
        factor.cal(overlapping, offset=100.0).update(force=True)
    assert "already ends at" in str(smuggled_error.value), str(
        smuggled_error.value
    )

    after = _stored(factor)
    assert after.sizes["timestamp"] == len(EARLY)
    np.testing.assert_array_equal(after["alpha"].values, before)


def test_the_widen_fill_seam_reaches_the_widening_call(tmp_path: Path) -> None:
    """DVAR-11: `_widen_fill_values()` is a real seam, asserted BEHAVIOURALLY.

    The default is `{}` because every factor and label panel is float today --
    KunQuant emits float arrays, and even `SpotBinaryReturn` builds its binary
    label from `op.ConstantOp(1.0)`/`(0.0)`, so it is float64 rather than
    bool. The seam is required regardless of that default: the widening
    refuses to backfill a non-float variable without an explicit fill, so
    without this hook a future non-float subclass has no way to widen AT ALL.

    Asserting only that the method exists and returns `{}` would not span
    that. So the span is a pair of runs on the same shape: the base subclass
    is REFUSED on a boolean variable, and the subclass overriding the seam
    succeeds and keeps the dtype. That is the mapping demonstrably arriving at
    the widening call.

    RED under: dropping the seam, hardcoding `{}` at the call site, or not
    passing `fill_values` through to `widen_and_append`.
    """
    assert Factor._widen_fill_values(  # type: ignore[misc]
        object.__new__(PanelFactor)
    ) == {}

    plain = _factor(tmp_path / "plain")
    plain.cal(EARLY).update()
    with pytest.raises(ValueError) as excinfo:
        plain.cal(
            LATER, variables={"alpha": "float64", "anomaly_flag": "bool"}
        ).update()
    assert "anomaly_flag" in str(excinfo.value), str(excinfo.value)
    assert "fill_values" in str(excinfo.value), str(excinfo.value)

    filled = _factor(tmp_path / "filled", cls=FilledPanelFactor)
    filled.cal(EARLY).update()
    filled.cal(
        LATER, variables={"alpha": "float64", "anomaly_flag": "bool"}
    ).update()

    after = _stored(filled)
    assert after["anomaly_flag"].dtype == np.dtype(bool)
    # False across the historical block: it was not flagged because there was
    # nothing to flag. True on the new window, where the panel says so.
    assert not after["anomaly_flag"].values[: len(EARLY)].any()
    assert after["anomaly_flag"].values[len(EARLY) :].all()


def test_save_is_unchanged_by_the_arrival_of_update(tmp_path: Path) -> None:
    """DVAR-12: a no-change lock, not a new behaviour.

    `save()` keeps its `mode="a"` default and keeps raising its wrapped,
    actionable `ValueError` on a differently-sized second range. Only its
    PROSE moved -- the two clauses asserting this method has no incremental
    route became false the moment `update()` landed and were re-pointed, not
    deleted.

    Whether that `"a"` default is now vestigial is a real question and is
    deliberately NOT answered here; it is filed for the developer at
    `.planning/todos/pending/`.

    RED under: changing `save()`'s default, its wrapping, or the substring it
    matches on.
    """
    assert inspect.signature(Factor.save).parameters["mode"].default == "a"

    factor = _factor(tmp_path)
    factor.cal(EARLY).save()

    with pytest.raises(ValueError) as excinfo:
        factor.cal(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-04"]).save()

    message = str(excinfo.value)
    assert 'save(mode="w")' in message, message
    assert str(factor.config.file_path) in message, message
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "already exists with different dimension sizes" in str(
        excinfo.value.__cause__
    )
