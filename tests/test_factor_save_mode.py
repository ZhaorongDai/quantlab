"""`Factor.save(mode="a")`'s failure must name its own fix (defect J).

Quick task 260907-fl6, batch 2. `base/factor.py:Factor.save` defaults to
`mode="a"`, which reads like "append", and is passed straight through to
`to_zarr`. zarr's `"a"` means "overwrite variables in an existing store", NOT
"append along time", so writing a second, differently-sized date range fails
with a message about internal dimension sizes:

    ValueError: variable 'timestamp' already exists with different dimension
    sizes: {'timestamp': 29} != {'timestamp': 31}. to_zarr() only supports
    changing dimension sizes when explicitly appending, but append_dim=None ...

**The default is deliberately NOT changed** -- that would be a behaviour change
for any caller relying on it. The harm here is an incomprehensible error two
layers below the `save()` the caller wrote, not a wrong default. So the fix is
to make the failure actionable: name `mode="w"`, name the store, and keep the
original exception on `__cause__`.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from quantlab.base.config import BaseFactorConfig
from quantlab.base.factor import Factor


class _FakeDataset:
    """Only what `Factor`'s config setter touches.

    `_reset_dataset_config()` writes `start_date`/`end_date` onto
    `config.dataset.config`; nothing else on the dataset is reached by the
    `save()` path.
    """

    def __init__(self):
        self.config = SimpleNamespace(start_date=None, end_date=None)


class PanelFactor(Factor):
    """The smallest concrete `Factor`: one variable over `(timestamp, symbol)`.

    `cal()` puts a panel of `periods` timestamps into the backend, which is
    all `save()` needs. No KunQuant graph, no compiled code, no real dataset.
    """

    SYMBOLS = ["AAA", "BBB"]

    def _get_factor_names(self) -> tuple[str, ...]:
        return ("alpha",)

    def cal(self, periods: int, start: str = "2024-02-01"):
        times = pd.date_range(start, periods=periods, freq="D")
        panel = xr.Dataset(
            {
                "alpha": (
                    ["timestamp", "symbol"],
                    np.zeros((periods, len(self.SYMBOLS))),
                )
            },
            coords={"timestamp": times, "symbol": self.SYMBOLS},
        )
        self.data_backend.to_internal(panel)
        return self


def _factor(tmp_path: Path) -> PanelFactor:
    return PanelFactor(
        BaseFactorConfig(
            window=1,
            dataset=_FakeDataset(),  # type: ignore[arg-type]
            file_path=str(tmp_path / "alpha.zarr"),
            start_date="2024-01-01",
            end_date="2024-12-31",
        )
    )


def test_second_date_range_under_mode_a_names_mode_w_as_the_fix(
    tmp_path: Path,
):
    """The real failure path, not a simulated one.

    Two genuine `save()` calls on the same store with different `timestamp`
    lengths -- exactly what a caller doing "compute February, then compute
    March" does. The raised message must be usable without reading
    `base/factor.py`: it names `mode="w"`, names the store, and says what
    zarr's `"a"` actually means.
    """
    factor = _factor(tmp_path)
    factor.cal(periods=29).save()  # default mode="a"

    with pytest.raises(ValueError) as excinfo:
        factor.cal(periods=31, start="2024-03-01").save()

    message = str(excinfo.value)
    assert 'save(mode="w")' in message, message
    assert str(tmp_path / "alpha.zarr") in message, message
    assert "NOT" in message and "append along time" in message, message
    assert "XrBackend.append()" in message, message


def test_the_original_zarr_error_is_preserved_as_the_cause(tmp_path: Path):
    """Wrapping must not destroy evidence. Anyone debugging the store itself
    still needs zarr's own words about which variable and which sizes."""
    factor = _factor(tmp_path)
    factor.cal(periods=29).save()

    with pytest.raises(ValueError) as excinfo:
        factor.cal(periods=31, start="2024-03-01").save()

    cause = excinfo.value.__cause__
    assert isinstance(cause, ValueError)
    assert "already exists with different dimension sizes" in str(cause)


def test_mode_w_is_the_documented_way_out(tmp_path: Path):
    """The advice in the message has to actually work, or it is worse than no
    message at all."""
    factor = _factor(tmp_path)
    factor.cal(periods=29).save()
    factor.cal(periods=31, start="2024-03-01").save(mode="w")

    reloaded = xr.open_dataset(str(tmp_path / "alpha.zarr"))
    try:
        assert reloaded.sizes["timestamp"] == 31
    finally:
        reloaded.close()


def test_the_default_is_still_a(tmp_path: Path):
    """Pinning the decision, not just the code: the default was left alone on
    purpose. Changing `"a"` to `"w"` would silently start truncating stores
    for any caller who relies on the current default."""
    import inspect

    default = inspect.signature(Factor.save).parameters["mode"].default
    assert default == "a"


def test_unrelated_value_errors_are_not_swallowed(tmp_path: Path):
    """The wrapper matches on zarr's dimension-size wording. Any other
    `ValueError` from the write must pass through untouched -- a catch-all
    that relabelled every failure as "use mode=w" would be a new bug."""
    factor = _factor(tmp_path)
    factor.cal(periods=4)

    def _boom(*args, **kwargs):
        raise ValueError("something else entirely")

    factor.data_backend.write = _boom  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="something else entirely"):
        factor.save()
