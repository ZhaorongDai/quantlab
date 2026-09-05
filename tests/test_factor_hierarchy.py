"""Factor-hierarchy boundary-contract tests (FACTOR-04 + D-03).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0); the `Factor` ABC extraction
assertions were added by **03-02**. The KunQuant/Polars interchangeability
integration test lands in **03-05**.

The single test below is not a placeholder: it is the cheap, permanent
regression lock on D-03's "seamless interchangeability" claim. It passes today
(`base/model.py` contains no `Factor` reference at all) and MUST keep passing
after 03-02 splits `FactorKunQuant` into `Factor` + backend subclasses and
03-04 adds `FactorPolars`. The moment the model layer has to know which factor
backend it was handed, the interchangeability claim is dead -- and this test
is what catches it.

Written in the established grep-style purity idiom of
`tests/test_extensibility_contract.py:test_core_layer_purity_no_market_
specific_logic`, including its comment-line exclusion so an architecture note
mentioning these names in prose is not a false positive.

Import-safety rule (tests/conftest.py module docstring): nothing here may
import `base.factor_polars` or `factor.momentum` at module level -- neither
exists yet.
"""

from pathlib import Path
from typing import Callable

from base.config import DatasetConfig, FactorConfig
from base.factor import Factor, FactorKunQuant
from dataset.spot import SpotKlineDataset
from label.spot import SpotBinaryReturn, SpotReturn

CONSUMER_FILE = "base/model.py"

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
