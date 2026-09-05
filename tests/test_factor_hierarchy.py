"""Factor-hierarchy boundary-contract tests (FACTOR-04 + D-03).

Scaffolded by 03-01 Task 2 (Nyquist Wave 0). The rest of the content -- the
`Factor` ABC extraction assertions and the KunQuant/Polars interchangeability
integration test -- lands in **03-02** and **03-05**.

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

CONSUMER_FILE = "base/model.py"

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
