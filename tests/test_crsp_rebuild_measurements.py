"""The W0 REAL-DATA gate: rebuild the on-disk CRSP panel and lock its numbers.

**This is not a unit test.** It DELETES and re-converts the real
`wrds_crsp_sp500_1d.zarr` under the repository's `data/` tree, then asserts the
five post-fix measurements phase 03.11 reasons from. Everything synthetic lives
in `tests/test_crsp_rebuild.py`; this file exists because a synthetic fixture
cannot tell you what the SHIPPED panel actually contains.

**It is deliberately EXCLUDED from the full regression command**, and excluded
by NOT BEING COLLECTED rather than by being skipped. Two independent reasons,
either of which alone would be sufficient:

1. `_data_root()` calls `pytest.fail` -- not `pytest.skip` -- when
   `QUANTLAB_DATA_ROOT` is unset, because a skip here is indistinguishable from
   a pass and this file's whole job is to refuse false greens. The regression
   command does not set that variable, so a collected copy of this file would
   make the gate structurally red forever.
2. `rebuild()` runs assert -> backup -> clear -> convert. Hanging that off the
   ordinary test loop would re-convert the real store on every `pytest` run,
   which is worse than a red gate.

So the phase-wide command is::

    uv run pytest -q \\
      --ignore=tests/test_factor_hierarchy.py \\
      --ignore=tests/test_crsp_rebuild_measurements.py \\
      -p no:cacheprovider

and this file runs on its own::

    QUANTLAB_DATA_ROOT="$(dirname "$(git rev-parse --path-format=absolute \\
      --git-common-dir)")" uv run pytest -q -s \\
      tests/test_crsp_rebuild_measurements.py -p no:cacheprovider

`.planning/config.json`'s `workflow.test_command` carries the same `--ignore`,
so the gate the tooling runs and the gate the plan describes are one gate.

**Why the root comes from `--git-common-dir`.** `data/` is gitignored, so it
does not exist inside a git worktree at all; executions are isolated into
worktrees while the data stays in the main checkout. `--git-common-dir` returns
the MAIN repository's `.git` from inside a worktree and the repository's own
`.git` from the main tree, so its parent is the main repository root in both
cases. Code under test therefore comes from the working tree while the data
comes from the main tree -- which is exactly the split that makes a false green
impossible to produce by accident (T-03.11-03).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from quantlab.base.config import CrspDatasetConfig
from quantlab.dataset.crsp_membership import CrspMembership
from quantlab.dataset.crsp_reference import CrspReference
from quantlab.dataset.crsp_rebuild import (
    CRSP_SIDECAR_SUFFIXES,
    CrspStoreRebuilder,
)

#: The window this gate rebuilds (03.11 D-15 / operator RULING 2). ONLY 2024,
#: not the raw tier's full 2019-2025 span: `03.11-RESEARCH.md` section R1 took
#: its pre/post comparison on exactly this slice, and widening the window would
#: leave the baseline numbers below with nothing comparable to be measured
#: against.
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

#: The 13 surviving anomalies on the post-fix panel: 12 split-day false
#: positives (`flag_anomalies` reads the RAW close, so a 10-for-1 split reads
#: as a -90% move) plus `GL 2024-04-11`, a real -53% day with
#: `splitFactor == 1.0`. Each is enumerated in RESEARCH section R1.
#:
#: Asserted as EQUALITY, never as an inequality. A different count means a fact
#: changed -- a different raw vintage, a different roster, a different cleaning
#: rule -- and relaxing the assertion would convert that signal into silence.
EXPECTED_ANOMALY_FLAG_TRUE = 13


def _data_root() -> Path:
    """The main repository root, from `QUANTLAB_DATA_ROOT`.

    `pytest.fail` rather than `pytest.skip` on every failure path, and no
    conditional-skip marker appears anywhere in this module. A skipped data
    gate reads as a pass in every summary line an operator will ever look at,
    which makes it worse than no gate at all.

    (The marker's name is deliberately not written out here: the acceptance
    criterion for this file is a `grep -c` for that token returning 0, and a
    mention in prose is indistinguishable from a use.)
    """
    raw = os.environ.get("QUANTLAB_DATA_ROOT")
    if not raw:
        pytest.fail(
            "QUANTLAB_DATA_ROOT is not set. This gate rebuilds the REAL CRSP "
            "store, which lives in the main repository (data/ is gitignored "
            "and therefore absent from any worktree). Run this file as:\n"
            '  QUANTLAB_DATA_ROOT="$(dirname "$(git rev-parse '
            '--path-format=absolute --git-common-dir)")" \\\n'
            "    uv run pytest -q -s tests/test_crsp_rebuild_measurements.py "
            "-p no:cacheprovider\n"
            "Failing rather than skipping on purpose: a skip here is "
            "indistinguishable from a pass."
        )

    root = Path(raw).resolve()
    if not root.is_dir():
        pytest.fail(
            f"QUANTLAB_DATA_ROOT resolved to {str(root)!r}, which is not a "
            f"directory. Expected the main repository root."
        )

    missing = [path for path in _input_paths(root) if not path.exists()]
    if missing:
        described = "\n".join(f"  - {str(path)}" for path in missing)
        pytest.fail(
            f"QUANTLAB_DATA_ROOT resolved to {str(root)!r} but the CRSP raw "
            f"tier is incomplete there:\n{described}\n"
            f"The rebuild reads the raw parquet tier and its sibling "
            f"_reference tier; without them there is nothing to convert. Pull "
            f"the raw tier, or point QUANTLAB_DATA_ROOT at the tree that "
            f"already holds it."
        )
    return root


def _input_paths(root: Path) -> tuple[Path, Path]:
    """`(raw vendor dir, reference dir)` under `root`.

    The raw path TERMINATES at `/wrds` (the vendor segment `_assert_vendor_root`
    demands) and `_reference` is a SIBLING of `wrds_crsp`, not a child of
    `wrds`. Both traps are documented on
    `CrspStoreRebuilder._required_inputs`; they are repeated in the layout here
    because this is where the literals are written.
    """
    vendor_parent = root / "data" / "downloads" / "us_equity" / "1d" / "wrds_crsp"
    return vendor_parent / "wrds", vendor_parent / "_reference"


def _config(root: Path) -> CrspDatasetConfig:
    """The sp500/2024 config, constructed DIRECTLY.

    Never through `quantlab/config`'s factory functions: this phase builds its
    configs by hand so the fields under test are visible at the call site
    rather than buried in a factory's defaults.
    """
    raw_dir, reference_dir = _input_paths(root)
    permnos = tuple(
        CrspMembership(CrspReference(reference_dir)).permnos_in_range(
            "crsp_sp500", START_DATE, END_DATE
        )
    )
    return CrspDatasetConfig(
        zarr_file_path=str(
            root
            / "data"
            / "data"
            / "us_equity"
            / "1d"
            / "wrds_crsp_sp500_1d.zarr"
        ),
        raw_data_dir_path=str(raw_dir),
        catalog_path=str(root / "data" / "data" / "catalog"),
        reference_dir=str(reference_dir),
        start_date=START_DATE,
        end_date=END_DATE,
        permnos=permnos,
        security_filter="equity_common",
        roster_universe="crsp_sp500",
    )


@pytest.fixture(scope="module")
def rebuilt():
    """Rebuild the real store ONCE and hand both tests the measurement.

    Module-scoped because the rebuild is destructive: running it per-test would
    delete and re-convert the panel twice for no additional evidence. The
    `started_at` companion is captured BEFORE the rebuild so the sidecar
    freshness assertion has a lower bound it can trust.
    """
    root = _data_root()
    started_at = time.time()
    config = _config(root)
    rebuilder = CrspStoreRebuilder(config, data_root=root)
    measurement = rebuilder.rebuild(
        backup_dir=root / "data" / "_backup_pre_03.11"
    )
    return measurement, started_at, config


def test_sp500_2024_rebuild_matches_post_fix_measurements(rebuilt):
    """The W0 acceptance gate: five numbers off the freshly-rebuilt panel."""
    measurement, _started_at, _config_used = rebuilt

    # These six lines ARE the evidence the SUMMARY transcribes. `data_root` in
    # particular is the visible proof of WHICH tree was read -- the only cheap
    # defence against a worktree false green.
    print(f"data_root:    {measurement.data_root}")
    print(f"store_path:   {measurement.store_path}")
    print(f"dims:         {measurement.dims}")
    print(f"metrics:      {measurement.metrics}")
    print(f"removed:      {measurement.removed}")
    print(f"backup_path:  {measurement.backup_path}")
    print(f"data_vars:    {measurement.data_var_count}")

    metrics = measurement.metrics

    # CR-01 / CR-02 go to exactly zero: no adjusted close is non-positive, and
    # the no-price sentinel is no longer read as a price.
    assert metrics["adj_close_le_zero"] == 0
    assert metrics["close_eq_zero"] == 0

    # Equality, not an inequality. See EXPECTED_ANOMALY_FLAG_TRUE: a different
    # count means a new fact, and must be investigated rather than absorbed.
    assert metrics["anomaly_flag_true"] == EXPECTED_ANOMALY_FLAG_TRUE, (
        f"expected {EXPECTED_ANOMALY_FLAG_TRUE} anomalies (12 split-day false "
        f"positives + GL 2024-04-11), measured "
        f"{metrics['anomaly_flag_true']}. Do NOT relax this to an inequality "
        f"-- a different count means a fact changed; enumerate the anomalies "
        f"and find out which."
    )

    # The strongest statement this phase makes. Not "there are fewer NaNs" but
    # "there is not ONE surplus NaN": every null in the adjusted series sits on
    # a cell where no bar exists at all, which is the dense panel's cartesian
    # product (D-06) and not a defect.
    assert metrics["adj_close_nan"] == metrics["structural_gaps"]
    assert metrics["adj_volume_nan"] == metrics["structural_gaps"]

    # The rebuild read the MAIN repository, not the worktree it ran from.
    assert measurement.data_root.endswith("/quantlab")
    assert Path(measurement.data_root).is_absolute()
    assert measurement.backup_path is not None
    assert (Path(measurement.backup_path) / "wrds_crsp_sp500_1d.zarr").exists()


def test_rebuild_refreshed_every_sidecar(rebuilt):
    """The reverse of Pitfall 8: every audit file describes THIS panel.

    `crsp.py:1414`'s store-exists guard means an append leaves the sidecars
    alone, so a rebuild that cleared only the store would finish with audit
    files describing the previous panel. Asserting each one's mtime is later
    than the moment the rebuild started is what makes that failure visible.
    """
    measurement, started_at, _config_used = rebuilt
    store = Path(measurement.store_path)

    for suffix in CRSP_SIDECAR_SUFFIXES:
        sidecar = Path(str(store) + suffix)
        if suffix == ".crsp_symbology_report.json" and not sidecar.exists():
            # Later plans in this phase stop GENERATING this report. It stays
            # on the clearing list regardless (a stale file describing a
            # deleted mechanism is exactly what clearing prevents), so its
            # absence after a rebuild is correct, not a failure.
            continue
        assert sidecar.exists(), f"{sidecar} was not written by the rebuild"
        assert sidecar.stat().st_mtime >= started_at, (
            f"{sidecar} has mtime {sidecar.stat().st_mtime} which predates "
            f"this rebuild ({started_at}) -- it describes the PREVIOUS panel."
        )
