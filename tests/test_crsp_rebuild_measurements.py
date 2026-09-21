"""The W0 REAL-DATA gate: rebuild the on-disk CRSP panel and lock its numbers.

**This is not a unit test.** It DELETES and re-converts the real
`wrds_crsp_sp500_1d.zarr` under the repository's `data/` tree, then asserts the
post-fix measurements phase 03.11 reasons from. Everything synthetic lives in
`tests/test_crsp_rebuild.py`; this file exists because a synthetic fixture
cannot tell you what the SHIPPED panel actually contains.

As of 03.11-10 it also gates the PERMNO-axis migration on the written store:
the `symbol` array's RAW Zarr dtype is integral, `.crsp_symbology_report.json`
is absent (its six keys were all statements about a ticker axis) and
`.crsp_tickers.json` is present and covers exactly the panel's own PERMNOs. The
four cleanliness assertions are stated as INVARIANTS (`adjClose <= 0 == 0`,
`close == 0 == 0`, both NaN counts `== structural_gaps`) rather than as counts,
which is why they survive the axis change unmodified; `symbol_count` is the one
number that may legitimately move, and it is reported rather than pinned.

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

import json
import os
import time
from pathlib import Path

import pytest
import zarr

from quantlab.base.config import CrspDatasetConfig
from quantlab.dataset.crsp_membership import CrspMembership
from quantlab.dataset.crsp_reference import CrspReference
from quantlab.dataset.crsp_rebuild import CrspStoreRebuilder

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

#: The `symbol_count` W0 measured on the TICKER axis (03.11-01's rebuild table).
#: Kept only to be PRINTED beside the current number, never asserted against.
#: On a PERMNO axis the column count may legitimately move in either direction:
#: a rename inside the window (FB -> META) merged two ticker columns into one
#: PERMNO column, and a ticker recycled inside the window split one ticker
#: column into two PERMNO columns. An equality here would flag a correct
#: migration as a regression.
W0_TICKER_AXIS_SYMBOL_COUNT = 523

#: The band a one-year S&P 500 roster plausibly occupies. Wide enough that the
#: rename/recycle arithmetic above cannot breach it, narrow enough that a
#: roster that silently collapsed (or swallowed the whole raw tier) does. A
#: value outside it is a signal to investigate, not a number to widen the band
#: around.
SYMBOL_COUNT_BAND = (500, 540)

#: The sidecars a rebuild must WRITE, each newer than the moment it started.
#:
#: This is deliberately NOT `CRSP_SIDECAR_SUFFIXES`. That tuple is the CLEARING
#: list -- what must be deleted before a rebuild -- and it still carries
#: `.crsp_symbology_report.json` precisely because that file is no longer
#: generated and may be lying around from an older store. The two lists answer
#: different questions, and conflating them would either stop clearing a stale
#: file or demand a file the current tree has no code to write.
EXPECTED_WRITTEN_SIDECARS: tuple[str, ...] = (
    ".chunks.json",
    ".crsp_adjustment.json",
    ".crsp_filter_report.json",
    ".crsp_tickers.json",
)

#: The audit file whose six keys were all statements about a TICKER axis
#: (resolved collisions, PERMNO seams, carried labels, class respellings,
#: unlabelled rows). On a PERMNO axis every one of them can only ever say
#: "nothing happened", which reads like evidence a check ran. 03.11-07 deleted
#: the mechanism and 03.11-09 replaced the sidecar; a rebuild must therefore
#: leave NO such file beside the store.
DEAD_SYMBOLOGY_SIDECAR = ".crsp_symbology_report.json"

#: The interval table that replaced it: PERMNO -> period-correct ticker.
TICKER_SIDECAR = ".crsp_tickers.json"


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

    **The backup directory is per-rebuild, and that matters.** `backup()` copies
    with `dirs_exist_ok=True`, so pointing two different rebuilds at one
    directory overwrites the older copy in place. `data/_backup_pre_03.11` holds
    the PRE-PHASE (pre-fix) store -- the only surviving record of what the 699 /
    686 / 2202 numbers were measured on, and unreproducible because the code
    that wrote it is deleted. This rebuild therefore writes its own
    `data/_backup_pre_03.11_10`, preserving both generations.
    """
    root = _data_root()
    started_at = time.time()
    config = _config(root)
    rebuilder = CrspStoreRebuilder(config, data_root=root)
    measurement = rebuilder.rebuild(
        backup_dir=root / "data" / "_backup_pre_03.11_10"
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

    # `symbol_count` is REPORTED against W0's number, not asserted equal to it.
    # See W0_TICKER_AXIS_SYMBOL_COUNT: on a PERMNO axis a rename inside the
    # window merges two ticker columns and a recycled ticker splits one, so
    # equality would report a correct migration as a regression. The band is
    # what still catches a roster that collapsed or swallowed the raw tier.
    low, high = SYMBOL_COUNT_BAND
    print(
        f"symbol_count: {metrics['symbol_count']} "
        f"(W0 ticker axis: {W0_TICKER_AXIS_SYMBOL_COUNT}, band: {low}-{high})"
    )
    assert low <= metrics["symbol_count"] <= high, (
        f"symbol_count {metrics['symbol_count']} falls outside the plausible "
        f"one-year S&P 500 band {low}-{high} (W0 measured "
        f"{W0_TICKER_AXIS_SYMBOL_COUNT} on the ticker axis). Do NOT widen the "
        f"band -- a count this far off means the roster or the security "
        f"filter changed, and which one is the question to answer."
    )

    # The rebuild read the MAIN repository, not the worktree it ran from.
    assert measurement.data_root.endswith("/quantlab")
    assert Path(measurement.data_root).is_absolute()
    assert measurement.backup_path is not None
    assert (Path(measurement.backup_path) / "wrds_crsp_sp500_1d.zarr").exists()


def test_the_written_axis_is_int64_permnos(rebuilt):
    """The identity axis on disk is integers, read from Zarr's OWN dtype.

    `xr.open_zarr` would hand back whatever the decoders made of the array, and
    a `VLenUTF8`-encoded axis of digit strings decodes into something that
    prints identically to an int64 axis in every log line an operator reads.
    Asking `zarr.open_group` for the raw `dtype` is the only form of this
    question that a string axis cannot pass by accident (D-01).
    """
    measurement, _started_at, _config_used = rebuilt

    stored = zarr.open_group(measurement.store_path, mode="r")["symbol"]
    print(f"symbol dtype: {stored.dtype} (kind {stored.dtype.kind!r})")
    assert stored.dtype.kind == "i", (
        f"the symbol axis is stored as {stored.dtype!r}, whose kind is "
        f"{stored.dtype.kind!r}, not 'i'. The panel's identity axis is the "
        f"PERMNO (D-01); a string axis here means the migration did not reach "
        f"the written store."
    )


def test_the_symbology_report_is_gone_and_the_ticker_table_arrived(rebuilt):
    """The dead sidecar is absent; the interval table names the panel.

    Both halves are needed. Asserting only the absence would pass on a rebuild
    that wrote no sidecars at all; asserting only the arrival would leave a file
    describing a deleted mechanism sitting beside the store it never described,
    which is the confidently-wrong audit trail the clearing rule exists to
    prevent.
    """
    measurement, _started_at, _config_used = rebuilt
    store = Path(measurement.store_path)

    dead = Path(str(store) + DEAD_SYMBOLOGY_SIDECAR)
    assert not dead.exists(), (
        f"{dead} still exists after the rebuild. All six of its keys were "
        f"statements about a ticker AXIS (resolved collisions, PERMNO seams, "
        f"carried labels, class respellings, unlabelled rows); 03.11-07 "
        f"deleted the mechanism, so nothing in the current tree can write it "
        f"and the clearing list must have removed it."
    )

    ticker_sidecar = Path(str(store) + TICKER_SIDECAR)
    assert ticker_sidecar.exists(), (
        f"{ticker_sidecar} was not written. An int64 axis cannot say what its "
        f"numbers are CALLED; this interval table is the only thing that can."
    )

    payload = json.loads(ticker_sidecar.read_text())
    intervals = payload["intervals"]
    print(f"ticker intervals: {len(intervals)} PERMNO(s)")
    print(f"vintage_product_end: {payload['vintage_product_end']}")

    # Only the PANEL's PERMNOs. `stksecurityinfohist` carries 40,518 of them;
    # writing the whole table would put megabytes of names for securities this
    # store has never heard of beside it.
    assert len(intervals) == measurement.metrics["symbol_count"], (
        f"the ticker sidecar names {len(intervals)} PERMNO(s) while the panel "
        f"carries {measurement.metrics['symbol_count']}. The sidecar is "
        f"supposed to cover exactly the panel's own axis -- more means it "
        f"leaked the reference table, fewer means some column on the axis has "
        f"no name the reference tier can supply."
    )
    axis = {
        str(int(label))
        for label in zarr.open_group(measurement.store_path, mode="r")["symbol"][:]
    }
    assert set(intervals) == axis, (
        f"the sidecar's PERMNOs and the panel's axis are the same SIZE but not "
        f"the same SET; symmetric difference: "
        f"{sorted(set(intervals) ^ axis)[:10]}"
    )


def test_rebuild_refreshed_every_sidecar(rebuilt):
    """The reverse of Pitfall 8: every audit file describes THIS panel.

    `crsp.py:1414`'s store-exists guard means an append leaves the sidecars
    alone, so a rebuild that cleared only the store would finish with audit
    files describing the previous panel. Asserting each one's mtime is later
    than the moment the rebuild started is what makes that failure visible.

    The list iterated is `EXPECTED_WRITTEN_SIDECARS`, not the clearing list --
    see that constant for why the two must not be the same tuple.
    """
    measurement, started_at, _config_used = rebuilt
    store = Path(measurement.store_path)

    for suffix in EXPECTED_WRITTEN_SIDECARS:
        sidecar = Path(str(store) + suffix)
        assert sidecar.exists(), f"{sidecar} was not written by the rebuild"
        assert sidecar.stat().st_mtime >= started_at, (
            f"{sidecar} has mtime {sidecar.stat().st_mtime} which predates "
            f"this rebuild ({started_at}) -- it describes the PREVIOUS panel."
        )
