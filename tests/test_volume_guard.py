"""Pre-flight acquisition volume guard (03.2 SC-6, D-09).

A high-frequency request is easy to write and expensive to discover. `us_all`
quotes over a decade is not a slow download, it is a multi-terabyte one, and
the current failure mode is that it starts, runs for hours and fills the disk.
SC-6 requires the refusal to happen BEFORE the client is constructed and before
any request is issued, to name a concrete narrowing (fewer symbols, a shorter
window, a coarser frequency) rather than just saying no, and to be overridable
with an explicit `--force-volume`.

Two grounded constants already live on `acquisition/universe.py:UniverseCatalog`
and the new acquisition-volume estimator is built beside them, sharing their
arithmetic. This module pins both, so an edit to either surfaces HERE -- next to
the guard whose thresholds it silently shifts -- rather than only in the
dense-panel tests it was written for.

`tick` frequency without an explicit `rows_per_symbol_day` must REFUSE rather
than guess: an invented row count produces an invented budget, and the whole
point of the guard is that its number is defensible (house rule: unknown is
represented by absence and never guessed).

Every test here is offline and allocates nothing: the estimator is arithmetic
over roster size, window length and frequency. No network call, no credential,
no real data volume.

This file lands in 03.2-01 (Wave 0) carrying its constant self-test; 03.2-04
Tasks 1-2 fill in the estimator and refusal tests. It is deliberately NOT an
empty placeholder: a pytest file with zero collected tests exits 5 ("no tests
ran"), which a later task's automated command reads as green.
"""


def test_the_grounded_constants_the_new_volume_guard_is_built_beside():
    """Self-test: the two existing `UniverseCatalog` constants the acquisition
    volume estimator shares arithmetic with.

    `MAX_DENSE_PANEL_BYTES` is 4 GiB -- a RAM budget, not a disk one (a dense
    float64 panel is materialised in memory before it is written).
    `TRADING_DAYS_PER_YEAR` is 252, the figure every window-length estimate in
    this codebase multiplies through.

    `UniverseCatalog` is imported INSIDE the test body on purpose. A module-scope
    import of `acquisition.universe` here would make this file a new
    collection-time liability of the same kind `tests/conftest.py`'s docstring
    forbids.
    """
    from acquisition.universe import UniverseCatalog

    assert UniverseCatalog.MAX_DENSE_PANEL_BYTES == 4 * 1024**3, (
        "MAX_DENSE_PANEL_BYTES changed; the acquisition volume guard's budget "
        "arithmetic is derived from it -- update both together, deliberately"
    )
    assert UniverseCatalog.TRADING_DAYS_PER_YEAR == 252, (
        "TRADING_DAYS_PER_YEAR changed; every window-length estimate in the "
        "volume guard multiplies through it"
    )
