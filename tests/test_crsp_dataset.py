"""CRSP panel semantics: prices, returns, events and the adjustment anchor.

The tracer (`tests/test_crsp_tracer.py`) proved ONE PERMNO-month travels the
whole path. This module is about what the numbers in that panel MEAN -- the
part a factor, a label or a backtest reads without knowing the vendor:

- the twelve Tiingo variables plus the CRSP extras, every one float64 (D-07);
- `close = abs(dlyprc)` with the bid/ask midpoint flagged rather than hidden,
  and a missing CRSP return kept as NaN rather than filled with 0 (D-09);
- the delisting loss in the series EXACTLY ONCE -- CIZ already puts it on its
  own `dlyret` row, so `stkdelists.delret` is event data and never compounded
  on top (D-10, D-19);
- distributions and splits on their ex-dates (D-11);
- one adjustment anchor per configured window whatever the chunking, recorded
  in a sidecar so a later run cannot splice a second anchor into the same
  column (D-08).

**Every quantlab import is INSIDE a test or helper body.** That is not style:
these tests are written before the names they assert on exist, and a
module-scope import would turn the RED run into a collection error -- zero
tests discovered, which proves nothing about the behaviour (TDD gate #3770).

**Provenance rule, inherited from `tests/crsp_fixtures.py`.** A row reused from
that module is VERBATIM live data; a row built here with `dsf_row` is invented
and carries a `# SYNTHETIC` comment. Every synthetic row keeps `dsf_row`'s
NS/EQTY/COM defaults, so plan 08's common-stock filter keeps it.
"""

from __future__ import annotations

from datetime import date

import pytest

#: A real PERMNO (Microsoft) used for every SYNTHETIC scenario, so the digits
#: in these tests are not a PERMNO that means something else.
SYNTHETIC_PERMNO = "10107"
SYNTHETIC_SYMBOL = "MSFT"

#: Apple and Lehman, the two PERMNOs whose VERBATIM rows the fixtures carry.
AAPL_PERMNO = "14593"
LEHMAN_PERMNO = "80599"

#: WestRock, whose 2024 delisting row is the MODERN CIZ shape: `dlyprcflg='DA'`
#: with `dlyprc = 0.0` as a NO-PRICE sentinel and the factor/volume columns
#: NULL. That shape is 5 of 5 delisting rows in the raw tier this phase pulled;
#: Lehman's `DP` shape is 0 of 5.
WESTROCK_PERMNO = "21186"
WESTROCK_SYMBOL = "WRK"

#: The SAME securities, spelled the way the PANEL spells them: its symbol axis
#: is the int64 PERMNO (D-01, phase 03.11), never the ticker. The `*_SYMBOL`
#: constants above stay because the reference-tier fixtures are keyed on the
#: ticker -- `stksecurityinfohist` is where a ticker legitimately lives.
SYNTHETIC_AXIS = int(SYNTHETIC_PERMNO)
AAPL_AXIS = int(AAPL_PERMNO)
LEHMAN_AXIS = int(LEHMAN_PERMNO)
WESTROCK_AXIS = int(WESTROCK_PERMNO)


# ---------------------------------------------------------------------------
# Helpers: raw tier -> reference tier -> converted store
# ---------------------------------------------------------------------------


def _synthetic_secinfo():
    """A `stksecurityinfohist` interval for `SYNTHETIC_PERMNO`.

    SYNTHETIC: the live check never sampled 10107's security info. The shape
    is `secinfo_row`'s ordinary-common-share default, and the span is wide
    enough to label every scenario below.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(
            int(SYNTHETIC_PERMNO),
            "1986-03-13",
            "2025-12-31",
            SYNTHETIC_SYMBOL,
            SYNTHETIC_SYMBOL,
            None,
        )
    ]


def _reference_rows(extra_secinfo=()):
    """The fixture module's default reference rows plus `extra_secinfo`."""
    from tests.crsp_fixtures import (
        CCM_ROWS,
        DELISTS_ROWS,
        DISTRIBUTION_ROWS,
        DSP500_ROWS,
        IDXCST_ROWS,
        SECINFO_ROWS,
    )

    return {
        "crsp_a_stock.stksecurityinfohist": list(SECINFO_ROWS) + list(extra_secinfo),
        "crsp_a_stock.stkdelists": list(DELISTS_ROWS),
        "crsp_a_stock.stkdistributions": list(DISTRIBUTION_ROWS),
        "crsp_a_indexes.dsp500list_v2": list(DSP500_ROWS),
        "comp.idxcst_his": list(IDXCST_ROWS),
        "crsp_a_ccm.ccmxpf_lnkhist": list(CCM_ROWS),
    }


def _pull(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    product_end="2025-12-31",
    extra_secinfo=(),
):
    """Serve `rows` through the fake session, land the raw + reference tiers.

    Production's own path: `run_crsp_pull` goes through the registry, and
    `write_reference_tables` writes the shape plan 04's real writer produces.
    """
    from quantlab.acquisition.wrds.crsp import WrdsCrspDailyAcquisition
    from tests.crsp_fixtures import (
        FakeCrspSession,
        run_crsp_pull,
        write_reference_tables,
    )

    FakeCrspSession.daily_rows = list(rows)
    FakeCrspSession.product_end = date.fromisoformat(product_end)

    cfg, result = run_crsp_pull(tmp_path, permnos, start_date=start, end_date=end)
    assert result.failures == {}, result.failures

    reference_dir = WrdsCrspDailyAcquisition.reference_dir_for(cfg)
    write_reference_tables(
        reference_dir,
        _reference_rows(extra_secinfo),
        product_end=product_end,
    )
    return cfg, str(reference_dir)


def _dataset_config(tmp_path, cfg, reference_dir, *, start, end, store="crsp.zarr"):
    from quantlab.base.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
    )


def _convert(dataset_config, granularity="year"):
    from quantlab import registry
    from quantlab.acquisition.wrds import WRDS_SOURCE

    return registry.convert(
        WRDS_SOURCE,
        dataset_config,
        data_type="crsp_daily",
        granularity=granularity,
    )


def _panel(dataset_config):
    import xarray as xr

    return xr.open_zarr(dataset_config.zarr_file_path).load()


def _build_store(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    granularity="year",
    extra_secinfo=(),
    product_end="2025-12-31",
):
    """Pull, convert, and return the CONVERTED store's config."""
    cfg, reference_dir = _pull(
        tmp_path,
        rows,
        permnos,
        start=start,
        end=end,
        product_end=product_end,
        extra_secinfo=extra_secinfo,
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start=start, end=end
    )
    _convert(dataset_config, granularity=granularity)
    return dataset_config


def _build(tmp_path, rows, permnos, **kwargs):
    """`_build_store`, opened -- the common three-line preamble."""
    return _panel(_build_store(tmp_path, rows, permnos, **kwargs))


def _at(panel, variable, day, symbol):
    return float(panel[variable].sel(timestamp=day, symbol=symbol).values)


# ---------------------------------------------------------------------------
# Task 1: prices, flags and returns
# ---------------------------------------------------------------------------


def test_the_panel_carries_the_tiingo_variables_plus_the_crsp_extras(
    mock_crsp_session, tmp_path
):
    """D-07: the variable set is exactly Tiingo's twelve, the CRSP extras and
    `anomaly_flag`, and every variable but the flag is float64.

    Float64 for EVERY extra, the two 1.0/0.0 indicators included,
    is RESEARCH Pitfall 7: a dense `[timestamp, symbol]` panel is a cartesian
    product, so a symbol that did not exist yet needs a NaN to say so -- which
    an integer or boolean column has no room for.
    """
    import numpy as np

    from quantlab.dataset.crsp import CRSP_EXTRA_VARIABLES, CrspStockDataset
    from tests.crsp_fixtures import AAPL_1980_ROWS

    panel = _build(
        tmp_path,
        AAPL_1980_ROWS,
        [AAPL_PERMNO],
        start="1980-12-01",
        end="1980-12-31",
    )

    # 14, not 15: `permno` left the variable set when it BECAME the axis
    # (D-01). A float64 copy of the symbol coordinate is a second source of
    # truth for the same number, and nothing keeps the two in step.
    assert len(CRSP_EXTRA_VARIABLES) == 14, CRSP_EXTRA_VARIABLES
    assert "permno" not in CRSP_EXTRA_VARIABLES, CRSP_EXTRA_VARIABLES
    assert "permco" in CRSP_EXTRA_VARIABLES, CRSP_EXTRA_VARIABLES
    assert set(panel.data_vars) == (
        set(CrspStockDataset.TIINGO_VARIABLES)
        | set(CRSP_EXTRA_VARIABLES)
        | {"anomaly_flag"}
    ), sorted(panel.data_vars)

    for name in sorted(set(panel.data_vars) - {"anomaly_flag"}):
        assert panel[name].dtype == np.dtype("float64"), (name, panel[name].dtype)


def test_aapl_1980_carries_the_crsp_extra_variables_in_panel_units(
    mock_crsp_session, tmp_path
):
    """The VERBATIM 1980 rows, read back through the panel's own units.

    `shrout` and `dlycap` are stored by CRSP in THOUSANDS; the panel states
    shares and dollars, so both are multiplied by 1000 here rather than at
    every call site. The pre-1992 Nasdaq era has no OHLC and no volume at all
    (RESEARCH Pitfall 11), so those four are NaN -- expected, not an error.
    """
    import numpy as np

    from tests.crsp_fixtures import AAPL_1980_ROWS

    panel = _build(
        tmp_path,
        AAPL_1980_ROWS,
        [AAPL_PERMNO],
        start="1980-12-01",
        end="1980-12-31",
    )

    assert _at(panel, "close", "1980-12-12", AAPL_AXIS) == pytest.approx(28.8125)
    assert _at(panel, "market_cap", "1980-12-12", AAPL_AXIS) == pytest.approx(
        1588606.0 * 1000
    )
    assert _at(panel, "shrout", "1980-12-12", AAPL_AXIS) == pytest.approx(55136 * 1000)
    assert _at(panel, "cumfacpr", "1980-12-12", AAPL_AXIS) == pytest.approx(224.0)
    assert _at(panel, "cumfacshr", "1980-12-12", AAPL_AXIS) == pytest.approx(224.0)
    # The security id is not READ off a variable any more -- it is the label
    # the row was selected by (D-01).
    assert panel["symbol"].values.tolist() == [AAPL_AXIS]

    for name in ("open", "high", "low", "volume", "close_trade", "numtrd"):
        assert np.isnan(_at(panel, name, "1980-12-12", AAPL_AXIS)), name

    # The return chain itself: the first row has no return, the second has the
    # live -5.2061% one.
    assert np.isnan(_at(panel, "ret", "1980-12-12", AAPL_AXIS))
    assert _at(panel, "ret", "1980-12-15", AAPL_AXIS) == pytest.approx(-0.052061)
    assert _at(panel, "retx", "1980-12-15", AAPL_AXIS) == pytest.approx(-0.052061)


def test_a_bidask_midpoint_day_is_priced_and_flagged(mock_crsp_session, tmp_path):
    """D-09/D-19: `dlyprcflg == 'BA'` is the NO-TRADE indicator, and the
    midpoint is a real price rather than a hole.

    AAPL's first three days are all `BA` (VERBATIM `C4_sample`), so the flag
    is 1.0 throughout and `bid`/`ask` bracket the midpoint. The flag comes
    from `dlyprcflg`, NOT from the sign of `dlyprc`: CIZ carries no negative
    prices at all (live `L10_1`: 0 negative, 122,471 `BA` rows in 2000), so a
    sign test would flag nothing and the no-trade days would be invisible.
    """
    from tests.crsp_fixtures import AAPL_1980_ROWS

    panel = _build(
        tmp_path,
        AAPL_1980_ROWS,
        [AAPL_PERMNO],
        start="1980-12-01",
        end="1980-12-31",
    )

    assert _at(panel, "prc_is_bidask", "1980-12-12", AAPL_AXIS) == pytest.approx(1.0)
    assert _at(panel, "bid", "1980-12-12", AAPL_AXIS) == pytest.approx(28.75)
    assert _at(panel, "ask", "1980-12-12", AAPL_AXIS) == pytest.approx(28.875)
    assert _at(panel, "is_delisting", "1980-12-12", AAPL_AXIS) == pytest.approx(0.0)


def test_a_negative_price_becomes_a_positive_close_and_keeps_its_flag(
    mock_crsp_session, tmp_path
):
    """D-09: `close = abs(dlyprc)`.

    A no-op guard on CIZ -- the live count of negative prices is ZERO -- kept
    because the legacy CRSP convention encoded "bid/ask midpoint" as a
    negative price, and a legacy-shaped row must not enter the panel as a
    negative price that every downstream ratio would silently invert.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: CIZ has no negative price to transcribe.
        dsf_row(SYNTHETIC_PERMNO, "2020-03-02", dlyprc="10.000000", dlyret="0.010000"),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-03",
            dlyprc="-10.500000",
            dlyprcflg="BA",
            dlyret="0.050000",
        ),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    assert _at(panel, "close", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(10.5)
    assert _at(panel, "prc_is_bidask", "2020-03-03", SYNTHETIC_AXIS) == (
        pytest.approx(1.0)
    )
    assert _at(panel, "prc_is_bidask", "2020-03-02", SYNTHETIC_AXIS) == (
        pytest.approx(0.0)
    )


def test_a_missing_ret_stays_nan_and_contributes_a_factor_of_one(
    mock_crsp_session, tmp_path
):
    """D-09 + the gap-spanning rule: a NULL `dlyret` is NaN in the panel, and
    the adjustment chain treats it as a factor of 1.

    Filling it with 0 would be wrong TWICE. In the stored `ret` it would
    invent a flat day a model would train on; in the chain it would double
    count, because CIZ returns span gaps back to `DlyPrevDt` -- so the NEXT
    valid return already covers the missing day. That is what the ratio below
    asserts: across the gap, `adjClose` moves by exactly day 4's own return.
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: a five-day window with a Missing-Price day 3.
        dsf_row(SYNTHETIC_PERMNO, "2020-03-02", dlyprc="100.000000", dlyret="0.010000"),
        dsf_row(SYNTHETIC_PERMNO, "2020-03-03", dlyprc="102.000000", dlyret="0.020000"),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-04",
            dlyprc=None,
            dlyprcflg=None,
            dlyret=None,
            dlyretmissflg="MP",
        ),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-05",
            dlyprc="105.060000",
            dlyret="0.030000",
            dlyretdurflg="D2",
        ),
        dsf_row(SYNTHETIC_PERMNO, "2020-03-06", dlyprc="103.000000", dlyret="-0.019608"),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    assert np.isnan(_at(panel, "ret", "2020-03-04", SYNTHETIC_AXIS))
    assert np.isnan(_at(panel, "close", "2020-03-04", SYNTHETIC_AXIS))
    assert np.isnan(_at(panel, "prc_is_bidask", "2020-03-04", SYNTHETIC_AXIS))

    spanning = _at(panel, "adjClose", "2020-03-05", SYNTHETIC_AXIS) / _at(
        panel, "adjClose", "2020-03-03", SYNTHETIC_AXIS
    )
    assert spanning == pytest.approx(1.03, rel=1e-9)


def test_lehman_delisting_loss_is_counted_exactly_once(mock_crsp_session, tmp_path):
    """D-10/D-19: the delisting return is a `dlyret` row, not an addition.

    `stkdelists` holds `delret = -0.6` for PERMNO 80599 (VERBATIM `L3_2`) and
    `dsf_v2` holds the SAME -0.6 on 2008-09-18 with `dlydelflg='Y'` (VERBATIM
    `L3_1`). Compounding both would apply the loss twice, and the panel would
    understate a delisted holding by 40% -- the exact direction that flatters
    a backtest. The three-day product below is what pins it.

    The delisting row's ticker is NULL and its security-info interval's ticker
    is NULL too, so the symbol survives only through symbology's carry rule.
    A test that let that row vanish would look like a passing test with
    survivorship bias restored.
    """
    from tests.crsp_fixtures import LEHMAN_2008_ROWS

    panel = _build(
        tmp_path,
        LEHMAN_2008_ROWS,
        [LEHMAN_PERMNO],
        start="2008-09-12",
        end="2008-09-30",
    )

    assert panel["symbol"].values.tolist() == [LEHMAN_AXIS]
    assert [
        _at(panel, "is_delisting", day, LEHMAN_AXIS)
        for day in (
            "2008-09-12",
            "2008-09-15",
            "2008-09-16",
            "2008-09-17",
            "2008-09-18",
        )
    ] == [0.0, 0.0, 0.0, 0.0, 1.0]

    assert _at(panel, "ret", "2008-09-18", LEHMAN_AXIS) == pytest.approx(-0.6)
    assert _at(panel, "close", "2008-09-18", LEHMAN_AXIS) == pytest.approx(0.052)
    # The delisting row is NO LONGER the anchor: the anchor is this PERMNO's
    # FIRST usable row in the window, so this row's adjusted close is only
    # APPROXIMATELY its own close -- the raw prices and the return chain agree
    # to 5.355e-06 here, not exactly. That relative difference is just outside
    # `pytest.approx`'s 1e-6 default, which is why the tolerance is spelled
    # out: the gap is the evidence that this row stopped being the anchor.
    assert _at(panel, "adjClose", "2008-09-18", LEHMAN_AXIS) == pytest.approx(
        0.052, rel=1e-4
    )

    assert _at(panel, "adjClose", "2008-09-18", LEHMAN_AXIS) / _at(
        panel, "adjClose", "2008-09-17", LEHMAN_AXIS
    ) == pytest.approx(0.4)
    assert _at(panel, "adjClose", "2008-09-18", LEHMAN_AXIS) / _at(
        panel, "adjClose", "2008-09-15", LEHMAN_AXIS
    ) == pytest.approx(1.428571 * 0.433333 * 0.4, rel=1e-9)


def test_a_modern_delisting_keeps_a_real_adjusted_level(
    mock_crsp_session, tmp_path
):
    """GAP-0: the OTHER delisting shape -- the one that is 5 of 5 in the data.

    Lehman's row above is `dlyprcflg='DP'`, a delisting PRICE: 0.052 is a real
    value. WestRock's 2024-07-08 row is `dlyprcflg='DA'`, a delisting AMOUNT:
    CRSP writes `dlyprc = 0.000000` there as a NO-PRICE SENTINEL and leaves
    `dlyclose`, `dlyvol`, `dlycumfacpr` and `dlycumfacshr` NULL.

    Reading that sentinel as a close breaks the panel twice over, through two
    independent mechanisms:

    - the anchor filter tests `close.is_not_null()`, and 0.0 is not null, so the
      sentinel row BECOMES the anchor; `_close_anchor = 0.0` makes `adjClose`,
      `_factor` and therefore `adjOpen/adjHigh/adjLow` exactly 0.0 for the
      security's WHOLE history (GAP-A);
    - `dlycumfacshr` is NULL on that same row, so `_cumfacshr_anchor` is null and
      `adjVolume = dlyvol * (dlycumfacshr / null)` is NaN for the whole history,
      while raw `volume` is fully populated -- survivorship bias through the
      volume column (GAP-B).

    The delisting RETURN is not in question: -0.005630 stays in `ret` and in the
    cumulative chain either way. What the fix changes is the LEVEL the chain is
    anchored to, which is why the ratio between the two priced days below is
    asserted alongside the levels themselves.

    Every number here is VERIFIED: 03.10-REVIEW.md CR-01 reproduced these raw
    rows read-only from the pulled tier, and the expected adjusted values follow
    from them by hand -- 49.75 is the FIRST real close, so it IS the anchor
    (phase 03.12), and 2024-07-05 sits one 3.5377% day above it.
    """
    import numpy as np

    from tests.crsp_fixtures import WESTROCK_2024_ROWS

    panel = _build(
        tmp_path,
        WESTROCK_2024_ROWS,
        [WESTROCK_PERMNO],
        start="2024-07-01",
        end="2024-07-31",
    )

    assert panel["symbol"].values.tolist() == [WESTROCK_AXIS]

    # (a) the adjusted LEVEL is real, and is not the zeroed column.
    #     The anchor is the FIRST usable row, 2024-07-03, so THAT row's
    #     adjusted close is its own raw close and the later day is carried up
    #     from it by the return chain. 2024-07-05's raw close (51.51) and the
    #     chained level (49.75 * 1.035377 = 51.51000575) differ by 1.1e-07 --
    #     CRSP's own rounding between its price and its return, which only
    #     becomes visible once that row stops being the anchor.
    anchor_day = _at(panel, "adjClose", "2024-07-03", WESTROCK_AXIS)
    later_day = _at(panel, "adjClose", "2024-07-05", WESTROCK_AXIS)
    assert anchor_day == pytest.approx(49.75, rel=1e-12)
    assert later_day == pytest.approx(49.75 * 1.035377, rel=1e-9)
    assert later_day == pytest.approx(51.51, rel=1e-5)
    assert anchor_day != 0.0
    assert later_day != 0.0
    assert later_day / anchor_day == pytest.approx(1.035377, rel=1e-9)

    adj_open = _at(panel, "adjOpen", "2024-07-05", WESTROCK_AXIS)
    assert adj_open == pytest.approx(
        50.78 * (49.75 * 1.035377 / 51.51), rel=1e-9
    )
    assert adj_open != 0.0

    # (b) adjVolume is finite wherever raw volume is -- `dlycumfacshr` is 1.0 on
    #     both priced days, so the adjusted volume IS the raw volume.
    for day, raw_volume in (("2024-07-03", 4435075.0), ("2024-07-05", 11862010.0)):
        adj_volume = _at(panel, "adjVolume", day, WESTROCK_AXIS)
        assert np.isfinite(adj_volume), day
        assert adj_volume == pytest.approx(raw_volume, rel=1e-9)
        assert _at(panel, "volume", day, WESTROCK_AXIS) == pytest.approx(
            raw_volume, rel=1e-9
        )

    # (c) the sentinel is not a trade: raw `close` is NaN, never 0.0.
    delisting_close = _at(panel, "close", "2024-07-08", WESTROCK_AXIS)
    assert np.isnan(delisting_close), delisting_close

    # The chain itself is untouched, and the row is still the delisting row
    # carrying WestRock's symbol through a NULL ticker.
    assert _at(panel, "ret", "2024-07-08", WESTROCK_AXIS) == pytest.approx(
        -0.005630, rel=1e-9
    )
    assert _at(panel, "is_delisting", "2024-07-08", WESTROCK_AXIS) == 1.0


def test_the_delisting_fixture_corpus_covers_both_ciz_shapes():
    """The corpus must never again carry only ONE delisting shape.

    Until this plan, `tests/crsp_fixtures.py` encoded exactly one: Lehman
    2008-09-18, `dlyprcflg='DP'`, `dlyprc = 0.052`, a REAL delisting price with
    ordinary factor and volume columns. That shape is **0 of 5** delisting rows
    in the raw tier this phase actually pulled; the delisting-AMOUNT shape --
    `dlyprcflg='DA'`, `dlyprc = 0.000000` as a no-price sentinel,
    `dlycumfacshr` NULL -- is **5 of 5**. Two blockers (GAP-A, GAP-B) shipped
    green across 179 passing CRSP tests for precisely that reason: the suite was
    complete about a shape the data no longer produces.

    This is the durable guard, not part of this plan's red set. It is GREEN as
    soon as the fixture rows land, and it fails the moment either shape is
    dropped from the corpus -- which is the only way this class of defect can
    become invisible again.
    """
    from tests.crsp_fixtures import LEHMAN_2008_ROWS, WESTROCK_2024_ROWS

    delisting_rows = [
        row
        for row in list(LEHMAN_2008_ROWS) + list(WESTROCK_2024_ROWS)
        if row["dlydelflg"] == "Y"
    ]
    assert delisting_rows

    flags = {row["dlyprcflg"] for row in delisting_rows}
    # Both CIZ delisting flags: PRICE (a real price) and AMOUNT (a sentinel).
    assert "DP" in flags, flags
    assert "DA" in flags, flags

    assert any(row["dlyprc"] == "0.000000" for row in delisting_rows), delisting_rows
    assert any(row["dlycumfacshr"] is None for row in delisting_rows)
    assert any(
        row["dlyprc"] is not None and float(row["dlyprc"]) > 0.0
        for row in delisting_rows
    )


# ---------------------------------------------------------------------------
# Task 2: events on their ex-dates, and the drop-in check
# ---------------------------------------------------------------------------

#: Where the SYNTHETIC 2:1 split falls inside `_split_series_rows`.
SPLIT_INDEX = 20
SERIES_DAYS = 40

#: SYNTHETIC filler securities, present for ONE structural reason: KunQuant's
#: `"TS"` input layout refuses a symbol axis whose length is not a multiple of
#: the SIMD width. Measured on this machine (arm64): 4, 8 and 16 symbols run;
#: 1, 2, 3 and 5 raise `RuntimeError: Bad shape at <column>`. Eight rather
#: than four because the width is 8 on x86 AVX2 and 4 on arm64 NEON, and a
#: suite that passes on one host must pass on the other.
#:
#: They pad the panel and nothing else -- every assertion below reads the
#: security the scenario is about (`MSFT` or `LEH`), never a companion.
COMPANIONS: tuple[tuple[int, str], ...] = tuple(
    (90000 + index, f"TST{index}") for index in range(1, 8)
)


def _companion_secinfo():
    """A `stksecurityinfohist` interval per companion. SYNTHETIC throughout."""
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(permno, "1900-01-01", "2025-12-31", ticker, ticker, None)
        for permno, ticker in COMPANIONS
    ]


def _companion_rows(days):
    """One ordinary `dsf_v2` row per companion per day. SYNTHETIC throughout."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for offset, (permno, _ticker) in enumerate(COMPANIONS):
        price = 20.0 + offset
        for index, day in enumerate(days):
            daily_return = round(0.03 - 0.05 * (index % 2) + 0.001 * offset, 6)
            price *= 1.0 + daily_return
            rows.append(
                dsf_row(
                    permno,
                    day,
                    dlyprc=f"{price:.6f}",
                    dlyclose=f"{price:.6f}",
                    dlyopen=f"{price * 0.99:.6f}",
                    dlyhigh=f"{price * 1.02:.6f}",
                    dlylow=f"{price * 0.98:.6f}",
                    dlyret=f"{daily_return:.6f}",
                    dlyretx=f"{daily_return:.6f}",
                )
            )
    return rows


def _companion_permnos():
    return [str(permno) for permno, _ in COMPANIONS]


def _split_series_rows():
    """SYNTHETIC: `SERIES_DAYS` business days with one 2:1 split, self-consistent.

    Returns `(rows, days, returns)`. The raw close is `2 x value` before the
    split and `value` from the split day on, where `value` compounds the
    return path -- so the RAW price ratio across the split is
    `(1 + r) / 2`, exactly as CRSP records it, while the total return is
    plain `r`. That is what makes "the label equals the next day's `ret`
    ACROSS the split" a real assertion rather than an arithmetic identity.

    Returns alternate around +-4% rather than +-0.4% deliberately: KunQuant
    computes in float32, and `close(t+1)/close(t) - 1` loses absolute
    precision to cancellation, so a tiny return would be compared at a
    relative tolerance float32 cannot hold.
    """
    import pandas as pd

    from tests.crsp_fixtures import dsf_row

    days = pd.bdate_range("2020-01-02", periods=SERIES_DAYS)
    returns = [
        round(0.05 - 0.09 * (index % 2) + 0.002 * (index % 5), 6)
        for index in range(SERIES_DAYS)
    ]

    rows = []
    value = 50.0
    for index, (day, daily_return) in enumerate(zip(days, returns)):
        value *= 1.0 + daily_return
        raw = value * (2.0 if index < SPLIT_INDEX else 1.0)
        split_day = index == SPLIT_INDEX
        rows.append(
            dsf_row(
                SYNTHETIC_PERMNO,
                day.date().isoformat(),
                dlyprc=f"{raw:.6f}",
                dlyclose=f"{raw:.6f}",
                dlyopen=f"{raw * 0.99:.6f}",
                dlyhigh=f"{raw * 1.02:.6f}",
                dlylow=f"{raw * 0.98:.6f}",
                dlyret=f"{daily_return:.6f}",
                dlyretx=f"{daily_return:.6f}",
                dlyfacprc="2.000000" if split_day else "1.000000",
                dlycumfacpr=(
                    "2.000000000000" if index < SPLIT_INDEX else "1.000000000000"
                ),
                dlycumfacshr=(
                    "2.000000000000" if index < SPLIT_INDEX else "1.000000000000"
                ),
            )
        )
    return rows, [day.date().isoformat() for day in days], returns


def _split_series_store(tmp_path, *, with_companions=False):
    """The split series, optionally padded to the eight-symbol KunQuant width."""
    rows, days, returns = _split_series_rows()
    permnos = [SYNTHETIC_PERMNO]
    extra_secinfo = list(_synthetic_secinfo())
    if with_companions:
        rows = rows + _companion_rows(days)
        permnos = permnos + _companion_permnos()
        extra_secinfo = extra_secinfo + _companion_secinfo()
    dataset_config = _build_store(
        tmp_path,
        rows,
        permnos,
        start=days[0],
        end=days[-1],
        extra_secinfo=extra_secinfo,
    )
    return dataset_config, days, returns


def _factor_config(dataset_config, *, factor_names, data_columns, tmp_path, **kwargs):
    """A `FactorConfig` over a CRSP store, built exactly as a Tiingo one is.

    `factor_names` is always explicit so the compiled graph stays tiny (the
    169-name Alpha158 default would compile for minutes), and `njobs=4` keeps
    the KunQuant executor from spawning the config default's 128 threads.
    """
    from quantlab.base.config import FactorConfig
    from quantlab.dataset.crsp import CrspStockDataset

    return FactorConfig(
        window=kwargs.pop("window", 10),
        dataset=CrspStockDataset(dataset_config),
        mode="batch",
        data_columns=tuple(data_columns),
        factor_names=tuple(factor_names),
        file_path=str(tmp_path / "factors" / "out.zarr"),
        njobs=4,
        **kwargs,
    )


def test_events_land_on_their_ex_dates(mock_crsp_session, tmp_path):
    """D-11: `divCash` on the ex-date, `splitFactor` and `facprc` on the split.

    VERBATIM `L4_1`: AAPL's 2020-08-07 dividend (0.82) and its 2020-08-31 4:1
    split, where `dlycumfacpr` steps 4 -> 1 and `dlyfacprc` reads 4. The
    panel says the same three things in Tiingo's vocabulary, so a consumer
    reading `divCash`/`splitFactor` needs no CRSP knowledge.
    """
    from tests.crsp_fixtures import AAPL_AUG_2020_ROWS

    panel = _build(
        tmp_path,
        AAPL_AUG_2020_ROWS,
        [AAPL_PERMNO],
        start="2020-08-01",
        end="2020-08-31",
    )

    assert _at(panel, "divCash", "2020-08-07", AAPL_AXIS) == pytest.approx(0.82)
    for day in ("2020-08-06", "2020-08-28", "2020-08-31"):
        assert _at(panel, "divCash", day, AAPL_AXIS) == pytest.approx(0.0), day

    assert _at(panel, "splitFactor", "2020-08-31", AAPL_AXIS) == pytest.approx(4.0)
    assert _at(panel, "facprc", "2020-08-31", AAPL_AXIS) == pytest.approx(4.0)
    # 1.0 on an ordinary day -- and on 08-06, the PERMNO's FIRST row in the
    # window, where there is no previous `dlycumfacpr` to divide by.
    for day in ("2020-08-06", "2020-08-07", "2020-08-28"):
        assert _at(panel, "splitFactor", day, AAPL_AXIS) == pytest.approx(1.0), day
        assert _at(panel, "facprc", day, AAPL_AXIS) == pytest.approx(1.0), day


def test_divcash_sums_ordinary_and_non_ordinary_distributions(
    mock_crsp_session, tmp_path
):
    """`divCash = dlyorddivamt + dlynonorddivamt`, unadjusted, on the ex-date.

    CRSP splits a day's cash into an ORDINARY and a NON-ORDINARY component;
    Tiingo's `divCash` is one number. Summing is therefore the drop-in answer,
    and both-null is 0.0 rather than NaN -- a day with no distribution paid
    nothing, which is a known amount, not a missing one.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: a special dividend beside an ordinary one.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-02",
            dlyprc="100.000000",
            dlyret="0.010000",
            dlyorddivamt="0.200000",
            dlynonorddivamt="1.500000",
        ),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-03",
            dlyprc="101.000000",
            dlyret="0.010000",
            dlyorddivamt=None,
            dlynonorddivamt=None,
        ),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    assert _at(panel, "divCash", "2020-03-02", SYNTHETIC_AXIS) == pytest.approx(1.7)
    assert _at(panel, "divCash", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(0.0)


def test_splitfactor_and_facprc_mark_the_split_day(mock_crsp_session, tmp_path):
    """The synthetic 2:1 split shows up as 2.0 in both event variables.

    `splitFactor` is DERIVED (the previous `dlycumfacpr` over today's) while
    `facprc` is CRSP's own per-day factor. Asserting both on the same row is
    what would catch a derivation that drifted off by one day.
    """
    dataset_config, days, _ = _split_series_store(tmp_path)
    panel = _panel(dataset_config)

    split_day = days[SPLIT_INDEX]
    assert _at(panel, "splitFactor", split_day, SYNTHETIC_AXIS) == pytest.approx(2.0)
    assert _at(panel, "facprc", split_day, SYNTHETIC_AXIS) == pytest.approx(2.0)
    # On a day that DID trade, `close_trade` (`dlyclose`) and `close`
    # (`abs(dlyprc)`) agree -- the two diverge only where there was no trade,
    # which is what makes `dlyprc` the right source for the panel's close.
    assert _at(panel, "close_trade", split_day, SYNTHETIC_AXIS) == pytest.approx(
        _at(panel, "close", split_day, SYNTHETIC_AXIS)
    )
    for index in (SPLIT_INDEX - 1, SPLIT_INDEX + 1):
        assert _at(panel, "splitFactor", days[index], SYNTHETIC_AXIS) == (
            pytest.approx(1.0)
        ), days[index]


def test_alpha158_computes_over_a_crsp_panel_with_no_consumer_change(
    mock_crsp_session, tmp_path
):
    """D-07's whole point: `Alpha158Stock` runs on a CRSP store unmodified.

    The factor class is constructed exactly as `tests/test_factor_kunquant.py`
    constructs it over a Tiingo store -- same `FactorConfig`, same five
    adjusted `data_columns` -- with only the `Dataset` subclass swapped. No
    CRSP branch exists in `quantlab/factor`, and this test is what would fail
    if one were needed.
    """
    import numpy as np

    from quantlab.factor.alpha158 import Alpha158Stock

    dataset_config, days, _ = _split_series_store(tmp_path, with_companions=True)
    factor = Alpha158Stock(
        _factor_config(
            dataset_config,
            factor_names=["KMID", "ROC5", "STD5"],
            data_columns=["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"],
            tmp_path=tmp_path,
        )
    )

    result = factor.cal().get_features()

    assert dict(result.sizes) == {"timestamp": SERIES_DAYS, "symbol": 8}
    assert sorted(result.data_vars) == ["KMID", "ROC5", "STD5"]
    series = result.sel(symbol=SYNTHETIC_AXIS)
    for name in ("KMID", "ROC5", "STD5"):
        finite = np.isfinite(series[name].to_numpy())
        # Rolling factors burn their window at the head; everything after it
        # must be a real number.
        assert finite.sum() >= SERIES_DAYS - 6, (name, finite.sum())


def test_the_return_label_equals_the_next_days_crsp_ret(mock_crsp_session, tmp_path):
    """The `Return` label over a CRSP panel IS CRSP's own next-day `dlyret`.

    Not a tautology: the label is computed by KunQuant from `adjClose`, while
    `ret` is the vendor's number carried through untouched. They agree only if
    the total-return adjustment reproduces the return chain exactly -- across
    the 2:1 split included, where the RAW price ratio is `(1 + r) / 2`.
    """
    import numpy as np

    from quantlab.label.fret import Return

    dataset_config, days, returns = _split_series_store(tmp_path, with_companions=True)
    label = Return(
        _factor_config(
            dataset_config,
            factor_names=["ret_1"],
            data_columns=["adjClose"],
            tmp_path=tmp_path,
            window=0,
            kwargs={"n_forward_periods": 1},
        )
    )

    labels = label.cal().get_labels().load()
    series = labels["ret_1"].sel(symbol=SYNTHETIC_AXIS).to_numpy()

    assert len(series) == SERIES_DAYS
    for index in range(SERIES_DAYS - 1):
        assert series[index] == pytest.approx(returns[index + 1], rel=1e-5), (
            index,
            days[index],
        )
    # The last bar has no next day, so it has no label.
    assert np.isnan(series[-1])


def test_return_label_over_lehmans_delisting_day(mock_crsp_session, tmp_path):
    """The label on 2008-09-17 is the -60% delisting loss.

    This is the survivorship-bias test stated in the vocabulary a model
    actually trains on. If the delisting row were dropped, or its return
    double-counted, this cell would hold NaN or -0.84 instead.
    """
    from quantlab.label.fret import Return
    from tests.crsp_fixtures import LEHMAN_2008_ROWS

    # Lehman's five VERBATIM rows, padded to the eight-symbol KunQuant width
    # by companions trading on the SAME five days. The padding changes no
    # value of `LEH`'s own series -- the panel is a cartesian product.
    days = [
        "2008-09-12",
        "2008-09-15",
        "2008-09-16",
        "2008-09-17",
        "2008-09-18",
    ]
    dataset_config = _build_store(
        tmp_path,
        list(LEHMAN_2008_ROWS) + _companion_rows(days),
        [LEHMAN_PERMNO] + _companion_permnos(),
        start="2008-09-12",
        end="2008-09-30",
        extra_secinfo=_companion_secinfo(),
    )
    label = Return(
        _factor_config(
            dataset_config,
            factor_names=["ret_1"],
            data_columns=["adjClose"],
            tmp_path=tmp_path,
            window=0,
            kwargs={"n_forward_periods": 1},
        )
    )

    labels = label.cal().get_labels().load()
    value = float(
        labels["ret_1"].sel(timestamp="2008-09-17", symbol=LEHMAN_AXIS).values
    )
    assert value == pytest.approx(-0.6, rel=1e-5)


# ---------------------------------------------------------------------------
# Task 3: one anchor per window, whatever the chunking
# ---------------------------------------------------------------------------

ANCHOR_START = "2019-11-01"
ANCHOR_END = "2020-09-30"
#: The 4:1 split and the dividend inside the anchor window.
ANCHOR_SPLIT_DAY = "2020-08-31"
ANCHOR_DIVIDEND_DAY = "2020-08-07"
ANCHOR_DIVIDEND = 0.82


def _anchor_series_rows():
    """SYNTHETIC: ~11 months of daily rows with one 4:1 split and a dividend.

    Long enough that `granularity="year"` plans TWO windows and
    `granularity="month"` plans ELEVEN -- which is the whole point. If the
    adjustment anchor were computed per window rather than once over
    `[start_date, end_date]`, the two stores would disagree at every window
    seam (RESEARCH Pitfall 1), and the warning sign named there is exactly
    "the year and month granularity stores differ".
    """
    import pandas as pd

    from tests.crsp_fixtures import dsf_row

    days = pd.bdate_range(ANCHOR_START, ANCHOR_END)
    rows = []
    value = 60.0
    previous_raw = None
    for index, day in enumerate(days):
        iso = day.date().isoformat()
        daily_return = round(0.02 - 0.035 * (index % 2) + 0.0005 * (index % 11), 6)
        value *= 1.0 + daily_return
        pre_split = 4.0 if iso < ANCHOR_SPLIT_DAY else 1.0
        raw = value * pre_split

        without_dividend = daily_return
        if iso == ANCHOR_DIVIDEND_DAY and previous_raw:
            without_dividend = round(daily_return - ANCHOR_DIVIDEND / previous_raw, 6)

        rows.append(
            dsf_row(
                SYNTHETIC_PERMNO,
                iso,
                dlyprc=f"{raw:.6f}",
                dlyclose=f"{raw:.6f}",
                dlyopen=f"{raw * 0.99:.6f}",
                dlyhigh=f"{raw * 1.02:.6f}",
                dlylow=f"{raw * 0.98:.6f}",
                dlyret=f"{daily_return:.6f}",
                dlyretx=f"{without_dividend:.6f}",
                dlyorddivamt=(
                    f"{ANCHOR_DIVIDEND:.6f}"
                    if iso == ANCHOR_DIVIDEND_DAY
                    else "0.000000"
                ),
                dlyfacprc="4.000000" if iso == ANCHOR_SPLIT_DAY else "1.000000",
                dlycumfacpr=(
                    "4.000000000000" if iso < ANCHOR_SPLIT_DAY else "1.000000000000"
                ),
                dlycumfacshr=(
                    "4.000000000000" if iso < ANCHOR_SPLIT_DAY else "1.000000000000"
                ),
            )
        )
        previous_raw = raw
    return rows


def test_year_and_month_granularity_produce_identical_stores(
    mock_crsp_session, tmp_path
):
    """D-08: the anchor is GLOBAL to the configured window, so chunking is
    invisible in the output.

    Two converts of the SAME raw tier into two paths, one planning two
    year-windows and one planning eleven month-windows. `assert_identical`
    compares every variable, coordinate and name -- a per-window anchor would
    put a fabricated return at each of the ten extra seams, so this is the
    test Pitfall 1's warning sign was written for.
    """
    import xarray as xr

    cfg, reference_dir = _pull(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    stores = {}
    for granularity in ("year", "month"):
        dataset_config = _dataset_config(
            tmp_path,
            cfg,
            reference_dir,
            start=ANCHOR_START,
            end=ANCHOR_END,
            store=f"crsp_{granularity}.zarr",
        )
        result = _convert(dataset_config, granularity=granularity)
        stores[granularity] = _panel(dataset_config)
        assert result.granularity == granularity

    xr.testing.assert_identical(stores["year"], stores["month"])
    assert stores["year"].sizes["timestamp"] > 200, stores["year"].sizes


def test_the_anchor_is_the_FIRST_usable_row(mock_crsp_session, tmp_path):
    """The anchor is the FIRST row with a usable level, not simply the first.

    A usable level is a strictly positive `close` AND a non-null
    `dlycumfacshr`, on ONE row. A PERMNO whose opening row in the window is a
    Missing-Price day must therefore push its anchor forward by a row;
    anchoring on that null would make every adjusted value in its entire
    history NaN. The null row itself still keeps a NaN `adjClose`: there was
    no price that day, and publishing the anchor's level there would invent
    one.

    **Why this test was rewritten rather than left alone** (phase 03.12). Its
    three assertions stayed green when the anchor moved from `.last()` to
    `.first()` -- the fixture's prices and returns are consistent, so both
    anchors produce the same numbers on it. But its NAME and its docstring
    said "last", and a test that passes while claiming to test something it
    no longer tests is worse than no test: it is a confident false statement
    about how the system works. The fixture's priceless row therefore moved
    from the END of the window to the START, which is where the anchor rule
    now has to do its work.
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the window's FIRST row has no price.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-01",
            dlyprc=None,
            dlyprcflg=None,
            dlyret=None,
            dlyretmissflg="MP",
        ),
        dsf_row(SYNTHETIC_PERMNO, "2020-03-02", dlyprc="100.000000", dlyret="0.010000"),
        dsf_row(SYNTHETIC_PERMNO, "2020-03-03", dlyprc="102.000000", dlyret="0.020000"),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    # The anchor skipped the priceless opening row and landed here, so this
    # row's adjusted close IS its own close.
    assert _at(panel, "adjClose", "2020-03-02", SYNTHETIC_AXIS) == pytest.approx(
        100.0
    )
    assert np.isnan(_at(panel, "adjClose", "2020-03-01", SYNTHETIC_AXIS))
    # And the rest of the series is carried forward from that close, not NaN.
    assert _at(panel, "adjClose", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        102.0
    )


def test_a_permno_with_only_sentinel_prices_is_refused_by_name(
    mock_crsp_session, tmp_path
):
    """GAP-A: "non-null" is not "usable". A 0.0 sentinel is not a level.

    The anchor guard above tests `close.is_not_null()`, and CRSP's no-price
    sentinel is `dlyprc = 0.000000` -- a number, not a NULL. A PERMNO whose only
    priced rows are sentinels therefore anchors on 0.0, and
    `adjClose = 0.0 * _G / _G_anchor` is exactly 0.0 on every day it existed.

    A refusal is the right outcome rather than a NaN column, let alone a zero
    one. Zero is a legal float that no consumer rejects: `alpha158` reads the
    five `adj*` columns and turns a zeroed column into `0/0 -> NaN` returns,
    `x/0 -> inf` ratios and a cross-sectional rank pinned to the bottom on every
    day -- silently, for exactly the securities that delisted. The conversion
    cannot compute an adjusted series here, and saying so by name is the only
    answer that cannot be mistaken for data.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: every priced row is a no-price sentinel.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-02",
            dlyprc="0.000000",
            dlyprcflg="DA",
            dlyret="0.010000",
        ),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-03",
            dlyprc="0.000000",
            dlyprcflg="DA",
            dlyret="-0.005000",
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        _build_store(
            tmp_path,
            rows,
            [SYNTHETIC_PERMNO],
            start="2020-03-01",
            end="2020-03-31",
            extra_secinfo=_synthetic_secinfo(),
        )

    assert SYNTHETIC_PERMNO in str(excinfo.value)


def test_a_permno_whose_priced_rows_lack_cumfacshr_is_refused_by_name(
    mock_crsp_session, tmp_path
):
    """GAP-B: every quantity read OFF the anchor row must be present ON it.

    The anchor is chosen on `close` alone, and `dlycumfacshr` is then read off
    that same row. When it is NULL there, `_cumfacshr_anchor` is null,
    `_volume_factor = dlycumfacshr / null` is null for every row of the PERMNO,
    and `adjVolume` is NaN for the security's whole history -- while raw
    `volume` is fully populated. This is an independent mechanism from GAP-A:
    fixing `close` does not fix it.

    A refusal beats an all-NaN column because of who reads it. Any liquidity
    screen, turnover factor or volume-weighted signal built on `adjVolume`
    silently drops every security whose anchor row lacked the factor -- which is
    every delisted security -- reintroducing the survivorship bias D-10 exists
    to remove, through a different door. A NaN column looks like "no data"; it is
    actually "the data was there and the adjustment lost it".
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: real prices, but no share-adjustment factor anywhere.
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-02",
            dlyprc="100.000000",
            dlyret="0.010000",
            dlycumfacshr=None,
        ),
        dsf_row(
            SYNTHETIC_PERMNO,
            "2020-03-03",
            dlyprc="102.000000",
            dlyret="0.020000",
            dlycumfacshr=None,
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        _build_store(
            tmp_path,
            rows,
            [SYNTHETIC_PERMNO],
            start="2020-03-01",
            end="2020-03-31",
            extra_secinfo=_synthetic_secinfo(),
        )

    message = str(excinfo.value)
    assert SYNTHETIC_PERMNO in message
    # Named by its CRSP column, so the refusal says WHICH quantity is missing.
    assert "dlycumfacshr" in message


def test_a_total_loss_return_chain_is_refused_rather_than_made_infinite(
    mock_crsp_session, tmp_path
):
    """GAP-A's third mechanism: a zero DENOMINATOR, not a zero anchor price.

    `_G` is `cum_prod(1 + dlyret)`. A `dlyret` of exactly -1.0 -- a total loss,
    which is a legal CRSP return -- makes `_G` exactly 0.0 from that row onward.
    When the ANCHOR row is one of those rows, `_G_anchor` is 0.0 too, and
    `adjClose = _close_anchor * _G / 0.0` is `inf` before the loss and
    `0/0 -> NaN` after it, under IEEE semantics and with no exception raised.

    **The fixture moved in phase 03.12, and that is the point.** The anchor is
    now a PERMNO's FIRST usable row, not its last, so the row that has to
    carry the -1.0 to make `_G_anchor` zero is the FIRST one. Under the old
    `.last()` anchor a mid-series loss put the zero in the denominator; under
    `.first()` it does not -- see
    `test_a_total_loss_after_the_anchor_is_not_refused_and_zeroes_the_tail`
    below, which pins what happens instead.

    No such row exists in the raw tier this phase pulled, and nothing in
    `_derivation` prevents one: the guard is being demanded here precisely
    because the defect is latent rather than observed. An infinite adjusted
    series is worse than a refusal for the same reason a zeroed one is -- `inf`
    propagates through every factor and every rank without ever raising.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the ANCHOR row -- the first one -- is a total loss.
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-02", dlyprc="100.000000", dlyret="-1.000000"
        ),
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-03", dlyprc="50.000000", dlyret="0.010000"
        ),
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-04", dlyprc="49.000000", dlyret="0.020000"
        ),
    ]

    with pytest.raises(ValueError) as excinfo:
        _build_store(
            tmp_path,
            rows,
            [SYNTHETIC_PERMNO],
            start="2020-03-01",
            end="2020-03-31",
            extra_secinfo=_synthetic_secinfo(),
        )

    assert SYNTHETIC_PERMNO in str(excinfo.value)


def test_a_total_loss_after_the_anchor_is_not_refused_and_zeroes_the_tail(
    mock_crsp_session, tmp_path
):
    """A KNOWN, CURRENTLY UNGUARDED consequence of the backward anchor.

    This is a CHARACTERIZATION test, not an endorsement. With the anchor at a
    PERMNO's FIRST usable row (phase 03.12), a `dlyret` of exactly -1.0 on any
    LATER row no longer lands in `_G_anchor`, so `_assert_anchor_usable` does
    not fire. `_G` is still exactly 0.0 from the loss onward, so every
    subsequent `adjClose` is exactly 0.0 -- a $0.00 adjusted price published
    on days whose raw `close` is 50.0 and 49.0.

    That is CR-01's harm (a zeroed adjusted column feeding `0/0 -> NaN`
    returns and a cross-sectional rank pinned to the bottom) reached through a
    door the `.last()` anchor happened to close and the `.first()` anchor does
    not. It is latent: no such row exists in the raw tier this project pulled,
    and the raw data would have to be self-contradictory to produce one (a
    -100% return followed by further trading).

    Phase 03.12 plan 01 is forbidden from touching `_assert_anchor_usable`
    (D-05 preservation), so the gap is RECORDED rather than patched here --
    see `.planning/WINDOWS.md`. **When the guard is extended to cover a
    post-anchor zero, DELETE this test**; it exists only to make sure the
    behaviour is discovered on purpose rather than in a factor library.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the MIDDLE row is a total loss, priced.
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-02", dlyprc="100.000000", dlyret="0.010000"
        ),
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-03", dlyprc="50.000000", dlyret="-1.000000"
        ),
        dsf_row(
            SYNTHETIC_PERMNO, "2020-03-04", dlyprc="49.000000", dlyret="0.020000"
        ),
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    # The anchor row itself is unharmed: it publishes its own close.
    assert _at(panel, "adjClose", "2020-03-02", SYNTHETIC_AXIS) == pytest.approx(
        100.0, rel=1e-12
    )
    # Everything from the loss onward is exactly 0.0 while the raw close is not.
    assert _at(panel, "adjClose", "2020-03-03", SYNTHETIC_AXIS) == 0.0
    assert _at(panel, "adjClose", "2020-03-04", SYNTHETIC_AXIS) == 0.0
    assert _at(panel, "close", "2020-03-03", SYNTHETIC_AXIS) == pytest.approx(
        50.0, rel=1e-12
    )
    assert _at(panel, "close", "2020-03-04", SYNTHETIC_AXIS) == pytest.approx(
        49.0, rel=1e-12
    )


def test_a_second_convert_over_the_same_window_resumes(mock_crsp_session, tmp_path):
    """Re-running the SAME window is a no-op resume, not a refusal.

    This used to be the honest-case companion of a cross-run gate that
    compared the conversion window against a record beside the store. That
    gate is gone (phase 03.12: the anchor is each PERMNO's first usable row,
    so it does not move when the window grows), but the claim it guarded here
    still has to hold on its own: a second conversion of an already-complete
    store finds every window in the chunk ledger and writes nothing.

    The extension case -- same `start_date`, later `end_date` -- is no longer
    a refusal at all; it is the supported append, pinned bit-for-bit by
    `tests/test_crsp_first_anchor.py::test_a_full_and_incremental_build_agree_bit_for_bit`.
    """
    dataset_config = _build_store(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    again = _convert(dataset_config, granularity="year")

    assert again.windows_written == 0, again
    assert again.windows_skipped == again.windows_planned, again
    assert again.resumed is True, again


# ---------------------------------------------------------------------------
# WR-02 / WR-03: both entry points record, and a refused run keeps
# the audit trail of the store that survived it
# ---------------------------------------------------------------------------

#: A second real PERMNO (Oracle), used only to WIDEN a roster -- the one
#: refusal that fires AFTER the identity reports have been written.
SECOND_PERMNO = "10104"
SECOND_SYMBOL = "ORCL"


def _second_secinfo():
    """A `stksecurityinfohist` interval for `SECOND_PERMNO`.

    SYNTHETIC, like `_synthetic_secinfo`: the live check never sampled 10104's
    security info, and the span only has to cover the anchor window.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(
            int(SECOND_PERMNO),
            "1986-03-12",
            "2025-12-31",
            SECOND_SYMBOL,
            SECOND_SYMBOL,
            None,
        )
    ]


def _second_security_rows():
    """SYNTHETIC: a handful of ordinary days for `SECOND_PERMNO`."""
    import pandas as pd

    from tests.crsp_fixtures import dsf_row

    rows = []
    value = 25.0
    for index, day in enumerate(pd.bdate_range(ANCHOR_START, ANCHOR_END)):
        value *= 1.001
        rows.append(
            dsf_row(
                SECOND_PERMNO,
                day.date().isoformat(),
                dlyprc=f"{value:.6f}",
                dlyclose=f"{value:.6f}",
                dlyopen=f"{value * 0.99:.6f}",
                dlyhigh=f"{value * 1.01:.6f}",
                dlylow=f"{value * 0.98:.6f}",
                dlyret="0.001000" if index else None,
            )
        )
    return rows


def _crsp_config(tmp_path, cfg, reference_dir, *, permnos, store="crsp.zarr"):
    """`_dataset_config` with an explicit PERMNO roster."""
    from quantlab.base.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        reference_dir=str(reference_dir),
        start_date=ANCHOR_START,
        end_date=ANCHOR_END,
        permnos=permnos,
    )


def _report_paths(dataset_config):
    """`(filter report, symbology report, ticker sidecar)` paths beside the store.

    The symbology suffix is spelled as a LITERAL here, not imported: its
    constant was deleted with the rest of the ticker-identity machinery in
    03.11-07, and the tests below assert that this file is NOT written. A test
    that the file is absent must be able to name the file without the code
    under test agreeing that the name exists.

    The other two ARE imported, for the mirror-image reason: they must be
    written, so their constants necessarily exist, and importing them means a
    renamed suffix is a failing assertion here rather than a silently skipped
    one.
    """
    from pathlib import Path

    from quantlab.dataset.crsp import (
        FILTER_REPORT_SUFFIX,
        TICKER_SIDECAR_SUFFIX,
    )

    base = str(dataset_config.zarr_file_path)
    return (
        Path(base + FILTER_REPORT_SUFFIX),
        Path(base + ".crsp_symbology_report.json"),
        Path(base + TICKER_SIDECAR_SUFFIX),
    )


def test_from_raw_data_leaves_a_complete_sidecar_set(mock_crsp_session, tmp_path):
    """WR-02: the NON-chunked entry point records its provenance too.

    `from_raw_data().save()` is the idiom every other dataset in this repo
    supports, and `scripts/ingest_wrds_crsp.py` uses it four lines from the CRSP
    call. The identity reports were once written only from
    `_raw_axes_in_range`, which ONLY the chunked path calls -- so that idiom
    produced a store with no provenance at all, and which conversion entry
    point had been used silently decided whether an audit trail existed.

    **This is the test for BOTH entry points leaving records.** The chunked one
    is covered everywhere else in this module; what is pinned HERE is that
    `from_raw_data` (through `_raw_data_to_xr`) leaves the same set as
    `from_raw_data_chunked` (through `_raw_axes_in_range`), because the two
    call different overrides and nothing else forces them to agree.

    A later chunked conversion may still refuse -- a store written in one
    `mode="w"` shot has no CHUNK ledger, and appending blind to a store whose
    written windows are unrecorded is refused for every dataset in the repo,
    CRSP included. That refusal names the ledger and states its remedy.
    """
    from quantlab.dataset.crsp import CrspStockDataset

    cfg, reference_dir = _pull(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start=ANCHOR_START, end=ANCHOR_END
    )

    CrspStockDataset(dataset_config).from_raw_data().save()

    filter_report, symbology_report, ticker_sidecar = _report_paths(
        dataset_config
    )
    for path in (filter_report, ticker_sidecar):
        assert path.exists(), sorted(item.name for item in tmp_path.iterdir())

    # The SYMBOLOGY report is not part of the set any more (D-01, phase
    # 03.11). Every one of its six keys -- collisions, seams, class_suffixed,
    # nonconforming_symbols, unlabelled, delisting_carried -- describes
    # something that can only happen while the panel is keyed on a ticker. On
    # a PERMNO axis they are all structurally empty, and a sidecar that can
    # only ever say "nothing happened" is worse than no sidecar: it reads like
    # evidence that the checks ran.
    assert not symbology_report.exists(), sorted(
        item.name for item in tmp_path.iterdir()
    )

    try:
        _convert(dataset_config)
    except ValueError as exc:
        message = str(exc)
        assert "chunk ledger" in message, message


def test_a_refused_reconversion_keeps_the_existing_identity_reports(
    mock_crsp_session, tmp_path
):
    """WR-03: a run that refuses must not erase the SURVIVING store's audit
    trail.

    `.crsp_filter_report.json` is the D-17 artifact whose whole purpose is to
    say what the store on disk dropped. `_write_identity_reports` ran
    unconditionally from `_raw_axes_in_range`, i.e. BEFORE
    `_reconcile_new_listings`' `on_new_listing='refuse'` arm and before
    `ChunkLedger.assert_consistent` -- either of which still aborts the run.
    So a wider-roster re-conversion overwrote both reports with the NEW roster's
    numbers, then refused and appended nothing: the store was unchanged and its
    two provenance sidecars now described a panel that was never written.

    The scenario has to be a refusal that fires AFTER those writes. Extending
    `end_date` does not: the anchor gate refuses inside the derivation, before
    any report is written, so it would pass this test with or without the guard.
    Widening the roster on the SAME window does -- the anchor record is
    identical, so the gate lets the run through to the axis reconciliation.
    """
    cfg, reference_dir = _pull(
        tmp_path,
        _anchor_series_rows() + _second_security_rows(),
        [SYNTHETIC_PERMNO, SECOND_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo() + _second_secinfo(),
    )

    narrow = _crsp_config(
        tmp_path, cfg, reference_dir, permnos=(SYNTHETIC_PERMNO,)
    )
    _convert(narrow)

    filter_report, symbology_report, ticker_sidecar = _report_paths(narrow)
    filter_bytes = filter_report.read_bytes()
    ticker_bytes = ticker_sidecar.read_bytes()
    # No symbology report to preserve on a PERMNO axis -- see
    # `test_from_raw_data_leaves_a_complete_sidecar_set` for why it is gone.
    assert not symbology_report.exists(), symbology_report

    widened = _crsp_config(
        tmp_path, cfg, reference_dir, permnos=(SYNTHETIC_PERMNO, SECOND_PERMNO)
    )
    with pytest.raises(ValueError):
        _convert(widened)

    assert filter_report.read_bytes() == filter_bytes
    # The ticker sidecar is under the same guard and for the same reason: the
    # widened roster would have named SECOND_PERMNO in a file sitting beside a
    # store that never carried it (03.11-09).
    assert ticker_sidecar.read_bytes() == ticker_bytes
    assert not symbology_report.exists(), symbology_report



# ---------------------------------------------------------------------------
# D-14 / RULING 1: admission without a ticker is COUNTED, not predicated away
# ---------------------------------------------------------------------------
#
# On the ticker axis, a PERMNO whose security-info intervals carried no ticker
# on a given day could not enter the panel at all: the daily labeller (deleted
# in 03.11-07) as-of joined onto the NAMED symbol intervals and DROPPED every
# row it could not label. That was an admission rule nobody had written down --
# "must have a ticker" was a side effect of needing a column name.
#
# On the PERMNO axis the rule simply stops applying, and the panel gets wider.
# D-10 originally asked for the implicit rule to be replaced by an explicit
# `securitytype`/`sharetype` predicate. RESEARCH R5c measured that this cannot
# be done: 1,003 of the 1,012 never-ticker PERMNOs read `EQTY/COM/NS`, the same
# combination as ordinary common stock, so any type predicate either excludes
# none of them or excludes real common stock with them. `securityactiveflg` is
# not a column of the daily raw tier at all.
#
# The operator's RULING 1 is therefore: LET THEM IN, and make the widening
# VISIBLE. `{zarr}.crsp_filter_report.json` carries an `admitted_without_ticker`
# key -- always, even when empty -- and a warning fires when it is not.
#
# The measured boundary: every never-ticker interval in the raw tier ends
# before 1983-04-13, so the widening for any window starting on or after
# 1990-08-20 is exactly zero. Both halves are asserted below.

TICKERLESS_PERMNO = "7000"
TICKERLESS_DAYS = ("1982-06-01", "1982-06-02", "1982-06-03")
MODERN_DAYS = ("1990-08-21", "1990-08-22", "1990-08-23")


def _filter_report(dataset_config):
    """The `{zarr}.crsp_filter_report.json` payload beside a converted store."""
    import json
    from pathlib import Path

    from quantlab.dataset.crsp import FILTER_REPORT_SUFFIX

    path = Path(str(dataset_config.zarr_file_path) + FILTER_REPORT_SUFFIX)
    return json.loads(path.read_text(encoding="utf-8"))


def _plain_rows(permnos_and_days):
    """SYNTHETIC ordinary daily rows: `[(permno, days), ...]`."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for permno, days in permnos_and_days:
        price = 20.0
        for day in days:
            price *= 1.01
            rows.append(
                dsf_row(
                    permno,
                    day,
                    dlyprc=f"{price:.6f}",
                    dlyclose=f"{price:.6f}",
                    dlyret="0.010000",
                    dlyretx="0.010000",
                )
            )
    return rows


def test_a_permno_that_never_had_a_ticker_is_admitted_and_counted(
    mock_crsp_session, tmp_path
):
    """RULING 1: it is IN the panel, and the report says how much that cost.

    PERMNO 7000's single interval carries a NULL ticker (see `SECINFO_ROWS`),
    so on the ticker axis every one of its rows was dropped as `unlabelled`.
    Here it is a column like any other -- and the widening is a counted,
    warned-about fact rather than a silent one.

    The four-digit PERMNO is deliberate on a second count: `[7000, 14593]` is
    the numeric axis order (D-19), while the lexicographic order this code used
    until plan 03 would have produced `['14593', '7000']`.
    """
    dataset_config = _build_store(
        tmp_path,
        _plain_rows(
            [(TICKERLESS_PERMNO, TICKERLESS_DAYS), (AAPL_PERMNO, TICKERLESS_DAYS)]
        ),
        [TICKERLESS_PERMNO, AAPL_PERMNO],
        start="1982-06-01",
        end="1982-06-30",
    )
    panel = _panel(dataset_config)

    # Admitted, not excluded -- and in numeric order.
    assert panel["symbol"].values.tolist() == [
        int(TICKERLESS_PERMNO),
        AAPL_AXIS,
    ], panel["symbol"].values.tolist()

    report = _filter_report(dataset_config)
    assert "admitted_without_ticker" in report, sorted(report)
    admitted = report["admitted_without_ticker"]
    assert admitted["permnos"] == [int(TICKERLESS_PERMNO)], admitted
    assert admitted["rows"] == len(TICKERLESS_DAYS), admitted
    # AAPL HAS a ticker on these days, so it is not counted -- without this the
    # field could pass by naming every PERMNO in the panel.
    assert AAPL_AXIS not in admitted["permnos"], admitted


def test_no_permno_is_admitted_without_a_ticker_after_1990_08_20(
    mock_crsp_session, tmp_path
):
    """The measured boundary (RESEARCH R5c), and the key's UNCONDITIONAL
    presence.

    Every never-ticker interval in the raw tier ends before 1983-04-13, so a
    window starting on or after 1990-08-20 admits nobody without a ticker --
    which is why the two stores already on disk are untouched by RULING 1.

    The key is written anyway. Absence and zero must stay distinguishable: a
    reader of a store built before this field existed would otherwise be unable
    to tell "nothing was admitted without a ticker" from "nobody counted".
    """
    dataset_config = _build_store(
        tmp_path,
        _plain_rows([(AAPL_PERMNO, MODERN_DAYS)]),
        [AAPL_PERMNO],
        start="1990-08-20",
        end="1990-09-30",
    )

    report = _filter_report(dataset_config)
    assert "admitted_without_ticker" in report, sorted(report)
    assert report["admitted_without_ticker"] == {"permnos": [], "rows": 0}, report[
        "admitted_without_ticker"
    ]


def test_a_delisted_permno_keeps_its_last_row(mock_crsp_session, tmp_path):
    """The delisting row survives the security filter, and MUST keep doing so.

    **This is a prohibition guard, not a feature test.** The filter's verdict
    inheritance (`quantlab/dataset/crsp/__init__.py`, `_apply_security_filter`) and
    `CrspSymbology`'s ticker carry are described side by side in one docstring
    paragraph. The symbology half is deleted in plan 07; the filter half must
    NOT be, and the two are one edit apart.

    What is lost if it goes: a delisted security's last row is exactly where
    CRSP's type columns go blank, AND it is the row carrying the delisting
    RETURN. Judged on its own blank types the row is dropped, every delisting
    loss silently disappears, survivorship bias walks back in one row at a
    time -- and the panel stays completely well-formed while it happens.

    WestRock's 2024-07-08 row is the modern CIZ shape, verbatim from the raw
    tier: `dlydelflg='Y'`, a `dlyprc = 0.0` no-price sentinel, and every one of
    the five TYPE columns NULL. The nullness is asserted from the fixture
    itself first, so this test cannot pass by accident on a row the filter
    would have kept on its own merits.
    """
    from tests.crsp_fixtures import WESTROCK_2024_ROWS

    delisting_row = next(
        row for row in WESTROCK_2024_ROWS if row["dlycaldt"] == "2024-07-08"
    )
    assert delisting_row["dlydelflg"] == "Y", delisting_row
    for column in (
        "sharetype",
        "securitytype",
        "securitysubtype",
        "issuertype",
        "usincflg",
    ):
        # If this ever stops being None the test below proves nothing: the row
        # would be kept on its own types, carry or no carry.
        assert delisting_row[column] is None, (column, delisting_row[column])

    dataset_config = _build_store(
        tmp_path,
        WESTROCK_2024_ROWS,
        [WESTROCK_PERMNO],
        start="2024-07-01",
        end="2024-07-31",
    )
    panel = _panel(dataset_config)

    assert "2024-07-08" in [
        str(value)[:10] for value in panel["timestamp"].values
    ], panel["timestamp"].values

    assert _at(panel, "is_delisting", "2024-07-08", WESTROCK_AXIS) == pytest.approx(
        1.0
    )
    assert _at(panel, "ret", "2024-07-08", WESTROCK_AXIS) == pytest.approx(
        -0.005630
    )

    # And the report agrees it was not dropped -- the row count is the other
    # half, because a row could be absent from `dropped_permnos` while never
    # having reached the filter at all.
    report = _filter_report(dataset_config)
    assert report["rows_dropped"] == 0, report
    assert report["dropped_permnos"] == {}, report["dropped_permnos"]
    assert report["rows_kept"] == len(WESTROCK_2024_ROWS), report
