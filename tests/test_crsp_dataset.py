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
    from quantlab.acquisition.wrds_crsp import WrdsCrspDailyAcquisition
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
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
    )


def _convert(dataset_config, granularity="year"):
    from quantlab.acquisition import registry
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

    Float64 for EVERY extra, `permno` and the two 1.0/0.0 indicators included,
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

    assert len(CRSP_EXTRA_VARIABLES) == 15, CRSP_EXTRA_VARIABLES
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

    assert _at(panel, "close", "1980-12-12", "AAPL") == pytest.approx(28.8125)
    assert _at(panel, "market_cap", "1980-12-12", "AAPL") == pytest.approx(
        1588606.0 * 1000
    )
    assert _at(panel, "shrout", "1980-12-12", "AAPL") == pytest.approx(55136 * 1000)
    assert _at(panel, "cumfacpr", "1980-12-12", "AAPL") == pytest.approx(224.0)
    assert _at(panel, "cumfacshr", "1980-12-12", "AAPL") == pytest.approx(224.0)
    assert _at(panel, "permno", "1980-12-12", "AAPL") == pytest.approx(14593.0)

    for name in ("open", "high", "low", "volume", "close_trade", "numtrd"):
        assert np.isnan(_at(panel, name, "1980-12-12", "AAPL")), name

    # The return chain itself: the first row has no return, the second has the
    # live -5.2061% one.
    assert np.isnan(_at(panel, "ret", "1980-12-12", "AAPL"))
    assert _at(panel, "ret", "1980-12-15", "AAPL") == pytest.approx(-0.052061)
    assert _at(panel, "retx", "1980-12-15", "AAPL") == pytest.approx(-0.052061)


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

    assert _at(panel, "prc_is_bidask", "1980-12-12", "AAPL") == pytest.approx(1.0)
    assert _at(panel, "bid", "1980-12-12", "AAPL") == pytest.approx(28.75)
    assert _at(panel, "ask", "1980-12-12", "AAPL") == pytest.approx(28.875)
    assert _at(panel, "is_delisting", "1980-12-12", "AAPL") == pytest.approx(0.0)


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

    assert _at(panel, "close", "2020-03-03", SYNTHETIC_SYMBOL) == pytest.approx(10.5)
    assert _at(panel, "prc_is_bidask", "2020-03-03", SYNTHETIC_SYMBOL) == (
        pytest.approx(1.0)
    )
    assert _at(panel, "prc_is_bidask", "2020-03-02", SYNTHETIC_SYMBOL) == (
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

    assert np.isnan(_at(panel, "ret", "2020-03-04", SYNTHETIC_SYMBOL))
    assert np.isnan(_at(panel, "close", "2020-03-04", SYNTHETIC_SYMBOL))
    assert np.isnan(_at(panel, "prc_is_bidask", "2020-03-04", SYNTHETIC_SYMBOL))

    spanning = _at(panel, "adjClose", "2020-03-05", SYNTHETIC_SYMBOL) / _at(
        panel, "adjClose", "2020-03-03", SYNTHETIC_SYMBOL
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

    assert [str(value) for value in panel["symbol"].values] == ["LEH"]
    assert [
        _at(panel, "is_delisting", day, "LEH")
        for day in (
            "2008-09-12",
            "2008-09-15",
            "2008-09-16",
            "2008-09-17",
            "2008-09-18",
        )
    ] == [0.0, 0.0, 0.0, 0.0, 1.0]

    assert _at(panel, "ret", "2008-09-18", "LEH") == pytest.approx(-0.6)
    assert _at(panel, "close", "2008-09-18", "LEH") == pytest.approx(0.052)
    # The delisting row is the last row with a price, so it IS the anchor.
    assert _at(panel, "adjClose", "2008-09-18", "LEH") == pytest.approx(0.052)

    assert _at(panel, "adjClose", "2008-09-18", "LEH") / _at(
        panel, "adjClose", "2008-09-17", "LEH"
    ) == pytest.approx(0.4)
    assert _at(panel, "adjClose", "2008-09-18", "LEH") / _at(
        panel, "adjClose", "2008-09-15", "LEH"
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
    from them by hand -- 51.51 is the last real close, so it IS the anchor, and
    2024-07-03 sits one 3.5377% day below it.
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

    assert [str(value) for value in panel["symbol"].values] == [WESTROCK_SYMBOL]

    # (a) the adjusted LEVEL is real, and is not the zeroed column.
    anchor_day = _at(panel, "adjClose", "2024-07-05", WESTROCK_SYMBOL)
    previous_day = _at(panel, "adjClose", "2024-07-03", WESTROCK_SYMBOL)
    assert anchor_day == pytest.approx(51.51, rel=1e-9)
    assert previous_day == pytest.approx(51.51 / 1.035377, rel=1e-9)
    assert anchor_day != 0.0
    assert previous_day != 0.0
    assert anchor_day / previous_day == pytest.approx(1.035377, rel=1e-9)

    adj_open = _at(panel, "adjOpen", "2024-07-05", WESTROCK_SYMBOL)
    assert adj_open == pytest.approx(50.78 * (51.51 / 51.51), rel=1e-9)
    assert adj_open != 0.0

    # (b) adjVolume is finite wherever raw volume is -- `dlycumfacshr` is 1.0 on
    #     both priced days, so the adjusted volume IS the raw volume.
    for day, raw_volume in (("2024-07-03", 4435075.0), ("2024-07-05", 11862010.0)):
        adj_volume = _at(panel, "adjVolume", day, WESTROCK_SYMBOL)
        assert np.isfinite(adj_volume), day
        assert adj_volume == pytest.approx(raw_volume, rel=1e-9)
        assert _at(panel, "volume", day, WESTROCK_SYMBOL) == pytest.approx(
            raw_volume, rel=1e-9
        )

    # (c) the sentinel is not a trade: raw `close` is NaN, never 0.0.
    delisting_close = _at(panel, "close", "2024-07-08", WESTROCK_SYMBOL)
    assert np.isnan(delisting_close), delisting_close

    # The chain itself is untouched, and the row is still the delisting row
    # carrying WestRock's symbol through a NULL ticker.
    assert _at(panel, "ret", "2024-07-08", WESTROCK_SYMBOL) == pytest.approx(
        -0.005630, rel=1e-9
    )
    assert _at(panel, "is_delisting", "2024-07-08", WESTROCK_SYMBOL) == 1.0


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

    assert _at(panel, "divCash", "2020-08-07", "AAPL") == pytest.approx(0.82)
    for day in ("2020-08-06", "2020-08-28", "2020-08-31"):
        assert _at(panel, "divCash", day, "AAPL") == pytest.approx(0.0), day

    assert _at(panel, "splitFactor", "2020-08-31", "AAPL") == pytest.approx(4.0)
    assert _at(panel, "facprc", "2020-08-31", "AAPL") == pytest.approx(4.0)
    # 1.0 on an ordinary day -- and on 08-06, the PERMNO's FIRST row in the
    # window, where there is no previous `dlycumfacpr` to divide by.
    for day in ("2020-08-06", "2020-08-07", "2020-08-28"):
        assert _at(panel, "splitFactor", day, "AAPL") == pytest.approx(1.0), day
        assert _at(panel, "facprc", day, "AAPL") == pytest.approx(1.0), day


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

    assert _at(panel, "divCash", "2020-03-02", SYNTHETIC_SYMBOL) == pytest.approx(1.7)
    assert _at(panel, "divCash", "2020-03-03", SYNTHETIC_SYMBOL) == pytest.approx(0.0)


def test_splitfactor_and_facprc_mark_the_split_day(mock_crsp_session, tmp_path):
    """The synthetic 2:1 split shows up as 2.0 in both event variables.

    `splitFactor` is DERIVED (the previous `dlycumfacpr` over today's) while
    `facprc` is CRSP's own per-day factor. Asserting both on the same row is
    what would catch a derivation that drifted off by one day.
    """
    dataset_config, days, _ = _split_series_store(tmp_path)
    panel = _panel(dataset_config)

    split_day = days[SPLIT_INDEX]
    assert _at(panel, "splitFactor", split_day, SYNTHETIC_SYMBOL) == pytest.approx(2.0)
    assert _at(panel, "facprc", split_day, SYNTHETIC_SYMBOL) == pytest.approx(2.0)
    # On a day that DID trade, `close_trade` (`dlyclose`) and `close`
    # (`abs(dlyprc)`) agree -- the two diverge only where there was no trade,
    # which is what makes `dlyprc` the right source for the panel's close.
    assert _at(panel, "close_trade", split_day, SYNTHETIC_SYMBOL) == pytest.approx(
        _at(panel, "close", split_day, SYNTHETIC_SYMBOL)
    )
    for index in (SPLIT_INDEX - 1, SPLIT_INDEX + 1):
        assert _at(panel, "splitFactor", days[index], SYNTHETIC_SYMBOL) == (
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
    series = result.sel(symbol=SYNTHETIC_SYMBOL)
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
    series = labels["ret_1"].sel(symbol=SYNTHETIC_SYMBOL).to_numpy()

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
        labels["ret_1"].sel(timestamp="2008-09-17", symbol="LEH").values
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


def _sidecar_path(dataset_config):
    from pathlib import Path

    from quantlab.dataset.crsp import CrspStockDataset

    return Path(
        str(dataset_config.zarr_file_path)
        + CrspStockDataset.ADJUSTMENT_SIDECAR_SUFFIX
    )


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


def test_the_anchor_is_the_last_non_null_close(mock_crsp_session, tmp_path):
    """The anchor is the last row WITH A PRICE, not simply the last row.

    A PERMNO whose final row in the window is a Missing-Price day would
    otherwise anchor its entire series on a null and every adjusted value
    would be NaN. The null row itself keeps a NaN `adjClose`: there is no
    price that day, and publishing the anchor's level there would invent one.
    """
    import numpy as np

    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the window's LAST row has no price.
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
    ]
    panel = _build(
        tmp_path,
        rows,
        [SYNTHETIC_PERMNO],
        start="2020-03-01",
        end="2020-03-31",
        extra_secinfo=_synthetic_secinfo(),
    )

    assert _at(panel, "adjClose", "2020-03-03", SYNTHETIC_SYMBOL) == pytest.approx(
        102.0
    )
    assert np.isnan(_at(panel, "adjClose", "2020-03-04", SYNTHETIC_SYMBOL))
    # And the rest of the series is still anchored on that close, not on NaN.
    assert _at(panel, "adjClose", "2020-03-02", SYNTHETIC_SYMBOL) == pytest.approx(
        102.0 / 1.02
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
    which is a legal CRSP return -- makes `_G` exactly 0.0 from that row onward,
    so `_G_anchor` is 0.0 too. `adjClose = _close_anchor * _G / 0.0` is then
    `inf` on every row before the loss and `0/0 -> NaN` on every row after it,
    under IEEE semantics and with no exception raised.

    No such row exists in the raw tier this phase pulled, and nothing in
    `_derivation` prevents one: the guard is being demanded here precisely
    because the defect is latent rather than observed. An infinite adjusted
    series is worse than a refusal for the same reason a zeroed one is -- `inf`
    propagates through every factor and every rank without ever raising.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [  # SYNTHETIC: the middle row is a total loss, priced.
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


def test_the_sidecar_records_the_anchor_beside_the_store(mock_crsp_session, tmp_path):
    """A store never exists without the record of WHICH anchor built it.

    `{zarr}.crsp_adjustment.json` is a SIBLING of the store, like the chunk
    ledger -- not a file inside it, which a `mode="w"` rewrite would drop and
    a zarr reader would surface as a stray array.
    """
    import json

    dataset_config = _build_store(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )

    sidecar = _sidecar_path(dataset_config)
    assert sidecar.exists(), sorted(p.name for p in tmp_path.iterdir())
    assert sidecar.parent == tmp_path, sidecar
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "start_date": ANCHOR_START,
        "end_date": ANCHOR_END,
        "product_end": "2025-12-31",
        "rule": "total_return_backward_from_last_close",
    }


def test_a_second_convert_with_the_same_anchor_resumes(mock_crsp_session, tmp_path):
    """The refusal must not fire on the honest case: the SAME window again.

    A resumed or re-run conversion reads the same anchor record it wrote, so
    every window is already in the ledger and the run is a no-op.
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


def test_extending_the_window_is_refused_naming_both_anchors(
    mock_crsp_session, tmp_path
):
    """D-08 / T-03.10-19: extending `end_date` in place would SPLICE anchors.

    Extending the window moves every still-listed PERMNO's anchor, which
    rescales every earlier adjusted value by a per-PERMNO constant. The chunk
    ledger appends windows and never rewrites finished ones, so the new
    windows would land beside old ones computed against the OLD anchor -- one
    column, two anchors, and a fabricated return at the join. Refusing BEFORE
    any write is the only place that cannot be half-done.
    """
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
    _convert(dataset_config)

    before = _panel(dataset_config)
    ledger = tmp_path / "crsp.zarr.chunks.json"
    ledger_before = ledger.read_text(encoding="utf-8")

    extended = _dataset_config(
        tmp_path, cfg, reference_dir, start=ANCHOR_START, end="2020-12-31"
    )
    with pytest.raises(ValueError) as raised:
        _convert(extended)

    message = str(raised.value)
    assert ANCHOR_END in message, message
    assert "2020-12-31" in message, message
    assert "crsp_adjustment" in message, message
    # The remedy, not only the diagnosis.
    assert "zarr_file_path" in message, message

    import xarray as xr

    xr.testing.assert_identical(before, _panel(dataset_config))
    assert ledger.read_text(encoding="utf-8") == ledger_before


def test_a_store_without_the_sidecar_is_refused(mock_crsp_session, tmp_path):
    """No sidecar, no proof of which anchor the store holds -- so no append.

    A store written before this record existed, or one whose sidecar was
    deleted, is indistinguishable from a store built against a different
    anchor. Guessing "probably the same one" is exactly the assumption the
    sidecar exists to stop being an assumption.
    """
    dataset_config = _build_store(
        tmp_path,
        _anchor_series_rows(),
        [SYNTHETIC_PERMNO],
        start=ANCHOR_START,
        end=ANCHOR_END,
        extra_secinfo=_synthetic_secinfo(),
    )
    sidecar = _sidecar_path(dataset_config)
    sidecar.unlink()

    with pytest.raises(ValueError) as raised:
        _convert(dataset_config)

    message = str(raised.value)
    assert sidecar.name in message, message
