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


def _build(
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
    """Pull, convert and open -- the common three-line preamble."""
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
    return _panel(dataset_config)


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
