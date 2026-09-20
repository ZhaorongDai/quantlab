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
