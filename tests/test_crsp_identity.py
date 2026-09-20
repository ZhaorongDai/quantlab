"""CRSP panel IDENTITY: who is in it, who owns a ticker, where a column changes
company, and the QQQ benchmark store that sits outside all of it.

`tests/test_crsp_dataset.py` is about what the NUMBERS mean. This module is
about WHICH SECURITY each number belongs to -- the four identity decisions of
phase 03.10:

- the common-stock filter and its report (D-06, D-17): which securities the
  equity panel holds, evaluated per DATE off the `dsf_v2` per-day type columns,
  with a delisting row inheriting its PERMNO's previous verdict so the single
  most consequential row a dead security has can never be filtered away;
- same-day ticker collisions (D-04): two PERMNOs heading for one column are
  resolved by stated rule -- active over delisting, then universe member -- or
  refused, never merged;
- PERMNO seams (D-18): where a ticker column changes company the incoming
  PERMNO's first row gets NaN adjusted values, so no return and no factor
  window spans two companies. A same-PERMNO rename (FB -> META) is NOT a seam;
- the QQQ benchmark store (D-15) built by `CrspDatasetConfig.qqq_benchmark`,
  which is DATA ONLY (D-16): nothing here touches the backtester's
  `benchmark_dataset`, which still raises `NotImplementedError`.

**Every quantlab import is INSIDE a test or helper body.** These tests are
written before the names they assert on exist, and a module-scope import would
turn the RED run into a collection error -- zero tests discovered, which proves
nothing about the behaviour (TDD gate #3770).

**Provenance rule, inherited from `tests/crsp_fixtures.py`.** A row reused from
that module is VERBATIM live data; a row built here with `dsf_row` is invented
and carries a `# SYNTHETIC` comment.
"""

from __future__ import annotations

from datetime import date

import pytest

# ---------------------------------------------------------------------------
# The security-type scenarios (SYNTHETIC)
# ---------------------------------------------------------------------------
#
# One PERMNO per type combination the live check found among S&P members since
# 2000 (`03.10-LIVE-CHECK-2.json` key `L11_1`), plus QQQ's live FUND/ETF shape.
# The PERMNO digits are invented; the five TYPE codes of each row are the live
# combination it stands for, which is the only thing the filter reads.
#
# (permno, symbol, sharetype, securitytype, securitysubtype, issuertype,
#  usincflg, primaryexch)

TYPE_SCENARIOS: tuple[tuple[str, ...], ...] = (
    # L11 1020 rows: ordinary US common stock. Kept by EVERY preset.
    ("10107", "MSFT", "NS", "EQTY", "COM", "CORP", "Y", "Q"),
    # L11 60 rows: common stock of a non-US-incorporated issuer.
    ("13901", "NONUS", "NS", "EQTY", "COM", "CORP", "N", "Q"),
    # L11 35 rows: a REIT with an ordinary share type.
    ("17144", "REITNS", "NS", "EQTY", "COM", "REIT", "Y", "N"),
    # L11 8 rows: a REIT whose share type is SB.
    ("22779", "REITSB", "SB", "EQTY", "COM", "REIT", "Y", "N"),
    # L11 83 rows: unknown types. Dropped by the default preset.
    ("35107", "UNKWN", "N/A", "N/A", "UNK", "CORP", "N", "Q"),
    # L11 1 row: an ADR. Satisfies EQTY/COM, which is why the default preset
    # also carries a `sharetype` allow-list (the D-17 reading).
    ("47896", "ADRCO", "AD", "EQTY", "COM", "CORP", "N", "Q"),
    # L11 1 row: a unit. Same shape as the ADR.
    ("58123", "UNITCO", "UG", "EQTY", "COM", "CORP", "N", "A"),
    # QQQ's live type combination (`03.10-LIVE-CHECK-NDX-QQQ.json` key
    # `C3_qqq_daily_sample`), on SYNTHETIC 2020 dates so one small window
    # carries every combination at once.
    ("86755", "QQQ", "NS", "FUND", "ETF", "ACOR", "Y", "Q"),
)

#: What the default `equity_common` preset keeps out of `TYPE_SCENARIOS`.
EQUITY_COMMON_KEPT = ["MSFT", "NONUS", "REITNS", "REITSB"]

#: What `shrcd_10_11` keeps: US-incorporated corporate NS common only.
SHRCD_10_11_KEPT = ["MSFT"]

#: The three days every scenario PERMNO trades on.
SCENARIO_DAYS = ("2020-03-02", "2020-03-03", "2020-03-04")
SCENARIO_START = "2020-03-01"
SCENARIO_END = "2020-03-31"

LEHMAN_PERMNO = "80599"
AAPL_PERMNO = "14593"
QQQ_PERMNO_TEXT = "86755"


# ---------------------------------------------------------------------------
# Helpers: raw tier -> reference tier -> converted store
# ---------------------------------------------------------------------------


def _scenario_rows():
    """Three ordinary daily rows per `TYPE_SCENARIOS` entry. SYNTHETIC."""
    from tests.crsp_fixtures import dsf_row

    rows = []
    for index, scenario in enumerate(TYPE_SCENARIOS):
        (
            permno,
            symbol,
            sharetype,
            securitytype,
            securitysubtype,
            issuertype,
            usincflg,
            primaryexch,
        ) = scenario
        price = 20.0 + index
        for day_index, day in enumerate(SCENARIO_DAYS):
            daily_return = round(0.01 + 0.001 * index - 0.004 * day_index, 6)
            price *= 1.0 + daily_return
            rows.append(
                dsf_row(
                    permno,
                    day,
                    sharetype=sharetype,
                    securitytype=securitytype,
                    securitysubtype=securitysubtype,
                    issuertype=issuertype,
                    usincflg=usincflg,
                    primaryexch=primaryexch,
                    dlyprc=f"{price:.6f}",
                    dlyclose=f"{price:.6f}",
                    dlyret=f"{daily_return:.6f}",
                    dlyretx=f"{daily_return:.6f}",
                    ticker=symbol,
                )
            )
    return rows


def _scenario_secinfo():
    """One wide security-info interval per scenario PERMNO.

    QQQ is EXCLUDED: `tests/crsp_fixtures.py` already carries its three live
    ticker intervals, and a second overlapping interval for the same PERMNO
    would be a fixture artefact rather than a CRSP shape.
    """
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(int(permno), "2000-01-01", "2025-12-31", symbol, symbol, None)
        for permno, symbol, *_ in TYPE_SCENARIOS
        if permno != QQQ_PERMNO_TEXT
    ]


def _scenario_permnos():
    return [permno for permno, *_ in TYPE_SCENARIOS]


def _reference_rows(extra_secinfo=(), dsp500_rows=None):
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
        "crsp_a_indexes.dsp500list_v2": (
            list(DSP500_ROWS) if dsp500_rows is None else list(dsp500_rows)
        ),
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
    dsp500_rows=None,
):
    """Serve `rows` through the fake session, land the raw + reference tiers."""
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
        _reference_rows(extra_secinfo, dsp500_rows),
        product_end=product_end,
    )
    return cfg, str(reference_dir)


def _dataset_config(
    tmp_path, cfg, reference_dir, *, start, end, store="crsp.zarr", **overrides
):
    from quantlab.base.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / store),
        raw_data_dir_path=cfg.raw_data_dir_path,
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(reference_dir),
        start_date=start,
        end_date=end,
        **overrides,
    )


def _bare_config(tmp_path, **overrides):
    """A `CrspDatasetConfig` over paths that need not exist.

    Only the config SETTER runs against it -- which is exactly where the filter
    is validated, so a malformed filter must fail here, long before a raw tier
    is read.
    """
    from quantlab.base.config import CrspDatasetConfig

    return CrspDatasetConfig(
        zarr_file_path=str(tmp_path / "crsp.zarr"),
        raw_data_dir_path=str(tmp_path / "raw"),
        catalog_path=str(tmp_path / "catalog"),
        reference_dir=str(tmp_path / "_reference"),
        start_date="2020-01-01",
        end_date="2020-12-31",
        **overrides,
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


def _symbols(panel):
    return [str(value) for value in panel["symbol"].values]


def _timestamps(panel):
    return [str(value)[:10] for value in panel["timestamp"].values]


def _at(panel, variable, day, symbol):
    return float(panel[variable].sel(timestamp=day, symbol=symbol).values)


def _build_store(
    tmp_path,
    rows,
    permnos,
    *,
    start,
    end,
    extra_secinfo=(),
    dsp500_rows=None,
    product_end="2025-12-31",
    granularity="year",
    store="crsp.zarr",
    **config_overrides,
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
        dsp500_rows=dsp500_rows,
    )
    dataset_config = _dataset_config(
        tmp_path,
        cfg,
        reference_dir,
        start=start,
        end=end,
        store=store,
        **config_overrides,
    )
    _convert(dataset_config, granularity=granularity)
    return dataset_config


def _scenario_store(tmp_path, **config_overrides):
    """Every `TYPE_SCENARIOS` PERMNO converted into one store."""
    return _build_store(
        tmp_path,
        _scenario_rows(),
        _scenario_permnos(),
        start=SCENARIO_START,
        end=SCENARIO_END,
        extra_secinfo=_scenario_secinfo(),
        **config_overrides,
    )


def _read_json(path):
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _filter_report(dataset_config):
    from pathlib import Path

    from quantlab.dataset.crsp import FILTER_REPORT_SUFFIX

    return _read_json(
        Path(str(dataset_config.zarr_file_path) + FILTER_REPORT_SUFFIX)
    )


def _symbology_report(dataset_config):
    from pathlib import Path

    from quantlab.dataset.crsp import SYMBOLOGY_REPORT_SUFFIX

    return _read_json(
        Path(str(dataset_config.zarr_file_path) + SYMBOLOGY_REPORT_SUFFIX)
    )


# ---------------------------------------------------------------------------
# Task 1: the security filter, per date, with a delisting carry and a report
# ---------------------------------------------------------------------------


def test_the_default_filter_keeps_equity_common_and_drops_the_rest(
    mock_crsp_session, tmp_path
):
    """D-17: `equity_common` keeps REITs (SB included) and non-US issuers, and
    drops ADRs, units, funds/ETFs and unknown types.

    The ADR and the unit satisfy `securitytype='EQTY' AND
    securitysubtype='COM'` on the live rows, so the two-column predicate alone
    would keep them. The preset therefore also carries a `sharetype`
    allow-list -- every ShareType code CRSP defines except `AD` and `UG` --
    which is what delivers exactly the drops D-17 states.
    """
    panel = _panel(_scenario_store(tmp_path))

    assert _symbols(panel) == sorted(EQUITY_COMMON_KEPT), _symbols(panel)


def test_the_shrcd_10_11_preset_keeps_only_us_corporate_common(
    mock_crsp_session, tmp_path
):
    """The legacy `shrcd in (10, 11)` replication, available by config.

    It adds `usincflg='Y'` and `issuertype IN ('ACOR','CORP')` to the default,
    which is precisely why it is NOT the default: it drops the REITs and the
    non-US-incorporated issuers that are legitimate index members (RESEARCH
    Q3, Pitfall 5).
    """
    panel = _panel(_scenario_store(tmp_path, security_filter="shrcd_10_11"))

    assert _symbols(panel) == SHRCD_10_11_KEPT, _symbols(panel)


def test_the_none_filter_keeps_every_security_type(mock_crsp_session, tmp_path):
    """`none` is the empty predicate: the panel holds the raw tier's roster.

    This is the preset `CrspDatasetConfig.qqq_benchmark` uses (D-15), and the
    escape hatch for anyone who wants CRSP's own universe unfiltered.
    """
    panel = _panel(_scenario_store(tmp_path, security_filter="none"))

    assert _symbols(panel) == sorted(
        symbol for _permno, symbol, *_ in TYPE_SCENARIOS
    ), _symbols(panel)


def test_a_dict_filter_selects_on_any_filterable_column(
    mock_crsp_session, tmp_path
):
    """A user-supplied `{column: allowed values}` replaces the preset entirely.

    `primaryexch` is deliberately NOT in any preset -- it changes over a
    security's life, so filtering on it punches holes in a series. It is still
    FILTERABLE, because a user who wants an NYSE-only panel is entitled to one
    as long as the choice is theirs and the report says what it cost.
    """
    panel = _panel(_scenario_store(tmp_path, security_filter={"primaryexch": ["N"]}))

    assert _symbols(panel) == ["REITNS", "REITSB"], _symbols(panel)


def test_a_non_filterable_column_is_refused_naming_the_filterable_ones(
    mock_crsp_session, tmp_path
):
    """T-03.10-28: the filter's key space is closed.

    `ticker` is a real `dsf_v2` column but not a TYPE column; filtering on it
    would silently build a ticker-picked panel that looks like a type-filtered
    one. The refusal names the offending key and the columns that are allowed.
    """
    from quantlab.dataset.crsp import FILTERABLE_COLUMNS, CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(_bare_config(tmp_path, security_filter={"ticker": ["X"]}))

    message = str(raised.value)
    assert "ticker" in message, message
    assert "securitytype" in message, message
    assert all(name in message for name in FILTERABLE_COLUMNS), message


def test_an_empty_allow_list_is_refused_by_the_filter(mock_crsp_session, tmp_path):
    """An empty allow-list matches nothing, so it would silently empty the
    panel. That is a typo's shape, not an intention's."""
    from quantlab.dataset.crsp import CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(
            _bare_config(tmp_path, security_filter={"securitytype": []})
        )

    message = str(raised.value)
    assert "securitytype" in message, message


def test_an_unknown_preset_is_refused_listing_the_presets(
    mock_crsp_session, tmp_path
):
    """A misspelt preset must not fall back to 'keep everything' or to the
    default -- both would be a silently different panel."""
    from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset

    with pytest.raises(ValueError) as raised:
        CrspStockDataset(_bare_config(tmp_path, security_filter="common_stock"))

    message = str(raised.value)
    assert "common_stock" in message, message
    assert all(name in message for name in SECURITY_FILTER_PRESETS), message


def test_the_filter_is_evaluated_per_date(mock_crsp_session, tmp_path):
    """D-17: the verdict is the security-info interval valid THAT DAY.

    A closed-end fund that used to be common stock keeps its common-stock era
    and loses the rest. Evaluating the filter once per PERMNO -- on its first
    or its last row -- would either keep the whole fund era or throw away the
    common-stock era with it.
    """
    from tests.crsp_fixtures import dsf_row, secinfo_row

    permno, symbol = "61241", "WASCOM"
    rows = [  # SYNTHETIC: COM through 2020-06-30, CEF from 2020-07-01.
        dsf_row(
            permno,
            day,
            securitysubtype="COM" if day <= "2020-06-30" else "CEF",
            dlyprc="30.000000",
            dlyclose="30.000000",
            dlyret="0.010000",
            ticker=symbol,
        )
        for day in ("2020-06-29", "2020-06-30", "2020-07-01", "2020-07-02")
    ]
    dataset_config = _build_store(
        tmp_path,
        rows,
        [permno],
        start="2020-06-01",
        end="2020-07-31",
        extra_secinfo=[
            secinfo_row(int(permno), "2000-01-01", "2025-12-31", symbol, symbol, None)
        ],
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == [symbol], _symbols(panel)
    assert _timestamps(panel) == ["2020-06-29", "2020-06-30"], _timestamps(panel)


def test_a_delisting_row_inherits_the_previous_verdict(mock_crsp_session, tmp_path):
    """D-10 + D-17: the filter may never eat a delisting return.

    Lehman's 2008-09-18 row is the delisting row, and a delisted security's
    last row is exactly where CRSP's type columns go blank. Filtering it on its
    OWN types would drop the -60% day and restore survivorship bias through the
    filter after symbology's carry rule had just rescued it from the ticker.
    """
    from tests.crsp_fixtures import LEHMAN_2008_ROWS

    rows = []
    for row in LEHMAN_2008_ROWS:
        row = dict(row)
        if row["dlycaldt"] == "2008-09-18":  # SYNTHETIC type blanking.
            row["securitytype"] = "N/A"
            row["securitysubtype"] = "UNK"
            row["sharetype"] = "N/A"
        rows.append(row)

    panel = _panel(
        _build_store(
            tmp_path,
            rows,
            [LEHMAN_PERMNO],
            start="2008-09-12",
            end="2008-09-30",
        )
    )

    assert _symbols(panel) == ["LEH"], _symbols(panel)
    assert "2008-09-18" in _timestamps(panel), _timestamps(panel)
    assert _at(panel, "ret", "2008-09-18", "LEH") == pytest.approx(-0.6)
    assert _at(panel, "is_delisting", "2008-09-18", "LEH") == pytest.approx(1.0)


def test_the_filter_report_says_what_was_dropped_and_why(
    mock_crsp_session, tmp_path
):
    """D-17 / T-03.10-27: universe shrinkage is never silent.

    The sidecar carries the RESOLVED filter (so a preset name is not the only
    record of what ran), the row arithmetic, a count per dropped type
    combination, and a line per dropped PERMNO with its last symbol -- which is
    what makes "the S&P panel lost its ADR member" a readable fact rather than
    a missing column nobody notices.
    """
    dataset_config = _scenario_store(tmp_path)

    report = _filter_report(dataset_config)

    assert report["filter"]["requested"] == "equity_common"
    assert report["filter"]["resolved"]["sharetype"] == ["NS", "SB", "CE"]
    assert report["rows_total"] == report["rows_kept"] + report["rows_dropped"]
    assert report["rows_total"] == len(TYPE_SCENARIOS) * len(SCENARIO_DAYS)
    assert report["rows_kept"] == len(EQUITY_COMMON_KEPT) * len(SCENARIO_DAYS)

    assert set(report["dropped_by_type"]) == {
        "N/A/N/A/UNK/CORP/N",
        "AD/EQTY/COM/CORP/N",
        "UG/EQTY/COM/CORP/N",
        "NS/FUND/ETF/ACOR/Y",
    }, report["dropped_by_type"]
    assert report["dropped_by_type"]["NS/FUND/ETF/ACOR/Y"] == len(SCENARIO_DAYS)

    qqq = report["dropped_permnos"][QQQ_PERMNO_TEXT]
    assert qqq["symbol"] == "QQQ", qqq
    assert qqq["types"] == ["NS/FUND/ETF/ACOR/Y"], qqq
    assert qqq["rows"] == len(SCENARIO_DAYS), qqq
    assert qqq["first"] == SCENARIO_DAYS[0], qqq
    assert qqq["last"] == SCENARIO_DAYS[-1], qqq


# ---------------------------------------------------------------------------
# Task 2: collisions, PERMNO seams and the symbology report
# ---------------------------------------------------------------------------

#: The reused ticker, and the two PERMNOs that wear it in turn. SYNTHETIC
#: throughout: the live check sampled no ticker-reuse pair, and inventing the
#: DATES is the whole point of the scenario.
REUSE_SYMBOL = "XYZ"
OLD_PERMNO = 11111
NEW_PERMNO = 22222
REUSE_SEAM_DAY = "2010-05-18"

#: The two PERMNOs that are BOTH active under one ticker on one day.
TIE_MEMBER_PERMNO = 33333
TIE_OUTSIDER_PERMNO = 44444
TIE_DAY = "2011-01-03"


def _reuse_rows(*, new_permno_first_day=REUSE_SEAM_DAY):
    """SYNTHETIC: `OLD_PERMNO` delists as XYZ, `NEW_PERMNO` takes the ticker.

    `new_permno_first_day` moves the incoming security's first row onto the
    delisting day itself, which turns the scenario from a SEAM into a same-day
    COLLISION -- the two halves of ticker reuse, from the same fixture.
    """
    from tests.crsp_fixtures import dsf_row

    rows = [
        dsf_row(
            OLD_PERMNO,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyopen=f"{price * 0.99:.6f}",
            dlyhigh=f"{price * 1.02:.6f}",
            dlylow=f"{price * 0.98:.6f}",
            dlyret="0.010000",
            dlyretx="0.010000",
            ticker=REUSE_SYMBOL,
        )
        for day, price in (
            ("2010-05-12", 10.0),
            ("2010-05-13", 10.1),
            ("2010-05-14", 10.201),
        )
    ]
    rows.append(
        dsf_row(
            OLD_PERMNO,
            "2010-05-17",
            dlydelflg="Y",
            dlyprc="4.080400",
            dlyprcflg="DP",
            dlyret="-0.600000",
            ticker=None,
        )
    )
    rows.extend(
        dsf_row(
            NEW_PERMNO,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyopen=f"{price * 0.99:.6f}",
            dlyhigh=f"{price * 1.02:.6f}",
            dlylow=f"{price * 0.98:.6f}",
            dlyret="0.020000",
            dlyretx="0.020000",
            ticker=REUSE_SYMBOL,
        )
        for day, price in (
            (new_permno_first_day, 50.0),
            ("2010-05-19", 51.0),
            ("2010-05-20", 52.02),
        )
    )
    return rows


def _reuse_secinfo(*, new_permno_start=REUSE_SEAM_DAY):
    """One interval each. The old PERMNO's ENDS before the delisting row, which
    is why that row reaches the panel only through symbology's carry rule."""
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(OLD_PERMNO, "2000-01-01", "2010-05-14", REUSE_SYMBOL,
                    REUSE_SYMBOL, None),
        secinfo_row(NEW_PERMNO, new_permno_start, "2025-12-31", REUSE_SYMBOL,
                    REUSE_SYMBOL, None),
    ]


def _reuse_store(tmp_path, **config_overrides):
    return _build_store(
        tmp_path,
        _reuse_rows(),
        [str(OLD_PERMNO), str(NEW_PERMNO)],
        start="2010-05-01",
        end="2010-05-31",
        extra_secinfo=_reuse_secinfo(),
        **config_overrides,
    )


def _tie_rows():
    """SYNTHETIC: two ACTIVE PERMNOs under one ticker on the same days."""
    from tests.crsp_fixtures import dsf_row

    return [
        dsf_row(
            permno,
            day,
            dlyprc=f"{price:.6f}",
            dlyclose=f"{price:.6f}",
            dlyret="0.010000",
            ticker=REUSE_SYMBOL,
        )
        for permno, price in (
            (TIE_MEMBER_PERMNO, 30.0),
            (TIE_OUTSIDER_PERMNO, 70.0),
        )
        for day in (TIE_DAY, "2011-01-04")
    ]


def _tie_secinfo():
    from tests.crsp_fixtures import secinfo_row

    return [
        secinfo_row(permno, "2000-01-01", "2025-12-31", REUSE_SYMBOL,
                    REUSE_SYMBOL, None)
        for permno in (TIE_MEMBER_PERMNO, TIE_OUTSIDER_PERMNO)
    ]


def _tie_dsp500_rows():
    """`dsp500list_v2` covering ONLY the member of the colliding pair."""
    return [
        {
            "permno": str(TIE_MEMBER_PERMNO),
            "indno": "1000500",
            "mbrstartdt": "2000-01-01",
            "mbrenddt": "2025-12-31",
            "mbrflg": "NORM",
            "indfam": "1100500",
        }
    ]


def test_a_permno_seam_nans_the_incoming_first_row(mock_crsp_session, tmp_path):
    """D-18 / T-03.10-26: a ticker column that changes company breaks there.

    Without the break, `adjClose(2010-05-18) / adjClose(2010-05-17)` would be a
    finite number computed from two DIFFERENT companies' anchors -- a return
    that never happened, indistinguishable from a real one, and silently
    inherited by every rolling factor and forward label spanning the day.

    Raw prices and `permno` are untouched: the seam removes the FABRICATED
    quantity, not the observation.
    """
    import numpy as np

    panel = _panel(_reuse_store(tmp_path))

    assert _symbols(panel) == [REUSE_SYMBOL], _symbols(panel)
    assert _at(panel, "permno", "2010-05-17", REUSE_SYMBOL) == pytest.approx(
        float(OLD_PERMNO)
    )
    assert _at(panel, "permno", REUSE_SEAM_DAY, REUSE_SYMBOL) == pytest.approx(
        float(NEW_PERMNO)
    )

    for name in ("adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"):
        assert np.isnan(_at(panel, name, REUSE_SEAM_DAY, REUSE_SYMBOL)), name
    assert np.isfinite(_at(panel, "close", REUSE_SEAM_DAY, REUSE_SYMBOL))

    # The day AFTER the seam is an ordinary adjusted day again: the break is
    # one row wide, not the end of the column.
    assert np.isfinite(_at(panel, "adjClose", "2010-05-19", REUSE_SYMBOL))


def test_the_seam_is_reported_whether_or_not_it_is_nanned(
    mock_crsp_session, tmp_path
):
    """The opt-out removes the NaN, never the RECORD.

    A user who wants a continuous column (say, to study the ticker rather than
    the company) may have one, but the panel must never be able to hide that
    the column changed hands.
    """
    import numpy as np

    expected = [
        {
            "date": REUSE_SEAM_DAY,
            "symbol": REUSE_SYMBOL,
            "old_permno": OLD_PERMNO,
            "new_permno": NEW_PERMNO,
        }
    ]

    on = _reuse_store(tmp_path)
    assert _symbology_report(on)["seams"] == expected

    off = _reuse_store(
        tmp_path, nan_adj_at_permno_seam=False, store="crsp_continuous.zarr"
    )
    assert _symbology_report(off)["seams"] == expected
    assert np.isfinite(
        _at(_panel(off), "adjClose", REUSE_SEAM_DAY, REUSE_SYMBOL)
    )


def test_a_same_permno_rename_is_not_a_seam(mock_crsp_session, tmp_path):
    """FB -> META is ONE company, so the adjusted series must not break.

    PERMNO 13407 throughout (VERBATIM `C5` security-info intervals). The two
    symbols are two COLUMNS of the panel, and the ratio across them is a real
    return -- which is exactly what a PERMNO-keyed seam rule has to get right,
    because a ticker-keyed one could not tell this case from ticker reuse.
    """
    from tests.crsp_fixtures import dsf_row

    days = ("2022-06-06", "2022-06-07", "2022-06-08", "2022-06-09", "2022-06-10")
    daily_return = 0.02
    rows = []
    price = 180.0
    for day in days:  # SYNTHETIC daily rows; the secinfo intervals are live.
        price *= 1.0 + daily_return
        rows.append(
            dsf_row(
                13407,
                day,
                dlyprc=f"{price:.6f}",
                dlyclose=f"{price:.6f}",
                dlyret=f"{daily_return:.6f}",
                dlyretx=f"{daily_return:.6f}",
                ticker="FB" if day <= "2022-06-08" else "META",
            )
        )

    dataset_config = _build_store(
        tmp_path, rows, ["13407"], start="2022-06-01", end="2022-06-30"
    )
    panel = _panel(dataset_config)

    assert _symbols(panel) == ["FB", "META"], _symbols(panel)
    assert _symbology_report(dataset_config)["seams"] == []
    assert _at(panel, "adjClose", "2022-06-09", "META") / _at(
        panel, "adjClose", "2022-06-08", "FB"
    ) == pytest.approx(1.0 + daily_return, rel=1e-9)


def test_a_collision_is_broken_by_the_configured_universe(
    mock_crsp_session, tmp_path
):
    """D-04 rule 2: when both securities are trading, the universe decides.

    The panel is being built FOR that universe, so its member is the one whose
    prices the panel is about. Without a universe there is no such fact, which
    is why `collision_universe` is a config field and not a default.
    """
    dataset_config = _build_store(
        tmp_path,
        _tie_rows(),
        [str(TIE_MEMBER_PERMNO), str(TIE_OUTSIDER_PERMNO)],
        start="2011-01-01",
        end="2011-01-31",
        extra_secinfo=_tie_secinfo(),
        dsp500_rows=_tie_dsp500_rows(),
        collision_universe="crsp_sp500",
    )
    panel = _panel(dataset_config)

    assert _at(panel, "permno", TIE_DAY, REUSE_SYMBOL) == pytest.approx(
        float(TIE_MEMBER_PERMNO)
    )

    resolutions = _symbology_report(dataset_config)["collisions"]
    assert [record["rule"] for record in resolutions] == ["universe_member"] * 2
    first = resolutions[0]
    assert first["date"] == TIE_DAY, first
    assert first["symbol"] == REUSE_SYMBOL, first
    assert first["kept"] == TIE_MEMBER_PERMNO, first
    assert first["dropped"] == [TIE_OUTSIDER_PERMNO], first


def test_an_unresolvable_collision_refuses_before_anything_is_written(
    mock_crsp_session, tmp_path
):
    """T-03.10-16: two securities are never merged into one column.

    Without a universe, nothing distinguishes the two active PERMNOs. Picking
    one by row order, or averaging them, would leave a well-formed panel in
    which every return across the join is fabricated. The refusal happens in
    the once-per-run axes hook, so the store does not exist afterwards.
    """
    from pathlib import Path

    cfg, reference_dir = _pull(
        tmp_path,
        _tie_rows(),
        [str(TIE_MEMBER_PERMNO), str(TIE_OUTSIDER_PERMNO)],
        start="2011-01-01",
        end="2011-01-31",
        extra_secinfo=_tie_secinfo(),
        dsp500_rows=_tie_dsp500_rows(),
    )
    dataset_config = _dataset_config(
        tmp_path, cfg, reference_dir, start="2011-01-01", end="2011-01-31"
    )

    with pytest.raises(ValueError) as raised:
        _convert(dataset_config)

    message = str(raised.value)
    assert TIE_DAY in message, message
    assert REUSE_SYMBOL in message, message
    assert str(TIE_MEMBER_PERMNO) in message, message
    assert str(TIE_OUTSIDER_PERMNO) in message, message
    assert not Path(dataset_config.zarr_file_path).exists(), message


def test_a_collision_between_a_delisting_and_an_active_row_keeps_the_active(
    mock_crsp_session, tmp_path
):
    """D-04 rule 1, the shape ticker reuse actually takes.

    The outgoing security's delisting row and the incoming security's first row
    land on the SAME day. The ticker belongs to whoever is still trading under
    it, and the rule needs no universe -- which is what keeps ordinary ticker
    reuse from requiring one.
    """
    dataset_config = _build_store(
        tmp_path,
        _reuse_rows(new_permno_first_day="2010-05-17"),
        [str(OLD_PERMNO), str(NEW_PERMNO)],
        start="2010-05-01",
        end="2010-05-31",
        extra_secinfo=_reuse_secinfo(new_permno_start="2010-05-17"),
    )
    panel = _panel(dataset_config)

    assert _at(panel, "permno", "2010-05-17", REUSE_SYMBOL) == pytest.approx(
        float(NEW_PERMNO)
    )

    resolutions = _symbology_report(dataset_config)["collisions"]
    assert len(resolutions) == 1, resolutions
    assert resolutions[0] == {
        "date": "2010-05-17",
        "symbol": REUSE_SYMBOL,
        "kept": NEW_PERMNO,
        "dropped": [OLD_PERMNO],
        "rule": "active_over_delisting",
    }


def test_the_symbology_report_carries_every_identity_key(
    mock_crsp_session, tmp_path
):
    """One sidecar holds every identity decision, not five scattered logs.

    `delisting_carried` is the one to read twice: it says the panel's XYZ
    column owes its 2010-05-17 row to the carry rule rather than to an
    interval, which is the difference between a -60% day in the series and a
    -60% day that quietly never happened.
    """
    report = _symbology_report(_reuse_store(tmp_path))

    assert set(report) >= {
        "seams",
        "collisions",
        "unlabelled",
        "delisting_carried",
        "class_suffixed",
        "nonconforming_symbols",
    }, sorted(report)
    carried = report["delisting_carried"][str(OLD_PERMNO)]
    assert carried["symbol"] == REUSE_SYMBOL, carried
    assert carried["rows"] == 1, carried
    assert report["unlabelled"] == {}, report["unlabelled"]
    assert report["nonconforming_symbols"] == [], report["nonconforming_symbols"]
