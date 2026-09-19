"""Read-only WRDS check: historical Nasdaq-100 constituents + QQQ ETF data (one connection = one Duo push).

Usage: WRDS_USERNAME=<you> uv run python <this file>
"""

import json
import os
import sys
import time
from pathlib import Path

import psycopg2

OUT = Path(__file__).with_name("ndx_qqq_check.json")
results: dict = {}


def run(conn, key, sql, limit_rows=200):
    t0 = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchmany(limit_rows)
            cols = [d[0] for d in cur.description]
        results[key] = {"ok": True, "cols": cols, "rows": [list(map(str, r)) for r in rows], "secs": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        conn.rollback() if not conn.autocommit else None
        results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "secs": round(time.time() - t0, 2)}
    print(f"[{key}] {'ok' if results[key]['ok'] else 'FAILED'} ({results[key]['secs']}s)", flush=True)
    return results[key]


def main() -> None:
    user = os.environ.get("WRDS_USERNAME") or sys.exit("Set WRDS_USERNAME first")
    print("Connecting — approve the Duo push on your phone if prompted...", flush=True)
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require", connect_timeout=60)
    conn.set_session(readonly=True, autocommit=True)

    # --- A. Any Nasdaq index family / series inside CRSP index files?
    run(conn, "A1_crsp_index_families",
        "SELECT * FROM crsp_a_indexes.indfamilyinfohdr_ind ORDER BY 1", limit_rows=300)
    run(conn, "A2_crsp_index_series_nasdaq",
        "SELECT * FROM crsp_a_indexes.indseriesinfohdr_ind WHERE indname ILIKE '%nasdaq%' OR indname ILIKE '%ndx%' "
        "OR indname ILIKE '%100%'", limit_rows=100)
    run(conn, "A3_idx_const_views", "SELECT table_schema, table_name, column_name FROM information_schema.columns "
        "WHERE table_name ILIKE 'idx_const%' ORDER BY 1,2,ordinal_position", limit_rows=300)
    run(conn, "A4_idx_const_indexes", "SELECT DISTINCT indno FROM crsp.idx_const_close_v2 LIMIT 50")

    # --- B. Compustat index constituents (Nasdaq-100 history lives here if entitled)
    run(conn, "B1_comp_schemas", "SELECT nspname, has_schema_privilege(nspname,'USAGE') FROM pg_namespace "
        "WHERE nspname IN ('comp','compa','comp_na_daily_all','compd','compm','compq','comp_global_daily') ORDER BY 1")
    run(conn, "B2_comp_idx_tables", "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_name ILIKE 'idx%' AND table_schema ILIKE 'comp%' ORDER BY 1,2", limit_rows=100)
    run(conn, "B3_comp_ndx_index", "SELECT * FROM comp.idx_index WHERE conm ILIKE '%nasdaq%100%' OR conm ILIKE '%nasdaq 100%' "
        "OR tic ILIKE 'NDX%' OR tic ILIKE 'I:NDX%'", limit_rows=50)
    run(conn, "B4_comp_ndx_members",
        "SELECT h.gvkeyx, count(*) AS n_spells, count(DISTINCT h.gvkey) AS n_companies, min(h.\"from\"), max(h.\"from\"), "
        "count(*) FILTER (WHERE h.thru IS NULL) AS open_spells "
        "FROM comp.idxcst_his h JOIN comp.idx_index i ON i.gvkeyx=h.gvkeyx "
        "WHERE i.conm ILIKE '%nasdaq%100%' GROUP BY 1", limit_rows=20)

    # --- C. QQQ (and QQQQ 2004-2011) in the CRSP stock file
    run(conn, "C1_qqq_names", "SELECT permno, permco, namedt, nameenddt, ticker, issuernm, sharetype, securitytype, "
        "securitysubtype, primaryexch FROM crsp_a_stock.stocknames_v2 WHERE ticker IN ('QQQ','QQQQ') ORDER BY namedt")
    run(conn, "C2_qqq_daily_range",
        "SELECT d.permno, min(d.dlycaldt), max(d.dlycaldt), count(*) FROM crsp_a_stock.stkdlysecuritydata d "
        "WHERE d.permno IN (SELECT permno FROM crsp_a_stock.stocknames_v2 WHERE ticker IN ('QQQ','QQQQ')) GROUP BY 1")
    run(conn, "C3_qqq_daily_sample",
        "SELECT * FROM crsp_a_stock.dsf_v2 WHERE permno IN (SELECT permno FROM crsp_a_stock.stocknames_v2 "
        "WHERE ticker IN ('QQQ','QQQQ')) AND dlycaldt IN ('1999-03-10','2010-06-01','2025-12-31')", limit_rows=10)

    # --- D. QQQ holdings in CRSP Mutual Funds (quarterly update schema is entitled)
    run(conn, "D1_mf_tables", "SELECT table_name FROM information_schema.tables WHERE table_schema='crsp_q_mutualfunds' "
        "ORDER BY 1", limit_rows=200)
    run(conn, "D2_qqq_fund", "SELECT * FROM crsp_q_mutualfunds.fund_hdr WHERE ticker IN ('QQQ','QQQQ')", limit_rows=20)
    run(conn, "D3_qqq_fund_hist", "SELECT * FROM crsp_q_mutualfunds.fund_hdr_hist WHERE ticker IN ('QQQ','QQQQ') ORDER BY 1", limit_rows=40)
    run(conn, "D4_qqq_portno",
        "SELECT * FROM crsp_q_mutualfunds.portnomap WHERE crsp_fundno IN "
        "(SELECT crsp_fundno FROM crsp_q_mutualfunds.fund_hdr_hist WHERE ticker IN ('QQQ','QQQQ'))", limit_rows=40)
    run(conn, "D5_qqq_holdings_coverage",
        "SELECT report_dt, count(*) AS n_holdings, count(DISTINCT permno) AS n_permno, round(sum(percent_tna)::numeric,2) AS pct_tna "
        "FROM crsp_q_mutualfunds.holdings WHERE crsp_portno IN (SELECT crsp_portno FROM crsp_q_mutualfunds.portnomap "
        "WHERE crsp_fundno IN (SELECT crsp_fundno FROM crsp_q_mutualfunds.fund_hdr_hist WHERE ticker IN ('QQQ','QQQQ'))) "
        "GROUP BY 1 ORDER BY 1", limit_rows=400)
    run(conn, "D6_holdings_cols", "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_schema='crsp_q_mutualfunds' AND table_name='holdings' ORDER BY ordinal_position")
    conn.close()
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
