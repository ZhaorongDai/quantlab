"""Read-only WRDS follow-up checks for phase 03.9 (one connection = one Duo push).

Usage: WRDS_USERNAME=<you> uv run python <this file>
"""

import json
import os
import sys
import time
from pathlib import Path

import psycopg2

OUT = Path(__file__).with_name("wrds_live_check2.json")
results: dict = {}


def run(conn, key, sql):
    t0 = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        results[key] = {"ok": True, "cols": cols, "rows": [list(map(str, r)) for r in rows], "secs": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "secs": round(time.time() - t0, 2)}
    print(f"[{key}] {'ok' if results[key]['ok'] else 'FAILED'} ({results[key]['secs']}s)", flush=True)


def main() -> None:
    user = os.environ.get("WRDS_USERNAME") or sys.exit("Set WRDS_USERNAME first")
    print("Connecting — approve the Duo push on your phone if prompted...", flush=True)
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require", connect_timeout=60)
    conn.set_session(readonly=True, autocommit=True)

    # M-1: which complete_nbbo tables carry time_m_nano, per year (first trading-day-ish table of each year)
    run(conn, "M1_nano_by_year",
        "SELECT table_schema, min(table_name) AS first_table, "
        "bool_or(EXISTS (SELECT 1 FROM information_schema.columns c WHERE c.table_schema=t.table_schema "
        "AND c.table_name=t.table_name AND c.column_name='time_m_nano')) AS any_nano, "
        "count(*) AS n_tables "
        "FROM information_schema.tables t WHERE table_schema ~ '^taqm_[0-9]{4}$' AND table_name ~ '^complete_nbbo_[0-9]{8}$' "
        "GROUP BY table_schema ORDER BY table_schema")
    # M-2: exact first table that has time_m_nano
    run(conn, "M2_first_nano_table",
        "SELECT min(table_name) FROM information_schema.columns WHERE table_schema ~ '^taqm_[0-9]{4}$' "
        "AND table_name ~ '^complete_nbbo_[0-9]{8}$' AND column_name='time_m_nano'")

    # M-3: timestamp-tie counts in pre-nano years on (symbol, time_m)
    for year, day in (("2012", "20120103"), ("2016", "20161207")):
        run(conn, f"M3_ties_{year}",
            f"SELECT count(*) AS n, count(DISTINCT (sym_root, coalesce(sym_suffix,''), time_m)) AS n_ts "
            f"FROM taqm_{year}.complete_nbbo_{day} WHERE sym_root IN ('AAPL','MSFT','BRK','XOM','JNJ')")
    # M-4: are ties in 2016 distinguishable by content (identical rows vs different NBBO state)?
    run(conn, "M4_tie_content_2016",
        "SELECT count(*) AS n, count(DISTINCT (sym_root, coalesce(sym_suffix,''), time_m, best_bid, best_bidsizeshares, "
        "best_ask, best_asksizeshares, qu_cond, natbbo_ind, qu_source)) AS n_distinct_rows "
        "FROM taqm_2016.complete_nbbo_20161207 WHERE sym_root IN ('AAPL','MSFT','BRK','XOM','JNJ')")

    # M-5: full-market day size for the volume model
    run(conn, "M5_fullday_2024", "SELECT count(*) AS n_rows, count(DISTINCT sym_root) AS n_roots FROM taqm_2024.complete_nbbo_20240124")
    # M-6: one-sided / empty rows share (2024, 5 names), and how many fall inside RTH
    run(conn, "M6_onesided_2024",
        "SELECT count(*) FILTER (WHERE best_bid IS NULL OR best_ask IS NULL) AS one_or_both_null, "
        "count(*) FILTER (WHERE (best_bid IS NULL OR best_ask IS NULL) AND time_m BETWEEN '09:30' AND '16:00') AS null_in_rth, "
        "count(*) FILTER (WHERE best_bid > best_ask) AS crossed, count(*) FILTER (WHERE best_bid = best_ask) AS locked, "
        "count(*) AS n FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root IN ('AAPL','MSFT','BRK','XOM','JNJ')")
    # M-7: which qu_cond / natbbo_ind values appear (2024, 5 names)
    run(conn, "M7_codes_2024",
        "SELECT qu_cond, natbbo_ind, qu_source, count(*) FROM taqm_2024.complete_nbbo_20240124 "
        "WHERE sym_root IN ('AAPL','MSFT','BRK','XOM','JNJ') GROUP BY 1,2,3 ORDER BY 4 DESC LIMIT 40")
    conn.close()
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
