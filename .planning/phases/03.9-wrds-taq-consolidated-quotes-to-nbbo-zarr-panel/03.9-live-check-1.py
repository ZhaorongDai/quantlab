"""Read-only WRDS live checklist L-0..L-9 for phase 03.9 (one connection = one Duo push).

Usage (from repo root):  WRDS_USERNAME=<you> uv run python <this file>
Password comes from ~/.pgpass via libpq; nothing is written on WRDS.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

import psycopg2

OUT = Path(__file__).with_name("wrds_live_check.json")
results: dict = {}


def run(conn, key, sql, fetch=True):
    t0 = time.time()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall() if fetch else None
            cols = [d[0] for d in cur.description] if cur.description else []
        results[key] = {"ok": True, "cols": cols, "rows": [list(map(str, r)) for r in rows or []], "secs": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        conn.rollback()
        results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500], "secs": round(time.time() - t0, 2)}
    print(f"[{key}] {'ok' if results[key]['ok'] else 'FAILED'} ({results[key]['secs']}s)", flush=True)


def main() -> None:
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        sys.exit("Set WRDS_USERNAME first")
    print("Connecting — approve the Duo push on your phone if prompted...", flush=True)
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require", connect_timeout=60)
    conn.set_session(readonly=True, autocommit=True)

    run(conn, "L0_entitlement", "SELECT has_schema_privilege('taqm_2024','USAGE')")
    run(conn, "L1_tables", "SELECT table_schema, table_name, table_type FROM information_schema.tables "
        "WHERE table_name IN ('complete_nbbo_20240124','nbbom_20240124') ORDER BY 1,2")
    run(conn, "L2_columns", "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
        "WHERE table_schema IN ('taqm_2012','taqm_2016','taqm_2024') AND table_name IN "
        "('complete_nbbo_20120103','complete_nbbo_20161207','complete_nbbo_20240124','nbbom_20240124') "
        "ORDER BY table_schema, table_name, ordinal_position")
    cols_2024 = {r[2] for r in results["L2_columns"].get("rows", []) if r[1] == "complete_nbbo_20240124"}
    has_seq = "qu_seqnum" in cols_2024
    run(conn, "L3_nano_range", "SELECT min(time_m_nano), max(time_m_nano), count(*) FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root='AAPL'")

    seq = ", qu_seqnum" if has_seq else ""
    for year, day in (("2024", "20240124"), ("2016", "20161207")):
        run(conn, f"L4_key_unique_{year}",
            f"SELECT count(*) AS n, "
            f"count(DISTINCT (sym_root, coalesce(sym_suffix,''), time_m, time_m_nano{seq})) AS n_key, "
            f"count(DISTINCT (sym_root, coalesce(sym_suffix,''), time_m, time_m_nano)) AS n_ts "
            f"FROM taqm_{year}.complete_nbbo_{day} WHERE sym_root IN ('AAPL','MSFT','BRK')")
    if has_seq:
        # seqnum monotonic in time for AAPL: count adjacent pairs where seqnum decreases (ordered by time only)
        run(conn, "L4b_seq_monotonic_2024",
            "SELECT count(*) FILTER (WHERE d < 0) AS decreasing, count(*) AS pairs FROM ("
            " SELECT qu_seqnum - lag(qu_seqnum) OVER (ORDER BY time_m, time_m_nano, qu_seqnum) AS d"
            " FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root='AAPL' AND sym_suffix IS NULL) s")

    run(conn, "L5_volume_2024", "SELECT sym_root, count(*) FROM taqm_2024.complete_nbbo_20240124 "
        "WHERE sym_root IN ('AAPL','MSFT','JNJ','BRK','XOM') GROUP BY 1 ORDER BY 1")

    t0 = time.time()
    try:
        buf = io.StringIO()
        with conn.cursor() as cur:
            cur.copy_expert("COPY (SELECT * FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root='AAPL' LIMIT 5) "
                            "TO STDOUT WITH (FORMAT csv, HEADER true)", buf)
        results["L6_copy"] = {"ok": True, "sample_csv": buf.getvalue(), "secs": round(time.time() - t0, 2)}
    except Exception as exc:  # noqa: BLE001
        results["L6_copy"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    print(f"[L6_copy] {'ok' if results['L6_copy']['ok'] else 'FAILED'}", flush=True)

    run(conn, "L7_conn_limit", "SELECT rolconnlimit FROM pg_roles WHERE rolname = current_user")
    run(conn, "L8_index", "EXPLAIN SELECT count(*) FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root = ANY(ARRAY['AAPL'])")
    run(conn, "L9_suffix", "SELECT DISTINCT sym_root, sym_suffix FROM taqm_2024.complete_nbbo_20240124 WHERE sym_root IN ('BRK','BF') ORDER BY 1,2")
    conn.close()

    results["_meta"] = {"has_qu_seqnum_2024": has_seq}
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
