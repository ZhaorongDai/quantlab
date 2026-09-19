"""Read-only WRDS CRSP Stock v2 schema check for phase 03.10 (one connection = one Duo push).

Usage: WRDS_USERNAME=<you> uv run python <this file>
"""

import json
import os
import sys
import time
from pathlib import Path

import psycopg2

OUT = Path(__file__).with_name("crsp_live_check.json")
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
        results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400], "secs": round(time.time() - t0, 2)}
    print(f"[{key}] {'ok' if results[key]['ok'] else 'FAILED'} ({results[key]['secs']}s)", flush=True)
    return results[key]


def main() -> None:
    user = os.environ.get("WRDS_USERNAME") or sys.exit("Set WRDS_USERNAME first")
    print("Connecting — approve the Duo push on your phone if prompted...", flush=True)
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require", connect_timeout=60)
    conn.set_session(readonly=True, autocommit=True)

    # C-1: CRSP schemas and whether we can use them
    run(conn, "C1_schemas",
        "SELECT nspname, has_schema_privilege(nspname,'USAGE') AS usable FROM pg_namespace "
        "WHERE nspname ILIKE 'crsp%' ORDER BY 1")
    # C-2: v2 / CIZ-style tables and views across crsp schemas
    run(conn, "C2_tables",
        "SELECT table_schema, table_name, table_type FROM information_schema.tables "
        "WHERE table_schema ILIKE 'crsp%' AND (table_name ILIKE '%v2%' OR table_name ILIKE 'stk%' "
        "OR table_name ILIKE '%ciz%' OR table_name ILIKE 'dsf%' OR table_name ILIKE '%dse%' "
        "OR table_name ILIKE '%delist%' OR table_name ILIKE '%dist%' OR table_name ILIKE '%names%' "
        "OR table_name ILIKE 'dsp500%' OR table_name ILIKE '%secinfo%' OR table_name ILIKE '%hdr%') "
        "ORDER BY 1,2", limit_rows=400)

    # C-3: columns of the most likely CIZ tables (whichever exist)
    names = ("dsf_v2", "stkdlysecuritydata", "stksecurityinfohist", "stkdelists", "stkdistributions",
             "wrds_dsfv2_query", "stknames", "stkissuerinfohist", "dsp500list_v2", "stocknames_v2")
    run(conn, "C3_columns",
        "SELECT table_schema, table_name, column_name, data_type FROM information_schema.columns "
        f"WHERE table_schema ILIKE 'crsp%' AND table_name IN {names} ORDER BY table_schema, table_name, ordinal_position",
        limit_rows=2000)

    usable = {r[0] for r in results["C1_schemas"].get("rows", []) if r[1] == "True"}
    present = {(r[0], r[1]) for r in results["C3_columns"].get("rows", [])}
    datecols = {}
    for s, t, c, _ in results["C3_columns"].get("rows", []):
        if c.lower() in ("dlycaldt", "date", "caldt", "secinfostartdt", "delistingdt", "disexdt"):
            datecols.setdefault((s, t), c)

    # C-4: per table: row count estimate (catalog), date range, and 3 AAPL-ish sample rows
    for (s, t) in sorted(present):
        if s not in usable:
            results[f"C4_{s}.{t}"] = {"ok": False, "error": "schema not usable (no USAGE privilege)"}
            continue
        run(conn, f"C4_est_{s}.{t}", f"SELECT reltuples::bigint FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
            f"WHERE n.nspname='{s}' AND c.relname='{t}'")
        dc = datecols.get((s, t))
        if dc:
            run(conn, f"C4_range_{s}.{t}", f"SELECT min({dc}), max({dc}) FROM {s}.{t}")
        run(conn, f"C4_sample_{s}.{t}", f"SELECT * FROM {s}.{t} WHERE permno = 14593 LIMIT 3")  # 14593 = Apple

    # C-5: ticker history shape for renamed / class-share names (FB->META, BRK.B, GOOGL)
    for s, t in sorted(present):
        if t in ("stksecurityinfohist", "stocknames_v2", "stknames") and s in usable:
            run(conn, f"C5_ticker_hist_{s}.{t}",
                f"SELECT * FROM {s}.{t} WHERE permno IN (13407, 83443, 90319) ORDER BY 1 LIMIT 60", limit_rows=60)
    conn.close()
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
