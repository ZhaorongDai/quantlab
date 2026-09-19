"""Read-only WRDS live checklist L1..L11 for phase 03.10, SQL taken verbatim from 03.10-RESEARCH.md.

Usage (repo root): WRDS_USERNAME=<you> uv run python <this file>
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import psycopg2

RESEARCH = Path(".planning/phases/03.10-crsp-stock-v2-daily-data-via-wrds/03.10-RESEARCH.md")
OUT = Path(__file__).with_name("crsp_live_check2.json")


def load_queries():
    text = RESEARCH.read_text()
    block = text.split("## User-Run Live Checklist", 1)[1].split("```sql", 1)[1].split("```", 1)[0]
    queries, label, buf, n = [], "L?", [], 0
    for line in block.splitlines():
        m = re.match(r"\s*--\s*(L\d+)", line)
        if m:
            label, n = m.group(1), 0
            continue
        s = re.sub(r"--.*$", "", line)
        if not s.strip():
            continue
        buf.append(s)
        if s.rstrip().endswith(";"):
            n += 1
            queries.append((f"{label}_{n}", "\n".join(buf).rstrip().rstrip(";")))
            buf = []
    return queries


def main() -> None:
    user = os.environ.get("WRDS_USERNAME") or sys.exit("Set WRDS_USERNAME first")
    queries = load_queries()
    for q in queries:
        assert re.match(r"^\s*(SELECT|WITH)\b", q[1], re.I), q  # read-only guard
    print(f"{len(queries)} queries loaded. Connecting — approve the Duo push if prompted...", flush=True)
    conn = psycopg2.connect(host="wrds-pgdata.wharton.upenn.edu", port=9737, dbname="wrds", user=user, sslmode="require", connect_timeout=60)
    conn.set_session(readonly=True, autocommit=True)
    results = {}
    for key, sql in queries:
        t0 = time.time()
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchmany(500)
                cols = [d[0] for d in cur.description]
            results[key] = {"ok": True, "sql": sql, "cols": cols, "rows": [list(map(str, r)) for r in rows], "secs": round(time.time() - t0, 2)}
        except Exception as exc:  # noqa: BLE001 - diagnostic script
            results[key] = {"ok": False, "sql": sql, "error": f"{type(exc).__name__}: {exc}"[:400], "secs": round(time.time() - t0, 2)}
        print(f"[{key}] {'ok' if results[key]['ok'] else 'FAILED'} ({results[key]['secs']}s)", flush=True)
    conn.close()
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
