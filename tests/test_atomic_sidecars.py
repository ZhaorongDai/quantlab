"""Home for the sidecar-durability proofs: requirement D-20.

D-20 — no same-source file lock is introduced (a lock was considered and
rejected), but the AMENDED decision brings atomic sidecar WRITES into scope:
watermark sidecars must be written the way `PageLedger._flush` already writes
its ledger, through ONE shared helper rather than a third hand-rolled copy
(L-4, "Don't Hand-Roll").

Scaffolded by plan 03.4-01 (Wave 0). `quantlab/utils/atomic.py:write_json_atomically`
does not exist yet; plan 03.4-03 extracts it and fills this file in.

The two tests below bracket that change from both sides:

- the PRECEDENT the extraction copies is proven to work before it is copied,
  so a regression in the extracted helper is attributable rather than being
  discovered as "the ledger was always like this";
- the BLAST-RADIUS BOUND the amendment relies on (a corrupt sidecar reads back
  as absent rather than raising) is asserted for the code as it stands TODAY,
  so it can be re-asserted unchanged after the atomic write lands. A property
  only asserted after a change cannot tell you the change preserved it.

RULES THIS FILE IS SUBJECT TO (incidents recorded in `.planning/STATE.md`):
every test must be a real assertion, because a pytest file with zero tests
exits **5** ("no tests ran") and a per-file command reads that as green; and
no test is named for a selector a later plan owns.
"""

import json
from pathlib import Path

from quantlab.acquisition.tiingo import TiingoAcquisition
from quantlab.base.pageledger import PageLedger


def test_page_ledger_flush_leaves_no_temp_file_and_writes_parseable_json(
    tmp_path: Path,
) -> None:
    """The in-repo atomic-write precedent, proven before plan 03 copies it.

    `PageLedger._flush` writes to a `NamedTemporaryFile` in the SAME directory
    (so `os.replace` is a rename within one filesystem, which is what makes it
    atomic) and unlinks the temp file on any exception. Two observable
    consequences, both asserted here:

    - after a successful write the directory contains the sidecar and NOTHING
      else -- a surviving `*.tmp` would accumulate one file per page on a real
      multi-thousand-page run, and would also mean the `os.replace` never
      happened;
    - the written file parses as JSON. That is the whole point of the temp-file
      dance: a crash mid-write must leave either the previous valid ledger or
      the new one, never a half-written file the next run cannot resume from.

    A second `record_page` follows the first, because the interesting case is
    the OVERWRITE: the first write into an empty directory would leave no temp
    file even from a naive implementation that simply opened the target path.
    """
    ledger_dir = tmp_path / "_watermarks" / "tiingo"
    ledger_path = ledger_dir / "batch0000.json"

    ledger = PageLedger(str(ledger_path), symbols=["AAPL", "MSFT"])
    ledger.record_page(
        index=0,
        next_token="token-0",
        rows=2,
        seen=["AAPL"],
        shard_paths=[str(tmp_path / "part-batch0000-00000.pqt")],
    )
    ledger.record_page(
        index=1,
        next_token=None,
        rows=3,
        seen=["MSFT"],
        shard_paths=[str(tmp_path / "part-batch0000-00001.pqt")],
    )

    assert ledger_path.is_file()
    assert list(ledger_dir.glob("*.tmp")) == [], (
        "a surviving temp file means os.replace never ran, or the writer "
        "leaked one file per page"
    )
    assert sorted(p.name for p in ledger_dir.iterdir()) == [ledger_path.name]

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert len(payload["pages"]) == 2
    assert sorted(payload["symbols_with_data"]) == ["AAPL", "MSFT"]


def test_a_half_written_watermark_sidecar_reads_back_as_absent(
    mock_tiingo_client, acquisition_config
) -> None:
    """The blast-radius bound L-4 relies on, asserted BEFORE the change.

    `Acquisition._read_sidecar` is the single tolerant read that both
    `_read_watermark` and `_read_coverage` share, so there is exactly one
    failure policy for a corrupt sidecar rather than two that could drift. Its
    policy is: absent or unparseable both mean `None`, i.e. "no watermark",
    i.e. re-fetch from `config.start_date`. The worst case is a
    wider-than-necessary re-fetch; the alternative -- raising -- would let one
    truncated file abort a whole refresh run.

    That is what bounds the damage a non-atomic write can do today, and it is
    exactly why the atomic write (D-20) is an improvement rather than a
    correctness fix. It must hold identically after plan 03 lands, so it is
    pinned here first: a bound only measured after a change cannot show the
    change preserved it.

    Both corrupt shapes are covered -- truncated JSON (what a crash mid-write
    actually produces) and well-formed JSON that is not an object (which
    `_read_sidecar` also rejects, since callers index it like a dict).
    """
    config = acquisition_config(vendor="tiingo", symbols=("AAPL", "MSFT"))
    acq = TiingoAcquisition(config)

    assert acq._read_sidecar("AAPL") is None, "no sidecar yet -- must be None"

    truncated_path = acq._watermark_path("AAPL")
    truncated_path.parent.mkdir(parents=True, exist_ok=True)
    truncated_path.write_text('{"last_date": "2024-01-3', encoding="utf-8")

    assert acq._read_sidecar("AAPL") is None
    assert acq._read_watermark("AAPL") is None

    not_an_object_path = acq._watermark_path("MSFT")
    not_an_object_path.write_text('["2024-01-31"]', encoding="utf-8")

    assert acq._read_sidecar("MSFT") is None
