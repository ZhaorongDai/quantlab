# Internals

This page describes the machinery that keeps quantlab's long-running jobs
safe to interrupt and its results safe to trust: how downloads and
conversions resume where they stopped, how a store is rebuilt without losing
the old one, why three package `__init__.py` files stay empty, how files are
written atomically, and how data fingerprints tie a
backtest to the data it read. It is written for contributors who change these
parts of the code or add a component that has to cooperate with them. Users
only need the behaviour, which [Data sources](../user-guide/data-sources.md),
[Datasets](../user-guide/datasets.md) and
[Backtesting](../user-guide/backtesting.md) describe.

The examples below were run offline; the file listing comes from the demo
vendor in [Extending quantlab](extending.md#a-data-source).

## Resumable downloads

A full-market download is thousands of requests and can take hours, so every
acquisition must survive being killed at any point and continue without
re-fetching finished work or skipping unfinished work. Three kinds of small
JSON file, all kept under the config's `watermark_path` and never inside the
raw data directory (a Parquet directory scan would choke on them), carry that
state:

```text
_watermarks/demo/AAA.json                          one watermark per symbol
_watermarks/demo/_failures.json                    the failure manifest
_watermarks/demo/_pages/5b7fc3dffa5ab4a0.pages.json one page ledger per batch
raw/demo/month=2024-01/part-5b7fc3dffa5ab4a0-00000.pqt
```

A *watermark* records the date range of a symbol's data already on disk, for
example `{"last_date": "2024-03-29", "start_date": "2024-01-02"}`. Before each
pass, `Acquisition._run` asks `CoverageLedger` (`quantlab.base.coverage`)
which requested symbols are not yet covered for the requested window, and
fetches only those. A watermark is written only after the symbol's whole
batch has been fetched and written, so a symbol interrupted mid-download has
no watermark and is fetched again. `refresh()` starts each symbol at its own `last_date` instead
of the config's start date. The failure manifest lists symbols whose download
failed, with the reason; it accumulates across runs, and a later success
removes the entry.

A *batch* is one request covering several symbols. Vendors such as Alpaca
answer it as a chain of pages linked by opaque tokens, so a batch can itself
be interrupted halfway. The *page ledger* (`quantlab.base.pageledger`)
records, per batch, every page fetched, the token the next request must send,
and the shard files each page was written to. `Acquisition._fetch_batch` is
the only pagination loop, and it resumes from the ledger:

```python
from quantlab.base.pageledger import PageLedger

roster = ["AAPL", "MSFT"]
key = PageLedger.batch_key("alpaca", "1m", "2024-01-02", "2024-01-05", roster)
path = PageLedger.default_path(str(root / "_watermarks"), key)
ledger = PageLedger(path, roster)
ledger.describe(key, "alpaca", "1m", "2024-01-02", "2024-01-05", roster)
ledger.record_page(0, "tok-1", rows=10_000, seen=["AAPL"],
                   shard_paths=[f"date=2024-01-02/part-{key}-00000.pqt"])

print(key, Path(path).relative_to(root))
print(PageLedger(path, roster).resume_point())            # a new process resumes here
print(PageLedger(path, ["AAPL", "NVDA"]).resume_point())  # another roster starts over
```

```text
91c9dc202fdc2cc1 _watermarks/_pages/91c9dc202fdc2cc1.pages.json
(1, 'tok-1')
(0, None)
```

Several properties make resuming safe, and a change to this code must keep
all of them:

- The batch key is a hash of vendor, frequency, dates and the sorted symbol
  list, and shard names are `part-{batch_key}-{page:05d}.pqt` with no
  timestamp or random part. Re-fetching a page overwrites the same file, so a
  retry can cost requests but never duplicates rows.
- A page's shard is written before the ledger records the page. A crash in
  between costs one re-fetch; the opposite order could record a page whose
  data never reached disk.
- The ledger stores a fingerprint of its roster. Opened with a different
  roster, it reads back empty rather than resuming pages fetched for other
  symbols.
- Before resuming, `PageLedger.assert_consistent` checks that every recorded
  shard still exists and raises if one is missing, since resuming past it
  would leave a hole no later read could detect.
- A vendor that returns the token it was given is refused with `ValueError`
  instead of looping forever and writing a new shard on every iteration.
- A missing or corrupt ledger file reads back as an empty ledger, which costs
  that one batch a re-fetch and nothing more.

Each ledger file has one writer, because each batch runs in one worker thread.
Merging ledgers into a shared file would need a lock around the update and the
write.

## Resumable conversions

Converting a large raw tier into a Zarr store is also resumable.
`BaseDataset.from_raw_data_chunked` splits the time axis into windows with
`TimeChunkPlanner` (`quantlab.base.chunking`), densifies each window onto one
symbol axis fixed for the whole range before the first window (so all windows
line up column by column), appends it to the store, and records it in a
`ChunkLedger` next to the store (`<store>.chunks.json`). A re-run skips the
windows the ledger lists.

The ledger and the store are two records of the same history, and an append
cannot be undone, so `ChunkLedger.assert_consistent` trusts neither alone. It
refuses to resume when the symbol axis changed between runs, when a store
exists but the ledger is empty, when the ledger lists windows but the store is
gone, or when the store's last timestamp differs from the ledger's last
window, which means the process died between appending and recording.

## Rebuilds

A store is derived data: when the conversion code or the symbol list changes,
the store on disk has to be rebuilt from the raw tier. Two mechanisms do
this, and both follow the same rule: the old store is never deleted until the
new one is complete.

When the raw tier gains symbols the store does not have, `on_new_listing`
decides what chunked conversion does. `"widen"` adds the new symbols as NaN
columns through `XrBackend.widen_symbol_axis`, which writes the widened store
beside the original (`.widening.tmp`), then swaps directories with two
renames, briefly parking the original as `.superseded.tmp`. `"rebuild"` renames
the store and its ledger aside with the same `.superseded.tmp` suffix and
converts every window afresh; if the rebuild fails or is cancelled, the
partial store is removed and the originals are renamed back. If even that
rename fails, the error is logged rather than raised, so it cannot hide the
error that stopped the rebuild, and the log names both copies for manual
recovery. A widen refuses to start while a non-empty `.superseded.tmp` copy
is left over from an earlier crash, because that copy may be the only
complete version of the data.

`BaseStoreRebuilder` (`quantlab.base.rebuild`) is the skeleton for an explicit
rebuild of a whole store, used by the CRSP rebuilder in
`quantlab/dataset/crsp/rebuild.py`. `rebuild()` runs
`assert_inputs_present`, `backup`, `clear`, `_convert` and `_measure` in that
order: a missing raw tier is reported while the old store is still on disk
(converting nothing would write an empty panel that looks like a period with
no trading), a backup exists before anything is deleted, sidecars are deleted
together with the store so no stale bookkeeping survives, and a measurement is
returned only after the conversion succeeded. `data_root` is required and has
no default, because a rebuild run from a git worktree, which has no `data/`
directory, must not report success against a tree it never read.

## Empty package `__init__` files

`quantlab/__init__.py`, `quantlab/acquisition/__init__.py` and
`quantlab/acquisition/_support/__init__.py` are empty. A package `__init__`
runs on every import beneath it, and the credential-free `SourceInspector`
(`quantlab.acquisition._support.inspector`) must be importable without loading
a vendor client: `tests/test_source_inspector.py` fails if a client becomes
reachable from it. Do not add imports to those files. Downloads are not
estimated or refused by size.

## Atomic writes

Every JSON sidecar (watermarks, ledgers, failure manifests, backtest
metrics) is written by `quantlab.utils.atomic.write_json_atomically`: the
payload goes to a temporary file in the destination's own directory, is
flushed and `fsync`ed, and is renamed over the destination with `os.replace`.
A crash leaves either the previous complete file or the new one, never a
truncated file that a resumed run would fail to parse. The temporary file has
to be in the same directory, because a rename is atomic only within one
filesystem. There is no lock; two concurrent writers still race, but the
loser sees a complete file. New code that persists state should use this
function rather than `open(...).write`.

Directories follow the same pattern. A backtest run directory is written as a
hidden `.{name}.partial` sibling and renamed into place only after every
artifact succeeded, and the staging directory is removed on any exception,
including `KeyboardInterrupt`. Widening a store swaps whole directories by
rename, as described above.

## Data fingerprints

A backtest is reproducible only if the data under it has not changed, and
data does change: stores are appended to, and adjusted prices are restated
after splits and dividends. `quantlab.utils.fingerprint.dataset_fingerprint`
records, for one panel and a list of variables, a SHA-256 digest over the
timestamps, the symbol names and the values, plus the first and last
timestamp and the axis sizes. The panel is sorted first so axis order does
not matter, and every NaN is rewritten to one canonical bit pattern and every
`-0.0` to `0.0`, because otherwise two reads of identical data could hash
differently.

The backtester records one fingerprint for the two price columns it trades
on, one per factor over the dataset columns that factor consumes (and over the
factor store itself when features are read from a store), and in train mode
one per factor and label over the training data. They are written to
`fingerprint.json` and into `config.json` as `data_fingerprint`. A backtester
rebuilt by `load_backtester_from_config` compares its own fingerprints with
the stored ones and logs a warning for every key that differs or is missing.
It never raises, because changed data can still be worth backtesting. If the
run fails partway, a partial comparison runs before the error propagates, with
a note that an interrupted read may explain the difference; that diagnostic
swallows its own errors so it can never replace the original exception.
