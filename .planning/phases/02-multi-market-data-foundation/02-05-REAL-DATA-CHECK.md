# 02-05 Real-Data Smoke Check (D-11)

Status: SUCCESS

## Commands run

Note: the task's default instruction ("first day of the most recent
fully-completed month") resolved to 2026-08 relative to this machine's system
clock (2026-09-05), but Binance's public data.vision archive has not yet
published a 2026-08 monthly file (`data/spot/monthly/klines/BTCUSDT/1d/`
lists monthly files only up through `2026-07`, confirmed via
`curl https://data.binance.vision/?prefix=data/spot/monthly/klines/BTCUSDT/1d/`).
2026-08-01..2026-08-31 returned 0/108 matching files. Fetched the most
recent month for which Binance has actually published data instead: 2026-07.

1. Downloaded the real monthly CSV via the user's separate `binance-data-downloader` tool:

```bash
OUTDIR="$(uv run python -c "from config import spot_kline_config; print(spot_kline_config().raw_data_dir_path)")"
uvx binance-data-downloader --data-type spot --interval monthly --symbol klines \
  --trading-pair BTCUSDT --time-interval 1d \
  --start-date 2026-07-01 --end-date 2026-07-31 \
  --output-dir "$OUTDIR" --extract
```

Result: `Downloaded 1/1 data files`, `Extracted 1/1 files successfully`, extracted to
`{raw_data_dir_path}/spot/monthly/klines/BTCUSDT/1d_extracted/BTCUSDT-1d-2026-07.csv`
(31 headerless rows, one per day of July 2026).

2. Ran the ingestion CLI built in Task 2:

```bash
uv run python ingest_binance_spot.py --symbols BTCUSDT
```

Result: completed without error (log: `SpotKlineDataset: from csv consumed time: 0.01s`,
`SpotKlineDataset: save consumed time: 0.27s`). `get_csv_files()`'s recursive glob found the
CSV under the nested `spot/monthly/klines/BTCUSDT/1d_extracted/` path with no manual
flattening/moving required, confirming the plan's `get_csv_files()`/`utils/file.py`
assumption.

## Date range fetched

2026-07-01 through 2026-07-31 (one full month, BTCUSDT, 1d klines).

## Resulting Zarr path

`data/data/crypto_spot/1d/klines.zarr` (relative to the worktree root; derived from
`spot_kline_config()`'s `market="crypto_spot", frequency="1d"` convention, per 02-02).

## Row count

31 rows (`timestamp` dim size = 31), 1 symbol (`symbol` dim size = 1) -- consistent with one
month of daily bars for one symbol (July has 31 days).

## Sample rows (real market data, not a fixture)

| timestamp  | symbol  | open     | high     | low      | close    | volume      |
|------------|---------|----------|----------|----------|----------|-------------|
| 2026-07-01 | BTCUSDT | 58624.71 | 61334.00 | 57800.19 | 60024.00 | 25093.44662 |
| 2026-07-02 | BTCUSDT | 60024.00 | 62200.00 | 59588.00 | 61560.00 | 21382.14163 |
| 2026-07-31 | BTCUSDT | 64780.03 | 65409.56 | 62466.00 | 62887.88 | 20475.60686 |

These OHLCV values were read directly from `xr.open_zarr(...)` after running
`ingest_binance_spot.py`, verifying the full round-trip: real Binance CSV ->
`SpotKlineDataset.from_raw_data()` (with the Task 1 dedup fix and `_clean()` override applied)
-> `.save()` -> Zarr -> `xr.open_zarr()` read-back.

## Conclusion

D-11 satisfied: a real, non-fixture BTCUSDT 1d sample was fetched via the user's
`binance-data-downloader` tool and round-tripped through `ingest_binance_spot.py` into Zarr,
with plausible OHLCV values. Per the task's own instructions, Tasks 1-2's fixture-based
automated tests remain the phase's authoritative, repeatable DATA-02 verification regardless of
this outcome -- this file is supplementary evidence, not a replacement for those tests.
