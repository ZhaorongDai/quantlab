# Download vendor data and keep it current

Two scripts take real market data from a vendor to dataset stores on your disk. Each is one
command for both jobs: the first run downloads and builds, and every later run adds only what is
new. An interrupted run is simply run again.

| Script | Downloads | Builds | Needs |
|--------|-----------|--------|-------|
| [`sharadar_data.py`](sharadar_data.py) | Sharadar's tables, as parquet | Every Sharadar store: prices, fundamentals, valuations, events, S&P 500 membership | A Sharadar subscription and `SHARADAR_API_KEY` |
| [`massive_data.py`](massive_data.py) | Massive's day, minute and trade files and its split and dividend tables | Daily bars by ticker, the Ticker map, daily bars by security with adjusted prices, and one-minute bars of a few tickers | A Massive Stocks Developer plan, `MASSIVE_API_KEY`, `MASSIVE_S3_ACCESS_KEY_ID` and the Sharadar stores |

Each script imports only quantlab, and every setting is a constant at the top of the file: the
first day (`START`), the tables or endpoints, the tickers of the minute bars. Copy a script and
edit it.

## First run

Choose a data root outside the repository, because the data is licensed for personal use (the
scripts refuse one inside it), then run Sharadar first:

```bash
export QUANTLAB_DATA_DIR=/data/quantlab
export SHARADAR_API_KEY=<your-sharadar-key>
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>

uv run python examples/data_update/sharadar_data.py
uv run python examples/data_update/massive_data.py
```

Measured on a 64-core server: Sharadar's first download took about 30 minutes and building its
stores from 2025 about 10, with 3 GB on disk; a later run took about 10 minutes. Massive's files
come to about 3 GB a day, and its first run needs about 2 minutes a day of minute bars. The warnings
of `massive_data.py` count what was left out or differs from Sharadar: splits and dividends from
before `START` have no Ticker map row, and the differences are listed in
`<store>.sep_check.json`. Neither stops the run.

## What lands on disk

```text
$QUANTLAB_DATA_DIR/
  downloads/                the raw tier: the vendors' files and their watermarks
    sharadar/<table>/
    massive/<data type>/<year>/<day>.csv.gz
    massive/rest/<endpoint>/
  market/                   the dataset stores, one folder each
    sharadar/sharadar_sep_1d/
      sharadar_sep_1d.zarr
      component.json        rebuilds the dataset: load_component(folder)
      README.md
    massive/massive_permaticker_day_bars_1d/
  universe/sharadar/sharadar_sp500_membership/
```

Read a store from its folder:

```python
from quantlab.core.store_folder import load_component

prices = load_component("/data/quantlab/market/sharadar/sharadar_sep_1d")
panel = prices.panel("2026-10-01", "2026-10-30")
```

## Update every day

Schedule the same two commands for the morning after the vendors publish. cron does not read your
shell profile, so put the `export` lines above in a file only you can read
(`chmod 600 ~/.config/quantlab/data.env`) and load it in the job. On a machine set to New York
time:

```text
0 5 * * 1-5  cd $HOME/quantlab && . $HOME/.config/quantlab/data.env && uv run python examples/data_update/sharadar_data.py && uv run python examples/data_update/massive_data.py >> $HOME/data_update.log 2>&1
```

Each script exits 1 when a part failed. Running it again is safe, so a retry is the same command.
A day the vendor has not published yet is not a failure: the next run picks it up.

Factors, labels and the risk model are computed from these stores, and are extended after them:
see [Keeping data current](../../docs/user-guide/data-update.md).
