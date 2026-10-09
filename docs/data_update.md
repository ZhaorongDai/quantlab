# Daily data update

One script brings every shared store up to the latest bar, in order: the raw vendor downloads, the
Zarr stores, the factors and the factor risk model. It runs on its own schedule. The paper trading
does not update data; it waits until the update reports that the day is done.

## Contents

- [What it runs](#what-it-runs)
- [Store configs](#store-configs)
- [The status file](#the-status-file)
- [Schedule](#schedule)
- [Add a store](#add-a-store)

## What it runs

`scripts/data_update/update.py` reads `config/data_update.yaml` and runs its stages in order under the data
root `--data-dir`. Each step is one action on one store folder (paths relative to the data root):

| action | what it does |
|--------|--------------|
| `sharadar` | `scripts/sharadar/update.py`: downloads every Sharadar table and updates `market/sharadar/` and `universe/sharadar/` |
| `fred` | downloads the FRED series, then updates the store in the named folder |
| `benchmarks` | `scripts/sharadar/price_return_benchmark.py --refresh` for the listed tickers (`market/benchmarks/`) |
| `update` | rebuilds the dataset of a store folder from its `component.json` and calls `update()` |
| `mirror` | appends a source dataset's new bars to a store that copies its variables (`quantlab.backtest.live.mirror_new_bars`) |
| `extend` | rebuilds a factor, or a factor risk model, from the folder's `component.json` and extends it to t |

t is the last bar of `market/sharadar/sharadar_sep_1d` once the `raw` stage is done. Until every
store in the file's `ready` list holds t, and t is newer than the last successful run's, the raw
stage is retried every `--retry-minutes` (15) until `--retry-until` (08:30 New York time). A factor
is extended only by the store's owner (`Factor.owns_store`), never by a view pinned to some of its
outputs.

The shipped file updates what the us3000 paper trading reads, plus the base data: every Sharadar
table, the FRED 3-month T-bill rate, the price-return benchmarks, the Barra exposures, the us3000
price slices and membership, the us3000 alphas and the USE4 risk stores. WRDS data is not in the
daily update; run `scripts/wrds/*.py` when the vendor publishes.

```bash
cd ~/projects/quantlab2
export SHARADAR_API_KEY=<your-sharadar-key>
QUANTLAB_DATA_DIR=/data/quantlab taskset -c 64-114 .venv/bin/python scripts/data_update/update.py \
    --data-dir /data/quantlab --download-dir /data/quantlab/downloads
```

`--dry-run` prints the steps without running them; `--stage NAME` (repeatable) runs only those
stages, without the retry and without writing the status file.

## Store configs

A store the update rebuilds holds a `component.json` beside its `README.md`: the component's
`get_config()`, as JSON. `rebuild` turns it back into the dataset, factor or risk model with every
nested component, so the update needs no recipe code. A recipe writes the configs of the stores it
defines; `python examples/sharadar_us_equity/us3000_h1_mvo.py store-configs` writes the us3000 ones.

## The status file

When it finishes, the update writes `<data-dir>/update_status.json`:

| field | value |
|-------|-------|
| `date` | the New York date of the run |
| `t` | the bar the stores were brought to |
| `state` | `running`, `done`, `no_new_bar` (nothing new by the cut-off, as on a market holiday) or `failed` |
| `steps` | each step's action, store, result and seconds |
| `reason` | why it stopped, for `no_new_bar` and `failed` |

quantlab-ibkr's `live_daily.sh` waits for `"state": "done"` with today's `date` before it runs the
prediction job; a day that never gets there holds.

## Schedule

`scripts/data_update/update_daily.sh` is the cron entry. cron calls it every hour and it checks the time in
New York itself, so daylight-saving changes need no edit; at 06 ET on a weekday it runs the update,
with `SHARADAR_API_KEY` from `~/.config/quantlab/sharadar.env`, on NUMA Node 1, under a lock, and
logs to `<data-dir>/logs/data_update/<date>.log`:

```
0 * * * * $HOME/projects/quantlab2/scripts/data_update/update_daily.sh
```

`update_daily.sh now` runs it at once.

## Add a store

1. Build the store in its folder (`<category>/<group>/<stem>/<stem>.zarr`) with a short
   `README.md`.
2. Write the component's config to `component.json` in the same folder (`json.dumps(to_jsonable(obj.get_config()))`).
3. Add an `update` or `extend` step to `config/data_update.yaml`, after the steps that update its
   inputs.
