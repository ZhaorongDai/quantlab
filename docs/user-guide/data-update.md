# Keeping data current

Vendor data reaches a model in three stages. Each stage has its own place on disk, and each
can be run again without touching the stage before it:

```text
vendor --download--> raw tier --convert--> dataset store --extend--> derived store
                     downloads/<vendor>/   market/<vendor>/<name>/   factors/  labels/  risk/
```

- The **raw tier** holds the vendor's data as it arrived, with a watermark per table saying how
  far it reaches.
- A **dataset store** is a `(timestamp, symbol)` Zarr panel, beside its `README.md` and
  `component.json`. Factors, models and backtests read these stores, never the raw tier.
- A **derived store** is computed from dataset stores: factors, labels and the factor risk model.

The first run and a daily run differ only in the download and in what the stores start from:

| Stage | First run | Every day after |
|---|---|---|
| Sharadar raw tier | `SharadarClient.bulk_table` for each table | `refresh_raw_tier` from the watermarks |
| Sharadar stores | `build_stores` | `update_stores` |
| Massive raw tier | `MassiveClient.download_days` and `refresh_raw_tier` from your start date | the same two calls, which resume after their watermarks |
| Massive stores | `update()` builds them | `update()` appends the new days |
| Derived stores | `build()`, then `save_component` | `load_component`, then `extend` |

## Start from the examples

[`examples/data_update/`](https://github.com/ZhaorongDai/quantlab/tree/main/examples/data_update)
holds one script per vendor that does all of this. Run the same script for the first build and
every day after it: it finds out by itself what is missing.

```bash
export QUANTLAB_DATA_DIR=/data/quantlab     # outside the repository: the data is licensed
export SHARADAR_API_KEY=<your-sharadar-key>
export MASSIVE_API_KEY=<your-api-key>
export MASSIVE_S3_ACCESS_KEY_ID=<your-s3-access-key-id>

uv run python examples/data_update/sharadar_data.py
uv run python examples/data_update/massive_data.py
```

Run Sharadar first: Massive's Ticker map needs Sharadar's closes of the day. The rest of this page
describes each step, for when you build your own job.

## The daily steps

Keeping a data root current is three steps, run in this order after the vendors publish a day:

1. refresh each vendor's raw tier from its watermarks;
2. bring the dataset stores up from the raw tier;
3. extend the derived stores.

Each step only appends: earlier rows are never rewritten, so a step that fails can be run again.

### Refresh the raw tiers

{py:func}`~quantlab.acquisition.registry.refresh_raw_tier` pulls everything a vendor published
since its last pull (a Sharadar table, a Massive REST endpoint, a FRED series), each part from its
own watermark. The caller names the vendor, not a roster or a window. It writes raw files only and
returns every part it tried; a part that fails does not stop the others. A Sharadar table must have
had its first bulk pull; the refresh refuses one that has not.

```python
from quantlab.acquisition.base import DataSourceRegistry
from quantlab.acquisition.registry import refresh_raw_tier

downloads = "/data/quantlab/downloads"
for vendor, options in [("sharadar", {}), ("fred", {"series": ["DTB3"]})]:
    result = refresh_raw_tier(DataSourceRegistry.get(vendor), downloads, **options)
    if result.failed:
        print(vendor, "failed:", result.failed)
```

Massive's refresh covers its REST tables only. Its daily files are downloaded with
{py:meth}`MassiveClient.download_days <quantlab.acquisition.massive.client.MassiveClient.download_days>`,
which resumes after each data type's watermark and leaves a day not published yet for the next run;
see [Massive](massive.md).

Vendors with a symbol-batched download and no raw-tier refresh (Tiingo, Alpaca) are pulled with
`download_symbols()` instead; see [Data sources](data-sources.md).

### Bring the dataset stores up

A dataset store appends the new bars from the raw tier with `update()`. For Sharadar,
{py:func}`~quantlab.dataset.sharadar.stores.update_stores` does it for every store at once, builds
a store that does not exist yet, and records a store that fails without stopping the rest:

```python
from quantlab.dataset.sharadar.stores import update_stores

outcome = update_stores(downloads, "/data/quantlab")
print(outcome.updated, outcome.failed)
```

A store refuses to resume when a symbol it holds has left the raw tier; `rebuild_dropped=True`
rebuilds it from the raw tier and keeps the old store beside it. See [Datasets](datasets.md) for
how `update()` handles new listings.

Massive's stores are updated one by one, in dependency order: the Day aggregate store, the Ticker
map (a day is mapped only once Sharadar's SEP store holds it), then the stores mapped through it.
`massive_data.py` shows the order; [Massive](massive.md) describes each store.

### Extend the derived stores

A factor, label or risk-model store records the component that wrote it. Saving that component
beside the store with {py:func}`~quantlab.core.store_folder.save_component` writes its config to
`component.json`; {py:func}`~quantlab.core.store_folder.load_component` rebuilds it, with every
nested component, so an update needs no recipe code:

```python
from quantlab.core.store_folder import load_component, save_component

# once, when the store is built
save_component(factor, "/data/quantlab/factors/market/alpha158", readme="# alpha158\n...")

# every day: rebuild the factor from its folder and append the new bars
factor = load_component("/data/quantlab/factors/market/alpha158")
factor.extend("2026-10-16")
```

{py:meth}`Factor.extend <quantlab.factor.base.Factor.extend>` computes the new bars with the
factor's warm-up read from history, so a store extended day by day equals one built at once. A
factor risk model's regression and estimate stores extend the same way
({py:meth}`RiskStore.extend <quantlab.risk.base.RiskStore.extend>`), the regression first. Extend a store only through the component
that owns it, never through a view of some of its outputs, and extend the inputs first: the
whole-market alphas before the labels and neutralised stores that read them, the style exposures
before the risk model.

## Scheduling

Run the steps as one scheduled job after the vendors publish. Sharadar's daily tables are out early
on the next morning, New York time, and a Massive day's files the morning after it. On a machine
set to New York time:

```text
0 5 * * 1-5  cd $HOME/quantlab && . $HOME/.config/quantlab/data.env && flock -n $HOME/data_update.lock sh -c 'uv run python examples/data_update/sharadar_data.py && uv run python examples/data_update/massive_data.py' >> $HOME/data_update.log 2>&1
```

- cron does not read your shell profile, so the job must load `QUANTLAB_DATA_DIR`, the keys and a
  `PATH` holding `uv` itself; here they are `export` lines in a file only you can read (`chmod 600`).
- `flock -n` skips a run while the last one is still going: two runs must not write the same stores
  at once, and a first build can outlast the time between two runs.
- Each script exits 1 when a part failed, and every part is safe to run again, so a retry is the
  same command. The `&&` keeps Massive from running on a day Sharadar failed.
- A day the vendor has not published yet is not an error: it is picked up by the next run.

Have any job that reads the data wait for this job rather than updating data itself: a prediction
made on a partly updated day reads stores that disagree on the last bar.
