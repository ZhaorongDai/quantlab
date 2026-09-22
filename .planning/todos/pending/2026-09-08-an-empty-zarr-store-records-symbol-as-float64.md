---
created: "2026-09-08T00:00:00.000Z"
title: An empty zarr store records symbol as float64
area: data
severity: minor
status: pending
---

# An empty zarr store records `symbol` as float64

## What was noticed

`data/data/us_equity/1d/stock_alpaca.zarr` records its `symbol` coordinate as
`float64`. A symbol coordinate should never be floating point.

Noticed during 260908-dvv while surveying which symbol-coordinate encodings are
live on real stores, and deliberately **not** investigated further — the brief
for that task was test integrity, not data repair. It is recorded here so the
observation is not lost.

## Diagnosis (not merely a hypothesis)

**The store is EMPTY.** Measured 2026-09-08: `{timestamp: 0, symbol: 0}`, while
carrying the full Alpaca variable set. So there is no symbol data in it at all,
and the `float64` is not a corrupted symbol coordinate — it is the dtype numpy
assigns when there are no strings to infer one from.

Reproduced from first principles in three lines (numpy 2.5.2 / xarray 2026.7.0 /
zarr 3.3.0):

```python
>>> np.asarray([]).dtype
dtype('float64')

>>> xr.Dataset(
...     {"close": (["timestamp", "symbol"], np.zeros((0, 0)))},
...     coords={"timestamp": pd.to_datetime([]), "symbol": []},
... ).to_zarr(path, mode="w")
>>> zarr.open_group(path, mode="r")["symbol"].dtype
dtype('float64')
```

An empty python list carries no string information, so numpy defaults its dtype
to `float64`, and writing a 0x0 panel reproduces the store's `symbol` dtype
exactly. This confirms the hypothesis the original todo
(`2026-09-08-widening-fixtures-bypass-the-real-coordinate-encoding-path.md`)
offered — "most likely an empty store whose coordinate was never populated" —
rather than leaving it open.

It is therefore **not a third string encoding**. 260908-dvv parametrised the
three owning widening suites over the two encodings that are genuinely live
(fixed-width unicode / `BytesCodec`, and object-encoded variable length /
`VLenUTF8Codec`) and deliberately excluded this one: a degenerate 0x0 SHAPE is
not a string encoding, and a widening test parametrised over it would be
asserting something about emptiness rather than about dtype.

## The one question left open

**Should an empty store be written at all?** Two sub-questions, and this todo
takes no position on either:

1. If a run produces no rows, is writing a 0x0 store the right outcome, or
   should the write be skipped so the absence of a store means "no data" rather
   than "a store that says nothing"? Downstream, `xr.open_zarr` on this store
   succeeds and yields an empty panel, so nothing currently fails loudly.
2. If an empty store IS the right outcome, should its `symbol` coordinate be
   pinned to a string dtype on the way out — so that a later append into it
   meets a string coordinate rather than a `float64` one? Note that
   `XrBackend.append`'s `_assert_append_compatible` compares coordinate dtypes,
   so the first non-empty append into this store is a plausible place for it to
   surface.

## Files

- `data/data/us_equity/1d/stock_alpaca.zarr` — the store in question
- `quantlab/base/data.py` — `from_raw_data_chunked` / `_raw_data_to_xr_window`,
  where an empty window would be produced
- `quantlab/backend.py` — `XrBackend.append`, whose dtype check is
  where the consequence would appear

## Discovered

2026-09-08, during 260908-dvv, while re-deriving which symbol dtypes are live on
disk. Not investigated further, per that task's brief.
