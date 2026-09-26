---
status: accepted
date: 2026-09-25
---

# A resampled panel is saved beside its source and read in preference to it

`resample(freq, how)` on a dataset or factor returns a copy whose config carries
`resample_freq` and `resample_how`. The copy is a view of the source store: it cannot be built
from raw files, updated or streamed. When it is saved, the panel goes to `store_path`, a store
beside the source named `<stem>_resample_<freq>.zarr`, never to the source path. When such a
store exists, `read()` opens it and does not resample; when it does not, `read()` opens the source
store and resamples it in memory.

We decided this rather than keeping resampled panels in memory only, or recording the resampled
store's path as a config field, for three reasons. Resampling a minute panel is slow enough that a
research loop over a daily factor should not repeat it on every read. A derived name keeps the
`config.json` of a run pointing at the source store, so a saved run still says where the bars came
from, and a rebuilt object finds the cache without a second path field that could drift from the
first. And writing to the source path would let a daily panel silently overwrite the minute store it
was made from.

The store name carries the frequency but not the aggregation methods: two resamples of one source
to the same frequency with different `how` share one path, and the second `save()` replaces the
first.

## Consequences

- The resampled store is a cache in the same sense a factor store is: rebuilding or updating the
  source store does not refresh it. Delete it, or `save()` again from a freshly resampled copy.
- A resampled object with the fields set reads the cache even when the source has newer bars.
  `read(overwrite=True)` re-reads the cache, not the source.
- Different `how` mappings at one frequency need different source stores or a manual rename;
  no path field is offered for this.
