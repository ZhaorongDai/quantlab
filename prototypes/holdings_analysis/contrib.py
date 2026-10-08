# PROTOTYPE, wipe me: per-bar holding contributions for the Holdings-analysis prototype.
import json, sys
import numpy as np, pandas as pd, xarray as xr
from quantlab.backtest.predefined.us_equity import USEquityCrossectionSelectStockVectorBt as B
R = sys.argv[1]
col = B.MARKET.valuation_price_column
h = xr.open_zarr(R + "/holdings.zarr")["holding"].transpose("timestamp", "symbol").load()
eq = xr.open_zarr(R + "/equity.zarr").load()
cfg = json.load(open(R + "/config.json"))
pdcfg = cfg["config"]["price_dataset"] if "config" in cfg else cfg["price_dataset"]
px = xr.open_zarr(pdcfg["zarr_file_path"])[col].sel(timestamp=h.timestamp, symbol=h.symbol).transpose("timestamp", "symbol").load()
p = px.to_pandas().ffill().to_numpy(float)
r = np.zeros_like(p); r[1:] = p[1:] / p[:-1] - 1
hv = np.nan_to_num(h.values.astype(float))
w = np.zeros_like(hv); w[1:] = hv[:-1]
c = np.where(w != 0, w * np.nan_to_num(r), 0.0)
nav = eq["returns"].values.astype(float)
T, S = c.shape
dec = np.zeros((T, 10)); cnt = np.zeros(T, int)
for i in range(T):
    held = np.flatnonzero(w[i] > 0)
    if held.size == 0: continue
    order = held[np.argsort(-w[i, held], kind="stable")]
    k = (np.arange(order.size) * 10) // order.size
    np.add.at(dec[i], k, c[i, order]); cnt[i] = order.size
years = pd.DatetimeIndex(h.timestamp.values).year.values
sym_total = c.sum(0); keep = np.flatnonzero(np.abs(w).sum(0) > 0)
yrs = sorted(set(years.tolist()))
sym_year = {str(y): c[years == y][:, keep].sum(0).round(8).tolist() for y in yrs}
held_days = (w[:, keep] > 0).sum(0).tolist()
avg_w = (w[:, keep].sum(0) / np.maximum(1, (w[:, keep] > 0).sum(0))).round(6).tolist()
out = dict(dates=[str(d)[:10] for d in h.timestamp.values], nav=np.nan_to_num(nav).round(8).tolist(),
           dec=dec.round(8).tolist(), cnt=cnt.tolist(), years=[int(y) for y in yrs],
           symbols=[str(s) for s in h.symbol.values[keep]], sym_total=sym_total[keep].round(8).tolist(),
           sym_year=sym_year, held_days=held_days, avg_w=avg_w, valuation=col)
json.dump(out, open("/data/quantlab/scratch/holdings_proto/contrib.json", "w"), separators=(",", ":"))
tot = dec.sum(0); print("valuation", col, "T", T, "names", keep.size)
print("decile sums", (tot*100).round(2).tolist(), "nav sum", round(nav.sum()*100,2), "residual", round((nav.sum()-tot.sum())*100,2))
