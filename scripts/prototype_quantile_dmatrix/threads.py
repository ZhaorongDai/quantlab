"""PROTOTYPE, throwaway: where does xgboost's DMatrix build lose its threads in the training process?

Times ``xgb.DMatrix`` at nthread 1 and 64 on a fixed random array (and, in
``_fit_model``, on the real training rows) after each stage: import, collect,
building the rows. Prints the OpenMP pools threadpoolctl sees and torch's
thread count each time. Run like bench.py.
"""

import dataclasses
import time
from pathlib import Path

import bench
import numpy as np
import torch
import xgboost as xgb
from bench import hf
from threadpoolctl import threadpool_info

from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.split import walk_forward_folds

PROBE = np.random.default_rng(0).standard_normal((2_000_000, 360), dtype=np.float32)


def probe(stage, x=PROBE):
    times = {}
    for n in (1, 64):
        started = time.perf_counter()
        xgb.DMatrix(x[:2_000_000], nthread=n)
        times[n] = round(time.perf_counter() - started, 1)
    pools = [(i["internal_api"], i["num_threads"], Path(i["filepath"]).parent.parent.name) for i in threadpool_info()]
    flags = {k: getattr(x, "flags")[k] for k in ("C_CONTIGUOUS", "OWNDATA")} | {"dtype": str(x.dtype)}
    print(f"[{stage}] dmatrix 1 thread {times[1]} s, 64 threads {times[64]} s; torch {torch.get_num_threads()}; "
          f"array {flags}; pools {pools}", flush=True)


class Probe(XGBoostRegressor):
    def _fit_model(self, train_rows, val_rows):
        probe("in _fit_model, random array")
        probe("in _fit_model, training rows", train_rows.x)
        probe("in _fit_model, copy of training rows", np.ascontiguousarray(train_rows.x[:2_000_000]).copy())
        raise SystemExit("probe done")


def main():
    probe("after import")
    head = hf.model("ALLMH", "xgb_base", 0)
    model = Probe(dataclasses.replace(head.config, model_save_dir=str(hf.WORK / "prototype_quantile_dmatrix_threads")))
    model.collect()
    probe("after collect")
    train_periods = int(np.searchsorted(model.walk_forward_bars(), np.datetime64(hf.FIRST_TEST)))
    folds = walk_forward_folds(model.walk_forward_bars(), train_periods, test_periods=hf.TEST_PERIODS,
                               expanding=True, purge_bars=model.purge_bars)
    model.train_fold(folds[bench.FOLD], Path(model.config.model_save_dir) / f"fold_{bench.FOLD}", "prototype")


if __name__ == "__main__":
    main()
