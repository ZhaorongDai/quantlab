"""PROTOTYPE, throwaway: the follow-up of bench.py.

1. Fixed ROUNDS rounds, no early stopping, so the cuts are the only thing that
   can differ: DMatrix against QuantileDMatrix from GPU blocks (several fixed
   row counts), and a QuantileDMatrix built with ``ref=`` today's DMatrix
   (the same cuts, quantized on the GPU) to show the cuts carry all of it.
2. CPU threads: build times of DMatrix and of QuantileDMatrix from numpy at
   nthread 1, 16 and 64 on a SUBSET_ROWS slice, with the process affinity and
   the thread environment, to see whether bench.py's CPU part ran on one thread.

Run like bench.py; writes ``bench2.json`` beside this file.
"""

import dataclasses
import json
import os
import time
from pathlib import Path

import bench
import cupy as cp
import numpy as np
import xgboost as xgb
from bench import Blocks, GpuPeak, cuts, hf

from quantlab.model.predefined.xgb import XGBoostRegressor
from quantlab.model.split import walk_forward_folds

ROUNDS = 150
BLOCK_ROWS = (500_000, 1_000_000, 2_000_000)
SUBSET_ROWS = 2_000_000
THREADS = (1, 16, 64)
bench.OUT = Path(__file__).with_name("bench2.json")
report = bench.report


class Bench2(XGBoostRegressor):
    boosters: dict

    def _train(self, params, dtrain, dval):
        return xgb.train(params, dtrain, num_boost_round=ROUNDS, evals=[(dval, "val")], verbose_eval=False)

    def _fit_model(self, train_rows, val_rows):
        self.boosters = {}
        params = dict(self._params)
        report("env", {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                                       "OMP_PROC_BIND", "OMP_PLACES")}
               | {"affinity": len(os.sched_getaffinity(0)), "params": params})

        with GpuPeak() as build:
            dtrain = xgb.DMatrix(train_rows.x, label=train_rows.y)
            dval = xgb.DMatrix(val_rows.x, label=val_rows.y)
        with GpuPeak() as fit:
            self.boosters["dmatrix"] = self._train(params, dtrain, dval)
        base_cuts = cuts(dtrain)
        report("gpu_dmatrix", {"build_s": build.seconds, "fit_s": fit.seconds, "fit_peak_mib": fit.peak})

        # The same cuts as today's DMatrix, quantized from GPU blocks.
        try:
            with GpuPeak() as build:
                qtrain = xgb.QuantileDMatrix(Blocks(train_rows.x, train_rows.y, 1_000_000), ref=dtrain)
                qval = xgb.QuantileDMatrix(Blocks(val_rows.x, val_rows.y, 1_000_000), ref=dtrain)
            with GpuPeak() as fit:
                self.boosters["qdm_ref_dmatrix"] = self._train(params, qtrain, qval)
            indptr, values = cuts(qtrain)
            report("gpu_qdm_ref_dmatrix", {"build_s": build.seconds, "fit_s": fit.seconds,
                                           "cuts_identical_to_dmatrix": bool(np.array_equal(values, base_cuts[1]))})
            del qtrain, qval
        except Exception as error:  # noqa: BLE001 - the prototype records it
            report("gpu_qdm_ref_dmatrix", {"error": f"{type(error).__name__}: {error}"})
        del dtrain, dval
        cp.get_default_memory_pool().free_all_blocks()

        for rows in BLOCK_ROWS:
            with GpuPeak() as build:
                qtrain = xgb.QuantileDMatrix(Blocks(train_rows.x, train_rows.y, rows), max_bin=params.get("max_bin", 256))
                qval = xgb.QuantileDMatrix(Blocks(val_rows.x, val_rows.y, rows), ref=qtrain)
            with GpuPeak() as fit:
                self.boosters[f"qdm_{rows}"] = self._train(params, qtrain, qval)
            indptr, values = cuts(qtrain)
            same = len(values) == len(base_cuts[1])
            report(f"gpu_qdm_{rows}", {
                "batches": -(-len(train_rows.x) // rows), "build_s": build.seconds, "build_peak_mib": build.peak,
                "fit_s": fit.seconds, "fit_peak_mib": fit.peak,
                "cuts_identical_to_dmatrix": bool(same and np.array_equal(values, base_cuts[1])),
                "num_cuts": [len(values), len(base_cuts[1])],
                "cuts_differing_share": float(np.mean(values != base_cuts[1])) if same else None})
            del qtrain, qval
            cp.get_default_memory_pool().free_all_blocks()

        # CPU threads on a slice.
        x, y = train_rows.x[:SUBSET_ROWS], train_rows.y[:SUBSET_ROWS]
        timings = {}
        for n in THREADS:
            started = time.perf_counter()
            xgb.DMatrix(x, label=y, nthread=n)
            dm = time.perf_counter() - started
            started = time.perf_counter()
            xgb.QuantileDMatrix(x, label=y, nthread=n, max_bin=params.get("max_bin", 256))
            timings[n] = {"dmatrix_s": round(dm, 1), "qdm_s": round(time.perf_counter() - started, 1)}
            report("cpu_threads", {"rows": SUBSET_ROWS, "timings": timings})
        self.model = self.boosters["dmatrix"]

    def _split_loss(self, split, panel, bars):
        if split == "test":
            import torch
            from scipy.stats import rankdata
            cells = panel.present & self._bar_mask(panel, bars)[:, None]
            t, s = torch.nonzero(cells, as_tuple=True)
            x = panel.x[t, s].numpy()
            y = panel.y_raw[t, s, 0].numpy()
            t = t.numpy()
            predictions = {name: b.inplace_predict(x) for name, b in self.boosters.items()}
            base = predictions["dmatrix"]
            for name, p in predictions.items():
                ics = [np.corrcoef(rankdata(p[on]), rankdata(y[on]))[0, 1]
                       for bar in np.unique(t) if (on := (t == bar) & np.isfinite(y)).sum() > 10]
                report(f"test_{name}", {"rank_ic": float(np.mean(ics)), "identical_to_dmatrix": bool(np.array_equal(p, base)),
                                        "max_abs_diff": float(np.max(np.abs(p - base))),
                                        "corr": float(np.corrcoef(p, base)[0, 1])})
        return super()._split_loss(split, panel, bars)


def main():
    head = hf.model("ALLMH", "xgb_base", 0)
    model = Bench2(dataclasses.replace(head.config, model_save_dir=str(hf.WORK / "prototype_quantile_dmatrix2")))
    model.collect()
    train_periods = int(np.searchsorted(model.walk_forward_bars(), np.datetime64(hf.FIRST_TEST)))
    folds = walk_forward_folds(model.walk_forward_bars(), train_periods, test_periods=hf.TEST_PERIODS,
                               expanding=True, purge_bars=model.purge_bars)
    fold = folds[bench.FOLD]
    model.train_fold(fold, Path(model.config.model_save_dir) / f"fold_{bench.FOLD}", "prototype")


if __name__ == "__main__":
    main()
