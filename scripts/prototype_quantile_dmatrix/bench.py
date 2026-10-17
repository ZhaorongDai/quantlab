"""PROTOTYPE, throwaway: QuantileDMatrix (GPU blocks / CPU numpy) against today's DMatrix in XGBoostRegressor.

Answers, on one real us3000 fold (ALLMH features, fold 6 of the 2026-10-16
experiment's walk-forward, ~17M training rows x 360 features), the questions
of the 2026-10-17 grilling:

1. speed: building the training and validation matrices and training, for
   today's ``xgb.DMatrix`` and for ``xgb.QuantileDMatrix`` fed by an
   ``xgb.DataIter`` that copies block-sized pieces of the rows to the GPU as
   cupy arrays, at several block budgets;
2. GPU memory: the device peak of each build and fit (nvidia-smi polled), and
   the quantized matrix's size, to calibrate the pre-check;
3. numbers: the quantile cuts, the test predictions and the test rank IC
   against today's path; whether the block budget changes anything;
4. CPU: on device "cpu", DMatrix against QuantileDMatrix built from numpy
   (build time, and predictions after a few rounds);
5. the CCC objective and early stopping run on a QuantileDMatrix.

Run on the training server with the GPU free::

    QUANTLAB_DATA_DIR=/data/quantlab OPENBLAS_NUM_THREADS=1 \
        numactl --cpunodebind=1 --preferred=1 python scripts/prototype_quantile_dmatrix/bench.py

Prints every measurement as it goes and writes ``bench.json`` beside this file.
"""

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np
import torch
import xgboost as xgb
from scipy.stats import rankdata

EXPERIMENT = Path.home() / "projects" / "quantlab2" / "experiments" / "2026-10-16-us3000-market-wide-hf-xgb" / "scripts"
sys.path.insert(0, str(EXPERIMENT))
import market_hf as hf

from quantlab.model.predefined.xgb import (
    XGBoostRegressor,
    ccc_loss_metric,
    ccc_objective,
)
from quantlab.model.split import walk_forward_folds

FOLD = 6
BUDGETS_GIB = (0.5, 1.0, 2.0, 4.0)
CPU_ROUNDS = 30
OUT = Path(__file__).with_name("bench.json")
results: dict = {}


def report(key, value):
    results[key] = value
    print(f"{key}: {json.dumps(value, default=str)}", flush=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))


class GpuPeak:
    """The process's largest GPU memory (MiB) while the block runs, polled from nvidia-smi."""

    def __enter__(self):
        self.peak, self.stop = 0, threading.Event()
        pid = str(os.getpid())

        def poll():
            while not self.stop.is_set():
                out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False).stdout
                for line in out.splitlines():
                    p, used = (part.strip() for part in line.split(","))
                    if p == pid:
                        self.peak = max(self.peak, int(used))
                time.sleep(0.2)

        self.thread = threading.Thread(target=poll, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self.started, 1)
        self.stop.set()
        self.thread.join()


class Blocks(xgb.DataIter):
    """The rows, ``block_rows`` at a time, copied to the GPU as cupy arrays.

    xgboost 3.4.1 reads every GPU array through cupy (``_transform_cupy_array``),
    so torch CUDA tensors alone fail with ``No module named 'cupy'``.
    """

    def __init__(self, x, y, block_rows):
        self.x, self.y, self.block, self.i = x, y, block_rows, 0
        super().__init__()

    def next(self, input_data):
        if self.i >= len(self.x):
            return False
        part = slice(self.i, self.i + self.block)
        input_data(data=cp.asarray(self.x[part]), label=cp.asarray(self.y[part]))
        self.i += self.block
        return True

    def reset(self):
        self.i = 0


def block_rows(budget_gib, num_features):
    return max(1, int(budget_gib * 2**30 // (4 * num_features)))


def cuts(matrix):
    indptr, values = matrix.get_quantile_cut()
    return np.asarray(indptr), np.asarray(values)


class Bench(XGBoostRegressor):
    """Runs every variant inside the fold's fit; the fold keeps today's booster."""

    boosters: dict

    def _train(self, params, dtrain, dval, rounds, early=True, obj=None):
        callbacks = [xgb.callback.EarlyStopping(rounds=self.early_stopping_patience, data_name="val",
                                                metric_name="rmse", save_best=True)] if early else []
        return xgb.train(params, dtrain, num_boost_round=rounds, evals=[(dval, "val")], callbacks=callbacks,
                         obj=obj, custom_metric=ccc_loss_metric if obj else None, verbose_eval=False)

    def _fit_model(self, train_rows, val_rows):
        self.boosters = {}
        params = dict(self._params)
        rounds = self._num_boost_round
        num_rows, num_features = train_rows.x.shape
        report("shape", {"train_rows": num_rows, "val_rows": len(val_rows.x), "features": num_features,
                         "train_gib": round(train_rows.x.nbytes / 2**30, 2),
                         "nan_share": float(np.isnan(train_rows.x).mean()), "params": params, "rounds": rounds})

        # 1. Today's path.
        with GpuPeak() as build:
            dtrain = xgb.DMatrix(train_rows.x, label=train_rows.y)
            dval = xgb.DMatrix(val_rows.x, label=val_rows.y)
        with GpuPeak() as fit:
            booster = self._train(params, dtrain, dval, rounds)
        self.boosters["dmatrix"] = booster
        base_cuts = cuts(dtrain)
        report("gpu_dmatrix", {"build_s": build.seconds, "build_peak_mib": build.peak, "fit_s": fit.seconds,
                               "fit_peak_mib": fit.peak, "best_iteration": booster.best_iteration,
                               "best_score": booster.best_score})
        del dtrain, dval

        # 2. QuantileDMatrix from GPU blocks.
        for budget in BUDGETS_GIB:
            rows = block_rows(budget, num_features)
            with GpuPeak() as build:
                qtrain = xgb.QuantileDMatrix(Blocks(train_rows.x, train_rows.y, rows), max_bin=params.get("max_bin", 256))
                qval = xgb.QuantileDMatrix(Blocks(val_rows.x, val_rows.y, rows), ref=qtrain)
            with GpuPeak() as fit:
                booster = self._train(params, qtrain, qval, rounds)
            self.boosters[f"qdm_gpu_{budget:g}"] = booster
            indptr, values = cuts(qtrain)
            same_cuts = bool(np.array_equal(indptr, base_cuts[0]) and np.array_equal(values, base_cuts[1]))
            report(f"gpu_qdm_{budget:g}gib", {
                "block_rows": rows, "build_s": build.seconds, "build_peak_mib": build.peak, "fit_s": fit.seconds,
                "fit_peak_mib": fit.peak, "best_iteration": booster.best_iteration, "best_score": booster.best_score,
                "cuts_identical_to_dmatrix": same_cuts,
                "cuts_max_abs_diff": None if same_cuts or len(values) != len(base_cuts[1])
                else float(np.max(np.abs(values - base_cuts[1]))),
                "num_cuts": [len(values), len(base_cuts[1])]})
            if budget == 1.0:
                # 5. The CCC objective with its metric, and early stopping, on the quantized matrix.
                ccc = {k: v for k, v in params.items() if k != "objective"}
                ccc["base_score"] = float(np.mean(train_rows.y, dtype=np.float64))
                started = time.perf_counter()
                try:
                    b = self._train({**ccc, "disable_default_eval_metric": 1}, qtrain, qval, 20, early=False,
                                    obj=ccc_objective)
                    report("ccc_on_qdm", {"ok": True, "rounds": b.num_boosted_rounds(),
                                          "seconds": round(time.perf_counter() - started, 1)})
                except Exception as error:  # noqa: BLE001 - the prototype records it
                    report("ccc_on_qdm", {"ok": False, "error": f"{type(error).__name__}: {error}"})
            del qtrain, qval
            cp.get_default_memory_pool().free_all_blocks()

        # 4. CPU: DMatrix against QuantileDMatrix from numpy, a few rounds.
        cpu = {**params, "device": "cpu", "nthread": 64}
        started = time.perf_counter()
        ctrain, cval = xgb.DMatrix(train_rows.x, label=train_rows.y, nthread=64), xgb.DMatrix(val_rows.x, label=val_rows.y, nthread=64)
        dm_build = time.perf_counter() - started
        started = time.perf_counter()
        cpu_dm = self._train(cpu, ctrain, cval, CPU_ROUNDS, early=False)
        dm_fit = time.perf_counter() - started
        del ctrain, cval
        started = time.perf_counter()
        qtrain = xgb.QuantileDMatrix(train_rows.x, label=train_rows.y, nthread=64, max_bin=params.get("max_bin", 256))
        qval = xgb.QuantileDMatrix(val_rows.x, label=val_rows.y, ref=qtrain, nthread=64)
        qdm_build = time.perf_counter() - started
        started = time.perf_counter()
        cpu_qdm = self._train(cpu, qtrain, qval, CPU_ROUNDS, early=False)
        qdm_fit = time.perf_counter() - started
        probe = train_rows.x[:: max(1, num_rows // 500_000)]
        a, b = cpu_dm.inplace_predict(probe), cpu_qdm.inplace_predict(probe)
        report("cpu", {"rounds": CPU_ROUNDS, "dmatrix_build_s": round(dm_build, 1), "dmatrix_fit_s": round(dm_fit, 1),
                       "qdm_build_s": round(qdm_build, 1), "qdm_fit_s": round(qdm_fit, 1),
                       "predictions_identical": bool(np.array_equal(a, b)),
                       "max_abs_diff": float(np.max(np.abs(a - b))), "corr": float(np.corrcoef(a, b)[0, 1])})
        del qtrain, qval

        self.model = self.boosters["dmatrix"]

    def _split_loss(self, split, panel, bars):
        if split == "test":
            cells = panel.present & self._bar_mask(panel, bars)[:, None]
            t, s = torch.nonzero(cells, as_tuple=True)
            x = panel.x[t, s].numpy()
            y = panel.y_raw[t, s, 0].numpy()
            t = t.numpy()
            predictions = {name: b.inplace_predict(x) for name, b in self.boosters.items()}
            base = predictions["dmatrix"]
            for name, p in predictions.items():
                ics = []
                for bar in np.unique(t):
                    on = (t == bar) & np.isfinite(y)
                    if on.sum() > 10:
                        ics.append(np.corrcoef(rankdata(p[on]), rankdata(y[on]))[0, 1])
                report(f"test_{name}", {"rank_ic": float(np.mean(ics)), "identical_to_dmatrix": bool(np.array_equal(p, base)),
                                        "max_abs_diff": float(np.max(np.abs(p - base))),
                                        "corr": float(np.corrcoef(p, base)[0, 1])})
        return super()._split_loss(split, panel, bars)


def main():
    head = hf.model("ALLMH", "xgb_base", 0)
    model = Bench(dataclasses.replace(head.config, model_save_dir=str(hf.WORK / "prototype_quantile_dmatrix")))
    model.collect()
    train_periods = int(np.searchsorted(model.walk_forward_bars(), np.datetime64(hf.FIRST_TEST)))
    folds = walk_forward_folds(model.walk_forward_bars(), train_periods, test_periods=hf.TEST_PERIODS,
                               expanding=True, purge_bars=model.purge_bars)
    fold = folds[FOLD]
    report("fold", {"features": "ALLMH", "index": fold.index, "train": fold.train_window, "test": fold.test_window,
                    "xgboost": xgb.__version__})
    model.train_fold(fold, Path(model.config.model_save_dir) / f"fold_{FOLD}", "prototype")


if __name__ == "__main__":
    main()
