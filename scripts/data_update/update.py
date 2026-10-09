"""The daily data update: raw downloads -> Zarr stores -> factors -> risk model.

Runs the stages of a YAML file (``config/data_update.yaml``) in order, under the data
root ``--data-dir``. A store a step names is a folder ``<category>/<group>/<stem>/``
holding ``<stem>.zarr``, its README.md and a ``component.json``: the component's
``get_config()``, from which the step rebuilds the dataset, factor or factor risk model
(``quantlab.core.component.rebuild``). Actions:

- ``sharadar``: ``scripts/sharadar/update.py`` (every Sharadar table);

A step with ``allow_failure: true`` that raises is recorded as failed and the run goes
on; whether the day is ready is still decided by the ``ready`` stores.
- ``fred``: download the FRED series, then ``update()`` the store in the named folder;
- ``benchmarks``: ``scripts/sharadar/price_return_benchmark.py --refresh``;
- ``update``: ``update()`` the dataset of a store folder;
- ``mirror``: append a source dataset's new bars to a store copying its variables
  (``quantlab.backtest.live.mirror_new_bars``);
- ``extend``: extend a factor (only the store's owner may) or a factor risk model's
  regression and estimate stores to t;
- ``run``: any script under the repository, with ``args`` in which ``{data_dir}`` and
  ``{download_dir}`` are filled in, and ``writes``: the store folders or folder prefixes
  it writes (``market/wrds/``), which ``--check`` orders against the other steps.

t is the last bar of ``market/sharadar/sharadar_sep_1d`` once the raw stage is done.
Until every store of the file's ``ready`` list holds t, and t is newer than the last
successful run's, the raw stage is retried every ``--retry-minutes`` until
``--retry-until`` (New York time). The outcome is written to
``<data-dir>/update_status.json``: ``{"date", "t", "state", "started", "finished",
"steps"}``, ``state`` being ``running``, ``done``, ``no_new_bar`` (the vendor published
nothing new by the cut-off, as on a holiday) or ``failed``. The paper trading waits for
``done`` on its day.

Usage::

    QUANTLAB_DATA_DIR=/data/quantlab python scripts/data_update/update.py \\
        --data-dir /data/quantlab --download-dir /data/quantlab/downloads

``SHARADAR_API_KEY`` must be set for the raw stage. ``--stage NAME`` runs only the
named stages (repeatable), without the retry; ``--dry-run`` prints the plan;
``--check`` downloads and computes nothing: it rebuilds every step's component and
checks that a factor owns its store and that no step reads a store a later step
writes, and exits 1 listing the problems.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import xarray as xr
import yaml
from loguru import logger

from quantlab.acquisition.config import AcquisitionConfig
from quantlab.acquisition.fred import FredAcquisition
from quantlab.backtest.live import mirror_new_bars
from quantlab.core.component import walk_components
from quantlab.core.store_folder import COMPONENT_FILE, load_component
from quantlab.dataset.base import BaseDataset
from quantlab.factor.base import Factor
from quantlab.risk.base import FactorRiskModel
from quantlab.utils.cli import add_output_dir_args, resolve_output_dirs
from quantlab.utils.jsonable import to_jsonable

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_YORK = ZoneInfo("America/New_York")
#: The store whose last bar is the day t the update brings everything to.
CALENDAR_STORE = "market/sharadar/sharadar_sep_1d"
STATUS_FILE = "update_status.json"


def store_path(folder: Path) -> Path:
    """``<folder>/<stem>.zarr`` of a store folder ``.../<stem>/``."""
    return folder / f"{folder.name}.zarr"


def component(folder: Path):
    """Rebuild the dataset, factor or factor risk model of a store folder from its component.json."""
    return load_component(folder)


def stores_read(item) -> set[Path]:
    """The stores a component reads: every dataset's and factor's store below it."""
    paths = set()
    for _, part in walk_components(item):
        if not isinstance(part, (BaseDataset, Factor)):
            continue
        try:
            store = part.store_path
        except ValueError:  # a merged dataset is a view; its inputs are walked too
            continue
        if store:
            paths.add(Path(store).absolute())
    return paths


def last_bar(path: Path) -> pd.Timestamp | None:
    """The last timestamp of a store, ``None`` when it does not exist or is empty."""
    if not path.exists():
        return None
    store = xr.open_zarr(path)
    try:
        values = store["timestamp"].values
        return pd.Timestamp(values[-1]) if len(values) else None
    finally:
        store.close()


class Update:
    """One run of the stages over a data root."""

    def __init__(self, plan: dict, data_dir: Path, download_dir: Path, dry_run: bool):
        self.plan = plan
        self.data_dir = data_dir
        self.download_dir = download_dir
        self.dry_run = dry_run
        self.steps: list[dict] = []
        self.t: pd.Timestamp | None = None

    def folder(self, relative: str) -> Path:
        return self.data_dir / relative

    # -- actions ------------------------------------------------------------------

    def sharadar(self, args: list | None = None) -> str:
        self._script("scripts/sharadar/update.py", "--download-dir", str(self.download_dir),
                     "--data-dir", str(self.data_dir), *(args or []))
        return "updated"

    def fred(self, series: list, store: str, start: str = "1954-01-04") -> str:
        if not self.dry_run:
            FredAcquisition(AcquisitionConfig(
                market="us_equity", frequency="1d", vendor="fred",
                raw_data_dir_path=str(self.download_dir / "fred"),
                watermark_path=str(self.download_dir / "_watermarks" / "fred"),
                symbols=tuple(series), start_date=start,
            )).download()
        return self.update(store)

    def benchmarks(self, tickers: list) -> str:
        sfp = store_path(self.folder("market/sharadar/sharadar_sfp_1d"))
        self._script("scripts/sharadar/price_return_benchmark.py", "--tickers", ",".join(tickers),
                     "--sfp-store", str(sfp), "--raw-dir", str(self.download_dir / "sharadar"),
                     "--data-dir", str(self.data_dir), "--refresh")
        return "rebuilt"

    def run(self, script: str, args: list | None = None, writes: list | None = None) -> str:
        fill = {"data_dir": str(self.data_dir), "download_dir": str(self.download_dir)}
        self._script(script, *(str(a).format(**fill) for a in (args or [])))
        return "ran"

    def update(self, store: str) -> str:
        if self.dry_run:
            return "would update"
        component(self.folder(store)).update()
        return f"last bar {last_bar(store_path(self.folder(store)))}"

    def mirror(self, store: str, source: str) -> str:
        if self.dry_run:
            return "would mirror"
        return mirror_new_bars(component(self.folder(source)), store_path(self.folder(store)),
                               self.t.date().isoformat())

    def extend(self, store: str) -> str:
        if self.dry_run:
            return "would extend"
        day = self.t.date().isoformat()
        item = component(self.folder(store))
        if isinstance(item, FactorRiskModel):
            done = []
            for part in (item.regression, item.estimate):
                _, end = part.store_range()
                if pd.Timestamp(end) < self.t:
                    part.extend(day)
                done.append(part.store_range()[1])
            return f"ranges end {done}"
        if not isinstance(item, Factor):
            raise TypeError(f"{store}: extend takes a factor or a factor risk model, got {type(item).__name__}")
        if not item.owns_store():
            raise ValueError(f"{store}: the config is not the store's owner; write the owner's config")
        _, end = item.store_range()
        if pd.Timestamp(end) < self.t:
            # A date, so the recorded range holds the whole day (a Timestamp ends it at midnight).
            item.extend(day)
            return f"extended to {day}"
        return "current"

    # -- running ------------------------------------------------------------------

    def _script(self, script: str, *args: str) -> None:
        command = [sys.executable, str(REPO_ROOT / script), *args]
        logger.info(f"run {' '.join(command)}")
        if not self.dry_run:
            subprocess.run(command, check=True)

    def run_stage(self, stage: dict) -> None:
        for step in stage["steps"]:
            (action, value), = step.items()
            kwargs = dict(value) if isinstance(value, dict) else {"store": value}
            allow_failure = kwargs.pop("allow_failure", False)
            began = time.perf_counter()
            try:
                result = getattr(self, action)(**kwargs)
            except Exception as error:
                if not allow_failure:
                    raise
                # Readiness is decided by the `ready` stores, not by this step.
                logger.opt(exception=error).warning(f"{stage['name']}: {action} failed; continuing")
                result = f"failed: {error!r}"
            record = {"stage": stage["name"], "action": action, **kwargs, "result": result,
                      "seconds": round(time.perf_counter() - began, 1)}
            self.steps.append(record)
            logger.info(json.dumps(to_jsonable(record)))

    def writes(self, action: str, kwargs: dict) -> list[str]:
        """The store folders (or folder prefixes, ending in /) a step writes."""
        match action:
            case "sharadar":
                return ["market/sharadar/", "universe/sharadar/"]
            case "benchmarks":
                return ["market/benchmarks/"]
            case "run":
                return list(kwargs.get("writes", []))
            case _:
                return [kwargs["store"]]

    def check(self) -> list[str]:
        """Rebuild every step's component and check ownership and order; the problems found."""
        problems, steps = [], []
        for stage in self.plan["stages"]:
            for step in stage["steps"]:
                (action, value), = step.items()
                kwargs = {k: v for k, v in (value if isinstance(value, dict) else {"store": value}).items()
                          if k != "allow_failure"}
                where = f"{stage['name']}: {action} {kwargs.get('store', kwargs.get('script', ''))}".strip()
                if not hasattr(self, action) or action in ("check", "writes", "ready", "folder"):
                    problems.append(f"{where}: unknown action")
                    continue
                reads: set[Path] = set()
                for key in ("store", "source") if action in ("update", "extend", "mirror", "fred") else ():
                    if key not in kwargs or (action == "mirror" and key == "store"):
                        continue
                    folder = self.folder(kwargs[key])
                    if not (folder / COMPONENT_FILE).is_file():
                        problems.append(f"{where}: {folder / COMPONENT_FILE} is missing")
                        continue
                    try:
                        item = component(folder)
                    except Exception as error:
                        problems.append(f"{where}: {kwargs[key]} does not rebuild: {error!r}")
                        continue
                    if action == "extend" and isinstance(item, Factor) and not item.owns_store():
                        problems.append(f"{where}: the component is not the owner of its store")
                    if action == "extend" and not isinstance(item, (Factor, FactorRiskModel)):
                        problems.append(f"{where}: extend takes a factor or a factor risk model")
                    if action == "update" and not isinstance(item, BaseDataset):
                        problems.append(f"{where}: update takes a dataset")
                    own = store_path(folder).absolute()
                    reads |= {p for p in stores_read(item) | {store_path(folder).absolute()} if p != own}
                if action == "mirror":
                    reads.add(store_path(self.folder(kwargs["source"])).absolute())
                if action == "run" and not (REPO_ROOT / kwargs["script"]).is_file():
                    problems.append(f"{where}: no script {kwargs['script']}")
                steps.append((where, reads, self.writes(action, kwargs)))
        for i, (where, reads, _) in enumerate(steps):
            for later, _, written in steps[i + 1:]:
                for target in written:
                    root = self.folder(target).absolute()
                    hit = [p for p in reads if p == store_path(root) or (target.endswith("/") and root in p.parents)]
                    if hit:
                        problems.append(f"{where} reads {hit[0]}, which a later step writes ({later})")
        return problems

    def ready(self) -> list[str]:
        """The ``ready`` stores that do not hold t."""
        return [s for s in self.plan.get("ready", [])
                if (bar := last_bar(store_path(self.folder(s)))) is None or bar < self.t]


def write_status(path: Path, status: dict) -> None:
    status = {**status, "updated": datetime.datetime.now(NEW_YORK).isoformat(timespec="seconds")}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(to_jsonable(status), indent=2))
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_output_dir_args(parser)
    parser.add_argument("--config", default=str(REPO_ROOT / "config" / "data_update.yaml"))
    parser.add_argument("--stage", action="append", help="Run only these stages (no retry).")
    parser.add_argument("--retry-minutes", type=int, default=15)
    parser.add_argument("--retry-until", default="08:30", help="New York time, HH:MM.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check the plan, run nothing.")
    args = parser.parse_args()
    download_dir, data_dir = resolve_output_dirs(args)
    plan = yaml.safe_load(Path(args.config).read_text())
    if args.check:
        problems = Update(plan, data_dir, download_dir, dry_run=True).check()
        for problem in problems:
            print(f"PROBLEM {problem}")
        print(f"{len(problems)} problem(s) in {args.config}")
        sys.exit(1 if problems else 0)
    stages = [s for s in plan["stages"] if not args.stage or s["name"] in args.stage]
    status_path = data_dir / STATUS_FILE
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    today = datetime.datetime.now(NEW_YORK).date().isoformat()
    status = {"date": today, "state": "running", "started": datetime.datetime.now(NEW_YORK).isoformat(timespec="seconds")}
    run = Update(plan, data_dir, download_dir, args.dry_run)
    if not args.dry_run and not args.stage:
        write_status(status_path, {**previous, **status})
    try:
        raw, rest = (stages[0], stages[1:]) if stages and stages[0]["name"] == "raw" else (None, stages)
        while raw is not None:
            run.run_stage(raw)
            run.t = last_bar(store_path(run.folder(CALENDAR_STORE)))
            waiting = run.ready() if run.t is not None else []
            new = previous.get("state") != "done" or previous.get("t") is None or run.t > pd.Timestamp(previous["t"])
            if args.stage or args.dry_run or (not waiting and new):
                break
            why = f"stores short of {run.t.date()}: {waiting}" if waiting else f"no bar after {previous.get('t')}"
            if datetime.datetime.now(NEW_YORK).strftime("%H:%M") >= args.retry_until:
                logger.warning(f"{why}; past {args.retry_until}, giving up")
                write_status(status_path, {**status, "t": str(run.t.date()), "state": "no_new_bar" if not waiting else "failed",
                                           "reason": why, "steps": run.steps})
                return
            logger.info(f"{why}; retrying in {args.retry_minutes} min")
            time.sleep(args.retry_minutes * 60)
        if run.t is None:
            run.t = last_bar(store_path(run.folder(CALENDAR_STORE)))
        for stage in rest:
            run.run_stage(stage)
    except Exception as error:
        logger.exception("data update failed")
        if not args.dry_run and not args.stage:
            write_status(status_path, {**status, "t": str(run.t.date()) if run.t is not None else None,
                                       "state": "failed", "reason": repr(error), "steps": run.steps})
        raise
    if not args.dry_run and not args.stage:
        write_status(status_path, {**status, "t": str(run.t.date()), "state": "done",
                                   "finished": datetime.datetime.now(NEW_YORK).isoformat(timespec="seconds"),
                                   "steps": run.steps})
    logger.info(f"data update done to {run.t.date() if run.t is not None else None}")


if __name__ == "__main__":
    main()
