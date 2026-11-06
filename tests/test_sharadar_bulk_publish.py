"""A Sharadar bulk pull commits in one rename.

The bulk file's name carries its pull time, so readers take the newest bulk and
ignore the window and updated pulls older than it. Nothing else has to happen in
order: what is left after the rename is repeatable cleanup. These tests stop a
publish directly and judge it by what ``scan_raw_table`` returns.
"""

import os
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from quantlab.acquisition.sharadar.raw_tier import bulk_scratch, publish_bulk
from quantlab.dataset.sharadar.tables import (
    bulk_file,
    latest_bulk,
    pull_time,
    raw_files,
    scan_raw_table,
    window_file,
)

MONDAY = datetime(2024, 1, 8, 8, 30, tzinfo=UTC)
TUESDAY = datetime(2024, 1, 9, 8, 30, tzinfo=UTC)
WEDNESDAY = datetime(2024, 1, 10, 8, 30, tzinfo=UTC)
DAY = date(2024, 1, 5)


def _rows(close):
    return pl.DataFrame(
        {"ticker": ["AAA"], "date": [DAY], "close": [close]},
        schema={"ticker": pl.String, "date": pl.Date, "close": pl.Float64},
    )


def _publish(root, pulled_at, close):
    with bulk_scratch(root, "sep") as scratch:
        return publish_bulk(root, "sep", pulled_at, _rows(close).lazy(), scratch)


def _window(root, pulled_at, close):
    path = window_file(root, "sep", pulled_at, DAY, DAY)
    path.parent.mkdir(parents=True, exist_ok=True)
    _rows(close).write_parquet(path)
    return path


def _empty_path(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _closes(root):
    return scan_raw_table(root, "sep").select("close").collect()["close"].to_list()


def test_a_published_bulk_reads_and_is_named_by_its_pull_time(tmp_path):
    path = _publish(tmp_path, MONDAY, 10.0)

    assert path == bulk_file(tmp_path, "sep", MONDAY) == latest_bulk(tmp_path, "sep")
    assert pull_time(path) == MONDAY
    assert _closes(tmp_path) == [10.0]


def test_a_bulk_killed_before_the_commit_leaves_the_old_data(tmp_path, monkeypatch):
    _publish(tmp_path, MONDAY, 10.0)
    _window(tmp_path, TUESDAY, 11.0)

    def killed(source, destination):
        raise OSError("killed before the rename")

    monkeypatch.setattr("quantlab.acquisition.sharadar.raw_tier.move_durably", killed)
    with pytest.raises(OSError):
        _publish(tmp_path, WEDNESDAY, 99.0)

    assert latest_bulk(tmp_path, "sep") == bulk_file(tmp_path, "sep", MONDAY)
    assert _closes(tmp_path) == [11.0]


def test_a_bulk_killed_right_after_the_commit_reads_new_with_nothing_older_over_it(tmp_path, monkeypatch):
    _publish(tmp_path, MONDAY, 10.0)
    stale = _window(tmp_path, TUESDAY, 11.0)

    def killed(vendor_root, code):
        raise OSError("killed after the rename")

    monkeypatch.setattr("quantlab.acquisition.sharadar.raw_tier._remove_superseded", killed)
    with pytest.raises(OSError):
        _publish(tmp_path, WEDNESDAY, 12.0)

    assert stale.exists()  # left behind, and ignored
    assert raw_files(tmp_path, "sep") == [bulk_file(tmp_path, "sep", WEDNESDAY)]
    assert _closes(tmp_path) == [12.0]


def test_a_pull_made_after_the_bulk_started_still_overlays_it(tmp_path):
    _publish(tmp_path, TUESDAY, 10.0)
    _window(tmp_path, WEDNESDAY, 11.0)

    assert _closes(tmp_path) == [11.0]


def test_the_next_bulk_finishes_the_cleanup(tmp_path, monkeypatch):
    _publish(tmp_path, MONDAY, 10.0)
    _window(tmp_path, TUESDAY, 11.0)
    with monkeypatch.context() as patch:
        patch.setattr(
            "quantlab.acquisition.sharadar.raw_tier._remove_superseded",
            lambda vendor_root, code: (_ for _ in ()).throw(OSError("killed")),
        )
        with pytest.raises(OSError):
            _publish(tmp_path, TUESDAY, 12.0)

    _publish(tmp_path, WEDNESDAY, 13.0)

    assert sorted(path.name for path in (tmp_path / "sep").glob("*.parquet")) == [
        bulk_file(tmp_path, "sep", WEDNESDAY).name
    ]


def test_two_bulks_on_disk_read_as_the_newest(tmp_path):
    _rows(10.0).write_parquet(_empty_path(bulk_file(tmp_path, "sep", MONDAY)))
    _rows(12.0).write_parquet(_empty_path(bulk_file(tmp_path, "sep", TUESDAY)))

    assert _closes(tmp_path) == [12.0]


def test_a_killed_pulls_scratch_is_removed_by_the_next_one(tmp_path):
    leftover = tmp_path / "sep" / ".pull-killed"
    leftover.mkdir(parents=True)

    _publish(tmp_path, MONDAY, 10.0)

    assert not leftover.exists()


def test_a_bulk_under_its_old_name_is_refused_not_read_as_only_its_later_pulls(tmp_path):
    (tmp_path / "sep").mkdir()
    _rows(10.0).write_parquet(tmp_path / "sep" / "sep.parquet")
    _window(tmp_path, TUESDAY, 11.0)

    with pytest.raises(ValueError, match="old name"):
        scan_raw_table(tmp_path, "sep")


def test_a_bulk_is_synced_on_both_sides_of_its_rename(tmp_path, monkeypatch):
    events = []
    real_replace, real_fsync = os.replace, os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("sync"), real_fsync(fd))[1])

    def replace(source, target):
        events.append(Path(target).name)
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", replace)
    _publish(tmp_path, MONDAY, 10.0)

    commit = events.index(bulk_file(tmp_path, "sep", MONDAY).name)
    assert events[commit - 1] == "sync" and events[commit + 1] == "sync"
