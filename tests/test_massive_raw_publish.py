"""A Massive pull appears in the raw tier in one step, or not at all.

``raw_tier.stage_snapshot`` and ``stage_as_of`` hold a pull's pages out of sight
until ``commit`` publishes them with one directory rename, so a pull interrupted
at any point leaves the readers the old whole data. These tests interrupt a
publish directly, with no HTTP fake, and judge it by what the readers return.
"""

import os
from datetime import UTC, date, datetime

import pytest

from quantlab.acquisition.massive.raw_tier import stage_as_of, stage_snapshot
from quantlab.dataset.massive.raw import (
    read_rest_as_of,
    read_rest_snapshot,
    rest_as_of_days,
    rest_as_of_dir,
    rest_snapshots,
)

MONDAY = datetime(2024, 3, 11, 21, tzinfo=UTC)
TUESDAY = datetime(2024, 3, 12, 21, tzinfo=UTC)
PAGES = [[{"ticker": "AAA"}], [{"ticker": "BBB"}], [{"ticker": "CCC"}]]
ROWS = [{"ticker": "AAA"}, {"ticker": "BBB"}, {"ticker": "CCC"}]


def _stage_whole(stage):
    staged = stage()
    for page in PAGES:
        staged.add_page(page)
    return staged


def _snapshot(root, pulled_at=MONDAY):
    return _stage_whole(lambda: stage_snapshot(root, "float", pulled_at))


def _as_of(root, pulled_at=TUESDAY):
    return _stage_whole(lambda: stage_as_of(root, "tickers", date(2024, 3, 11), pulled_at))


def _refuse_the_rename(monkeypatch):
    def refuse(self, target):
        raise OSError("killed before the rename")

    monkeypatch.setattr("pathlib.Path.replace", refuse)


def test_a_snapshot_not_committed_is_not_there(tmp_path):
    staged = stage_snapshot(tmp_path, "float", MONDAY)
    staged.add_page(PAGES[0])

    assert rest_snapshots(tmp_path, "float") == []
    with pytest.raises(ValueError):
        read_rest_snapshot(tmp_path, "float")


def test_a_committed_snapshot_reads_whole_in_the_vendors_order(tmp_path):
    files = _snapshot(tmp_path).commit()

    assert rest_snapshots(tmp_path, "float") == [MONDAY]
    assert read_rest_snapshot(tmp_path, "float") == ROWS
    assert [path.name for path in files] == ["000001.jsonl", "000002.jsonl", "000003.jsonl"]


def test_a_snapshot_killed_at_the_rename_leaves_the_old_one_whole(tmp_path, monkeypatch):
    _snapshot(tmp_path).commit()
    newer = stage_snapshot(tmp_path, "float", TUESDAY)
    newer.add_page([{"ticker": "ZZZ"}])

    with monkeypatch.context() as patch:
        _refuse_the_rename(patch)
        with pytest.raises(OSError):
            newer.commit()

    assert rest_snapshots(tmp_path, "float") == [MONDAY]
    assert read_rest_snapshot(tmp_path, "float") == ROWS


def test_a_snapshot_killed_after_the_rename_reads_whole(tmp_path, monkeypatch):
    staged = _snapshot(tmp_path)

    def die_after_the_rename(path):
        raise OSError("killed after the rename")

    monkeypatch.setattr("quantlab.acquisition.massive.raw_tier.sync_directory", die_after_the_rename)
    with pytest.raises(OSError):
        staged.commit()

    assert read_rest_snapshot(tmp_path, "float") == ROWS


def test_the_next_pull_takes_over_from_an_interrupted_one(tmp_path):
    interrupted = stage_snapshot(tmp_path, "float", MONDAY)
    interrupted.add_page([{"ticker": "OLD"}])

    _snapshot(tmp_path, TUESDAY).commit()

    assert rest_snapshots(tmp_path, "float") == [TUESDAY]
    assert read_rest_snapshot(tmp_path, "float") == ROWS
    assert not (tmp_path / "rest" / "float" / "_pending").exists()


def test_a_pull_with_no_pages_publishes_nothing(tmp_path):
    assert stage_snapshot(tmp_path, "float", MONDAY).commit() == ()
    assert rest_snapshots(tmp_path, "float") == []


def test_snapshots_of_one_year_sit_side_by_side(tmp_path):
    _snapshot(tmp_path, MONDAY).commit()
    newer = stage_snapshot(tmp_path, "float", TUESDAY)
    newer.add_page([{"ticker": "ZZZ"}])
    newer.commit()

    assert rest_snapshots(tmp_path, "float") == [MONDAY, TUESDAY]
    assert read_rest_snapshot(tmp_path, "float", MONDAY) == ROWS
    assert read_rest_snapshot(tmp_path, "float") == [{"ticker": "ZZZ"}]


def test_a_session_not_committed_is_not_kept(tmp_path):
    staged = stage_as_of(tmp_path, "tickers", date(2024, 3, 11), TUESDAY)
    staged.add_page(PAGES[0])

    assert rest_as_of_days(tmp_path, "tickers") == []
    with pytest.raises(ValueError):
        read_rest_as_of(tmp_path, "tickers", date(2024, 3, 11))


def test_a_session_killed_at_the_rename_is_not_kept(tmp_path, monkeypatch):
    staged = _as_of(tmp_path)

    with monkeypatch.context() as patch:
        _refuse_the_rename(patch)
        with pytest.raises(OSError):
            staged.commit()

    assert rest_as_of_days(tmp_path, "tickers") == []


def test_a_session_pulled_again_reads_its_newest_pull_only(tmp_path):
    _as_of(tmp_path, MONDAY).commit()
    again = stage_as_of(tmp_path, "tickers", date(2024, 3, 11), TUESDAY)
    again.add_page([{"ticker": "ZZZ"}])
    again.commit()

    assert rest_as_of_days(tmp_path, "tickers") == [date(2024, 3, 11)]
    assert read_rest_as_of(tmp_path, "tickers", date(2024, 3, 11)) == [{"ticker": "ZZZ"}]


def test_a_session_killed_after_the_rename_reads_whole(tmp_path, monkeypatch):
    staged = _as_of(tmp_path)

    def die_after_the_rename(self, *args, **kwargs):
        raise OSError("killed after the rename")

    monkeypatch.setattr("quantlab.acquisition.massive.raw_tier.shutil.rmtree", die_after_the_rename)
    with pytest.raises(OSError):
        staged.commit()
    monkeypatch.undo()

    assert read_rest_as_of(tmp_path, "tickers", date(2024, 3, 11)) == ROWS


def test_the_next_session_takes_over_from_an_interrupted_one(tmp_path):
    interrupted = stage_as_of(tmp_path, "tickers", date(2024, 3, 11), MONDAY)
    interrupted.add_page([{"ticker": "OLD"}])

    _as_of(tmp_path, TUESDAY).commit()

    assert read_rest_as_of(tmp_path, "tickers", date(2024, 3, 11)) == ROWS
    assert not (tmp_path / "rest" / "tickers" / "_pending").exists()


def test_pages_kept_outside_a_pull_directory_are_refused_not_read_as_empty(tmp_path):
    session = rest_as_of_dir(tmp_path, "tickers", date(2024, 3, 11))
    session.mkdir(parents=True)
    (session / "20240312T210000000000_000001.jsonl").write_text('{"ticker": "AAA"}\n')

    with pytest.raises(ValueError, match="pull"):
        rest_as_of_days(tmp_path, "tickers")
    with pytest.raises(ValueError, match="pull"):
        read_rest_as_of(tmp_path, "tickers", date(2024, 3, 11))


def test_a_flat_snapshot_page_is_refused_with_the_reason(tmp_path):
    year = tmp_path / "rest" / "float" / "2024"
    year.mkdir(parents=True)
    (year / "20240312T210000000000_000001.jsonl").write_text('{"ticker": "AAA"}\n')

    with pytest.raises(ValueError, match="pull"):
        rest_snapshots(tmp_path, "float")


def test_every_page_is_on_disk_before_the_pull_is_published(tmp_path, monkeypatch):
    events = []
    real_fsync, real_rename = os.fsync, os.replace
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append("sync"), real_fsync(fd))[1])

    def rename(source, target):
        events.append("dir rename" if os.path.isdir(source) else "page rename")
        return real_rename(source, target)

    monkeypatch.setattr(os, "replace", rename)
    _snapshot(tmp_path).commit()

    assert events.index("dir rename") > max(i for i, event in enumerate(events) if event == "page rename")
    assert events[-1] == "sync"


def test_a_new_session_is_durable_up_to_the_endpoint(tmp_path, monkeypatch):
    # A watermark outlives the session unless each new directory's entry is synced.
    synced = []
    real_open = os.open

    def recording_open(path, flags, *args, **kwargs):
        if os.path.isdir(path):
            synced.append(os.path.relpath(path, tmp_path))
        return real_open(path, flags, *args, **kwargs)

    staged = stage_as_of(tmp_path, "tickers", date(2024, 3, 11), TUESDAY)
    staged.add_page(PAGES[0])
    monkeypatch.setattr(os, "open", recording_open)
    staged.commit()

    assert {"rest/tickers", "rest/tickers/2024", "rest/tickers/2024/2024-03-11"} <= set(synced)
