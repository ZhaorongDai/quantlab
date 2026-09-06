"""Proof that `ingest_tiingo.py --universe`/`--as-of-date` wiring resolves a
symbol list purely through `UniverseCatalog.get_symbols_as_of()`, never
touching `TiingoAcquisition`/`StockDataset` inside `_build_configs()` itself.
"""

import argparse
from unittest.mock import patch

import pytest

import ingest_tiingo


class _FakeCatalog:
    def get_symbols_as_of(self, category: str, as_of_date: str) -> list[str]:
        return ["AAPL", "MSFT"]


def _make_args(**overrides) -> argparse.Namespace:
    base = dict(
        universe=None,
        as_of_date=None,
        symbols=None,
        start_date=None,
        end_date=None,
        refresh=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_configs_resolves_symbols_from_universe(monkeypatch):
    monkeypatch.setattr(
        ingest_tiingo.UniverseCatalog, "load", classmethod(lambda cls, config: _FakeCatalog())
    )

    args = _make_args(universe="sp500", as_of_date="2020-01-01")
    acq_config, ds_config = ingest_tiingo._build_configs(args)

    assert tuple(acq_config.symbols) == ("AAPL", "MSFT")
    assert list(ds_config.symbols) == ["AAPL", "MSFT"]


def test_build_configs_never_instantiates_tiingo_acquisition_or_stock_dataset(
    monkeypatch,
):
    monkeypatch.setattr(
        ingest_tiingo.UniverseCatalog, "load", classmethod(lambda cls, config: _FakeCatalog())
    )

    args = _make_args(universe="sp500", as_of_date="2020-01-01")
    with patch("ingest_tiingo.TiingoAcquisition") as mock_acquisition, patch(
        "ingest_tiingo.StockDataset"
    ) as mock_dataset:
        ingest_tiingo._build_configs(args)

        mock_acquisition.assert_not_called()
        mock_dataset.assert_not_called()


def test_build_configs_keeps_explicit_symbols_path_unchanged():
    args = _make_args(symbols="AAPL,MSFT")
    acq_config, ds_config = ingest_tiingo._build_configs(args)

    assert acq_config.symbols == ("AAPL", "MSFT")
    assert ds_config.symbols == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# D-14: the shared argument groups live once in `utils/cli.py` (03.2-07 Task 1)
#
# The extraction is a RELOCATION. What these tests defend is that it stayed
# one: the same flags, from one definition, reachable from every script that
# calls the group -- and that the two roster modes did NOT collapse into one
# with a default, which is the single highest-cost mistake available here
# because the resulting survivorship bias is invisible to every test that does
# not specifically look for delisted tickers.
# ---------------------------------------------------------------------------


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }


def test_the_shared_groups_register_the_same_flags_from_either_parser():
    """One definition, N callers: adding a shared group to two independently
    constructed parsers must produce identical option sets.

    Asserted per GROUP rather than by diffing the two scripts' whole parsers,
    because the scripts legitimately differ elsewhere -- `--refresh` on one,
    `--category`/`--stamp-legacy-watermarks` on the other. A whole-parser diff
    would fail for the right reason today and the wrong reason tomorrow.
    """
    import utils.cli as cli

    for add_group, kwargs in (
        (cli.add_universe_args, {}),
        (cli.add_window_args, {}),
        (cli.add_chunk_args, {}),
        (cli.add_concurrency_args, {"default_max_workers": 7}),
    ):
        first, second = argparse.ArgumentParser(), argparse.ArgumentParser()
        add_group(first, **kwargs)
        add_group(second, **kwargs)
        assert _option_strings(first) == _option_strings(second) != set(), add_group


def test_both_existing_scripts_take_their_window_flags_from_the_shared_group():
    """`--start-date` / `--end-date` reach both scripts, and each keeps its own
    default -- `ingest_tiingo.py` has none, `ingest_us_equity.py` defaults to
    the D-05 backfill start. A shared group that flattened the defaults would
    silently widen or narrow one script's window."""
    import ingest_us_equity

    tiingo = ingest_tiingo._build_arg_parser()
    us_equity = ingest_us_equity._build_arg_parser()

    for parser in (tiingo, us_equity):
        assert {"--start-date", "--end-date"} <= _option_strings(parser)

    assert tiingo.parse_args([]).start_date is None
    assert (
        us_equity.parse_args([]).start_date == ingest_us_equity.DEFAULT_START_DATE
    )


def test_the_universe_flags_reach_ingest_tiingo_and_the_choices_are_unchanged():
    parser = ingest_tiingo._build_arg_parser()

    assert {"--symbols", "--universe", "--as-of-date"} <= _option_strings(parser)

    choices = next(
        action.choices
        for action in parser._actions
        if "--universe" in action.option_strings
    )
    assert set(choices) == {"sp500", "nasdaq100", "nasdaq_all", "us_all"}


def test_resolve_symbols_refuses_to_pick_a_roster_mode_for_the_caller():
    """`mode` is keyword-only with NO default. Point-in-time membership and
    interval overlap are a deliberate semantic difference; a default would make
    the wrong one silent."""
    import inspect

    import utils.cli as cli

    mode = inspect.signature(cli.resolve_symbols).parameters["mode"]
    assert mode.default is inspect.Parameter.empty
    assert mode.kind is inspect.Parameter.KEYWORD_ONLY

    args = _make_args(universe="sp500", as_of_date="2020-01-01")
    with pytest.raises(TypeError):
        cli.resolve_symbols(args, _FakeCatalog())  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="neither is a default"):
        cli.resolve_symbols(args, _FakeCatalog(), mode="whatever")  # type: ignore[arg-type]


def test_both_roster_modes_are_reachable_and_call_different_catalog_methods():
    """The bias-safe mode is not merely present -- it dispatches somewhere
    else. A resolver that accepted `"in_range"` and still called
    `get_symbols_as_of` would pass every signature assertion above."""
    import utils.cli as cli

    calls: list[str] = []

    class _RecordingCatalog:
        def get_symbols_as_of(self, category, as_of_date):
            calls.append(f"as_of:{category}:{as_of_date}")
            return ["AAPL"]

        def get_symbols_in_range(self, category, start_date, end_date):
            calls.append(f"in_range:{category}:{start_date}:{end_date}")
            return ["AAPL", "DELISTED"]

    as_of_args = _make_args(universe="sp500", as_of_date="2020-01-01")
    assert cli.resolve_symbols(as_of_args, _RecordingCatalog(), mode="as_of") == (
        "AAPL",
    )

    range_args = argparse.Namespace(
        universe=None,
        symbols=None,
        as_of_date=None,
        category="us_all",
        start_date="2016-01-01",
        end_date="2026-01-01",
        limit=None,
    )
    assert cli.resolve_symbols(
        range_args, _RecordingCatalog(), mode="in_range"
    ) == ("AAPL", "DELISTED")

    assert calls == [
        "as_of:sp500_constituent:2020-01-01",
        "in_range:us_all:2016-01-01:2026-01-01",
    ]


def test_the_category_map_exists_in_exactly_one_module():
    """The `--universe` choices are derived from one map. A per-script copy is
    how the map and the choices drifted apart the first time."""
    import ast

    import utils.cli as cli

    assert set(cli.UNIVERSE_CATEGORY_MAP) == {
        "sp500",
        "nasdaq100",
        "nasdaq_all",
        "us_all",
    }

    for path in ("ingest_tiingo.py", "ingest_us_equity.py", "ingest_alpaca.py"):
        try:
            tree = ast.parse(open(path).read())
        except FileNotFoundError:
            continue
        assigned = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        assert "_UNIVERSE_CATEGORY_MAP" not in assigned, path
