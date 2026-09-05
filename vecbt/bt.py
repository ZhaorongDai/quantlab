import pandas as pd
import vectorbt as vbt


def backtest_from_signals(
    close: pd.Series,
    long_entries: pd.Series,
    long_exits: pd.Series,
    short_entries: pd.Series,
    short_exits: pd.Series,
    index: pd.Index,
):
    close = close.reindex(index)

    p = vbt.Portfolio.from_signals(
        close,
        entries=long_entries,
        exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
    )

    return p
