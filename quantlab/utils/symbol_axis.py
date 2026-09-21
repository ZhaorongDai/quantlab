"""The symbol axis's two contracts, each stated ONCE (03.11-02).

RED skeleton: the signatures exist so `tests/test_symbol_axis_contract.py`
fails on the BEHAVIOUR it asserts rather than on a collection error. The
bodies land in the GREEN commit.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def sort_symbol_axis(values: Iterable) -> list:
    raise NotImplementedError("sort_symbol_axis: RED skeleton")


def normalize_to_axis_dtype(labels: Iterable, stored_index: pd.Index) -> list:
    raise NotImplementedError("normalize_to_axis_dtype: RED skeleton")
