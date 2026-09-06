"""Dataset-hierarchy boundary-contract tests (DATA-06 + D-03).

Created by 03.1-01 Task 2, which splits `base/data.py:Dataset` into a shared
`BaseDataset(ABC)` and a `MarketDataset(BaseDataset)` carrying every
nautilus/KunQuant-specific member -- the dataset-layer twin of the
`Factor`/`FactorKunQuant` split that `tests/test_factor_hierarchy.py` locks.

Like that file, several tests here are **source-introspection rather than
behavioural**, because nothing at runtime fails today if the invariants break.
The breakage only surfaces later, in the index-constituent backend that does
not exist yet: a `catalog_path` leaking onto the shared config would only bite
the first dataset constructed without one, and a reversed `__init__` would only
bite through `_reset_symbols()`. These tests move both failures to test time.

The two genuinely behavioural tests are the last two. `PanelDataset` below is
the non-market dataset DATA-06 is about: no OHLCV, no bar representation, no
catalog, no KunQuant graph -- one boolean membership variable over
`(timestamp, symbol)`, which it round-trips through the whole storage
lifecycle using nothing but `BaseDataset` plus one implemented abstract method.
"""

import dataclasses
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from base.config import BaseDatasetConfig, ConstituentDatasetConfig, DatasetConfig
from base.data import BaseDataset, MarketDataset

# The nautilus/KunQuant-specific members. A membership panel has no bar and no
# compiled-graph representation, so none of these may live on the shared base
# -- especially not as an `@abstractmethod`, which would force every future
# dataset kind to carry a meaningless `raise NotImplementedError` stub (D-03).
MARKET_ONLY_MEMBERS = (
    "to_kunquant",
    "_to_kunquant",
    "to_nautilus",
    "_to_nautilus",
    "_write_catalog",
)

# The market-only `DatasetConfig` fields that must NOT be on the shared
# `BaseDatasetConfig` (CONFLICT 4). `catalog_path` is the load-bearing one:
# it is the nautilus write destination, and a non-market dataset must not be
# able to be handed one at all.
MARKET_ONLY_CONFIG_FIELDS = {
    "raw_data_dir_path",
    "catalog_path",
    "market",
    "frequency",
}

# The two fields `ConstituentDatasetConfig` adds: `cache_dir`, the cached
# source snapshot directory an index-membership fetcher writes to, and
# `as_of`, which pins the panel's right edge so a rebuild is a function of the
# config rather than of the wall clock (WR-07).
#
# Both are deliberately NOT on `BaseDatasetConfig`: neither means anything to
# a market dataset, whose right edge comes from its bars.
CONSTITUENT_ONLY_CONFIG_FIELDS = {"cache_dir", "as_of"}

_PANEL_SYMBOLS = ("PANEL_A", "PANEL_B")
_PANEL_PERIODS = 4

# Hand-checkable membership pattern, shaped (timestamp, symbol): PANEL_A is a
# member throughout, PANEL_B joins on the third day. Asserted byte-identical
# on the way back out of Zarr.
_PANEL_IS_MEMBER = np.array(
    [
        [True, False],
        [True, False],
        [True, True],
        [True, True],
    ]
)


class PanelDataset(BaseDataset):
    """A test-only non-market dataset: a boolean membership panel.

    Implements exactly ONE abstract method, `_raw_data_to_xr()`. It overrides
    `_clean()` because the inherited default runs the OHLCV-shaped
    `clean_market_data()` pipeline, and `_reset_symbols()` because a panel
    derives its symbol axis from its own source rather than from the store
    (CONFLICT 1). It defines no bar conversion, no catalog write and no
    KunQuant input path -- that is the whole point.
    """

    def __init__(self, config: BaseDatasetConfig):
        self.raw_data_calls = 0
        super().__init__(config)

    def _raw_data_to_xr(self) -> xr.Dataset:
        self.raw_data_calls += 1
        return xr.Dataset(
            {"is_member": (["timestamp", "symbol"], _PANEL_IS_MEMBER)},
            coords={
                "timestamp": pd.date_range(
                    "2024-01-01", periods=_PANEL_PERIODS, freq="D"
                ),
                "symbol": list(_PANEL_SYMBOLS),
            },
        )

    def _clean(self, data: xr.Dataset) -> xr.Dataset:
        # A membership panel has no OHLCV columns to validate or flag.
        return data

    def _reset_symbols(self) -> None:
        # The seam (CONFLICT 1): the symbol axis comes from the membership
        # source, not from the store, so construction must not touch disk.
        return None


def test_base_dataset_is_abstract_and_market_dataset_subclasses_it() -> None:
    """D-03/DATA-06: a shared abstract `BaseDataset` exists and `MarketDataset`
    is one implementation of it, not the root of the hierarchy.

    `_raw_data_to_xr` is the ONE abstract method on the shared base -- the
    single obligation every dataset kind genuinely has. The two KunQuant/
    nautilus abstract methods are added by `MarketDataset` and by it alone,
    which is the mechanism that keeps them off a membership panel.
    """
    assert inspect.isabstract(BaseDataset)
    assert issubclass(MarketDataset, BaseDataset)
    assert BaseDataset.__abstractmethods__ == frozenset({"_raw_data_to_xr"})
    assert set(MarketDataset.__abstractmethods__) - set(
        BaseDataset.__abstractmethods__
    ) == {"_to_kunquant", "_to_nautilus"}


def test_market_only_members_stay_on_market_dataset() -> None:
    """D-03: every nautilus/KunQuant member stays on `MarketDataset` and is
    absent from `BaseDataset`.

    If any of these leaked onto the shared base, a dataset with no bar and no
    catalog representation would inherit an obligation it cannot honestly
    meet -- exactly the two `raise NotImplementedError` stubs DATA-06 exists
    to remove.
    """
    own = set(MarketDataset.__dict__)
    shared = set(BaseDataset.__dict__)

    missing = [name for name in MARKET_ONLY_MEMBERS if name not in own]
    leaked = [name for name in MARKET_ONLY_MEMBERS if name in shared]

    assert not missing, f"no longer defined on MarketDataset: {missing}"
    assert not leaked, f"leaked onto the shared BaseDataset base: {leaked}"


def test_market_only_config_fields_stay_off_the_shared_base() -> None:
    """CONFLICT 4: the nautilus-only config fields are exactly the delta on
    `DatasetConfig`, and `ConstituentDatasetConfig` adds only `cache_dir`.

    Stated positively rather than as a bare absence check, so moving a field
    onto `BaseDatasetConfig` fails here rather than silently making a
    catalog write path available to a dataset that has no catalog.
    """
    assert issubclass(DatasetConfig, BaseDatasetConfig)
    assert issubclass(ConstituentDatasetConfig, BaseDatasetConfig)

    base_fields = {f.name for f in dataclasses.fields(BaseDatasetConfig)}
    market_fields = {f.name for f in dataclasses.fields(DatasetConfig)}
    constituent_fields = {
        f.name for f in dataclasses.fields(ConstituentDatasetConfig)
    }

    assert market_fields - base_fields == MARKET_ONLY_CONFIG_FIELDS
    assert constituent_fields - base_fields == CONSTITUENT_ONLY_CONFIG_FIELDS


def test_base_dataset_init_assigns_the_storage_backend_before_the_config() -> (
    None
):
    """CONFLICT 2: `BaseDataset.__init__` must assign the storage backend
    BEFORE `self.config`.

    This is deliberately the INVERSE of
    `tests/test_factor_hierarchy.py:test_factor_init_assigns_config_before_
    the_storage_backend`. The two hierarchies hold opposite invariants for
    different, equally correct reasons: no method reachable from the `Factor`
    config setter may touch the backend, whereas the `BaseDataset` config
    setter DOES reach it -- through `_reset_symbols()` -> `read()`. Reversing
    the two lines here raises `AttributeError` on every dataset construction
    that pins `config.symbols`. Do not "harmonize" the two files.
    """
    init_source = inspect.getsource(BaseDataset.__init__)
    backend_index = init_source.index("self.data_backend =")
    config_index = init_source.index("self.config = config")

    assert backend_index < config_index, (
        "BaseDataset.__init__ assigns self.config before the storage "
        "backend; _reset_symbols() would then run against a half-built "
        "instance and raise AttributeError (CONFLICT 2)"
    )


def test_reset_symbols_seam_suppresses_construction_time_io(
    tmp_path: Path,
) -> None:
    """CONFLICT 1: `_reset_symbols()` is an overridable seam, and overriding
    it to a no-op demonstrably stops ALL construction-time I/O.

    The inherited default reads the store and, on `FileNotFoundError`, falls
    back to `from_raw_data()`. For a dataset whose raw source is a remote
    fetch that means merely constructing the object would hit the network,
    and it would also overwrite the caller's symbol request with whatever the
    store happens to hold. Both are asserted suppressed here: the store path
    below does not exist, yet construction succeeds, the caller's tuple
    survives, and `_raw_data_to_xr()` is never reached.
    """
    config = BaseDatasetConfig(
        zarr_file_path=str(tmp_path / "does" / "not" / "exist.zarr"),
        symbols=("PANEL_A",),
    )

    dataset = PanelDataset(config)

    assert dataset.config.symbols == ("PANEL_A",), (
        "the caller-supplied symbols tuple was overwritten during "
        "construction; the _reset_symbols() override did not take effect"
    )
    assert dataset.raw_data_calls == 0, (
        "constructing the dataset reached _raw_data_to_xr(); for a dataset "
        "whose raw source is a remote fetch that is an unannounced outbound "
        "request from __init__"
    )

    assert "_reset_symbols" in BaseDataset.__dict__, (
        "_reset_symbols() left the shared base; it is called from the shared "
        "config setter, so pushing it down would force the setter to split too"
    )
    assert "_reset_symbols" not in MarketDataset.__dict__, (
        "MarketDataset overrides _reset_symbols(); market datasets must keep "
        "inheriting the eager default unchanged"
    )


def test_non_market_dataset_round_trips_through_base_dataset(
    tmp_path: Path,
) -> None:
    """DATA-06 in its most direct form: a dataset with no bar, no catalog and
    no KunQuant representation completes the entire storage lifecycle by
    implementing one abstract method.

    `from_raw_data()` -> `save()` -> `read(overwrite=True)` ->
    `get_xarray_dataset()` runs entirely on `BaseDataset`. The boolean
    variable must survive the Zarr round trip as dtype `bool` (not as a
    float or an int8), and the membership pattern must come back
    byte-identical -- a silently-widened dtype or a transposed panel would
    both read downstream as a plausible-but-wrong universe mask.
    """
    config = BaseDatasetConfig(
        zarr_file_path=str(tmp_path / "panel" / "membership.zarr")
    )
    dataset = PanelDataset(config)

    dataset.from_raw_data().save()
    result = PanelDataset(config).read(overwrite=True).get_xarray_dataset()

    assert isinstance(result, xr.Dataset)
    assert result["is_member"].dims == ("timestamp", "symbol")
    assert result["is_member"].dtype == np.dtype("bool")
    assert result.symbol.values.tolist() == list(_PANEL_SYMBOLS)
    np.testing.assert_array_equal(
        result["is_member"].values, _PANEL_IS_MEMBER
    )


def test_base_data_module_exposes_only_the_two_split_classes() -> None:
    """The pre-split `Dataset` name was RETIRED, not kept as an alias.

    03.1-01 introduced `Dataset = MarketDataset` as a transitional alias so
    the split could land with a green suite, and removed it in the same plan
    once every call site was re-pointed. Two live names for one class is
    exactly the ambiguity a later reader "fixes" in the wrong direction, and
    an unused alias is dead code (QUAL-02). This test exists so nobody
    reintroduces it out of caution: if you are here because an import of
    `base.data.Dataset` failed, the answer is `MarketDataset` for a dataset
    with bars and a catalog, `BaseDataset` for anything else.
    """
    import base.data as base_data_module

    assert hasattr(base_data_module, "BaseDataset")
    assert hasattr(base_data_module, "MarketDataset")
    assert not hasattr(base_data_module, "Dataset"), (
        "base.data.Dataset is back; the pre-split name was retired rather "
        "than aliased (03.1-01 Task 3)"
    )
