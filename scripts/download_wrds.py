# %%
import argparse
from dataclasses import replace

from quantlab.base.config import (
    ConstituentDatasetConfig,
    CrspDatasetConfig,
)
from quantlab.config import get_data_root
from quantlab.dataset.constituent import (
    CompustatNasdaq100ConstituentDataset,
    CrspMarketConstituentDataset,
    CrspSP500ConstituentDataset,
)
from quantlab.dataset.crsp import SECURITY_FILTER_PRESETS, CrspStockDataset
from quantlab.dataset.crsp.market import MARKET, CrspMarketRoster
from quantlab.dataset.crsp.membership import CrspMembership
from quantlab.dataset.crsp.reference import CrspReference
from quantlab.registry import DataSourceRegistry, convert, run
from quantlab.dataset._support.masking import UniverseMask

#: The one place this script's vendor is named, as a registry token.
SOURCE = DataSourceRegistry.get("wrds")

#: The capability this script drives -- the same one the index shell drives.
#: One WRDS account serves several products (03.10 D-12), so BOTH the
#: acquisition class and the config factory are read off this key.
CAPABILITY = ("us_equity", "1d", "crsp_daily")

#: The CRSP daily acquisition class, resolved -- never named.
ACQ = SOURCE.acquisition_cls_for(*CAPABILITY)

#: The equity panel's store name. `all` rather than `us_all`: the latter is
#: already taken on disk by a Tiingo-era store, and two stores under one name
#: is the collision the sibling shell's `custom` fallback exists to avoid.
STORE = "wrds_crsp_all_1d.zarr"

#: The whole-market in-listing mask's store.
MEMBERSHIP_STORE = "wrds_crsp_all_membership.zarr"


INDEX_PANELS: dict[str, tuple[str, type]] = {
    CrspMembership.SP500: ("sp500", CrspSP500ConstituentDataset),
    CrspMembership.NASDAQ100: (
        "nasdaq100",
        CompustatNasdaq100ConstituentDataset,
    ),
}


# %%
batch_size = ACQ.DEFAULT_BATCH_SIZE
kwargs = {"batch_size": batch_size, "clip_to_product_end": True}
window = {"start_date": "2010-01-01", "end_date": "2025-01-01"}
refresh = True
to_zarr = True

# %%

acq_config = SOURCE.config_factory_for(*CAPABILITY)(
    symbols=(),
    start_date=window["start_date"],
    end_date=window["end_date"],
    kwargs=kwargs,
)
reference_dir = ACQ.reference_dir_for(acq_config)

# %%

reference_dir


# %%

result = run(SOURCE, acq_config, refresh=refresh)
print(
    f"{len(result.succeeded):,} PERMNO(s) succeeded, "
    f"{len(result.failures):,} failed"
)
print(f"Raw data written under: {acq_config.raw_data_dir_path}")

# %%

# roster_source = CrspMarketRoster(CrspReference(reference_dir))
# roster = roster_source.permnos_in_range(
#     window["start_date"],
#     window["end_date"],
#     security_filter="equity_common",
# )


# %%

data_dir = get_data_root() / "data" / "us_equity" / "1d"
catalog_path = str(get_data_root() / "data" / "catalog")

ds_config = CrspDatasetConfig(
    zarr_file_path=str(data_dir / STORE),
    raw_data_dir_path=acq_config.raw_data_dir_path,
    catalog_path=catalog_path,
    reference_dir=str(reference_dir),
    start_date=window["start_date"],
    end_date=window["end_date"],
    # permnos=tuple(roster),
    security_filter="equity_common",
    # NOT `roster_universe=MARKET`. That field means "an index
    # provider decided membership, so a member is exempt from the
    # type filter inside its spell" (GAP-C), and it is resolved
    # through `CrspMembership.permno_intervals`, which serves
    # indexes only. Neither half fits a whole-market roster: this
    # roster IS the type filter's own output, so exempting it from
    # the type filter would be circular, and `crsp_all` is not an
    # index `CrspMembership` can answer for.
)
probe_dataset = CrspStockDataset(ds_config).from_raw_data()

# %%
nasdaq100_mask = CompustatNasdaq100ConstituentDataset(
    ConstituentDatasetConfig(
        zarr_file_path=str(
            data_dir / "wrds_crsp_nasdaq100_membership.zarr"
        ),
        cache_dir=str(reference_dir),
        start_date=window["start_date"],
        end_date=window["end_date"],
    )
)
# %%
nasdaq100_mask.from_raw_data()
# %%
nasdaq100 = UniverseMask(probe_dataset.get_xarray_dataset(), nasdaq100_mask.get_xarray_dataset()).apply()


# %%
nasdaq100.to_zarr('/home/zhrdai/projects/quantlab2/data/data/us_equity/1d/wrds_crsp_nasdaq100_1d.zarr')

# %%

ds_config = CrspDatasetConfig(
    zarr_file_path='/home/zhrdai/projects/quantlab2/data/data/us_equity/1d/wrds_crsp_nasdaq100_1d.zarr',
    raw_data_dir_path='',
    catalog_path='',
    reference_dir=str(reference_dir),
    start_date=window["start_date"],
    end_date=window["end_date"],
    # permnos=tuple(roster),
    # security_filter="equity_common",
    # NOT `roster_universe=MARKET`. That field means "an index
    # provider decided membership, so a member is exempt from the
    # type filter inside its spell" (GAP-C), and it is resolved
    # through `CrspMembership.permno_intervals`, which serves
    # indexes only. Neither half fits a whole-market roster: this
    # roster IS the type filter's own output, so exempting it from
    # the type filter would be circular, and `crsp_all` is not an
    # index `CrspMembership` can answer for.
)
nasdaq100 = CrspStockDataset(ds_config).read()
# %%
nasdaq100.get_lazyframe().collect()

# %%

# if args.with_index_membership:
#     for universe, (short_name, cls) in INDEX_PANELS.items():
#         panel = cls(
#             ConstituentDatasetConfig(
#                 zarr_file_path=str(
#                     data_dir
#                     / MEMBERSHIP_TEMPLATE.format(name=short_name)
#                 ),
#                 cache_dir=str(reference_dir),
#                 start_date=window["start_date"],
#                 end_date=window["end_date"],
#                 kwargs={"allow_unlinked": args.allow_unlinked_ndx},
#             )
#         )
#         panel.from_raw_data().save()
#         print(
#             f"{universe} membership panel: "
#             f"{panel.config.zarr_file_path}"
#         )
