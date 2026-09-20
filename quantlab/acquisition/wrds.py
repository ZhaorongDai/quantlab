"""The `wrds` source descriptor: ONE account, SEVERAL products (03.10 D-12).

A WRDS subscription is not a product, it is a door. Behind it sit NYSE TAQ
millisecond quotes, CRSP daily stock, Compustat and more -- reached through one
username, one `~/.pgpass` entry and (D-20) one connection, but through
different tables, different entitlements and different acquisition classes. The
registry's D-01 rule is ONE descriptor per VENDOR, so all of that is one
descriptor here, with the differences expressed as `Capability` rows.

**Why the descriptor lives in this neutral module rather than beside a provider
class, which is where every other vendor's descriptor sits (03.4 D-05).**
Tiingo and Alpaca each have exactly one acquisition class, so "beside the class"
and "beside the vendor" are the same place. WRDS has several. A registration
written inside `wrds_taq.py` would have to name `wrds_crsp`'s classes to
declare the CRSP capability, so `wrds_taq` would import `wrds_crsp` (or the
reverse) purely in order to be registered -- and since the registering module
also imports the registry, which imports the vendor modules, importing either
provider first would close the loop.

Putting the descriptor one level up breaks it by construction: the providers
import each other not at all, this module imports both, and
`quantlab/acquisition/registry.py`'s bottom vendor import names THIS module. No
import order can cycle, which `tests/test_wrds_vendor_seam.py` asserts in a
fresh interpreter for both orders.

The providers therefore stay free of the registry, and the shared
`WrdsSession` offers provider-neutral `schema_usable` / `fetch_rows` /
`copy_csv` so a second product needs no new plumbing -- only a `Capability`
row naming its own `acquisition_cls` and `config_factory`.
"""

from quantlab.acquisition import wrds_crsp, wrds_taq
from quantlab.acquisition.registry import (
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.nbbo import NbboPanelDataset

#: The registry descriptor for WRDS -- "who I am" for the whole account.
#:
#: `acquisition_cls` / `config_factory` at the DESCRIPTOR level are the vendor
#: DEFAULT (03.10 D-12's `add-alongside` decision): they stay required, because
#: `scripts/ingest_wrds_taq.py` reads `SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE`
#: directly, and every other ingest shell does the same for its own vendor. The
#: nbbo capability ALSO names them, so the pair a request resolves to comes off
#: the capability that answers it rather than off whichever product happened to
#: be the default.
#:
#: `display_name` now names BOTH products, because as of 03.10 the descriptor
#: serves both: an operator picking this source is picking an ACCOUNT, and a
#: name mentioning only one of its two capabilities would under-advertise it
#: exactly as a name mentioning CRSP before plan 02 would have over-advertised
#: it.
WRDS_SOURCE = register_source(
    SourceDescriptor(
        vendor="wrds",
        display_name="WRDS (NYSE TAQ millisecond NBBO; CRSP Stock v2 daily)",
        acquisition_cls=wrds_taq.WrdsTaqNbboAcquisition,
        config_factory=wrds_taq.WrdsTaqNbboAcquisition.build_config,
        capabilities=(
            Capability(
                market="us_equity",
                frequency="tick",
                data_type="nbbo",
                dataset_cls=NbboPanelDataset,
                earliest_available="2003-09-10",
                entitlement="WRDS NYSE TAQ millisecond subscription",
                acquisition_cls=wrds_taq.WrdsTaqNbboAcquisition,
                config_factory=wrds_taq.WrdsTaqNbboAcquisition.build_config,
            ),
            #: CRSP Stock v2 daily (03.10). A SECOND acquisition class and a
            #: SECOND config factory under the SAME vendor -- which is the
            #: whole point of plan 01's per-capability resolution: the
            #: descriptor-level pair stays the TAQ one (four ingest shells
            #: read `SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE` directly), and
            #: a `crsp_daily` request resolves off this row instead.
            #:
            #: `earliest_available` is the table's own first day, live-verified
            #: (`C4_range_crsp_a_stock.dsf_v2`); it is advisory, as the field's
            #: docstring says, and nothing gates on it.
            Capability(
                market="us_equity",
                frequency="1d",
                data_type="crsp_daily",
                dataset_cls=CrspStockDataset,
                earliest_available="1925-12-31",
                entitlement=(
                    "WRDS CRSP annual-update Stock v2 (crsp_a_stock)"
                ),
                acquisition_cls=wrds_crsp.WrdsCrspDailyAcquisition,
                config_factory=wrds_crsp.WrdsCrspDailyAcquisition.build_config,
            ),
        ),
        #: A LITERAL, restated rather than derived from `CREDENTIAL_ENV_VARS`
        #: (the D-04 pinning test would otherwise be `x == x`). One name for
        #: the whole account: every WRDS product authenticates identically.
        required_env=("WRDS_USERNAME",),
        universe_categories=("sp500_constituent", "nasdaq100_constituent"),
    )
)
