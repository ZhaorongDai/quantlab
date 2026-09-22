"""The `wrds` source descriptor: ONE account, SEVERAL products (03.10 D-12).

A WRDS subscription is not a product, it is a door. Behind it sit NYSE TAQ
millisecond quotes, CRSP daily stock, Compustat and more -- reached through one
username, one `~/.pgpass` entry and (D-20) one connection, but through
different tables, different entitlements and different acquisition classes. The
registry's D-01 rule is ONE descriptor per VENDOR, so all of that is one
descriptor here, with the differences expressed as `Capability` rows.

**Why the descriptor lives in the package entry point rather than beside a
provider class, which is where every other vendor's descriptor sits (03.4
D-05).** Tiingo and Alpaca each have exactly one acquisition class, so "beside
the class" and "beside the vendor" are the same place. WRDS has several. A
registration written inside one provider submodule would still have to name a
sibling submodule's classes to declare the other capability -- being siblings
in one package changes nothing about that. So the descriptor sits one level up,
in this `__init__`, which imports the provider submodules; they import neither
the registry nor this descriptor.

**No import order cycles, and here is why.** `quantlab/acquisition/registry.py`
binds this package as a MODULE OBJECT at its bottom
(`from quantlab.acquisition import wrds as _wrds`), never an attribute off it,
and `from package import submodule` is safe while the package is only partially
initialised. Measured in three fresh interpreters with the editable-install
finder removed -- `quantlab.acquisition.wrds.taq` first,
`quantlab.acquisition.wrds` first, `quantlab.acquisition.registry` first -- all
three print `['alpaca', 'tiingo', 'wrds']` with no traceback.
`tests/test_wrds_vendor_seam.py` now pins all three orders permanently.

**What packaging these modules traded away, plainly.** Importing ANY WRDS
provider now runs this `__init__`, so it loads the registry,
`quantlab.dataset.crsp` and `quantlab.dataset.nbbo`. Before the providers were
packaged it did not: importing a provider pulled in 1457 modules and left both
`quantlab.acquisition.registry` and `quantlab.dataset.crsp` absent from
`sys.modules`; it now pulls in 1600 (0.87s -> 0.85s, so the module count moved
and the wall time did not). "The providers stay free of the registry" is
therefore a SOURCE-TEXT rule from here on, enforced by the `ast` scan in
`tests/test_wrds_vendor_seam.py`, not a runtime fact you can observe in
`sys.modules`. It is still worth enforcing: it is what keeps a registration
from drifting back into a provider.

And the older claim that the providers "import each other not at all" was
already false before the packaging: the CRSP provider reaches the shared
session through the TAQ module's attribute (`_wrds.WrdsSession`, read at call
time). The real rule, and the narrow one the tests enforce, is that no provider
imports the registry or this descriptor.

The shared `WrdsSession` offers provider-neutral `schema_usable` / `fetch_rows`
/ `copy_csv` so a second product needs no new plumbing -- only a `Capability`
row naming its own `acquisition_cls` and `config_factory`.
"""

from quantlab.acquisition.wrds import crsp, taq
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
        acquisition_cls=taq.WrdsTaqNbboAcquisition,
        config_factory=taq.WrdsTaqNbboAcquisition.build_config,
        capabilities=(
            Capability(
                market="us_equity",
                frequency="tick",
                data_type="nbbo",
                dataset_cls=NbboPanelDataset,
                earliest_available="2003-09-10",
                entitlement="WRDS NYSE TAQ millisecond subscription",
                acquisition_cls=taq.WrdsTaqNbboAcquisition,
                config_factory=taq.WrdsTaqNbboAcquisition.build_config,
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
                acquisition_cls=crsp.WrdsCrspDailyAcquisition,
                config_factory=crsp.WrdsCrspDailyAcquisition.build_config,
            ),
        ),
        #: A LITERAL, restated rather than derived from `CREDENTIAL_ENV_VARS`
        #: (the D-04 pinning test would otherwise be `x == x`). One name for
        #: the whole account: every WRDS product authenticates identically.
        required_env=("WRDS_USERNAME",),
        universe_categories=("sp500_constituent", "nasdaq100_constituent"),
    )
)
