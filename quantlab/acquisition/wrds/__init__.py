"""WRDS vendor package: the registry descriptor for one WRDS account.

A WRDS subscription is one login that reaches several products. This package
holds two of them, each as its own submodule and acquisition class:
``quantlab.acquisition.wrds.taq`` (NYSE TAQ millisecond NBBO quotes,
``WrdsTaqNbboAcquisition``) and ``quantlab.acquisition.wrds.crsp`` (CRSP
Stock v2 daily bars, ``WrdsCrspDailyAcquisition``). Both authenticate the same
way and share one PostgreSQL connection, ``taq.WrdsSession``, which reads the
username from ``WRDS_USERNAME`` and leaves the password to libpq's
``~/.pgpass``; every new connection can trigger a Duo prompt, so the session
is opened once per process.

``WRDS_SOURCE`` below registers the vendor with ``quantlab.registry`` as a
single descriptor whose ``Capability`` rows name the product-specific class
and config factory. The descriptor lives here, rather than beside one of the
provider classes, because either provider would otherwise have to import its
sibling to declare the other capability. The providers import nothing from the
registry or from this module.

Example:
    >>> from quantlab.acquisition.wrds import WRDS_SOURCE
    >>> sorted((c.market, c.frequency, c.data_type)
    ...        for c in WRDS_SOURCE.capabilities)
    [('us_equity', '1d', 'crsp_daily'), ('us_equity', 'tick', 'nbbo')]
    >>> WRDS_SOURCE.required_env
    ('WRDS_USERNAME',)
"""

from quantlab.acquisition.wrds import crsp, taq
from quantlab.registry import (
    Capability,
    SourceDescriptor,
    register_source,
)
from quantlab.dataset.crsp import CrspStockDataset
from quantlab.dataset.nbbo import NbboPanelDataset

#: The registered descriptor for the WRDS account.
#:
#: The descriptor-level ``acquisition_cls`` / ``config_factory`` pair is the
#: vendor default (the TAQ product); ingest scripts read
#: ``SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE`` from it. Each capability row
#: also names its own pair, so a request for ``crsp_daily`` resolves to the
#: CRSP class rather than to the default. ``display_name`` mentions both
#: products because choosing this source means choosing the account.
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
            #: CRSP Stock v2 daily: a second acquisition class and config
            #: factory under the same vendor. ``earliest_available`` is the
            #: first day of ``crsp_a_stock.dsf_v2``; it is advisory and
            #: nothing gates on it.
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
        #: Written as a literal rather than derived from the classes'
        #: ``CREDENTIAL_ENV_VARS`` so the two can be checked against each
        #: other. One variable serves the whole account.
        required_env=("WRDS_USERNAME",),
        universe_categories=("sp500_constituent", "nasdaq100_constituent"),
    )
)
