"""Registry of every data source quantlab can download from.

A data source is one vendor, such as Tiingo, Alpaca or WRDS. Each vendor is
described once by a ``SourceDescriptor``: the names of the environment
variables that hold its credentials, the ``Acquisition`` class that downloads
from it, and the ``Capability`` rows it actually serves. A capability is one
``(market, frequency, data_type)`` combination, for example US equities at
daily frequency.

Downloading and converting are two separate steps. An acquisition writes a
raw tier: the vendor's data as parquet files on disk, plus small watermark
files that record how far each symbol has been downloaded so that a later run
can resume. A dataset class then converts the raw tier into a Zarr store
holding a panel, an ``xarray.Dataset`` indexed by ``timestamp`` and
``symbol``. The module-level functions ``run()`` and ``convert()`` perform
these two steps without the caller naming any vendor class.

Descriptors register themselves by calling ``register_source`` beside the
acquisition class they describe. The vendor modules are imported at the
bottom of this file, so ``import quantlab.registry`` alone makes every source
available.

Descriptors hold environment variable names only. No function here returns,
logs or embeds a credential value, not even a masked one, and no descriptor
holds a base URL or host.

Examples
--------
>>> from quantlab.registry import DataSourceRegistry
>>> [d.vendor for d in DataSourceRegistry.all()]
['alpaca', 'tiingo', 'wrds']
"""

import dataclasses
import os
from dataclasses import dataclass
from typing import Callable

from quantlab.base.acquisition import Acquisition, AcquisitionResult
from quantlab.base.config import AcquisitionConfig, DatasetConfig
from quantlab.base.data import ConversionResult, MarketDataset
from quantlab.base.progress import CancelToken, ProgressReporter
from quantlab.enums.data import Frequency, Market, UniverseCategory, Vendor


@dataclass(frozen=True)
class Capability:
    """One ``(market, frequency, data_type)`` combination a vendor serves.

    A descriptor lists its capabilities one by one rather than as every
    pairing of its markets and frequencies, because vendors serve irregular
    combinations. For example, Alpaca serves tick data for US equities only,
    and splits it into quotes and trades. Anything that differs between
    combinations, such as the earliest available date or the dataset class
    that converts the data, is stored here rather than on the descriptor.
    The class is frozen; extend it by adding optional fields.

    Parameters
    ----------
    market : Market
        The market, for example ``"us_equity"``.
    frequency : Frequency
        The bar frequency, for example ``"1d"`` or ``"tick"``.
    data_type : str or None, default None
        ``"bars"``, ``"quotes"``, ``"trades"`` and so on, or ``None`` when
        the vendor makes no such distinction. ``None`` means "no data type",
        not "any data type".
    earliest_available : str or None, default None
        ISO date of the earliest data the vendor serves, when known. It is
        informational; nothing checks against it.
    entitlement : str or None, default None
        The subscription this capability needs, when the vendor sells
        several.
    dataset_cls : type[MarketDataset] or None, default None
        The dataset class that converts this capability's raw tier into
        Zarr, or ``None`` when no dataset can hold the data. Tick data, for
        example, arrives at irregular times and does not fit a regular
        ``(timestamp, symbol)`` grid. ``convert()`` refuses a capability
        whose ``dataset_cls`` is ``None``, so adding a conversion means
        filling this field, not adding a branch to ``convert()``.
    acquisition_cls : type[Acquisition] or None, default None
        The class that downloads this capability, or ``None`` to use the
        descriptor's own ``acquisition_cls``. One vendor account can serve
        several products through different classes; WRDS, for example,
        serves both TAQ quote data and CRSP daily data.
    config_factory : callable or None, default None
        The function that builds an ``AcquisitionConfig`` for this
        capability, or ``None`` to use the descriptor's own
        ``config_factory``.

    Notes
    -----
    The class fields are direct class references, never dotted import
    paths. They are annotated with the abstract base classes so that the top
    of this module imports no vendor module.

    Examples
    --------
    >>> from quantlab.registry import Capability
    >>> cap = Capability(market="us_equity", frequency="1d", data_type="bars")
    >>> cap.data_type
    'bars'
    >>> cap.dataset_cls is None
    True
    """

    market: Market
    frequency: Frequency
    data_type: str | None = None
    earliest_available: str | None = None
    entitlement: str | None = None
    dataset_cls: type[MarketDataset] | None = None
    acquisition_cls: type[Acquisition] | None = None
    config_factory: Callable[..., AcquisitionConfig] | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything the registry knows about one vendor.

    Facts about the vendor as a whole (credential names, display name, the
    default acquisition class and config factory) are stated once here.
    Everything that differs by market, frequency or data type is stored in
    ``capabilities``. Exactly one descriptor is registered per vendor.

    Parameters
    ----------
    vendor : Vendor
        The vendor's short name, for example ``"tiingo"``.
    display_name : str
        A human-readable name for listings.
    acquisition_cls : type[Acquisition]
        The default class that downloads from this vendor. A capability may
        name a different one.
    config_factory : callable
        The default function that builds an ``AcquisitionConfig`` for this
        vendor, usually a ``functools.partial`` over the shared config
        factory with ``vendor`` filled in.
    capabilities : tuple of Capability
        Every ``(market, frequency, data_type)`` combination the vendor
        serves. Must not be empty.
    required_env : tuple of str
        Names of the environment variables that hold the vendor's
        credentials. Never a value, and never anything derived from one.
        They are written out here rather than read from ``acquisition_cls``
        so that a test can check the two against each other.
    universe_categories : tuple of UniverseCategory, default ()
        Symbol universes, such as S&P 500 constituents, that a user interface
        might offer beside this source. Informational only: the actual symbol
        list comes from ``UniverseCatalog``, and ``run()`` never reads this
        field.

    Notes
    -----
    ``acquisition_cls`` is a direct class reference, not a dotted path
    resolved at runtime. The cost is that importing this module imports
    every vendor's client library. The annotation names the ``Acquisition``
    base class, so the top of this module imports no vendor module, and each
    descriptor can be defined beside the class it describes.

    Examples
    --------
    >>> from quantlab.registry import DataSourceRegistry
    >>> source = DataSourceRegistry.get("tiingo")
    >>> source.display_name
    'Tiingo EOD'
    >>> source.required_env
    ('TIINGO_API_KEY',)
    >>> [(c.market, c.frequency) for c in source.capabilities]
    [('us_equity', '1d')]
    """

    vendor: Vendor
    display_name: str
    acquisition_cls: type[Acquisition]
    config_factory: Callable[..., AcquisitionConfig]
    capabilities: tuple[Capability, ...]
    required_env: tuple[str, ...]
    universe_categories: tuple[UniverseCategory, ...] = ()

    def supports(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> bool:
        """Return whether this source serves ``(market, frequency, data_type)``.

        This is the yes/no form of ``capabilities_for``, which holds the
        matching rule.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        bool
            True if at least one capability matches.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> source.supports("us_equity", "tick")
        True
        >>> source.supports("us_equity", "tick", "quotes")
        True
        >>> DataSourceRegistry.get("tiingo").supports("us_equity", "tick")
        False
        """
        return bool(self.capabilities_for(market, frequency, data_type))

    def capabilities_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> tuple[Capability, ...]:
        """Return every capability matching ``(market, frequency, data_type)``.

        With ``data_type=None``, a vendor that serves two data types at one
        ``(market, frequency)`` returns both. ``convert()`` relies on seeing
        all of them, so it can refuse an ambiguous request instead of
        picking one.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        tuple of Capability
            The matching capabilities in declaration order, or ``()`` when
            none match.

        Examples
        --------
        >>> source = DataSourceRegistry.get("alpaca")
        >>> [c.data_type for c in source.capabilities_for("us_equity", "tick")]
        ['quotes', 'trades']
        >>> DataSourceRegistry.get("tiingo").capabilities_for("us_equity", "tick")
        ()
        """
        return tuple(
            capability
            for capability in self.capabilities
            if capability.market == market
            and capability.frequency == frequency
            and (data_type is None or capability.data_type == data_type)
        )

    def _resolve_capability_field(
        self,
        field: str,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> object:
        """Resolve ``field`` for a request, preferring the capability's value.

        This is the shared rule behind ``acquisition_cls_for`` and
        ``config_factory_for``. With no matching capability, the descriptor's
        own field is returned. A matching capability that leaves the field
        ``None`` contributes the descriptor's default, so several capabilities
        served by one class agree. If the matches disagree, ``ValueError`` is
        raised instead of picking the first one, because the wrong class
        would ask for the wrong credential and reach the wrong product.
        """
        matches = self.capabilities_for(market, frequency, data_type)
        if not matches:
            return getattr(self, field)

        resolved = [
            getattr(capability, field) or getattr(self, field)
            for capability in matches
        ]
        # Two `functools.partial` factories are never identical objects, so
        # fall back to equality; class references compare by identity.
        first = resolved[0]
        if any(value is not first and value != first for value in resolved[1:]):
            requested = (market, frequency, data_type)
            raise ValueError(
                f"{self.display_name}: {requested!r} matched {len(matches)} "
                f"capabilities carrying different {field} values, with "
                f"data_type={[c.data_type for c in matches]!r}. Pass "
                f"data_type= to say which one you mean; this layer will not "
                f"choose for you, because the wrong choice reaches a different "
                f"vendor product."
            )
        return first

    def acquisition_cls_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> type[Acquisition]:
        """Return the ``Acquisition`` subclass that serves a request.

        This is the matching capability's ``acquisition_cls`` when it names
        one, and the descriptor's default otherwise.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        type[Acquisition]
            The class to construct for the download.

        Raises
        ------
        ValueError
            If two matching capabilities name different classes.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("wrds")
        >>> source.acquisition_cls_for("us_equity", "1d").__name__
        'WrdsCrspDailyAcquisition'
        >>> source.acquisition_cls_for("us_equity", "tick", "nbbo").__name__
        'WrdsTaqNbboAcquisition'
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "acquisition_cls", market, frequency, data_type
        )

    def config_factory_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> Callable[..., AcquisitionConfig]:
        """Return the ``AcquisitionConfig`` factory that serves a request.

        Resolved by the same rule as ``acquisition_cls_for``.

        Parameters
        ----------
        market : Market
            The requested market, for example ``"us_equity"``.
        frequency : Frequency
            The requested frequency, for example ``"1d"``.
        data_type : str or None, default None
            The requested data type, or ``None`` to match any data type.

        Returns
        -------
        callable
            The function that builds the download's ``AcquisitionConfig``.

        Raises
        ------
        ValueError
            If two matching capabilities name different factories.
            Pass ``data_type`` to disambiguate.

        Examples
        --------
        >>> source = DataSourceRegistry.get("tiingo")
        >>> source.config_factory_for("us_equity", "1d") is source.config_factory
        True
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "config_factory", market, frequency, data_type
        )


class DataSourceRegistry:
    """The collection of every registered ``SourceDescriptor``.

    ``register_source`` adds a descriptor when its vendor module is imported,
    so defining a source registers it. Read the registry through ``all()``
    and ``get()``; the class is never instantiated.

    Attributes
    ----------
    SOURCES : tuple of SourceDescriptor
        Every registered descriptor, in registration order.

    Examples
    --------
    >>> from quantlab.registry import DataSourceRegistry
    >>> [d.vendor for d in DataSourceRegistry.all()]
    ['alpaca', 'tiingo', 'wrds']
    >>> DataSourceRegistry.get("tiingo").display_name
    'Tiingo EOD'
    """

    # A tuple that `register_source` replaces and never mutates in place, so a
    # test that saves this attribute and restores it later gets the old
    # contents back.
    SOURCES: tuple[SourceDescriptor, ...] = ()

    @classmethod
    def all(cls) -> tuple[SourceDescriptor, ...]:
        """Return every registered descriptor, sorted by vendor.

        Sorting makes the result independent of which vendor module the
        caller happened to import first.

        Returns
        -------
        tuple of SourceDescriptor
            The descriptors sorted by ``vendor``, or ``()`` when nothing is
            registered.

        Examples
        --------
        >>> len(DataSourceRegistry.all())
        3
        >>> DataSourceRegistry.all()[0].vendor
        'alpaca'
        """
        return tuple(sorted(cls.SOURCES, key=lambda d: d.vendor))

    @classmethod
    def get(cls, vendor: str) -> SourceDescriptor:
        """Return the descriptor registered for ``vendor``.

        Parameters
        ----------
        vendor : str
            The vendor's short name, for example ``"tiingo"``.

        Returns
        -------
        SourceDescriptor
            The registered descriptor.

        Raises
        ------
        ValueError
            If no descriptor is registered for ``vendor``. The message lists
            the vendors that are registered.

        Examples
        --------
        >>> DataSourceRegistry.get("tiingo").vendor
        'tiingo'
        >>> DataSourceRegistry.get("bloomberg")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: No data source is registered for vendor 'bloomberg'. ...
        """
        for descriptor in cls.SOURCES:
            if descriptor.vendor == vendor:
                return descriptor
        raise ValueError(
            f"No data source is registered for vendor {vendor!r}. "
            f"Registered vendors: {sorted(d.vendor for d in cls.SOURCES)}. "
            f"A source registers itself when its module is imported; if this "
            f"vendor's module was never imported, the registry cannot know "
            f"about it (see the vendor imports at the bottom of "
            f"quantlab/registry.py)."
        )


def register_source(descriptor: SourceDescriptor) -> SourceDescriptor:
    """Register ``descriptor`` and return it unchanged.

    Call it around a module-level descriptor defined beside the acquisition
    class it describes, so the module-level name is the descriptor itself.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The vendor's descriptor.

    Returns
    -------
    SourceDescriptor
        ``descriptor`` itself.

    Raises
    ------
    ValueError
        If ``descriptor.capabilities`` is empty, or if a descriptor for the
        same vendor is already registered. A second market or frequency for
        a vendor is another ``Capability`` on its existing descriptor, not a
        second descriptor.

    Examples
    --------
    Defined beside the acquisition class it describes::

        EXAMPLE_SOURCE = register_source(
            SourceDescriptor(
                vendor="example",
                display_name="Example Vendor",
                acquisition_cls=ExampleAcquisition,
                config_factory=functools.partial(
                    stock_acquisition_config, vendor="example"
                ),
                capabilities=(
                    Capability(market="us_equity", frequency="1d"),
                ),
                required_env=("EXAMPLE_API_KEY",),
            )
        )
    """
    if not descriptor.capabilities:
        raise ValueError(
            f"Refusing to register vendor {descriptor.vendor!r} with an empty "
            f"`capabilities` tuple. A source that serves nothing is a "
            f"definition error; declare at least one "
            f"Capability(market=..., frequency=...)."
        )

    for existing in DataSourceRegistry.SOURCES:
        if existing.vendor == descriptor.vendor:
            raise ValueError(
                f"vendor {descriptor.vendor!r} is already registered "
                f"({existing.display_name!r}). Each vendor has exactly one "
                f"descriptor; express a second market, frequency or data type "
                f"as another Capability on the existing descriptor."
            )

    # Replace the tuple rather than mutate it; see the `SOURCES` comment.
    DataSourceRegistry.SOURCES += (descriptor,)
    return descriptor


def is_configured(descriptor: SourceDescriptor) -> bool:
    """Return whether every credential variable the source names is set.

    Answers ``False`` rather than raising on an unconfigured machine, and
    never turns a credential value into anything but this boolean. A variable
    set to the empty string counts as unset, which matches the check the
    vendor clients apply when they are constructed.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to check.

    Returns
    -------
    bool
        True if every name in ``descriptor.required_env`` is set and
        non-empty.

    Examples
    --------
    With ``TIINGO_API_KEY`` unset in the environment:

    >>> from quantlab.registry import DataSourceRegistry, is_configured
    >>> is_configured(DataSourceRegistry.get("tiingo"))
    False
    """
    return all(os.environ.get(name) for name in descriptor.required_env)


def credential_status(descriptor: SourceDescriptor) -> dict[str, bool]:
    """Return ``{variable_name: is_set}`` for each credential the source names.

    The values are booleans only; nothing here returns, logs or masks a
    credential value.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to check.

    Returns
    -------
    dict of str to bool
        One entry per name in ``descriptor.required_env``, or ``{}`` when the
        source needs no credentials.

    Examples
    --------
    With neither Alpaca variable set in the environment:

    >>> from quantlab.registry import DataSourceRegistry, credential_status
    >>> credential_status(DataSourceRegistry.get("alpaca"))
    {'APCA_API_KEY_ID': False, 'APCA_API_SECRET_KEY': False}
    """
    return {name: bool(os.environ.get(name)) for name in descriptor.required_env}


def run(
    descriptor: SourceDescriptor,
    config: AcquisitionConfig,
    *,
    refresh: bool = False,
    reporter: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> AcquisitionResult:
    """Download a raw tier in the current process and return the outcome.

    The acquisition class is looked up from the descriptor's capability for
    ``(config.market, config.frequency, config.kwargs["data_type"])``, so the
    caller never names a vendor class. Constructing that class is the first
    point at which a credential is required. The run writes raw parquet files
    (shards) and watermark files and stops there; converting them to Zarr is
    the separate ``convert()`` step.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source to download from.
    config : AcquisitionConfig
        What to download; its ``market``, ``frequency`` and optional
        ``kwargs["data_type"]`` pick the capability.
    refresh : bool, default False
        If True, call ``refresh()``, which downloads the covered window
        again. Otherwise call ``download()``, which fills only what is
        missing.
    reporter : ProgressReporter or None, default None
        Receives a ``ProgressEvent`` per batch. ``None`` keeps the default
        progress bar on stderr.
    cancel : CancelToken or None, default None
        Setting the token stops the run at the next batch boundary.
        Completed batches stay on disk, and a later call resumes from them.

    Returns
    -------
    AcquisitionResult
        The result the acquisition stored on its ``last_result`` attribute.
        Its ``failures`` are the ones this run met. The failure manifest on
        disk (a JSON file of failed symbols) accumulates across runs and may
        name more symbols; read it through ``SourceInspector.failures``.

    Examples
    --------
    Needs the vendor's credential in the environment and network access::

        from quantlab.base.config import AcquisitionConfig
        from quantlab.base.progress import CancelToken
        from quantlab.registry import DataSourceRegistry, run

        source = DataSourceRegistry.get("tiingo")
        config = AcquisitionConfig(
            market="us_equity", frequency="1d", vendor="tiingo",
            raw_data_dir_path="data/us_equity/1d/nasdaq_data/tiingo",
            watermark_path="data/us_equity/1d/nasdaq_data/_watermarks/tiingo",
            symbols=("AAPL", "MSFT"),
            start_date="2024-01-01", end_date="2024-05-31",
        )
        token = CancelToken()
        result = run(source, config, cancel=token)
        result.failures  # {symbol: reason} for this run only
    """
    acquisition_cls = descriptor.acquisition_cls_for(
        config.market, config.frequency, (config.kwargs or {}).get("data_type")
    )
    acquisition = acquisition_cls(config)
    acquisition.attach(reporter=reporter, cancel=cancel)
    if refresh:
        acquisition.refresh()
    else:
        acquisition.download()
    return acquisition.last_result


def convert(
    descriptor: SourceDescriptor,
    dataset_config: DatasetConfig,
    *,
    data_type: str | None = None,
    granularity: str = "year",
    on_new_listing: str = "refuse",
    predicted_peak_bytes: int | None = None,
    reporter: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> ConversionResult:
    """Convert an already downloaded raw tier into a Zarr store, in-process.

    This is the second step after ``run()``. It reads the raw parquet tier
    and writes the store through the capability's ``dataset_cls``, one time
    window at a time. It uses no vendor client and no credential. The
    capability is looked up by ``(dataset_config.market,
    dataset_config.frequency, data_type)``.

    No memory check runs here. A window too large for memory fails with an
    out-of-memory error, so choose the window size (``granularity``)
    yourself. ``predicted_peak_bytes`` is only copied into the result for
    reporting.

    Parameters
    ----------
    descriptor : SourceDescriptor
        The source whose raw tier is being converted.
    dataset_config : DatasetConfig
        Where the raw tier is and where the store goes.
    data_type : str or None, default None
        Picks one capability when the vendor serves several at this
        ``(market, frequency)``.
    granularity : str, default "year"
        Length of each conversion window, such as ``"year"`` or
        ``"quarter"``.
    on_new_listing : str, default "refuse"
        What to do with a symbol that is not on the store's fixed symbol
        axis; passed to ``from_raw_data_chunked``.
    predicted_peak_bytes : int or None, default None
        The caller's own memory estimate, copied into the result.
    reporter : ProgressReporter or None, default None
        Receives a ``ProgressEvent`` per window.
    cancel : CancelToken or None, default None
        Checked between windows. Windows already written stay in the store
        and in its record of finished windows, so a later call with the same
        config resumes at the first unwritten window and reports
        ``resumed=True``.

    Returns
    -------
    ConversionResult
        The result the dataset stored on ``last_chunk_result``, with
        ``predicted_peak_bytes`` filled in.

    Raises
    ------
    ValueError
        In three cases: the descriptor serves no such capability (the
        message lists what it does serve); several capabilities match with
        different dataset classes (pass ``data_type``); or the matching
        capability has no ``dataset_cls``, because its data is a stream of
        irregularly timed events that no panel can hold.

    Examples
    --------
    Needs a raw tier already downloaded by ``run()``::

        from quantlab.base.config import DatasetConfig
        from quantlab.registry import DataSourceRegistry, convert

        source = DataSourceRegistry.get("tiingo")
        dataset_config = DatasetConfig(
            raw_data_dir_path="data/us_equity/1d/nasdaq_data/tiingo",
            zarr_file_path="data/us_equity/1d/tiingo_1d.zarr",
            market="us_equity", frequency="1d", vendor="tiingo",
            start_date="2024-01-01", end_date="2024-05-31",
        )
        result = convert(source, dataset_config, granularity="year")
        result.windows_written

    A capability the vendor serves but no dataset can express is refused
    without touching disk:

    >>> tick_config = DatasetConfig(
    ...     raw_data_dir_path="data/alpaca", zarr_file_path="data/out.zarr",
    ...     market="us_equity",
    ...     frequency="tick", vendor="alpaca",
    ...     start_date="2024-01-01", end_date="2024-01-31",
    ... )
    >>> convert(DataSourceRegistry.get("alpaca"), tick_config,
    ...         data_type="quotes")  # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    ValueError: Alpaca Market Data: no raw-to-Zarr conversion exists for ...
    """
    matches = descriptor.capabilities_for(
        dataset_config.market, dataset_config.frequency, data_type
    )
    requested = (dataset_config.market, dataset_config.frequency, data_type)

    # No match: list what the descriptor does serve in the same message.
    if not matches:
        served = [
            (capability.market, capability.frequency, capability.data_type)
            for capability in descriptor.capabilities
        ]
        raise ValueError(
            f"{descriptor.display_name} serves no capability for "
            f"(market, frequency, data_type)={requested!r}. It serves "
            f"{served!r}. Adding one is adding a Capability row beside the "
            f"vendor class, never adding a branch here."
        )

    # Ambiguous: several capabilities match and name different dataset
    # classes. Refuse rather than let declaration order pick the one that
    # writes the store.
    targets = {capability.dataset_cls for capability in matches}
    if len(targets) > 1:
        raise ValueError(
            f"{descriptor.display_name}: {requested!r} matched "
            f"{len(matches)} capabilities carrying {len(targets)} different "
            f"conversion targets, with data_type="
            f"{[capability.data_type for capability in matches]!r}. Pass "
            f"data_type= to say which one you mean; this layer will not "
            f"choose for you, because the wrong choice writes a real store."
        )

    capability = matches[0]

    # The vendor serves this capability, but no dataset class can hold its
    # data; a missing `dataset_cls` is how a capability says so.
    if capability.dataset_cls is None:
        raise ValueError(
            f"{descriptor.display_name}: no raw-to-Zarr conversion exists for "
            f"{requested!r}. This capability's raw data is a stream of "
            f"individually timestamped events at irregular times (an "
            f"irregular event axis), and the regular [timestamp, symbol] "
            f"panel that every Dataset subclass writes cannot hold it. "
            f"Forcing it onto a regular grid would produce a panel that "
            f"looks plausible but is wrong. Conversion for this kind of data "
            f"is not supported yet; the raw parquet files that the download "
            f"writes are the usable output and can be queried with polars."
        )

    dataset = capability.dataset_cls(dataset_config)
    dataset.from_raw_data_chunked(
        granularity=granularity,
        on_new_listing=on_new_listing,
        reporter=reporter,
        cancel=cancel,
    )
    result = dataset.last_chunk_result
    if result is None:  # pragma: no cover -- defensive
        raise RuntimeError(
            f"{type(dataset).__name__}.from_raw_data_chunked() returned "
            f"without publishing a ConversionResult on `last_chunk_result`."
        )
    return dataclasses.replace(
        result, predicted_peak_bytes=predicted_peak_bytes
    )


# Vendor module imports, deliberately last.
#
# Importing a vendor module registers its descriptor, so this module imports
# all of them to make `import quantlab.registry` alone list every source.
# They cannot go at the top, because each vendor module imports this module
# for `register_source`. They also stay out of
# `quantlab/acquisition/__init__.py`, which is kept empty so that the
# credential-free inspector in `quantlab.acquisition._support` can be
# imported without loading any vendor client.
#
# Import the module objects, not names from them: if a caller imports a
# vendor module first, that module is only partly initialised while this code
# runs, and reading an attribute from it would fail. `wrds` holds the single
# WRDS descriptor and imports the TAQ and CRSP modules itself.
from quantlab.acquisition import alpaca as _alpaca  # noqa: E402,F401
from quantlab.acquisition import tiingo as _tiingo  # noqa: E402,F401
from quantlab.acquisition import wrds as _wrds  # noqa: E402,F401
