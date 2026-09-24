"""Registry of every data source quantlab can download from.

Each vendor is described once by a ``SourceDescriptor``: the names of the
environment variables that hold its credentials, the ``Acquisition`` class
that downloads from it, and the ``Capability`` rows (market, frequency, data
type) it actually serves. ``DataSourceRegistry`` enumerates the descriptors,
and the two module-level entry points ``run()`` and ``convert()`` let a caller
download a raw tier and convert it to Zarr without naming a vendor class.
Descriptors register themselves through ``register_source`` beside the
acquisition class they describe; the vendor modules are imported at the bottom
of this file so that ``import quantlab.registry`` alone enumerates every
source.

Descriptors carry environment variable names only. No function here returns,
logs or embeds a credential value, not even a masked one, and no descriptor
carries a base URL or host. See ``docs/registry.md`` for a tour.
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

    A descriptor lists capabilities explicitly rather than as a product of
    markets and frequencies, because vendors serve irregular combinations
    (Alpaca's tick data is US-equity only and splits into quotes and trades).
    Anything that varies per combination, such as the earliest available date
    or the dataset class that converts it, lives here rather than on the
    descriptor. The class is frozen; extend it by adding optional fields.

    Example:
        >>> from quantlab.registry import Capability
        >>> cap = Capability(market="us_equity", frequency="1d", data_type="bars")
        >>> cap.data_type
        'bars'
        >>> cap.dataset_cls is None
        True
    """

    market: Market
    frequency: Frequency
    #: ``"bars"``, ``"quotes"``, ``"trades"`` and so on, or ``None`` when the
    #: vendor draws no such distinction. ``None`` is the absence of a data
    #: type, not a wildcard.
    data_type: str | None = None
    #: ISO date of the earliest data the vendor serves, when known. Advisory;
    #: nothing gates on it.
    earliest_available: str | None = None
    #: The subscription tier this capability needs, when the vendor has tiers.
    entitlement: str | None = None
    #: The ``MarketDataset`` subclass that converts this capability's raw tier
    #: into Zarr, or ``None`` when no dataset can express its axis (tick data
    #: on an irregular event axis). ``convert()`` refuses a capability whose
    #: ``dataset_cls`` is ``None``, so adding a conversion means filling this
    #: field, not adding a branch. A direct class reference, never a dotted
    #: path; annotated with the abstract base so this module's top stays free
    #: of vendor imports.
    dataset_cls: type[MarketDataset] | None = None
    #: The ``Acquisition`` subclass that downloads this capability, or ``None``
    #: to use the descriptor's own ``acquisition_cls``. One vendor account may
    #: serve several products through several classes (WRDS serves both TAQ
    #: NBBO and CRSP daily data), which makes the downloading class a
    #: per-capability fact. A direct class reference, never a dotted path.
    acquisition_cls: type[Acquisition] | None = None
    #: The ``AcquisitionConfig`` factory for this capability, or ``None`` to
    #: use the descriptor's own ``config_factory``. Resolved by the same rule
    #: as ``acquisition_cls``.
    config_factory: Callable[..., AcquisitionConfig] | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    """Everything the registry knows about one vendor.

    Vendor-level facts (credential names, display name, the default
    acquisition class and config factory) are stated once here; everything
    that varies by market, frequency or data type lives in ``capabilities``.
    Exactly one descriptor is registered per vendor.

    ``acquisition_cls`` is a direct class reference rather than a dotted path
    resolved at runtime. The accepted cost is that importing this module
    imports every vendor SDK. The annotation names the ``Acquisition`` base
    class so the top of this module imports no vendor module, which is what
    lets each descriptor be defined beside the class it describes.

    Example:
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
    #: Builds an ``AcquisitionConfig`` for this vendor; usually a
    #: ``functools.partial`` over the shared config factory with ``vendor``
    #: bound.
    config_factory: Callable[..., AcquisitionConfig]
    capabilities: tuple[Capability, ...]
    #: Names of the environment variables that hold this vendor's credentials.
    #: Never a value, and never anything derived from one. Restated here as
    #: literals rather than read off ``acquisition_cls`` so that a test can
    #: check the two against each other.
    required_env: tuple[str, ...]
    #: Universe categories a console might offer beside this source. Advisory
    #: only: the roster comes from ``UniverseCatalog``, and ``run()`` never
    #: reads this field.
    universe_categories: tuple[UniverseCategory, ...] = ()

    def supports(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> bool:
        """Return whether this source serves ``(market, frequency, data_type)``.

        ``data_type=None`` means "any data type". The match rule itself lives
        in ``capabilities_for``; this is its boolean form.

        Example:
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

        ``data_type=None`` matches any data type, so a vendor that serves two
        data types at one ``(market, frequency)`` returns both; ``convert()``
        relies on seeing all of them so it can refuse an ambiguous request
        instead of picking one. No match returns ``()`` rather than raising.

        Example:
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

        The one rule behind ``acquisition_cls_for`` and ``config_factory_for``.
        With no matching capability the descriptor's own field is returned.
        Matching capabilities that leave the field ``None`` contribute the
        descriptor default, so several rows served by one class agree. Matches
        that disagree raise ``ValueError`` rather than being resolved by
        declaration order, because the wrong class demands the wrong
        credential.
        """
        matches = self.capabilities_for(market, frequency, data_type)
        if not matches:
            return getattr(self, field)

        resolved = [
            getattr(capability, field) or getattr(self, field)
            for capability in matches
        ]
        # Identity-then-equality: a class reference compares by identity, a
        # `functools.partial` config factory does not, and both must count as
        # "the same value" here.
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
        """Return the ``Acquisition`` subclass serving a request.

        The matching capability's ``acquisition_cls`` when it names one, this
        descriptor's default otherwise.

        Raises:
            ValueError: If two matching capabilities name different classes.
                Pass ``data_type`` to disambiguate.

        Example:
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
        """Return the ``AcquisitionConfig`` factory serving a request.

        The companion of ``acquisition_cls_for``, resolved by the same rule
        and raising ``ValueError`` on the same ambiguity.

        Example:
            >>> source = DataSourceRegistry.get("tiingo")
            >>> source.config_factory_for("us_equity", "1d") is source.config_factory
            True
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "config_factory", market, frequency, data_type
        )


class DataSourceRegistry:
    """Enumeration of every registered ``SourceDescriptor``.

    Descriptors are added by ``register_source`` when their vendor module is
    imported, so defining a source registers it. Read the registry through
    ``all()`` and ``get()``.

    Example:
        >>> from quantlab.registry import DataSourceRegistry
        >>> [d.vendor for d in DataSourceRegistry.all()]
        ['alpaca', 'tiingo', 'wrds']
        >>> DataSourceRegistry.get("tiingo").display_name
        'Tiingo EOD'
    """

    #: Every registered descriptor, in registration order; ``all()`` sorts.
    #: A tuple that ``register_source`` rebinds and never mutates in place, so
    #: that a caller who saves this attribute and restores it later (a test
    #: fixture isolating the registry, for instance) gets the old contents
    #: back.
    SOURCES: tuple[SourceDescriptor, ...] = ()

    @classmethod
    def all(cls) -> tuple[SourceDescriptor, ...]:
        """Return every registered descriptor, sorted by vendor.

        Sorted rather than in import order so that a display built from this
        call does not depend on which module the caller imported first. An
        empty registry returns ``()``.

        Example:
            >>> len(DataSourceRegistry.all())
            3
            >>> DataSourceRegistry.all()[0].vendor
            'alpaca'
        """
        return tuple(sorted(cls.SOURCES, key=lambda d: d.vendor))

    @classmethod
    def get(cls, vendor: str) -> SourceDescriptor:
        """Return the descriptor registered for ``vendor``.

        Raises:
            ValueError: If no descriptor is registered for ``vendor``. The
                message lists the vendors that are registered.

        Example:
            >>> DataSourceRegistry.get("tiingo").vendor
            'tiingo'
            >>> DataSourceRegistry.get("bloomberg")
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

    Meant to wrap a module-level descriptor literal beside the acquisition
    class it describes, so the module-level name is the descriptor itself.

    Raises:
        ValueError: If a descriptor for the same vendor is already registered
            (a second market or frequency is another ``Capability`` on the
            existing descriptor, not a second descriptor), or if
            ``descriptor.capabilities`` is empty.

    Example:
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
            f"definition error, not an enumerable source -- declare at least "
            f"one Capability(market=..., frequency=...)."
        )

    for existing in DataSourceRegistry.SOURCES:
        if existing.vendor == descriptor.vendor:
            raise ValueError(
                f"vendor {descriptor.vendor!r} is already registered "
                f"({existing.display_name!r}). ONE descriptor per VENDOR "
                f"(D-01) -- express a second market/frequency/data_type as "
                f"another Capability on the existing descriptor, not as a "
                f"second descriptor."
            )

    # Rebound, never appended in place: see the `SOURCES` comment above.
    DataSourceRegistry.SOURCES += (descriptor,)
    return descriptor


def is_configured(descriptor: SourceDescriptor) -> bool:
    """Return whether every env var the source names is set and non-empty.

    Answers ``False`` rather than raising on an unconfigured machine, and
    never reads a credential value into anything but the boolean. An env var
    set to the empty string counts as unset, which is the same predicate the
    vendor clients apply at construction.

    Example:
        With ``TIINGO_API_KEY`` unset in the environment:

        >>> from quantlab.registry import DataSourceRegistry, is_configured
        >>> is_configured(DataSourceRegistry.get("tiingo"))
        False
    """
    return all(os.environ.get(name) for name in descriptor.required_env)


def credential_status(descriptor: SourceDescriptor) -> dict[str, bool]:
    """Return ``{env_var_name: is_set}`` for each credential the source names.

    The values are booleans only; no code path here returns, logs or masks a
    credential value. A descriptor with no ``required_env`` returns ``{}``
    without reading the environment.

    Example:
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
    """Download a raw tier in-process and return the run's outcome.

    The acquisition class is resolved from the descriptor's capability for
    ``(config.market, config.frequency, config.kwargs["data_type"])``, so the
    caller never names a vendor class. Constructing that class is the first
    point at which a credential is demanded. The run writes raw parquet
    shards and watermark sidecars and stops there; converting them to Zarr is
    the separate ``convert()`` call below.

    Args:
        descriptor: The source to download from.
        config: What to download; its ``market``, ``frequency`` and optional
            ``kwargs["data_type"]`` pick the capability.
        refresh: Call ``refresh()`` (re-download the covered window) instead
            of ``download()`` (fill what is missing).
        reporter: Receives a ``ProgressEvent`` per batch. ``None`` keeps the
            default stderr progress bar.
        cancel: A ``CancelToken``; setting it stops the run at the next batch
            boundary. Completed batches stay on disk and a later call resumes.

    Returns:
        The ``AcquisitionResult`` the run published on ``last_result``. Its
        ``failures`` are the ones this run met; the on-disk failure manifest
        accumulates across runs and may name more symbols (read it through
        ``SourceInspector.failures``).

    Example:
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
    """Convert an already-downloaded raw tier into a Zarr store, in-process.

    The counterpart of ``run()``: it reads the raw parquet tier and writes the
    store through the capability's ``dataset_cls`` and its chunked conversion
    loop, and touches no vendor client or credential. The capability is
    looked up by ``(dataset_config.market, dataset_config.frequency,
    data_type)``.

    No memory guard runs here. An over-sized window goes straight to an
    out-of-memory error, so size the window (or pass a smaller
    ``granularity``) yourself; ``predicted_peak_bytes`` is only echoed into
    the result for reporting.

    Args:
        descriptor: The source whose raw tier is being converted.
        dataset_config: Where the raw tier is and where the store goes.
        data_type: Picks one capability when the vendor serves several at
            this ``(market, frequency)``.
        granularity: Window size of the chunked conversion (``"year"``,
            ``"quarter"``, ...).
        on_new_listing: How a symbol absent from the pinned axis is handled;
            forwarded to ``from_raw_data_chunked``.
        predicted_peak_bytes: A caller's own estimate, copied into the result.
        reporter: Receives a ``ProgressEvent`` per window.
        cancel: A ``CancelToken``, observed at window boundaries. Windows
            already written stay in the store and the chunk ledger, so a later
            call over the same config resumes at the first unwritten window
            and reports ``resumed=True``.

    Returns:
        The ``ConversionResult`` the dataset published on
        ``last_chunk_result``, with ``predicted_peak_bytes`` filled in.

    Raises:
        ValueError: In three distinguishable cases: the descriptor serves no
            such capability (the message lists what it does serve); several
            capabilities match with different conversion targets (pass
            ``data_type``); or the matching capability has no ``dataset_cls``
            because its raw tier is an irregular event stream no panel can
            express.

    Example:
        Needs a raw tier already downloaded by ``run()``::

            from quantlab.base.config import DatasetConfig
            from quantlab.registry import DataSourceRegistry, convert

            source = DataSourceRegistry.get("tiingo")
            dataset_config = DatasetConfig(
                raw_data_dir_path="data/us_equity/1d/nasdaq_data/tiingo",
                zarr_file_path="data/us_equity/1d/tiingo_1d.zarr",
                catalog_path="data/catalog",
                market="us_equity", frequency="1d", vendor="tiingo",
                start_date="2024-01-01", end_date="2024-05-31",
            )
            result = convert(source, dataset_config, granularity="year")
            result.windows_written

        A capability the vendor serves but no dataset can express is refused
        without touching disk:

        >>> tick_config = DatasetConfig(
        ...     raw_data_dir_path="data/alpaca", zarr_file_path="data/out.zarr",
        ...     catalog_path="data/catalog", market="us_equity",
        ...     frequency="tick", vendor="alpaca",
        ...     start_date="2024-01-01", end_date="2024-01-31",
        ... )
        >>> convert(DataSourceRegistry.get("alpaca"), tick_config, data_type="quotes")
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

    # Ambiguous: several rows matched and disagree about the conversion
    # target. Refuse rather than let declaration order pick the class that
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

    # No conversion target: the vendor serves this capability, but no dataset
    # class can express its axis. The absent field is the refusal.
    if capability.dataset_cls is None:
        raise ValueError(
            f"{descriptor.display_name}: no raw-to-Zarr conversion exists for "
            f"{requested!r}. This capability's raw tier is a stream of "
            f"individually-timestamped events on an irregular event axis, and "
            f"the dense [timestamp, symbol] panel every Dataset subclass "
            f"writes cannot express one -- flattening it onto a dense grid "
            f"would produce a plausible-looking panel that is scientifically "
            f"wrong, and a wrong panel that loads is worse than this refusal. "
            f"That axis is phase 03.3's work (03.4 D-18). Until it lands, the "
            f"raw parquet shards this capability acquires ARE the deliverable "
            f"and are already queryable with polars."
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


# ---------------------------------------------------------------------------
# Vendor module imports, deliberately last.
#
# The registry is populated by `register_source` as vendor modules are
# imported, so this module must import every vendor module for a cold
# `import quantlab.registry` to enumerate every source. They cannot go at the
# top: each vendor module imports this module for the decorator, so the
# descriptor and the class it describes could not share a file. They also do
# not go in `quantlab/acquisition/__init__.py`, which stays empty so that the
# credential-free read surface (`quantlab.acquisition._support.inspector`) can
# be imported without any vendor client being loaded ahead of it.
#
# Module-object form (`from quantlab.acquisition import tiingo`) rather than
# `from quantlab.acquisition.tiingo import TiingoAcquisition`: when a caller
# imports a vendor module first, this module runs while that module is only
# partially initialised, and binding the module object is safe where reading
# an attribute off it would raise. Order between the three lines does not
# matter; `all()` sorts.
#
# `wrds` is a neutral module holding the one WRDS descriptor and importing the
# WRDS provider modules (TAQ, CRSP) itself, because one WRDS account serves
# several products through several acquisition classes.
# ---------------------------------------------------------------------------
from quantlab.acquisition import alpaca as _alpaca  # noqa: E402,F401
from quantlab.acquisition import tiingo as _tiingo  # noqa: E402,F401
from quantlab.acquisition import wrds as _wrds  # noqa: E402,F401
