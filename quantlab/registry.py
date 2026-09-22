"""Every acquirable data source, described ONCE and enumerable from here.

This module is what an operator surface -- quantlab's own ingest shells today,
the out-of-repo `quantlab-console` tomorrow -- asks "what can this project
download, what does each source serve, and is it configured?". Adding a vendor
is registering a descriptor beside its acquisition class, not editing five call
sites.

**ONE class-level registration tuple**, in the spirit of
`quantlab/universe.py:1249` (`UniverseCatalog.MEMBERSHIP_FETCHERS`)
and consumed the same way: a plain loop over the tuple with the discriminator
read off each element (`descriptor.vendor`), never an `if` branch on a vendor
literal.

**Its elements are INSTANCES where `MEMBERSHIP_FETCHERS` holds CLASSES**, and
that difference is a deliberate decision (03.4-CONTEXT.md D-05), not drift. Two
reasons:

- the `Acquisition` subclasses are already heavy with `RAW_SCHEMA`,
  `QUOTA_STATUS_CODES`, `FIELD_MAP`, `ENDPOINT_MAP`,
  `RAW_COLUMNS_BY_DATA_TYPE`, `LEGACY_WATERMARK_POLICIES` ...  -- hanging "who I
  am" off the same class as "how I download" makes both harder to read;
- separating them lets ONE acquisition class carry SEVERAL descriptors later
  (alpaca paper vs live, say) without the class knowing anything about it.

It is still one registry in one style, which is what the ROADMAP's "do not
invent a second registry style" asks for.

**The credential rule is absolute.** A descriptor carries env var NAMES. There
is no API here that returns, logs, or embeds a credential VALUE -- not even a
masked or partially-redacted one, which looks responsible and is the shape the
next leak takes. This repository has already leaked one real Tiingo key
(Phase 1); `is_configured` returns a `bool` and `credential_status` returns a
`bool` per NAME, full stop.

**A descriptor also carries no transport target** -- no base URL, no host, no
endpoint prefix. Making the wire destination configurable turns an operator
console into a credential-exfiltration surface;
`_AlpacaMarketDataClient.BASE_URL` stays pinned in its own module with its own
comment.
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
    """ONE `(market, frequency, data_type)` combination a vendor actually serves.

    A LIST of these, never a cross-product of two flat tuples (D-01/D-02).
    Alpaca's `tick` is `us_equity`-only and splits further into `quotes` and
    `trades`, each with its own `ENDPOINT_MAP` entry, so a
    `markets x frequencies` product would admit combinations the vendor
    rejects -- and a registry that advertises a combination the vendor refuses
    is worse than one that advertises nothing.

    **Per-capability data goes HERE, never on `SourceDescriptor`.** That is the
    lesson `UniverseCatalog`'s own registry comments already argue one level
    up: a roster has no `PIT_COVERAGE_START`, so registering it beside the
    membership fetchers would impose a boundary it must not have. A field only
    some rows can populate is a signal to move the field down, not to widen the
    container. `earliest_available` and `entitlement` are exactly such fields.

    Frozen, and extensible by adding OPTIONAL fields (D-02) -- never by
    smuggling a dict in, which would put the vendor-specific part beyond the
    reach of any type checker or any test that enumerates it.
    """

    market: Market
    frequency: Frequency
    #: `"bars"` / `"quotes"` / `"trades"`, or `None` where the vendor has no
    #: such concept (Tiingo's EOD endpoint serves one thing and names it
    #: nothing). `None` is the ABSENCE of a distinction, not a wildcard.
    data_type: str | None = None
    #: ISO date of the earliest data the vendor will serve, when known.
    #: Advisory: nothing gates on it yet.
    earliest_available: str | None = None
    #: The subscription tier this capability needs, when the vendor has tiers.
    entitlement: str | None = None
    #: The `MarketDataset` subclass that materialises THIS capability's raw
    #: tier into Zarr, or `None` where no Dataset can express the axis yet
    #: (tick -> phase 03.3). `None` IS the SC-7 refusal, expressed as DATA:
    #: `convert()` below carries no `if frequency == "tick"` branch and no
    #: vendor literal, because the absence of a conversion target is the
    #: refusal. Phase 03.3 turns the refusal off by FILLING two fields, not by
    #: deleting a branch.
    #:
    #: On `Capability` rather than on `SourceDescriptor` (03.5 D-01) for the
    #: reason this class's own docstring above already states: only some rows
    #: can populate it. It is also exactly `Capability`'s key --
    #: `(market, frequency, data_type)` -- that Dataset subclasses divide on,
    #: so `StockDataset` appearing on three rows across two vendors is one
    #: correct answer to three questions, not a duplicated decision.
    #:
    #: Annotated with the `MarketDataset` ABC, following the rule
    #: `SourceDescriptor.acquisition_cls` states below: the TOP of this module
    #: must stay vendor-free, or the descriptors could not be defined beside
    #: the classes they describe. The concrete class reference is DIRECT
    #: (03.4 D-03), never a dotted path resolved at runtime.
    dataset_cls: type[MarketDataset] | None = None
    #: The `Acquisition` subclass that downloads THIS capability's raw tier, or
    #: `None` to mean "the descriptor's own `acquisition_cls`" (03.10 D-12).
    #:
    #: One VENDOR may serve several products through several classes: the WRDS
    #: account carries both NYSE TAQ millisecond NBBO and CRSP daily stock, and
    #: D-01 keeps that as ONE descriptor, so "which class downloads this" stops
    #: being a vendor-level fact and becomes a capability-level one -- exactly
    #: the move this class's own docstring argues for ("a field only some rows
    #: can populate is a signal to move the field down").
    #:
    #: `None` is the ABSENCE of an override, never "no class": a capability
    #: declared before this field existed keeps resolving to the vendor
    #: default, which is what makes the field purely additive. The
    #: descriptor-level pair stays REQUIRED and is the default rather than a
    #: fallback of last resort, because four ingest shells read
    #: `SOURCE.acquisition_cls.DEFAULT_BATCH_SIZE` (and friends) directly.
    #:
    #: A DIRECT class reference (03.4 D-03), never a dotted path, and annotated
    #: with the `Acquisition` ABC for the reason `SourceDescriptor` states
    #: below: the top of this module must stay vendor-free.
    acquisition_cls: type[Acquisition] | None = None
    #: The config factory for THIS capability, or `None` for the descriptor's
    #: own `config_factory`. The companion of `acquisition_cls` above and
    #: resolved by the same rule: a vendor whose two products need differently
    #: shaped `AcquisitionConfig`s (different raw roots, different required
    #: `kwargs`) would otherwise have to pick one factory for both.
    config_factory: Callable[..., AcquisitionConfig] | None = None


@dataclass(frozen=True)
class SourceDescriptor:
    """"Who I am" for one acquirable data source, keyed by VENDOR (D-01).

    Vendor-level facts -- credentials, display name, the acquisition class --
    are stated once here; everything that varies per market/frequency lives in
    `capabilities`. One descriptor per vendor, enforced at registration.

    `acquisition_cls` is a DIRECT class reference (D-03), matching
    `UniverseCatalog.MEMBERSHIP_FETCHERS`, not a dotted path resolved through
    `quantlab/utils/module.py:get_cls_from_path`. The dotted-path idiom exists
    in this repo (`Acquisition.import_path` feeds it) and was considered and
    rejected: the accepted cost is that importing this registry imports every
    vendor SDK, which is an import cost rather than a fragility because
    `tiingo` and `requests` are already hard quantlab dependencies.

    The type annotation names the `Acquisition` ABC from
    `quantlab.base.acquisition`, which imports no vendor module -- the TOP of
    this file must stay vendor-free, or the descriptors could not be defined
    beside the classes they describe (see the bottom of this file).
    """

    vendor: Vendor
    display_name: str
    acquisition_cls: type[Acquisition]
    #: Usually `functools.partial(stock_acquisition_config, vendor=...)`. There
    #: is exactly one acquisition config factory serving both vendors, so a
    #: partial over it is the whole "factory" -- a per-vendor function would be
    #: a second copy of the same body differing by one keyword.
    config_factory: Callable[..., AcquisitionConfig]
    capabilities: tuple[Capability, ...]
    #: Credential env var NAMES. Never a value, and never anything derived from
    #: a value. Restated as literals on each descriptor rather than sourced
    #: from `acquisition_cls.CREDENTIAL_ENV_VARS`, because D-04 demands a test
    #: pinning the two against each other and deriving one from the other
    #: makes that test the tautology `x == x`.
    required_env: tuple[str, ...]
    #: ADVISORY ONLY, and deliberately so: the roster comes from
    #: `UniverseCatalog`, not from the vendor, so this is a display hint about
    #: which categories a console might offer beside this source. It is never a
    #: gate, and nothing in `run()` reads it. Both vendors are US-equity and
    #: every category resolves against either of them today, so a narrower
    #: tuple would encode a distinction that does not exist.
    universe_categories: tuple[UniverseCategory, ...] = ()

    def supports(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> bool:
        """Whether this source serves `(market, frequency[, data_type])`.

        Answered from `capabilities` alone. There is no vendor literal in this
        body and there must never be one: the whole point of the capability
        list is that adding a market or a frequency is adding a `Capability`,
        not editing a branch.

        `data_type=None` means "do not care", which is why
        `TIINGO_SOURCE.supports("us_equity", "tick")` is `False` (Tiingo serves
        no tick capability at all) rather than accidentally `True` via
        Tiingo's own `data_type=None` capability.

        The PREDICATE itself lives in `capabilities_for()` below and is not
        restated here: two copies of a match rule is two places for the
        "`data_type=None` means do not care" semantics to drift apart, and the
        drift would be silent -- a boolean that disagrees with the tuple the
        conversion entry point actually resolves.
        """
        return bool(self.capabilities_for(market, frequency, data_type))

    def capabilities_for(
        self,
        market: Market,
        frequency: Frequency,
        data_type: str | None = None,
    ) -> tuple[Capability, ...]:
        """EVERY capability matching `(market, frequency[, data_type])`.

        The tuple-valued half of `supports()`, and the lookup
        `quantlab/registry.py:convert()` resolves a conversion
        target through (03.5 D-03). Same match rule, same
        "`data_type=None` means do not care" semantics -- stated ONCE, here.

        Returns ALL matches rather than the first, and the plurality is
        load-bearing: a vendor serving two shapes at one `(market, frequency)`
        is exactly the case 03.5 D-02 restored `DatasetConfig.market` for, and
        `convert()` must be able to REFUSE it rather than silently pick a row.
        `()` over no match is a legitimate answer a caller renders, not an
        error condition -- the raising happens in `convert()`, where there is
        a request to name.
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
        """`field` for `(market, frequency[, data_type])`, capability first.

        The ONE resolution rule behind `acquisition_cls_for` and
        `config_factory_for`, stated once so the two cannot drift -- the same
        reason `supports()` defers its predicate to `capabilities_for()`.

        Three arms, in the order a caller meets them:

        - **no match** -> the descriptor's own field. That is today's behaviour
          exactly, and it is what keeps this change additive: every `run()`
          call site that predates the per-capability fields reaches the vendor
          default through this arm, unchanged.
        - **matches that AGREE** -> the single value. A capability leaving the
          field `None` contributes the descriptor default, so a vendor serving
          two shapes through one class is not an ambiguity.
        - **matches that DISAGREE** -> `ValueError`. Refusing is the only
          honest answer, for the reason `convert()` states one axis over:
          picking `matches[0]` would let capability declaration ORDER -- an
          authoring detail invisible at the call site -- decide which vendor
          class receives the config, and so which credential is demanded
          (T-03.10-41).

        There is no vendor literal and no frequency literal in this body: every
        message is built from the descriptor's own data, which is what keeps
        the capability list, not this method, the place a new combination is
        added.
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
        """The `Acquisition` subclass serving `(market, frequency[, data_type])`.

        The capability's own `acquisition_cls` when it names one, this
        descriptor's default otherwise. Raises when two matching capabilities
        disagree -- see `_resolve_capability_field` for the rule.
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
        """The config factory serving `(market, frequency[, data_type])`.

        The `acquisition_cls_for` companion, same rule, same refusal.
        """
        return self._resolve_capability_field(  # type: ignore[return-value]
            "config_factory", market, frequency, data_type
        )


class DataSourceRegistry:
    """The one registry of acquirable data sources.

    Populated by `@register_source` at descriptor-definition time (D-06), so
    defining a source registers it and it cannot be forgotten -- the failure
    mode a hand-maintained tuple has is a new vendor that works everywhere
    except in the enumeration an operator sees.
    """

    #: Every registered descriptor, in REGISTRATION order. Read through
    #: `all()`, which sorts.
    #:
    #: A `tuple`, REBOUND by `register_source` (`SOURCES += (descriptor,)`) and
    #: never mutated in place. That is not a style preference: the
    #: `isolated_registry` test fixture isolates by `monkeypatch.setattr`-ing
    #: this attribute, which saves the old OBJECT and puts it back on teardown.
    #: A `list.append` would mutate the very object being restored, the fixture
    #: would silently stop isolating, and a fake descriptor would leak into
    #: every later test in the session.
    #: `tests/test_source_registry.py::test_registration_tuple_shape_...` is
    #: what keeps this honest.
    SOURCES: tuple[SourceDescriptor, ...] = ()

    @classmethod
    def all(cls) -> tuple[SourceDescriptor, ...]:
        """Every registered descriptor, SORTED BY VENDOR.

        Returns `()` over an empty registry rather than raising: "nothing is
        registered" is a legitimate state for a caller to render, and an
        exception would make the empty case the caller's problem.
        """
        #: Sorted rather than import order, deliberately. Enumeration order
        #: becomes an operator surface's DISPLAY order, and import order is a
        #: function of which module the caller happened to touch first --
        #: `import quantlab.acquisition.tiingo` and
        #: `import quantlab.registry` would render the same
        #: installation's sources in two different orders. Sorting also makes
        #: an assertion on this method a literal tuple comparison.
        return tuple(sorted(cls.SOURCES, key=lambda d: d.vendor))

    @classmethod
    def get(cls, vendor: str) -> SourceDescriptor:
        """The single descriptor registered for `vendor`.

        Raises `ValueError` naming the requested vendor AND listing the
        registered ones -- a caller who mistypes a token learns what was
        available in the same message, rather than getting a bare `KeyError`
        that says nothing about a registry they cannot see.
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
    """Register `descriptor` and return it UNCHANGED, by identity.

    Written to be used as a decorator over a module-level descriptor literal,
    beside the acquisition class it describes::

        TIINGO_SOURCE = register_source(SourceDescriptor(...))

    Returning the descriptor itself is what makes the decorated module-level
    name the descriptor and never `None` -- the classic decorator bug, and one
    that would only surface at the first attribute read far from here.

    Refuses two things, both `ValueError`:

    - a `vendor` that is already registered (D-01). A second market or
      frequency is another `Capability`, not another descriptor. The collision
      never merges capability lists and never silently replaces the first
      descriptor: a registry that quietly took the last definition would make
      import order decide what a vendor can do.
    - an EMPTY `capabilities` tuple (D-03). A source that serves nothing is a
      definition error rather than an enumerable source; letting it register
      puts a row in an operator's list that can answer no question.
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

    # REBOUND, never `.append`ed -- see the `SOURCES` comment above for why the
    # `isolated_registry` fixture's isolation depends on it.
    DataSourceRegistry.SOURCES += (descriptor,)
    return descriptor


def is_configured(descriptor: SourceDescriptor) -> bool:
    """Whether every env var this source NAMES is present and non-empty.

    Returns a `bool`. It never returns, logs, or embeds a credential VALUE,
    and it never requires one to be present in order to answer -- an operator
    browsing sources on an unconfigured laptop gets a real answer, not an
    exception.

    Presence-and-non-empty is exactly the predicate both vendors already
    enforce at construction -- `acquisition/tiingo.py` (`if not
    os.environ.get(KEY_ENV)`) and `acquisition/alpaca.py` (`if not key or not
    secret`) -- so a source this reports as configured is one whose client will
    construct, and an env var set to the empty string reads as NOT configured
    in both places identically.
    """
    return all(os.environ.get(name) for name in descriptor.required_env)


def credential_status(descriptor: SourceDescriptor) -> dict[str, bool]:
    """Per-NAME presence, for an operator who needs to know WHICH one is missing.

    Values are booleans. There is no code path here that reads a value into a
    return, a log line, or an exception message, and there must never be a
    "masked" variant either: a partially-redacted key still narrows the search
    space for whoever reads the screenshot.

    A descriptor with `required_env=()` gets `{}` -- and gets there without
    reading any environment variable at all, because the comprehension has
    nothing to iterate.
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
    """Start an acquisition IN-PROCESS and return its outcome (D-12 / D-14).

    The vendor is reached through `descriptor.acquisition_cls_for(...)`, so no
    caller names an acquisition class. `download()` and `refresh()` keep their
    `-> Self` chaining contract; the outcome is read off `last_result`.

    **The class comes from the CAPABILITY, never from a vendor branch**
    (03.10 D-12). The lookup key is
    `(config.market, config.frequency, config.kwargs["data_type"])` -- exactly
    `Capability`'s own key, the same triple `convert()` resolves a conversion
    target through. A capability that names no `acquisition_cls` resolves to
    the descriptor's default, so every call site that predates the
    per-capability fields constructs precisely the class it constructed
    before; a request matching two capabilities that disagree is REFUSED
    rather than resolved by declaration order, because the wrong class demands
    the wrong credential.

    **What the returned object says, and what it does not.** Its `failures`
    are the ones THIS run discovered, always inside its own `requested`. The
    crash-durable `_failures.json` beside the watermarks is the wider
    cross-run record and may name symbols this run never attempted, so the two
    are related by `set(result.failures) <= set(manifest)` -- containment, not
    equality (D-18; the equality was retired by REVIEW CR-01, where sharing
    one dict made a `--symbols AAPL` run report another roster's 404s as its
    own). Read the wider set through `SourceInspector.failures()`.

    **ACQUISITION-ONLY, and that is D-14's amendment stated as code.** This
    downloads to the raw parquet tier and STOPS. It performs no raw-to-Zarr
    conversion, mirroring `Acquisition`'s own contract that acquisition classes
    never touch xarray/Zarr storage. Conversion is the SEPARATE registry-level
    call `convert()` directly below, never a flag here.

    Why the separation, now that 03.5 D-06 has retired 03.4's original
    argument for it (conversion is ONE chunked path with ONE guard, and all
    three ingest shells reach it through `convert()`): folding conversion in
    here would turn an ACQUISITION-ONLY contract into a
    conversion-sometimes contract, and the two-call shape is precisely what
    lets a caller acquire without converting -- a credentialled backfill onto
    a machine that will never densify a panel -- and convert without
    acquiring, which is what every re-derivation of an existing raw tier is.
    `run()` returning an `AcquisitionResult` while `convert()` returns a
    `ConversionResult` is that difference stated in the type system.

    Constructing the resolved class is the FIRST point a credential is
    demanded, deliberately: that is the vendor class's own fail-fast guard, and
    moving it later would turn fail-fast into fail-late.

    **`reporter` and `cancel` are the console's two handles on a running
    acquisition** (03.4 D-16 / D-17). Both are KEYWORD-ONLY with `None`
    defaults, so every existing call site of `run(descriptor, config)` is
    unchanged and a caller that wants neither gets today's behaviour exactly:
    the incumbent stderr bar, and no way to stop the run early.

    They are CALL ARGUMENTS, forwarded through `Acquisition.attach()`, and are
    never assigned onto `config` -- `AcquisitionConfig.to_dict()` is
    `asdict(self)` and lands on disk beside model checkpoints, where a
    `threading.Event` cannot be serialised and a live reporter object is not
    reproducible configuration. See `attach`'s docstring.

    Cancellation is a TOKEN and not the reporter's return value, so a reporter
    that only wants to log cannot halt a multi-hour backfill by forgetting to
    return the right value -- and a reporter that raises cannot end the run
    either (`Acquisition._emit` catches and logs).
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
    """Convert an already-acquired RAW tier into Zarr, IN-PROCESS (03.5 D-03).

    **The counterpart to `run()` above, and a SEPARATE function rather than a
    flag on it (03.4 D-14).** `run()` downloads to the raw parquet tier and
    stops; this reads that tier and writes the Zarr store. They are adjacent
    so the two-calls contract is visible while reading, and nothing here
    touches a vendor client, an endpoint, or a credential.

    **The conversion target comes from the CAPABILITY, never from a branch**
    (D-01). The lookup key is `(dataset_config.market, dataset_config.frequency,
    data_type)` -- exactly `Capability`'s own key, which is why D-02 reinstated
    `DatasetConfig.market`. There is no vendor literal in this body and no
    frequency literal either: a capability with no `dataset_cls` is refused
    BECAUSE it has none, so phase 03.3 lands tick conversion by filling two
    fields rather than by deleting an `if`.

    **It WRITES, period.** That is half the reason it exists: `from_raw_data()`
    returns `Self` and leaves writing to a separate `.save()`, while
    `from_raw_data_chunked()` writes inside its loop. A caller reaching the
    Dataset layer by hand has to know which. A caller here does not.

    **It does NOT run the RAM guard, and that is a recorded, accepted risk
    (D-11).** SUPERSEDED by phase 03.6 (SC-3). The original sentence read:
    "The guarantee lives at every call site -- the catalog's per-chunk RAM
    guard before `convert(...)`, as `ingest_us_equity.py` already does -- not
    at the one place that allocates." It named that guard by its method name;
    the name is not restated here because phase 03.6's SC-3 gate asserts the
    symbol appears nowhere in the tree. Phase 03.6 DELETED the guard, so
    there is no call-site guarantee left to rely on: the recorded risk is now
    UNMITIGATED rather than mitigated-at-the-call-site, and every caller --
    new integrator or in-repo shell alike -- goes straight to OOM on an
    over-sized window. `predicted_peak_bytes` survives, defaulting to `None`,
    because an out-of-repo caller with its own estimate may still pass one; it
    is echoed into the result so a report can put prediction beside outcome.
    Passing it buys no protection, and pretending otherwise here would retire a
    recorded risk silently instead of by decision.

    **Refuses in THREE distinguishable ways, all `ValueError`.** A caller who
    cannot tell "nobody serves that combination" from "nobody can convert it
    yet" has to read this function's source in order to act, so each arm names
    the descriptor, the requested tuple, and the one thing that would fix it.
    Every message is built from the descriptor's own data: there is no vendor
    literal and no frequency literal in this body, which is what makes the
    capability list -- not this function -- the place a new combination is
    added.

    **`reporter` and `cancel` are the console's two handles on a running
    conversion** (03.5 D-05), word for word the pair `run()` above carries and
    for the identical reason: 03.4 D-17's argument -- cancellation cannot be
    added by the console from outside, because the loop lives in quantlab --
    applies to the chunk loop too, and a chunked conversion of a full-market
    panel is the longest-running operation this project has. Both are
    KEYWORD-ONLY with `None` defaults, so every existing
    `convert(descriptor, dataset_config)` call site is unchanged and a caller
    that wants neither gets today's behaviour exactly.

    They are CALL ARGUMENTS, forwarded straight into
    `from_raw_data_chunked()`, and are never assigned onto `dataset_config` --
    `DatasetConfig.to_dict()` is `asdict(self)` and lands on disk beside model
    checkpoints, where a `threading.Event` cannot be serialised and a live
    reporter object is not reproducible configuration.

    Cancellation is a TOKEN and not the reporter's return value, so a reporter
    that only wants to log cannot halt a multi-hour conversion by forgetting to
    return the right value -- and a reporter that raises cannot end it either
    (`BaseDataset._emit_progress` catches and logs).

    The conversion-specific half: the token is observed at WINDOW BOUNDARIES,
    never mid-window, and `ChunkLedger` already supplies the
    completed-work-stays-resumable precondition that the acquisition side had
    to retrofit with atomic sidecars. So a cancelled conversion is not work
    thrown away -- every window it finished is still there, and the next
    `convert()` over the same config resumes at the first unwritten one and
    reports `resumed`.
    """
    matches = descriptor.capabilities_for(
        dataset_config.market, dataset_config.frequency, data_type
    )
    requested = (dataset_config.market, dataset_config.frequency, data_type)

    # (1) NO MATCH. Enumerate what the descriptor DOES serve in the same
    # message, the courtesy `DataSourceRegistry.get()` already extends for a
    # mistyped vendor token: a caller who got one element wrong learns which.
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

    # (2) AMBIGUOUS. Several rows matched and they disagree about what
    # materialises them. This is the case D-02 reinstated
    # `DatasetConfig.market` for, one axis over: refusing is the only honest
    # answer, because picking `matches[0]` would let capability ORDER -- an
    # authoring detail nobody reading the call site can see -- decide which
    # Dataset subclass writes the store.
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

    # (3) NO CONVERSION TARGET (SC-7). The refusal IS the absent field: the
    # lookup above MATCHED, the vendor genuinely serves this capability, and
    # what is missing is a Dataset subclass that can express its axis. Voiced
    # after the parser-level refusal one of the ingest shells already carries
    # for this same capability, which stays where it is -- that one fires
    # before any work is done, and this is the second, programmatic layer for
    # callers that never touch argparse. Naming that shell (or the frequency
    # token) HERE would put the very literal in this body that the capability
    # lookup exists to keep out.
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
# Vendor module imports -- LAST, and here rather than in `__init__.py`.
#
# A decorator-populated registry is only as complete as the set of modules that
# have been imported, so importing THIS module must import every vendor module
# (D-07): a cold `import quantlab.registry` in a fresh process must
# enumerate every source.
#
# They do NOT go in `quantlab/acquisition/__init__.py`, which stays 0 bytes.
# A non-empty package `__init__` runs on EVERY `import
# quantlab.acquisition.<anything>` -- including `quantlab.acquisition.inspector`,
# the read surface whose entire structural guarantee is that no acquisition
# client is reachable from it, whatever the call order. That property is what
# makes a read refuse-before-any-client-exists rather than
# refuse-if-called-in-the-right-order. Worse, it would erode SILENTLY:
# `tests/test_source_inspector.py`'s structural arm is an `ast` scan of
# `inspector.py`'s OWN source plus a `vars(inspector_module)` sweep, and
# neither can see a transitive import dragged in by a package `__init__`.
#
# The same argument used to cover `quantlab.acquisition.universe` and the
# volume guard. That half has TRANSFERRED UP: the universe module is now
# `quantlab/universe.py`, a top-level sibling, so `import quantlab.universe`
# runs ONE package `__init__` (`quantlab/`, also 0 bytes) where it used to run
# two -- the guarantee got strictly easier to hold, and its prose now lives on
# the module that HAS it. `tests/test_volume_guard.py` asserts the emptiness
# rather than trusting a comment.
#
# Bottom placement is also what lets each descriptor be defined beside the
# class it describes: `tiingo.py`/`alpaca.py` import THIS module for the
# decorator, so this module cannot import them at its top.
#
# The MODULE-OBJECT form (`from quantlab.acquisition import tiingo`), never
# `from quantlab.acquisition.tiingo import TiingoAcquisition`: when a caller
# imports `tiingo` first, this module runs while `tiingo` is only partially
# initialised, and binding the module object is safe where reading an attribute
# off it would raise. Order between the three lines is irrelevant -- `all()`
# sorts.
#
# That form is now carrying MORE weight than it used to. Since 260922-lu2 this
# module is `quantlab/registry.py`, a top-level sibling, so the cycle it closes
# CROSSES A PACKAGE BOUNDARY where it used to stay inside one:
#
#     quantlab.registry -> quantlab.acquisition.wrds -> quantlab.registry
#
# The mechanics are unchanged -- `from package import submodule` is defined to
# work during partial initialisation, and the module object is safe to bind
# where an attribute read would raise -- but the blast radius is wider, because
# the partially-initialised module now lives outside the package whose
# `__init__` the importer just ran. The five import orders that could expose it
# are pinned by
# `tests/test_wrds_vendor_seam.py::test_enumeration_survives_any_import_order`:
# `quantlab.acquisition.wrds.taq`, `quantlab.acquisition.wrds`,
# `quantlab.registry`, `quantlab.acquisition.alpaca`, `quantlab.universe` --
# each in a fresh interpreter, each required to enumerate
# `['alpaca', 'tiingo', 'wrds']`.
#
# The argument depends on TWO `__init__.py` files being 0 bytes: `quantlab/`
# (which now runs on every `import quantlab.registry`) and
# `quantlab/acquisition/` (which runs on every vendor import below). Both are
# asserted by tests -- `tests/test_volume_guard.py` and
# `tests/test_source_inspector.py` -- rather than by this comment.
#
# `wrds` is a NEUTRAL module rather than a provider (03.10 D-12): it holds the
# one `wrds` descriptor and imports the WRDS PROVIDER modules itself, because
# one WRDS account serves several products through several acquisition classes.
# A descriptor placed inside one provider would have to name the other
# provider's classes, and that edge -- plus this import -- would cycle.
# ---------------------------------------------------------------------------
from quantlab.acquisition import alpaca as _alpaca  # noqa: E402,F401
from quantlab.acquisition import tiingo as _tiingo  # noqa: E402,F401
from quantlab.acquisition import wrds as _wrds  # noqa: E402,F401
