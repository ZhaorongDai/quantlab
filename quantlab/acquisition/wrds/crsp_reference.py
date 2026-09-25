"""Pull of the CRSP, Compustat and CCM reference tables into ``_reference/``.

``quantlab.dataset.crsp.reference`` declares the six reference tables (their
schemas, server column order and types) and reads them back without a
credential. ``CrspReferenceTables`` in this module is the other half: it
copies each table whole through the shared WRDS session and writes one
parquet file per table plus a ``manifest.json`` recording the CRSP vintage.

This is deliberately not an ``Acquisition``. That base class downloads a
per-symbol time series with batches, pages, watermarks and a resumable
ledger; these tables have no symbol axis and no window, and each one is
either present or not. The destination is ``.../wrds_crsp/_reference``, a
sibling of the raw root, never inside it, because the dataset's raw scan
walks every file below the raw root and a stray parquet or manifest there
would break every conversion.

``stkdelists`` is stored as event data only: the CIZ daily table already puts
the delisting return on its own daily row, so nothing here chains ``delret``
onto ``dlyret``, which would apply the delisting loss twice.
"""

from __future__ import annotations

import io
import os
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
from loguru import logger
from psycopg2 import sql

from quantlab.acquisition.wrds.crsp import CrspQueries
from quantlab.dataset.crsp.reference import (
    MANIFEST_NAME,
    REFERENCE_TABLES_BY_NAME,
    ReferenceTableSpec,
)
from quantlab.utils.atomic import write_json_atomically


class CrspReferenceTables:
    """Pull the CRSP, Compustat and CCM reference tables into one directory.

    An instance holds only the shared ``WrdsSession`` and the destination
    directory; the vintage, the universes and the refresh policy are
    arguments to ``pull``, so one instance can serve several vintages.

    Safety, in the order the code performs it: every table is counted before
    it is copied and refused above ``MAX_REFERENCE_ROWS`` (the pull is
    whole-table, so the count is the only thing between a renamed table and
    an unbounded download); the COPY must return exactly that many rows;
    each file is written to a temporary file in the destination directory
    and renamed into place; and the manifest is written last, so an
    interrupted pull never publishes a manifest naming a table it did not
    finish. Every value reaching SQL travels as ``sql.Literal`` and every
    identifier as ``sql.Identifier``.

    Examples
    --------
    Needs a live ``WrdsSession``; the reference directory comes from
    ``WrdsCrspDailyAcquisition.reference_dir_for(cfg)``.

    >>> tables = CrspReferenceTables(WrdsSession.shared(), reference_dir)
    >>> manifest = tables.pull(product_end="2025-12-31", include_sp500=True)
    >>> sorted(manifest)
    ['product_end', 'pulled_at', 'tables']
    >>> sorted(manifest["tables"])
    ['dsp500list_v2', 'stkdelists', 'stkdistributions', 'stksecurityinfohist']
    """

    #: Refuse any single reference table larger than this. Generous against
    #: the live sizes (the biggest, ``stkdistributions``, is about 1.1M rows)
    #: and still small enough that a table which became something else, such
    #: as a view over the daily panel, stops the run instead of streaming.
    MAX_REFERENCE_ROWS = 5_000_000

    #: Always pulled: the symbology source and the two event tables.
    STOCK_TABLES = ("stksecurityinfohist", "stkdelists", "stkdistributions")

    #: CRSP's own S&P 500 membership, by PERMNO.
    SP500_TABLES = ("dsp500list_v2",)

    #: Compustat index membership and the CRSP/Compustat link, in this order:
    #: the CCM pull asks only for the gvkeys ``idxcst_his`` returned, so it
    #: cannot run first.
    NASDAQ100_TABLES = ("idxcst_his", "ccmxpf_lnkhist")

    #: ``comp.idx_index``'s gvkeyx for the Nasdaq 100.
    NDX_GVKEYX = "000208"

    def __init__(self, session, reference_dir) -> None:
        """Bind the shared session and the destination directory."""
        self.session = session
        self.reference_dir = Path(reference_dir)

    # -- paths and the manifest ---------------------------------------------

    def path_for(self, name: str) -> Path:
        """Return the parquet path for the reference table ``name``.

        Examples
        --------
        >>> tables.path_for("stkdelists").name
        stkdelists.parquet
        """
        return self.reference_dir / f"{name}.parquet"

    @property
    def manifest_path(self) -> Path:
        """Return the ``manifest.json`` path inside the reference directory.

        Examples
        --------
        >>> tables.manifest_path.name
        manifest.json
        """
        return self.reference_dir / MANIFEST_NAME

    def read_manifest(self) -> dict | None:
        """Return the manifest on disk, or ``None`` if there is none.

        A manifest that cannot be parsed is treated as absent rather than
        raised on: the only thing it can make the caller do is pull again,
        which is the safe direction, and refusing to pull because a sidecar
        is corrupt would leave the tier unrepairable except by hand.

        Examples
        --------
        >>> tables.read_manifest() is None
        True
        """
        import json

        path = self.manifest_path
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            logger.warning(
                f"{type(self).__name__}: {str(path)!r} is unreadable; treating "
                f"the reference tier as un-pulled."
            )
            return None
        return manifest if isinstance(manifest, dict) else None

    def requested_tables(
        self, *, include_sp500: bool, include_nasdaq100: bool
    ) -> tuple[str, ...]:
        """Return the table names a pull covers, in pull order.

        Examples
        --------
        >>> tables.requested_tables(include_sp500=True, include_nasdaq100=False)
        ('stksecurityinfohist', 'stkdelists', 'stkdistributions', 'dsp500list_v2')
        """
        names = list(self.STOCK_TABLES)
        if include_sp500:
            names.extend(self.SP500_TABLES)
        if include_nasdaq100:
            names.extend(self.NASDAQ100_TABLES)
        return tuple(names)

    # -- the pull -----------------------------------------------------------

    def pull(
        self,
        *,
        product_end,
        include_sp500: bool = True,
        include_nasdaq100: bool = False,
        refresh: bool = False,
    ) -> dict:
        """Pull the requested reference tables for one CRSP vintage.

        The vintage check comes first, before entitlement and before any
        count: if the manifest on disk already covers every requested table
        for this ``product_end``, nothing is queried and the existing
        manifest is returned. A different ``product_end`` is a different
        tier, because CRSP revises history at the annual refresh, so every
        requested table is pulled again rather than mixed with files from the
        previous vintage. Entitlement is checked over exactly the requested
        tables' schemas, so an S&P-only pull never asks whether the account
        can read ``comp``.

        Parameters
        ----------
        product_end
            The CRSP product's last day (ISO string or date),
            as probed by ``CrspQueries.product_end``.
        include_sp500 : bool
            Also pull ``dsp500list_v2``.
        include_nasdaq100 : bool
            Also pull ``idxcst_his`` and
            ``ccmxpf_lnkhist``, which need the Compustat and CCM
            subscriptions.
        refresh : bool
            Re-pull tables that are already on disk for this
            vintage.

        Returns
        -------
        dict
            The manifest written to ``manifest.json``: ``product_end``,
            ``pulled_at`` and a ``tables`` map of ``{schema, table, rows,
            where}`` entries.

        Examples
        --------
        >>> manifest = tables.pull(product_end="2025-12-31")
        >>> manifest["tables"]["stkdelists"]["schema"]
        crsp_a_stock
        >>> tables.pull(product_end="2025-12-31") == manifest
        True
        """
        product_end = _as_date(product_end)
        requested = self.requested_tables(
            include_sp500=include_sp500, include_nasdaq100=include_nasdaq100
        )
        specs = [REFERENCE_TABLES_BY_NAME[name] for name in requested]

        existing = self.read_manifest()
        same_vintage = (
            existing is not None
            and str(existing.get("product_end")) == product_end.isoformat()
        )
        entries: dict[str, dict] = (
            dict(existing.get("tables") or {}) if same_vintage and existing else {}
        )

        if refresh:
            to_pull = list(specs)
        else:
            to_pull = [
                spec
                for spec in specs
                if spec.name not in entries or not self.path_for(spec.name).exists()
            ]
        if not to_pull:
            logger.debug(
                f"{type(self).__name__}: {str(self.reference_dir)!r} already "
                f"holds every requested table for product end "
                f"{product_end.isoformat()}; no query issued."
            )
            return existing  # type: ignore[return-value]

        # Entitlement over exactly the requested tables' schemas, so an
        # S&P-only pull never asks whether this account can read `comp`.
        CrspQueries.assert_entitled(self.session, _schemas_of(specs))

        gvkeys: list[str] | None = None
        for spec in to_pull:
            if spec.name == "ccmxpf_lnkhist" and gvkeys is None:
                gvkeys = self._gvkeys_on_disk()
            where, described = self._where_for(spec, gvkeys)
            frame = self._pull_table(spec, where)
            self._write_parquet_atomically(frame, self.path_for(spec.name))
            entries[spec.name] = {
                "schema": spec.schema,
                "table": spec.table,
                "rows": frame.height,
                "where": described,
            }
            if spec.name == "idxcst_his":
                gvkeys = self._gvkeys_of(frame)

        manifest = {
            "product_end": product_end.isoformat(),
            "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": entries,
        }
        # Last, and atomically: a manifest published before the final COPY
        # would name a table that is not on disk.
        write_json_atomically(self.manifest_path, manifest, indent=2, sort_keys=True)
        return manifest

    # -- one table ----------------------------------------------------------

    def _pull_table(self, spec: ReferenceTableSpec, where) -> pl.DataFrame:
        """Count, refuse or copy, check the shape, and cast; no file I/O."""
        rows = CrspQueries.count(self.session, spec.schema, spec.table, where)
        if rows > self.MAX_REFERENCE_ROWS:
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} has {rows} "
                f"row(s), over the {self.MAX_REFERENCE_ROWS}-row reference "
                f"ceiling (MAX_REFERENCE_ROWS). The reference tier is pulled "
                f"WHOLE, so a table this size is refused BEFORE its COPY rather "
                f"than streamed; nothing was downloaded for it."
            )

        raw = CrspQueries.copy(
            self.session, spec.schema, spec.table, spec.columns, where
        )
        frame = pl.read_csv(io.BytesIO(raw), infer_schema=False)
        if tuple(frame.columns) != tuple(spec.columns):
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} COPY "
                f"returned columns {frame.columns}, not the requested "
                f"{list(spec.columns)}. The table layout no longer matches the "
                f"one this phase was verified against."
            )
        if frame.height != rows:
            raise ValueError(
                f"{type(self).__name__}: {spec.schema}.{spec.table} COPY "
                f"returned {frame.height} row(s) but count(*) with the same "
                f"WHERE reported {rows}; the table is incomplete and is not "
                f"written, so the next pull re-fetches it."
            )
        return spec.cast(frame)

    def _where_for(self, spec: ReferenceTableSpec, gvkeys):
        """Return ``(sql WHERE or None, description or None)`` for one table.

        Four of the six tables are pulled whole and have no WHERE. The two
        Compustat-side tables are predicated, with the index id as a
        ``sql.Literal``, the gvkeys as a ``sql.Literal`` list and every
        column name as a ``sql.Identifier``, which also quotes
        ``idxcst_his``'s reserved ``from`` column. The description is what
        the manifest records; it is deliberately not the rendered statement,
        because rendering a composable needs a live connection and the
        manifest is read on machines with no driver at all.
        """
        if spec.name == "idxcst_his":
            where = sql.SQL("{column} = {value}").format(
                column=sql.Identifier("gvkeyx"),
                value=sql.Literal(self.NDX_GVKEYX),
            )
            return where, f"gvkeyx = '{self.NDX_GVKEYX}'"
        if spec.name == "ccmxpf_lnkhist":
            values = list(gvkeys or ())
            where = sql.SQL("{column} = ANY({values})").format(
                column=sql.Identifier("gvkey"), values=sql.Literal(values)
            )
            return where, (
                f"gvkey in the {len(values)} gvkey(s) comp.idxcst_his returned "
                f"for gvkeyx '{self.NDX_GVKEYX}'"
            )
        return None, None

    def _write_parquet_atomically(self, frame: pl.DataFrame, path: Path) -> None:
        """Write ``frame`` to ``path`` through a temp file in the same directory.

        The temp file must share a filesystem with the destination for
        ``os.replace`` to be an atomic rename rather than an interruptible
        copy. On any failure the temp file is removed, so a crashed pull
        leaves neither a half-written table nor a ``.tmp`` beside the good
        ones.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=str(path.parent),
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        )
        handle.close()
        try:
            frame.write_parquet(handle.name)
            os.replace(handle.name, path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    # -- Nasdaq-100 ---------------------------------------------------------

    def _gvkeys_of(self, frame: pl.DataFrame) -> list[str]:
        """Return the membership's distinct gvkeys, sorted; empty is an error.

        An empty set would make the CCM predicate ``= ANY(ARRAY[])``, which
        matches nothing, and the pull would quietly write two empty tables.
        That would surface much later as a Nasdaq-100 universe with no
        members, indistinguishable from a roster that legitimately selected
        nothing, with the manifest saying the tier is complete.
        """
        gvkeys = sorted(
            {
                str(value)
                for value in frame.get_column("gvkey").drop_nulls().to_list()
            }
        )
        if not gvkeys:
            raise ValueError(
                f"{type(self).__name__}: comp.idxcst_his returned no rows for "
                f"gvkeyx {self.NDX_GVKEYX!r} (Nasdaq 100), so there is no "
                f"membership to link. Refusing to query "
                f"crsp_a_ccm.ccmxpf_lnkhist for an empty gvkey set and "
                f"refusing to publish a manifest for an empty universe; check "
                f"the Compustat index id before re-running."
            )
        return gvkeys

    def _gvkeys_on_disk(self) -> list[str]:
        """Return the gvkeys of an ``idxcst_his`` pulled in a previous run.

        Reached only in the incremental case: a vintage whose membership
        table is on disk but whose CCM table is not, for example a pull that
        was interrupted between the two or one that gained
        ``include_nasdaq100`` after the fact. Re-reading the parquet is
        cheaper than re-copying a table this vintage already has.
        """
        path = self.path_for("idxcst_his")
        if not path.exists():
            raise ValueError(
                f"{type(self).__name__}: crsp_a_ccm.ccmxpf_lnkhist is pulled "
                f"for the gvkeys comp.idxcst_his returned, but no "
                f"{path.name} is on disk under {str(self.reference_dir)!r}. "
                f"Pull the Nasdaq-100 membership first (or pass refresh=True)."
            )
        return self._gvkeys_of(pl.read_parquet(path))


def _schemas_of(specs) -> tuple[str, ...]:
    """Return the specs' schemas, de-duplicated, in first-seen order."""
    seen: list[str] = []
    for spec in specs:
        if spec.schema not in seen:
            seen.append(spec.schema)
    return tuple(seen)


def _as_date(value) -> date:
    """Coerce a ``datetime``, ``date`` or ISO string to a ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
