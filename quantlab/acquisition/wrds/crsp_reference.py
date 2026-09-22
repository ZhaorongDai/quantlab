"""The CRSP reference PULL: six whole tables into `_reference/` (phase 03.10).

`quantlab/dataset/crsp_reference.py` DECLARES the six tables -- their schemas,
their server column order, their types -- and reads them back without a
credential. This module is the other half: it puts them on disk.

**Why this is not an `Acquisition`** (RESEARCH Pattern 3). An `Acquisition`
downloads a per-symbol time series: batches of symbols, pages inside a window,
a watermark per symbol, a resumable page ledger. These tables have no symbol
axis and no window -- `crsp_a_stock.stkdelists` is 29,833 rows TOTAL and is
either present or not. Wrapping one `COPY` per table in that machinery would
be ceremony with nothing to resume, so the pull is a plain class with one
public method.

**Where the files live, and why it matters.** `reference_dir` comes from
`WrdsCrspDailyAcquisition.reference_dir_for(config)`, which is
`.../wrds_crsp/_reference`, a SIBLING of the raw root `.../wrds_crsp/wrds`.
Never inside it: `StockDataset._scan_raw` globs every file below the raw root
and polars refuses a tree whose file extensions disagree, so one `.parquet` --
or the `manifest.json` -- dropped in there breaks every conversion, including
the ones that never asked for the reference tier. Nothing in this module
writes anywhere but `reference_dir`.

**`stkdelists` is EVENT DATA ONLY** (D-10/D-19). It is pulled and stored raw,
and nothing here derives, merges or nets anything out of it. CRSP's CIZ daily
table already puts the delisting return on its own daily row
(`dlydelflg='Y'`), so chaining `delret` on top of `dlyret` would apply the
delisting loss twice -- a survivorship error in the flattering direction,
which is the kind that does not look wrong in a backtest.

**Safety, in the order the code performs it.** Every table is COUNTED before
it is copied and refused above `MAX_REFERENCE_ROWS` (T-03.10-14: the pull is
whole-table, so the count is the only thing between a renamed table and an
unbounded download); its COPY must return exactly that many rows; each file
is written to a temp file in the destination directory and `os.replace`d into
place; and the manifest is written LAST, so an interrupted pull never
publishes a manifest naming a table it did not finish (T-03.10-12). Every
value reaching SQL travels as `sql.Literal` and every identifier as
`sql.Identifier` (T-03.10-13), which is also what quotes `comp.idxcst_his`'s
reserved `from` column for free (Pitfall 8).
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
from quantlab.dataset.crsp_reference import (
    MANIFEST_NAME,
    REFERENCE_TABLES_BY_NAME,
    ReferenceTableSpec,
)
from quantlab.utils.atomic import write_json_atomically


class CrspReferenceTables:
    """Pull the CRSP / Compustat / CCM reference tables into one directory.

    Holds the ONE shared `WrdsSession` it was handed (D-20: a second
    connection can push a second Duo prompt) and the destination directory.
    Nothing else: the vintage, the universes and the refresh policy are
    arguments to `pull`, so one instance can answer for several vintages and a
    caller never has to rebuild it to change its mind.
    """

    #: Refuse any single reference table larger than this. Generous against
    #: the live sizes (the biggest, `stkdistributions`, is ~1.1M rows) and
    #: still small enough that a table which became something else -- a view
    #: over the daily panel, say -- stops the run instead of streaming.
    MAX_REFERENCE_ROWS = 5_000_000

    #: Always pulled: the symbology source and the two event tables.
    STOCK_TABLES = ("stksecurityinfohist", "stkdelists", "stkdistributions")

    #: CRSP's own S&P 500 membership, by PERMNO (D-05).
    SP500_TABLES = ("dsp500list_v2",)

    #: Compustat index membership and the CRSP/Compustat link (D-14), in this
    #: ORDER: the CCM pull asks only for the gvkeys `idxcst_his` returned, so
    #: it cannot run first.
    NASDAQ100_TABLES = ("idxcst_his", "ccmxpf_lnkhist")

    #: `comp.idx_index`'s gvkeyx for "Nasdaq 100" (live check L8/NDX-QQQ).
    NDX_GVKEYX = "000208"

    def __init__(self, session, reference_dir) -> None:
        self.session = session
        self.reference_dir = Path(reference_dir)

    # -- paths and the manifest ---------------------------------------------

    def path_for(self, name: str) -> Path:
        return self.reference_dir / f"{name}.parquet"

    @property
    def manifest_path(self) -> Path:
        return self.reference_dir / MANIFEST_NAME

    def read_manifest(self) -> dict | None:
        """The manifest on disk, or `None` if there is none.

        A manifest that cannot be parsed is treated as absent rather than
        raised on: the only thing it can make the caller do is pull again,
        which is the safe direction, and refusing to pull because a sidecar
        is corrupt would leave the tier unrepairable except by hand.
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
        """The table names this pull covers, in pull order."""
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

        Returns the manifest (the dict written to `manifest.json`).

        The vintage SKIP comes first, before entitlement and before any
        count: a complete tier for this `product_end` is already the answer,
        and every probe is a round trip on the single shared session. A
        different `product_end` is a different tier -- CRSP revises history at
        the annual refresh -- so the table map starts empty and every
        requested table is pulled again rather than being mixed with files
        from the previous vintage.
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

        # Entitlement over exactly the REQUESTED tables' schemas, so an
        # S&P-only pull never asks whether this account can read `comp` --
        # a question whose answer would be "no" for most CRSP subscriptions
        # and which nothing in that pull needs (D-03/D-14).
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
        # LAST, and atomically: a manifest published before the final COPY
        # would name a table that is not on disk, and every reader trusts it
        # for exactly that.
        write_json_atomically(self.manifest_path, manifest, indent=2, sort_keys=True)
        return manifest

    # -- one table ----------------------------------------------------------

    def _pull_table(self, spec: ReferenceTableSpec, where) -> pl.DataFrame:
        """Count, refuse-or-copy, check the shape, cast. No file I/O."""
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
        """`(sql WHERE or None, human-readable description or None)`.

        Four of the six tables are pulled WHOLE and have no WHERE at all. The
        two Compustat-side ones are predicated, and both predicates are built
        with `psycopg2.sql` only: the index id is a `sql.Literal`, the gvkeys
        are a `sql.Literal` LIST (server-returned values going back into a
        query, T-03.10-13), and every column name is a `sql.Identifier` --
        which is also what quotes `idxcst_his`'s reserved `from` (Pitfall 8).

        The description is what the manifest records. Deliberately NOT the
        rendered statement: rendering `sql.Composed` needs a live connection's
        quoting context, and the manifest is read by `CrspReference` on
        machines with no driver at all.
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
        """Write `frame` to `path` via a temp file in the SAME directory.

        Same idiom, and the same reason, as `utils/atomic.py`: the temp file
        must share a filesystem with the destination for `os.replace` to be an
        atomic rename rather than an interruptible copy. On any failure the
        temp file is removed, so a crashed pull leaves neither a half-written
        table nor a `.tmp` beside the good ones.
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

    # -- Nasdaq-100 (plan 03.10-04 Task 2) ----------------------------------

    def _gvkeys_of(self, frame: pl.DataFrame) -> list[str]:
        """The membership's distinct gvkeys, sorted; empty is a FAILURE.

        An empty set would make the CCM predicate `= ANY(ARRAY[])`, which
        matches nothing, and the pull would quietly write two empty tables.
        That surfaces much later as a Nasdaq-100 universe with no members --
        indistinguishable from a roster that legitimately selected nothing,
        and by then the vintage manifest says the tier is complete.
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
        """The gvkeys of an `idxcst_his` already pulled in a previous run.

        Reached only in the incremental case -- a vintage whose membership
        table is on disk but whose CCM table is not, e.g. a pull that was
        interrupted between the two, or one that gained `include_nasdaq100`
        after the fact. Re-reading the parquet is cheaper and more honest than
        re-COPYing a table this vintage already has.
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
    """The specs' schemas, de-duplicated, in first-seen order."""
    seen: list[str] = []
    for spec in specs:
        if spec.schema not in seen:
            seen.append(spec.schema)
    return tuple(seen)


def _as_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])
