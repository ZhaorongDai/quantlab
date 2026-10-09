"""Check a day of one-minute Trade bars against Massive's own minute aggregates.

Massive's minute aggregates are kept only to check quantlab's Trade bars
against (ADR 0030). ``check_minute_bars`` compares one session's bars with
the vendor's bars of the same day, after two alignments:

- **Labels.** Massive labels a bar at its start (``window_start``); Trade
  bars are labelled at their end, so a vendor bar starting at ``t`` is
  compared with the Trade bar labelled ``t + 1 minute``. Vendor bars whose
  shifted label is not a bar of the session (pre- and post-market) are
  counted as ``vendor_outside_session`` and not compared. The two still
  close bars on opposite sides: Massive's bar is ``[t, t + 1m)``, a Trade
  bar ``(t, t + 1m]``, so a trade stamped exactly on a minute (to the
  nanosecond) falls in neighbouring bars on the two sides.
- **Symbols.** A vendor ticker is put on the permaticker the conversion
  mapped it to that day; one the conversion did not keep (unmapped, the
  busier ticker of a pair won, or off the roster: all three are counted
  together) is counted as ``vendor_unmapped`` and not compared.

Massive emits a minute bar only when it holds a trade that sets a price,
so a Trade bar counts as present when it has prices (a ``close``); a bar of
volume-only trades (odd lots, mostly) is counted as ``ours_volume_only``
and not compared. Over the bars present on both sides, each of ``open``,
``high``, ``low``, ``close``, ``volume`` and ``n_trades`` (the vendor's
``transactions``) agrees when the two values are equal; a NaN price never
agrees with a number. The result counts the agreements, names the bar with
the largest difference of each variable, and samples the bars present on
one side only. Differences are reported, never raised: ``CHECKED`` says the
check ran, not that it passed, until tolerances are set.

Examples
--------
>>> check = check_minute_bars(bars, vendor, mapping, labels)
>>> check["status"], check["both"] == check["agree"]["close"]
('checked', True)
"""

from __future__ import annotations

import polars as pl

#: The variables compared, each with its vendor column.
COMPARED = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "n_trades": "transactions",
}

#: The ``status`` of a day checked against the vendor's bars.
CHECKED = "checked"

#: The ``status`` of a one-minute day whose minute aggregates are not in the raw tier.
NO_MINUTE_AGGREGATES = "no minute aggregates"

#: The ``status`` of a day converted at another bar interval, never checked.
NOT_ONE_MINUTE = "not one-minute bars"

#: Bars sampled from each one-sided set.
SAMPLE = 5

_MINUTE_NS = 60 * 1_000_000_000


def check_minute_bars(
    bars: pl.DataFrame, vendor: pl.DataFrame, mapping: dict[str, int], labels: pl.Series
) -> dict:
    """Compare one session's one-minute Trade bars with Massive's minute aggregates.

    Parameters
    ----------
    bars : pl.DataFrame
        The Trade bars of the day: ``symbol`` (the permaticker, Int64),
        ``timestamp`` (the end label) and the ``COMPARED`` variables.
    vendor : pl.DataFrame
        The day's minute aggregates as ``raw.read_aggregates`` returns them.
    mapping : dict[str, int]
        The tickers the conversion kept that day, each with its permaticker.
    labels : pl.Series
        The session's bar labels (``Datetime("ns")``).

    Returns
    -------
    dict
        JSON-ready: ``status`` (``"checked"``); ``ours_bars`` (with
        prices), ``ours_volume_only``, ``vendor_bars``, ``both``,
        ``ours_only``, ``vendor_only``;
        ``vendor_outside_session``, ``vendor_unmapped``; ``agree`` (bars
        agreeing, per variable); ``worst`` (per disagreeing variable, the
        ``ticker``, ``permaticker``, ``bar``, ``ours`` and ``vendor`` values
        of the largest difference, a NaN price shown as ``None``); and up to
        ``SAMPLE`` bars of each one-sided set as ``ours_only_sample`` and
        ``vendor_only_sample``.

    Examples
    --------
    With the bars and vendor file of 2016-11-25 (a half day)::

        check = check_minute_bars(bars, read_aggregates(path), mapping, labels)
        check["both"], check["agree"]["close"]  # (616733, 616723)
    """
    tickers = pl.DataFrame(
        {"ticker": list(mapping), "permaticker": list(mapping.values())},
        schema={"ticker": pl.String, "permaticker": pl.Int64},
    )
    grid = pl.DataFrame({"timestamp": labels.cast(pl.Datetime("ns"))})
    theirs = vendor.with_columns(
        (pl.col("window_start") + _MINUTE_NS).cast(pl.Datetime("ns")).alias("timestamp")
    )
    in_session = theirs.join(grid, on="timestamp", how="semi")
    mapped = in_session.join(tickers, on="ticker", how="inner")
    theirs = mapped.select(
        "ticker",
        pl.col("permaticker").alias("symbol"),
        "timestamp",
        *(pl.col(column).cast(pl.Float64).alias(f"{name}_vendor") for name, column in COMPARED.items()),
    )
    ours = (
        bars.filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
        .select(
            pl.col("symbol").cast(pl.Int64),
            pl.col("timestamp").cast(pl.Datetime("ns")),
            *(pl.col(name).cast(pl.Float64).alias(f"{name}_ours") for name in COMPARED),
        )
        .join(tickers.rename({"permaticker": "symbol"}), on="symbol", how="left")
    )
    keys = ["symbol", "timestamp"]
    both = ours.drop("ticker").join(theirs, on=keys, how="inner")
    ours_only = ours.join(theirs, on=keys, how="anti")
    vendor_only = theirs.join(ours, on=keys, how="anti")

    agree: dict[str, int] = {}
    worst: dict[str, dict] = {}
    for name in COMPARED:
        mine, their = pl.col(f"{name}_ours"), pl.col(f"{name}_vendor")
        equal = (mine == their).fill_null(False) & ~mine.is_nan() & ~their.is_nan()
        agree[name] = int(both.select(equal.sum()).item())
        differ = both.filter(~equal)
        if differ.height:
            # A NaN on one side is the worst difference there is.
            row = (
                differ.with_columns((mine - their).abs().fill_nan(float("inf")).alias("_diff"))
                .sort("_diff", descending=True)
                .row(0, named=True)
            )
            worst[name] = {
                "ticker": row["ticker"],
                "permaticker": int(row["symbol"]),
                "bar": row["timestamp"].isoformat(),
                "ours": _json_float(row[f"{name}_ours"]),
                "vendor": _json_float(row[f"{name}_vendor"]),
            }
    volume_only = bars.filter((pl.col("n_trades") > 0) & (pl.col("close").is_null() | pl.col("close").is_nan()))
    return {
        "status": CHECKED,
        "ours_bars": ours.height,
        "ours_volume_only": volume_only.height,
        "vendor_bars": theirs.height,
        "both": both.height,
        "ours_only": ours_only.height,
        "vendor_only": vendor_only.height,
        "vendor_outside_session": vendor.height - in_session.height,
        "vendor_unmapped": in_session.height - mapped.height,
        "agree": agree,
        "worst": worst,
        "ours_only_sample": _sample(ours_only),
        "vendor_only_sample": _sample(vendor_only),
    }


def _json_float(value: float) -> float | None:
    """Return ``value``, or ``None`` for NaN, which JSON cannot hold."""
    return None if value != value else float(value)


def _sample(frame: pl.DataFrame) -> list[dict]:
    """Return the first ``SAMPLE`` bars of ``frame`` as ticker, permaticker and bar."""
    return [
        {"ticker": row["ticker"], "permaticker": int(row["symbol"]), "bar": row["timestamp"].isoformat()}
        for row in frame.sort("timestamp", "symbol").head(SAMPLE).iter_rows(named=True)
    ]
