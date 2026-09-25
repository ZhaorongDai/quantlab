# Universes

A *universe* is the set of symbols a strategy is allowed to consider on a
given day. Choosing it carelessly is one of the easiest ways to produce a
backtest that looks excellent and means nothing. This page explains
survivorship bias, shows how quantlab records point-in-time index membership
and exchange listings, how to turn membership into a mask over a price
panel, how the price and liquidity filter restricts factors and labels to
tradeable symbols, and what happens in a backtest when a held symbol leaves
the universe. Read [Datasets](datasets.md) first; the snippets here come from
the same runnable example,
[`examples/build_panel.py`](../../examples/build_panel.py).

## Survivorship bias

Suppose you backtest a strategy on "the S&P 500" using today's list of
members. Every company on that list survived to today, and many are there
precisely because they grew. Companies that went bankrupt, were acquired
cheaply or shrank out of the index are missing. A backtest over 2010 to 2020
on today's members therefore buys stocks with hindsight knowledge that they
will do well. This is *survivorship bias*, and it can inflate backtested
returns by a wide margin while looking entirely plausible.

Two things remove it. First, the price data must include delisted
securities, so that a stock that went to zero is in the panel until the day
it stopped trading. Second, the universe on each date must be the set of
symbols that were actually eligible on that date, not the set known today.
Information that was available on a date is called *point-in-time*; a
point-in-time universe changes as index additions, removals, listings and
delistings happen.

## Rosters and memberships

quantlab keeps two kinds of point-in-time universe information:

- An exchange *roster* lists every common stock ever listed on a set of
  exchanges, with its listing and delisting dates.
- An index *membership* lists the periods during which each symbol was a
  member of an index.

Both are stored as intervals: one row per symbol and period, with a
`start_date` and an `end_date` (empty while the interval is still open).
Intervals are closed at both ends, so a symbol removed on date D is still a
member on D and not on D + 1.

### The universe catalog

`quantlab.universe.UniverseCatalog` merges four categories into one Parquet
table with columns `symbol`, `category`, `start_date`, `end_date` and
`end_date_is_inferred`:

| Category | Contents | Source | Coverage starts |
|---|---|---|---|
| `nasdaq_all` | every NASDAQ-listed USD common stock, delisted included | Tiingo ticker directory | no boundary |
| `us_all` | the same across NYSE, NASDAQ and AMEX (about 15,000 tickers) | Tiingo ticker directory | no boundary |
| `sp500_constituent` | S&P 500 membership | Wikipedia change log | 1976-07-01 |
| `nasdaq100_constituent` | Nasdaq-100 membership | Wikipedia change log | 2007-02-01 |

The index histories are reconstructed by replaying Wikipedia's change log
backwards from a current-constituents snapshot. That removes most of the
survivorship bias but is not an exact official history: early years are
incomplete, and where the log never recorded a removal, the end date is
inferred and `end_date_is_inferred` is `True`. `nasdaq100_constituent` is
about 100 names at a time and has nothing to do with `nasdaq_all`, which has
no index concept; they only share a word.

Build or refresh the table with the script below. It downloads from Tiingo
and Wikipedia, needs no API key, and writes
`data/reference/universe.parquet` under the data root:

```bash
uv run python scripts/refresh_us_equity_universe.py
```

The script refuses to save a table rebuilt from a cached copy of a source
that failed to download, because a stale table looks exactly like a fresh
one; `--allow-stale` accepts it knowingly.

### Query the catalog

Two queries answer two different questions. `get_symbols_as_of(category,
date)` returns the symbols whose interval contains one date: this is what a
strategy may trade on that day, and a walk-forward backtest must ask it on
every rebalance date rather than once at the start. `get_symbols_in_range(
category, start, end)` returns every symbol whose interval overlaps the
window, including those that delisted inside it: this is the list to download
for a backfill.

The example writes a four-row catalog by hand, in which `DDD` delisted on
2024-01-31 and `EEE` listed on 2024-03-01:

```python
from quantlab.base.config import UniverseConfig
from quantlab.universe import UniverseCatalog

catalog = UniverseCatalog.load(
    UniverseConfig(output_path="universe.parquet", cache_dir="_cache")
)
catalog.get_symbols_as_of("us_all", "2024-03-15")
catalog.get_symbols_in_range("us_all", "2024-01-01", "2024-03-31")
```

```text
8. listed on 2024-03-15: ['AAA', 'BBB', 'EEE']
   listed at any time in Q1 2024: ['AAA', 'BBB', 'DDD', 'EEE']
```

Downloading only the first list for a Q1 backtest would silently drop `DDD`,
which is exactly survivorship bias. Results are sorted and de-duplicated.

The queries refuse rather than guess. An unknown category or a non-ISO date
raises `ValueError`, and so does a date before an index category's coverage
start, since membership before that date cannot be answered:

```text
ValueError: Cannot answer sp500_constituent membership before 1976-07-01 -- as_of_date='1970-01-01' precedes it. The Wikipedia-sourced change log is left-censored at that date and this query cannot be answered correctly, rather than silently defaulting to an incomplete/wrong answer.
```

The download scripts resolve their symbol lists from this table; see
[Data sources](data-sources.md).

## Membership panels

The catalog answers questions about symbols. To use membership inside the
pipeline, turn it into a panel: a dataset whose single boolean variable
`is_member` is `True` where a symbol belonged to the index on that day. These
datasets subclass `quantlab.base.constituent.IndexConstituentDataset` and are
configured with `ConstituentDatasetConfig`:

| Class (in `quantlab.dataset.constituent`) | Index | `symbol` axis | Source | Coverage starts |
|---|---|---|---|---|
| `SP500ConstituentDataset` | S&P 500 | ticker | Wikipedia change log | 1976-07-01 |
| `Nasdaq100ConstituentDataset` | Nasdaq-100 | ticker | Wikipedia change log | 2007-02-01 |
| `CrspSP500ConstituentDataset` | S&P 500 | PERMNO | CRSP reference tables | 1925-12-31 |
| `CompustatNasdaq100ConstituentDataset` | Nasdaq-100 | PERMNO | Compustat, linked to CRSP | 1995-01-01 |
| `CrspMarketConstituentDataset` | whole market (listed securities) | PERMNO | CRSP reference tables | 1925-12-31 |

Pick the class whose `symbol` axis matches your price panel. The
Wikipedia-based classes are keyed by ticker and pair with a Tiingo or Alpaca
`StockDataset`. The CRSP-based classes are keyed by PERMNO, CRSP's permanent
security identifier, and line up with a `CrspStockDataset` column for column;
they read the CRSP reference tables already on disk (`cache_dir`) and make no
WRDS connection. `CrspMarketConstituentDataset` answers "was this security
listed on this day", which lets you tell "not listed" apart from "listed, no
trade" in a wide panel. `scripts/ingest_wrds_crsp.py` can build the price
store and its membership panel together; see [WRDS](wrds.md).

A few properties of these panels matter when you use them:

- The `timestamp` axis is every calendar day, weekends and holidays included.
  Join it to a trading-day price panel by selecting onto the price
  timestamps; `UniverseMask` below does this for you.
- The `symbol` axis is every symbol that was ever a member, so a symbol whose
  membership ended before your window is an all-`False` column rather than a
  missing one.
- `start_date` is raised to the source's coverage start, with a warning if
  you asked for an earlier date.
- `as_of` pins the right edge. When it is unset, open intervals run to today,
  and the same config produces a different store tomorrow. Set it whenever
  the panel has to be reproducible.
- Ticker-keyed sources can disagree about spellings. Wikipedia records the
  ticker in use on each event date (`FB` before 2022), while Tiingo files a
  renamed company's whole history under its current ticker (`META`), and the
  two spell share classes differently (`BRK.B` against `BRK-B`). Such members
  find no column in the price panel. The PERMNO-keyed CRSP pair does not have
  this problem, which is a strong reason to prefer it for long histories.

### Build a universe panel and mask prices with it

A concrete membership dataset implements two hooks: the earliest date its
source can answer, and the interval table. The example uses a toy index in
which `AAA` is always a member, `BBB` leaves on 2024-02-15, `DDD` is a member
in January, and `ZZZ` was a member for three weeks but is absent from the
price data:

```python
import polars as pl
from quantlab.base.config import ConstituentDatasetConfig
from quantlab.base.constituent import IndexConstituentDataset
from quantlab.dataset._support.masking import UniverseMask

class DemoIndex(IndexConstituentDataset):
    def _pit_coverage_start(self) -> str:
        return "2024-01-01"

    def _build_intervals(self) -> pl.DataFrame:
        return pl.DataFrame(
            [
                ("AAA", "2024-01-01", None),
                ("BBB", "2024-01-01", "2024-02-15"),
                ("DDD", "2024-01-01", "2024-01-31"),
                ("ZZZ", "2024-01-01", "2024-01-19"),
            ],
            schema=["symbol", "start_date", "end_date"],
            orient="row",
        )

index_config = ConstituentDatasetConfig(
    zarr_file_path="demo_index.zarr",
    cache_dir="_cache",
    start_date="2024-01-01",
    end_date="2024-03-29",
    as_of="2024-03-29",
)
DemoIndex(index_config).from_raw_data().save()
membership = DemoIndex(index_config).read().get_xarray_dataset()

mask = UniverseMask(prices, membership)
masked = mask.apply()
```

`UniverseMask` intersects the two panels on both axes and sets every
non-member cell to NaN. Before masking it reports every in-window index
member that the price panel does not carry at all, in full, because each one
is a hole through which survivorship bias returns (delisted names are the
hardest to obtain). Price symbols outside the index are dropped silently.

```text
WARNING: UniverseMask: 1 of 4 in-window index member(s) are absent from the market panel entirely and are dropped by the alignment. Every dropped name is a survivorship-bias gap, so the complete list follows: ['ZZZ']
   report: {'in_window_members': 4, 'missing_count': 1, 'missing_symbols': ['ZZZ'], 'missing_labels': ['ZZZ']}
   masked close, first trading day of each month:
symbol        AAA    BBB   DDD
timestamp
2024-01-02  50.06  20.16  30.0
2024-02-01  48.44  21.04   NaN
2024-03-01  51.46    NaN   NaN
```

Treat a non-empty report as a finding to resolve (download the missing
names, or switch to a PERMNO-keyed pair), not as noise. In a pipeline with
both panels saved, `UniverseMask.from_datasets(price_dataset,
constituent_dataset)` reads both stores and, for a CRSP store, spells the
report with period-correct tickers.

## The price and liquidity filter

Index membership says whether a symbol belonged to a list. A second question
is whether it was tradeable at all: a cross-sectional ranking over the whole
US market is easily dominated by penny stocks, warrants and illiquid names
whose prices jump by orders of magnitude, and a model will happily buy them.
`quantlab.factor.universe_filter.UniverseFilteredFactor` answers that
question with a point-in-time rule. A symbol is in the universe at bar t when

- its raw `close` at t is at least `min_price` (default 5.0), and
- the mean of raw `close * volume` over the trailing `window` bars (default
  20) is at least `min_dollar_volume` (default 1,000,000).

Raw prices are used, never adjusted ones: adjusted history is scaled by later
splits and dividends, so an adjusted price cannot say what a stock cost at
the time. A window that is not yet full or contains a NaN counts as out of
the universe, and nothing after t affects the mask at t. Security types
(common stock against ADRs, funds, units) are not decided here; for CRSP data
that is the `security_filter` of the dataset config.

The filter is a wrapper around any KunQuant factor or label and is itself a
factor, so it drops into a model config unchanged. Wrap both the factors and
the labels: wrapping only the factors leaves training rows for
out-of-universe symbols, and wrapping only the labels leaves the factors'
cross-sections polluted by them.

```python
from quantlab.factor.alpha101 import Alpha101Stock
from quantlab.factor.universe_filter import UniverseFilteredFactor
from quantlab.label.fret import Return

factors = [UniverseFilteredFactor(Alpha101Stock(factor_config))]
labels = [UniverseFilteredFactor(Return(label_config), min_price=5.0,
                                 min_dollar_volume=1_000_000.0, window=20)]
```

The wrapper does more than blank its outputs. It rewrites the inner factor's
graph so that every cross-sectional operator (a rank or a cross-sectional
z-score across symbols) sees out-of-universe symbols as NaN, which keeps a
penny stock from shifting the ranks of every other symbol. Time-series
operators still see full history. Outputs are then set to NaN wherever the
mask is out. The symbol axis never shrinks: a symbol that is out for the
whole window stays as an all-NaN column, so a model trained on one window
can predict on another. The dataset's start date is also pulled earlier so
the dollar-volume window is warm on the first requested bar.

You can inspect the mask on any panel with raw `close` and `volume`. The
example uses `min_dollar_volume=2_000_000` and `window=5`:

```python
filtered = UniverseFilteredFactor(label, min_price=5.0,
                                  min_dollar_volume=2_000_000.0, window=5)
mask = filtered.compute_universe_mask(prices)   # 1.0 in, NaN out
mask.to_pandas().notna().mean()
```

```text
7. price/liquidity mask, share of bars in the universe:
symbol
AAA    0.94
BBB    0.92
DDD    0.28
EEE    0.27
PNY    0.00
```

`PNY` trades around 1.5 and is never in. `DDD` and `EEE` are in only for the
part of the window in which they traded, after their first five bars.

Three limits apply. The number of symbols must be a multiple of the SIMD
block width KunQuant compiles for on your machine (16 works on common
hardware; 13 does not). A time-series operator applied on top of a
cross-sectional one, such as a 10-bar correlation of two ranks, is NaN for a
full window after a symbol re-enters the universe, because its input was NaN
while the symbol was out; this matches what a trader could actually have
computed. And the model heads treat masked cells differently: tree models
such as `XGBoostRegressor` drop rows whose label is NaN, while the MLP head
replaces NaN with 0 before training, so masked cells become zero-valued
samples. See [Models](models.md).

The filter and index membership are independent and can be stacked: mask the
price panel to an index with `UniverseMask`, and wrap the factors and labels
with `UniverseFilteredFactor`.

## Symbols leaving the universe in a backtest

The backtester needs no universe setting of its own; the universe reaches it
through the model's predictions. A symbol that leaves the universe has NaN
factors, so the model predicts NaN for it, and on the next rebalance bar the
selector does not consider it. Its target weight becomes 0, and because a
signal formed at bar t fills at bar t + 1's open, the position is sold at the
open of the bar after that rebalance. Eligibility is evaluated only on
rebalance bars, so with `rebalance_periods=5` a position can be held up to
four bars after it left the universe; with `rebalance_periods=1` it is sold
the next day. See [Backtesting](backtesting.md).

Keep the backtester's `price_dataset` unfiltered: do not apply a universe
mask to it. Prices of held positions must stay available until they are sold,
otherwise a held symbol would have no price to trade at.

A symbol that leaves because it delisted, so that its price disappears
altogether, is handled separately. If a symbol is held after a rebalance and
has no fill price on the next bar, the engine sells it at its last known
price and records a *forced liquidation* (symbol, signal and fill timestamps,
price) in the run's `liquidations.json`. A symbol that has no price at the
start of the window and was never held is simply not listed yet, and trades
normally once it lists.

## See also

- [Datasets](datasets.md): build the price panels that universes restrict.
- [Factors](factors.md): the factors and labels the filter wraps.
- [WRDS](wrds.md): CRSP reference tables, PERMNOs and `security_filter`.
- The docstrings of `quantlab.universe.UniverseCatalog`,
  `quantlab.base.constituent.IndexConstituentDataset`,
  `quantlab.dataset._support.masking.UniverseMask` and
  `quantlab.factor.universe_filter.UniverseFilteredFactor` for every
  parameter.
