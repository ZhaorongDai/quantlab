"""Alphalens-style factor analysis over a ``(timestamp, symbol)`` panel.

The analysis asks how well a factor orders symbols by their forward return.
It pairs every analyzed factor variable with every forward-return variable
(a *fret*) and reports three groups of metrics, following the alphalens
library:

- Information analysis. The IC (information coefficient) of a period is the
  Spearman rank correlation between the factor and the forward return across
  the symbols of that timestamp. The report gives the IC series, its mean,
  standard deviation, IR (mean over standard deviation), t-statistic and
  p-value against zero, skew, excess kurtosis and monthly means.
- Returns analysis. At every timestamp the symbols are split into equal-count
  quantile buckets by factor value (bucket 1 holds the lowest values). The
  report gives the mean forward return per bucket and period, the top-minus-
  bottom spread, and cumulative returns per bucket and for the long-short
  spread.
- Turnover analysis. Quantile turnover is the fraction of symbols in a bucket
  that were not in that bucket at the previous timestamp. Rank
  autocorrelation is the Spearman correlation of the factor with itself one
  period earlier; a value near 1 means a slowly changing factor.

``FactorAnalyzer`` computes the metrics, ``FactorReportFigure`` draws one
composite matplotlib figure per pair, and ``FactorAnalysis`` holds the
results and writes them to disk. ``Factor.analyze`` is the usual entry point.
The inputs are ``xarray`` panels; the results are small tidy pandas tables.
matplotlib is imported only when a figure is drawn.

Examples
--------
The method examples in this module share this session: a signal over 20
symbols and 100 days, and a one-bar forward return it partly predicts.

>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": pd.date_range("2024-01-01", periods=100),
...           "symbol": [f"S{i}" for i in range(20)]}
>>> raw = rng.normal(size=(100, 20))
>>> signal = xr.DataArray(raw, coords=coords, dims=("timestamp", "symbol"))
>>> ret = xr.DataArray(0.01 * (0.3 * raw + rng.normal(size=(100, 20))),
...                    coords=coords, dims=("timestamp", "symbol"))
>>> pair = FactorAnalyzer().analyze_pair(signal, ret, "signal", "ret_1")
>>> analysis = FactorAnalysis(pairs={pair.key: pair},
...                           figures={pair.key: FactorReportFigure().render(pair)})
>>> round(pair.summary["ic_mean"], 4)
0.2384
"""

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats

from quantlab.utils.jsonable import to_jsonable

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def most_common_spacing(timestamps: xr.DataArray | pd.Index | np.ndarray) -> np.timedelta64:
    """Return the most common spacing between consecutive timestamps.

    This is the rule ``BaseDataset.time_interval`` uses: the mode rather than
    the minimum, so weekend and holiday gaps do not change the answer.

    Parameters
    ----------
    timestamps : xarray.DataArray, pandas.Index or numpy.ndarray
        Sorted ``datetime64`` timestamps; at least two.

    Returns
    -------
    numpy.timedelta64
        The most common spacing.

    Raises
    ------
    ValueError
        If fewer than two timestamps are given.

    Examples
    --------
    >>> most_common_spacing(pd.date_range("2024-01-01", periods=10, freq="B"))
    np.timedelta64(86400000000,'us')
    """
    values = pd.DatetimeIndex(np.asarray(timestamps))
    if len(values) < 2:
        raise ValueError(
            f"need at least two timestamps to infer a bar spacing, got {len(values)}"
        )
    diffs = pd.Series(values[1:] - values[:-1])
    return diffs.mode().to_numpy()[0]


def _per_bar_rate(returns: pd.DataFrame | pd.Series, horizon: int):
    """Convert ``horizon``-bar returns to the equivalent one-bar rate."""
    if horizon == 1:
        return returns
    return (1.0 + returns) ** (1.0 / horizon) - 1.0


#: Name of the forward-return column in the long frame of ``_collect``.
FRET_COLUMN = "__fret"
#: Resolution of the saved PNG figures.
FIGURE_DPI = 110


def pair_cumulative_ic(ic: pd.Series) -> pd.Series:
    """Return the running sum of ``ic`` with missing periods counted as 0."""
    return ic.fillna(0.0).cumsum().rename("cumulative_ic")


def _render_and_save(pair: "PairAnalysis", path: str) -> str:
    """Draw ``pair`` and save it to ``path``; the process-pool worker of ``save``."""
    FactorReportFigure().render(pair).savefig(path, dpi=FIGURE_DPI)
    return path


@dataclass
class PairAnalysis:
    """Metrics of one factor variable against one forward-return variable.

    Every time series is indexed by ``timestamp``; bucket columns are the
    integers ``1..quantiles``, bucket 1 holding the lowest factor values.

    Attributes
    ----------
    factor_name, fret_name : str
        The analyzed factor variable and forward-return variable.
    horizon : int
        Bars the forward return spans; cumulative returns compound the
        per-bar rate ``(1 + r) ** (1 / horizon) - 1``.
    quantiles : int
        Number of buckets.
    ic : pandas.Series
        Per-period Spearman IC.
    monthly_ic : pandas.Series
        Mean IC per calendar month, indexed by month end.
    quantile_returns : pandas.DataFrame
        Mean forward return per bucket and period.
    mean_quantile_returns : pandas.Series
        Time mean of ``quantile_returns`` per bucket.
    spread : pandas.Series
        Top-bucket minus bottom-bucket mean forward return per period.
    cumulative_quantile_returns : pandas.DataFrame
        Compounded per-bar return of each bucket.
    cumulative_long_short : pandas.Series
        Compounded per-bar return of long the top bucket, short the bottom.
    turnover : pandas.DataFrame
        Fraction of each bucket's symbols that were not in it the period
        before.
    rank_autocorrelation : pandas.Series
        Spearman correlation of the factor with its value one period
        earlier.
    cumulative_ic : pandas.Series
        Running sum of ``ic`` (a property).
    summary : dict
        The scalar metrics; one row of ``FactorAnalysis.summary_table()``.
    """

    factor_name: str
    fret_name: str
    horizon: int
    quantiles: int
    ic: pd.Series
    monthly_ic: pd.Series
    quantile_returns: pd.DataFrame
    mean_quantile_returns: pd.Series
    spread: pd.Series
    cumulative_quantile_returns: pd.DataFrame
    cumulative_long_short: pd.Series
    turnover: pd.DataFrame
    rank_autocorrelation: pd.Series
    summary: dict = field(default_factory=dict)

    @property
    def cumulative_ic(self) -> pd.Series:
        """Running sum of the per-period IC, periods without an IC adding 0.

        Examples
        --------
        >>> pair.cumulative_ic.iloc[-1] == pair.ic.fillna(0).sum()
        True
        """
        return pair_cumulative_ic(self.ic)

    @property
    def key(self) -> str:
        """``"<factor>__<fret>"``, the pair's name in files and dicts.

        Examples
        --------
        >>> pair.key
        'signal__ret_1'
        """
        return f"{self.factor_name}__{self.fret_name}"

    @property
    def start(self) -> pd.Timestamp:
        """First timestamp of the aligned panel.

        Examples
        --------
        >>> pair.start
        Timestamp('2024-01-01 00:00:00')
        """
        return self.ic.index[0]

    @property
    def end(self) -> pd.Timestamp:
        """Last timestamp of the aligned panel.

        Examples
        --------
        >>> pair.end
        Timestamp('2024-04-09 00:00:00')
        """
        return self.ic.index[-1]


@dataclass
class FactorAnalysis:
    """The result of ``Factor.analyze``: one ``PairAnalysis`` per pair.

    Attributes
    ----------
    pairs : dict[str, PairAnalysis]
        Keyed by ``"<factor>__<fret>"``.
    figures : dict[str, matplotlib.figure.Figure]
        The composite figure of each pair, same keys. The figures are built
        without ``pyplot``, so they are not registered with any GUI and need
        no closing; ``fig.savefig(path)`` writes one.
    config : dict
        ``{"factor": factor config, "frets": [fret configs]}``, the dicts
        ``load_factor_from_config`` rebuilds each object from.
    """

    pairs: dict[str, PairAnalysis]
    figures: dict[str, "Figure"] = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    def summary_table(self) -> pd.DataFrame:
        """Return the scalar metrics, one row per pair.

        Examples
        --------
        >>> table = analysis.summary_table()
        >>> table[["factor", "fret", "ic_mean", "mean_spread"]].round(4)
           factor   fret  ic_mean  mean_spread
        0  signal  ret_1   0.2384       0.0069
        """
        return pd.DataFrame([pair.summary for pair in self.pairs.values()])

    def ic_table(self) -> pd.DataFrame:
        """Return every pair's IC series as one tidy table.

        Columns are ``timestamp``, ``factor``, ``fret`` and ``ic``.

        Examples
        --------
        >>> analysis.ic_table().head(2).round({"ic": 4})
           timestamp  factor   fret      ic
        0 2024-01-01  signal  ret_1  0.1308
        1 2024-01-02  signal  ret_1 -0.1353
        """
        return self._tidy(lambda p: p.ic.rename("ic").to_frame())

    def quantile_returns_table(self) -> pd.DataFrame:
        """Return every pair's mean return per bucket and period, tidy.

        Columns are ``timestamp``, ``factor``, ``fret``, ``quantile`` and
        ``mean_return``.

        Examples
        --------
        >>> list(analysis.quantile_returns_table().columns)
        ['timestamp', 'factor', 'fret', 'quantile', 'mean_return']
        """
        return self._tidy_by_quantile(lambda p: p.quantile_returns, "mean_return")

    def turnover_table(self) -> pd.DataFrame:
        """Return every pair's bucket turnover and rank autocorrelation, tidy.

        Columns are ``timestamp``, ``factor``, ``fret``, ``quantile`` and
        ``turnover``; the factor's lag-1 rank autocorrelation is repeated on
        each bucket row as ``rank_autocorrelation``.

        Examples
        --------
        >>> list(analysis.turnover_table().columns)
        ['timestamp', 'factor', 'fret', 'quantile', 'turnover', 'rank_autocorrelation']
        """
        table = self._tidy_by_quantile(lambda p: p.turnover, "turnover")
        autocorr = self._tidy(
            lambda p: p.rank_autocorrelation.rename("rank_autocorrelation").to_frame()
        )
        return table.merge(autocorr, on=["timestamp", "factor", "fret"], how="left")

    def monthly_ic_table(self) -> pd.DataFrame:
        """Return every pair's monthly mean IC, tidy.

        Columns are ``month`` (month end), ``factor``, ``fret`` and ``ic``.

        Examples
        --------
        >>> analysis.monthly_ic_table().round({"ic": 4})
               month  factor   fret      ic
        0 2024-01-31  signal  ret_1  0.2159
        1 2024-02-29  signal  ret_1  0.1772
        2 2024-03-31  signal  ret_1  0.3103
        3 2024-04-30  signal  ret_1  0.2655
        """
        table = self._tidy(lambda p: p.monthly_ic.rename("ic").to_frame())
        return table.rename(columns={"timestamp": "month"})

    def save(self, output_dir: str | Path, workers: int | None = None) -> Path:
        """Write the analysis to ``output_dir``, creating it if needed.

        Files written: ``summary.json`` (every scalar metric, per pair),
        ``summary.csv``, ``ic.csv``, ``monthly_ic.csv``,
        ``quantile_returns.csv``, ``turnover.csv``, one
        ``<factor>__<fret>.png`` per pair and ``config.json`` (see
        ``config``). Floats that are NaN or infinite are written to JSON as
        ``null``. Figures held in ``figures`` are saved as they are; when
        none is held, every pair is drawn from its metrics and saved, on
        ``workers`` processes, without being kept.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Directory to write into.
        workers : int, optional
            Processes that draw the figures; ``None`` uses every CPU.

        Returns
        -------
        pathlib.Path
            ``output_dir``.

        Examples
        --------
        >>> out = analysis.save("report")
        >>> sorted(p.name for p in out.iterdir())
        ['config.json', 'ic.csv', 'monthly_ic.csv', 'quantile_returns.csv', 'signal__ret_1.png', 'summary.csv', 'summary.json', 'turnover.csv']
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summaries = [pair.summary for pair in self.pairs.values()]
        (out / "summary.json").write_text(
            json.dumps(to_jsonable({"pairs": summaries}), indent=2)
        )
        (out / "config.json").write_text(json.dumps(to_jsonable(self.config), indent=2))
        self.summary_table().to_csv(out / "summary.csv", index=False)
        self.ic_table().to_csv(out / "ic.csv", index=False)
        self.monthly_ic_table().to_csv(out / "monthly_ic.csv", index=False)
        self.quantile_returns_table().to_csv(out / "quantile_returns.csv", index=False)
        self.turnover_table().to_csv(out / "turnover.csv", index=False)
        if self.figures:
            for key, figure in self.figures.items():
                figure.savefig(out / f"{key}.png", dpi=FIGURE_DPI)
        elif self.pairs:
            self._render_to(out, workers)
        return out

    def _render_to(self, out: Path, workers: int | None) -> None:
        """Draw and save one PNG per pair, on ``workers`` processes.

        A single pair, or ``workers=1``, renders in this process; otherwise
        the pairs are spread over ``joblib.Parallel`` worker processes (the
        ``loky`` backend), since matplotlib draws on one thread and the
        figures dominate the cost of a large report. joblib is the one
        fan-out this repository uses (see ``tests/test_acquisition_progress
        .py::test_no_task_isolation_was_added``).
        """
        jobs = [(pair, str(out / f"{key}.png")) for key, pair in self.pairs.items()]
        count = workers if workers is not None else (os.cpu_count() or 1)
        if len(jobs) == 1 or count <= 1:
            for pair, path in jobs:
                _render_and_save(pair, path)
            return
        Parallel(n_jobs=min(count, len(jobs)), backend="loky")(
            delayed(_render_and_save)(pair, path) for pair, path in jobs
        )

    def _tidy(self, frame_of) -> pd.DataFrame:
        """Stack ``frame_of(pair)`` of every pair with factor/fret columns."""
        frames = []
        for pair in self.pairs.values():
            frame = frame_of(pair)
            frame.index.name = "timestamp"
            frame = frame.reset_index()
            frame.insert(1, "factor", pair.factor_name)
            frame.insert(2, "fret", pair.fret_name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def _tidy_by_quantile(self, frame_of, value_name: str) -> pd.DataFrame:
        """Stack a per-bucket wide table of every pair into long form."""
        frames = []
        for pair in self.pairs.values():
            wide = frame_of(pair).copy()
            wide.index.name = "timestamp"
            wide.columns.name = "quantile"
            long = wide.stack()
            long = long.rename(value_name).reset_index()
            long.insert(1, "factor", pair.factor_name)
            long.insert(2, "fret", pair.fret_name)
            frames.append(long)
        return pd.concat(frames, ignore_index=True)


class FactorAnalyzer:
    """Compute alphalens-style metrics of factor variables against frets.

    The metrics are computed with polars: the aligned panels become one
    long ``(timestamp, symbol)`` frame per chunk of factor variables, every
    per-period statistic of every variable in the chunk is one lazy plan,
    and the plan is collected once. Only the small per-period series are
    then finished in pandas.

    Parameters
    ----------
    quantiles : int, default 5
        Number of equal-count buckets per timestamp. Timestamps with fewer
        usable symbols than buckets get no bucket returns.
    rolling_window : int, default 22
        Window, in periods, of the rolling mean IC drawn in the figure.
    plot : bool, default True
        Draw figures. With an ``output_dir`` they are drawn on
        ``workers`` processes straight to PNG files and not kept; without
        one they are held in ``FactorAnalysis.figures``.
    workers : int, optional
        Processes that draw the figures of a saved report; ``None`` uses
        every CPU.
    chunk_size : int, default 32
        Factor variables per lazy plan. Each plan holds ``chunk_size + 3``
        float columns of ``timestamps * symbols`` rows in memory.

    Examples
    --------
    >>> analyzer = FactorAnalyzer(quantiles=5)
    >>> pair = analyzer.analyze_pair(signal, ret, "signal", "ret_1")
    >>> round(pair.summary["ic_mean"], 4), pair.mean_quantile_returns.round(4).tolist()
    (0.2384, [-0.0037, -0.0013, -0.0005, 0.0018, 0.0032])
    """

    def __init__(
        self,
        quantiles: int = 5,
        rolling_window: int = 22,
        plot: bool = True,
        workers: int | None = None,
        chunk_size: int = 32,
    ):
        """Initialize the analyzer; see the class docstring for parameters."""
        if quantiles < 2:
            raise ValueError(f"quantiles must be at least 2, got {quantiles}")
        self.quantiles = int(quantiles)
        self.rolling_window = int(rolling_window)
        self.plot = plot
        self.workers = workers
        self.chunk_size = max(int(chunk_size), 1)

    def run(
        self,
        factor,
        frets: Sequence,
        factor_names: Sequence[str] | None = None,
        output_dir: str | Path | None = None,
    ) -> FactorAnalysis:
        """Analyze ``factor`` against every fret; see ``Factor.analyze``.

        Parameters
        ----------
        factor : Factor
            A factor whose panel is computed or read; ``get_features()`` is
            analyzed.
        frets : sequence of Factor
            Forward-return labels whose panels are computed or read;
            ``get_labels()`` of each is used.
        factor_names : sequence of str, optional
            Factor variables to analyze; all of ``get_factor_names()`` when
            None.
        output_dir : str or pathlib.Path, optional
            When given, ``FactorAnalysis.save`` writes the results there and
            the figures are not kept in memory.

        Returns
        -------
        FactorAnalysis
            The metrics, figures and configs.

        Raises
        ------
        ValueError
            If no fret is given, a factor name is unknown, a fret panel's bar
            spacing differs from the factor's, the aligned panel is empty, or
            two pairs share a name.

        Examples
        --------
        >>> analysis = FactorAnalyzer(quantiles=4).run(factor, [fwd])
        >>> list(analysis.pairs)
        ['momentum_5__ret_1']
        """
        if not frets:
            raise ValueError("analyze needs at least one forward-return label in `frets`")
        features = factor.get_features()
        available = list(features.data_vars)
        names = list(factor.get_factor_names() if factor_names is None else factor_names)
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(
                f"factor_names {unknown} are not in {factor.class_name}'s panel; "
                f"available: {available}"
            )

        pairs: dict[str, PairAnalysis] = {}
        for fret in frets:
            labels = fret.get_labels()
            self.check_frequency(
                features, labels, factor.class_name, type(fret).__name__
            )
            aligned_features, aligned_labels = xr.align(
                features[names], labels, join="inner"
            )
            if aligned_features.sizes.get("timestamp", 0) == 0 or aligned_features.sizes.get(
                "symbol", 0
            ) == 0:
                raise ValueError(
                    f"{factor.class_name} and {type(fret).__name__} share no "
                    f"(timestamp, symbol) cells"
                )
            for pair in self.analyze_many(
                aligned_features, aligned_labels, horizon=self._horizon_of(fret)
            ):
                if pair.key in pairs:
                    raise ValueError(
                        f"two factor/fret pairs are both named {pair.key!r}; "
                        f"give the frets distinct variable names"
                    )
                pairs[pair.key] = pair

        config = {
            "factor": factor.get_config(),
            "frets": [fret.get_config() for fret in frets],
        }
        analysis = FactorAnalysis(pairs=pairs, figures={}, config=config)
        if output_dir is not None:
            analysis.save(output_dir, workers=self.workers if self.plot else 0)
        elif self.plot:
            renderer = FactorReportFigure(rolling_window=self.rolling_window)
            analysis.figures = {key: renderer.render(pair) for key, pair in pairs.items()}
        return analysis

    @staticmethod
    def check_frequency(
        features: xr.Dataset,
        labels: xr.Dataset,
        factor_label: str = "factor",
        fret_label: str = "fret",
    ) -> np.timedelta64:
        """Refuse panels whose most common bar spacing differs.

        Parameters
        ----------
        features, labels : xarray.Dataset
            The factor and forward-return panels.
        factor_label, fret_label : str
            Names used in the error message.

        Returns
        -------
        numpy.timedelta64
            The shared spacing.

        Raises
        ------
        ValueError
            If the spacings differ.

        Examples
        --------
        >>> FactorAnalyzer.check_frequency(daily, weekly, "signal", "ret")
        Traceback (most recent call last):
        ...
        ValueError: bar spacing differs: signal has 1 days 00:00:00 bars, ret has 7 days 00:00:00 bars; resample one panel to the other's frequency first
        """
        factor_step = most_common_spacing(features["timestamp"].values)
        fret_step = most_common_spacing(labels["timestamp"].values)
        if factor_step != fret_step:
            raise ValueError(
                f"bar spacing differs: {factor_label} has {pd.Timedelta(factor_step)} "
                f"bars, {fret_label} has {pd.Timedelta(fret_step)} bars; resample "
                f"one panel to the other's frequency first"
            )
        return factor_step

    def analyze_many(
        self, features: xr.Dataset, labels: xr.Dataset, horizon: int = 1
    ) -> list[PairAnalysis]:
        """Compute every metric of every factor variable against every fret.

        Parameters
        ----------
        features, labels : xarray.Dataset
            Aligned ``(timestamp, symbol)`` panels: the factor variables and
            the forward-return variables. Only cells finite in both a factor
            and a fret are used for that pair.
        horizon : int, default 1
            Bars the forward returns span.

        Returns
        -------
        list of PairAnalysis
            One per ``(factor variable, fret variable)``, frets outermost.

        Examples
        --------
        >>> pairs = FactorAnalyzer().analyze_many(signals, rets, horizon=1)
        >>> [pair.key for pair in pairs]
        ['signal__ret_1', 'noise__ret_1']
        """
        features = features.transpose("timestamp", "symbol")
        labels = labels.transpose("timestamp", "symbol")
        timestamps = pd.DatetimeIndex(features["timestamp"].values, name="timestamp")
        n_symbols = features.sizes["symbol"]
        names = list(features.data_vars)
        base = {
            "timestamp": np.repeat(timestamps.values.astype("datetime64[ns]"), n_symbols),
            "symbol": np.tile(np.arange(n_symbols, dtype=np.int32), len(timestamps)),
        }
        pairs = []
        for fret_name in labels.data_vars:
            r = labels[fret_name].values.astype(np.float64)
            for start in range(0, len(names), self.chunk_size):
                chunk = names[start : start + self.chunk_size]
                data = dict(base)
                data[FRET_COLUMN] = r.ravel()
                counts = {}
                for i, name in enumerate(chunk):
                    f = features[name].values.astype(np.float64)
                    data[f"f{i}"] = f.ravel()
                    counts[name] = int((np.isfinite(f) & np.isfinite(r)).any(axis=0).sum())
                table = self._collect(pl.DataFrame(data).lazy(), len(chunk))
                for i, name in enumerate(chunk):
                    pairs.append(
                        self._finish_pair(
                            table, i, name, str(fret_name), horizon, counts[name], timestamps
                        )
                    )
        return pairs

    def _collect(self, lf: pl.LazyFrame, n_factors: int) -> pd.DataFrame:
        """Run the per-period plan of ``n_factors`` factor columns and collect it.

        The frame is timestamp-major, so a shift within a symbol is the
        previous period. NaN cells are nulls, buckets are assigned per
        period by ordinal rank, and every aggregation of every factor is one
        ``group_by("timestamp")`` plan collected once.
        """
        q_count = self.quantiles
        fret = pl.col(FRET_COLUMN)

        row_exprs = []
        for i in range(n_factors):
            f = pl.col(f"f{i}")
            both = f.is_not_null() & fret.is_not_null()
            fv = pl.when(both).then(f)
            n_valid = fv.count().over("timestamp")
            bucket = (
                (fv.rank(method="ordinal").over("timestamp").cast(pl.Int64) - 1) * q_count
            ) // n_valid + 1
            row_exprs += [
                fv.alias(f"fv{i}"),
                pl.when(both).then(fret).alias(f"rv{i}"),
                pl.when(n_valid >= q_count).then(bucket).alias(f"b{i}"),
                f.shift(1).over("symbol").alias(f"lag{i}"),
            ]
        lf = lf.with_columns(pl.all().exclude("timestamp", "symbol").fill_nan(None))
        lf = lf.with_columns(row_exprs)
        lf = lf.with_columns(
            [pl.col(f"b{i}").shift(1).over("symbol").alias(f"pb{i}") for i in range(n_factors)]
        )

        aggs = []
        for i in range(n_factors):
            fv, rv, bucket, prev = (pl.col(f"{c}{i}") for c in ("fv", "rv", "b", "pb"))
            f, lag = pl.col(f"f{i}"), pl.col(f"lag{i}")
            both = f.is_not_null() & lag.is_not_null()
            had_previous = prev.is_not_null().any()
            aggs.append(pl.corr(fv.rank(), rv.rank()).alias(f"ic{i}"))
            aggs.append(
                pl.corr(pl.when(both).then(f).rank(), pl.when(both).then(lag).rank())
                .alias(f"rac{i}")
            )
            for q in range(1, q_count + 1):
                in_q = bucket == q
                count = in_q.sum()
                new = (in_q & prev.ne_missing(q)).sum()
                aggs.append(rv.filter(in_q).mean().alias(f"q{i}_{q}"))
                aggs.append(
                    pl.when((count > 0) & had_previous)
                    .then(new / count)
                    .alias(f"t{i}_{q}")
                )
        return (
            lf.group_by("timestamp", maintain_order=True)
            .agg(aggs)
            .sort("timestamp")
            .collect()
            .to_pandas()
            .set_index("timestamp")
        )

    def _finish_pair(
        self,
        table: pd.DataFrame,
        i: int,
        factor_name: str,
        fret_name: str,
        horizon: int,
        n_symbols: int,
        index: pd.DatetimeIndex,
    ) -> PairAnalysis:
        """Build the ``PairAnalysis`` of factor column ``i`` from the collected table."""
        columns = pd.Index(range(1, self.quantiles + 1), name="quantile")
        ic = table[f"ic{i}"].astype(np.float64).reindex(index).rename("ic")
        monthly_ic = ic.groupby(pd.Grouper(freq="ME")).mean()
        monthly_ic.index.name = "timestamp"
        quantile_returns = pd.DataFrame(
            {q: table[f"q{i}_{q}"].to_numpy(dtype=np.float64) for q in columns},
            index=index, columns=columns,
        )
        turnover = pd.DataFrame(
            {q: table[f"t{i}_{q}"].to_numpy(dtype=np.float64) for q in columns},
            index=index, columns=columns,
        )
        rank_autocorr = (
            table[f"rac{i}"].astype(np.float64).reindex(index).rename("rank_autocorrelation")
        )

        spread = (quantile_returns[self.quantiles] - quantile_returns[1]).rename("spread")
        per_bar = _per_bar_rate(quantile_returns, horizon)
        cumulative_quantile = (1.0 + per_bar.fillna(0.0)).cumprod() - 1.0
        long_short_rate = (per_bar[self.quantiles] - per_bar[1]).fillna(0.0)
        cumulative_long_short = ((1.0 + long_short_rate).cumprod() - 1.0).rename(
            "cumulative_long_short"
        )
        pair = PairAnalysis(
            factor_name=factor_name,
            fret_name=fret_name,
            horizon=int(horizon),
            quantiles=self.quantiles,
            ic=ic,
            monthly_ic=monthly_ic,
            quantile_returns=quantile_returns,
            mean_quantile_returns=quantile_returns.mean().rename("mean_return"),
            spread=spread,
            cumulative_quantile_returns=cumulative_quantile,
            cumulative_long_short=cumulative_long_short,
            turnover=turnover,
            rank_autocorrelation=rank_autocorr,
        )
        pair.summary = self._summary(pair, n_symbols=n_symbols)
        return pair

    def analyze_pair(
        self,
        factor: xr.DataArray,
        fret: xr.DataArray,
        factor_name: str,
        fret_name: str,
        horizon: int = 1,
    ) -> PairAnalysis:
        """Compute every metric of one aligned factor/fret pair.

        Parameters
        ----------
        factor, fret : xarray.DataArray
            Aligned ``(timestamp, symbol)`` panels. Only cells finite in both
            are used.
        factor_name, fret_name : str
            Names recorded in the result.
        horizon : int, default 1
            Bars the forward return spans.

        Returns
        -------
        PairAnalysis
            The metrics.

        Examples
        --------
        >>> pair = FactorAnalyzer().analyze_pair(signal, ret, "signal", "ret_1")
        >>> pair.key, pair.quantile_returns.shape
        ('signal__ret_1', (100, 5))
        """
        return self.analyze_many(
            factor.to_dataset(name=factor_name),
            fret.to_dataset(name=fret_name),
            horizon=horizon,
        )[0]

    @staticmethod
    def _horizon_of(fret) -> int:
        """Read the label horizon from ``config.kwargs["n_forward_periods"]``, else 1."""
        kwargs = getattr(getattr(fret, "config", None), "kwargs", None) or {}
        horizon = kwargs.get("n_forward_periods", 1)
        return max(int(horizon), 1)

    def _summary(self, pair: PairAnalysis, n_symbols: int) -> dict[str, Any]:
        """Collect the scalar metrics of ``pair`` into a flat dict."""
        ic = pair.ic.dropna().to_numpy()
        n = len(ic)
        mean = float(ic.mean()) if n else math.nan
        std = float(ic.std(ddof=1)) if n > 1 else math.nan
        with np.errstate(invalid="ignore", divide="ignore"):
            ir = np.float64(mean) / np.float64(std)
            t_stat = ir * np.sqrt(n) if n > 1 else math.nan
        p_value = float(2.0 * stats.t.sf(abs(t_stat), n - 1)) if n > 1 else math.nan
        varies = n > 2 and std > 0
        summary: dict[str, Any] = {
            "factor": pair.factor_name,
            "fret": pair.fret_name,
            "horizon": pair.horizon,
            "quantiles": pair.quantiles,
            "start": pair.start,
            "end": pair.end,
            "n_periods": len(pair.ic),
            "n_symbols": n_symbols,
            "ic_mean": mean,
            "ic_std": std,
            "ir": float(ir),
            "ic_t_stat": float(t_stat),
            "ic_p_value": p_value,
            "ic_skew": float(stats.skew(ic)) if varies else math.nan,
            "ic_kurtosis": float(stats.kurtosis(ic)) if varies else math.nan,
            "ic_positive_ratio": float((ic > 0).mean()) if n else math.nan,
        }
        for q, value in pair.mean_quantile_returns.items():
            summary[f"mean_return_q{q}"] = float(value)
        summary["mean_spread"] = float(pair.spread.mean())
        summary["cumulative_long_short"] = float(pair.cumulative_long_short.iloc[-1])
        summary["mean_turnover_top"] = float(pair.turnover[pair.quantiles].mean())
        summary["mean_turnover_bottom"] = float(pair.turnover[1].mean())
        summary["mean_rank_autocorrelation"] = float(pair.rank_autocorrelation.mean())
        return summary


# Muted palette. Blue and red are the two diverging poles (top and bottom
# bucket), gray is the neutral midpoint; ink and grid colors keep text and
# chrome recessive.
_BLUE = "#2a78d6"
_BLUE_LIGHT = "#9ec5f4"
_BLUE_DARK = "#184f95"
_RED = "#e34948"
_ORANGE = "#eb6834"
_NEUTRAL = "#b8b7b2"
_NEUTRAL_LIGHT = "#f0efec"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e0"
_SURFACE = "#fcfcfb"


class FactorReportFigure:
    """Draw one composite matplotlib figure of a ``PairAnalysis``.

    The figure is a grid of panels: the IC series with its rolling mean;
    the IC histogram with a fitted normal curve; a normal QQ plot of the IC;
    the monthly mean IC heatmap; mean forward return by bucket; cumulative
    return by bucket; the cumulative long-short return; bucket turnover;
    factor rank autocorrelation; and a table of the scalar metrics. The
    suptitle names the factor, the fret and the date range.

    Parameters
    ----------
    rolling_window : int, default 22
        Window, in periods, of the rolling mean IC.

    Examples
    --------
    >>> fig = FactorReportFigure().render(pair)
    >>> type(fig).__name__, len(fig.axes)
    ('Figure', 11)
    >>> fig.savefig("signal__ret_1.png")
    """

    def __init__(self, rolling_window: int = 22):
        """Initialize the renderer; see the class docstring for parameters."""
        self.rolling_window = int(rolling_window)

    def render(self, pair: PairAnalysis) -> "Figure":
        """Return the composite figure of ``pair``.

        The figure is built with ``matplotlib.figure.Figure`` rather than
        ``pyplot``, so it is never shown and needs no closing.

        Parameters
        ----------
        pair : PairAnalysis
            The metrics to draw.

        Returns
        -------
        matplotlib.figure.Figure
            The figure.

        Examples
        --------
        >>> fig = FactorReportFigure(rolling_window=10).render(pair)
        >>> fig.get_suptitle()
        'signal  vs  ret_1   |   2024-01-01 to 2024-04-09'
        """
        from matplotlib.figure import Figure

        fig = Figure(figsize=(16, 25), facecolor=_SURFACE, layout="constrained")
        grid = fig.add_gridspec(6, 2, height_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.15])
        fig.suptitle(
            f"{pair.factor_name}  vs  {pair.fret_name}   |   "
            f"{pair.start:%Y-%m-%d} to {pair.end:%Y-%m-%d}",
            fontsize=18,
            fontweight="bold",
            color=_INK,
        )
        self._ic_series(fig.add_subplot(grid[0, :]), pair)
        self._ic_histogram(fig.add_subplot(grid[1, 0]), pair)
        self._ic_qq(fig.add_subplot(grid[1, 1]), pair)
        self._monthly_heatmap(fig, fig.add_subplot(grid[2, 0]), pair)
        self._mean_quantile_returns(fig.add_subplot(grid[2, 1]), pair)
        self._cumulative_quantiles(fig.add_subplot(grid[3, 0]), pair)
        self._cumulative_long_short(fig.add_subplot(grid[3, 1]), pair)
        self._turnover(fig.add_subplot(grid[4, 0]), pair)
        self._rank_autocorrelation(fig.add_subplot(grid[4, 1]), pair)
        self._summary_table(fig.add_subplot(grid[5, :]), pair)
        return fig

    @staticmethod
    def _quantile_colors(quantiles: int) -> list:
        """Diverging colors from red (bottom bucket) through gray to blue (top)."""
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("quantiles", [_RED, _NEUTRAL, _BLUE])
        return [cmap(i / (quantiles - 1)) for i in range(quantiles)]

    @staticmethod
    def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
        """Apply the shared axis style: recessive spines and grid, titles."""
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=_INK)
        ax.set_xlabel(xlabel, color=_INK_SECONDARY)
        ax.set_ylabel(ylabel, color=_INK_SECONDARY)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_NEUTRAL)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    @staticmethod
    def _no_data(ax) -> None:
        """Mark an axis as having nothing to draw."""
        ax.text(0.5, 0.5, "no data", ha="center", va="center", color=_INK_SECONDARY,
                transform=ax.transAxes)

    @staticmethod
    def _percent(ax, axis: str = "y") -> None:
        """Format an axis as percentages."""
        from matplotlib.ticker import PercentFormatter

        target = ax.yaxis if axis == "y" else ax.xaxis
        target.set_major_formatter(PercentFormatter(1.0))

    def _ic_series(self, ax, pair: PairAnalysis) -> None:
        """IC per period with its rolling mean, and the cumulative IC on a right axis."""
        self._style(ax, "Information coefficient (Spearman)", "", "IC")
        ic = pair.ic
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        lines = ax.plot(ic.index, ic.to_numpy(), color=_BLUE_LIGHT, linewidth=1, label="IC")
        window = max(1, min(self.rolling_window, len(ic)))
        rolling = ic.rolling(window, min_periods=max(1, window // 2)).mean()
        lines += ax.plot(rolling.index, rolling.to_numpy(), color=_BLUE_DARK, linewidth=2,
                         label=f"{window}-period mean")
        lines.append(ax.axhline(pair.summary["ic_mean"], color=_ORANGE, linewidth=1.5,
                                linestyle="--", label=f"mean {pair.summary['ic_mean']:.3f}"))
        right = ax.twinx()
        cumulative = pair.cumulative_ic
        lines += right.plot(cumulative.index, cumulative.to_numpy(), color=_RED,
                            linewidth=1.8, label="cumulative IC")
        right.set_ylabel("cumulative IC", color=_RED)
        right.tick_params(colors=_RED, labelsize=9)
        right.grid(False)
        for side in ("top", "left", "bottom"):
            right.spines[side].set_visible(False)
        right.spines["right"].set_color(_RED)
        ax.legend(lines, [line.get_label() for line in lines], loc="lower right",
                  bbox_to_anchor=(1.0, 1.0), frameon=False, ncols=4)

    def _ic_histogram(self, ax, pair: PairAnalysis) -> None:
        """IC histogram with the normal density of the same mean and std."""
        self._style(ax, "IC distribution", "IC", "density")
        ic = pair.ic.dropna().to_numpy()
        if len(ic) == 0:
            self._no_data(ax)
            return
        ax.hist(ic, bins=min(40, max(5, len(ic) // 5)), density=True, color=_BLUE,
                alpha=0.75, edgecolor=_SURFACE, linewidth=1)
        mean, std = pair.summary["ic_mean"], pair.summary["ic_std"]
        if np.isfinite(std) and std > 0:
            x = np.linspace(ic.min() - std, ic.max() + std, 200)
            ax.plot(x, stats.norm.pdf(x, mean, std), color=_INK, linewidth=2,
                    label="normal fit")
            ax.legend(loc="upper left", frameon=False)
        ax.axvline(mean, color=_ORANGE, linestyle="--", linewidth=1.5)

    def _ic_qq(self, ax, pair: PairAnalysis) -> None:
        """Normal QQ plot of the IC."""
        self._style(ax, "IC normal QQ plot", "normal quantile", "observed IC quantile")
        ic = np.sort(pair.ic.dropna().to_numpy())
        if len(ic) < 2:
            self._no_data(ax)
            return
        theoretical = stats.norm.ppf((np.arange(1, len(ic) + 1) - 0.5) / len(ic))
        ax.scatter(theoretical, ic, s=14, color=_BLUE, alpha=0.8, edgecolors="none")
        mean, std = pair.summary["ic_mean"], pair.summary["ic_std"]
        if np.isfinite(std):
            ax.plot(theoretical, mean + std * theoretical, color=_INK, linewidth=1.5)

    def _monthly_heatmap(self, fig, ax, pair: PairAnalysis) -> None:
        """Monthly mean IC as a year-by-month heatmap."""
        self._style(ax, "Monthly mean IC", "month", "year")
        ax.grid(False)
        monthly = pair.monthly_ic
        if monthly.dropna().empty:
            self._no_data(ax)
            return
        table = (
            monthly.to_frame("ic")
            .assign(year=monthly.index.year, month=monthly.index.month)
            .pivot(index="year", columns="month", values="ic")
            .reindex(columns=range(1, 13))
        )
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("ic", [_RED, _NEUTRAL_LIGHT, _BLUE])
        limit = float(np.nanmax(np.abs(table.to_numpy()))) or 1.0
        image = ax.imshow(table.to_numpy(), cmap=cmap, vmin=-limit, vmax=limit,
                          aspect="auto")
        ax.set_xticks(range(12), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
        ax.set_yticks(range(len(table.index)), [str(y) for y in table.index])
        if table.size <= 120:
            for (row, col), value in np.ndenumerate(table.to_numpy()):
                if np.isfinite(value):
                    ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                            fontsize=8, color=_INK)
        fig.colorbar(image, ax=ax, shrink=0.85)

    def _mean_quantile_returns(self, ax, pair: PairAnalysis) -> None:
        """Mean forward return per bucket."""
        self._style(ax, "Mean forward return by quantile", "quantile", "mean return")
        values = pair.mean_quantile_returns
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.bar([str(q) for q in values.index], values.to_numpy(),
               color=self._quantile_colors(pair.quantiles), width=0.7,
               edgecolor=_SURFACE, linewidth=2)
        self._percent(ax)

    def _cumulative_quantiles(self, ax, pair: PairAnalysis) -> None:
        """Cumulative return of every bucket."""
        self._style(ax, "Cumulative return by quantile", "", "cumulative return")
        colors = self._quantile_colors(pair.quantiles)
        for q, color in zip(pair.cumulative_quantile_returns.columns, colors):
            series = pair.cumulative_quantile_returns[q]
            ax.plot(series.index, series.to_numpy(), color=color, linewidth=2,
                    label=f"Q{q}")
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.legend(loc="upper left", ncols=min(pair.quantiles, 5), facecolor=_SURFACE,
                  edgecolor=_GRID, framealpha=0.9)
        self._percent(ax)

    def _cumulative_long_short(self, ax, pair: PairAnalysis) -> None:
        """Cumulative return of long the top bucket, short the bottom."""
        self._style(ax, f"Cumulative long-short return (Q{pair.quantiles} - Q1)", "",
                    "cumulative return")
        series = pair.cumulative_long_short
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        ax.plot(series.index, series.to_numpy(), color=_BLUE, linewidth=2)
        ax.fill_between(series.index, series.to_numpy(), 0.0, color=_BLUE, alpha=0.12)
        self._percent(ax)

    def _band(self, ax, series: pd.Series, color: str, label: str,
              band_label: str | None = None) -> None:
        """Draw a series as its rolling range and rolling mean.

        The translucent band spans the rolling minimum to maximum over
        ``rolling_window`` periods; the line is the rolling mean. The raw
        series is not drawn, so a noisy per-period statistic reads as a
        level and a spread rather than as spikes. The band enters the
        legend only when ``band_label`` is given.
        """
        window = max(1, min(self.rolling_window, len(series)))
        rolling = series.rolling(window, min_periods=max(1, window // 2))
        low, high, mean = rolling.min(), rolling.max(), rolling.mean()
        ax.fill_between(series.index, low.to_numpy(), high.to_numpy(), color=color,
                        alpha=0.18, linewidth=0,
                        label=band_label if band_label is not None else "_nolegend_")
        ax.plot(mean.index, mean.to_numpy(), color=color, linewidth=2, label=label)

    @staticmethod
    def _inset_legend(ax) -> None:
        """A small legend inside the axes, on a translucent panel."""
        ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.85,
                  facecolor=_SURFACE, edgecolor=_GRID, ncols=3)

    def _window_of(self, series) -> int:
        """The rolling window actually used for ``series``."""
        return max(1, min(self.rolling_window, len(series)))

    def _turnover(self, ax, pair: PairAnalysis) -> None:
        """Rolling range and mean of the top and bottom buckets' turnover."""
        window = self._window_of(pair.turnover)
        self._style(ax, f"Quantile turnover, {window}-period", "", "turnover")
        self._band(ax, pair.turnover[pair.quantiles], _BLUE,
                   f"Q{pair.quantiles} (top) mean", band_label="range")
        self._band(ax, pair.turnover[1], _RED, "Q1 (bottom) mean")
        ax.set_ylim(bottom=max(-0.02, ax.get_ylim()[0]))
        self._inset_legend(ax)
        self._percent(ax)

    def _rank_autocorrelation(self, ax, pair: PairAnalysis) -> None:
        """Rolling range and mean of the lag-1 factor rank autocorrelation."""
        window = self._window_of(pair.rank_autocorrelation)
        self._style(ax, f"Rank autocorrelation (lag 1), {window}-period", "", "autocorrelation")
        ax.axhline(0.0, color=_INK_SECONDARY, linewidth=1)
        self._band(ax, pair.rank_autocorrelation, _BLUE_DARK, "mean", band_label="range")
        self._inset_legend(ax)

    def _summary_table(self, ax, pair: PairAnalysis) -> None:
        """The scalar metrics as a two-block text table."""
        ax.axis("off")
        ax.set_title("Summary", loc="left", fontsize=13, fontweight="bold", color=_INK)
        s = pair.summary

        def fmt(value, kind="f"):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "n/a"
            if kind == "%":
                return f"{value:.3%}"
            if kind == "e":
                return f"{value:.2e}"
            if kind == "d":
                return f"{value:d}"
            return f"{value:.4f}"

        left = [
            ("IC mean", fmt(s["ic_mean"])),
            ("IC std", fmt(s["ic_std"])),
            ("IR (mean / std)", fmt(s["ir"])),
            ("t-stat", fmt(s["ic_t_stat"])),
            ("p-value", fmt(s["ic_p_value"], "e")),
            ("IC skew", fmt(s["ic_skew"])),
            ("IC excess kurtosis", fmt(s["ic_kurtosis"])),
            ("IC > 0", fmt(s["ic_positive_ratio"], "%")),
        ]
        right = [
            ("periods / symbols", f"{s['n_periods']} / {s['n_symbols']}"),
            ("horizon (bars)", fmt(s["horizon"], "d")),
            (f"mean return Q{pair.quantiles}", fmt(s[f"mean_return_q{pair.quantiles}"], "%")),
            ("mean return Q1", fmt(s["mean_return_q1"], "%")),
            ("mean spread", fmt(s["mean_spread"], "%")),
            ("cumulative long-short", fmt(s["cumulative_long_short"], "%")),
            ("turnover top / bottom",
             f"{fmt(s['mean_turnover_top'], '%')} / {fmt(s['mean_turnover_bottom'], '%')}"),
            ("rank autocorrelation", fmt(s["mean_rank_autocorrelation"])),
        ]
        rows = [[a, b, c, d] for (a, b), (c, d) in zip(left, right)]
        table = ax.table(cellText=rows, colLabels=["metric", "value", "metric", "value"],
                         loc="upper center", cellLoc="left", colLoc="left",
                         colWidths=[0.25, 0.2, 0.3, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.0, 1.9)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor(_GRID)
            cell.set_facecolor(_NEUTRAL_LIGHT if row == 0 else _SURFACE)
            cell.get_text().set_color(_INK if row == 0 else _INK_SECONDARY)
            if row == 0:
                cell.get_text().set_fontweight("bold")
