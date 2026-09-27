"""How much the factor variables of one analysis say the same thing.

The correlation of two factor variables is the Spearman rank correlation of
their values across the symbols of one timestamp, averaged over timestamps,
the same per-period statistic the IC uses between a factor and a forward
return. Each variable is ranked once among its own finite symbols of the
timestamp, and a pair's correlation of those ranks uses the symbols finite
in both; a timestamp with fewer than three such symbols is left out of that
pair. When both variables cover the same symbols this is exactly Spearman's
correlation. When their coverage differs it is a close approximation, since
the ranks are not recomputed on the shared symbols; that keeps hundreds of
variables to a few matrix products per timestamp instead of re-ranking
every pair.

The variables are clustered by the distance ``1 - |mean correlation|``
(average linkage), so a factor and its negation fall in one cluster, and a
cluster cut at ``threshold`` groups variables whose correlation is mostly at
least that strong. The clustering orders the matrix, which is what makes a
heatmap of hundreds of variables readable: redundant factors form blocks on
the diagonal.

``FactorCorrelation`` computes and holds the result and
``FactorCorrelationFigure`` draws it. ``Factor.analyze`` computes it
whenever two or more factor variables are analyzed.

Examples
--------
>>> rng = np.random.default_rng(0)
>>> coords = {"timestamp": pd.date_range("2024-01-01", periods=60),
...           "symbol": [f"S{i}" for i in range(30)]}
>>> a = rng.normal(size=(60, 30))
>>> panel = xr.Dataset({"a": (("timestamp", "symbol"), a),
...                     "a_neg": (("timestamp", "symbol"), -a),
...                     "noise": (("timestamp", "symbol"), rng.normal(size=(60, 30)))},
...                    coords=coords)
>>> corr = FactorCorrelation.compute(panel)
>>> corr.mean.round(2)
       noise     a  a_neg
noise   1.00 -0.01   0.01
a      -0.01  1.00  -1.00
a_neg   0.01 -1.00   1.00
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Symbols a timestamp needs, finite in both variables, to enter a pair.
MIN_SYMBOLS = 3

#: Bytes of the ``[block, factor, factor]`` work arrays one block may use.
_BLOCK_BYTES = 256 * 2**20


@dataclass
class FactorCorrelation:
    """Mean cross-sectional rank correlation between factor variables.

    Every matrix is indexed by factor name on both axes, in the cluster
    order, so correlated variables sit next to each other.

    Attributes
    ----------
    mean : pd.DataFrame
        Mean over timestamps of the per-timestamp Spearman correlation.
        NaN for a pair that shares no usable timestamp.
    std : pd.DataFrame
        Standard deviation (``ddof=1``) of the per-timestamp correlation.
    periods : pd.DataFrame
        Timestamps each pair was computed on.
    clusters : pd.Series
        Cluster number of each variable, numbered ``1, 2, ...`` along the
        order.
    threshold : float
        The ``|correlation|`` the clusters were cut at.
    """

    mean: pd.DataFrame
    std: pd.DataFrame
    periods: pd.DataFrame
    clusters: pd.Series
    threshold: float

    @classmethod
    def compute(
        cls,
        features: xr.Dataset,
        threshold: float = 0.7,
        block_size: int | None = None,
    ) -> "FactorCorrelation":
        """Correlate every pair of variables of ``features``.

        Parameters
        ----------
        features : xr.Dataset
            A ``(timestamp, symbol)`` panel, one variable per factor.
        threshold : float, default 0.7
            ``|correlation|`` the clusters are cut at, in ``(0, 1]``.
        block_size : int, optional
            Timestamps processed at once. By default as many as keep the
            work arrays near 256 MiB, counting both symbols and factors;
            the result does not depend on it. The panel itself is never
            copied whole: each block is stacked on its own.

        Returns
        -------
        FactorCorrelation
            The matrices in cluster order.

        Raises
        ------
        ValueError
            If ``features`` has fewer than two variables or ``threshold`` is
            not in ``(0, 1]``.

        Examples
        --------
        >>> corr = FactorCorrelation.compute(panel, threshold=0.7)
        >>> corr.clusters.to_dict()
        {'noise': 1, 'a': 2, 'a_neg': 2}
        """
        names = [str(n) for n in features.data_vars]
        if len(names) < 2:
            raise ValueError(
                f"a factor correlation needs at least two variables, got {names}"
            )
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")
        features = features[names].transpose("timestamp", "symbol")
        n_times = features.sizes["timestamp"]
        n_symbols, n_factors = features.sizes["symbol"], len(names)
        if block_size is None:
            # float64 work arrays per timestamp: about seven [symbol, factor]
            # (the block, its ranks, the masked ranks and their temporaries)
            # and ten [factor, factor] (the masked sums and the correlation).
            per_time = 8 * (7 * n_symbols * n_factors + 10 * n_factors * n_factors)
            block_size = max(1, _BLOCK_BYTES // per_time)
        total = np.zeros((n_factors, n_factors))
        total_sq = np.zeros((n_factors, n_factors))
        count = np.zeros((n_factors, n_factors))
        for start in range(0, n_times, block_size):
            # Only this block is stacked into one array, never the whole panel.
            block = np.asarray(
                features.isel(timestamp=slice(start, start + block_size))
                .to_array("factor")
                .transpose("timestamp", "symbol", "factor")
                .values,
                dtype=np.float64,
            )
            corr = _block_correlations(block)
            valid = np.isfinite(corr)
            total += np.where(valid, corr, 0.0).sum(axis=0)
            total_sq += np.where(valid, corr * corr, 0.0).sum(axis=0)
            count += valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / count, np.nan)
            var = (total_sq - count * mean**2) / (count - 1)
            std = np.where(count > 1, np.sqrt(np.clip(var, 0.0, None)), np.nan)
        order, clusters = _cluster(mean, threshold)
        ordered = [names[i] for i in order]

        def frame(values):
            return pd.DataFrame(
                values[np.ix_(order, order)], index=ordered, columns=ordered
            )

        return cls(
            mean=frame(mean),
            std=frame(std),
            periods=frame(count.astype(np.int64)),
            clusters=pd.Series(clusters, index=ordered, name="cluster"),
            threshold=float(threshold),
        )

    @property
    def summary(self) -> dict[str, Any]:
        """Counts that describe the matrix at a glance.

        ``n_factors``; ``n_clusters``; ``n_pairs_above_threshold``, the
        pairs whose ``|mean|`` reaches ``threshold``; ``mean_abs_correlation``
        over all pairs; and ``threshold``.

        Examples
        --------
        >>> corr.summary["n_clusters"], corr.summary["n_pairs_above_threshold"]
        (2, 1)
        """
        pairs = self.pairs_table()
        strength = pairs["mean"].abs()
        return {
            "n_factors": int(len(self.mean)),
            "n_clusters": int(self.clusters.nunique()),
            "n_pairs_above_threshold": int((strength >= self.threshold).sum()),
            "mean_abs_correlation": float(strength.mean()),
            "threshold": self.threshold,
        }

    def pairs_table(self) -> pd.DataFrame:
        """Return one row per pair of variables, strongest ``|mean|`` first.

        Columns are ``factor_a``, ``factor_b``, ``mean``, ``std``,
        ``periods`` and ``same_cluster``. Pairs with no usable timestamp
        come last.

        Examples
        --------
        >>> corr.pairs_table()[["factor_a", "factor_b", "mean"]].round(2)
          factor_a factor_b  mean
        0        a    a_neg -1.00
        1    noise        a -0.01
        2    noise    a_neg  0.01
        """
        names = list(self.mean.index)
        rows, cols = np.triu_indices(len(names), k=1)
        mean = self.mean.to_numpy()[rows, cols]
        table = pd.DataFrame(
            {
                "factor_a": [names[i] for i in rows],
                "factor_b": [names[j] for j in cols],
                "mean": mean,
                "std": self.std.to_numpy()[rows, cols],
                "periods": self.periods.to_numpy()[rows, cols],
                "same_cluster": self.clusters.to_numpy()[rows]
                == self.clusters.to_numpy()[cols],
            }
        )
        strength = np.nan_to_num(np.abs(mean), nan=-1.0)
        return table.iloc[np.argsort(-strength, kind="stable")].reset_index(drop=True)

    def cluster_summary(self) -> pd.DataFrame:
        """Return one row per cluster of two or more variables, largest first.

        Columns are ``cluster``, ``size``, ``mean_abs_correlation`` (the
        mean ``|mean|`` over the pairs inside the cluster) and ``members``
        (the variable names, in matrix order).

        Examples
        --------
        >>> corr.cluster_summary()
           cluster  size  mean_abs_correlation     members
        0        2     2                   1.0  [a, a_neg]
        """
        values = np.abs(self.mean.to_numpy())
        rows = []
        for cluster, members in self.clusters.groupby(self.clusters, sort=False):
            if len(members) < 2:
                continue
            idx = [self.mean.index.get_loc(name) for name in members.index]
            inner = values[np.ix_(idx, idx)][np.triu_indices(len(idx), k=1)]
            rows.append({
                "cluster": int(cluster),
                "size": len(idx),
                "mean_abs_correlation": float(np.nanmean(inner)) if np.isfinite(inner).any() else np.nan,
                "members": list(members.index),
            })
        table = pd.DataFrame(rows, columns=["cluster", "size", "mean_abs_correlation", "members"])
        return table.sort_values(["size", "cluster"], ascending=[False, True], kind="stable").reset_index(drop=True)

    def cluster_table(self) -> pd.DataFrame:
        """Return each variable's cluster, the cluster size and its position.

        Columns are ``factor``, ``cluster``, ``cluster_size`` and
        ``position`` (0-based, along the matrix order).

        Examples
        --------
        >>> corr.cluster_table()
          factor  cluster  cluster_size  position
        0  noise        1             1         0
        1      a        2             2         1
        2  a_neg        2             2         2
        """
        sizes = self.clusters.map(self.clusters.value_counts())
        return pd.DataFrame(
            {
                "factor": self.clusters.index,
                "cluster": self.clusters.to_numpy(),
                "cluster_size": sizes.to_numpy(),
                "position": np.arange(len(self.clusters)),
            }
        )


def _block_correlations(block: np.ndarray) -> np.ndarray:
    """Return ``[time, factor, factor]`` rank correlations of a ``[time, symbol, factor]`` block.

    Each factor is ranked among its own finite symbols; a pair's Pearson
    correlation of those ranks uses the symbols finite in both, from
    masked sums, so every pair of every timestamp is a few batched matrix
    products.
    """
    finite = np.isfinite(block)
    ranks = stats.rankdata(np.where(finite, block, np.nan), axis=1, nan_policy="omit")
    mask = finite.astype(np.float64)
    x = np.where(finite, ranks, 0.0)
    xt = x.transpose(0, 2, 1)
    n = mask.transpose(0, 2, 1) @ mask
    sx = xt @ mask
    sxx = (xt * xt) @ mask
    sxy = xt @ x
    sy, syy = sx.transpose(0, 2, 1), sxx.transpose(0, 2, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        var_x = sxx - sx * sx / n
        var_y = syy - sy * sy / n
        corr = cov / np.sqrt(var_x * var_y)
    usable = (n >= MIN_SYMBOLS) & (var_x > 1e-12) & (var_y > 1e-12)
    return np.where(usable, np.clip(corr, -1.0, 1.0), np.nan)


def _cluster(mean: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the leaf order and the cluster numbers along it.

    The distance is ``1 - |mean|``, with a pair that has no correlation
    treated as unrelated. Clusters are numbered in order of appearance.
    """
    distance = 1.0 - np.nan_to_num(np.abs(mean), nan=0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    tree = linkage(squareform(distance, checks=False), method="average")
    order = leaves_list(tree)
    labels = fcluster(tree, t=1.0 - threshold, criterion="distance")[order]
    renumber: dict[int, int] = {}
    numbered = np.array([renumber.setdefault(c, len(renumber) + 1) for c in labels])
    return order, numbered


_RED = "#e34948"
_BLUE = "#2a78d6"
_NEUTRAL = "#b8b7b2"
_NEUTRAL_LIGHT = "#f0efec"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_GRID = "#e6e5e0"
_SURFACE = "#fcfcfb"


class FactorCorrelationFigure:
    """Draw a ``FactorCorrelation`` so that hundreds of variables stay readable.

    The heatmap shows the mean correlation in cluster order on a fixed
    ``-1..1`` diverging scale (red negative, blue positive), with every
    cluster of two or more variables outlined. Up to ``label_limit``
    variables every row and column is named; beyond that the axes name the
    clusters of two or more variables instead, and ``factor_clusters.csv``
    maps every variable to its cluster and position. Beside the heatmap, the
    strongest pairs are listed by name as bars, the largest clusters are
    listed with their size, mean inner ``|correlation|`` and first
    members, and a histogram shows how the correlations of all pairs are
    distributed against the threshold.

    Parameters
    ----------
    label_limit : int, default 60
        Most variables whose names are written on the heatmap axes.
    top_pairs : int, default 25
        Strongest pairs listed beside the heatmap.
    top_clusters : int, default 15
        Largest clusters listed beside the heatmap.

    Examples
    --------
    >>> fig = FactorCorrelationFigure().render(corr)
    >>> fig.savefig("factor_correlation.png")
    """

    def __init__(self, label_limit: int = 60, top_pairs: int = 25, top_clusters: int = 15):
        """Initialize the renderer; see the class docstring for parameters."""
        self.label_limit = int(label_limit)
        self.top_pairs = int(top_pairs)
        self.top_clusters = int(top_clusters)

    def render(self, corr: FactorCorrelation) -> "Figure":
        """Return the figure of ``corr``.

        Built with ``matplotlib.figure.Figure`` rather than ``pyplot``, so
        it is never shown and needs no closing.

        Examples
        --------
        >>> FactorCorrelationFigure().render(corr).get_suptitle()
        'Factor correlation   |   3 factors, 2 clusters at |corr| >= 0.7'
        """
        from matplotlib.figure import Figure

        n = len(corr.mean)
        side = float(np.clip(7.0 + 0.03 * n, 8.0, 22.0))
        fig = Figure(figsize=(side + 9.0, max(side, 14.0) + 1.0), facecolor=_SURFACE,
                     layout="constrained")
        grid = fig.add_gridspec(3, 2, width_ratios=[side, 8.5], height_ratios=[1.2, 1.0, 0.8])
        summary = corr.summary
        fig.suptitle(
            f"Factor correlation   |   {summary['n_factors']} factors, "
            f"{summary['n_clusters']} clusters at |corr| >= {corr.threshold:g}",
            fontsize=18, fontweight="bold", color=_INK,
        )
        self._heatmap(fig, fig.add_subplot(grid[:, 0]), corr)
        self._top_pairs(fig.add_subplot(grid[0, 1]), corr)
        self._top_clusters(fig.add_subplot(grid[1, 1]), corr)
        self._histogram(fig.add_subplot(grid[2, 1]), corr)
        return fig

    @staticmethod
    def _cmap():
        """Diverging map: red at -1, light neutral at 0, blue at +1."""
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list("corr", [_RED, _NEUTRAL_LIGHT, _BLUE])

    @staticmethod
    def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
        """Recessive spines and grid, left-aligned bold title."""
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold", color=_INK)
        ax.set_xlabel(xlabel, color=_INK_SECONDARY)
        ax.set_ylabel(ylabel, color=_INK_SECONDARY)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_NEUTRAL)

    def _heatmap(self, fig, ax, corr: FactorCorrelation) -> None:
        """The ordered matrix with clusters outlined and readable axis labels."""
        from matplotlib.patches import Rectangle

        n = len(corr.mean)
        self._style(ax, "Mean rank correlation, clustered")
        values = corr.mean.to_numpy()
        image = ax.imshow(
            np.ma.masked_invalid(values), cmap=self._cmap(), vmin=-1.0, vmax=1.0,
            interpolation="nearest", aspect="equal",
        )
        clusters = corr.clusters.to_numpy()
        starts = np.flatnonzero(np.r_[True, clusters[1:] != clusters[:-1]])
        ends = np.r_[starts[1:], n]
        edge = max(0.6, 2.0 - n / 200)
        for start, end in zip(starts, ends):
            if end - start >= 2:
                ax.add_patch(Rectangle(
                    (start - 0.5, start - 0.5), end - start, end - start,
                    fill=False, edgecolor=_INK, linewidth=edge,
                ))
        if n <= self.label_limit:
            size = float(np.clip(260 / n, 5, 10))
            names = list(corr.mean.index)
            ax.set_xticks(range(n), names, rotation=90, fontsize=size)
            ax.set_yticks(range(n), names, fontsize=size)
            if n <= 15:
                for (row, col), value in np.ndenumerate(values):
                    if np.isfinite(value) and row != col:
                        ax.text(col, row, f"{value:.2f}", ha="center", va="center",
                                fontsize=8, color=_INK)
        else:
            groups = [(s, e) for s, e in zip(starts, ends) if e - s >= 2]
            groups = sorted(groups, key=lambda g: g[0] - g[1])[: self.label_limit]
            groups.sort()
            ticks = [(s + e - 1) / 2 for s, e in groups]
            labels = [f"C{clusters[s]} ({e - s})" for s, e in groups]
            ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
            ax.set_yticks(ticks, labels, fontsize=7)
            ax.set_xlabel(
                "clusters of 2+ factors; every factor's cluster and position "
                "is in factor_clusters.csv",
                color=_INK_SECONDARY,
            )
        ax.tick_params(length=0)
        fig.colorbar(image, ax=ax, shrink=0.6, label="mean Spearman correlation")

    def _top_pairs(self, ax, corr: FactorCorrelation) -> None:
        """The strongest pairs by ``|mean|``, named, as horizontal bars."""
        pairs = corr.pairs_table().dropna(subset=["mean"]).head(self.top_pairs)
        self._style(ax, f"Strongest {len(pairs)} pairs", "mean correlation")
        if pairs.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=_INK_SECONDARY, transform=ax.transAxes)
            return
        rows = np.arange(len(pairs))[::-1]
        colors = [_BLUE if v >= 0 else _RED for v in pairs["mean"]]
        ax.barh(rows, pairs["mean"], color=colors, height=0.7)
        ax.set_yticks(rows, [
            f"{_short(a)}  ×  {_short(b)}" for a, b in zip(pairs["factor_a"], pairs["factor_b"])
        ], fontsize=8)
        ax.set_xlim(-1.0, 1.0)
        ax.axvline(0.0, color=_INK_SECONDARY, linewidth=1)
        for x in (-corr.threshold, corr.threshold):
            ax.axvline(x, color=_NEUTRAL, linewidth=1, linestyle="--")
        ax.grid(True, axis="x", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def _top_clusters(self, ax, corr: FactorCorrelation) -> None:
        """The largest clusters: size as bars, members and inner |corr| as labels."""
        clusters = corr.cluster_summary().head(self.top_clusters)
        self._style(ax, f"Largest {len(clusters)} clusters", "factors in cluster")
        if clusters.empty:
            ax.text(0.5, 0.5, f"no two factors reach |corr| >= {corr.threshold:g}",
                    ha="center", va="center", color=_INK_SECONDARY, transform=ax.transAxes)
            ax.set_yticks([])
            return
        rows = np.arange(len(clusters))[::-1]
        ax.barh(rows, clusters["size"], color=_BLUE, height=0.7)
        ax.set_yticks(rows, [f"C{c}" for c in clusters["cluster"]], fontsize=8)
        largest = int(clusters["size"].max())
        # The right part of the axis holds each bar's label.
        ax.set_xlim(0, largest * 2.6)
        ax.set_xticks([t for t in ax.get_xticks() if 0 <= t <= largest])
        for row, (_, item) in zip(rows, clusters.iterrows()):
            shown = ", ".join(_short(m, 14) for m in item["members"][:3])
            more = len(item["members"]) - 3
            text = f" {item['size']}  |corr| {item['mean_abs_correlation']:.2f}  {shown}" + (
                f" +{more}" if more > 0 else ""
            )
            ax.text(item["size"], row, text, va="center", fontsize=7.5, color=_INK,
                    clip_on=True)
        ax.grid(True, axis="x", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    def _histogram(self, ax, corr: FactorCorrelation) -> None:
        """Distribution of every pair's mean correlation, threshold marked."""
        values = corr.pairs_table()["mean"].dropna().to_numpy()
        summary = corr.summary
        self._style(ax, "All pairs", "mean correlation", "pairs")
        if values.size == 0:
            return
        ax.hist(values, bins=np.linspace(-1.0, 1.0, 41), color=_NEUTRAL, edgecolor=_SURFACE)
        for x in (-corr.threshold, corr.threshold):
            ax.axvline(x, color=_INK_SECONDARY, linewidth=1, linestyle="--")
        ax.set_xlim(-1.0, 1.0)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.text(
            0.02, 0.97,
            f"{summary['n_pairs_above_threshold']} of {values.size} pairs "
            f"at |corr| >= {corr.threshold:g}\nmean |corr| "
            f"{summary['mean_abs_correlation']:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, color=_INK,
        )


def _short(name: str, limit: int = 22) -> str:
    """Shorten a long factor name for a tick label."""
    return name if len(name) <= limit else name[: limit - 1] + "…"
