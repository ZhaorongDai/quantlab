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


# Chart chrome and ink of the reference palette.
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_SECONDARY = "#52514e"
_INK_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_BLUE = "#2a78d6"
_RED = "#e34948"
#: Diverging ramp for correlations: red arm, neutral gray midpoint, blue arm,
#: with the same number of steps and matching lightness per arm.
_DIVERGING = [
    "#8f2a2a", "#c23b3a", "#e8736f", "#f4b4ae",
    "#f0efec",
    "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95",
]


class FactorCorrelationFigure:
    """Draw a ``FactorCorrelation`` so that hundreds of variables stay readable.

    The main panel is the mean correlation in cluster order on a fixed
    ``-1..1`` diverging scale (red negative, gray none, blue positive), so
    redundant factors form blocks on the diagonal; every cluster of two or
    more variables is outlined, and the diagonal, always 1, is left blank.
    Up to ``label_limit`` variables every row and column is named; beyond
    that the axes name the clusters of two or more variables, and
    ``factor_clusters.csv`` maps every variable to its cluster and
    position. Beside it: the strongest pairs by ``|correlation|``, the
    largest clusters with their first members, and the distribution of all
    pairs on a log count scale so the few strong pairs stay visible.

    Parameters
    ----------
    label_limit : int, default 60
        Most variables whose names are written on the heatmap axes.
    top_pairs : int, default 20
        Strongest pairs listed beside the heatmap.
    top_clusters : int, default 12
        Largest clusters listed beside the heatmap.

    Examples
    --------
    >>> fig = FactorCorrelationFigure().render(corr)
    >>> fig.savefig("factor_correlation.png")
    """

    def __init__(self, label_limit: int = 60, top_pairs: int = 20, top_clusters: int = 12):
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
        named = n <= self.label_limit
        # Layout in inches: a square heatmap under a header, and a right
        # column of three panels spanning the same height.
        side = float(np.clip(6.0 + 0.035 * n, 7.5, 16.0))
        left, bottom = (1.5, 1.5) if named else (0.7, 1.0)
        header, gap, right_w, margin = 1.55, 2.3, 5.6, 0.35
        height = max(header + side + bottom, 11.0)
        width = left + side + gap + right_w + margin
        fig = Figure(figsize=(width, height), facecolor=_SURFACE)

        def box(x, y_top, w, h):
            return [x / width, 1 - (y_top + h) / height, w / width, h / height]

        summary = corr.summary
        fig.suptitle(x=0.35 / width, y=1 - 0.3 / height, t=
                 f"Factor correlation   |   {summary['n_factors']} factors, "
                 f"{summary['n_clusters']} clusters at |corr| >= {corr.threshold:g}",
                 ha="left", va="top", fontsize=16, fontweight="bold", color=_INK)
        fig.text(0.35 / width, 1 - 0.72 / height,
                 f"Mean cross-sectional rank correlation over "
                 f"{int(corr.periods.to_numpy().max())} periods; "
                 f"{summary['n_pairs_above_threshold']} of {n * (n - 1) // 2} pairs at "
                 f"|corr| >= {corr.threshold:g}, mean |corr| "
                 f"{summary['mean_abs_correlation']:.2f}",
                 ha="left", va="top", fontsize=10, color=_INK_SECONDARY)
        heat = fig.add_axes(box(left, header, side, side))
        legend = fig.add_axes(box(left + side * 0.6, header - 0.42, side * 0.4, 0.12))
        self._heatmap(heat, legend, corr)

        column_h = height - header - 0.8
        x = left + side + gap
        spacing = 0.95
        heights = np.array([0.46, 0.31, 0.23]) * (column_h - 2 * spacing)
        tops = [header, header + heights[0] + spacing,
                header + heights[0] + heights[1] + 2 * spacing]
        self._top_pairs(fig.add_axes(box(x, tops[0], right_w, heights[0])), corr,
                        rows=int(max(5, min(self.top_pairs, heights[0] / 0.27))))
        self._top_clusters(fig.add_axes(box(x, tops[1], right_w, heights[1])), corr,
                           rows=int(max(3, min(self.top_clusters, heights[1] / 0.3))))
        self._histogram(fig.add_axes(box(x, tops[2], right_w, heights[2])), corr)
        return fig

    @staticmethod
    def _cmap():
        """The diverging ramp as a colormap, gray at 0."""
        from matplotlib.colors import LinearSegmentedColormap

        return LinearSegmentedColormap.from_list("corr", _DIVERGING)

    @staticmethod
    def _style(ax, title: str, xlabel: str = "") -> None:
        """Hairline recessive chrome and a left-aligned title."""
        ax.set_facecolor(_SURFACE)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=_INK, pad=8)
        ax.set_xlabel(xlabel, color=_INK_SECONDARY, fontsize=9)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=8.5, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_AXIS)
        ax.spines["bottom"].set_linewidth(0.8)

    def _heatmap(self, ax, legend, corr: FactorCorrelation) -> None:
        """The ordered matrix, clusters outlined, diagonal blank."""
        from matplotlib.patches import Rectangle

        n = len(corr.mean)
        ax.set_title("Clustered correlation matrix", loc="left", fontsize=12,
                     fontweight="bold", color=_INK, pad=8)
        values = corr.mean.to_numpy().copy()
        np.fill_diagonal(values, np.nan)
        cmap = self._cmap()
        cmap.set_bad(_SURFACE)
        # A surface gap between cells while they are large enough to see it.
        gap = _SURFACE if n <= 40 else "face"
        image = ax.pcolormesh(
            np.ma.masked_invalid(values), cmap=cmap, vmin=-1.0, vmax=1.0,
            edgecolors=gap, linewidth=1.0 if n <= 40 else 0.0, rasterized=n > 40,
        )
        ax.set_xlim(0, n)
        ax.set_ylim(n, 0)
        ax.set_aspect("equal")
        for spine in ax.spines.values():
            spine.set_visible(False)
        clusters = corr.clusters.to_numpy()
        starts = np.flatnonzero(np.r_[True, clusters[1:] != clusters[:-1]])
        ends = np.r_[starts[1:], n]
        edge = float(np.clip(1.6 - n / 300, 0.6, 1.6))
        for start, end in zip(starts, ends):
            if end - start >= 2:
                ax.add_patch(Rectangle(
                    (start, start), end - start, end - start,
                    fill=False, edgecolor=_INK, linewidth=edge,
                ))
        ax.tick_params(colors=_INK_SECONDARY, length=0)
        if n <= self.label_limit:
            size = float(np.clip(300 / n, 5.5, 10))
            names = [_short(name, 24) for name in corr.mean.index]
            ax.set_xticks(np.arange(n) + 0.5, names, rotation=90, fontsize=size)
            ax.set_yticks(np.arange(n) + 0.5, names, fontsize=size)
        else:
            groups = [(s, e) for s, e in zip(starts, ends) if e - s >= 2]
            groups = sorted(groups, key=lambda g: g[0] - g[1])[: self.label_limit]
            groups.sort()
            ticks = [(s + e) / 2 for s, e in groups]
            labels = [f"C{clusters[s]}" for s, e in groups]
            ax.set_xticks(ticks, labels, rotation=90, fontsize=7.5)
            ax.set_yticks(ticks, labels, fontsize=7.5)
            ax.set_xlabel(
                "Outlined: clusters of 2+ factors, labelled C<n>. Each factor's cluster "
                "and position: factor_clusters.csv",
                color=_INK_MUTED, fontsize=9,
            )
        bar = ax.figure.colorbar(image, cax=legend, orientation="horizontal")
        bar.set_ticks([-1, -0.5, 0, 0.5, 1])
        bar.ax.tick_params(labelsize=8, colors=_INK_SECONDARY, length=0)
        bar.ax.xaxis.set_ticks_position("top")
        bar.outline.set_visible(False)

    def _top_pairs(self, ax, corr: FactorCorrelation, rows: int) -> None:
        """The strongest pairs: ``|corr|`` as thin bars, sign as color, value labelled."""
        pairs = corr.pairs_table().dropna(subset=["mean"]).head(rows)
        self._style(ax, f"Strongest {len(pairs)} pairs", "|mean correlation|")
        if pairs.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    color=_INK_MUTED, transform=ax.transAxes)
            return
        rows = np.arange(len(pairs))[::-1]
        strength = pairs["mean"].abs().to_numpy()
        colors = [_BLUE if v >= 0 else _RED for v in pairs["mean"]]
        ax.barh(rows, strength, color=colors, height=0.56)
        for row, value, s in zip(rows, pairs["mean"], strength):
            ax.text(s + 0.015, row, f"{value:+.2f}", va="center", fontsize=8,
                    color=_INK_SECONDARY)
        ax.set_yticks(rows, [
            f"{_short(a, 18)}  ·  {_short(b, 18)}"
            for a, b in zip(pairs["factor_a"], pairs["factor_b"])
        ], fontsize=8)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylim(-0.7, len(pairs) - 0.3)
        ax.axvline(corr.threshold, color=_INK_MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.grid(True, axis="x", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.text(1.0, 1.02, "blue positive · red negative", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=_INK_MUTED)

    def _top_clusters(self, ax, corr: FactorCorrelation, rows: int) -> None:
        """The largest clusters: size as thin bars, inner |corr| and members as columns.

        The bars take the left third of the axis; the columns to their right
        are placed in axis fractions, so they never overlap the bars.
        """
        from matplotlib.transforms import blended_transform_factory

        clusters = corr.cluster_summary().head(rows)
        self._style(ax, f"Largest {len(clusters)} clusters", "")
        if clusters.empty:
            ax.text(0.5, 0.5, f"no two factors reach |corr| >= {corr.threshold:g}",
                    ha="center", va="center", color=_INK_MUTED, transform=ax.transAxes)
            ax.set_yticks([])
            ax.set_xticks([])
            return
        slots = max(rows, len(clusters))
        positions = np.arange(len(clusters))
        largest = int(clusters["size"].max())
        ax.barh(positions, clusters["size"], color=_BLUE, height=0.56)
        ax.set_yticks(positions, [f"C{c}" for c in clusters["cluster"]], fontsize=8)
        ax.set_ylim(slots - 0.4, -1.1)
        ax.set_xlim(0, largest / 0.3)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        text = blended_transform_factory(ax.transAxes, ax.transData)
        for column, heading in ((0.32, "size"), (0.43, "|corr|"), (0.56, "members")):
            ax.text(column, -0.95, heading, transform=text, fontsize=7.5,
                    color=_INK_MUTED, va="center")
        for row, (_, item) in zip(positions, clusters.iterrows()):
            shown = ", ".join(_short(m, 12) for m in item["members"][:3])
            more = len(item["members"]) - 3
            ax.text(0.32, row, f"{item['size']}", transform=text, va="center",
                    fontsize=8, color=_INK)
            ax.text(0.43, row, f"{item['mean_abs_correlation']:.2f}", transform=text,
                    va="center", fontsize=8, color=_INK)
            ax.text(0.56, row, shown + (f" +{more}" if more > 0 else ""), transform=text,
                    va="center", fontsize=7.5, color=_INK_SECONDARY, clip_on=True)

    def _histogram(self, ax, corr: FactorCorrelation) -> None:
        """Every pair's mean correlation, on a log count scale, threshold shaded."""
        values = corr.pairs_table()["mean"].dropna().to_numpy()
        self._style(ax, "All pairs", "mean correlation")
        if values.size == 0:
            return
        edges = np.linspace(-1.0, 1.0, 41)
        counts, _ = np.histogram(values, bins=edges)
        centers = (edges[:-1] + edges[1:]) / 2
        colors = [
            _RED if c <= -corr.threshold else _BLUE if c >= corr.threshold else "#b8b7b2"
            for c in centers
        ]
        ax.bar(centers, np.where(counts > 0, counts, np.nan), width=0.045,
               color=colors, bottom=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(counts.max(), 1) * 3)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylabel("pairs (log)", color=_INK_SECONDARY, fontsize=9)
        ax.grid(True, axis="y", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def _short(name: str, limit: int = 22) -> str:
    """Shorten a long factor name for a tick label."""
    return name if len(name) <= limit else name[: limit - 1] + "…"
