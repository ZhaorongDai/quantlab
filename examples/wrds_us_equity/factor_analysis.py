"""Alpha101 and Alpha158 factor analysis on WRDS CRSP daily data.

Data -> Alpha101 + Alpha158 factors -> forward-return label ->
``Factor.analyze()``, an alphalens-style report per factor column: IC
series and distribution, monthly IC, quantile returns, long-short curve and
turnover, drawn as one figure per column and written with the tables and
the configs to an output directory per library. Edit ``Settings`` below and
run ``uv run python examples/wrds_us_equity/factor_analysis.py``, or step
through the ``# %%`` cells. The data root is ``common.DATA_ROOT``.
"""

# %% Settings
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from common import (
    DataSettings,
    Paths,
    check_settings,
    compute_factors,
    factor_objects,
    prepare_stores,
)


@dataclass
class Settings:
    """Everything this pipeline needs; nothing is read from argv."""

    #: Use ``alpha101_names`` / ``alpha158_names`` to analyze a subset; the
    #: whole libraries give 82 + 169 figures.
    data: DataSettings = field(default_factory=DataSettings)
    #: Number of equal-count factor buckets per day.
    quantiles: int = 5
    #: ``False`` reads the stores an earlier run wrote instead of rebuilding
    #: the derived stores and recomputing the factors and the label.
    recompute: bool = True
    #: Report directory; ``None`` means ``<work>/analysis/<library>`` under
    #: the universe's pipeline directory.
    output_dir: str | None = None


SETTINGS = Settings()


# %% Analyze
def analyze(s: Settings, paths: Paths) -> dict:
    """Run ``analyze()`` for each library; returns ``{library: FactorAnalysis}``."""
    factors, labels = factor_objects(s.data, paths)
    for label in labels:
        label.read()
    results = {}
    for factor in factors:
        factor.read()
        library = type(factor).__name__.removesuffix("Stock").lower()
        out = Path(s.output_dir) / library if s.output_dir else paths.work / "analysis" / library
        results[library] = factor.analyze(
            frets=labels, quantiles=s.quantiles, output_dir=str(out)
        )
        table = results[library].summary_table()
        best = table.sort_values("ic_mean", key=abs, ascending=False).head(10)
        logger.info(
            f"{library}: {len(table)} column(s) analyzed -> {out}\n"
            f"{best[['factor', 'fret', 'ic_mean', 'ic_t_stat', 'mean_spread']]}"
        )
    return results


# %% Run everything
def main(s: Settings = SETTINGS) -> dict:
    check_settings(s.data)
    paths = Paths.build(s.data)
    if s.recompute:
        prepare_stores(s.data, paths)
        compute_factors(s.data, paths)
    return analyze(s, paths)


if __name__ == "__main__":
    main()
