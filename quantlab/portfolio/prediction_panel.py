"""The prediction panel file of a backtest run, and the rule rebuilt from a run.

A backtest run with a model writes the predictions its portfolio
construction rule read into ``predictions.zarr``: one variable per label on
``(timestamp, symbol)``, with ``attrs`` holding ``format_version`` and
``labels``, a JSON list of the labels' ``LabelSpec`` fields. With that file
and the run's ``config.json`` an executor outside the research pipeline (an
event-driven backtest, a live trader) rebuilds the run's rule, bound to the
same label specs, and feeds it the same predictions without loading the
model, factor or label layers: ``load_constructor`` does exactly that.

``PredictionPanel`` itself, with its ``write`` and ``read``, is defined in
``quantlab.base.portfolio``, so that the backtester (in the base layer) can
write the file; it is importable from here too. This module imports only the
portfolio base and the class loader, never the model, factor, label or
backtest layers.
"""

import json
from os import PathLike
from pathlib import Path

from quantlab.base.portfolio import PortfolioConstructor, PredictionPanel
from quantlab.utils.module import get_cls_from_path


def load_constructor(run_dir: str | PathLike) -> PortfolioConstructor:
    """Rebuild a backtest run's portfolio construction rule, bound to its label specs.

    The rule is rebuilt from ``config.json["constructor"]`` by the
    ``from_config`` of the class it names and bound to the label specs of
    the run's ``predictions.zarr``; the predictions themselves are not
    read. Neither the model nor the factor or label layers are imported.

    Parameters
    ----------
    run_dir : str or os.PathLike
        A run directory written by ``run()`` or ``run_cv()``.

    Returns
    -------
    PortfolioConstructor
        The bound rule, equal to the one the run used.

    Raises
    ------
    FileNotFoundError
        If the run has no ``config.json`` or no ``predictions.zarr`` (a
        ``run_weights()`` run has no model and so no panel).
    ValueError
        If ``config.json`` records no constructor, or the rule refuses the
        specs.

    Examples
    --------
    With ``run_dir`` the directory of a ``run()`` with a top-2 rule:

    >>> rule = load_constructor(run_dir)
    >>> rule
    TopNConstructor(direction='long_only', top_n=2, score_label=None)
    """
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    recorded = config.get("constructor")
    if recorded is None:
        raise ValueError(f"{run_dir / 'config.json'} records no constructor")
    path = run_dir / PredictionPanel.FILE_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no {PredictionPanel.FILE_NAME}; only a run with a "
            f"model (run() or run_cv()) writes one"
        )
    labels = PredictionPanel.read_labels(path)
    rule = get_cls_from_path(recorded["name"]).from_config(recorded)
    rule.bind(labels)
    return rule
