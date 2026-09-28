"""Stopping-rule contract of the torch model variant.

A deep head declares a ``StoppingRule`` (``DLModel.stopping``); at the start
of every fit ``DLModel`` asks it for a fresh ``EpochMonitor``, feeds the
monitor each epoch's mean training and validation losses, and lets it choose
the weights that are kept. The concrete rules, ``ValLossPatience`` and
``TrainLossThreshold``, live in ``quantlab.dl_model.training``.
"""

from abc import ABC, abstractmethod

import torch


class EpochMonitor(ABC):
    """The per-fit state of a stopping rule.

    ``DLModel`` asks ``StoppingRule.monitor()`` for a fresh one at the start
    of every fit, calls ``update`` after each epoch and ``finish`` once the
    loop ends, so a rule declared once on a head class is never shared
    between fits or cross-validation folds.

    Examples
    --------
    A monitor that stops after a fixed number of epochs::

        >>> class Fixed(EpochMonitor):
        ...     def __init__(self, epochs):
        ...         self.left = epochs
        ...     def update(self, train_loss, val_loss, model):
        ...         self.left -= 1
        ...         return self.left == 0
        >>> monitor = Fixed(2)
        >>> [monitor.update(1.0, None, torch.nn.Linear(1, 1)) for _ in range(2)]
        [False, True]
    """

    @abstractmethod
    def update(
        self, train_loss: float, val_loss: float | None, model: torch.nn.Module
    ) -> bool:
        """Record a finished epoch; return True to stop training.

        Parameters
        ----------
        train_loss : float
            Mean of the head's training-step losses over the epoch.
        val_loss : float or None
            Mean of the head's validation losses, None without a validation
            segment.
        model : torch.nn.Module
            The network, for monitors that snapshot weights.

        Returns
        -------
        bool
            True to stop after this epoch.

        Examples
        --------
        >>> from quantlab.dl_model.training import TrainLossThreshold
        >>> TrainLossThreshold(1.0, 10).monitor().update(0.5, None, torch.nn.Linear(1, 1))
        True
        """

    def finish(self, model: torch.nn.Module) -> None:
        """Load the chosen weights into ``model``; the default keeps the last.

        Examples
        --------
        >>> from quantlab.dl_model.training import ValLossPatience
        >>> net = torch.nn.Linear(1, 1)
        >>> monitor = ValLossPatience(1).monitor()
        >>> _ = monitor.update(0.0, 1.0, net)
        >>> monitor.finish(net)   # restores the weights of the best epoch
        """


class StoppingRule(ABC):
    """A head's declared rule for when training stops and which weights it keeps.

    Examples
    --------
    >>> from quantlab.dl_model.training import ValLossPatience
    >>> isinstance(ValLossPatience(3), StoppingRule)
    True
    """

    @abstractmethod
    def monitor(self) -> EpochMonitor:
        """Return fresh per-fit state for this rule.

        Examples
        --------
        >>> from quantlab.dl_model.training import ValLossPatience
        >>> ValLossPatience(3).monitor() is ValLossPatience(3).monitor()
        False
        """
