"""joblib-backed checkpoint storage for non-torch model heads.

``MlBackend`` implements the ``ModelBackend`` contract and is how ``MLModel``
(the base class for tree and tabular models such as XGBoost) writes and reads
its ``.joblib`` checkpoints. It holds one model object and knows only how to
dump it to a path and load it back. It has no notion of panels, dimensions or
coordinates. joblib files are pickles, which can run arbitrary code when
loaded, so only load files you trust.
"""

import joblib
from pathlib import Path
from typing import Self

from quantlab.base.backend import ModelBackend


class MlBackend(ModelBackend):
    """Persist one model object with joblib.

    ``write``, ``read`` and ``to_internal`` all return ``self`` so calls can
    be chained. ``write`` creates missing parent directories. The constructor
    takes no arguments; the backend is empty until ``read`` or
    ``to_internal`` gives it a model.

    Examples
    --------
    >>> MlBackend().to_internal({"coef": 2.5}).write("ckpt/model.joblib")
    MlBackend()
    >>> MlBackend().read("ckpt/model.joblib").get_model()
    {'coef': 2.5}
    """

    def get_model(self):
        """Return the held model object.

        Raises
        ------
        AttributeError
            If nothing has been loaded with ``read`` or ``to_internal`` yet.

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5}).get_model()
        {'coef': 2.5}
        """
        return self.model

    def write(self, path: str, **kwargs) -> Self:
        """Dump the held model to ``path`` with ``joblib.dump`` and return ``self``.

        Missing parent directories of ``path`` are created.

        Parameters
        ----------
        path : str
            Destination file, conventionally ending in ``.joblib``.
        **kwargs
            Forwarded to ``joblib.dump``, for example ``compress=3``.

        Returns
        -------
        MlBackend
            This backend, for chaining.

        Examples
        --------
        >>> backend = MlBackend().to_internal({"coef": 2.5})
        >>> backend.write("ckpt/model.joblib", compress=3)
        MlBackend()
        """
        if not Path(path).parent.exists():
            Path(path).parent.mkdir(parents=True)
        joblib.dump(self.model, path, **kwargs)
        return self

    def read(self, path: str, **kwargs) -> Self:
        """Load the model at ``path`` with ``joblib.load`` and return ``self``.

        Parameters
        ----------
        path : str
            A file previously written by ``write`` or ``joblib.dump``.
        **kwargs
            Forwarded to ``joblib.load``.

        Returns
        -------
        MlBackend
            This backend, now holding the loaded model.

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.

        Examples
        --------
        >>> MlBackend().read("ckpt/model.joblib").get_model()
        {'coef': 2.5}
        """
        self.model = joblib.load(path, **kwargs)
        return self

    def to_internal(self, model) -> Self:
        """Adopt an in-memory model object and return ``self``.

        Parameters
        ----------
        model : object
            Any picklable object, typically a fitted estimator.

        Returns
        -------
        MlBackend
            This backend, now holding ``model``.

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5})
        MlBackend()
        """
        self.model = model
        return self
