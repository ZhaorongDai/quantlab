"""joblib-backed checkpoint persistence for non-torch model heads.

``MlBackend`` implements ``ModelBackend`` and is how ``MLModel`` writes and
reads its ``.joblib`` checkpoints. It holds one model object and knows only
how to dump it to a path and load it back; it has no notion of panels,
dimensions or coordinates. joblib files are pickles, so only load files you
trust.
"""

import joblib
from pathlib import Path
from typing import Self

from quantlab.base.backend import ModelBackend


class MlBackend(ModelBackend):
    """Persist one model object with joblib.

    ``write``, ``read`` and ``to_internal`` all return ``self`` so calls can
    be chained. ``write`` creates missing parent directories.

    Examples
    --------
    >>> MlBackend().to_internal({"coef": 2.5}).write("ckpt/model.joblib")
    MlBackend()
    >>> MlBackend().read("ckpt/model.joblib").get_model()
    {'coef': 2.5}
    """

    def get_model(self):
        """Return the held model object.

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5}).get_model()
        {'coef': 2.5}
        """
        return self.model

    def write(self, path: str, **kwargs) -> Self:
        """Dump the held model to ``path`` with ``joblib.dump`` and return ``self``.

        Missing parent directories of ``path`` are created. Extra keyword
        arguments are forwarded to ``joblib.dump`` (for example
        ``compress=3``).

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

        Extra keyword arguments are forwarded to ``joblib.load``.

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

        Examples
        --------
        >>> MlBackend().to_internal({"coef": 2.5})
        MlBackend()
        """
        self.model = model
        return self
