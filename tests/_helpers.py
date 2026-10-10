"""Test helpers.

``NotImplementedError`` is a subclass of ``RuntimeError``, so a plain
``pytest.raises(RuntimeError)`` is satisfied by a stub that has not been written
yet. :func:`raises_real_runtime_error` closes that hole: it swallows a genuine
``RuntimeError`` and re-raises ``NotImplementedError`` so the surrounding
``xfail(raises=NotImplementedError)`` marker still accounts for the stub.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def raises_real_runtime_error() -> Iterator[None]:
    """Require a RuntimeError that is not merely the stub's NotImplementedError."""
    try:
        yield
    except NotImplementedError:
        raise
    except RuntimeError:
        return
    raise AssertionError("expected a RuntimeError, none was raised")
