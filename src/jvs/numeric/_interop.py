"""Shared NumPy scalar comparison dispatch without arithmetic or conversion."""

from typing import Protocol

import numpy as np


class _OrderedScalar(Protocol):
    def __lt__(self, other: object) -> bool: ...
    def __le__(self, other: object) -> bool: ...
    def __gt__(self, other: object) -> bool: ...
    def __ge__(self, other: object) -> bool: ...


def scalar_comparison_ufunc(
    value: _OrderedScalar,
    ufunc: np.ufunc,
    method: str,
    *inputs: object,
    **kwargs: object,
) -> object:
    """Preserve equality deferral and the original direction of scalar ordering."""
    if (
        method != "__call__"
        or kwargs
        or len(inputs) != 2
        or ufunc
        not in (
            np.equal,
            np.not_equal,
            np.less,
            np.less_equal,
            np.greater,
            np.greater_equal,
        )
    ):
        raise TypeError(
            f"NumPy ufuncs cannot bypass {type(value).__name__} checks; "
            "use wrapper operators or extract .value explicitly"
        )
    on_left = inputs[0] is value
    other = inputs[1] if on_left else inputs[0]
    # NumPy may transport a scalar comparison operand as a 0-D integer array.
    # This exception does not enable array construction or arithmetic.
    if isinstance(other, np.ndarray):
        if other.ndim != 0 or other.dtype.kind not in "iu":
            raise TypeError("Comparison requires supported scalar operands")
        other = other[()]
    if ufunc is np.less:
        return value.__lt__(other) if on_left else value.__gt__(other)
    if ufunc is np.less_equal:
        return value.__le__(other) if on_left else value.__ge__(other)
    if ufunc is np.greater:
        return value.__gt__(other) if on_left else value.__lt__(other)
    if ufunc is np.greater_equal:
        return value.__ge__(other) if on_left else value.__le__(other)
    result = value.__eq__(other)
    if result is NotImplemented:
        # Let NumPy try the other operand's implementation, as Python == does.
        # If every implementation defers, NumPy raises TypeError.
        return NotImplemented
    return result if ufunc is np.equal else not result
