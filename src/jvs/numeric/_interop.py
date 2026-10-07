"""Shared NumPy equality dispatch without implicit arithmetic or conversion."""

import numpy as np


def scalar_equality_ufunc(
    value: object,
    ufunc: np.ufunc,
    method: str,
    *inputs: object,
    **kwargs: object,
) -> object:
    """Handle scalar equality, deferring unresolved comparisons to other operands."""
    if (
        method != "__call__"
        or kwargs
        or len(inputs) != 2
        or (ufunc is not np.equal and ufunc is not np.not_equal)
    ):
        raise TypeError(
            f"NumPy ufuncs cannot bypass {type(value).__name__} checks; "
            "use wrapper operators or extract .value explicitly"
        )
    other = inputs[1] if inputs[0] is value else inputs[0]
    # NumPy may transport a scalar equality operand as a 0-D integer array.
    # This exception does not enable array construction or arithmetic.
    if isinstance(other, np.ndarray):
        if other.ndim != 0 or other.dtype.kind not in "iu":
            raise TypeError("Equality requires supported scalar operands")
        other = other[()]
    result = value.__eq__(other)
    if result is NotImplemented:
        # Let NumPy try the other operand's implementation, as Python == does.
        # If every implementation defers, NumPy raises TypeError.
        return NotImplemented
    return result if ufunc is np.equal else not result
