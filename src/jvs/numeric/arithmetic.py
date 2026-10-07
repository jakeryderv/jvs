"""Checked arithmetic in an explicitly selected operand representation."""

from __future__ import annotations

import operator
from collections.abc import Callable

from .casting import NumericValue, ScalarDType, cast


def _apply(
    operation: Callable[[NumericValue, NumericValue], NumericValue],
    a: NumericValue,
    b: NumericValue,
    dtype: ScalarDType,
    approximate: bool,
) -> NumericValue:
    left = cast(a, dtype, approximate=approximate)
    right = cast(b, dtype, approximate=approximate)
    return operation(left, right)


def add(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue:
    """Cast operands, then add with checked arithmetic in the requested dtype.

    ``approximate`` controls operand conversion only; floating/complex arithmetic
    retains its normal rounding rules even when this option is false.
    """
    return _apply(operator.add, a, b, dtype, approximate)


def subtract(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue:
    """Cast operands, then subtract b from a; results retain the operand dtype."""
    return _apply(operator.sub, a, b, dtype, approximate)


def multiply(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue:
    """Cast operands, then multiply using the target wrapper's checked rules."""
    return _apply(operator.mul, a, b, dtype, approximate)


def divide(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue:
    """Cast operands, then divide; integer and fixed operands yield exact rationals.

    Other operand families retain their dtype. Zero divisors raise explicitly.
    """
    return _apply(operator.truediv, a, b, dtype, approximate)
