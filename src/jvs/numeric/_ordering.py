"""Exact real ordering, independent of arithmetic promotion and rounding."""

import numbers
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from types import NotImplementedType

import numpy as np
import sympy as sp


def _real_ratio(value: object) -> Fraction | NotImplementedType:
    # Resolve value classes only when called, avoiding cycles as the wrappers
    # import this shared comparison implementation.
    from .complex import ComplexValue
    from .fixed import FixedValue
    from .floating import FloatingValue
    from .integer import IntegerValue, _input_integer
    from .rational import RationalValue

    if isinstance(value, IntegerValue):
        return Fraction(int(value))
    if isinstance(value, RationalValue):
        return Fraction(value.value.p, value.value.q)
    if isinstance(value, FloatingValue):
        return Fraction(*value.value.as_integer_ratio())
    if isinstance(value, FixedValue):
        return value._ratio()
    if isinstance(value, ComplexValue):
        raise TypeError("ComplexValue is unordered; use .to_real() explicitly")
    if isinstance(value, (numbers.Number, Decimal, np.generic, sp.Basic)):
        return Fraction(_input_integer(value))
    if isinstance(value, np.ndarray):
        raise TypeError("Ordering requires supported scalar operands")
    return NotImplemented


def compare_real(
    left: object, right: object, operation: Callable[[Fraction, Fraction], bool]
) -> bool | NotImplementedType:
    """Compare stored ratios, preserving Python's reflected-dispatch fallback."""
    a, b = _real_ratio(left), _real_ratio(right)
    if a is NotImplemented or b is NotImplemented:
        return NotImplemented
    return operation(a, b)
