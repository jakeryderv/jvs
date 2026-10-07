"""Explicit wrapper-to-wrapper conversion using the existing checked scalar APIs."""

from __future__ import annotations

from .complex import ComplexValue
from .dtype import ExactDType, FixedDType, NumericKind, NumPyDType
from .fixed import FixedValue
from .floating import FloatingValue
from .integer import IntegerValue
from .rational import RationalValue

type NumericValue = (
    IntegerValue | RationalValue | FloatingValue | ComplexValue | FixedValue
)
type ScalarDType = NumPyDType | ExactDType | FixedDType


def cast(
    value: NumericValue, dtype: ScalarDType, *, approximate: bool = False
) -> NumericValue:
    """Convert a wrapper, preserving its stored value unless rounding is requested.

    Approximation is available for floating/complex/fixed targets. Integer targets
    always require integrality and range checks. Real targets require zero
    imaginary parts. Integer/rational targets discard zero signs and rounding history;
    floating-to-rational conversion extracts the stored binary ratio.
    """
    if not isinstance(
        value, (IntegerValue, RationalValue, FloatingValue, ComplexValue, FixedValue)
    ):
        raise TypeError(
            "cast requires a numeric value wrapper; construct raw inputs explicitly"
        )
    if not isinstance(dtype, (NumPyDType, ExactDType, FixedDType)):
        raise TypeError(
            "cast requires an explicit NumPyDType, ExactDType, or FixedDType"
        )
    if not isinstance(approximate, bool):
        raise TypeError("approximate must be a bool")
    if approximate and dtype.kind not in (
        NumericKind.FLOATING,
        NumericKind.COMPLEX,
        NumericKind.FIXED,
    ):
        raise ValueError(
            "Approximation is supported only for floating, complex, or fixed-point targets"
        )

    if isinstance(dtype, NumPyDType) and dtype.kind is NumericKind.COMPLEX:
        if isinstance(value, ComplexValue):
            return value.to(dtype, approximate=approximate)
        constructor = ComplexValue.approx if approximate else ComplexValue
        return constructor(value, dtype=dtype)

    if isinstance(value, ComplexValue):
        value = value.to_real()

    if isinstance(dtype, FixedDType):
        return (
            FixedValue.approx(value, dtype=dtype)
            if approximate
            else FixedValue(value, dtype=dtype)
        )

    if dtype.kind is NumericKind.INTEGER:
        if isinstance(value, IntegerValue):
            return value.to(dtype)
        return value.to_integer(dtype)

    if isinstance(dtype, ExactDType):
        # ExactDType supports only INTEGER (handled above) and RATIONAL.
        if isinstance(value, IntegerValue):
            return RationalValue(int(value), dtype=dtype)
        if isinstance(value, RationalValue):
            return value
        return value.to_rational()

    if dtype.kind is NumericKind.FLOATING:
        constructor = FloatingValue.approx if approximate else FloatingValue
        return constructor(value, dtype=dtype)
    raise TypeError("Unsupported scalar target dtype")
