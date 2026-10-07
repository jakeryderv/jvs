"""Fixed-point lattices, checked coefficient storage, and explicit rescaling."""

import operator
import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    Backend,
    ComplexValue,
    DType,
    ExactDType,
    FixedDType,
    FixedValue,
    FloatingValue,
    Integer,
    IntegerValue,
    Natural,
    NumericKind,
    NumPyDType,
    PrecisionLossError,
    Rational,
    RationalValue,
    Whole,
    add,
    cast,
    divide,
    multiply,
    subtract,
)

I8 = NumPyDType("int8")
INTEGER = ExactDType.integer()
RATIONAL = ExactDType.rational()
TENTHS = FixedDType(NumPyDType("int16"), step=Fraction(1, 10))
HALVES = FixedDType(I8, step=Fraction(1, 2))


@pytest.mark.parametrize(
    "step", [1, np.uint64(1), sp.Integer(1), sp.Rational(2, 2), Fraction(1)]
)
def test_descriptor_normalizes_exact_steps(step: Any) -> None:
    dtype = FixedDType(I8, step=step)
    assert isinstance(dtype, DType)
    assert dtype == FixedDType(I8, step=Fraction(1))
    assert len({dtype, FixedDType(I8, step=sp.Integer(1))}) == 1
    assert dtype.step == Fraction(1)
    assert dtype.backend is Backend.NUMPY
    assert dtype.kind is NumericKind.FIXED
    assert dtype.storage_bits == 8 and dtype.is_fixed_width
    assert not dtype.supports_nonfinite
    assert "int8" in dtype.name and "1/1" in dtype.name


def test_descriptor_identity_and_unbounded_storage() -> None:
    assert FixedDType(I8, step=sp.Rational(2, 6)) == FixedDType(I8, step=Fraction(1, 3))
    assert FixedDType(I8, step=1) != HALVES
    assert FixedDType(NumPyDType("int16"), step=Fraction(1, 2)) != HALVES
    dtype = FixedDType(INTEGER, step=Fraction(3, 7))
    assert dtype.backend is Backend.SYMPY
    assert dtype.storage_bits is None and not dtype.is_fixed_width
    with pytest.raises(PrecisionLossError):
        FixedValue(1, dtype=dtype)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        mutable: Any = dtype
        mutable.step = Fraction(1)


@pytest.mark.parametrize("step", [0, -1, Fraction(-1, 3), sp.Integer(0)])
def test_step_must_be_positive(step: Any) -> None:
    with pytest.raises(ValueError, match="positive"):
        FixedDType(I8, step=step)


@pytest.mark.parametrize(
    "step",
    [
        True,
        np.bool_(True),
        sp.true,
        0.1,
        np.float64(0.1),
        "0.1",
        Decimal("0.1"),
        sp.Float("0.1"),
        sp.sqrt(2),
        sp.oo,
        sp.nan,
        np.array(1),
        np.timedelta64(1, "ns"),
        object(),
    ],
)
def test_step_requires_explicit_exact_rational_input(step: Any) -> None:
    with pytest.raises(TypeError):
        FixedDType(I8, step=step)


@pytest.mark.parametrize(
    "storage",
    ["int8", None, NumPyDType("float64"), NumPyDType("complex64"), RATIONAL, HALVES],
)
def test_coefficient_dtype_must_be_explicit_integer(storage: Any) -> None:
    with pytest.raises(TypeError):
        FixedDType(storage, step=1)


@pytest.mark.parametrize(
    "step", [Fraction(1, 100), Fraction(1, 256), Fraction(3, 7), Fraction(2)]
)
@pytest.mark.parametrize("storage", [I8, NumPyDType("uint8"), INTEGER])
def test_value_and_coefficient_construction_preserve_exact_lattice(
    step: Fraction, storage: Any
) -> None:
    dtype = FixedDType(storage, step=step)
    for coefficient in range(0 if storage == NumPyDType("uint8") else -128, 128):
        expected = coefficient * step
        a = FixedValue(expected, dtype=dtype)
        b = FixedValue.from_coefficient(coefficient, dtype=dtype)
        assert a == b
        assert int(a.coefficient) == coefficient
        assert a.value == sp.Rational(expected.numerator, expected.denominator)
        assert a.to_rational() == RationalValue(
            expected.numerator, expected.denominator, dtype=RATIONAL
        )
        assert not a.rounded
        assert bool(a) is (coefficient != 0)
        assert isinstance(
            a.coefficient.value, sp.Integer if storage == INTEGER else np.integer
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        sp.true,
        "1.5",
        Decimal("1.5"),
        1 + 0j,
        np.complex64(1),
        sp.Float(1),
        sp.sqrt(2),
        sp.Symbol("x"),
        sp.oo,
        sp.nan,
        np.array(1),
        object(),
    ],
)
def test_unsupported_value_inputs_rejected(value: Any) -> None:
    for constructor in (FixedValue, FixedValue.approx):
        with pytest.raises(TypeError):
            constructor(value, dtype=TENTHS)


@pytest.mark.parametrize(
    "value", [float("inf"), float("-inf"), float("nan"), np.float32("nan")]
)
def test_nonfinite_floats_rejected(value: Any) -> None:
    for constructor in (FixedValue, FixedValue.approx):
        with pytest.raises(ValueError, match="finite"):
            constructor(value, dtype=TENTHS)


@pytest.mark.parametrize(
    "value",
    [
        1.0,
        Fraction(1),
        True,
        np.bool_(True),
        IntegerValue(1, dtype=INTEGER),
        np.array(1),
    ],
)
def test_coefficient_constructor_requires_raw_integer(value: Any) -> None:
    with pytest.raises(TypeError):
        FixedValue.from_coefficient(value, dtype=TENTHS)


@pytest.mark.parametrize("value", [1, np.int64(1), sp.Integer(1)])
def test_value_and_coefficient_are_distinct_inputs(value: Any) -> None:
    assert FixedValue(value, dtype=TENTHS) == 1
    assert int(FixedValue(value, dtype=TENTHS).coefficient) == 10
    assert FixedValue.from_coefficient(value, dtype=TENTHS).value == sp.Rational(1, 10)


def test_raw_float_means_stored_binary_value() -> None:
    with pytest.raises(PrecisionLossError):
        FixedValue(0.1, dtype=TENTHS)
    value = FixedValue.approx(0.1, dtype=TENTHS)
    assert value.value == sp.Rational(1, 10) and value.rounded
    binary_dtype = FixedDType(INTEGER, step=Fraction(1, 2**60))
    exact = FixedValue(0.1, dtype=binary_dtype)
    assert exact.to_rational().value == sp.Rational(*(0.1).as_integer_ratio())
    assert not exact.rounded


@pytest.mark.parametrize("step", [Fraction(1, 10), Fraction(1, 256), Fraction(3, 7)])
def test_quantization_matches_nearest_lattice_oracle(step: Fraction) -> None:
    dtype = FixedDType(I8, step=step)
    # Search neighboring lattice points instead of repeating the implementation's
    # quotient/remainder rounding algorithm; include both signs and halfway ties.
    for eighths in range(-128 * 8, 127 * 8 + 1):
        coefficient = Fraction(eighths, 8)
        exact = coefficient * step
        center = eighths // 8
        expected = min(
            range(center - 1, center + 3), key=lambda n: (abs(n - coefficient), n % 2)
        )
        result = FixedValue.approx(exact, dtype=dtype)
        assert int(result.coefficient) == expected
        assert result.rounded is (coefficient.denominator != 1)
        if coefficient.denominator != 1:
            with pytest.raises(PrecisionLossError):
                FixedValue(exact, dtype=dtype)


@pytest.mark.parametrize("storage", ["int8", "uint8", "int64", "uint64"])
def test_range_checked_before_rounding_and_storage(storage: str) -> None:
    dtype = FixedDType(NumPyDType(storage), step=Fraction(3, 7))
    limits = dtype.coefficient_dtype
    assert isinstance(limits, NumPyDType)
    lo, hi = int(limits.integer_info.min), int(limits.integer_info.max)
    for boundary in (lo, hi):
        assert (
            int(FixedValue(boundary * dtype.step, dtype=dtype).coefficient) == boundary
        )
    for outside in (Fraction(lo) - Fraction(1, 4), Fraction(hi) + Fraction(1, 4)):
        for constructor in (FixedValue, FixedValue.approx):
            with pytest.raises(OverflowError):
                constructor(outside * dtype.step, dtype=dtype)
    for outside in (lo - 1, hi + 1, 10**5000):
        with pytest.raises(OverflowError):
            FixedValue.from_coefficient(outside, dtype=dtype)


def test_rescaling_and_rounding_history() -> None:
    a = FixedValue(Fraction(5, 4), dtype=FixedDType(I8, step=Fraction(1, 4)))
    with pytest.raises(PrecisionLossError):
        a.to(TENTHS)
    b = a.to(TENTHS, approximate=True)
    assert b.value == sp.Rational(6, 5) and b.rounded
    finer = FixedDType(INTEGER, step=Fraction(1, 100))
    assert b.to(finer).rounded
    assert (-b).rounded and (+b) is b
    assert (b - b).rounded and not (b - b)
    assert (b + b).rounded
    with pytest.raises(TypeError, match="bool"):
        b.to(finer, approximate=np.bool_(True))  # ty: ignore[invalid-argument-type]
    for field, replacement in (
        ("rounded", False),
        ("coefficient", 0),
        ("dtype", finer),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(b, field, replacement)


def test_zero_rounding_is_explicit_and_zero_sign_is_discarded() -> None:
    for source in (Fraction(1, 100), Fraction(-1, 100)):
        with pytest.raises(PrecisionLossError):
            FixedValue(source, dtype=TENTHS)
        zero = FixedValue.approx(source, dtype=TENTHS)
        assert not zero and zero.rounded
    value = FixedValue(-0.0, dtype=TENTHS)
    assert not value and not value.rounded
    floating = cast(value, NumPyDType("float64"))
    assert isinstance(floating, FloatingValue)
    assert not np.signbit(floating.value)


def test_addition_subtraction_and_negation_check_coefficient_bounds() -> None:
    low = FixedValue.from_coefficient(-128, dtype=HALVES)
    high = FixedValue.from_coefficient(127, dtype=HALVES)
    unit = FixedValue.from_coefficient(1, dtype=HALVES)
    assert (high - unit).coefficient == 126
    assert (low + unit).coefficient == -127
    assert (high + low).value == sp.Rational(-1, 2)
    for operation in (lambda: high + unit, lambda: low - unit, lambda: -low):
        with pytest.raises(OverflowError):
            operation()
    unsigned = FixedValue.from_coefficient(
        1, dtype=FixedDType(NumPyDType("uint8"), step=1)
    )
    with pytest.raises(OverflowError):
        operator.neg(unsigned)


def test_products_and_quotients_are_exact_rationals_without_coefficient_overflow() -> (
    None
):
    a = FixedValue.from_coefficient(127, dtype=HALVES)
    b = FixedValue.from_coefficient(3, dtype=HALVES)
    assert (a * a).value == sp.Rational(16129, 4)
    assert (a / b).value == sp.Rational(127, 3)
    assert isinstance(a * b, RationalValue)
    assert (a / a).dtype == RATIONAL
    with pytest.raises(ZeroDivisionError):
        a / FixedValue(0, dtype=HALVES)
    rounded = FixedValue.approx(Fraction(1, 3), dtype=HALVES)
    result = rounded * rounded
    assert result.value == sp.Rational(1, 4)
    restored = cast(result, FixedDType(INTEGER, step=Fraction(1, 4)))
    assert isinstance(restored, FixedValue) and not restored.rounded


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_binary_operations_reject_implicit_promotion(operation: Any) -> None:
    a = FixedValue(1, dtype=HALVES)
    for other in (
        1,
        1.0,
        sp.Integer(1),
        IntegerValue(1, dtype=INTEGER),
        FixedValue(1, dtype=TENTHS),
        FixedValue(1, dtype=FixedDType(INTEGER, step=Fraction(1, 2))),
    ):
        for left, right in ((a, other), (other, a)):
            with pytest.raises(TypeError):
                operation(left, right)


@pytest.mark.parametrize("value", [-1, 0, 1, Fraction(1, 2), Fraction(3, 2)])
def test_classification_uses_value_instead_of_coefficient(
    value: int | Fraction,
) -> None:
    a = FixedValue(value, dtype=HALVES)
    ratio = Fraction(value)
    integral = ratio.denominator == 1
    assert a in Rational
    assert (a in Integer) is integral
    assert (a in Whole) is (integral and value >= 0)
    assert (a in Natural) is (integral and value >= 1)


def test_huge_values_and_steps_have_bounded_diagnostics() -> None:
    huge = 10**5000
    before = sys.get_int_max_str_digits()
    dtype = FixedDType(INTEGER, step=Fraction(huge + 1, huge))
    value = FixedValue.from_coefficient(-huge, dtype=dtype)
    assert value.value == -(huge + 1)
    assert len(repr(value)) < 650
    assert len(dtype.name) < 250
    assert "bits=" in repr(value)
    assert sys.get_int_max_str_digits() == before


def test_non_native_storage_rejected_by_scalar_construction() -> None:
    order = ">" if sys.byteorder == "little" else "<"
    dtype = FixedDType(NumPyDType(order + "i2"), step=Fraction(1, 10))
    for constructor in (FixedValue, FixedValue.approx, FixedValue.from_coefficient):
        with pytest.raises(ValueError, match="native"):
            constructor(1, dtype=dtype)


def test_casting_and_explicit_arithmetic() -> None:
    a = IntegerValue(1, dtype=INTEGER)
    half = RationalValue(1, 2, dtype=RATIONAL)
    assert add(a, half, dtype=TENTHS).value == sp.Rational(3, 2)
    assert subtract(a, half, dtype=TENTHS).value == sp.Rational(1, 2)
    assert multiply(a, half, dtype=TENTHS).dtype == RATIONAL
    assert divide(a, half, dtype=TENTHS) == 2
    third = RationalValue(1, 3, dtype=RATIONAL)
    with pytest.raises(PrecisionLossError):
        add(a, third, dtype=TENTHS)
    result = add(a, third, dtype=TENTHS, approximate=True)
    assert isinstance(result, FixedValue) and result.rounded
    assert result.value == sp.Rational(13, 10)
    with pytest.raises(OverflowError):
        subtract(
            IntegerValue(1000, dtype=INTEGER),
            IntegerValue(1000, dtype=INTEGER),
            dtype=HALVES,
        )


def test_conversion_history_and_real_component_checks() -> None:
    source = FloatingValue.approx(Fraction(1, 3), dtype=NumPyDType("float64"))
    fixed = cast(source, FixedDType(INTEGER, step=Fraction(1, 2**60)))
    assert isinstance(fixed, FixedValue) and fixed.rounded
    for target in (NumPyDType("float64"), NumPyDType("complex128")):
        result = cast(fixed, target)
        assert isinstance(result, (FloatingValue, ComplexValue)) and result.rounded
        assert result == source
    assert not FixedValue(fixed.to_rational(), dtype=fixed.dtype).rounded
    with pytest.raises(ValueError):
        fixed.to_integer(INTEGER)
    assert FixedValue(2, dtype=TENTHS).to_integer(I8) == 2
    with pytest.raises(OverflowError):
        FixedValue(1000, dtype=TENTHS).to_integer(I8)
    for approximate in (False, True):
        with pytest.raises(ValueError, match="imaginary"):
            cast(
                ComplexValue(1, 1, dtype=NumPyDType("complex64")),
                TENTHS,
                approximate=approximate,
            )
    real = cast(ComplexValue(1, -0.0, dtype=NumPyDType("complex64")), TENTHS)
    assert real == 1


def test_implicit_scalar_conversions_and_sympification_rejected() -> None:
    value: Any = FixedValue(1, dtype=TENTHS)
    for conversion in (int, float, complex, operator.index):
        with pytest.raises(TypeError):
            conversion(value)
    with pytest.raises(sp.SympifyError):
        sp.sympify(value)


@pytest.mark.parametrize("dtype", [None, "int8", I8, INTEGER, RATIONAL])
def test_construction_requires_fixed_descriptor(dtype: Any) -> None:
    for constructor in (FixedValue, FixedValue.approx, FixedValue.from_coefficient):
        with pytest.raises(TypeError, match="FixedDType"):
            constructor(1, dtype=dtype)


def test_nonbinary_ratios_do_not_compare_equal_to_binary_approximations() -> None:
    for exact in (Fraction(1, 10), Fraction(-1, 3), Fraction(1, sys.hash_info.modulus)):
        dtype = FixedDType(INTEGER, step=abs(exact))
        fixed = FixedValue(exact, dtype=dtype)
        rational = RationalValue(exact.numerator, exact.denominator, dtype=RATIONAL)
        floating = FloatingValue.approx(exact, dtype=NumPyDType("float64"))
        assert fixed == rational and rational == fixed
        assert hash(fixed) == hash(rational) == hash(exact)
        assert fixed != floating and floating != fixed
        for raw in (
            exact,
            sp.Rational(exact.numerator, exact.denominator),
            float(exact),
            True,
        ):
            with pytest.raises(TypeError):
                fixed.__eq__(raw)


def test_extended_float_conversion_does_not_narrow_through_python_float() -> None:
    precision = int(np.finfo(np.longdouble).nmant) + 1
    source = np.longdouble(1) + np.ldexp(np.longdouble(1), 1 - precision)
    dtype = FixedDType(INTEGER, step=Fraction(1, 2 ** (precision - 1)))
    fixed = FixedValue(source, dtype=dtype)
    assert fixed.value == sp.Rational(*source.as_integer_ratio())
    restored = cast(fixed, NumPyDType(np.longdouble))
    assert isinstance(restored, FloatingValue)
    assert restored.value == source and not restored.rounded
