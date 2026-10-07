"""Explicit operand representation, result types, and failure propagation."""

import operator
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from jvs.numeric import (
    ComplexValue,
    ExactDType,
    FloatingValue,
    IntegerValue,
    NumPyDType,
    PrecisionLossError,
    RationalValue,
    UnderflowError,
    add,
    divide,
    multiply,
    subtract,
)

I8 = NumPyDType("int8")
I16 = NumPyDType("int16")
F32 = NumPyDType("float32")
F64 = NumPyDType("float64")
C64 = NumPyDType("complex64")
C128 = NumPyDType("complex128")
INTEGER = ExactDType.integer()
RATIONAL = ExactDType.rational()
OPERATIONS = [add, subtract, multiply, divide]
TARGETS = [
    (I8, IntegerValue),
    (INTEGER, IntegerValue),
    (RATIONAL, RationalValue),
    (F32, FloatingValue),
    (C64, ComplexValue),
]


def sources(value: int) -> list[Any]:
    return [
        IntegerValue(value, dtype=I16),
        IntegerValue(value, dtype=INTEGER),
        RationalValue(value, dtype=RATIONAL),
        FloatingValue(value, dtype=F64),
        ComplexValue(value, 0, dtype=C128),
    ]


@pytest.mark.parametrize(
    ("operation", "expected"), [(add, 15), (subtract, 9), (multiply, 36), (divide, 4)]
)
@pytest.mark.parametrize(("dtype", "wrapper"), TARGETS)
def test_source_family_pairs_and_result_type_table(
    operation: Any, expected: int, dtype: Any, wrapper: Any
) -> None:
    for left in sources(12):
        for right in sources(3):
            result = operation(left, right, dtype=dtype)
            result_type = (
                RationalValue
                if operation is divide and wrapper is IntegerValue
                else wrapper
            )
            result_dtype = RATIONAL if result_type is RationalValue else dtype
            assert isinstance(result, result_type)
            assert result.dtype == result_dtype
            assert result == expected
            assert left == 12 and right == 3
            assert left.dtype in (I16, INTEGER, RATIONAL, F64, C128)


@pytest.mark.parametrize("dtype", [I8, INTEGER])
def test_integer_division_is_rational_not_an_integer_target_cast(dtype: Any) -> None:
    result = divide(
        IntegerValue(1, dtype=INTEGER), IntegerValue(3, dtype=I8), dtype=dtype
    )
    assert isinstance(result, RationalValue)
    assert result == RationalValue(1, 3, dtype=RATIONAL)
    assert result.dtype == RATIONAL
    widened_result = divide(
        IntegerValue(-128, dtype=I8), IntegerValue(-1, dtype=I8), dtype=I8
    )
    assert widened_result == 128
    assert widened_result.dtype == RATIONAL


def test_exact_rational_arithmetic_with_mixed_source_families() -> None:
    half = RationalValue(1, 2, dtype=RATIONAL)
    three = IntegerValue(3, dtype=I8)
    assert add(half, three, dtype=RATIONAL) == RationalValue(7, 2, dtype=RATIONAL)
    assert subtract(half, three, dtype=RATIONAL) == RationalValue(-5, 2, dtype=RATIONAL)
    assert multiply(half, three, dtype=RATIONAL) == RationalValue(3, 2, dtype=RATIONAL)
    assert divide(half, three, dtype=RATIONAL) == RationalValue(1, 6, dtype=RATIONAL)
    floating = FloatingValue(0.5, dtype=F32)
    assert divide(floating, three, dtype=RATIONAL) == RationalValue(
        1, 6, dtype=RATIONAL
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("fractional_on_left", [False, True])
def test_operand_conversion_rounding_requires_explicit_request(
    operation: Any, fractional_on_left: bool
) -> None:
    third = RationalValue(1, 3, dtype=RATIONAL)
    one = IntegerValue(1, dtype=I8)
    left, right = (third, one) if fractional_on_left else (one, third)
    for dtype in (F32, C64):
        with pytest.raises(PrecisionLossError):
            operation(left, right, dtype=dtype)
        result = operation(left, right, dtype=dtype, approximate=True)
        assert isinstance(result, (FloatingValue, ComplexValue))
        assert result.dtype == dtype
        assert result.rounded


@pytest.mark.parametrize("dtype", [F32, C64])
def test_exact_operand_conversion_still_allows_normal_arithmetic_rounding(
    dtype: NumPyDType,
) -> None:
    one = IntegerValue(1, dtype=I8)
    three = IntegerValue(3, dtype=I8)
    result = divide(one, three, dtype=dtype)
    assert isinstance(result, (FloatingValue, ComplexValue))
    assert result.rounded
    component = result.real if isinstance(result, ComplexValue) else result
    assert component.value.as_integer_ratio() == np.float32(1 / 3).as_integer_ratio()


@pytest.mark.parametrize("operation", OPERATIONS)
def test_inputs_must_fit_even_when_final_value_could_fit(operation: Any) -> None:
    too_large = IntegerValue(256, dtype=INTEGER)
    with pytest.raises(OverflowError):
        operation(too_large, too_large, dtype=I8)
    nonintegral = RationalValue(1, 2, dtype=RATIONAL)
    with pytest.raises(ValueError):
        operation(nonintegral, nonintegral, dtype=I8)


@pytest.mark.parametrize("operation", OPERATIONS)
def test_nonreal_inputs_cannot_enter_real_arithmetic(operation: Any) -> None:
    real = IntegerValue(1, dtype=I8)
    nonreal = ComplexValue(1, 2, dtype=C64)
    for left, right in ((real, nonreal), (nonreal, real)):
        with pytest.raises(ValueError, match="imaginary"):
            operation(left, right, dtype=F32, approximate=True)
        result = operation(left, right, dtype=C128)
        assert isinstance(result, ComplexValue)
        assert result.dtype == C128


def test_fixed_width_arithmetic_failures_are_not_automatically_widened() -> None:
    high, one, two = (IntegerValue(x, dtype=INTEGER) for x in (127, 1, 2))
    for operation, left, right in (
        (add, high, one),
        (subtract, IntegerValue(-128, dtype=INTEGER), one),
        (multiply, high, two),
    ):
        with pytest.raises(OverflowError):
            operation(left, right, dtype=I8)
    assert add(high, one, dtype=I16) == 128
    assert multiply(high, two, dtype=INTEGER) == 254


@pytest.mark.parametrize(("dtype", "unused"), TARGETS)
def test_division_by_zero_in_all_target_families(dtype: Any, unused: Any) -> None:
    one = IntegerValue(1, dtype=I8)
    zero = ComplexValue(-0.0, -0.0, dtype=C128)
    with pytest.raises(ZeroDivisionError):
        divide(one, zero, dtype=dtype)
    with pytest.raises(ZeroDivisionError):
        divide(zero, zero, dtype=dtype)


@pytest.mark.parametrize("dtype", [F32, C64])
def test_floating_result_failures_propagate(dtype: NumPyDType) -> None:
    large = FloatingValue(np.finfo(np.float32).max, dtype=F32)
    small = FloatingValue(np.finfo(np.float32).smallest_subnormal, dtype=F32)
    two = IntegerValue(2, dtype=INTEGER)
    with pytest.raises(OverflowError):
        multiply(large, two, dtype=dtype)
    with pytest.raises(UnderflowError):
        divide(small, two, dtype=dtype)
    tiny_exact = RationalValue(1, 2**150, dtype=RATIONAL)
    with pytest.raises(UnderflowError):
        multiply(tiny_exact, two, dtype=dtype, approximate=True)


def test_complex_dispatch_retains_final_rounding_contract() -> None:
    eps = Fraction(*np.finfo(np.float32).eps.as_integer_ratio())
    left = ComplexValue(1 + eps, 1, dtype=C128)
    right = ComplexValue(1 - eps, 1, dtype=C64)
    result = multiply(left, right, dtype=C64)
    assert result == ComplexValue(-eps * eps, 2, dtype=C64)
    assert isinstance(result, ComplexValue)
    assert not result.rounded
    huge = ComplexValue(np.finfo(np.float32).max, np.finfo(np.float32).max, dtype=C64)
    assert divide(huge, huge, dtype=C64) == 1


def test_history_and_signed_zero_follow_selected_representation() -> None:
    third = FloatingValue.approx(Fraction(1, 3), dtype=F32)
    zero = -(third - third)
    one = IntegerValue(1, dtype=I8)
    result = multiply(zero, one, dtype=F64)
    assert isinstance(result, FloatingValue)
    assert result.rounded and np.signbit(result.value)
    exact = multiply(zero, one, dtype=RATIONAL)
    assert isinstance(exact, RationalValue)
    assert exact == 0


@pytest.mark.parametrize("operation", OPERATIONS)
def test_public_api_requires_wrappers_keyword_dtype_and_valid_policy(
    operation: Any,
) -> None:
    one = IntegerValue(1, dtype=I8)
    for left, right in ((1, one), (one, np.int64(1))):
        with pytest.raises(TypeError, match="wrapper"):
            operation(left, right, dtype=I8)
    with pytest.raises(TypeError):
        operation(one, one)
    with pytest.raises(TypeError):
        operation(one, one, I8)
    with pytest.raises(TypeError, match="explicit"):
        operation(one, one, dtype="int8")
    with pytest.raises(TypeError, match="bool"):
        operation(one, one, dtype=F32, approximate=1)
    for dtype in (I8, INTEGER, RATIONAL):
        with pytest.raises(ValueError, match="Approximation"):
            operation(one, one, dtype=dtype, approximate=True)


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_existing_operators_still_reject_mixed_dtypes(operation: Any) -> None:
    a = IntegerValue(1, dtype=I8)
    b = IntegerValue(1, dtype=INTEGER)
    with pytest.raises(TypeError):
        operation(a, b)
