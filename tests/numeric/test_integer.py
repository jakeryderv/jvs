"""Integer representation, overflow, inference, and backend dispatch contracts."""

import operator
import sys
import warnings
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    Algebraic,
    ExactDType,
    Integer,
    IntegerValue,
    Irrational,
    Natural,
    NumPyDType,
    Rational,
    RationalValue,
    Real,
    Truth,
    Whole,
    classify,
)

DTYPES = [
    NumPyDType(f"{kind}{bits}") for kind in ("int", "uint") for bits in (8, 16, 32, 64)
]
EXACT = ExactDType.integer()


@pytest.mark.parametrize("dtype", [*DTYPES, EXACT])
@pytest.mark.parametrize("value", [3, np.int32(3), np.uint64(3), sp.Integer(3)])
def test_construction_preserves_value_and_selected_backend(
    dtype: Any, value: Any
) -> None:
    wrapped = IntegerValue(value, dtype=dtype)
    assert int(wrapped) == 3
    assert wrapped.dtype == dtype
    if isinstance(dtype, NumPyDType):
        assert isinstance(wrapped.value, np.integer)
        assert wrapped.value.dtype == dtype.numpy_dtype
    else:
        assert isinstance(wrapped.value, sp.Integer)


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_boundaries_and_out_of_range(dtype: NumPyDType) -> None:
    limits = dtype.integer_info
    for value in (int(limits.min), int(limits.max)):
        assert int(IntegerValue(value, dtype=dtype)) == value
    for value in (int(limits.min) - 1, int(limits.max) + 1, 10**10000):
        with pytest.raises(OverflowError):
            IntegerValue(value, dtype=dtype)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        sp.true,
        3.0,
        np.float64(3),
        sp.Float(3),
        3 + 0j,
        Fraction(3, 1),
        Decimal(3),
        "3",
        np.array(3),
        np.timedelta64(3, "D"),
        sp.Symbol("n", integer=True),
    ],
)
def test_noninteger_representations_are_not_coerced(value: Any) -> None:
    for dtype in (NumPyDType("int64"), EXACT):
        with pytest.raises(TypeError):
            IntegerValue(value, dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        "int32",
        np.int32,
        np.dtype("int32"),
        None,
        NumPyDType("float32"),
        NumPyDType("complex128"),
        ExactDType.rational(),
    ],
)
def test_dtype_must_be_explicit_and_integer(dtype: Any) -> None:
    with pytest.raises(TypeError):
        IntegerValue(3, dtype=dtype)


def test_nonnative_storage_descriptor_is_rejected_for_scalar() -> None:
    order = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="native byte order"):
        IntegerValue(3, dtype=NumPyDType(f"{order}i4"))


def test_only_supported_integer_types_are_converted() -> None:
    class PretendInteger:
        def __int__(self) -> int:
            raise AssertionError("Must not invoke arbitrary conversion")

    value: Any = PretendInteger()
    with pytest.raises(TypeError):
        IntegerValue(value, dtype=EXACT)


@pytest.mark.parametrize("dtype", [NumPyDType("int32"), EXACT])
@pytest.mark.parametrize(
    ("value", "whole", "natural"),
    [(-3, False, False), (0, True, False), (3, True, True)],
)
def test_membership_uses_value_not_just_dtype(
    dtype: Any, value: int, whole: bool, natural: bool
) -> None:
    wrapped = IntegerValue(value, dtype=dtype)
    facts = classify(wrapped)
    for domain in (Integer, Rational, Real, Algebraic):
        assert facts[domain].state is Truth.TRUE
    assert (wrapped in Whole) is whole
    assert (wrapped in Natural) is natural
    assert wrapped not in Irrational
    assert "IntegerValue" in facts[Integer].reason


@pytest.mark.parametrize("dtype", [*DTYPES, EXACT])
def test_same_dtype_arithmetic(dtype: Any) -> None:
    a = IntegerValue(12, dtype=dtype)
    b = IntegerValue(3, dtype=dtype)
    for result, expected in ((a + b, 15), (a - b, 9), (a * b, 36)):
        assert isinstance(result, IntegerValue)
        assert result.dtype == dtype
        assert int(result) == expected
    assert int(a) == 12
    assert +a is a


@pytest.mark.parametrize("dtype", DTYPES)
def test_arithmetic_overflow_and_underflow_raise_before_backend_wrap(
    dtype: NumPyDType,
) -> None:
    limits = dtype.integer_info
    high = IntegerValue(limits.max, dtype=dtype)
    low = IntegerValue(limits.min, dtype=dtype)
    one = IntegerValue(1, dtype=dtype)
    two = IntegerValue(2, dtype=dtype)
    # A warning is not an acceptable substitute for the requested exception.
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        for operation in (lambda: high + one, lambda: low - one, lambda: high * two):
            with pytest.raises(OverflowError):
                operation()
    assert int(high) == int(limits.max)
    assert int(low) == int(limits.min)


@pytest.mark.parametrize("dtype", DTYPES)
def test_negation_range(dtype: NumPyDType) -> None:
    assert int(-IntegerValue(0, dtype=dtype)) == 0
    if dtype.integer_info.signed:
        assert int(-IntegerValue(1, dtype=dtype)) == -1
        value = int(dtype.integer_info.min)
    else:
        value = 1
    with pytest.raises(OverflowError):
        _ = -IntegerValue(value, dtype=dtype)


def test_unbounded_exact_arithmetic() -> None:
    huge = 10**1000
    a = IntegerValue(huge, dtype=EXACT)
    b = IntegerValue(huge + 1, dtype=EXACT)
    assert int(a + b) == 2 * huge + 1
    assert int(a * b) == huge * (huge + 1)
    assert int(a - b) == -1
    assert isinstance((a * b).value, sp.Integer)
    assert int(-a) == -huge


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize(
    "other",
    [
        IntegerValue(3, dtype=NumPyDType("int16")),
        IntegerValue(3, dtype=EXACT),
        3,
        True,
        np.int8(3),
        np.uint64(3),
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(3),
        3.0,
        np.float32(3),
        sp.Rational(3, 2),
        np.array([3]),
    ],
)
def test_mixed_arithmetic_cannot_bypass_dtype_contract(
    operation: Any, other: Any
) -> None:
    value = IntegerValue(3, dtype=NumPyDType("int8"))
    with pytest.raises(TypeError):
        operation(value, other)
    with pytest.raises(TypeError):
        operation(other, value)


def test_explicit_conversion_is_checked() -> None:
    original = IntegerValue(255, dtype=NumPyDType("uint8"))
    widened = original.to(NumPyDType("int16"))
    assert widened == original
    assert widened.dtype == NumPyDType("int16")
    assert widened.to(EXACT).to(original.dtype) == original
    with pytest.raises(OverflowError):
        original.to(NumPyDType("int8"))
    with pytest.raises(OverflowError):
        IntegerValue(-1, dtype=EXACT).to(NumPyDType("uint64"))
    with pytest.raises(TypeError):
        original.to(NumPyDType("float64"))


@pytest.mark.parametrize("value", [-1, 0, 1, 2**53 + 1, 2**63 - 1])
def test_equal_integer_representations_share_hashes(value: int) -> None:
    representations = [
        IntegerValue(value, dtype=NumPyDType("int64")),
        IntegerValue(value, dtype=EXACT),
        value,
        np.int64(value),
        sp.Integer(value),
    ]
    for a in representations:
        for b in representations:
            assert a == b
            assert bool(operator.ne(a, b)) is False
            assert hash(a) == hash(b)
    assert len(set(representations)) == 1
    assert IntegerValue(value + 1, dtype=EXACT) != representations[0]


def test_unsigned_and_large_exact_equality_never_uses_float() -> None:
    maximum = 2**64 - 1
    a = IntegerValue(maximum, dtype=NumPyDType("uint64"))
    assert a == IntegerValue(maximum, dtype=EXACT)
    assert np.uint64(maximum) == a
    assert a != IntegerValue(maximum - 1, dtype=EXACT)
    assert hash(a) == hash(maximum)


@pytest.mark.parametrize(
    "other",
    [
        True,
        np.bool_(True),
        3.0,
        np.float64(3),
        sp.Float(3),
        Fraction(3, 1),
        Decimal(3),
        3 + 0j,
    ],
)
def test_noninteger_numeric_equality_requires_explicit_conversion(other: Any) -> None:
    value = IntegerValue(3, dtype=EXACT)
    with pytest.raises(TypeError):
        _ = value == other
    with pytest.raises((TypeError, sp.SympifyError)):
        _ = other == value


def test_integer_protocols_and_immutability() -> None:
    value = IntegerValue(2, dtype=NumPyDType("int8"))
    assert operator.index(value) == 2
    assert ["a", "b", "c"][value] == "c"
    assert bool(value)
    assert not bool(IntegerValue(0, dtype=EXACT))
    assert "IntegerValue(2" in repr(value)
    assert value != object()
    for attribute, replacement in (("dtype", EXACT), ("_value", np.int8(1))):
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, replacement)


def test_unsupported_ufuncs_and_floor_division_fail() -> None:
    value: Any = IntegerValue(3, dtype=EXACT)
    with pytest.raises(TypeError, match="ufunc"):
        np.add(value, value)
    with pytest.raises(TypeError):
        operator.floordiv(value, value)
    with pytest.raises(TypeError, match="ufunc"):
        np.divide(value, value)
    with pytest.raises(TypeError):
        operator.truediv(sp.Integer(1), value)
    with pytest.raises(TypeError):
        np.equal(value, value, where=True)


@pytest.mark.parametrize("dtype", [*DTYPES, EXACT])
@pytest.mark.parametrize(("numerator", "denominator"), [(1, 3), (12, 3), (0, 3)])
def test_integer_division_returns_exact_rational(
    dtype: Any, numerator: int, denominator: int
) -> None:
    a = IntegerValue(numerator, dtype=dtype)
    b = IntegerValue(denominator, dtype=dtype)
    result = a / b
    expected = Fraction(numerator, denominator)
    assert isinstance(result, RationalValue)
    assert result.dtype == ExactDType.rational()
    assert result.numerator == expected.numerator
    assert result.denominator == expected.denominator
    assert result in Rational
    assert (result in Integer) is (expected.denominator == 1)
    assert b.__rtruediv__(a) == result
    assert int(a) == numerator
    assert int(b) == denominator
    if expected.denominator == 1:
        assert result.to_integer(dtype) == expected.numerator


@pytest.mark.parametrize("dtype", [*DTYPES, EXACT])
@pytest.mark.parametrize("numerator", [0, 1])
def test_integer_division_by_zero_raises(dtype: Any, numerator: int) -> None:
    a = IntegerValue(numerator, dtype=dtype)
    zero = IntegerValue(0, dtype=dtype)
    with pytest.raises(ZeroDivisionError):
        _ = a / zero


@pytest.mark.parametrize("dtype", DTYPES)
def test_integer_division_does_not_overflow_or_round(dtype: NumPyDType) -> None:
    limits = dtype.integer_info
    numerator = int(limits.min) if limits.signed else int(limits.max)
    denominator = -1 if limits.signed else 3
    a = IntegerValue(numerator, dtype=dtype)
    b = IntegerValue(denominator, dtype=dtype)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        result = a / b
    expected = Fraction(numerator, denominator)
    assert (result.numerator, result.denominator) == (
        expected.numerator,
        expected.denominator,
    )
    if limits.signed:
        # min / -1 is exact in the rational result but cannot fit back in dtype.
        with pytest.raises(OverflowError):
            result.to_integer(dtype)


@pytest.mark.parametrize(
    ("p", "q"),
    [(-1, 3), (1, -3), (-1, -3), (2**53 + 1, 2**53), (10**1000 + 1, -(10**500 + 3))],
)
def test_exact_integer_division_preserves_sign_and_large_values(p: int, q: int) -> None:
    result = IntegerValue(p, dtype=EXACT) / IntegerValue(q, dtype=EXACT)
    expected = Fraction(p, q)
    assert (result.numerator, result.denominator) == (
        expected.numerator,
        expected.denominator,
    )


def test_numpy_zero_dimensional_equality_transport() -> None:
    value = IntegerValue(2**64 - 1, dtype=EXACT)
    scalar_array = np.array(2**64 - 1, dtype=np.uint64)
    assert scalar_array == value
    assert value == scalar_array
    for other in (np.array([3]), np.array(3.0)):
        with pytest.raises(TypeError):
            _ = other == value
        with pytest.raises(TypeError):
            _ = value == other
