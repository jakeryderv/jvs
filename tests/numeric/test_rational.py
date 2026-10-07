"""Exact ratios, conversion, membership, and foreign-backend boundaries."""

import operator
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

EXACT = ExactDType.rational()


def ratio(p: Any, q: Any = 1) -> RationalValue:
    return RationalValue(p, q, dtype=EXACT)


@pytest.mark.parametrize("scalar", [int, np.int64, np.uint64, sp.Integer])
def test_supported_integer_components(scalar: Any) -> None:
    value = ratio(scalar(6), scalar(8))
    assert value.numerator == 3
    assert value.denominator == 4
    assert isinstance(value.numerator, sp.Integer)
    assert isinstance(value.denominator, sp.Integer)
    assert isinstance(value.value, sp.Rational)
    assert value.value == sp.Rational(3, 4)
    assert value.dtype == EXACT


@pytest.mark.parametrize(
    ("p", "q", "expected"),
    [
        (6, 8, (3, 4)),
        (6, -8, (-3, 4)),
        (-6, -8, (3, 4)),
        (0, -9, (0, 1)),
        (6, 3, (2, 1)),
    ],
)
def test_canonical_reduction(p: int, q: int, expected: tuple[int, int]) -> None:
    value = ratio(p, q)
    assert (value.numerator, value.denominator) == expected
    assert value.dtype == EXACT
    if expected[1] == 1:
        assert isinstance(value.value, sp.Integer)


@pytest.mark.parametrize("zero", [0, np.int8(0), np.uint64(0), sp.Integer(0)])
@pytest.mark.parametrize("p", [0, 1, -1])
def test_zero_denominator_always_raises(p: int, zero: Any) -> None:
    with pytest.raises(ZeroDivisionError):
        ratio(p, zero)


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        sp.true,
        3.0,
        np.float32(3),
        sp.Float(3),
        float("inf"),
        float("nan"),
        sp.oo,
        sp.nan,
        3 + 0j,
        Fraction(3, 1),
        sp.Rational(3, 2),
        Decimal(3),
        "3",
        np.array(3),
        np.timedelta64(3, "D"),
        sp.Symbol("n", integer=True),
        IntegerValue(3, dtype=ExactDType.integer()),
    ],
)
def test_components_require_integer_representations(value: Any) -> None:
    with pytest.raises(TypeError):
        ratio(value, 1)
    with pytest.raises(TypeError):
        ratio(1, value)


def test_duck_conversion_is_not_called() -> None:
    class PretendInteger:
        def __int__(self) -> int:
            raise AssertionError("Must not invoke arbitrary conversion")

    with pytest.raises(TypeError):
        ratio(PretendInteger())


@pytest.mark.parametrize(
    "dtype",
    [
        None,
        "Rational",
        sp.Rational,
        ExactDType.integer(),
        NumPyDType("float64"),
        NumPyDType("int64"),
    ],
)
def test_requires_explicit_rational_dtype(dtype: Any) -> None:
    with pytest.raises(TypeError):
        RationalValue(1, 2, dtype=dtype)


def test_dtype_cannot_be_omitted() -> None:
    constructor: Any = RationalValue
    with pytest.raises(TypeError):
        constructor(1, 2)


@pytest.mark.parametrize(("p", "q"), [(1, 2), (-5, 3), (-4, 2), (0, 7), (6, 3)])
def test_membership_uses_reduced_value(p: int, q: int) -> None:
    value = ratio(p, q)
    facts = classify(value)
    for domain in (Rational, Real, Algebraic):
        assert facts[domain].state is Truth.TRUE
    assert (value in Integer) is (p % q == 0)
    assert (value in Whole) is (p % q == 0 and p >= 0)
    assert (value in Natural) is (p % q == 0 and p > 0)
    assert value not in Irrational
    assert "RationalValue" in facts[Rational].reason


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
def test_arithmetic_matches_independent_exact_oracle(operation: Any) -> None:
    # Include cancellation, negative signs, zero, and values beyond machine range.
    pairs = [(1, 3), (-5, 7), (0, 1), (10**200 + 1, 10**100 + 3)]
    for p, q in pairs:
        for r, s in pairs:
            a, b = ratio(p, q), ratio(r, s)
            if operation is operator.truediv and r == 0:
                with pytest.raises(ZeroDivisionError):
                    operation(a, b)
                continue
            result = operation(a, b)
            expected = operation(Fraction(p, q), Fraction(r, s))
            assert isinstance(result, RationalValue)
            assert result.dtype == EXACT
            assert (result.numerator, result.denominator) == (
                expected.numerator,
                expected.denominator,
            )
            assert (a.numerator, a.denominator) == (p, q)


def test_unary_and_integral_results_preserve_wrapper() -> None:
    value = ratio(1, 3)
    assert +value is value
    assert -value == ratio(-1, 3)
    assert value - value == ratio(0)
    assert value / value == ratio(1)
    assert isinstance((value / value).value, sp.Integer)
    assert (value / value).dtype == EXACT


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize(
    "other",
    [
        0,
        1,
        True,
        0.5,
        np.int64(1),
        np.uint64(1),
        np.float64(0.5),
        sp.Integer(0),
        sp.Integer(1),
        sp.Rational(1, 2),
        Fraction(1, 2),
        sp.Float(0.5),
        IntegerValue(1, dtype=ExactDType.integer()),
        np.array([1]),
    ],
)
def test_mixed_arithmetic_requires_explicit_conversion(
    operation: Any, other: Any
) -> None:
    value = ratio(1, 2)
    with pytest.raises(TypeError):
        operation(value, other)
    with pytest.raises(TypeError):
        operation(other, value)


@pytest.mark.parametrize("p", [-1, 0, 1, 2**53 + 1, 2**63 - 1])
def test_integer_equality_is_symmetric_and_hash_consistent(p: int) -> None:
    value = ratio(p * 3, 3)
    peers = [
        value,
        ratio(p),
        p,
        np.int64(p),
        sp.Integer(p),
        IntegerValue(p, dtype=ExactDType.integer()),
        IntegerValue(p, dtype=NumPyDType("int64")),
    ]
    for other in peers:
        assert value == other
        assert other == value
        assert bool(operator.ne(value, other)) is False
        assert bool(operator.ne(other, value)) is False
        assert hash(value) == hash(other)
    assert len(set(peers)) == 1
    assert value != ratio(p + 1)


def test_fractional_equality_and_hashing() -> None:
    a, b = ratio(1, 3), ratio(7, 21)
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1
    assert {a: "exact"}[b] == "exact"
    assert a != ratio(1, 2)
    for other in (
        0,
        np.int64(0),
        sp.Integer(0),
        IntegerValue(0, dtype=ExactDType.integer()),
    ):
        assert a != other
        assert other != a
    assert a != object()
    maximum = np.uint64(2**64 - 1)
    assert ratio(maximum) == maximum
    assert maximum == ratio(maximum)
    assert hash(ratio(maximum)) == hash(maximum)


@pytest.mark.parametrize(
    "other",
    [
        True,
        np.bool_(True),
        0.5,
        np.float64(0.5),
        sp.Float(0.5),
        Fraction(1, 2),
        sp.Rational(1, 2),
        Decimal("0.5"),
        0.5 + 0j,
    ],
)
def test_other_numeric_equality_requires_explicit_conversion(other: Any) -> None:
    value = ratio(1, 2)
    for operation in (operator.eq, operator.ne):
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(value, other)
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(other, value)


def test_exact_integer_conversion_and_range_checks() -> None:
    value = ratio(510, 2)
    assert value.to_integer(NumPyDType("uint8")) == 255
    assert value.to_integer(ExactDType.integer()) == 255
    with pytest.raises(OverflowError):
        value.to_integer(NumPyDType("int8"))
    with pytest.raises(OverflowError):
        ratio(-1).to_integer(NumPyDType("uint64"))
    with pytest.raises(TypeError):
        value.to_integer(NumPyDType("float64"))
    with pytest.raises(TypeError):
        value.to_integer(EXACT)
    huge = 10**10000
    assert ratio(huge * 3, 3).to_integer(ExactDType.integer()) == huge
    with pytest.raises(OverflowError):
        ratio(huge).to_integer(NumPyDType("int64"))
    for fractional in (ratio(1, 2), ratio(-1, 2)):
        with pytest.raises(ValueError, match="denominator of one"):
            fractional.to_integer(ExactDType.integer())


def test_protocols_and_immutability() -> None:
    value: Any = ratio(2, 3)
    assert bool(value)
    assert not bool(ratio(0, -2))
    assert "RationalValue(2, 3" in repr(value)
    for attribute, replacement in (("dtype", EXACT), ("_value", sp.Integer(1))):
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, replacement)
    for conversion in (int, float, operator.index):
        with pytest.raises(TypeError):
            conversion(value)
    with pytest.raises(sp.SympifyError):
        sp.sympify(value)
    for operation in (operator.floordiv, operator.mod, operator.pow, operator.lt):
        with pytest.raises(TypeError):
            operation(value, value)


def test_numpy_dispatch_cannot_bypass_contract() -> None:
    # NumPy's scalar overloads do not describe third-party ufunc dispatch.
    value: Any = ratio(2)
    equal: Any = ratio(2)
    different: Any = ratio(1, 2)
    scalar_array = np.array(2, dtype=np.uint64)
    assert scalar_array == value
    assert value == scalar_array
    assert np.equal(value, equal)
    assert np.not_equal(value, different)
    for other in (np.array([2]), np.array(2.0)):
        with pytest.raises(TypeError):
            _ = value == other
        with pytest.raises(TypeError):
            _ = other == value
    for ufunc in (np.add, np.subtract, np.multiply, np.divide):
        with pytest.raises(TypeError, match="ufunc"):
            ufunc(value, value)
    with pytest.raises(TypeError):
        np.equal(value, value, where=True)
    with pytest.raises(TypeError):
        np.equal(value, value, out=np.array(False))
