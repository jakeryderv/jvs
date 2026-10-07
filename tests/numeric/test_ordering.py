"""Exact real ordering across representations and foreign dispatch boundaries."""

import operator
from decimal import Decimal
from fractions import Fraction
from itertools import product
from typing import Any

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    ComplexValue,
    ExactDType,
    FixedDType,
    FixedValue,
    FloatingValue,
    IntegerValue,
    NumPyDType,
    RationalValue,
)

INTEGER = ExactDType.integer()
RATIONAL = ExactDType.rational()
F64 = NumPyDType("float64")
OPERATIONS = [operator.lt, operator.le, operator.gt, operator.ge]
# NumPy's stubs do not describe custom wrapper dispatch or invalid inputs.
UFUNCS: list[Any] = [np.less, np.less_equal, np.greater, np.greater_equal]
COMPARISONS = list(zip(OPERATIONS, UFUNCS, strict=True))


def real_values() -> list[tuple[Any, Fraction]]:
    result: list[tuple[Any, Fraction]] = []
    for n in (-2, 0, 2):
        expected = Fraction(n)
        result.extend(
            [
                (IntegerValue(n, dtype=INTEGER), expected),
                (IntegerValue(n, dtype=NumPyDType("int8")), expected),
            ]
        )
    for expected in map(Fraction, (-2, Fraction(-1, 2), 0, Fraction(1, 2), 2)):
        result.extend(
            [
                (
                    RationalValue(
                        expected.numerator, expected.denominator, dtype=RATIONAL
                    ),
                    expected,
                ),
                (
                    FixedValue(
                        expected, dtype=FixedDType(INTEGER, step=Fraction(1, 10))
                    ),
                    expected,
                ),
                (
                    FixedValue(
                        expected,
                        dtype=FixedDType(NumPyDType("int8"), step=Fraction(1, 2)),
                    ),
                    expected,
                ),
                (FloatingValue(expected, dtype=NumPyDType("float32")), expected),
            ]
        )
    result.append((FloatingValue(-0.0, dtype=F64), Fraction(0)))
    return result


@pytest.mark.parametrize(("operation", "ufunc"), COMPARISONS)
def test_all_real_representation_pairs_match_exact_value_oracle(
    operation: Any, ufunc: Any
) -> None:
    for (left, a), (right, b) in product(real_values(), repeat=2):
        expected = operation(a, b)
        assert operation(left, right) is expected
        assert ufunc(left, right) is expected
        assert (left <= right and right <= left) is (left == right)
        assert (left < right) is (right > left)
        assert (left <= right) is (right >= left)


def test_transitivity_sorting_and_extrema_across_representations() -> None:
    values = [value for value, _ in real_values()]
    for a, b, c in product(values, repeat=3):
        if a <= b <= c:
            assert a <= c
            if a < b or b < c:
                assert a < c
    source = list(reversed(real_values()))
    expected = [value for value, _ in sorted(source, key=lambda pair: pair[1])]
    assert sorted([value for value, _ in source]) == expected
    assert min(values) == -2
    assert max(values) == 2


@pytest.mark.parametrize(
    "raw",
    [-2, 0, 2, np.int8(-2), np.uint64(2), sp.Integer(-2), sp.Integer(0), sp.Integer(2)],
)
@pytest.mark.parametrize(("operation", "ufunc"), COMPARISONS)
def test_raw_integer_scalars_work_in_both_directions(
    raw: Any, operation: Any, ufunc: Any
) -> None:
    for value, expected in real_values():
        assert operation(value, raw) is operation(expected, int(raw))
        assert operation(raw, value) is operation(int(raw), expected)
        assert ufunc(value, raw) is operation(expected, int(raw))
        assert ufunc(raw, value) is operation(int(raw), expected)


def test_large_integer_and_rational_boundaries_do_not_narrow() -> None:
    floating = FloatingValue(2**53, dtype=F64)
    above = IntegerValue(2**53 + 1, dtype=INTEGER)
    assert floating < above and above > floating
    assert not above <= floating
    unsigned = IntegerValue(2**64 - 1, dtype=NumPyDType("uint64"))
    rounded_up = FloatingValue(2**64, dtype=F64)
    assert unsigned < rounded_up
    assert IntegerValue(-1, dtype=NumPyDType("int64")) < unsigned
    huge = 10**5000
    denominator = huge + 1
    below_one = RationalValue(huge, denominator, dtype=RATIONAL)
    above_one = RationalValue(huge + 2, denominator, dtype=RATIONAL)
    assert below_one < FloatingValue(1, dtype=F64) < above_one
    assert (
        IntegerValue(-huge, dtype=INTEGER)
        < floating
        < IntegerValue(huge, dtype=INTEGER)
    )
    fixed = FixedValue.from_coefficient(
        huge, dtype=FixedDType(INTEGER, step=Fraction(1, denominator))
    )
    assert fixed == below_one and fixed <= below_one and not fixed < below_one


def test_decimal_tenths_compare_to_stored_binary_tenths() -> None:
    exact = RationalValue(1, 10, dtype=RATIONAL)
    fixed = FixedValue(
        Fraction(1, 10), dtype=FixedDType(INTEGER, step=Fraction(1, 100))
    )
    binary = FloatingValue(0.1, dtype=F64)
    assert exact == fixed
    assert exact < binary and fixed < binary
    assert -binary < -fixed


def test_extended_floats_and_subnormals_keep_original_precision() -> None:
    formats = [np.float16, np.float32, np.float64, np.longdouble]
    for scalar in formats:
        dtype = NumPyDType(scalar)
        one = FloatingValue(1, dtype=dtype)
        successor = np.nextafter(scalar(1), scalar(2))
        above = FloatingValue(successor, dtype=dtype)
        exact = Fraction(*successor.as_integer_ratio())
        midpoint = (Fraction(1) + exact) / 2
        between = RationalValue(
            midpoint.numerator, midpoint.denominator, dtype=RATIONAL
        )
        assert one < between < above
        tiny = scalar(np.finfo(scalar).smallest_subnormal)
        subnormal = FloatingValue(tiny, dtype=dtype)
        ratio = Fraction(*tiny.as_integer_ratio()) / 2
        smaller = FixedValue(ratio, dtype=FixedDType(INTEGER, step=ratio))
        assert 0 < smaller < subnormal
        assert -subnormal < -smaller < 0


def test_zero_signs_and_rounding_history_are_preserved() -> None:
    positive = FloatingValue(0.0, dtype=F64)
    negative = FloatingValue(-0.0, dtype=F64)
    rounded = FixedValue.approx(Fraction(1, 100), dtype=FixedDType(INTEGER, step=1))
    pristine = FixedValue(0, dtype=rounded.dtype)
    values = [positive, negative, rounded, pristine]
    before = [repr(value) for value in values]
    for a, b in product(values, repeat=2):
        assert a == b and a <= b and a >= b
        assert not a < b and not a > b
    assert [repr(value) for value in values] == before
    assert np.signbit(negative.value) and not np.signbit(positive.value)
    assert rounded.rounded and not pristine.rounded
    assert negative.dtype == F64


@pytest.mark.parametrize(
    "raw",
    [
        True,
        np.bool_(True),
        sp.true,
        1.0,
        np.float32(1),
        Fraction(1),
        sp.Rational(1, 2),
        sp.Float(1),
        Decimal(1),
        1 + 0j,
        np.complex64(1),
        float("nan"),
        float("inf"),
        sp.oo,
        sp.Symbol("x"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("method", ["__lt__", "__le__", "__gt__", "__ge__"])
def test_wrapper_methods_reject_other_raw_numeric_representations(
    raw: Any, method: str
) -> None:
    for value, _ in real_values():
        with pytest.raises(TypeError):
            getattr(value, method)(raw)


@pytest.mark.parametrize("imag", [0, -0.0, 1])
@pytest.mark.parametrize(("operation", "ufunc"), COMPARISONS)
def test_complex_wrappers_are_unordered_in_both_directions(
    imag: Any, operation: Any, ufunc: Any
) -> None:
    z = ComplexValue(1, imag, dtype=NumPyDType("complex128"))
    others = [value for value, _ in real_values()] + [z, 1, np.int64(1), sp.Integer(1)]
    for other in others:
        for left, right in ((z, other), (other, z)):
            with pytest.raises(TypeError, match="unordered"):
                operation(left, right)
            with pytest.raises(TypeError, match="unordered"):
                ufunc(left, right)
    if not imag:
        assert z.to_real() <= IntegerValue(1, dtype=INTEGER)
    else:
        with pytest.raises(ValueError, match="imaginary"):
            z.to_real()


@pytest.mark.parametrize(("operation", "ufunc"), COMPARISONS)
def test_numpy_dispatch_preserves_direction_and_rejects_options(
    operation: Any, ufunc: Any
) -> None:
    # Distinct operands detect incorrectly reversed comparisons in ufunc dispatch.
    for value, ratio in real_values():
        transport = np.array(3, dtype=np.int64)
        assert ufunc(value, transport) is operation(ratio, 3)
        assert ufunc(transport, value) is operation(3, ratio)
        with pytest.raises(TypeError):
            ufunc.reduce(value)
        for kwargs in ({"where": True}, {"out": np.array(False)}, {"dtype": bool}):
            with pytest.raises(TypeError):
                ufunc(value, value, **kwargs)
        for raw in (
            np.array([1]),
            np.array(1.0),
            np.array(1 + 0j),
            np.array(True),
            np.array(value, dtype=object),
            1.0,
            True,
            object(),
        ):
            for left, right in ((value, raw), (raw, value)):
                with pytest.raises(TypeError):
                    ufunc(left, right)


def test_unrelated_objects_keep_reflected_dispatch_and_normal_failure() -> None:
    class Foreign:
        def __gt__(self, other: object) -> bool:
            return True

    for value, _ in real_values():
        unrelated: Any = object()
        for method in ("__lt__", "__le__", "__gt__", "__ge__"):
            assert getattr(value, method)(object()) is NotImplemented
        for operation in OPERATIONS:
            with pytest.raises(TypeError):
                operation(value, unrelated)
            with pytest.raises(TypeError):
                operation(unrelated, value)
        assert (value < Foreign()) is True


def test_foreign_ordering_can_answer_without_wrapper_dispatch() -> None:
    class Foreign:
        def __lt__(self, other: object) -> bool:
            return True

    complex_value = ComplexValue(1, dtype=NumPyDType("complex64"))
    assert Foreign() < complex_value
    with pytest.raises(TypeError, match="unordered"):
        operator.gt(complex_value, Foreign())
