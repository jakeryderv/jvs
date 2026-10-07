"""Checked complex composition, classification, conversion, and dispatch."""

import operator
import sys
import warnings
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    Algebraic,
    Complex,
    ComplexValue,
    ExactDType,
    FloatingValue,
    Integer,
    IntegerValue,
    Irrational,
    Natural,
    NumPyDType,
    PrecisionLossError,
    Rational,
    RationalValue,
    Real,
    Transcendental,
    UnderflowError,
    Whole,
    classify,
)

C64 = NumPyDType("complex64")
C128 = NumPyDType("complex128")
DTYPES = list(
    dict.fromkeys(NumPyDType(t) for t in (np.complex64, np.complex128, np.clongdouble))
)


def make(real: Any, imag: Any = 0, dtype: NumPyDType = C64) -> ComplexValue:
    return ComplexValue(real, imag, dtype=dtype)


def supported(dtype: NumPyDType) -> None:
    if dtype.component_dtype.floating_info.significand_bits not in (24, 53, 64, 113):
        with pytest.raises(NotImplementedError):
            make(1, dtype=dtype)
        pytest.skip("Unsupported extended component format")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "component",
    [
        3,
        np.int64(3),
        np.uint64(3),
        sp.Integer(3),
        3.0,
        np.float32(3),
        Fraction(3),
        sp.Rational(6, 2),
        IntegerValue(3, dtype=ExactDType.integer()),
        RationalValue(3, dtype=ExactDType.rational()),
        FloatingValue(3, dtype=NumPyDType("float32")),
    ],
)
def test_exact_component_construction(dtype: NumPyDType, component: Any) -> None:
    supported(dtype)
    z = make(component, component, dtype)
    assert isinstance(z.real, FloatingValue)
    assert isinstance(z.imag, FloatingValue)
    assert z.real == z.imag == 3
    assert z.real.dtype == z.imag.dtype == dtype.component_dtype
    assert isinstance(z.value, np.complexfloating)
    assert z.value.dtype == dtype.numpy_dtype
    assert z.value.real.as_integer_ratio() == (3, 1)
    assert z.value.imag.as_integer_ratio() == (3, 1)
    assert not z.rounded
    assert not make(component, dtype=dtype).imag


@pytest.mark.parametrize(
    "dtype",
    [
        None,
        "complex64",
        np.complex64,
        np.dtype("complex64"),
        NumPyDType("float64"),
        NumPyDType("int64"),
        ExactDType.rational(),
    ],
)
def test_explicit_complex_dtype_required(dtype: Any) -> None:
    with pytest.raises(TypeError):
        make(1, dtype=dtype)


def test_missing_dtype_and_nonnative_storage() -> None:
    constructor: Any = ComplexValue
    with pytest.raises(TypeError):
        constructor(1, 2)
    order = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="native byte order"):
        make(1, dtype=NumPyDType(f"{order}c16"))


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        1 + 2j,
        np.complex64(1 + 2j),
        sp.I,
        sp.Float(1),
        "1",
        np.array(1),
        sp.Symbol("x"),
        make(1),
    ],
)
def test_unsupported_components_raise(value: Any) -> None:
    for constructor in (ComplexValue, ComplexValue.approx):
        with pytest.raises(TypeError):
            constructor(value, 0, dtype=C64)
        with pytest.raises(TypeError):
            constructor(0, value, dtype=C64)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("imaginary", [False, True])
def test_component_failures_apply_in_both_positions(
    dtype: NumPyDType, imaginary: bool
) -> None:
    supported(dtype)
    info = dtype.component_dtype.floating_info
    cases = [
        (float("nan"), ValueError),
        (float("inf"), ValueError),
        (Fraction(*info.max.as_integer_ratio()) + 1, OverflowError),
        (Fraction(*info.smallest_subnormal.as_integer_ratio()) / 2, UnderflowError),
    ]
    for value, error in cases:
        real, imag = (0, value) if imaginary else (value, 0)
        for constructor in (ComplexValue, ComplexValue.approx):
            with pytest.raises(error):
                constructor(real, imag, dtype=dtype)
    real, imag = (0, Fraction(1, 3)) if imaginary else (Fraction(1, 3), 0)
    with pytest.raises(PrecisionLossError):
        make(real, imag, dtype)
    result = ComplexValue.approx(real, imag, dtype=dtype)
    assert result.rounded
    assert result.real.rounded is (not imaginary)
    assert result.imag.rounded is imaginary


@pytest.mark.parametrize("dtype", DTYPES)
def test_add_subtract_conjugate_and_rounding_history(dtype: NumPyDType) -> None:
    supported(dtype)
    a, b = make(1, 2, dtype), make(3, -4, dtype)
    assert a + b == make(4, -2, dtype)
    assert a - b == make(-2, 6, dtype)
    assert b.__rsub__(a) == a - b
    assert b.__radd__(a) == a + b
    assert -a == make(-1, -2, dtype)
    assert +a is a
    assert a.conjugate() == make(1, -2, dtype)
    assert a.conjugate().conjugate() == a
    assert (a + b).dtype == dtype
    half_eps = Fraction(*dtype.component_dtype.floating_info.eps.as_integer_ratio()) / 2
    rounded = a + make(half_eps, 0, dtype)
    assert rounded == a
    assert rounded.real.rounded and not rounded.imag.rounded
    assert rounded.conjugate().real.rounded
    assert (-rounded).rounded
    third = ComplexValue.approx(0, Fraction(1, 3), dtype=dtype)
    canceled = third - third
    assert canceled == 0
    assert canceled.imag.rounded and not canceled.real.rounded
    assert canceled.to_real().rounded


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("imaginary", [False, True])
def test_component_overflow_is_explicit_and_inputs_unchanged(
    dtype: NumPyDType, imaginary: bool
) -> None:
    supported(dtype)
    maximum = dtype.component_dtype.floating_info.max
    z = make(0, maximum, dtype) if imaginary else make(maximum, 0, dtype)
    original = z.value
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        settings = np.geterr()
        with pytest.raises(OverflowError):
            _ = z + z
        with pytest.raises(OverflowError):
            _ = z - (-z)
        assert np.geterr() == settings
    assert z.value == original


def test_conversion_history_and_nonreal_rejection() -> None:
    z = make(Fraction(1, 2), Fraction(1, 4))
    assert z.to(C128) == z
    assert z.to(C128).to(C64) == z
    original = make(0.1, 0, C128)
    with pytest.raises(PrecisionLossError):
        original.to(C64)
    rounded = original.to(C64, approximate=True)
    assert rounded.real.rounded and not rounded.imag.rounded
    assert rounded.to(C128).rounded
    assert rounded.to(C128) != original
    assert make(rounded.real, rounded.imag, C128).rounded
    with pytest.raises(ValueError, match="zero imaginary"):
        z.to_real()
    real = make(Fraction(1, 2), -0.0).to_real()
    assert real.dtype == C64.component_dtype
    assert real.to_rational() == RationalValue(1, 2, dtype=ExactDType.rational())
    invalid_targets: list[Any] = [None, NumPyDType("float64"), ExactDType.rational()]
    for target in invalid_targets:
        with pytest.raises(TypeError):
            z.to(target)
    invalid: Any = 1
    with pytest.raises(TypeError):
        z.to(C64, approximate=invalid)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("real", [0.0, -0.0])
@pytest.mark.parametrize("imag", [0.0, -0.0])
def test_signed_zero_preserved(dtype: NumPyDType, real: float, imag: float) -> None:
    supported(dtype)
    z = make(real, imag, dtype)
    for copy in (z, z.to(C128), ComplexValue.approx(z.real, z.imag, dtype=dtype)):
        assert np.signbit(copy.value.real) == np.signbit(real)
        assert np.signbit(copy.value.imag) == np.signbit(imag)
        assert copy == 0
        assert hash(copy) == hash(0)
        assert not copy
    assert np.signbit(z.to_real().value) == np.signbit(real)
    assert np.signbit((-z).value.real) != np.signbit(real)
    assert np.signbit((-z).value.imag) != np.signbit(imag)
    assert np.signbit(z.conjugate().value.real) == np.signbit(real)
    assert np.signbit(z.conjugate().value.imag) != np.signbit(imag)


@pytest.mark.parametrize("dtype", DTYPES)
def test_extended_values_and_exact_subnormals_are_not_narrowed(
    dtype: NumPyDType,
) -> None:
    supported(dtype)
    info = dtype.component_dtype.floating_info
    re = Fraction(1) + Fraction(*info.eps.as_integer_ratio())
    im = info.smallest_subnormal
    z = make(re, im, dtype)
    assert z.value.real.as_integer_ratio() == (re.numerator, re.denominator)
    assert z.value.imag.as_integer_ratio() == im.as_integer_ratio()
    assert make(z.value.real, z.value.imag, dtype) == z
    assert hash(z) == hash(make(z.real, z.imag, dtype))
    if info.significand_bits > 53:
        with pytest.raises((PrecisionLossError, UnderflowError)):
            z.to(C128)
        if info.max_exponent > 1024:
            large = make(1 << 2000, 1 << 2001, dtype)
            assert large.value.real.as_integer_ratio() == (1 << 2000, 1)
            with pytest.raises(OverflowError):
                large.to(C128)


@pytest.mark.parametrize(
    ("real", "imag"), [(0, 0), (1, 0), (-1, 0), (Fraction(1, 2), 0), (0, 1), (1, -2)]
)
def test_classification_and_real_comparison(real: Any, imag: int) -> None:
    z = make(real, imag)
    assert z in Complex and z in Algebraic
    assert z not in Irrational and z not in Transcendental
    assert (z in Real) is (imag == 0)
    assert (z in Rational) is (imag == 0)
    integral = imag == 0 and Fraction(real).denominator == 1
    assert (z in Integer) is integral
    assert (z in Whole) is (integral and real >= 0)
    assert (z in Natural) is (integral and real > 0)
    assert "ComplexValue" in classify(z)[Real].reason
    peers: list[Any] = [
        FloatingValue(real, dtype=C64.component_dtype),
        RationalValue(
            Fraction(real).numerator,
            Fraction(real).denominator,
            dtype=ExactDType.rational(),
        ),
    ]
    if Fraction(real).denominator == 1:
        peers += [
            IntegerValue(real, dtype=ExactDType.integer()),
            real,
            np.int64(real),
            sp.Integer(real),
        ]
    for peer in peers:
        assert (z == peer) is (imag == 0)
        assert bool(peer == z) is (imag == 0)
        assert bool(operator.ne(z, peer)) is (imag != 0)
        assert bool(operator.ne(peer, z)) is (imag != 0)
        if imag == 0:
            assert hash(z) == hash(peer)
            assert len({z, peer}) == 1
    assert z != object()


@pytest.mark.parametrize(
    ("real", "imag"), [(1, 2), (-1, -2), (0.5, 0.25), (2**53, 1), (-(2**53), 0.5)]
)
def test_complex_equality_and_hash_convention(real: Any, imag: Any) -> None:
    a, b = make(real, imag, C128), make(real, imag, C64)
    assert a == b
    assert hash(a) == hash(b) == hash(complex(real, imag))
    assert len({a, b}) == 1
    assert {a: "value"}[b] == "value"
    assert a != make(real, -imag, C128)


@pytest.mark.parametrize(
    "operation", [operator.add, operator.sub, operator.mul, operator.truediv]
)
@pytest.mark.parametrize(
    "other",
    [
        0,
        1,
        True,
        1.0,
        1 + 0j,
        np.int64(1),
        np.complex64(1),
        sp.Integer(0),
        sp.Integer(1),
        sp.I,
        np.array([1]),
        make(1, dtype=C128),
        FloatingValue(1, dtype=C64.component_dtype),
    ],
)
def test_mixed_arithmetic_raises(operation: Any, other: Any) -> None:
    z = make(1)
    with pytest.raises(TypeError):
        operation(z, other)
    with pytest.raises(TypeError):
        operation(other, z)


@pytest.mark.parametrize(
    "other",
    [
        True,
        1.0,
        1 + 0j,
        np.float32(1),
        np.complex64(1),
        sp.Float(1),
        sp.I,
        Fraction(1),
        sp.Rational(1, 2),
    ],
)
@pytest.mark.parametrize("imag", [0, 1])
def test_unsupported_numeric_equality_always_raises(other: Any, imag: int) -> None:
    z = make(1, imag)
    for operation in (operator.eq, operator.ne):
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(z, other)
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(other, z)


def test_deferred_operations_and_backend_boundaries() -> None:
    z: Any = make(1, 2)
    for operation in (operator.pow, operator.lt):
        with pytest.raises(TypeError):
            operation(z, z)
    for conversion in (int, float, complex, abs):
        with pytest.raises(TypeError):
            conversion(z)
    with pytest.raises(sp.SympifyError):
        sp.sympify(z)
    for ufunc in (np.add, np.subtract, np.multiply, np.divide, np.conjugate):
        with pytest.raises(TypeError, match="ufunc"):
            if ufunc is np.conjugate:
                ufunc(z)
            else:
                ufunc(z, z)
    real: Any = make(1)
    assert real == np.array(1)
    assert np.array(1) == real
    for other in (np.array([1]), np.array(1.0), np.array(1 + 0j)):
        with pytest.raises(TypeError):
            _ = real == other
        with pytest.raises(TypeError):
            _ = other == real
    with pytest.raises(TypeError):
        np.equal(real, real, where=True)
    for attribute, replacement in (
        ("real", real.real),
        ("imag", real.imag),
        ("dtype", C128),
        ("_value", np.complex64(0)),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(z, attribute, replacement)
    assert "ComplexValue" in repr(z)


def assert_nearest(actual: FloatingValue, exact: sp.Rational) -> None:
    """Compare against the neighboring representable value, not our converter."""
    ratio = Fraction(int(exact.p), int(exact.q))
    stored = Fraction(*actual.value.as_integer_ratio())
    if stored == ratio:
        assert not actual.rounded
        return
    assert actual.rounded
    scalar = actual.dtype.numpy_dtype.type
    direction = scalar("inf" if ratio > stored else "-inf")
    with np.errstate(all="ignore"):
        neighbor = np.nextafter(actual.value, direction)
    other = Fraction(*neighbor.as_integer_ratio())
    assert abs(stored - ratio) <= abs(other - ratio)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("operation", [operator.mul, operator.truediv])
@pytest.mark.parametrize(
    "parts", [(1, 2, 3, -4), (-5, 7, 2, 3), (0, 1, -3, 0), (7, 0, 0, -2)]
)
def test_products_and_quotients_against_symbolic_oracle(
    dtype: NumPyDType, operation: Any, parts: tuple[int, int, int, int]
) -> None:
    supported(dtype)
    a, b, c, d = parts
    left, right = make(a, b, dtype), make(c, d, dtype)
    expected = sp.expand_complex(
        operation(sp.Integer(a) + sp.I * b, sp.Integer(c) + sp.I * d)
    )
    real, imag = expected.as_real_imag()
    result = operation(left, right)
    assert result.dtype == dtype
    assert_nearest(result.real, real)
    assert_nearest(result.imag, imag)
    reflected = (
        right.__rmul__(left) if operation is operator.mul else right.__rtruediv__(left)
    )
    assert reflected == result
    assert left == make(a, b, dtype)
    assert right == make(c, d, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_product_cancellation_rounds_only_final_components(dtype: NumPyDType) -> None:
    supported(dtype)
    eps = Fraction(*dtype.component_dtype.floating_info.eps.as_integer_ratio())
    result = make(1 + eps, 1, dtype) * make(1 - eps, 1, dtype)
    assert result == make(-eps * eps, 2, dtype)
    assert not result.rounded
    # The real intermediate product would round to 1 before subtracting 1.
    for sign in (-1, 1):
        tied = make(sign * Fraction(3, 2), 0, dtype) * make(1 + eps, 0, dtype)
        assert tied == make(sign * (Fraction(3, 2) + 2 * eps), 0, dtype)
        assert tied.real.rounded and not tied.imag.rounded


@pytest.mark.parametrize("dtype", DTYPES)
def test_intermediate_overflow_does_not_reject_finite_results(
    dtype: NumPyDType,
) -> None:
    supported(dtype)
    maximum = Fraction(*dtype.component_dtype.floating_info.max.as_integer_ratio())
    left = make(maximum, maximum / 4, dtype)
    right = make(Fraction(9, 8), Fraction(1, 2), dtype)
    # ac exceeds the format range, but ac - bd is exactly the maximum.
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        settings = np.geterr()
        result = left * right
        assert result.real == left.real
        assert_nearest(
            result.imag, sp.Rational(25 * maximum.numerator, 32 * maximum.denominator)
        )
        huge = make(maximum, maximum, dtype)
        assert huge / huge == make(1, 0, dtype)
        assert huge / huge.conjugate() == make(0, 1, dtype)
        assert np.geterr() == settings


@pytest.mark.parametrize("dtype", DTYPES)
def test_intermediate_underflow_does_not_reject_exact_results(
    dtype: NumPyDType,
) -> None:
    supported(dtype)
    info = dtype.component_dtype.floating_info
    step = Fraction(*info.smallest_subnormal.as_integer_ratio())
    tiny = Fraction(*info.smallest_normal.as_integer_ratio())
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        settings = np.geterr()
        smallest = make(step, step, dtype)
        result = smallest * make(Fraction(1, 2), Fraction(1, 2), dtype)
        assert result == make(0, step, dtype)
        assert not result.rounded
        assert smallest / smallest == make(1, 0, dtype)
        assert make(1, 0, dtype) / make(tiny, 0, dtype) == make(1 / tiny, 0, dtype)
        assert make(tiny, 0, dtype) / make(2, 0, dtype) == make(tiny / 2, 0, dtype)
        assert np.geterr() == settings


@pytest.mark.parametrize("dtype", DTYPES)
def test_final_range_failures_are_not_hidden_by_exact_intermediates(
    dtype: NumPyDType,
) -> None:
    supported(dtype)
    info = dtype.component_dtype.floating_info
    maximum = Fraction(*info.max.as_integer_ratio())
    step = Fraction(*info.smallest_subnormal.as_integer_ratio())
    tiny = Fraction(*info.smallest_normal.as_integer_ratio())
    eps = Fraction(*info.eps.as_integer_ratio())
    for operation in (
        lambda: make(maximum, 0, dtype) * make(2, 0, dtype),
        lambda: make(0, maximum, dtype) * make(2, 0, dtype),
        lambda: make(maximum, 0, dtype) / make(Fraction(1, 2), 0, dtype),
        lambda: make(0, maximum, dtype) / make(Fraction(1, 2), 0, dtype),
        lambda: make(1, 0, dtype) / make(step, 0, dtype),
        # Strict range check even when the excess could round back to max.
        lambda: make(maximum, step, dtype) * make(1, -step, dtype),
    ):
        with pytest.raises(OverflowError):
            operation()
    for operation in (
        lambda: make(step, step, dtype) * make(Fraction(1, 4), Fraction(1, 4), dtype),
        lambda: make(step, 0, dtype) / make(2, 0, dtype),
        lambda: make(tiny, 0, dtype) / make(3, 0, dtype),
        lambda: make(0, tiny, dtype) / make(3, 0, dtype),
        # Tininess before rounding: this would round up to smallest_normal.
        lambda: make(tiny, 0, dtype) * make(1 - eps / 2, 0, dtype),
    ):
        with pytest.raises(UnderflowError):
            operation()


@pytest.mark.parametrize("real", [0.0, -0.0])
@pytest.mark.parametrize("imag", [0.0, -0.0])
@pytest.mark.parametrize("numerator", [(0, 0), (1, 2)])
def test_all_signed_complex_zero_divisors_raise(
    real: float, imag: float, numerator: tuple[int, int]
) -> None:
    with pytest.raises(ZeroDivisionError):
        _ = make(*numerator) / make(real, imag)
    with pytest.raises(ZeroDivisionError):
        make(real, imag).__rtruediv__(make(*numerator))


@pytest.mark.parametrize("a", [0.0, -0.0])
@pytest.mark.parametrize("b", [0.0, -0.0])
@pytest.mark.parametrize("c", [0.0, -0.0, 1.0, -1.0])
@pytest.mark.parametrize("d", [0.0, -0.0, 1.0, -1.0])
def test_zero_signs_follow_exact_product_and_sum_formulas(
    a: float, b: float, c: float, d: float
) -> None:
    left, right = make(a, b), make(c, d)
    product = left * right
    # Only zero products are evaluated here, so the independent native-real
    # formula oracle cannot overflow, underflow, or round a nonzero value.
    assert np.signbit(product.value.real) == np.signbit(a * c - b * d)
    assert np.signbit(product.value.imag) == np.signbit(a * d + b * c)
    assert product == 0
    assert not product.rounded
    if c or d:
        quotient = left / right
        assert np.signbit(quotient.value.real) == np.signbit(
            (a * c + b * d) / (c * c + d * d)
        )
        assert np.signbit(quotient.value.imag) == np.signbit(
            (b * c - a * d) / (c * c + d * d)
        )
        assert quotient == 0
        assert not quotient.rounded


@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_cancellation_has_positive_zero(dtype: NumPyDType) -> None:
    supported(dtype)
    z = make(-1, -2, dtype)
    assert not np.signbit((z / z).value.imag)
    assert not np.signbit((z * z.conjugate()).value.imag)


def test_product_and_quotient_propagate_all_input_history() -> None:
    third = ComplexValue.approx(0, Fraction(1, 3), dtype=C64)
    one = make(1)
    for result in (third * one, one * third, third / one, one / third, third * make(0)):
        assert result.real.rounded
        assert result.imag.rounded
    result = make(1) / make(3)
    assert result.real.rounded and not result.imag.rounded
    assert (result * make(0)).to_real().rounded
