"""Floating precision, exceptional results, provenance, and backend boundaries."""

import operator
import sys
import warnings
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    Algebraic,
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
    UnderflowError,
    Whole,
    classify,
)

DTYPES = list(
    dict.fromkeys(
        NumPyDType(t) for t in (np.float16, np.float32, np.float64, np.longdouble)
    )
)
F32 = NumPyDType("float32")
F64 = NumPyDType("float64")


def ratio(value: np.floating[Any]) -> Fraction:
    return Fraction(*value.as_integer_ratio())


def make(value: Any, dtype: NumPyDType = F32) -> FloatingValue:
    return FloatingValue(value, dtype=dtype)


def supported(dtype: NumPyDType) -> None:
    # Extended formats are explicitly platform dependent; IBM double-double is
    # rejected by contract, not accidentally interpreted as a regular lattice.
    if dtype.floating_info.significand_bits not in (11, 24, 53, 64, 113):
        with pytest.raises(NotImplementedError):
            make(1, dtype)
        pytest.skip("Unsupported extended floating format on this platform")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "value",
    [
        3,
        np.int64(3),
        np.uint64(3),
        sp.Integer(3),
        3.0,
        np.float64(3),
        Fraction(3),
        sp.Rational(6, 2),
        IntegerValue(3, dtype=ExactDType.integer()),
        RationalValue(6, 2, dtype=ExactDType.rational()),
    ],
)
def test_exact_inputs_and_backend(dtype: NumPyDType, value: Any) -> None:
    supported(dtype)
    result = make(value, dtype)
    assert isinstance(result.value, np.floating)
    assert result.value.dtype == dtype.numpy_dtype
    assert result == 3
    assert result.dtype == dtype
    assert not result.rounded
    assert make(result, dtype) == result


@pytest.mark.parametrize("dtype", DTYPES)
def test_exact_limits_and_subnormals(dtype: NumPyDType) -> None:
    supported(dtype)
    info = dtype.floating_info
    for value in (
        info.max,
        -info.max,
        info.smallest_normal,
        info.smallest_subnormal,
        -info.smallest_subnormal,
    ):
        assert ratio(make(value, dtype).value) == ratio(value)
        assert ratio(make(Fraction(*value.as_integer_ratio()), dtype).value) == ratio(
            value
        )
    for value in (ratio(info.max) + 1, -ratio(info.max) - 1, 10**10000):
        for constructor in (FloatingValue, FloatingValue.approx):
            with pytest.raises(OverflowError):
                constructor(value, dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_precision_loss_and_single_nearest_even_rounding(dtype: NumPyDType) -> None:
    supported(dtype)
    spacing = ratio(dtype.floating_info.eps)
    # Midpoints on both sides, with both even and odd lower significands.
    for sign in (-1, 1):
        for offset, expected in ((Fraction(1, 2), 0), (Fraction(3, 2), 2)):
            exact = sign * (1 + offset * spacing)
            with pytest.raises(PrecisionLossError):
                make(exact, dtype)
            result = FloatingValue.approx(exact, dtype=dtype)
            assert ratio(result.value) == sign * (1 + expected * spacing)
            assert result.rounded
    # This would double-round to 1 if first converted through a narrower float.
    exact = 1 + spacing / 2 + spacing / 2**100
    assert ratio(FloatingValue.approx(exact, dtype=dtype).value) == 1 + spacing
    assert not FloatingValue.approx(Fraction(1, 2), dtype=dtype).rounded


@pytest.mark.parametrize("dtype", DTYPES)
def test_underflow_including_round_up_to_normal(dtype: NumPyDType) -> None:
    supported(dtype)
    tiny = ratio(dtype.floating_info.smallest_normal)
    step = ratio(dtype.floating_info.smallest_subnormal)
    for exact in (step / 2, 3 * step / 2, tiny - step / 4):
        for sign in (-1, 1):
            for constructor in (FloatingValue, FloatingValue.approx):
                with pytest.raises(UnderflowError):
                    constructor(sign * exact, dtype=dtype)
    assert ratio(make(tiny - step, dtype).value) == tiny - step


@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(False),
        sp.true,
        1 + 0j,
        np.complex64(1),
        "0.5",
        Decimal("0.5"),
        sp.Float("0.5"),
        sp.Symbol("x"),
        sp.sqrt(2),
        sp.oo,
        np.array(1.0),
        np.timedelta64(1, "D"),
        object(),
    ],
)
def test_unsupported_inputs_are_not_coerced(value: Any) -> None:
    for constructor in (FloatingValue, FloatingValue.approx):
        with pytest.raises(TypeError):
            constructor(value, dtype=F32)


@pytest.mark.parametrize(
    "scalar", [float, np.float16, np.float32, np.float64, np.longdouble]
)
@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_inputs_raise(scalar: Any, value: str) -> None:
    for constructor in (FloatingValue, FloatingValue.approx):
        with pytest.raises(ValueError):
            constructor(scalar(value), dtype=F32)


@pytest.mark.parametrize(
    "dtype",
    [
        None,
        "float32",
        np.float32,
        np.dtype("float32"),
        ExactDType.rational(),
        NumPyDType("int32"),
        NumPyDType("complex64"),
    ],
)
def test_dtype_validation(dtype: Any) -> None:
    with pytest.raises(TypeError):
        FloatingValue(1, dtype=dtype)


def test_nonnative_and_unsupported_formats_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    order = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="native byte order"):
        make(1, NumPyDType(f"{order}f8"))
    info = F64.floating_info
    for changed in (
        replace(info, significand_bits=106),
        replace(info, smallest_subnormal=info.smallest_normal),
    ):
        monkeypatch.setattr(
            NumPyDType, "floating_info", property(lambda self, info=changed: info)
        )
        with pytest.raises(NotImplementedError):
            make(1, F64)


def test_no_arbitrary_conversion_or_implicit_dtype() -> None:
    class PretendFloat:
        def __float__(self) -> float:
            raise AssertionError("Must not call duck conversion")

    with pytest.raises(TypeError):
        make(PretendFloat())
    constructor: Any = FloatingValue
    with pytest.raises(TypeError):
        constructor(1)


@pytest.mark.parametrize("dtype", DTYPES)
def test_arithmetic_rounding_and_history(dtype: NumPyDType) -> None:
    supported(dtype)
    a, b = make(1, dtype), make(3, dtype)
    third = a / b
    assert third.rounded
    assert third.dtype == dtype
    assert ratio(third.value) == ratio(
        dtype.numpy_dtype.type(1) / dtype.numpy_dtype.type(3)
    )
    assert (a + b) == 4
    assert (a - b) == -2
    assert (a * b) == 3
    assert not (a + b).rounded
    assert (third - third).rounded
    assert (third - third) == 0
    assert third.to(dtype).rounded
    assert (+third) is third
    assert (-third).rounded
    assert operator.neg(-third) == third
    assert b.__rsub__(a) == -2
    assert b.__rtruediv__(a) == third
    # A tie rounded away from the odd lower significand.
    eps = make(dtype.floating_info.eps, dtype)
    half_eps = make(ratio(dtype.floating_info.eps) / 2, dtype)
    assert a + half_eps == a
    assert (a + half_eps).rounded
    assert (a + eps) + half_eps == a + eps + eps


@pytest.mark.parametrize("dtype", DTYPES)
def test_arithmetic_errors_do_not_depend_on_numpy_settings(dtype: NumPyDType) -> None:
    supported(dtype)
    info = dtype.floating_info
    maximum, two = make(info.max, dtype), make(2, dtype)
    smallest = make(info.smallest_subnormal, dtype)
    for mode in ("ignore", "warn", "raise"):
        with warnings.catch_warnings(), np.errstate(all=mode):
            warnings.simplefilter("error")
            settings = np.geterr()
            for operation in (
                lambda: maximum * two,
                lambda: maximum + maximum,
                lambda: -maximum - maximum,
            ):
                with pytest.raises(OverflowError):
                    operation()
            for operation in (
                lambda: smallest / two,
                lambda: smallest * make(Fraction(1, 2), dtype),
            ):
                with pytest.raises(UnderflowError):
                    operation()
            for zero in (0.0, -0.0):
                with pytest.raises(ZeroDivisionError):
                    _ = make(0, dtype) / make(zero, dtype)
            assert smallest * two == make(2 * ratio(info.smallest_subnormal), dtype)
            assert np.geterr() == settings


def test_casting_preserves_stored_value_and_rounding_history() -> None:
    decimal_tenth = Fraction(1, 10)
    original = make(0.1, F64)
    assert not original.rounded  # No claim about accuracy of the supplied float.
    with pytest.raises(PrecisionLossError):
        make(decimal_tenth, F64)
    assert FloatingValue.approx(decimal_tenth, dtype=F64).rounded
    with pytest.raises(PrecisionLossError):
        original.to(F32)
    narrowed = original.to(F32, approximate=True)
    assert narrowed.rounded
    assert narrowed.to(F64).rounded
    assert narrowed.to(F64) != original
    with pytest.raises(TypeError):
        invalid_policy: Any = 1
        original.to(F32, approximate=invalid_policy)


def test_extended_precision_never_narrows_through_python_float() -> None:
    dtype = NumPyDType(np.longdouble)
    supported(dtype)
    if dtype.floating_info.significand_bits <= 53:
        pytest.skip("longdouble has no additional precision on this platform")
    exact = Fraction(1) + ratio(dtype.floating_info.eps)
    value = make(exact, dtype)
    assert ratio(value.value) == exact
    assert make(value.value, dtype) == value
    assert value != make(1, dtype)
    assert hash(value) == hash(exact)
    with pytest.raises(PrecisionLossError):
        value.to(F64)
    assert value.to(F64, approximate=True) == 1
    if dtype.floating_info.max_exponent > 1024:
        large = make(1 << 2000, dtype)
        assert large.to_integer(ExactDType.integer()) == 1 << 2000
        with pytest.raises(OverflowError):
            large.to(F64)


@pytest.mark.parametrize("dtype", DTYPES)
def test_signed_zero(dtype: NumPyDType) -> None:
    supported(dtype)
    pos, neg, one = make(0.0, dtype), make(-0.0, dtype), make(1, dtype)
    assert pos == neg
    assert hash(pos) == hash(neg) == hash(0)
    assert not pos and not neg
    assert np.signbit(neg.value)
    assert not np.signbit((-neg).value)
    assert np.signbit(neg.to(F64).value)
    assert np.signbit((neg * one).value)
    assert np.signbit((pos / -one).value)
    assert not np.signbit((one - one).value)
    assert np.signbit((neg + neg).value)
    assert neg.to_rational() == 0


@pytest.mark.parametrize(("p", "q"), [(0, 1), (1, 1), (-3, 1), (1, 2), (3, 2)])
def test_membership_and_exact_wrapper_interoperability(p: int, q: int) -> None:
    value = make(Fraction(p, q))
    exact = RationalValue(p, q, dtype=ExactDType.rational())
    assert value == exact
    assert exact == value
    assert hash(value) == hash(exact)
    assert len({value, exact}) == 1
    assert value.to_rational() == exact
    for domain in (Rational, Algebraic):
        assert value in domain
    assert value not in Irrational
    assert (value in Integer) is (q == 1)
    assert (value in Whole) is (q == 1 and p >= 0)
    assert (value in Natural) is (q == 1 and p > 0)
    assert "stored binary value" in classify(value)[Rational].reason
    if q == 1:
        peers = [
            make(p, F64),
            IntegerValue(p, dtype=ExactDType.integer()),
            p,
            np.int64(p),
            sp.Integer(p),
        ]
        for other in peers:
            assert value == other
            assert other == value
            assert bool(operator.ne(value, other)) is False
            assert bool(operator.ne(other, value)) is False
            assert hash(value) == hash(other)
        assert value.to_integer(NumPyDType("int8")) == p
    else:
        with pytest.raises(ValueError):
            value.to_integer(ExactDType.integer())
    assert value != object()


def test_checked_integer_extraction() -> None:
    value = make(256)
    with pytest.raises(OverflowError):
        value.to_integer(NumPyDType("uint8"))
    with pytest.raises(TypeError):
        value.to_integer(F64)


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
        np.float32(0.5),
        np.int64(1),
        sp.Integer(0),
        sp.Integer(1),
        sp.Rational(1, 2),
        sp.Float(0.5),
        Fraction(1, 2),
        np.array([1]),
        make(1, F64),
        IntegerValue(1, dtype=ExactDType.integer()),
        RationalValue(1, 2, dtype=ExactDType.rational()),
    ],
)
def test_mixed_arithmetic_cannot_bypass_checks(operation: Any, other: Any) -> None:
    value = make(Fraction(1, 2))
    with pytest.raises(TypeError):
        operation(value, other)
    with pytest.raises(TypeError):
        operation(other, value)


@pytest.mark.parametrize(
    "other",
    [
        True,
        np.bool_(False),
        0.5,
        np.float32(0.5),
        np.longdouble(0.5),
        sp.Float(0.5),
        Fraction(1, 2),
        sp.Rational(1, 2),
        Decimal("0.5"),
        0.5 + 0j,
    ],
)
def test_raw_numeric_equality_requires_supported_integer_or_wrapping(
    other: Any,
) -> None:
    value = make(Fraction(1, 2))
    for operation in (operator.eq, operator.ne):
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(value, other)
        with pytest.raises((TypeError, sp.SympifyError)):
            operation(other, value)


def test_protocols_immutability_and_backend_boundaries() -> None:
    value: Any = make(1)
    for conversion in (int, float, operator.index):
        with pytest.raises(TypeError):
            conversion(value)
    for operation in (operator.floordiv, operator.mod, operator.pow, operator.lt):
        with pytest.raises(TypeError):
            operation(value, value)
    for ufunc in (np.add, np.subtract, np.multiply, np.divide):
        with pytest.raises(TypeError, match="ufunc"):
            ufunc(value, value)
    with pytest.raises(sp.SympifyError):
        sp.sympify(value)
    for attribute, replacement in (
        ("dtype", F64),
        ("rounded", True),
        ("_value", np.float32(2)),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, attribute, replacement)
    assert "rounded=False" in repr(value)
    assert np.array(1) == value
    assert value == np.array(1)
    for other in (np.array([1]), np.array(1.0)):
        with pytest.raises(TypeError):
            _ = value == other
        with pytest.raises(TypeError):
            _ = other == value
    with pytest.raises(TypeError):
        np.equal(value, value, where=True)


def test_backend_mismatch_fails_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    real_ldexp = np.ldexp
    monkeypatch.setattr(np, "ldexp", lambda value, exponent: np.float32(0))
    with pytest.raises(FloatingPointError, match="Backend"):
        make(1)
    monkeypatch.setattr(np, "ldexp", real_ldexp)
    value = make(1)

    def wrong_operation(a: Any, b: Any) -> Any:
        return Fraction(2) if isinstance(a, Fraction) else np.float32(3)

    with pytest.raises(FloatingPointError, match="Backend arithmetic"):
        value._binary(value, wrong_operation)
