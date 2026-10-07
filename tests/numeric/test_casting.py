"""Cross-family casts and explicit loss policy at the public dispatch boundary."""

import sys
from fractions import Fraction
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
    PrecisionLossError,
    RationalValue,
    UnderflowError,
    cast,
)

I8 = NumPyDType("int8")
U8 = NumPyDType("uint8")
F32 = NumPyDType("float32")
F64 = NumPyDType("float64")
C64 = NumPyDType("complex64")
C128 = NumPyDType("complex128")
INTEGER = ExactDType.integer()
RATIONAL = ExactDType.rational()
FIXED = FixedDType(NumPyDType("int16"), step=Fraction(1, 10))
TARGETS = [
    (I8, IntegerValue),
    (INTEGER, IntegerValue),
    (RATIONAL, RationalValue),
    (F32, FloatingValue),
    (C64, ComplexValue),
    (FIXED, FixedValue),
]


def sources(value: int) -> list[Any]:
    return [
        IntegerValue(value, dtype=I8),
        IntegerValue(value, dtype=INTEGER),
        RationalValue(value, dtype=RATIONAL),
        FloatingValue(value, dtype=F64),
        ComplexValue(value, -0.0, dtype=C128),
        FixedValue(value, dtype=FIXED),
    ]


@pytest.mark.parametrize("value", [-12, 0, 12])
@pytest.mark.parametrize(("dtype", "wrapper"), TARGETS)
def test_every_source_target_pair_preserves_integral_values(
    value: int, dtype: Any, wrapper: Any
) -> None:
    for original in sources(value):
        result = cast(original, dtype)
        assert isinstance(result, wrapper)
        assert result.dtype == dtype
        assert result == original == value
        assert hash(result) == hash(original)
        assert cast(result, original.dtype) == original
        # Successful casts do not change the original representation.
        assert original.dtype in (I8, INTEGER, RATIONAL, F64, C128, FIXED)


@pytest.mark.parametrize(("dtype", "wrapper"), TARGETS)
def test_fractional_source_conversion_matrix(dtype: Any, wrapper: Any) -> None:
    inputs = [
        RationalValue(1, 2, dtype=RATIONAL),
        FloatingValue(0.5, dtype=F64),
        ComplexValue(0.5, 0, dtype=C128),
        FixedValue(Fraction(1, 2), dtype=FIXED),
    ]
    for original in inputs:
        if wrapper is IntegerValue:
            with pytest.raises(ValueError):
                cast(original, dtype)
        else:
            result = cast(original, dtype)
            assert isinstance(result, wrapper)
            assert result.dtype == dtype
            assert result == original


@pytest.mark.parametrize("dtype", [I8, INTEGER, RATIONAL, F32, F64])
@pytest.mark.parametrize("imag", [1, -1])
def test_nonreal_input_cannot_be_cast_to_real_target(dtype: Any, imag: int) -> None:
    value = ComplexValue(1, imag, dtype=C128)
    with pytest.raises(ValueError, match="imaginary"):
        cast(value, dtype)
    if dtype in (F32, F64):
        with pytest.raises(ValueError, match="imaginary"):
            cast(value, dtype, approximate=True)
    result = cast(value, C64)
    assert isinstance(result, ComplexValue)
    assert result == value


@pytest.mark.parametrize("dtype", [F32, C64])
@pytest.mark.parametrize(
    "value",
    [
        IntegerValue(2**24 + 1, dtype=INTEGER),
        RationalValue(1, 3, dtype=RATIONAL),
        FloatingValue(0.1, dtype=F64),
        ComplexValue(0.1, 0, dtype=C128),
    ],
)
def test_rounding_requires_explicit_approximation(value: Any, dtype: Any) -> None:
    with pytest.raises(PrecisionLossError):
        cast(value, dtype)
    result = cast(value, dtype, approximate=True)
    assert isinstance(result, (FloatingValue, ComplexValue))
    assert result.rounded
    assert result.dtype == dtype
    assert result != value
    widened = cast(result, C128 if dtype == C64 else F64)
    assert isinstance(widened, (FloatingValue, ComplexValue))
    assert widened == result
    assert widened.rounded


@pytest.mark.parametrize("dtype", [I8, U8, INTEGER, RATIONAL])
@pytest.mark.parametrize("value", sources(1))
def test_approximation_is_rejected_for_integer_and_rational_targets(
    value: Any, dtype: Any
) -> None:
    with pytest.raises(ValueError, match="Approximation"):
        cast(value, dtype, approximate=True)


@pytest.mark.parametrize(
    "value",
    [
        1,
        1.0,
        True,
        1 + 0j,
        np.int64(1),
        np.float32(1),
        np.complex64(1),
        Fraction(1),
        sp.Integer(1),
        sp.Rational(1, 2),
        np.array(1),
        None,
        object(),
    ],
)
def test_raw_values_are_not_implicitly_wrapped(value: Any) -> None:
    with pytest.raises(TypeError, match="wrapper"):
        cast(value, F64)


@pytest.mark.parametrize(
    "dtype", [None, "float64", float, np.float64, np.dtype("float64"), object()]
)
def test_raw_dtype_specifications_are_rejected(dtype: Any) -> None:
    with pytest.raises(TypeError, match="explicit"):
        cast(IntegerValue(1, dtype=INTEGER), dtype)


@pytest.mark.parametrize("policy", [0, 1, "yes", None, np.bool_(True)])
def test_approximation_policy_requires_bool(policy: Any) -> None:
    with pytest.raises(TypeError, match="bool"):
        cast(IntegerValue(1, dtype=INTEGER), F32, approximate=policy)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda x: IntegerValue(x, dtype=INTEGER),
        lambda x: RationalValue(x, dtype=RATIONAL),
        lambda x: FloatingValue(x, dtype=F64),
        lambda x: ComplexValue(x, 0, dtype=C128),
    ],
)
def test_integer_range_checked_for_all_sources(constructor: Any) -> None:
    with pytest.raises(OverflowError):
        cast(constructor(128), I8)
    with pytest.raises(OverflowError):
        cast(constructor(-1), U8)
    assert cast(constructor(127), I8) == 127
    assert cast(constructor(255), U8) == 255


@pytest.mark.parametrize("dtype", [F32, C64])
@pytest.mark.parametrize("approximate", [False, True])
def test_floating_range_policy_remains_strict(dtype: Any, approximate: bool) -> None:
    huge = IntegerValue(10**10000, dtype=INTEGER)
    with pytest.raises(OverflowError):
        cast(huge, dtype, approximate=approximate)
    tiny = RationalValue(1, 2**150, dtype=RATIONAL)
    with pytest.raises(UnderflowError):
        cast(tiny, dtype, approximate=approximate)
    exact_subnormal = RationalValue(1, 2**149, dtype=RATIONAL)
    result = cast(exact_subnormal, dtype, approximate=approximate)
    assert result == exact_subnormal
    assert isinstance(result, (FloatingValue, ComplexValue))
    assert not result.rounded


def test_nonreal_complex_cast_checks_imaginary_precision_and_range() -> None:
    value = ComplexValue(0, 0.1, dtype=C128)
    with pytest.raises(PrecisionLossError):
        cast(value, C64)
    result = cast(value, C64, approximate=True)
    assert isinstance(result, ComplexValue)
    assert not result.real.rounded
    assert result.imag.rounded
    huge_imag = ComplexValue(0, 2**200, dtype=C128)
    for approximate in (False, True):
        with pytest.raises(OverflowError):
            cast(huge_imag, C64, approximate=approximate)


def test_signed_zero_and_component_history_across_floating_families() -> None:
    zero = FloatingValue(-0.0, dtype=F32)
    wider = cast(zero, F64)
    assert isinstance(wider, FloatingValue)
    assert np.signbit(wider.value)
    z = cast(zero, C128)
    assert isinstance(z, ComplexValue)
    assert np.signbit(z.value.real)
    assert not np.signbit(z.value.imag)
    both_negative = ComplexValue(-0.0, -0.0, dtype=C128)
    narrowed = cast(both_negative, C64)
    assert isinstance(narrowed, ComplexValue)
    assert np.signbit(narrowed.value.real) and np.signbit(narrowed.value.imag)
    real = cast(narrowed, F64)
    assert isinstance(real, FloatingValue)
    assert np.signbit(real.value)
    third = ComplexValue.approx(0, Fraction(1, 3), dtype=C64)
    canceled = third - third
    assert canceled.imag.rounded and not canceled.real.rounded
    real = cast(canceled, F64)
    assert isinstance(real, FloatingValue)
    assert real.rounded
    reconstructed = cast(real, C64)
    assert isinstance(reconstructed, ComplexValue)
    assert reconstructed.real.rounded and not reconstructed.imag.rounded


@pytest.mark.parametrize("dtype", [I8, INTEGER, RATIONAL])
def test_exact_targets_explicitly_discard_sign_and_rounding_history(dtype: Any) -> None:
    rounded = FloatingValue.approx(Fraction(1, 3), dtype=F32)
    zero = -(rounded - rounded)
    assert zero.rounded and np.signbit(zero.value)
    for original in (zero, ComplexValue(zero, -0.0, dtype=C64)):
        exact = cast(original, dtype)
        assert isinstance(exact, (IntegerValue, RationalValue))
        assert exact == 0
        restored = cast(exact, F32)
        assert isinstance(restored, FloatingValue)
        assert not restored.rounded
        assert not np.signbit(restored.value)
        assert original.rounded  # Metadata extraction does not mutate the source.


def test_binary_ratio_extraction_does_not_claim_intended_exactness() -> None:
    third = RationalValue(1, 3, dtype=RATIONAL)
    rounded = cast(third, F32, approximate=True)
    assert isinstance(rounded, FloatingValue)
    binary = cast(rounded, RATIONAL)
    assert isinstance(binary, RationalValue)
    p, q = rounded.value.as_integer_ratio()
    assert binary == RationalValue(p, q, dtype=RATIONAL)
    assert binary != third
    restored = cast(binary, F32)
    assert isinstance(restored, FloatingValue)
    assert restored == rounded
    assert not restored.rounded and rounded.rounded


def test_extended_precision_and_range_survive_direct_casts() -> None:
    dtype = NumPyDType(np.longdouble)
    info = dtype.floating_info
    if info.significand_bits not in (53, 64, 113):
        with pytest.raises(NotImplementedError):
            cast(IntegerValue(1, dtype=INTEGER), dtype)
        pytest.skip("Unsupported extended format")
    if info.significand_bits <= 53:
        pytest.skip("No additional floating precision on this platform")
    numerator, denominator = (1 + info.eps).as_integer_ratio()
    original = RationalValue(numerator, denominator, dtype=RATIONAL)
    floating = cast(original, dtype)
    assert isinstance(floating, FloatingValue)
    assert floating.value.as_integer_ratio() == (numerator, denominator)
    assert cast(floating, RATIONAL) == original
    z = cast(floating, NumPyDType(np.clongdouble))
    assert isinstance(z, ComplexValue)
    assert cast(z, RATIONAL) == original
    if info.max_exponent > 1024:
        huge = IntegerValue(1 << 2000, dtype=INTEGER)
        assert cast(cast(huge, dtype), INTEGER) == huge


def test_non_native_targets_are_not_silently_normalized() -> None:
    order = ">" if sys.byteorder == "little" else "<"
    value = IntegerValue(1, dtype=INTEGER)
    for code in ("i4", "f8", "c16"):
        with pytest.raises(ValueError, match="native byte order"):
            cast(value, NumPyDType(f"{order}{code}"))
