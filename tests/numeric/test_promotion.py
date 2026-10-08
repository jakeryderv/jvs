"""Explicit descriptor promotion and complete operand-domain coverage."""

import sys
from fractions import Fraction
from itertools import product
from typing import Any, get_type_hints

import numpy as np
import pytest

from jvs.numeric import (
    Backend,
    ComplexValue,
    DType,
    ExactDType,
    FixedDType,
    FixedValue,
    FloatingValue,
    IntegerValue,
    NumericKind,
    NumPyDType,
    RationalValue,
    add,
    arithmetic,
    cast,
    common_dtype,
    divide,
    multiply,
)
from jvs.numeric.floating import _format

I8, U8 = NumPyDType("int8"), NumPyDType("uint8")
INTEGER, RATIONAL = ExactDType.integer(), ExactDType.rational()
F32, F64 = NumPyDType("float32"), NumPyDType("float64")
C64, C128 = NumPyDType("complex64"), NumPyDType("complex128")
FIXED = FixedDType(I8, step=Fraction(1, 10))
FIXED16 = FixedDType(NumPyDType("int16"), step=Fraction(1, 10))
INTEGERS = [
    NumPyDType(f"{kind}{bits}") for kind in ("int", "uint") for bits in (8, 16, 32, 64)
]
FAMILIES = [I8, RATIONAL, FIXED, F32, C64]
FAMILY_RESULTS = [
    [I8, RATIONAL, FIXED16, None, None],
    [RATIONAL, RATIONAL, RATIONAL, None, None],
    [FIXED16, RATIONAL, FIXED, None, None],
    [None, None, None, F32, C64],
    [None, None, None, C64, C64],
]


@pytest.mark.parametrize(("i", "j"), list(product(range(5), repeat=2)))
def test_complete_family_decision_table(i: int, j: int) -> None:
    left, right = FAMILIES[i], FAMILIES[j]
    expected = FAMILY_RESULTS[i][j]
    if expected is None:
        with pytest.raises(TypeError, match="explicit target"):
            common_dtype(left, right)
    else:
        assert common_dtype(left, right) == expected
        assert common_dtype(right, left) == expected


@pytest.mark.parametrize(("left", "right"), list(product(INTEGERS, repeat=2)))
def test_integer_range_union_and_exact_boundary_casts(
    left: NumPyDType, right: NumPyDType
) -> None:
    low = min(int(left.integer_info.min), int(right.integer_info.min))
    high = max(int(left.integer_info.max), int(right.integer_info.max))
    if low < 0 and high > 2**63 - 1:
        with pytest.raises(ValueError, match="explicit target"):
            common_dtype(left, right)
        return
    target = common_dtype(left, right)
    assert isinstance(target, NumPyDType)
    assert target == common_dtype(right, left)
    info = target.integer_info
    assert info.signed is (low < 0)
    assert int(info.min) <= low <= high <= int(info.max)
    if info.bits > 8:
        previous_bits = info.bits // 2
        previous_low = -(2 ** (previous_bits - 1)) if info.signed else 0
        previous_high = 2 ** (previous_bits - int(info.signed)) - 1
        assert low < previous_low or high > previous_high
    for source in (left, right):
        for value in (source.integer_info.min, source.integer_info.max):
            original = IntegerValue(value, dtype=source)
            converted = cast(original, target)
            assert converted == original and converted.dtype == target


@pytest.mark.parametrize(
    "dtype",
    [*INTEGERS, INTEGER, RATIONAL, FIXED, FixedDType(INTEGER, step=Fraction(3, 7))],
)
def test_explicit_exact_backends_and_idempotence(dtype: Any) -> None:
    assert common_dtype(dtype, dtype) == dtype
    assert common_dtype(dtype, RATIONAL) == common_dtype(RATIONAL, dtype) == RATIONAL
    if isinstance(dtype, FixedDType):
        expected = FixedDType(INTEGER, step=Fraction(1, dtype.step.denominator))
    else:
        expected = RATIONAL if dtype == RATIONAL else INTEGER
    assert common_dtype(dtype, INTEGER) == common_dtype(INTEGER, dtype) == expected


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (
            FIXED,
            FixedDType(I8, step=Fraction(1, 100)),
            FixedDType(NumPyDType("int16"), step=Fraction(1, 100)),
        ),
        (
            FixedDType(I8, step=Fraction(1, 6)),
            FIXED,
            FixedDType(NumPyDType("int16"), step=Fraction(1, 30)),
        ),
        (FixedDType(I8, step=2), I8, FixedDType(NumPyDType("int16"), step=1)),
        (
            FixedDType(U8, step=Fraction(1, 10)),
            U8,
            FixedDType(NumPyDType("uint16"), step=Fraction(1, 10)),
        ),
        (
            FixedDType(I8, step=Fraction(3, 10)),
            FixedDType(U8, step=Fraction(1, 4)),
            FixedDType(NumPyDType("int16"), step=Fraction(1, 20)),
        ),
        (
            FixedDType(INTEGER, step=Fraction(2, 3)),
            FixedDType(I8, step=Fraction(4, 5)),
            FixedDType(INTEGER, step=Fraction(2, 15)),
        ),
    ],
)
def test_fixed_step_fixtures_and_coefficient_coverage(
    left: Any, right: Any, expected: FixedDType
) -> None:
    target = common_dtype(left, right)
    assert target == common_dtype(right, left) == expected
    assert isinstance(target, FixedDType)
    for source in (left, right):
        step = source.step if isinstance(source, FixedDType) else Fraction(1)
        storage = source.coefficient_dtype if isinstance(source, FixedDType) else source
        assert (step / target.step).denominator == 1
        if isinstance(storage, NumPyDType):
            coefficients = [
                int(storage.integer_info.min),
                0,
                1,
                int(storage.integer_info.max),
            ]
        else:
            coefficients = [-(10**1000), 0, 1, 10**1000]
        for coefficient in coefficients:
            value = (
                FixedValue.from_coefficient(coefficient, dtype=source)
                if isinstance(source, FixedDType)
                else IntegerValue(coefficient, dtype=source)
            )
            converted = cast(value, target)
            assert isinstance(converted, FixedValue)
            assert converted == value and converted.dtype == target
            assert int(converted.coefficient) * target.step == coefficient * step
            assert not converted.rounded


def test_fixed_range_exhaustion_and_huge_steps_fail_without_fallback() -> None:
    a = FixedDType(NumPyDType("int64"), step=Fraction(1, 10))
    b = FixedDType(NumPyDType("int64"), step=Fraction(1, 100))
    for left, right in ((a, b), (b, a)):
        with pytest.raises(ValueError, match="explicit target"):
            common_dtype(left, right)
    huge = 10**5000
    huge_step = FixedDType(I8, step=Fraction(huge + 1, huge))
    with pytest.raises(ValueError) as error:
        common_dtype(huge_step, I8)
    assert len(str(error.value)) < 700
    result = common_dtype(huge_step, INTEGER)
    assert result == FixedDType(INTEGER, step=Fraction(1, huge))


def supported_inexact() -> list[NumPyDType]:
    result = []
    for scalar in (
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
    ):
        dtype = NumPyDType(scalar)
        component = (
            dtype.component_dtype if dtype.kind is NumericKind.COMPLEX else dtype
        )
        try:
            _format(component)
        except NotImplementedError:
            continue
        if dtype not in result:
            result.append(dtype)
    return result


@pytest.mark.parametrize(
    ("left", "right"), list(product(supported_inexact(), repeat=2))
)
def test_floating_component_coverage_and_boundary_casts(
    left: NumPyDType, right: NumPyDType
) -> None:
    target = common_dtype(left, right)
    assert isinstance(target, NumPyDType)
    assert target == common_dtype(right, left)
    if left == right:
        assert target == left
    complex_target = NumericKind.COMPLEX in (left.kind, right.kind)
    assert target.kind is (
        NumericKind.COMPLEX if complex_target else NumericKind.FLOATING
    )
    target_component = target.component_dtype if complex_target else target
    ti = target_component.floating_info
    for source in (left, right):
        component = (
            source.component_dtype if source.kind is NumericKind.COMPLEX else source
        )
        si = component.floating_info
        assert ti.significand_bits >= si.significand_bits
        assert Fraction(*ti.max.as_integer_ratio()) >= Fraction(
            *si.max.as_integer_ratio()
        )
        assert Fraction(*ti.smallest_subnormal.as_integer_ratio()) <= Fraction(
            *si.smallest_subnormal.as_integer_ratio()
        )
        for boundary in (
            si.min,
            si.max,
            si.smallest_normal,
            si.smallest_subnormal,
            -si.smallest_subnormal,
            component.numpy_dtype.type(-0.0),
        ):
            original = (
                ComplexValue(boundary, boundary, dtype=source)
                if source.kind is NumericKind.COMPLEX
                else FloatingValue(boundary, dtype=source)
            )
            result = cast(original, target)
            assert isinstance(result, (FloatingValue, ComplexValue))
            assert result == original and result.dtype == target and not result.rounded
            real = result.real if isinstance(result, ComplexValue) else result
            assert bool(np.signbit(real.value)) == bool(np.signbit(boundary))
            if isinstance(result, ComplexValue):
                assert bool(np.signbit(result.imag.value)) is (
                    bool(np.signbit(boundary))
                    if isinstance(original, ComplexValue)
                    else False
                )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (NumPyDType("float16"), F32, F32),
        (F32, F64, F64),
        (C64, C128, C128),
        (NumPyDType("float16"), C64, C64),
        (F64, C64, C128),
    ],
)
def test_portable_floating_selection_table(
    left: NumPyDType, right: NumPyDType, expected: NumPyDType
) -> None:
    assert common_dtype(left, right) == common_dtype(right, left) == expected


@pytest.mark.parametrize(
    "exact", [*INTEGERS, INTEGER, RATIONAL, FIXED, FixedDType(INTEGER, step=1)]
)
@pytest.mark.parametrize("approximate", [F32, C64])
def test_exact_approximate_mixtures_require_explicit_target(
    exact: Any, approximate: NumPyDType
) -> None:
    for a, b in ((exact, approximate), (approximate, exact)):
        with pytest.raises(TypeError, match="explicit target"):
            common_dtype(a, b)


class OtherDType(DType):
    name = "custom"
    backend = Backend.NUMPY
    kind = NumericKind.INTEGER
    storage_bits = 8
    supports_nonfinite = False


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        1,
        True,
        "int8",
        int,
        np.int8,
        np.dtype("int8"),
        np.int8(1),
        np.array(1),
        IntegerValue(1, dtype=I8),
        OtherDType(),
    ],
)
def test_only_explicit_supported_descriptors_are_accepted(invalid: Any) -> None:
    for left, right in ((invalid, I8), (I8, invalid), (invalid, invalid)):
        with pytest.raises(TypeError, match="descriptor"):
            common_dtype(left, right)


@pytest.mark.parametrize("kind", ["i2", "f8", "c16", "fixed"])
def test_non_native_descriptors_rejected_before_identity_and_family_selection(
    kind: str,
) -> None:
    order = ">" if sys.byteorder == "little" else "<"
    invalid = (
        FixedDType(NumPyDType(order + "i2"), step=Fraction(1, 10))
        if kind == "fixed"
        else NumPyDType(order + kind)
    )
    for other in (invalid, I8, RATIONAL, F64):
        for left, right in ((invalid, other), (other, invalid)):
            with pytest.raises(ValueError, match="native byte order"):
                common_dtype(left, right)


def test_unsupported_input_format_is_not_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = arithmetic._format

    def reject(dtype: NumPyDType) -> Any:
        if dtype == F64:
            raise NotImplementedError("unsupported test format")
        return original(dtype)

    monkeypatch.setattr(arithmetic, "_format", reject)
    for left, right in (
        (F64, F64),
        (C128, C128),
        (F64, F32),
        (F32, F64),
        (F64, INTEGER),
    ):
        with pytest.raises(NotImplementedError, match="test format"):
            common_dtype(left, right)


def test_unsupported_candidate_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    original = arithmetic._format

    def reject(dtype: NumPyDType) -> Any:
        if dtype == NumPyDType("float16"):
            raise NotImplementedError("unsupported candidate")
        return original(dtype)

    monkeypatch.setattr(arithmetic, "_format", reject)
    assert common_dtype(F32, F64) == F64


def test_no_covering_candidate_raises_instead_of_returning_lossy_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate exhaustion independently of the host's available extended format.
    monkeypatch.setattr(arithmetic, "_covers_float", lambda target, source: False)
    with pytest.raises(ValueError, match="explicit target"):
        common_dtype(F32, F64)


def test_promotion_is_opt_in_and_does_not_guarantee_result_range() -> None:
    a, b = IntegerValue(100, dtype=I8), IntegerValue(100, dtype=U8)
    target = common_dtype(a.dtype, b.dtype)
    with pytest.raises(TypeError):
        _ = a + b
    assert add(a, b, dtype=target) == 200
    assert target == NumPyDType("int16")
    with pytest.raises(OverflowError):
        add(a, a, dtype=common_dtype(a.dtype, a.dtype))
    assert isinstance(divide(a, b, dtype=target), RationalValue)
    fixed = FixedValue(1, dtype=FIXED)
    assert isinstance(
        multiply(fixed, fixed, dtype=common_dtype(FIXED, FIXED)), RationalValue
    )


def test_rounding_history_follows_existing_cast_rules() -> None:
    floating = FloatingValue.approx(Fraction(1, 3), dtype=F32)
    widened = cast(floating, common_dtype(F32, C128))
    assert isinstance(widened, ComplexValue) and widened.rounded
    assert widened.real == floating and not widened.imag
    fixed = FixedValue.approx(Fraction(1, 3), dtype=FIXED)
    target = common_dtype(FIXED, FixedDType(I8, step=Fraction(1, 100)))
    rescaled = cast(fixed, target)
    assert isinstance(rescaled, FixedValue) and rescaled.rounded and rescaled == fixed
    extracted = cast(fixed, common_dtype(FIXED, RATIONAL))
    assert isinstance(extracted, RationalValue) and extracted == fixed
    assert not FixedValue(extracted, dtype=FIXED).rounded
    assert floating.rounded and fixed.rounded


def test_documented_grouping_difference_and_runtime_annotations() -> None:
    fixed = FixedDType(I8, step=Fraction(1, 100))
    left_group = common_dtype(common_dtype(I8, U8), fixed)
    right_group = common_dtype(I8, common_dtype(U8, fixed))
    assert left_group == FixedDType(NumPyDType("int32"), step=Fraction(1, 100))
    assert right_group == FixedDType(NumPyDType("int16"), step=Fraction(1, 100))
    assert get_type_hints(common_dtype)["return"] is arithmetic.ScalarDType
