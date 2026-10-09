"""Direct buffer conversion preserves checked values, history, and ownership."""

import sys
import warnings
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from jvs.numeric import (
    ComplexValue,
    ExactDType,
    FixedDType,
    FloatingValue,
    NumericBuffer,
    NumPyDType,
    PrecisionLossError,
    UnderflowError,
    floating,
)

I8 = NumPyDType("int8")
F32, F64 = NumPyDType("float32"), NumPyDType("float64")
C64, C128 = NumPyDType("complex64"), NumPyDType("complex128")


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (3,), (2, 3)])
@pytest.mark.parametrize("source_type", ["int16", "uint16", "float64", "complex128"])
@pytest.mark.parametrize("target", ["int8", "uint8", "float32", "complex64"])
def test_conversion_preserves_shape_values_and_independent_storage(
    shape: tuple[int, ...], source_type: str, target: str
) -> None:
    raw = np.arange(int(np.prod(shape))).reshape(shape).astype(source_type)
    source = NumericBuffer(raw, dtype=NumPyDType(source_type))
    result = source.to(NumPyDType(target))
    assert result.shape == shape and result.dtype == NumPyDType(target)
    assert result is not source and not result.rounded
    for name in ("_data", "_rounding"):
        original, converted = getattr(source, name), getattr(result, name)
        assert converted.flags.owndata and converted.flags.c_contiguous
        assert not converted.flags.writeable
        assert converted is not original and not np.shares_memory(original, converted)
    first, second = result.to_numpy(), result.to_numpy()
    first[...] = 99
    np.testing.assert_array_equal(second, raw)
    np.testing.assert_array_equal(source.to_numpy(), raw)
    for index in np.ndindex(shape):
        assert result[index] == source[index]


@pytest.mark.parametrize("target", [C64, C128])
def test_same_dtype_and_widening_retain_each_component_history(
    target: NumPyDType,
) -> None:
    raw = np.array([0.5 + 0.5j, 0.1 + 0.5j, 0.5 + 0.1j, 0.1 + 0.1j])
    source = NumericBuffer(raw, dtype=C64, approximate=True)
    result = source.to(target)
    assert result is not source and result.rounded
    assert not np.shares_memory(source._data, result._data)
    assert not np.shares_memory(source._rounding, result._rounding)
    for buffer in (source, result, result.to(C64)):
        for i, flags in enumerate(
            [(False, False), (True, False), (False, True), (True, True)]
        ):
            value = buffer[i]
            assert isinstance(value, ComplexValue)
            assert (value.real.rounded, value.imag.rounded) == flags
            assert value == source[i]
    # The separate NumPy export boundary intentionally cannot retain history.
    assert not NumericBuffer(result.to_numpy(), dtype=target).rounded


def test_new_rounding_combines_with_history_and_exact_targets_discard_it() -> None:
    source = NumericBuffer(
        np.array([2**53 + 1, 2**24 + 1], dtype=np.int64),
        dtype=F64,
        approximate=True,
    )
    result = source.to(F32, approximate=True)
    for i, expected in enumerate((2**53, 2**24)):
        value = result[i]
        assert isinstance(value, FloatingValue) and value.rounded and value == expected
        complex_value = result.to(C128)[i]
        assert isinstance(complex_value, ComplexValue)
        assert complex_value.real.rounded and not complex_value.imag.rounded
    integral = result.to(NumPyDType("int64"))
    assert not integral.rounded
    assert integral[0] == 2**53 and integral[1] == 2**24
    assert not integral.to(F64).rounded
    real = result.to(C128).to(F64)
    assert real.rounded and real[1] == 2**24
    assert source[1] == 2**24 + 1


@pytest.mark.parametrize(
    ("raw", "target", "approximate", "error_type"),
    [
        (np.array([[1, 128]], dtype=np.int16), I8, False, OverflowError),
        (
            np.array([[1, -1]], dtype=np.int16),
            NumPyDType("uint8"),
            False,
            OverflowError,
        ),
        (np.array([[1, 1.5]]), I8, False, ValueError),
        (np.array([[1, 0.1]]), F32, False, PrecisionLossError),
        (np.array([[1, 2.0**128]]), F32, True, OverflowError),
        (np.array([[1, 2.0**-150]]), F32, True, UnderflowError),
        (np.array([[1, 1 + 1j]]), F32, True, ValueError),
    ],
)
def test_failed_conversion_reports_coordinate_and_leaves_source_unchanged(
    raw: Any, target: NumPyDType, approximate: bool, error_type: Any
) -> None:
    source = NumericBuffer(raw, dtype=NumPyDType(raw.dtype))
    previous_errors = np.geterr()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        with pytest.raises(error_type) as error:
            source.to(target, approximate=approximate)
    assert any("(0, 1)" in note for note in error.value.__notes__)
    np.testing.assert_array_equal(source.to_numpy(), raw)
    assert not source.rounded and np.geterr() == previous_errors


def test_failed_conversion_preserves_preexisting_rounding_history() -> None:
    source = NumericBuffer(
        np.array([2**53 + 1, 2**24 + 1], dtype=np.int64),
        dtype=F64,
        approximate=True,
    )
    before = source.to_numpy()
    with pytest.raises(PrecisionLossError) as error:
        source.to(F32)
    assert "(1,)" in error.value.__notes__[0]
    np.testing.assert_array_equal(source.to_numpy(), before)
    first, second = source[0], source[1]
    assert isinstance(first, FloatingValue) and first.rounded
    assert isinstance(second, FloatingValue) and not second.rounded


def test_large_unsigned_conversion_never_narrows_before_validation() -> None:
    source = NumericBuffer(
        np.array([2**64 - 1], dtype=np.uint64), dtype=NumPyDType("uint64")
    )
    with pytest.raises(OverflowError):
        source.to(NumPyDType("int64"))
    with pytest.raises(PrecisionLossError):
        source.to(F64)
    result = source.to(F64, approximate=True)
    assert result[0] == 2**64 and result.rounded
    assert source[0] == 2**64 - 1


def test_signed_zeros_and_exact_subnormals_survive_conversion() -> None:
    source = NumericBuffer(np.array([complex(-0.0, -0.0)]), dtype=C128)
    value = source.to(C64)[0]
    assert isinstance(value, ComplexValue)
    assert np.signbit(value.real.value) and np.signbit(value.imag.value)
    real = source.to(F32)[0]
    assert isinstance(real, FloatingValue) and np.signbit(real.value)
    tiny = NumericBuffer(np.array([2.0**-149]), dtype=F64).to(F32)
    assert not tiny.rounded and tiny.to(F64)[0] == tiny[0]


def test_extended_precision_conversion_avoids_python_float_narrowing() -> None:
    dtype = NumPyDType(np.longdouble)
    try:
        floating._format(dtype)
    except NotImplementedError:
        pytest.skip("Host extended format is not supported by scalar wrappers")
    raw = np.nextafter(np.longdouble(1), np.longdouble(2))
    source = NumericBuffer(np.array([raw], dtype=np.longdouble), dtype=dtype)
    value = source.to(dtype)[0]
    assert isinstance(value, FloatingValue)
    assert value.value.as_integer_ratio() == raw.as_integer_ratio()
    if np.finfo(np.longdouble).nmant > np.finfo(np.float64).nmant:
        with pytest.raises(PrecisionLossError):
            source.to(F64)
        rounded = source.to(F64, approximate=True)
        assert rounded[0] == 1 and rounded.rounded
        assert rounded.to(dtype).rounded


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3)])
@pytest.mark.parametrize(
    "target",
    [
        "float32",
        np.dtype("float32"),
        ExactDType.integer(),
        ExactDType.rational(),
        FixedDType(I8, step=Fraction(1, 10)),
        None,
    ],
)
def test_invalid_targets_fail_even_for_empty_buffers(
    shape: tuple[int, ...], target: Any
) -> None:
    source = NumericBuffer(np.zeros(shape), dtype=F64)
    with pytest.raises(TypeError, match="NumPyDType"):
        source.to(target)


@pytest.mark.parametrize("flag", [1, 0, None, np.bool_(True), "yes"])
def test_invalid_approximation_flag_rejected_for_empty_buffers(flag: Any) -> None:
    source = NumericBuffer(np.empty((0,)), dtype=F64)
    with pytest.raises(TypeError, match="bool"):
        source.to(F32, approximate=flag)


def test_empty_conversion_validates_integer_policy_byte_order_and_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = NumericBuffer(np.empty((0,)), dtype=F64)
    with pytest.raises(ValueError, match="Approximation"):
        source.to(I8, approximate=True)
    order = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="native"):
        source.to(NumPyDType(order + "f8"))
    original = floating._format

    def reject(dtype: NumPyDType) -> Any:
        if dtype == F32:
            raise NotImplementedError("unsupported test format")
        return original(dtype)

    monkeypatch.setattr(floating, "_format", reject)
    with pytest.raises(NotImplementedError):
        source.to(F32)
