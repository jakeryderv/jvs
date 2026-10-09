"""Checked elementwise arithmetic without broadcasting or metadata loss."""

import sys
import warnings
from fractions import Fraction
from typing import Any, assert_type, get_type_hints

import numpy as np
import pytest

from jvs.numeric import (
    ComplexValue,
    ExactDType,
    FixedDType,
    FloatingValue,
    IntegerValue,
    NumericBuffer,
    NumPyDType,
    PrecisionLossError,
    UnderflowError,
    add,
    divide,
    floating,
    multiply,
    subtract,
)
from jvs.numeric.casting import NumericValue

I8, I16 = NumPyDType("int8"), NumPyDType("int16")
F32, F64 = NumPyDType("float32"), NumPyDType("float64")
C64, C128 = NumPyDType("complex64"), NumPyDType("complex128")
OPERATIONS = [add, subtract, multiply]


def buffer(values: Any, dtype: NumPyDType) -> NumericBuffer:
    return NumericBuffer(np.array(values, dtype=dtype.numpy_dtype), dtype=dtype)


@pytest.mark.parametrize(
    ("operation", "expected"), [(add, 10), (subtract, 6), (multiply, 16)]
)
@pytest.mark.parametrize("shape", [(), (2, 3), (2, 0, 3)])
@pytest.mark.parametrize(
    ("left_type", "right_type"), [(I8, I16), (F64, I8), (C128, F32), (C64, C64)]
)
@pytest.mark.parametrize("target", [I16, F32, C128])
def test_values_shape_dtype_and_owned_output(
    operation: Any,
    expected: int,
    shape: tuple[int, ...],
    left_type: NumPyDType,
    right_type: NumPyDType,
    target: NumPyDType,
) -> None:
    left = buffer(np.full(shape, 8), left_type)
    right = buffer(np.full(shape, 2), right_type)
    result = operation(left, right, dtype=target)
    assert isinstance(result, NumericBuffer)
    assert result.shape == shape and result.dtype == target and not result.rounded
    for name in ("_data", "_rounding"):
        output = getattr(result, name)
        assert output.flags.owndata and output.flags.c_contiguous
        assert not output.flags.writeable
        for operand in (left, right):
            assert output is not getattr(operand, name)
            assert not np.shares_memory(output, getattr(operand, name))
    exported = result.to_numpy()
    exported[...] = 99
    for index in np.ndindex(shape):
        assert result[index] == expected
        assert left[index] == 8 and right[index] == 2


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("reverse", [False, True])
def test_operand_rounding_is_explicit_and_retained(
    operation: Any, reverse: bool
) -> None:
    left, right = buffer([0.1], F64), buffer([1], I8)
    if reverse:
        left, right = right, left
    with pytest.raises(PrecisionLossError) as error:
        operation(left, right, dtype=F32)
    assert "(0,)" in error.value.__notes__[0]
    result = operation(left, right, dtype=F32, approximate=True)
    value = result[0]
    assert isinstance(value, FloatingValue) and value.rounded and result.rounded
    assert not left.rounded and not right.rounded


@pytest.mark.parametrize("target", [F32, C64])
def test_arithmetic_rounding_does_not_require_approximate_flag(
    target: NumPyDType,
) -> None:
    result = add(buffer([2**24], NumPyDType("int64")), buffer([1], I8), dtype=target)
    assert result[0] == 2**24 and result.rounded
    cancellation = subtract(result, result, dtype=target)
    assert cancellation[0] == 0 and cancellation.rounded
    exact = subtract(result, result, dtype=NumPyDType("int64"))
    assert exact[0] == 0 and not exact.rounded


def test_complex_component_history_follows_scalar_dependencies() -> None:
    left = buffer([0.1 + 0.5j, 0.5 + 0.1j], C128).to(C64, approximate=True)
    zero = buffer([0, 0], C64)
    for operation in (add, subtract):
        result = operation(left, zero, dtype=C128)
        for index, flags in enumerate([(True, False), (False, True)]):
            value = result[index]
            assert isinstance(value, ComplexValue)
            assert (value.real.rounded, value.imag.rounded) == flags
    product = multiply(left, buffer([1, 1], C64), dtype=C128)
    for i in range(2):
        value = product[i]
        assert isinstance(value, ComplexValue)
        assert value.real.rounded and value.imag.rounded
        assert value == left[i]


def test_complex_product_rounds_only_final_components() -> None:
    eps = 2.0**-23
    result = multiply(
        buffer([complex(1 + eps, 1)], C128),
        buffer([complex(1 - eps, 1)], C64),
        dtype=C64,
    )
    assert result[0] == ComplexValue(-Fraction(1, 2**46), 2, dtype=C64)
    assert not result.rounded


@pytest.mark.parametrize(
    ("operation", "left", "right", "target", "error_type"),
    [
        (add, buffer([[1, 127]], I8), buffer([[1, 1]], I8), I8, OverflowError),
        (subtract, buffer([[1, -128]], I8), buffer([[1, 1]], I8), I8, OverflowError),
        (multiply, buffer([[1, 64]], I8), buffer([[1, 2]], I8), I8, OverflowError),
        (
            subtract,
            buffer([[1, 0]], I8),
            buffer([[1, 1]], I8),
            NumPyDType("uint8"),
            OverflowError,
        ),
        (
            add,
            buffer([[1, 2**64 - 1]], NumPyDType("uint64")),
            buffer([[1, 1]], I8),
            NumPyDType("uint64"),
            OverflowError,
        ),
        (subtract, buffer([[1, 256]], I16), buffer([[1, 256]], I16), I8, OverflowError),
        (add, buffer([[1, 1.5]], F64), buffer([[1, 1]], I8), I8, ValueError),
        (add, buffer([[1, 1j]], C64), buffer([[1, 1]], I8), F32, ValueError),
        (
            multiply,
            buffer([[1, np.finfo(np.float32).max]], F32),
            buffer([[1, 2]], I8),
            F32,
            OverflowError,
        ),
        (
            multiply,
            buffer([[1, 2.0**-149]], F32),
            buffer([[1, 0.5]], F32),
            F32,
            UnderflowError,
        ),
    ],
)
def test_failures_report_coordinate_and_preserve_inputs(
    operation: Any,
    left: NumericBuffer,
    right: NumericBuffer,
    target: NumPyDType,
    error_type: Any,
) -> None:
    before = [b.to_numpy() for b in (left, right)]
    previous_errors = np.geterr()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        with pytest.raises(error_type) as error:
            operation(left, right, dtype=target)
    assert "(0, 1)" in error.value.__notes__[0]
    assert np.geterr() == previous_errors
    for operand, original in zip((left, right), before, strict=True):
        np.testing.assert_array_equal(operand.to_numpy(), original)
        assert not operand.rounded


def test_failure_preserves_existing_history_and_stops_at_first_pair() -> None:
    left = buffer([0.1, np.finfo(np.float32).max, 0.1], F64).to(F32, approximate=True)
    before = left.to_numpy()
    with pytest.raises(OverflowError) as error:
        multiply(left, buffer([1, 2, 0.1], F64), dtype=F32)
    assert "(1,)" in error.value.__notes__[0]
    np.testing.assert_array_equal(left.to_numpy(), before)
    value = left[0]
    assert isinstance(value, FloatingValue) and value.rounded


def test_exact_subnormal_and_negative_zero_results() -> None:
    tiny = multiply(
        buffer([np.finfo(np.float32).tiny], F32), buffer([0.5], F32), dtype=F32
    )
    value = tiny[0]
    assert isinstance(value, FloatingValue)
    assert value.value.as_integer_ratio() == (1, 2**127) and not tiny.rounded
    negative_zero = multiply(buffer([-0.0], F64), buffer([2], I8), dtype=F32)[0]
    assert isinstance(negative_zero, FloatingValue) and np.signbit(negative_zero.value)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("left_shape", "right_shape"),
    [((2, 1), (1, 2)), ((), (1,)), ((0,), (2, 0)), ((2,), (3,))],
)
def test_shape_mismatches_do_not_broadcast(
    operation: Any, left_shape: tuple[int, ...], right_shape: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        operation(
            buffer(np.zeros(left_shape), I8),
            buffer(np.zeros(right_shape), I8),
            dtype=I8,
        )


@pytest.mark.parametrize("operation", OPERATIONS)
def test_public_boundary_and_empty_policy_validation(operation: Any) -> None:
    empty = buffer([], I8)
    for other in (IntegerValue(1, dtype=I8), 1, np.array([], dtype=np.int8)):
        for left, right in ((empty, other), (other, empty)):
            with pytest.raises(TypeError, match="two NumericBuffer"):
                operation(left, right, dtype=I8)
    for dtype in (
        "int8",
        np.dtype("int8"),
        ExactDType.integer(),
        ExactDType.rational(),
        FixedDType(I8, step=Fraction(1, 10)),
    ):
        with pytest.raises(TypeError, match="NumPyDType"):
            operation(empty, empty, dtype=dtype)
    for flag in (1, None, np.bool_(True)):
        with pytest.raises(TypeError, match="bool"):
            operation(empty, empty, dtype=F32, approximate=flag)
    with pytest.raises(ValueError, match="Approximation"):
        operation(empty, empty, dtype=I8, approximate=True)
    order = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="native"):
        operation(empty, empty, dtype=NumPyDType(order + "f8"))
    with pytest.raises(TypeError):
        operation(empty, empty)
    with pytest.raises(TypeError):
        operation(empty, empty, I8)


def test_empty_buffers_validate_both_source_formats_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    integer, floating_buffer = buffer([], I8), buffer([], F64)
    original = floating._format

    def reject(dtype: NumPyDType) -> Any:
        if dtype == F64:
            raise NotImplementedError("unsupported test format")
        return original(dtype)

    monkeypatch.setattr(floating, "_format", reject)
    for left, right, dtype in (
        (integer, floating_buffer, I8),
        (floating_buffer, integer, I8),
        (integer, integer, F64),
    ):
        with pytest.raises(NotImplementedError):
            add(left, right, dtype=dtype)


def test_division_remains_scalar_only() -> None:
    left: Any = buffer([1], I8)
    for target in (I8, F32, C64, ExactDType.rational()):
        with pytest.raises(TypeError, match="Buffer division"):
            divide(left, left, dtype=target)


def test_public_return_annotations_distinguish_scalars_and_buffers() -> None:
    left = buffer([1], I8)
    scalar = IntegerValue(1, dtype=I8)
    assert_type(add(left, left, dtype=I8), NumericBuffer)
    assert_type(subtract(left, left, dtype=I8), NumericBuffer)
    assert_type(multiply(left, left, dtype=I8), NumericBuffer)
    assert_type(add(scalar, scalar, dtype=I8), NumericValue)
    for operation in OPERATIONS:
        get_type_hints(operation)
