"""Owned storage, checked scalar conversion, and detached NumPy exports."""

import inspect
import operator
import sys
import warnings
from dataclasses import FrozenInstanceError
from fractions import Fraction
from typing import Any, get_type_hints

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
    classify,
    floating,
)

I8 = NumPyDType("int8")
F32, F64 = NumPyDType("float32"), NumPyDType("float64")
C64 = NumPyDType("complex64")
TARGETS = [
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


@pytest.mark.parametrize("shape", [(), (0,), (2, 0, 3), (4,), (2, 3)])
@pytest.mark.parametrize("target", TARGETS)
def test_shape_scalar_extraction_and_exact_owned_storage(
    shape: tuple[int, ...], target: str
) -> None:
    source = np.arange(int(np.prod(shape)), dtype=np.int16).reshape(shape)
    buffer = NumericBuffer(source, dtype=NumPyDType(target))
    assert buffer.shape == shape and buffer.ndim == len(shape)
    assert buffer.size == source.size and not buffer.rounded
    if shape:
        assert len(buffer) == shape[0]
    else:
        with pytest.raises(TypeError):
            len(buffer)
    exported = buffer.to_numpy()
    assert exported.shape == shape and exported.dtype == np.dtype(target)
    assert (
        exported.flags.c_contiguous
        and exported.flags.writeable
        and exported.flags.owndata
    )
    assert buffer._data.flags.owndata and not buffer._data.flags.writeable
    assert buffer._rounding.flags.owndata and not buffer._rounding.flags.writeable
    assert not np.shares_memory(exported, buffer._data)
    wrapper = (
        IntegerValue
        if exported.dtype.kind in "iu"
        else FloatingValue
        if exported.dtype.kind == "f"
        else ComplexValue
    )
    for coordinate in np.ndindex(shape):
        value = buffer[coordinate]
        assert isinstance(value, wrapper)
        assert value.dtype == buffer.dtype and value == int(source[coordinate])
    assert "shape=" in repr(buffer) and "dtype=" in repr(buffer)


@pytest.mark.parametrize(
    "layout", ["transpose", "reverse", "stride", "fortran", "readonly"]
)
def test_source_aliases_and_each_export_are_independent(layout: str) -> None:
    base = np.arange(24, dtype=np.int32).reshape(4, 6)
    source = {
        "transpose": base.T,
        "reverse": base[::-1, ::-1],
        "stride": base[::2, ::2],
        "fortran": np.asfortranarray(base),
        "readonly": base.view(),
    }[layout]
    if layout == "readonly":
        source.setflags(write=False)
    expected = source.copy()
    buffer = NumericBuffer(source, dtype=NumPyDType("int64"))
    first, second = buffer.to_numpy(), buffer.to_numpy()
    assert not np.shares_memory(first, source)
    assert not np.shares_memory(first, second)
    base[:] = -1
    if source.flags.writeable:
        source[:] = -2
    first[:] = 99
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(buffer.to_numpy(), expected)
    assert buffer[0, 0] == int(expected[0, 0])


@pytest.mark.parametrize(
    "source_type", ["int64", "uint64", "float32", "float64", "complex64", "complex128"]
)
@pytest.mark.parametrize("target", ["int16", "float64", "complex128"])
def test_cross_family_conversion_matches_stored_values(
    source_type: str, target: str
) -> None:
    source = np.array([0, 1, 100], dtype=source_type)
    buffer = NumericBuffer(source, dtype=NumPyDType(target))
    for i, value in enumerate((0, 1, 100)):
        assert buffer[i] == value


def test_large_unsigned_values_do_not_narrow_before_checks() -> None:
    source = np.array([2**64 - 1], dtype=np.uint64)
    assert NumericBuffer(source, dtype=NumPyDType("uint64"))[0] == 2**64 - 1
    with pytest.raises(OverflowError):
        NumericBuffer(source, dtype=NumPyDType("int64"))
    with pytest.raises(PrecisionLossError):
        NumericBuffer(source, dtype=F64)
    rounded = NumericBuffer(source, dtype=F64, approximate=True)
    value = rounded[0]
    assert isinstance(value, FloatingValue)
    assert value == 2**64 and value.rounded and rounded.rounded


@pytest.mark.parametrize(
    ("source", "target", "approximate", "error_type"),
    [
        (np.array([[1, 128]], dtype=np.int16), I8, False, OverflowError),
        (
            np.array([[1, -1]], dtype=np.int16),
            NumPyDType("uint8"),
            False,
            OverflowError,
        ),
        (np.array([[1, 1.5]], dtype=np.float64), I8, False, ValueError),
        (np.array([[1, 0.1]], dtype=np.float64), F32, False, PrecisionLossError),
        (np.array([[1, 2.0**128]], dtype=np.float64), F32, True, OverflowError),
        (np.array([[1, 2.0**-150]], dtype=np.float64), F32, True, UnderflowError),
        (np.array([[1, 1 + 1j]], dtype=np.complex128), F32, True, ValueError),
        (np.array([[1, np.inf]], dtype=np.float64), F64, False, ValueError),
        (np.array([[1, np.nan]], dtype=np.float64), F64, True, ValueError),
        (
            np.array([[1, complex(1, np.inf)]], dtype=np.complex128),
            C64,
            False,
            ValueError,
        ),
    ],
)
def test_element_failures_keep_exception_type_and_add_coordinate(
    source: Any, target: NumPyDType, approximate: bool, error_type: Any
) -> None:
    before = source.copy()
    previous_errors = np.geterr()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        with pytest.raises(error_type) as error:
            NumericBuffer(source, dtype=target, approximate=approximate)
    assert any("(0, 1)" in note for note in error.value.__notes__)
    np.testing.assert_array_equal(source, before)
    assert np.geterr() == previous_errors


def test_rounding_metadata_is_retained_per_element_and_component() -> None:
    source = np.array(
        [complex(0.5, 0.5), complex(0.1, 0.5), complex(0.5, 0.1), complex(0.1, 0.1)],
        dtype=np.complex128,
    )
    buffer = NumericBuffer(source, dtype=C64, approximate=True)
    assert buffer.rounded
    for i, expected in enumerate(
        [(False, False), (True, False), (False, True), (True, True)]
    ):
        value = buffer[i]
        assert isinstance(value, ComplexValue)
        assert (value.real.rounded, value.imag.rounded) == expected
    exported = buffer.to_numpy()
    imported = NumericBuffer(exported, dtype=C64)
    assert not imported.rounded
    assert buffer.rounded


@pytest.mark.parametrize("complex_source", [False, True])
def test_signed_zeros_are_preserved_without_arithmetic_packing(
    complex_source: bool,
) -> None:
    source = (
        np.array([complex(-0.0, -0.0), complex(0.0, -0.0)])
        if complex_source
        else np.array([-0.0, 0.0])
    )
    buffer = NumericBuffer(source, dtype=C64)
    for i in range(2):
        value = buffer[i]
        assert isinstance(value, ComplexValue)
        assert bool(np.signbit(value.real.value)) is (i == 0)
        assert bool(np.signbit(value.imag.value)) is complex_source
    exported = buffer.to_numpy()
    np.testing.assert_array_equal(np.signbit(exported.real), [True, False])
    np.testing.assert_array_equal(
        np.signbit(exported.imag), [complex_source, complex_source]
    )
    integer = NumericBuffer(source, dtype=I8)
    assert integer[0] == integer[1] == 0 and not integer.rounded


def test_exact_subnormal_and_extended_precision_storage() -> None:
    source = np.array([2.0**-149], dtype=np.float64)
    tiny = NumericBuffer(source, dtype=F32)
    assert not tiny.rounded
    value = tiny[0]
    assert isinstance(value, FloatingValue)
    assert value.value.as_integer_ratio() == source[0].as_integer_ratio()
    dtype = NumPyDType(np.longdouble)
    try:
        floating._format(dtype)
    except NotImplementedError:
        pytest.skip("Host extended format is not supported by scalar wrappers")
    raw = np.nextafter(np.longdouble(1), np.longdouble(2))
    buffer = NumericBuffer(np.array([raw], dtype=np.longdouble), dtype=dtype)
    value = buffer[0]
    assert isinstance(value, FloatingValue)
    assert value.value.as_integer_ratio() == raw.as_integer_ratio()
    assert buffer.to_numpy()[0].as_integer_ratio() == raw.as_integer_ratio()


@pytest.mark.parametrize("kind", ["i4", "f8", "c16"])
def test_non_native_sources_are_decoded_but_non_native_targets_rejected(
    kind: str,
) -> None:
    order = ">" if sys.byteorder == "little" else "<"
    source = np.array([1, 2], dtype=order + kind)
    native = NumPyDType(np.dtype(kind).newbyteorder("="))
    buffer = NumericBuffer(source, dtype=native)
    assert buffer[0] == 1 and buffer[1] == 2
    for data in (source, source[:0]):
        with pytest.raises(ValueError, match="native"):
            NumericBuffer(data, dtype=NumPyDType(order + kind))


@pytest.mark.parametrize(
    "data",
    [
        [1, 2],
        1,
        np.int8(1),
        np.array([True]),
        np.array([1], dtype=object),
        np.array(["1"]),
        np.array([1], dtype="timedelta64[D]"),
        np.zeros(1, dtype=[("a", "i4")]),
        np.array([1], dtype=np.dtype("int32", metadata={"unit": "x"})),
        np.ma.array([1], mask=[True]),
    ],
)
def test_unsupported_input_representation_is_rejected(data: Any) -> None:
    with pytest.raises(TypeError):
        NumericBuffer(data, dtype=F64)


def test_array_subclasses_are_rejected_even_without_a_mask() -> None:
    class ArraySubclass(np.ndarray):
        pass

    data = np.array([1]).view(ArraySubclass)
    with pytest.raises(TypeError, match="plain"):
        NumericBuffer(data, dtype=I8)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        np.int8,
        np.dtype("int8"),
        ExactDType.integer(),
        ExactDType.rational(),
        FixedDType(I8, step=Fraction(1, 10)),
        None,
    ],
)
def test_only_explicit_numpy_targets_are_accepted(dtype: Any) -> None:
    with pytest.raises(TypeError, match="NumPyDType"):
        NumericBuffer(np.array([], dtype=np.int8), dtype=dtype)


@pytest.mark.parametrize("flag", [1, 0, None, np.bool_(True), "yes"])
def test_approximation_flag_must_be_python_bool_even_for_empty_buffers(
    flag: Any,
) -> None:
    with pytest.raises(TypeError, match="bool"):
        NumericBuffer(np.array([], dtype=np.float64), dtype=F32, approximate=flag)


def test_empty_buffers_do_not_bypass_policy_or_format_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Approximation"):
        NumericBuffer(np.array([], dtype=np.int8), dtype=I8, approximate=True)
    original = floating._format

    def reject(dtype: NumPyDType) -> Any:
        if dtype == F64:
            raise NotImplementedError("unsupported test format")
        return original(dtype)

    monkeypatch.setattr(floating, "_format", reject)
    with pytest.raises(NotImplementedError):
        NumericBuffer(np.array([], dtype=np.float64), dtype=I8)
    with pytest.raises(NotImplementedError):
        NumericBuffer(np.array([], dtype=np.int8), dtype=F64)


def test_complete_indices_and_negative_positions() -> None:
    buffer = NumericBuffer(np.arange(6, dtype=np.int8).reshape(2, 3), dtype=I8)
    assert buffer[-1, -1] == buffer[np.int64(1), np.uint64(2)] == 5
    vector = NumericBuffer(np.array([1, 2], dtype=np.int8), dtype=I8)
    assert vector[-1] == vector[-1,] == 2
    scalar = NumericBuffer(np.array(7, dtype=np.int8), dtype=I8)
    assert scalar[()] == 7


@pytest.mark.parametrize(
    "key",
    [
        True,
        np.bool_(False),
        0.0,
        slice(None),
        [0],
        Ellipsis,
        None,
        (0, slice(None)),
        (0, True),
        np.array([0]),
        (0, np.timedelta64(1, "ns")),
    ],
)
def test_index_types_cannot_enable_views_or_advanced_indexing(key: Any) -> None:
    buffer = NumericBuffer(np.ones((2, 2), dtype=np.int8), dtype=I8)
    with pytest.raises(TypeError):
        buffer[key]


@pytest.mark.parametrize("key", [0, (), (0, 0, 0), (-3, 0), (0, 2), (10**1000, 0)])
def test_partial_and_out_of_bounds_coordinates_raise(key: Any) -> None:
    buffer = NumericBuffer(np.ones((2, 2), dtype=np.int8), dtype=I8)
    with pytest.raises(IndexError):
        buffer[key]
    empty = NumericBuffer(np.empty((0,), dtype=np.int8), dtype=I8)
    with pytest.raises(IndexError):
        empty[0]


def test_public_boundary_prevents_mutation_and_implicit_operations() -> None:
    buffer: Any = NumericBuffer(np.array([1], dtype=np.int8), dtype=I8)
    for name, replacement in (
        ("dtype", F32),
        ("_data", np.array([99])),
        ("_rounding", np.array([1])),
    ):
        with pytest.raises((FrozenInstanceError, AttributeError)):
            setattr(buffer, name, replacement)
    with pytest.raises(TypeError):
        buffer[0] = 99
    for convert in (bool, iter, np.array, np.asarray):
        with pytest.raises(TypeError):
            convert(buffer)
    for operation in (np.add, np.equal, np.less, np.divide, operator.add):
        with pytest.raises(TypeError):
            operation(buffer, buffer)
    with pytest.raises(TypeError):
        classify(buffer)
    other = NumericBuffer(np.array([1], dtype=np.int8), dtype=I8)
    assert buffer == buffer and buffer != other
    assert buffer[0] == other[0] == 1


def test_runtime_annotations_resolve() -> None:
    get_type_hints(NumericBuffer)
    for member in vars(NumericBuffer).values():
        if isinstance(member, property):
            member = member.fget
        if inspect.isfunction(member):
            get_type_hints(member)
