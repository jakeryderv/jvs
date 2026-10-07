"""Representation contracts, strict dtype boundaries, and platform precision."""

import sys
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest

from jvs.numeric import (
    Backend,
    ByteOrder,
    DType,
    ExactDType,
    NumericKind,
    NumPyDType,
)


@pytest.mark.parametrize("spec", ["int32", np.int32, np.dtype("int32")])
def test_equivalent_specs_are_hashable_descriptors(spec: Any) -> None:
    descriptor = NumPyDType(spec)
    assert isinstance(descriptor, DType)
    assert descriptor == NumPyDType("int32")
    assert len({descriptor, NumPyDType(np.int32)}) == 1
    assert descriptor.backend is Backend.NUMPY
    assert descriptor.kind is NumericKind.INTEGER
    assert descriptor.name == "int32"
    assert descriptor.is_fixed_width
    assert descriptor.storage_bits == 32
    assert not descriptor.supports_nonfinite


@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("bits", [8, 16, 32, 64])
def test_integer_ranges_are_exact(signed: bool, bits: int) -> None:
    name = f"{'int' if signed else 'uint'}{bits}"
    descriptor = NumPyDType(name)
    info = descriptor.integer_info
    assert info.signed is signed
    assert info.bits == bits == descriptor.storage_bits
    assert int(info.min) == -(2 ** (bits - 1)) if signed else int(info.min) == 0
    assert int(info.max) == 2 ** (bits - int(signed)) - 1
    assert info.min.dtype == descriptor.numpy_dtype
    assert info.max.dtype == descriptor.numpy_dtype


@pytest.mark.parametrize(
    ("name", "storage", "precision", "exponent", "minexp", "maxexp"),
    [
        ("float16", 16, 11, 5, -14, 16),
        ("float32", 32, 24, 8, -126, 128),
        ("float64", 64, 53, 11, -1022, 1024),
    ],
)
def test_float_formats(
    name: str, storage: int, precision: int, exponent: int, minexp: int, maxexp: int
) -> None:
    descriptor = NumPyDType(name)
    info = descriptor.floating_info
    assert descriptor.kind is NumericKind.FLOATING
    assert descriptor.storage_bits == storage
    assert descriptor.supports_nonfinite
    assert info.significand_bits == precision
    assert info.exponent_bits == exponent
    assert (info.min_exponent, info.max_exponent) == (minexp, maxexp)
    assert info.min == -info.max
    assert bool(np.isfinite(info.max))
    assert 0 < info.smallest_subnormal < info.smallest_normal < info.eps
    assert info.max.dtype == descriptor.numpy_dtype
    assert info.eps.dtype == descriptor.numpy_dtype


def test_extended_precision_metadata_stays_in_backend_format() -> None:
    descriptor = NumPyDType(np.longdouble)
    info = descriptor.floating_info
    backend = np.finfo(np.longdouble)
    assert descriptor.storage_bits == np.dtype(np.longdouble).itemsize * 8
    assert info.significand_bits == backend.nmant + 1
    assert info.exponent_bits == backend.nexp
    assert info.max == backend.max
    assert isinstance(info.max, np.longdouble)
    assert isinstance(info.smallest_subnormal, np.longdouble)
    assert bool(np.isfinite(info.max))
    assert info.smallest_subnormal > 0
    if backend.maxexp > np.finfo(np.float64).maxexp:
        assert info.max > np.longdouble(np.finfo(np.float64).max)
    if backend.nmant + backend.nexp + 1 < descriptor.storage_bits:
        assert info.significand_bits < descriptor.storage_bits


@pytest.mark.parametrize(
    ("spec", "component"),
    [
        (np.complex64, np.float32),
        (np.complex128, np.float64),
        (np.clongdouble, np.longdouble),
    ],
)
def test_complex_components(spec: Any, component: Any) -> None:
    descriptor = NumPyDType(spec)
    assert descriptor.kind is NumericKind.COMPLEX
    assert descriptor.supports_nonfinite
    assert descriptor.component_dtype == NumPyDType(component)
    assert descriptor.storage_bits == 2 * descriptor.component_dtype.storage_bits
    with pytest.raises(TypeError, match="component_dtype"):
        _ = descriptor.floating_info


def test_byte_order_is_preserved_and_part_of_identity() -> None:
    little = NumPyDType("<i4")
    big = NumPyDType(">i4")
    assert little != big
    assert little.byteorder is ByteOrder.LITTLE
    assert big.byteorder is ByteOrder.BIG
    assert NumPyDType("int32").byteorder is ByteOrder(sys.byteorder)
    assert NumPyDType("int8").byteorder is ByteOrder.NOT_APPLICABLE
    assert NumPyDType(">c16").component_dtype == NumPyDType(">f8")
    assert NumPyDType(">f8").floating_info.significand_bits == 53
    assert int(big.integer_info.max) == 2**31 - 1


@pytest.mark.parametrize(
    "spec",
    [
        None,
        int,
        float,
        complex,
        bool,
        32,
        3.0,
        np.int32(3),
        np.array([3]),
        np.generic,
        np.number,
        np.integer,
        np.floating,
        np.complexfloating,
        "bool",
        "object",
        "datetime64[ns]",
        "timedelta64[D]",
        "U4",
        "S4",
        "V4",
        np.dtype([("x", "i4")]),
        np.dtype(("f8", (3,))),
        np.dtype("i4", metadata={"unit": "count"}),
        np.dtype("i4", metadata={}),
    ],
)
def test_unsupported_or_implicit_specs_raise(spec: Any) -> None:
    with pytest.raises(TypeError):
        NumPyDType(spec)


def test_invalid_dtype_name_raises() -> None:
    with pytest.raises(TypeError):
        NumPyDType("not-a-dtype")


def test_wrong_kind_metadata_requests_raise() -> None:
    with pytest.raises(TypeError):
        _ = NumPyDType("float32").integer_info
    with pytest.raises(TypeError):
        _ = NumPyDType("int32").floating_info
    with pytest.raises(TypeError):
        _ = NumPyDType("float32").component_dtype


@pytest.mark.parametrize("descriptor", [NumPyDType("int32"), ExactDType.integer()])
def test_descriptors_are_immutable(descriptor: DType) -> None:
    with pytest.raises((FrozenInstanceError, AttributeError)):
        setattr(
            descriptor,
            "numpy_dtype" if isinstance(descriptor, NumPyDType) else "_kind",
            np.dtype("float64"),
        )


def test_exact_representations_have_no_fixed_width() -> None:
    integer = ExactDType.integer()
    rational = ExactDType.rational()
    for descriptor in (integer, rational):
        assert isinstance(descriptor, DType)
        assert descriptor.backend is Backend.SYMPY
        assert descriptor.storage_bits is None
        assert not descriptor.is_fixed_width
        assert not descriptor.supports_nonfinite
    assert integer.kind is NumericKind.INTEGER
    assert rational.kind is NumericKind.RATIONAL
    assert integer != rational
    assert integer != NumPyDType("int64")
    assert len({integer, ExactDType(NumericKind.INTEGER)}) == 1
    assert integer.name == "Integer"
    assert rational.name == "Rational"


@pytest.mark.parametrize("kind", [NumericKind.FLOATING, NumericKind.COMPLEX])
def test_unsupported_exact_families_raise(kind: NumericKind) -> None:
    with pytest.raises(ValueError):
        ExactDType(kind)


@pytest.mark.parametrize("kind", ["integer", None, 1])
def test_exact_kind_must_be_explicit_enum(kind: Any) -> None:
    with pytest.raises(TypeError):
        ExactDType(kind)
