"""Representation descriptors, independent of mathematical sets and values.

Descriptors report backend capabilities; they do not cast values, establish
membership, choose arithmetic promotion, or promise numerical accuracy.
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

import numpy as np
import sympy as sp

from ._repr import integer_repr


class Backend(Enum):
    """Backends currently described by the dtype layer."""

    NUMPY = "numpy"
    SYMPY = "sympy"


class NumericKind(Enum):
    """Representation families, not mathematical membership classifications."""

    INTEGER = "integer"
    FLOATING = "floating"
    COMPLEX = "complex"
    RATIONAL = "rational"
    FIXED = "fixed"


class ByteOrder(Enum):
    """Resolved byte order, including types for which it is not applicable."""

    LITTLE = "little"
    BIG = "big"
    NOT_APPLICABLE = "not_applicable"


class DType(ABC):
    """Common descriptor interface for fixed-width and exact representations."""

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Representation label; inspect descriptor fields for full identity."""

    @property
    @abstractmethod
    def backend(self) -> Backend:
        """Backend that supplies the representation."""

    @property
    @abstractmethod
    def kind(self) -> NumericKind:
        """Representation family, independent of a particular value."""

    @property
    @abstractmethod
    def storage_bits(self) -> int | None:
        """Fixed storage width including padding, or None for variable storage."""

    @property
    @abstractmethod
    def supports_nonfinite(self) -> bool:
        """Whether this representation can encode infinities or NaNs.

        This capability does not authorize nonfinite values in a finite domain.
        """

    @property
    def is_fixed_width(self) -> bool:
        return self.storage_bits is not None


@dataclass(frozen=True, slots=True)
class IntegerInfo:
    """Integer format metadata; bounds retain the NumPy scalar type."""

    signed: bool
    bits: int
    min: np.integer[Any]
    max: np.integer[Any]


@dataclass(frozen=True, slots=True)
class FloatingInfo:
    """NumPy-reported binary floating format, not a guarantee of accuracy.

    Significand precision includes the leading bit for normal values. Subnormal
    values have reduced precision. Exponent bounds follow finfo: min_exponent
    gives the smallest normal power of two; max_exponent is exclusive.
    These fields do not specify a portable bit layout for extended formats.
    """

    significand_bits: int
    exponent_bits: int
    min_exponent: int
    max_exponent: int
    min: np.floating[Any]
    max: np.floating[Any]
    eps: np.floating[Any]
    smallest_normal: np.floating[Any]
    smallest_subnormal: np.floating[Any]


type NumPyDTypeLike = str | np.dtype[Any] | type[np.generic]

_ABSTRACT_NUMPY_TYPES = (
    np.generic,
    np.number,
    np.inexact,
    np.integer,
    np.signedinteger,
    np.unsignedinteger,
    np.floating,
    np.complexfloating,
)


@dataclass(frozen=True, slots=True, init=False)
class NumPyDType(DType):
    """An explicit NumPy integer, floating, or complex scalar representation.

    Accept dtype strings, concrete NumPy scalar classes, or np.dtype objects.
    Reject implicit defaults, Python scalar classes, scalar values, and dtypes
    with fields, subarrays, or metadata. Dtype aliases resolve on this platform;
    use explicit width names such as 'int32' for portable width requirements.
    """

    numpy_dtype: np.dtype[Any]

    def __init__(self, spec: NumPyDTypeLike) -> None:
        if isinstance(spec, type):
            if not issubclass(spec, np.generic) or spec in _ABSTRACT_NUMPY_TYPES:
                raise TypeError("Use a concrete NumPy scalar type, such as np.int32")
        elif not isinstance(spec, (str, np.dtype)):
            raise TypeError(
                "Expected a dtype string, np.dtype, or concrete NumPy scalar type"
            )
        dtype = np.dtype(spec)
        if dtype.fields is not None or dtype.subdtype is not None:
            raise TypeError(
                "Structured and subarray dtypes are not scalar numeric representations"
            )
        if dtype.metadata is not None:
            raise TypeError(
                "Dtype metadata is unsupported; keep domain metadata separate"
            )
        if dtype.kind not in "iufc":
            raise TypeError(f"Unsupported numeric dtype: {dtype!r}")
        object.__setattr__(self, "numpy_dtype", dtype)

    @property
    def name(self) -> str:
        return self.numpy_dtype.name

    @property
    def backend(self) -> Backend:
        return Backend.NUMPY

    @property
    def kind(self) -> NumericKind:
        if self.numpy_dtype.kind in "iu":
            return NumericKind.INTEGER
        if self.numpy_dtype.kind == "f":
            return NumericKind.FLOATING
        return NumericKind.COMPLEX

    @property
    def storage_bits(self) -> int:
        return self.numpy_dtype.itemsize * 8

    @property
    def supports_nonfinite(self) -> bool:
        return self.numpy_dtype.kind in "fc"

    @property
    def byteorder(self) -> ByteOrder:
        """Actual storage byte order, resolving NumPy's native '=' marker."""
        order = self.numpy_dtype.byteorder
        if order == "|":
            return ByteOrder.NOT_APPLICABLE
        if order == "=":
            return ByteOrder(sys.byteorder)
        return ByteOrder.LITTLE if order == "<" else ByteOrder.BIG

    @property
    def integer_info(self) -> IntegerInfo:
        """Exact finite range for integer formats; other kinds raise TypeError."""
        if self.kind is not NumericKind.INTEGER:
            raise TypeError(f"{self.name} is not an integer representation")
        info = np.iinfo(self.numpy_dtype)
        scalar = self.numpy_dtype.type
        return IntegerInfo(
            signed=self.numpy_dtype.kind == "i",
            bits=int(info.bits),
            min=scalar(info.min),
            max=scalar(info.max),
        )

    @property
    def floating_info(self) -> FloatingInfo:
        """Real floating format; complex types must use component_dtype first."""
        if self.kind is not NumericKind.FLOATING:
            raise TypeError(
                f"{self.name} is not a real floating representation; complex types expose component_dtype"
            )
        info = np.finfo(self.numpy_dtype)
        return FloatingInfo(
            significand_bits=int(info.nmant) + 1,
            exponent_bits=int(info.nexp),
            min_exponent=int(info.minexp),
            max_exponent=int(info.maxexp),
            min=info.min,
            max=info.max,
            eps=info.eps,
            smallest_normal=info.smallest_normal,
            smallest_subnormal=info.smallest_subnormal,
        )

    @property
    def component_dtype(self) -> NumPyDType:
        """Real component format of a complex dtype, preserving byte order."""
        if self.kind is not NumericKind.COMPLEX:
            raise TypeError(f"{self.name} is not a complex representation")
        component = np.finfo(self.numpy_dtype).dtype
        return NumPyDType(component.newbyteorder(self.numpy_dtype.byteorder))


@dataclass(frozen=True, slots=True, init=False)
class ExactDType(DType):
    """SymPy exact integer or rational representation with variable storage.

    No fixed bit width, finite range bound, or floating working precision applies.
    This describes a representation; it does not construct or convert values.
    """

    _kind: NumericKind

    def __init__(self, kind: NumericKind) -> None:
        if not isinstance(kind, NumericKind):
            raise TypeError("Expected a NumericKind")
        if kind not in (NumericKind.INTEGER, NumericKind.RATIONAL):
            raise ValueError("ExactDType currently supports only INTEGER and RATIONAL")
        object.__setattr__(self, "_kind", kind)

    @classmethod
    def integer(cls) -> ExactDType:
        return cls(NumericKind.INTEGER)

    @classmethod
    def rational(cls) -> ExactDType:
        return cls(NumericKind.RATIONAL)

    @property
    def name(self) -> str:
        return "Integer" if self._kind is NumericKind.INTEGER else "Rational"

    @property
    def backend(self) -> Backend:
        return Backend.SYMPY

    @property
    def kind(self) -> NumericKind:
        return self._kind

    @property
    def storage_bits(self) -> None:
        return None

    @property
    def supports_nonfinite(self) -> bool:
        return False


type FixedStep = int | np.integer[Any] | sp.Rational | Fraction


@dataclass(frozen=True, slots=True, init=False)
class FixedDType(DType):
    """Integer coefficient storage and a positive exact rational lattice step.

    Storage width and backend describe the coefficient, excluding step metadata.
    Unbounded coefficients still represent only integer multiples of the step.
    """

    coefficient_dtype: NumPyDType | ExactDType
    step: Fraction

    def __init__(
        self, coefficient_dtype: NumPyDType | ExactDType, *, step: FixedStep
    ) -> None:
        if not isinstance(coefficient_dtype, (NumPyDType, ExactDType)) or (
            coefficient_dtype.kind is not NumericKind.INTEGER
        ):
            raise TypeError("FixedDType requires an explicit integer coefficient dtype")
        if isinstance(step, bool):
            raise TypeError("Fixed-point step must be an exact rational, not a boolean")
        if isinstance(step, Fraction):
            ratio = step
        elif isinstance(step, sp.Rational):
            ratio = Fraction(step.p, step.q)
        elif isinstance(step, int) or (
            isinstance(step, np.integer) and step.dtype.kind in "iu"
        ):
            ratio = Fraction(int(step))
        else:
            raise TypeError(
                "Fixed-point step requires an integer, Fraction, or SymPy rational"
            )
        if ratio <= 0:
            raise ValueError("Fixed-point step must be positive")
        object.__setattr__(self, "coefficient_dtype", coefficient_dtype)
        object.__setattr__(self, "step", ratio)

    def __repr__(self) -> str:
        return (
            f"FixedDType({self.coefficient_dtype!r}, "
            f"step=Fraction({integer_repr(self.step.numerator)}, "
            f"{integer_repr(self.step.denominator)}))"
        )

    @property
    def name(self) -> str:
        return (
            f"fixed[{self.coefficient_dtype.name}, "
            f"step={integer_repr(self.step.numerator)}/{integer_repr(self.step.denominator)}]"
        )

    @property
    def backend(self) -> Backend:
        return self.coefficient_dtype.backend

    @property
    def kind(self) -> NumericKind:
        return NumericKind.FIXED

    @property
    def storage_bits(self) -> int | None:
        return self.coefficient_dtype.storage_bits

    @property
    def supports_nonfinite(self) -> bool:
        return False
