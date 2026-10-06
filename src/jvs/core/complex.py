# src/jvs/complex.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

import numpy as np

from .number import Number

TComplex = TypeVar("TComplex", bound="Complex")


@dataclass(frozen=True, slots=True)
class Complex(Number):
    """Base class for immutable, fixed-width complex values."""

    _value: np.complexfloating[Any, Any] = field(
        init=False,
        repr=False,
    )

    dtype: ClassVar[type[np.complexfloating[Any, Any]] | None] = None

    def __init__(self, value: complex) -> None:
        dtype = self.dtype

        if dtype is None:
            raise TypeError(f"{type(self).__name__} is not a concrete complex type")

        if isinstance(value, bool):
            raise TypeError("bool cannot be implicitly converted to Complex")

        if not isinstance(value, complex):
            raise TypeError(
                f"{type(self).__name__} requires complex, got {type(value).__name__}"
            )

        converted = dtype(value)

        real_overflow = np.isfinite(value.real) and not np.isfinite(converted.real)

        imag_overflow = np.isfinite(value.imag) and not np.isfinite(converted.imag)

        if real_overflow or imag_overflow:
            raise OverflowError(
                f"{value} cannot be represented by {type(self).__name__}"
            )

        object.__setattr__(self, "_value", converted)

    @property
    def value(self) -> np.complexfloating[Any, Any]:
        """Underlying NumPy scalar."""
        return self._value

    @property
    def real(self) -> float:
        """Real component as a Python float."""
        return float(self._value.real)

    @property
    def imag(self) -> float:
        """Imaginary component as a Python float."""
        return float(self._value.imag)

    @classmethod
    def bits(cls) -> int:
        """Total number of bits used by the complex value."""
        dtype = cls.dtype

        if dtype is None:
            raise TypeError(f"{cls.__name__} is not a concrete complex type")

        return int(np.dtype(dtype).itemsize * 8)

    def to(self, target: type[TComplex]) -> TComplex:
        """
        Convert to another complex type.

        IEEE-754 rounding is allowed, but component overflow is rejected.
        """
        if not issubclass(target, Complex):
            raise TypeError(f"target must be a Complex type, got {target!r}")

        return target(complex(self))

    def __complex__(self) -> complex:
        return complex(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({complex(self)!r})"


class Complex64(Complex):
    dtype = np.complex64


class Complex128(Complex):
    dtype = np.complex128
