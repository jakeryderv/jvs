# src/jvs/integer.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

import numpy as np

from .number import Number

TInteger = TypeVar("TInteger", bound="Integer")


@dataclass(frozen=True, slots=True)
class Integer(Number):
    """Base class for immutable, fixed-width integer values."""

    _value: np.integer[Any] = field(init=False, repr=False)

    dtype: ClassVar[type[np.integer[Any]] | None] = None
    signed: ClassVar[bool | None] = None

    def __init__(self, value: int) -> None:
        dtype = self.dtype

        if dtype is None:
            raise TypeError(f"{type(self).__name__} is not a concrete integer type")

        if isinstance(value, bool):
            raise TypeError("bool cannot be implicitly converted to Integer")

        if not isinstance(value, int):
            raise TypeError(
                f"{type(self).__name__} requires int, got {type(value).__name__}"
            )

        info = np.iinfo(dtype)

        if not info.min <= value <= info.max:
            raise OverflowError(
                f"{value} cannot be represented by "
                f"{type(self).__name__} "
                f"[{info.min}, {info.max}]"
            )

        object.__setattr__(self, "_value", dtype(value))

    @property
    def value(self) -> np.integer[Any]:
        """Underlying NumPy scalar."""
        return self._value

    @classmethod
    def bits(cls) -> int:
        """Number of bits used by this integer type."""
        return int(cls._info().bits)

    @classmethod
    def min(cls) -> int:
        """Minimum representable value."""
        return int(cls._info().min)

    @classmethod
    def max(cls) -> int:
        """Maximum representable value."""
        return int(cls._info().max)

    @classmethod
    def _info(cls) -> np.iinfo:
        dtype = cls.dtype

        if dtype is None:
            raise TypeError(f"{cls.__name__} is not a concrete integer type")

        return np.iinfo(dtype)

    def to(self, target: type[TInteger]) -> TInteger:
        """
        Convert to another integer type.

        The conversion succeeds only if the value is representable
        by the target type.
        """
        if not issubclass(target, Integer):
            raise TypeError(f"target must be an Integer type, got {target!r}")

        return target(int(self))

    def __int__(self) -> int:
        return int(self._value)

    def __index__(self) -> int:
        return int(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({int(self)})"


class SignedInteger(Integer):
    """Base class for signed fixed-width integer values."""

    signed = True


class UnsignedInteger(Integer):
    """Base class for unsigned fixed-width integer values."""

    signed = False


class Int8(SignedInteger):
    dtype = np.int8


class Int16(SignedInteger):
    dtype = np.int16


class Int32(SignedInteger):
    dtype = np.int32


class Int64(SignedInteger):
    dtype = np.int64


class UInt8(UnsignedInteger):
    dtype = np.uint8


class UInt16(UnsignedInteger):
    dtype = np.uint16


class UInt32(UnsignedInteger):
    dtype = np.uint32


class UInt64(UnsignedInteger):
    dtype = np.uint64
