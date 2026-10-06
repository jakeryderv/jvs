# src/jvs/floating.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, TypeVar

import numpy as np

from .number import Number

TFloating = TypeVar("TFloating", bound="Floating")


@dataclass(frozen=True, slots=True)
class Floating(Number):
    """Base class for immutable, fixed-width floating-point values."""

    _value: np.floating[Any] = field(init=False, repr=False)

    dtype: ClassVar[type[np.floating[Any]] | None] = None

    def __init__(self, value: float) -> None:
        dtype = self.dtype

        if dtype is None:
            raise TypeError(f"{type(self).__name__} is not a concrete floating type")

        if isinstance(value, bool):
            raise TypeError("bool cannot be implicitly converted to Floating")

        if not isinstance(value, float):
            raise TypeError(
                f"{type(self).__name__} requires float, got {type(value).__name__}"
            )

        converted = dtype(value)

        # Finite input becoming infinity means the target type overflowed.
        if np.isfinite(value) and not np.isfinite(converted):
            raise OverflowError(
                f"{value} cannot be represented by {type(self).__name__}"
            )

        object.__setattr__(self, "_value", converted)

    @property
    def value(self) -> np.floating[Any]:
        """Underlying NumPy scalar."""
        return self._value

    @classmethod
    def bits(cls) -> int:
        """Number of bits used by this floating-point type."""
        return int(cls._info().bits)

    @classmethod
    def min(cls) -> float:
        """Most negative finite representable value."""
        return float(cls._info().min)

    @classmethod
    def max(cls) -> float:
        """Largest finite representable value."""
        return float(cls._info().max)

    @classmethod
    def eps(cls) -> float:
        """Difference between 1 and the next representable value."""
        return float(cls._info().eps)

    @classmethod
    def _info(cls) -> np.finfo:
        dtype = cls.dtype

        if dtype is None:
            raise TypeError(f"{cls.__name__} is not a concrete floating type")

        return np.finfo(dtype)

    def to(self, target: type[TFloating]) -> TFloating:
        """
        Convert to another floating-point type.

        IEEE-754 rounding is allowed, but overflow to infinity is rejected.
        """
        if not issubclass(target, Floating):
            raise TypeError(f"target must be a Floating type, got {target!r}")

        return target(float(self))

    def __float__(self) -> float:
        return float(self._value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({float(self)!r})"


class Float16(Floating):
    dtype = np.float16


class Float32(Floating):
    dtype = np.float32


class Float64(Floating):
    dtype = np.float64
