"""Finite floating scalars with explicit approximation and checked arithmetic."""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, ClassVar, Never

import numpy as np
import sympy as sp

from . import fixed
from ._interop import scalar_comparison_ufunc
from ._ordering import compare_real
from .dtype import ExactDType, FloatingInfo, NumericKind, NumPyDType
from .integer import IntegerDType, IntegerValue, _input_integer
from .rational import RationalValue


class PrecisionLossError(ValueError):
    """An exact conversion would require rounding."""


class UnderflowError(FloatingPointError):
    """A nonzero value below the smallest normal cannot be preserved exactly."""


type FloatingInput = (
    int
    | float
    | np.integer[Any]
    | np.floating[Any]
    | sp.Rational
    | Fraction
    | IntegerValue
    | RationalValue
    | FloatingValue
    | fixed.FixedValue
)


def _power2(exponent: int) -> Fraction:
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def _ratio(value: np.floating[Any]) -> Fraction:
    return Fraction(*value.as_integer_ratio())


def _format(dtype: NumPyDType) -> FloatingInfo:
    if not isinstance(dtype, NumPyDType) or dtype.kind is not NumericKind.FLOATING:
        raise TypeError("FloatingValue requires an explicit floating NumPyDType")
    if not dtype.numpy_dtype.isnative:
        raise ValueError("FloatingValue requires native byte order for NumPy scalars")
    info = dtype.floating_info
    p, emin, emax = info.significand_bits, info.min_exponent, info.max_exponent
    if (
        (p, emin, emax)
        not in {
            (11, -14, 16),
            (24, -126, 128),
            (53, -1022, 1024),
            (64, -16382, 16384),
            (113, -16382, 16384),
        }
        or _ratio(info.smallest_normal) != _power2(emin)
        or _ratio(info.smallest_subnormal) != _power2(emin - p + 1)
        or _ratio(info.max) != ((1 << p) - 1) * _power2(emax - p)
    ):
        raise NotImplementedError(
            "Unsupported NumPy floating format or subnormal spacing"
        )
    return info


def _input(value: object) -> tuple[Fraction, bool, bool]:
    """Return stored/exact ratio, negative-zero sign, and known rounding history."""
    if isinstance(value, FloatingValue):
        return _ratio(value.value), bool(np.signbit(value.value)), value.rounded
    if isinstance(value, fixed.FixedValue):
        return value._ratio(), False, value.rounded
    if isinstance(value, IntegerValue):
        return Fraction(int(value)), False, False
    if isinstance(value, RationalValue):
        return Fraction(value.value.p, value.value.q), False, False
    if isinstance(value, Fraction):
        return value, False, False
    if isinstance(value, sp.Rational):
        return Fraction(value.p, value.q), False, False
    if isinstance(value, (float, np.floating)):
        if not bool(np.isfinite(value)):
            raise ValueError("FloatingValue requires finite inputs")
        return Fraction(*value.as_integer_ratio()), bool(np.signbit(value)), False
    return Fraction(_input_integer(value)), False, False


def _round_ratio(exact: Fraction, info: FloatingInfo, *, approximate: bool) -> Fraction:
    """Round once on the target binary lattice, using integer ties-to-even."""
    magnitude = abs(exact)
    if magnitude > _ratio(info.max):
        raise OverflowError("Value exceeds the target floating finite range")
    if not magnitude:
        return exact
    p, q = magnitude.numerator, magnitude.denominator
    exponent = p.bit_length() - q.bit_length()
    if magnitude < _power2(exponent):
        exponent -= 1
    step = _power2(max(exponent, info.min_exponent) - info.significand_bits + 1)
    scaled = magnitude / step
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = 2 * remainder
    if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
        whole += 1
    rounded = whole * step
    if rounded != magnitude:
        if magnitude < _ratio(info.smallest_normal):
            raise UnderflowError("Inexact value below the target smallest normal")
        if not approximate:
            raise PrecisionLossError(
                "Exact conversion requires rounding; use explicit approximation"
            )
    return -rounded if exact < 0 else rounded


def _store(exact: Fraction, dtype: NumPyDType, negative_zero: bool) -> np.floating[Any]:
    """Encode an already rounded binary ratio without a Python-float intermediate."""
    scalar = dtype.numpy_dtype.type
    numerator, denominator = abs(exact.numerator), exact.denominator
    with np.errstate(all="ignore"):
        result = scalar(0)
        if numerator:
            trailing = (numerator & -numerator).bit_length() - 1
            significand = numerator >> trailing
            exponent = trailing - (denominator.bit_length() - 1)
            # Build at most p significant bits from exactly representable bytes.
            for byte in significand.to_bytes(
                (significand.bit_length() + 7) // 8, "big"
            ):
                result = np.ldexp(result, 8) + scalar(byte)
            result = np.ldexp(result, exponent)
        if exact < 0 or (not exact and negative_zero):
            result = -result
    if not bool(np.isfinite(result)) or _ratio(result) != exact:
        raise FloatingPointError(
            "Backend failed to preserve the requested binary value"
        )
    return result


def _prepare(
    value: object, dtype: NumPyDType, *, approximate: bool
) -> tuple[np.floating[Any], bool]:
    info = _format(dtype)
    exact, negative_zero, history = _input(value)
    rounded = _round_ratio(exact, info, approximate=approximate)
    return _store(rounded, dtype, negative_zero), history or rounded != exact


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FloatingValue:
    """A finite NumPy scalar; exact conversion by default, rounded arithmetic.

    ``rounded`` tracks known rounding inside this API, not external accuracy.
    Raw floats require explicit wrapping for equality and arithmetic.
    """

    dtype: NumPyDType
    rounded: bool
    _value: np.floating[Any] = field(repr=False)
    _op_priority: ClassVar[float] = 1000.0

    def __init__(self, value: FloatingInput, *, dtype: NumPyDType) -> None:
        stored, rounded = _prepare(value, dtype, approximate=False)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "rounded", rounded)
        object.__setattr__(self, "_value", stored)

    @classmethod
    def _from_stored(
        cls, value: np.floating[Any], dtype: NumPyDType, rounded: bool
    ) -> FloatingValue:
        result = object.__new__(cls)
        object.__setattr__(result, "dtype", dtype)
        object.__setattr__(result, "rounded", rounded)
        object.__setattr__(result, "_value", value)
        return result

    @classmethod
    def approx(cls, value: FloatingInput, *, dtype: NumPyDType) -> FloatingValue:
        """Explicit nearest-even approximation; overflow and underflow still raise."""
        stored, rounded = _prepare(value, dtype, approximate=True)
        return cls._from_stored(stored, dtype, rounded)

    @property
    def value(self) -> np.floating[Any]:
        """Immutable backend scalar; extracting it transfers control to NumPy."""
        return self._value

    def to(self, dtype: NumPyDType, *, approximate: bool = False) -> FloatingValue:
        if not isinstance(approximate, bool):
            raise TypeError("approximate must be a bool")
        return (
            self.approx(self, dtype=dtype)
            if approximate
            else FloatingValue(self, dtype=dtype)
        )

    def to_rational(self) -> RationalValue:
        """Extract the stored binary ratio, discarding floating provenance."""
        p, q = self._value.as_integer_ratio()
        return RationalValue(p, q, dtype=ExactDType.rational())

    def to_integer(self, dtype: IntegerDType) -> IntegerValue:
        return self.to_rational().to_integer(dtype)

    def __bool__(self) -> bool:
        return bool(self._value != 0)

    def __repr__(self) -> str:
        return f"FloatingValue(value={self._value!r}, dtype={self.dtype!r}, rounded={self.rounded!r})"

    def __hash__(self) -> int:
        return hash(_ratio(self._value))

    def __eq__(self, other: object) -> bool:
        if isinstance(
            other, (FloatingValue, IntegerValue, RationalValue, fixed.FixedValue)
        ):
            exact, _, _ = _input(other)
        elif isinstance(other, (numbers.Number, Decimal, np.generic, sp.Basic)):
            exact = Fraction(_input_integer(other))
        else:
            return NotImplemented
        return _ratio(self._value) == exact

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def __lt__(self, other: object) -> bool:
        return compare_real(self, other, operator.lt)

    def __le__(self, other: object) -> bool:
        return compare_real(self, other, operator.le)

    def __gt__(self, other: object) -> bool:
        return compare_real(self, other, operator.gt)

    def __ge__(self, other: object) -> bool:
        return compare_real(self, other, operator.ge)

    def _peer(self, other: object) -> FloatingValue:
        if not isinstance(other, FloatingValue):
            raise TypeError(
                "Arithmetic requires FloatingValue operands with matching dtypes"
            )
        if self.dtype != other.dtype:
            raise TypeError(
                "Mixed floating dtypes require explicit conversion with .to(dtype)"
            )
        return other

    def _binary(self, other: object, operation: Callable[..., Any]) -> FloatingValue:
        peer = self._peer(other)
        if operation is operator.truediv and not peer:
            raise ZeroDivisionError("Cannot divide FloatingValue by zero")
        exact = operation(_ratio(self._value), _ratio(peer._value))
        expected = _round_ratio(exact, _format(self.dtype), approximate=True)
        with np.errstate(all="ignore"):
            stored = operation(self._value, peer._value)
        if (
            not isinstance(stored, np.floating)
            or stored.dtype != self.dtype.numpy_dtype
            or not bool(np.isfinite(stored))
            or _ratio(stored) != expected
        ):
            raise FloatingPointError(
                "Backend arithmetic did not produce the required nearest-even result"
            )
        return self._from_stored(
            stored, self.dtype, self.rounded or peer.rounded or expected != exact
        )

    def __add__(self, other: object) -> FloatingValue:
        return self._binary(other, operator.add)

    def __radd__(self, other: object) -> FloatingValue:
        return self._peer(other).__add__(self)

    def __sub__(self, other: object) -> FloatingValue:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: object) -> FloatingValue:
        return self._peer(other).__sub__(self)

    def __mul__(self, other: object) -> FloatingValue:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: object) -> FloatingValue:
        return self._peer(other).__mul__(self)

    def __truediv__(self, other: object) -> FloatingValue:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: object) -> FloatingValue:
        return self._peer(other).__truediv__(self)

    def __neg__(self) -> FloatingValue:
        return self._from_stored(-self._value, self.dtype, self.rounded)

    def __pos__(self) -> FloatingValue:
        return self

    def _sympy_(self) -> Never:
        raise sp.SympifyError(self, TypeError("Extract .value explicitly"))

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        return scalar_comparison_ufunc(self, ufunc, method, *inputs, **kwargs)
