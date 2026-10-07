"""Checked integer coefficients on an explicitly selected rational lattice."""

from __future__ import annotations

import numbers
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import ClassVar, Never

import numpy as np
import sympy as sp

from . import floating
from ._interop import scalar_equality_ufunc
from ._repr import integer_repr
from .dtype import ExactDType, FixedDType, NumPyDType
from .integer import IntegerDType, IntegerInput, IntegerValue, _input_integer
from .rational import RationalValue


def _prepare(
    value: object, dtype: FixedDType, *, approximate: bool
) -> tuple[IntegerValue, bool]:
    if not isinstance(dtype, FixedDType):
        raise TypeError("FixedValue requires an explicit FixedDType")
    exact, _, history = floating._input(value)
    scaled = exact / dtype.step
    storage = dtype.coefficient_dtype
    if isinstance(storage, NumPyDType):
        limits = storage.integer_info
        if not int(limits.min) <= scaled <= int(limits.max):
            raise OverflowError("Value exceeds the fixed-point coefficient range")
    whole, remainder = divmod(scaled.numerator, scaled.denominator)
    if remainder:
        if not approximate:
            raise floating.PrecisionLossError(
                "Exact fixed-point conversion requires rounding; use explicit approximation"
            )
        twice = 2 * remainder
        if twice > scaled.denominator or (twice == scaled.denominator and whole % 2):
            whole += 1
    return IntegerValue(whole, dtype=storage), history or remainder != 0


@dataclass(frozen=True, slots=True, init=False, eq=False)
class FixedValue:
    """An exact stored multiple of a rational step, with known rounding history.

    Construction takes a mathematical value; from_coefficient takes storage
    units. Multiplication/division extract exact rational results and discard
    history, as does to_rational(). Neither operation recovers input intent.
    """

    dtype: FixedDType
    coefficient: IntegerValue
    rounded: bool
    _op_priority: ClassVar[float] = 1000.0

    def __init__(self, value: floating.FloatingInput, *, dtype: FixedDType) -> None:
        coefficient, rounded = _prepare(value, dtype, approximate=False)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "coefficient", coefficient)
        object.__setattr__(self, "rounded", rounded)

    @classmethod
    def _from_coefficient(
        cls, coefficient: IntegerValue, dtype: FixedDType, rounded: bool
    ) -> FixedValue:
        result = object.__new__(cls)
        object.__setattr__(result, "dtype", dtype)
        object.__setattr__(result, "coefficient", coefficient)
        object.__setattr__(result, "rounded", rounded)
        return result

    @classmethod
    def from_coefficient(
        cls, coefficient: IntegerInput, *, dtype: FixedDType
    ) -> FixedValue:
        """Construct from integer storage units, with range checks and no rounding."""
        if not isinstance(dtype, FixedDType):
            raise TypeError("FixedValue requires an explicit FixedDType")
        stored = IntegerValue(coefficient, dtype=dtype.coefficient_dtype)
        return cls._from_coefficient(stored, dtype, False)

    @classmethod
    def approx(cls, value: floating.FloatingInput, *, dtype: FixedDType) -> FixedValue:
        """Round to the nearest coefficient, ties to even; range checks remain strict."""
        coefficient, rounded = _prepare(value, dtype, approximate=True)
        return cls._from_coefficient(coefficient, dtype, rounded)

    def _ratio(self) -> Fraction:
        return int(self.coefficient) * self.dtype.step

    @property
    def value(self) -> sp.Rational:
        """Reconstructed mathematical value; coefficient.value exposes storage."""
        ratio = self._ratio()
        return sp.Rational(ratio.numerator, ratio.denominator)

    def to(self, dtype: FixedDType, *, approximate: bool = False) -> FixedValue:
        if not isinstance(approximate, bool):
            raise TypeError("approximate must be a bool")
        return (
            self.approx(self, dtype=dtype)
            if approximate
            else FixedValue(self, dtype=dtype)
        )

    def to_rational(self) -> RationalValue:
        """Extract the stored rational value, discarding rounding history."""
        ratio = self._ratio()
        return RationalValue(
            ratio.numerator, ratio.denominator, dtype=ExactDType.rational()
        )

    def to_integer(self, dtype: IntegerDType) -> IntegerValue:
        return self.to_rational().to_integer(dtype)

    def __bool__(self) -> bool:
        return bool(self.coefficient)

    def __repr__(self) -> str:
        return (
            f"FixedValue(coefficient={integer_repr(int(self.coefficient))}, "
            f"dtype={self.dtype!r}, rounded={self.rounded!r})"
        )

    def __hash__(self) -> int:
        return hash(self._ratio())

    def __eq__(self, other: object) -> bool:
        if isinstance(
            other, (FixedValue, floating.FloatingValue, IntegerValue, RationalValue)
        ):
            ratio, _, _ = floating._input(other)
        elif isinstance(other, (numbers.Number, Decimal, np.generic, sp.Basic)):
            ratio = Fraction(_input_integer(other))
        else:
            # ComplexValue compares its real component through FloatingValue,
            # which understands FixedValue. Preserve reflected dispatch.
            return NotImplemented
        return self._ratio() == ratio

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def _peer(self, other: object) -> FixedValue:
        if not isinstance(other, FixedValue) or self.dtype != other.dtype:
            raise TypeError(
                "Arithmetic requires FixedValue operands with matching dtypes; convert explicitly"
            )
        return other

    def __add__(self, other: object) -> FixedValue:
        peer = self._peer(other)
        return self._from_coefficient(
            self.coefficient + peer.coefficient,
            self.dtype,
            self.rounded or peer.rounded,
        )

    def __radd__(self, other: object) -> FixedValue:
        return self._peer(other).__add__(self)

    def __sub__(self, other: object) -> FixedValue:
        peer = self._peer(other)
        return self._from_coefficient(
            self.coefficient - peer.coefficient,
            self.dtype,
            self.rounded or peer.rounded,
        )

    def __rsub__(self, other: object) -> FixedValue:
        return self._peer(other).__sub__(self)

    def __mul__(self, other: object) -> RationalValue:
        return self.to_rational() * self._peer(other).to_rational()

    def __rmul__(self, other: object) -> RationalValue:
        return self._peer(other).__mul__(self)

    def __truediv__(self, other: object) -> RationalValue:
        return self.to_rational() / self._peer(other).to_rational()

    def __rtruediv__(self, other: object) -> RationalValue:
        return self._peer(other).__truediv__(self)

    def __neg__(self) -> FixedValue:
        return self._from_coefficient(-self.coefficient, self.dtype, self.rounded)

    def __pos__(self) -> FixedValue:
        return self

    def _sympy_(self) -> Never:
        raise sp.SympifyError(self, TypeError("Extract .value explicitly"))

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        return scalar_equality_ufunc(self, ufunc, method, *inputs, **kwargs)
