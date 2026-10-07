"""Immutable exact rational values with explicit SymPy representation."""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import ClassVar, Never

import numpy as np
import sympy as sp

from . import integer
from ._interop import scalar_comparison_ufunc
from ._ordering import compare_real
from ._repr import integer_repr
from .dtype import ExactDType, NumericKind


@dataclass(frozen=True, slots=True, init=False, eq=False)
class RationalValue:
    """A finite exact ratio constructed from supported integer components.

    Arithmetic requires RationalValue operands. Equality supports rational and
    integer wrappers and Python/NumPy/SymPy integer scalars. Other numeric
    representations require explicit conversion, including raw rationals whose
    backend equality and hashing conventions need not agree.
    """

    dtype: ExactDType
    _value: sp.Rational = field(repr=False)

    _op_priority: ClassVar[float] = 1000.0

    def __init__(
        self,
        numerator: integer.IntegerInput,
        denominator: integer.IntegerInput = 1,
        *,
        dtype: ExactDType,
    ) -> None:
        if not isinstance(dtype, ExactDType) or dtype.kind is not NumericKind.RATIONAL:
            raise TypeError("RationalValue requires an explicit ExactDType.rational()")
        p = integer._input_integer(numerator)
        q = integer._input_integer(denominator)
        if q == 0:
            raise ZeroDivisionError("RationalValue denominator must be nonzero")
        object.__setattr__(self, "dtype", dtype)
        # SymPy reduces the ratio and normalizes the sign. Integer results are
        # Rational subclasses too; the wrapper retains its rational dtype.
        object.__setattr__(self, "_value", sp.Rational(p, q))

    @property
    def value(self) -> sp.Rational:
        """Immutable backend scalar; extracted values follow SymPy's rules."""
        return self._value

    @property
    def numerator(self) -> sp.Integer:
        """Reduced numerator as an exact SymPy integer."""
        return sp.Integer(self._value.p)

    @property
    def denominator(self) -> sp.Integer:
        """Positive reduced denominator as an exact SymPy integer."""
        return sp.Integer(self._value.q)

    def to_integer(self, dtype: integer.IntegerDType) -> integer.IntegerValue:
        """Convert exactly, rejecting fractional values and target overflow."""
        if self._value.q != 1:
            raise ValueError("Exact integer conversion requires a denominator of one")
        return integer.IntegerValue(self.numerator, dtype=dtype)

    def __bool__(self) -> bool:
        return self._value.p != 0

    def __repr__(self) -> str:
        return (
            f"RationalValue({integer_repr(self._value.p)}, "
            f"{integer_repr(self._value.q)}, dtype={self.dtype!r})"
        )

    def __hash__(self) -> int:
        # Follow Python's numeric hash convention, including integer results.
        # SymPy nonintegral rational hashes use a different convention.
        return hash(Fraction(self._value.p, self._value.q))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, RationalValue):
            return self._value.p == other._value.p and self._value.q == other._value.q
        if isinstance(other, integer.IntegerValue):
            integral = int(other)
        elif isinstance(other, (numbers.Number, Decimal, np.generic, sp.Basic)):
            integral = integer._input_integer(other)
        else:
            return NotImplemented
        return self._value.q == 1 and self._value.p == integral

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

    def _peer(self, other: object) -> RationalValue:
        if not isinstance(other, RationalValue):
            raise TypeError(
                "Arithmetic requires RationalValue operands; convert explicitly"
            )
        return other

    def _binary(
        self,
        other: object,
        operation: Callable[[sp.Rational, sp.Rational], sp.Rational],
    ) -> RationalValue:
        peer = self._peer(other)
        result = operation(self._value, peer._value)
        return RationalValue(result.p, result.q, dtype=self.dtype)

    def __add__(self, other: object) -> RationalValue:
        return self._binary(other, operator.add)

    def __radd__(self, other: object) -> RationalValue:
        return self._peer(other).__add__(self)

    def __sub__(self, other: object) -> RationalValue:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: object) -> RationalValue:
        return self._peer(other).__sub__(self)

    def __mul__(self, other: object) -> RationalValue:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: object) -> RationalValue:
        return self._peer(other).__mul__(self)

    def __truediv__(self, other: object) -> RationalValue:
        peer = self._peer(other)
        if not peer:
            raise ZeroDivisionError("Cannot divide RationalValue by zero")
        return self._binary(peer, operator.truediv)

    def __rtruediv__(self, other: object) -> RationalValue:
        return self._peer(other).__truediv__(self)

    def __neg__(self) -> RationalValue:
        return RationalValue(-self._value.p, self._value.q, dtype=self.dtype)

    def __pos__(self) -> RationalValue:
        return self

    def _sympy_(self) -> Never:
        """Require explicit extraction instead of discarding wrapper semantics."""
        raise sp.SympifyError(self, TypeError("Extract .value explicitly"))

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        return scalar_comparison_ufunc(self, ufunc, method, *inputs, **kwargs)
