"""Immutable integer values with explicit representation and checked arithmetic."""

from __future__ import annotations

import numbers
import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, ClassVar, Never

import numpy as np
import sympy as sp

# Module-qualified references keep runtime annotations resolvable across the
# integer/rational dependency without reading a class during partial import.
from . import rational
from ._interop import scalar_equality_ufunc
from ._repr import integer_repr
from .dtype import ExactDType, NumericKind, NumPyDType

type IntegerInput = int | np.integer[Any] | sp.Integer
type IntegerDType = NumPyDType | ExactDType


def _input_integer(value: object) -> int:
    """Read a supported integer exactly, without duck-typed numeric conversion."""
    if isinstance(value, bool):
        raise TypeError("Booleans are not integer inputs")
    if isinstance(value, (int, sp.Integer)):
        return int(value)
    if isinstance(value, np.integer) and value.dtype.kind in "iu":
        return int(value)
    raise TypeError(
        "Expected a Python, NumPy, or SymPy integer; "
        "convert other representations explicitly"
    )


@dataclass(frozen=True, slots=True, init=False, eq=False)
class IntegerValue:
    """An integer stored in an explicitly selected NumPy or SymPy representation.

    Construction rejects booleans and non-integer input representations, even
    when their values are integral. Binary arithmetic requires matching wrapper
    dtypes. Equality and hashing compare exact integer values across dtypes and
    supported Python/NumPy/SymPy integers; other numeric comparisons raise.
    """

    dtype: IntegerDType
    _value: np.integer[Any] | sp.Integer = field(repr=False)

    # SymPy must dispatch mixed arithmetic to our reflected methods instead of
    # silently sympifying this value and discarding its representation contract.
    _op_priority: ClassVar[float] = 1000.0

    def __init__(self, value: IntegerInput, *, dtype: IntegerDType) -> None:
        if not isinstance(dtype, (NumPyDType, ExactDType)):
            raise TypeError("dtype must be an explicit NumPyDType or ExactDType")
        if dtype.kind is not NumericKind.INTEGER:
            raise TypeError("IntegerValue requires an integer dtype")
        exact = _input_integer(value)
        if isinstance(dtype, NumPyDType):
            if not dtype.numpy_dtype.isnative:
                raise ValueError(
                    "IntegerValue requires native byte order for NumPy scalars; "
                    "non-native byte order belongs to array/storage representations"
                )
            limits = dtype.integer_info
            if not int(limits.min) <= exact <= int(limits.max):
                # Do not stringify an arbitrarily large input: Python's decimal
                # digit limit must not change OverflowError into ValueError.
                raise OverflowError(
                    f"Integer is outside {dtype.name} range "
                    f"[{int(limits.min)}, {int(limits.max)}]"
                )
            stored = dtype.numpy_dtype.type(exact)
        else:
            stored = sp.Integer(exact)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "_value", stored)

    @property
    def value(self) -> np.integer[Any] | sp.Integer:
        """Immutable backend scalar; operations on it follow backend rules."""
        return self._value

    def to(self, dtype: IntegerDType) -> IntegerValue:
        """Explicit, exact conversion; target range and dtype checks still apply."""
        return IntegerValue(int(self), dtype=dtype)

    def __int__(self) -> int:
        return int(self._value)

    def __index__(self) -> int:
        return int(self)

    def __bool__(self) -> bool:
        return int(self) != 0

    def __repr__(self) -> str:
        return f"IntegerValue({integer_repr(int(self))}, dtype={self.dtype!r})"

    def __hash__(self) -> int:
        return hash(int(self))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, IntegerValue):
            return int(self) == int(other)
        if isinstance(other, (numbers.Number, Decimal, np.generic, sp.Basic)):
            return int(self) == _input_integer(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def _peer(self, other: object) -> IntegerValue:
        if not isinstance(other, IntegerValue):
            raise TypeError(
                "Arithmetic requires IntegerValue operands with matching dtypes"
            )
        if self.dtype != other.dtype:
            raise TypeError(
                "Mixed integer dtypes require explicit conversion with .to(dtype)"
            )
        return other

    def _binary(
        self, other: object, operation: Callable[[int, int], int]
    ) -> IntegerValue:
        peer = self._peer(other)
        # Exact unbounded intermediates avoid performing overflowing NumPy
        # arithmetic first. Construction checks before storing the result.
        return IntegerValue(operation(int(self), int(peer)), dtype=self.dtype)

    def __add__(self, other: object) -> IntegerValue:
        return self._binary(other, operator.add)

    def __radd__(self, other: object) -> IntegerValue:
        return self._peer(other).__add__(self)

    def __sub__(self, other: object) -> IntegerValue:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: object) -> IntegerValue:
        return self._peer(other).__sub__(self)

    def __mul__(self, other: object) -> IntegerValue:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: object) -> IntegerValue:
        return self._peer(other).__mul__(self)

    def __truediv__(self, other: object) -> rational.RationalValue:
        """Divide matching integer representations into an exact rational."""
        peer = self._peer(other)
        if not peer:
            raise ZeroDivisionError("Cannot divide IntegerValue by zero")
        return rational.RationalValue(int(self), int(peer), dtype=ExactDType.rational())

    def __rtruediv__(self, other: object) -> rational.RationalValue:
        return self._peer(other).__truediv__(self)

    def __neg__(self) -> IntegerValue:
        return IntegerValue(-int(self), dtype=self.dtype)

    def __pos__(self) -> IntegerValue:
        return self

    def _sympy_(self) -> Never:
        """Require explicit backend extraction instead of implicit sympification."""
        raise sp.SympifyError(
            self, TypeError("Extract .value or int(value) explicitly")
        )

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        return scalar_equality_ufunc(self, ufunc, method, *inputs, **kwargs)
