"""Finite complex values composed from checked floating components."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Never

import numpy as np
import sympy as sp

from ._interop import scalar_equality_ufunc
from .dtype import NumericKind, NumPyDType
from .floating import FloatingInput, FloatingValue, _ratio


def _exact_product_sum(
    a: FloatingValue,
    b: FloatingValue,
    c: FloatingValue,
    d: FloatingValue,
    *,
    subtract: bool = False,
) -> tuple[Fraction, bool]:
    """Return ab +/- cd exactly, with the formula's negative-zero flag."""
    left = _ratio(a.value) * _ratio(b.value)
    right = _ratio(c.value) * _ratio(d.value)
    result = left - right if subtract else left + right
    left_negative = bool(np.signbit(a.value)) ^ bool(np.signbit(b.value))
    right_negative = bool(np.signbit(c.value)) ^ bool(np.signbit(d.value)) ^ subtract
    # Nonzero cancellation yields +0. Two zero terms yield -0 only when both
    # effective signs are negative, including the sign flip for subtraction.
    negative_zero = not left and not right and left_negative and right_negative
    return result, negative_zero


def _rounded_component(
    exact: Fraction, negative_zero: bool, dtype: NumPyDType, history: bool
) -> FloatingValue:
    result = FloatingValue.approx(exact, dtype=dtype)
    if negative_zero:
        result = -result
    return FloatingValue._from_stored(result.value, dtype, history or result.rounded)


def _components(
    real: FloatingInput, imag: FloatingInput, dtype: NumPyDType, *, approximate: bool
) -> tuple[FloatingValue, FloatingValue]:
    if not isinstance(dtype, NumPyDType) or dtype.kind is not NumericKind.COMPLEX:
        raise TypeError("ComplexValue requires an explicit complex NumPyDType")
    if not dtype.numpy_dtype.isnative:
        raise ValueError("ComplexValue requires native byte order for NumPy scalars")
    component_dtype = dtype.component_dtype
    constructor = FloatingValue.approx if approximate else FloatingValue
    return constructor(real, dtype=component_dtype), constructor(
        imag, dtype=component_dtype
    )


def _pack(
    real: FloatingValue, imag: FloatingValue, dtype: NumPyDType
) -> np.complexfloating[Any, Any]:
    # Assign components independently: real + 1j * imag can lose zero signs and
    # routing through Python complex would narrow extended-precision components.
    buffer = np.empty((), dtype=dtype.numpy_dtype)
    buffer.real[()] = real.value
    buffer.imag[()] = imag.value
    # NumPy's static indexing signature does not identify this 0-D scalar result.
    stored: Any = buffer[()]
    if not isinstance(stored, np.complexfloating) or stored.dtype != dtype.numpy_dtype:
        raise FloatingPointError("Backend did not produce the requested complex scalar")
    for actual, expected in ((stored.real, real.value), (stored.imag, imag.value)):
        if (
            not bool(np.isfinite(actual))
            or actual.as_integer_ratio() != expected.as_integer_ratio()
            or bool(np.signbit(actual)) != bool(np.signbit(expected))
        ):
            raise FloatingPointError(
                "Backend failed to preserve a checked complex component"
            )
    return stored


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ComplexValue:
    """A NumPy complex scalar with finite checked floating components.

    Construction and conversion are exact unless approximation is requested.
    Arithmetic requires matching complex dtypes. Multiplication and division use
    exact intermediates and round each final component once, nearest-even.
    """

    dtype: NumPyDType
    real: FloatingValue
    imag: FloatingValue
    _value: np.complexfloating[Any, Any] = field(repr=False)
    _op_priority: ClassVar[float] = 1000.0

    def __init__(
        self, real: FloatingInput, imag: FloatingInput = 0, *, dtype: NumPyDType
    ) -> None:
        re, im = _components(real, imag, dtype, approximate=False)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "real", re)
        object.__setattr__(self, "imag", im)
        object.__setattr__(self, "_value", _pack(re, im, dtype))

    @classmethod
    def _from_components(
        cls, real: FloatingValue, imag: FloatingValue, dtype: NumPyDType
    ) -> ComplexValue:
        result = object.__new__(cls)
        object.__setattr__(result, "dtype", dtype)
        object.__setattr__(result, "real", real)
        object.__setattr__(result, "imag", imag)
        object.__setattr__(result, "_value", _pack(real, imag, dtype))
        return result

    @classmethod
    def approx(
        cls, real: FloatingInput, imag: FloatingInput = 0, *, dtype: NumPyDType
    ) -> ComplexValue:
        """Explicit componentwise approximation with floating range checks."""
        re, im = _components(real, imag, dtype, approximate=True)
        return cls._from_components(re, im, dtype)

    @property
    def value(self) -> np.complexfloating[Any, Any]:
        """Immutable backend scalar; extracted values follow NumPy's rules."""
        return self._value

    @property
    def rounded(self) -> bool:
        """Whether either component carries known rounding inside the wrapper API."""
        return self.real.rounded or self.imag.rounded

    def to(self, dtype: NumPyDType, *, approximate: bool = False) -> ComplexValue:
        if not isinstance(approximate, bool):
            raise TypeError("approximate must be a bool")
        re, im = _components(self.real, self.imag, dtype, approximate=approximate)
        return self._from_components(re, im, dtype)

    def to_real(self) -> FloatingValue:
        """Extract a real value only for zero imaginary part, retaining history."""
        if self.imag:
            raise ValueError("Real conversion requires a zero imaginary component")
        return FloatingValue._from_stored(
            self.real.value, self.real.dtype, self.rounded
        )

    def __bool__(self) -> bool:
        return bool(self.real) or bool(self.imag)

    def __repr__(self) -> str:
        return f"ComplexValue(real={self.real!r}, imag={self.imag!r}, dtype={self.dtype!r})"

    def __hash__(self) -> int:
        combined = hash(self.real) + sys.hash_info.imag * hash(self.imag)
        half_range = 1 << (sys.hash_info.width - 1)
        combined = (combined + half_range) % (2 * half_range) - half_range
        return -2 if combined == -1 else combined

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ComplexValue):
            return self.real == other.real and self.imag == other.imag
        # Validate the comparison domain even for a nonzero imaginary component.
        result = self.real.__eq__(other)
        if result is NotImplemented:
            return NotImplemented
        return not self.imag and result

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result

    def _peer(self, other: object) -> ComplexValue:
        if not isinstance(other, ComplexValue):
            raise TypeError(
                "Arithmetic requires ComplexValue operands with matching dtypes"
            )
        if self.dtype != other.dtype:
            raise TypeError(
                "Mixed complex dtypes require explicit conversion with .to(dtype)"
            )
        return other

    def __add__(self, other: object) -> ComplexValue:
        peer = self._peer(other)
        return self._from_components(
            self.real + peer.real, self.imag + peer.imag, self.dtype
        )

    def __radd__(self, other: object) -> ComplexValue:
        return self._peer(other).__add__(self)

    def __sub__(self, other: object) -> ComplexValue:
        peer = self._peer(other)
        return self._from_components(
            self.real - peer.real, self.imag - peer.imag, self.dtype
        )

    def __rsub__(self, other: object) -> ComplexValue:
        return self._peer(other).__sub__(self)

    def __neg__(self) -> ComplexValue:
        return self._from_components(-self.real, -self.imag, self.dtype)

    def __pos__(self) -> ComplexValue:
        return self

    def conjugate(self) -> ComplexValue:
        return self._from_components(self.real, -self.imag, self.dtype)

    def _product_result(
        self,
        peer: ComplexValue,
        real: tuple[Fraction, bool],
        imag: tuple[Fraction, bool],
    ) -> ComplexValue:
        history = self.rounded or peer.rounded
        component_dtype = self.real.dtype
        return self._from_components(
            _rounded_component(*real, component_dtype, history),
            _rounded_component(*imag, component_dtype, history),
            self.dtype,
        )

    def __mul__(self, other: object) -> ComplexValue:
        peer = self._peer(other)
        real = _exact_product_sum(
            self.real, peer.real, self.imag, peer.imag, subtract=True
        )
        imag = _exact_product_sum(self.real, peer.imag, self.imag, peer.real)
        return self._product_result(peer, real, imag)

    def __rmul__(self, other: object) -> ComplexValue:
        return self._peer(other).__mul__(self)

    def __truediv__(self, other: object) -> ComplexValue:
        peer = self._peer(other)
        if not peer:
            raise ZeroDivisionError("Cannot divide ComplexValue by zero")
        denominator = _ratio(peer.real.value) ** 2 + _ratio(peer.imag.value) ** 2
        real, real_negative_zero = _exact_product_sum(
            self.real, peer.real, self.imag, peer.imag
        )
        imag, imag_negative_zero = _exact_product_sum(
            self.imag, peer.real, self.real, peer.imag, subtract=True
        )
        return self._product_result(
            peer,
            (real / denominator, real_negative_zero),
            (imag / denominator, imag_negative_zero),
        )

    def __rtruediv__(self, other: object) -> ComplexValue:
        return self._peer(other).__truediv__(self)

    def _sympy_(self) -> Never:
        raise sp.SympifyError(self, TypeError("Extract .value explicitly"))

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> object:
        return scalar_equality_ufunc(self, ufunc, method, *inputs, **kwargs)
