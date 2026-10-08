"""Owned homogeneous buffers with checked scalar construction and extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Never

import numpy as np
from numpy.typing import NDArray

from .casting import cast
from .complex import ComplexValue
from .dtype import NumPyDType
from .floating import FloatingValue
from .integer import IntegerValue

type BufferScalar = IntegerValue | FloatingValue | ComplexValue
type BufferIndex = int | np.integer[Any] | tuple[int | np.integer[Any], ...]


def _wrap_scalar(value: object, dtype: NumPyDType) -> BufferScalar:
    if isinstance(value, np.integer) and value.dtype.kind in "iu":
        return IntegerValue(value, dtype=dtype)
    if isinstance(value, np.floating):
        return FloatingValue(value, dtype=dtype)
    if isinstance(value, np.complexfloating):
        return ComplexValue(value.real, value.imag, dtype=dtype)
    raise TypeError("NumericBuffer requires integer, floating, or complex scalars")


@dataclass(frozen=True, slots=True, init=False, eq=False)
class NumericBuffer:
    """A private copied array; public exports never share its storage.

    Input must be a plain typed ndarray. Conversion uses scalar rules and keeps
    per-component rounding flags. Indexing extracts a scalar, never a view.
    """

    dtype: NumPyDType
    _data: NDArray[Any] = field(repr=False)
    _rounding: NDArray[np.uint8] = field(repr=False)

    def __init__(
        self, data: NDArray[Any], *, dtype: NumPyDType, approximate: bool = False
    ) -> None:
        if type(data) is not np.ndarray:
            raise TypeError("NumericBuffer requires a plain typed numpy.ndarray")
        if not isinstance(dtype, NumPyDType):
            raise TypeError("NumericBuffer requires an explicit NumPyDType")
        # Array indexing decodes non-native source byte order into native scalars.
        # Validate source metadata before resolving byte order or copying data.
        source = NumPyDType(data.dtype)
        source = NumPyDType(source.numpy_dtype.newbyteorder("="))
        # Validate both scalar formats and approximation policy even for empties.
        sample = _wrap_scalar(source.numpy_dtype.type(0), source)
        cast(sample, dtype, approximate=approximate)
        snapshot = data.copy(order="C")
        stored = np.empty(snapshot.shape, dtype=dtype.numpy_dtype, order="C")
        rounding = np.zeros(snapshot.shape, dtype=np.uint8)
        for index in np.ndindex(snapshot.shape):
            try:
                result = cast(
                    _wrap_scalar(snapshot[index], source),
                    dtype,
                    approximate=approximate,
                )
            except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
                error.add_note(f"NumericBuffer conversion failed at index {index}.")
                raise
            stored[index] = result.value
            if isinstance(result, ComplexValue):
                rounding[index] = int(result.real.rounded) | (
                    int(result.imag.rounded) << 1
                )
            elif isinstance(result, FloatingValue):
                rounding[index] = int(result.rounded)
        stored.setflags(write=False)
        rounding.setflags(write=False)
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "_data", stored)
        object.__setattr__(self, "_rounding", rounding)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def rounded(self) -> bool:
        """Whether any component was rounded during checked construction."""
        return bool(np.any(self._rounding))

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"NumericBuffer(shape={self.shape!r}, dtype={self.dtype!r}, rounded={self.rounded!r})"

    def __getitem__(self, key: BufferIndex) -> BufferScalar:
        coordinates = key if isinstance(key, tuple) else (key,)
        if any(
            isinstance(i, bool)
            or not isinstance(i, (int, np.integer))
            or (isinstance(i, np.integer) and i.dtype.kind not in "iu")
            for i in coordinates
        ):
            raise TypeError("NumericBuffer indices must be Python or NumPy integers")
        if len(coordinates) != self.ndim:
            raise IndexError(
                "NumericBuffer requires one integer coordinate per dimension"
            )
        normalized = []
        for axis, (coordinate, length) in enumerate(
            zip(coordinates, self.shape, strict=True)
        ):
            position = int(coordinate)
            if position < 0:
                position += length
            if not 0 <= position < length:
                raise IndexError(
                    f"NumericBuffer index out of range for axis {axis} of length {length}"
                )
            normalized.append(position)
        index = tuple(normalized)
        value = _wrap_scalar(self._data[index], self.dtype)
        flags = int(self._rounding[index])
        if isinstance(value, FloatingValue):
            return FloatingValue._from_stored(value.value, self.dtype, bool(flags & 1))
        if isinstance(value, ComplexValue):
            return ComplexValue._from_components(
                FloatingValue._from_stored(
                    value.real.value, value.real.dtype, bool(flags & 1)
                ),
                FloatingValue._from_stored(
                    value.imag.value, value.imag.dtype, bool(flags & 2)
                ),
                self.dtype,
            )
        return value

    def to_numpy(self) -> NDArray[Any]:
        """Export a fresh writable copy, explicitly discarding wrapper history."""
        return self._data.copy(order="C")

    def __bool__(self) -> Never:
        raise TypeError("NumericBuffer has no scalar truth value; inspect its elements")

    def __iter__(self) -> Never:
        raise TypeError("Use complete scalar indices or to_numpy() explicitly")

    def __array__(self, dtype: object = None, copy: object = None) -> Never:
        raise TypeError("Use NumericBuffer.to_numpy() explicitly")

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: object, **kwargs: object
    ) -> Never:
        raise TypeError(
            "NumericBuffer does not implement ufuncs; use to_numpy() explicitly"
        )
