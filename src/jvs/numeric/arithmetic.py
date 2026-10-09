"""Checked arithmetic in an explicitly selected operand representation."""

from __future__ import annotations

import operator
from collections.abc import Callable
from fractions import Fraction
from math import gcd, lcm
from typing import overload

import numpy as np

from .casting import NumericValue, ScalarDType, cast
from .dtype import ExactDType, FixedDType, FloatingInfo, NumericKind, NumPyDType
from .floating import _format, _ratio
from .integer import IntegerDType
from .storage import NumericBuffer, _validate_conversion


def _validate_scalar_dtype(dtype: ScalarDType) -> None:
    if not isinstance(dtype, (NumPyDType, ExactDType, FixedDType)):
        raise TypeError("common_dtype requires explicit scalar dtype descriptors")
    if isinstance(dtype, FixedDType):
        _validate_scalar_dtype(dtype.coefficient_dtype)
    elif isinstance(dtype, NumPyDType):
        if not dtype.numpy_dtype.isnative:
            raise ValueError(f"Scalar promotion requires native byte order: {dtype!r}")
        if dtype.kind is NumericKind.FLOATING:
            _format(dtype)
        elif dtype.kind is NumericKind.COMPLEX:
            _format(dtype.component_dtype)


def _no_common_dtype(left: ScalarDType, right: ScalarDType) -> ValueError:
    return ValueError(
        f"No permitted common dtype for {left!r} and {right!r}; "
        "choose an explicit target and conversion policy"
    )


def _covers_float(target: FloatingInfo, source: FloatingInfo) -> bool:
    return (
        target.significand_bits >= source.significand_bits
        and _ratio(target.max) >= _ratio(source.max)
        and _ratio(target.smallest_subnormal) <= _ratio(source.smallest_subnormal)
    )


def _common_inexact(left: NumPyDType, right: NumPyDType) -> NumPyDType:
    complex_target = NumericKind.COMPLEX in (left.kind, right.kind)
    sources = [
        _format(dtype.component_dtype if dtype.kind is NumericKind.COMPLEX else dtype)
        for dtype in (left, right)
    ]
    candidates = (
        (np.complex64, np.complex128, np.clongdouble)
        if complex_target
        else (np.float16, np.float32, np.float64, np.longdouble)
    )
    seen: set[NumPyDType] = set()
    for scalar in candidates:
        candidate = NumPyDType(scalar)
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            info = _format(candidate.component_dtype if complex_target else candidate)
        except NotImplementedError:
            # Unsupported inputs were already rejected; only candidates may be skipped.
            continue
        if all(_covers_float(info, source) for source in sources):
            return candidate
    raise _no_common_dtype(left, right)


def _lattice(dtype: ScalarDType) -> tuple[IntegerDType, Fraction]:
    if isinstance(dtype, FixedDType):
        return dtype.coefficient_dtype, dtype.step
    # Called only after excluding rational and floating/complex families.
    return dtype, Fraction(1)


def _common_lattice(left: ScalarDType, right: ScalarDType) -> ScalarDType:
    a, sa = _lattice(left)
    b, sb = _lattice(right)
    step = Fraction(
        gcd(sa.numerator, sb.numerator), lcm(sa.denominator, sb.denominator)
    )
    storage: IntegerDType
    if isinstance(a, ExactDType) or isinstance(b, ExactDType):
        storage = ExactDType.integer()
    else:
        # The rational gcd guarantees integral positive scale factors. Compute
        # full scaled ranges in unbounded integers, before choosing storage.
        ka, kb = (sa / step).numerator, (sb / step).numerator
        ai, bi = a.integer_info, b.integer_info
        low = min(int(ai.min) * ka, int(bi.min) * kb)
        high = max(int(ai.max) * ka, int(bi.max) * kb)
        for bits in (8, 16, 32, 64):
            candidate = NumPyDType(f"{'int' if low < 0 else 'uint'}{bits}")
            limits = candidate.integer_info
            if int(limits.min) <= low and high <= int(limits.max):
                storage = candidate
                break
        else:
            raise _no_common_dtype(left, right)
    if isinstance(left, FixedDType) or isinstance(right, FixedDType):
        return FixedDType(storage, step=step)
    return storage


def common_dtype(left: ScalarDType, right: ScalarDType) -> ScalarDType:
    """Select a dtype covering both complete finite operand-value domains exactly.

    Selection is explicit, symmetric, and independent of actual operand values.
    Exact/approximate family mixtures require a caller-selected target. NumPy
    range exhaustion raises rather than switching backends. A selected dtype
    does not guarantee that subsequent arithmetic results fit or remain exact.
    """
    _validate_scalar_dtype(left)
    _validate_scalar_dtype(right)
    if left == right:
        return left
    inexact = (NumericKind.FLOATING, NumericKind.COMPLEX)
    if left.kind in inexact or right.kind in inexact:
        if left.kind not in inexact or right.kind not in inexact:
            raise TypeError(
                f"Cannot promote exact/approximate families {left!r} and {right!r}; "
                "choose an explicit target"
            )
        if isinstance(left, NumPyDType) and isinstance(right, NumPyDType):
            return _common_inexact(left, right)
        raise TypeError("Floating/complex promotion requires NumPy descriptors")
    if NumericKind.RATIONAL in (left.kind, right.kind):
        return ExactDType.rational()
    return _common_lattice(left, right)


def _apply(
    operation: Callable[[NumericValue, NumericValue], NumericValue],
    a: NumericValue,
    b: NumericValue,
    dtype: ScalarDType,
    approximate: bool,
) -> NumericValue:
    left = cast(a, dtype, approximate=approximate)
    right = cast(b, dtype, approximate=approximate)
    return operation(left, right)


def _dispatch(
    operation: Callable[[NumericValue, NumericValue], NumericValue],
    a: NumericValue | NumericBuffer,
    b: NumericValue | NumericBuffer,
    dtype: ScalarDType,
    approximate: bool,
) -> NumericValue | NumericBuffer:
    if isinstance(a, NumericBuffer) or isinstance(b, NumericBuffer):
        if not isinstance(a, NumericBuffer) or not isinstance(b, NumericBuffer):
            raise TypeError("Buffer arithmetic requires two NumericBuffer operands")
        if not isinstance(dtype, NumPyDType):
            raise TypeError("Buffer arithmetic requires an explicit NumPyDType")
        if a.shape != b.shape:
            raise ValueError(
                f"Buffer arithmetic requires identical shapes: {a.shape!r} != {b.shape!r}"
            )
        _validate_conversion(a.dtype, dtype, approximate)
        _validate_conversion(b.dtype, dtype, approximate)
        return NumericBuffer._from_elements(
            a.shape,
            dtype,
            lambda index: _apply(operation, a[index], b[index], dtype, approximate),
            context="arithmetic",
        )
    return _apply(operation, a, b, dtype, approximate)


@overload
def add(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue: ...


@overload
def add(
    a: NumericBuffer,
    b: NumericBuffer,
    *,
    dtype: NumPyDType,
    approximate: bool = False,
) -> NumericBuffer: ...


def add(
    a: NumericValue | NumericBuffer,
    b: NumericValue | NumericBuffer,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue | NumericBuffer:
    """Cast operands, then add with checked arithmetic in the requested dtype.

    ``approximate`` controls operand conversion only; floating/complex arithmetic
    retains its normal rounding rules even when this option is false.
    Two buffers require identical shapes and a native NumPy target.
    """
    return _dispatch(operator.add, a, b, dtype, approximate)


@overload
def subtract(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue: ...


@overload
def subtract(
    a: NumericBuffer,
    b: NumericBuffer,
    *,
    dtype: NumPyDType,
    approximate: bool = False,
) -> NumericBuffer: ...


def subtract(
    a: NumericValue | NumericBuffer,
    b: NumericValue | NumericBuffer,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue | NumericBuffer:
    """Subtract scalars or matching-shape buffers in the requested representation."""
    return _dispatch(operator.sub, a, b, dtype, approximate)


@overload
def multiply(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue: ...


@overload
def multiply(
    a: NumericBuffer,
    b: NumericBuffer,
    *,
    dtype: NumPyDType,
    approximate: bool = False,
) -> NumericBuffer: ...


def multiply(
    a: NumericValue | NumericBuffer,
    b: NumericValue | NumericBuffer,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue | NumericBuffer:
    """Multiply scalars or matching-shape buffers using checked scalar rules."""
    return _dispatch(operator.mul, a, b, dtype, approximate)


def divide(
    a: NumericValue,
    b: NumericValue,
    *,
    dtype: ScalarDType,
    approximate: bool = False,
) -> NumericValue:
    """Cast operands, then divide; integer and fixed operands yield exact rationals.

    Other operand families retain their dtype. Zero divisors raise explicitly.
    """
    if isinstance(a, NumericBuffer) or isinstance(b, NumericBuffer):
        raise TypeError("Buffer division is not implemented")
    return _apply(operator.truediv, a, b, dtype, approximate)
