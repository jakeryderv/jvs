# Floating Values

`FloatingValue` is an immutable finite NumPy floating scalar with an explicit
`NumPyDType`. Construction and conversion preserve the supplied value exactly by
default. Approximation is a separate, explicit operation. The dtype describes
storage precision; it does not promise accuracy relative to a user's intended
number or measurement.

## Construction and conversion

The shared [`cast(value, dtype)` API](casting.md) provides the same checked
conversion policies between existing scalar wrappers. Raw inputs still use the
constructors described here.

```python
from fractions import Fraction
from jvs.numeric import FloatingValue, NumPyDType

f32 = NumPyDType("float32")
half = FloatingValue(Fraction(1, 2), dtype=f32)
third = FloatingValue.approx(Fraction(1, 3), dtype=f32)
assert third.rounded
wider = third.to(NumPyDType("float64"))
assert wider.rounded  # Widening preserves known rounding history.
```

Supported inputs are Python and NumPy integer/floating scalars, exact SymPy
integers/rationals, `Fraction`, and the integer, rational, floating, and fixed-point wrappers.
Booleans, complex values, strings, arrays, decimals, SymPy floats and symbolic
expressions are excluded from this first API. Unsupported inputs raise
`TypeError`; nonfinite floating inputs raise `ValueError`.

Python/NumPy floating inputs are interpreted as their stored binary value.
For example, preserving Python `0.1` in float64 does not assert that it equals the
exact decimal `1/10`. Use an exact ratio to specify that intent. There is no
implicit conversion through Python float, including for extended NumPy scalars.

`FloatingValue(value, dtype=...)` and `.to(dtype)` raise `PrecisionLossError`
when preserving the value requires rounding. `FloatingValue.approx(value,
dtype=...)` explicitly permits round-to-nearest, ties-to-even; `.to(dtype,
approximate=True)` uses the same rule. Neither path permits overflow or inexact
underflow. Explicit approximation does not emit routine rounding warnings.

Supported native-byte-order formats are binary16, binary32, binary64, x87 binary
extended precision, and binary128 when actually supplied by NumPy. Format
metadata and subnormal spacing are validated; other formats, including IBM
double-double, raise `NotImplementedError`. NumPy's `longdouble` is platform
dependent, not a portable precision promise. Non-native storage byte order is
rejected for scalar payloads.

## Arithmetic and exceptional results

The [explicit arithmetic functions](arithmetic.md) can cast mixed source wrappers
into a chosen floating dtype first. Their `approximate` option controls operand
conversion, while the arithmetic rounding rules below always apply.

`+`, `-`, `*`, and `/` require two floating wrappers with equal dtypes. Results
retain that dtype. Normal rounding is part of floating arithmetic and uses
round-to-nearest, ties-to-even. Mixed dtypes and raw operands require explicit
conversion. Unary plus and negation preserve the representation and rounding
history.

- Values or exact arithmetic results outside the finite representable range raise
  `OverflowError`, even if ordinary rounding could produce the maximum finite value.
- Division by either sign of zero raises `ZeroDivisionError`, including `0/0`.
- An inexact nonzero value/result with magnitude below `smallest_normal` raises
  `UnderflowError`, including rounding to zero or up to the smallest normal value.
  This is a deliberate tininess-before-rounding policy.
- Exactly representable subnormals are accepted. Nonfinite results are never stored.

Checks compare exact integer ratios of the supplied/stored values, independently
of NumPy warning settings. NumPy supplies the stored scalars and arithmetic;
backend results are checked against the specified rounding rule. An unexpected
backend result raises `FloatingPointError`. Error settings are scoped and restored.
These scalar checks prioritize correctness and are not a vectorized fast path.

## Transparency and interoperability

`value` exposes the NumPy scalar. `rounded` records rounding detected during
construction, conversion, or arithmetic within this wrapper API and propagates
through subsequent wrapper operations. It is not an error bound: false means no
known rounding in that history, not proof of accurate external input. Arithmetic
operates on stored values and cannot recover discarded information.

`.to_rational()` returns the exact ratio of the stored binary value, explicitly
discarding floating provenance. `.to_integer(dtype)` additionally requires that
the stored value be integral and fit the requested integer representation.
Neither operation claims to recover the intended source value. Integer/rational
wrappers convert into floating values through the constructor or `approx`.
Fixed-point wrappers also use those constructors, retaining known rounding
history; their coefficient times step is converted without an intermediate float.
Implicit `int`, index, and Python `float` conversion are absent; extract `.value`
when backend behavior is desired.

Equality compares stored mathematical values across floating, integer, rational, fixed-point,
and [complex wrappers](complex.md), and supported Python/NumPy/SymPy integer
scalars, without rounding. A complex wrapper can compare equal only when its
imaginary component is zero. Hashes
follow Python's exact numeric convention. Raw floating and rational comparisons
require explicit wrapping, consistent with the other wrappers; external extended
floating and symbolic rational hashes need not follow that convention. Dtype and
rounding history do not affect equality.

Signed zero is preserved on input, conversion, and arithmetic. The two signs
compare equal, share a hash, and are false in Boolean context. Exact rational or
integer extraction discards the zero sign. Inspect `np.signbit(x.value)` for the
representation sign. Mathematical classification uses the stored finite value,
not intended exactness or rounding history.

NumPy equality/inequality supports the documented wrapper and integer comparisons
in either operand order. Scalar integer equality may arrive through
zero-dimensional integer arrays.
Other array operations and ufunc arithmetic are rejected, as are implicit SymPy
coercions. Foreign comparison methods can return without calling our methods;
unsupported-operand rejection applies when our methods run. See
[comparison dispatch boundaries](integer.md#backend-boundaries).
`<`, `<=`, `>`, and `>=` implement [exact ordering](ordering.md) across the four
real-valued wrappers and supported raw integers. Comparisons retain the stored
binary ratio, including extended precision and subnormal values, without any
rounding or history changes. Both zero signs occupy the same place in the
ordering. NumPy ordering ufuncs use these rules too. Complex wrappers require
explicit `.to_real()` before ordering, even with zero imaginary part.

Powers, floor division, modulo, user-selected rounding modes,
permissive warning fallbacks, and arbitrary precision are deferred. `ComplexValue`
composes two checked floating components. Pass this wrapper as its real component
to preserve history; a complex wrapper's `to_real()` requires a zero imaginary
component before any further floating conversion.

## References

- [NumPy floating formats and platform differences](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
- [Scoped NumPy error handling](https://numpy.org/doc/stable/reference/generated/numpy.errstate.html)
