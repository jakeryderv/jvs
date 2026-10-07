# Rational Values

`RationalValue` is an immutable finite exact ratio stored with SymPy and an
explicit `ExactDType.rational()` descriptor. It does not inherit the mathematical
`Rational` set and is independent of `jvs.core`.

## Construction and Membership

```python
from jvs.numeric import (
    ExactDType, Integer, NumPyDType, Rational, RationalValue,
)

rational = ExactDType.rational()
x = RationalValue(6, -8, dtype=rational)
assert x.numerator == -3
assert x.denominator == 4
assert x in Rational
assert x not in Integer

whole = RationalValue(6, 3, dtype=rational)
assert whole in Integer
assert whole.dtype == rational
```

The numerator and denominator accept Python, NumPy, or SymPy integer scalars.
The denominator defaults to one; `dtype` is required. Booleans, floats, decimals,
strings, arrays, symbolic variables, and raw nonintegral rationals are rejected.
Integer wrappers require explicit extraction with `int(x)` or `x.value`.
For an existing `Fraction`, pass its `numerator` and `denominator`; for a SymPy
rational, pass its `p` and `q`. These are explicit exact conversions.

A zero denominator raises `ZeroDivisionError`, including `0/0`. SymPy reduces
the ratio and makes its denominator positive; zero becomes `0/1`.
`numerator` and `denominator` return exact SymPy integers. `value` exposes the
immutable SymPy scalar, which can be an `Integer` when the ratio reduces to one.
The wrapper and its rational dtype are retained. Membership reflects the actual
value, including integrality, `Whole`, and `Natural`.

`repr(x)` formats each integer component using the [bounded integer diagnostic
format](integer.md#construction-and-membership): ordinary values remain decimal,
and values above 256 bits are explicitly abbreviated. This does not change the
ratio or Python's global digit limit and is not a serialization format.

The restricted constructor deliberately validates before calling
[SymPy's Rational constructor](https://docs.sympy.org/latest/modules/core.html#sympy.core.numbers.Rational),
which also accepts floats and strings. Those broader backend inputs are outside
this wrapper's contract.

## Arithmetic and Integer Conversion

The shared [`cast(value, dtype)` API](casting.md) also converts rational wrappers
to supported scalar families, with exact conversion by default.
Use the [explicit arithmetic functions](arithmetic.md) to combine different source
families in a chosen rational or other supported operand representation.

```python
a = RationalValue(1, 3, dtype=rational)
b = RationalValue(2, 3, dtype=rational)

assert a + b == RationalValue(1, dtype=rational)
assert a - b == -a
assert a * b == RationalValue(2, 9, dtype=rational)
assert a / b == RationalValue(1, 2, dtype=rational)

integer = (a + b).to_integer(NumPyDType("int8"))
assert int(integer) == 1
```

Addition, subtraction, multiplication, and true division require two rational
wrappers. SymPy computes exact results, which retain the rational dtype, including
integral results. Division by zero raises `ZeroDivisionError` before calling the
backend. Unary negation stays exact; unary plus returns the same immutable object.
Mixed arithmetic with integer wrappers or raw scalars raises `TypeError`.

`to_integer(dtype)` requires a reduced denominator of one, otherwise it raises
`ValueError`. It constructs an `IntegerValue` using the requested integer dtype
and its existing range and representation checks. Out-of-range values raise
`OverflowError`. No truncation, rounding, or automatic widening occurs.

There is no implicit `int`, index, or float conversion. `bool(x)` is false only
for zero. Ordering, floor division, modulo, powers, and mixed-type promotion
remain deferred. Dividing two matching-dtype `IntegerValue` operands also produces
a `RationalValue` directly, with these same exact ratio semantics; see
[integer division](integer.md#checked-arithmetic-and-conversion).

## Equality, Hashing, and Backend Boundaries

Equality compares reduced values across rational wrappers, integer wrappers,
[floating wrappers](floating.md), [complex wrappers](complex.md),
[fixed-point wrappers](fixed.md), and supported
Python/NumPy/SymPy integer scalars. A complex value can compare equal only when its
stored imaginary component is zero.
Floating comparison uses the stored binary value, not intended exactness.
Dtype does not determine equality.
Hashes follow Python's numeric convention using the canonical exact ratio, so
equal supported values behave consistently as dictionary keys and set members.

When our comparison method runs, other numeric comparisons raise and require
explicit conversion, including raw
`Fraction` and nonintegral SymPy `Rational` values. This restriction avoids
promising compatibility across conflicting backend hash conventions: the current
SymPy backend can compare equal to a Python `Fraction` while hashing differently.
An integral SymPy rational is canonicalized to a SymPy `Integer` and is accepted.
Unrelated nonnumeric objects use Python's `NotImplemented` fallback.

NumPy equality/inequality supports the documented wrapper and integer comparisons
in either operand order, including its zero-dimensional integer array transport,
as with `IntegerValue`. Arrays with axes, noninteger arrays, ufunc
arithmetic, and equality options such as `out` or `where` raise `TypeError`.
Implicit SymPy coercion raises `SympifyError`; this can also be the error from a
foreign SymPy operation or comparison.

Foreign comparisons may return before reaching our methods, so unsupported
operand rejection cannot be guaranteed in that case; see
[comparison dispatch boundaries](integer.md#backend-boundaries).

Extracting `.value` explicitly transfers control to SymPy. Subsequent operations
follow that backend's semantics, including its coercion, equality, hashing, and
nonfinite-result behavior.

Use `FloatingValue(x, dtype=...)` to convert a rational wrapper exactly to a floating
format, or `FloatingValue.approx(x, dtype=...)` to explicitly permit rounding.
The floating wrapper's `to_rational()` extracts its stored binary ratio; it does
not recover the original intended value or carry floating provenance.
