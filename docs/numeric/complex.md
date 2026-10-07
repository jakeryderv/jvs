# Complex Values

`ComplexValue` is an immutable finite complex value with an explicit native NumPy
complex dtype. Its `real` and `imag` components are checked `FloatingValue`
wrappers using the dtype's `component_dtype`. It does not inherit the mathematical
`Complex` set and remains independent of `jvs.core`.

## Construction and representation

```python
from fractions import Fraction
from jvs.numeric import ComplexValue, FloatingValue, NumPyDType

c64 = NumPyDType("complex64")
z = ComplexValue(Fraction(1, 2), -2, dtype=c64)
assert z.real == ComplexValue(0.5, dtype=c64).real
assert z.imag == -2
assert z.real.dtype == c64.component_dtype

third = ComplexValue.approx(Fraction(1, 3), 0, dtype=c64)
assert third.real.rounded
assert third.rounded
```

The constructor takes separate real and imaginary inputs; the imaginary input
defaults to zero. Each component accepts exactly the inputs supported by
[`FloatingValue`](floating.md). To import an existing Python/NumPy complex scalar,
pass its `.real` and `.imag` explicitly. Raw complex objects, arrays, booleans, and
symbolic expressions are not accepted as components. For an existing complex
wrapper, use `.to(dtype)` or pass its component wrappers to preserve history.

Construction preserves each supplied stored/exact value or raises
`PrecisionLossError`. `ComplexValue.approx(real, imag, dtype=...)` explicitly
permits nearest-even rounding. Overflow, inexact underflow, nonfinite inputs,
unsupported component formats, and non-native scalar byte order fail according
to the floating contract. No intermediate Python complex/float conversion occurs.
The NumPy scalar is assembled from the checked components without arithmetic.

`value` exposes the immutable NumPy complex scalar. `real` and `imag` retain
component-specific rounding history; `rounded` is true if either component has
known rounding. It is not a claim about accuracy of the original input.
Extracting `.value` transfers control to NumPy and discards wrapper history.

## Operations and conversions

The shared [`cast(value, dtype)` API](casting.md) dispatches conversions between
existing scalar wrappers. Real targets require a stored zero imaginary component;
approximation never permits discarding a nonzero imaginary part.
The [explicit arithmetic functions](arithmetic.md) also accept a chosen operand
dtype, casting both wrappers before applying the operations described here.

Addition and subtraction require complex wrappers with equal dtypes. They apply
the checked floating operations separately to the real and imaginary components.
Normal rounding is permitted; range failures raise and no partial result is
returned. Each result retains the complex dtype and per-component history.
Unary plus returns the same immutable object. Unary negation negates both
components; `.conjugate()` negates only the imaginary component.

```python
w = ComplexValue(1, 2, dtype=c64)
assert z + w == ComplexValue(1.5, 0, dtype=c64)
assert z.conjugate() == ComplexValue(0.5, 2, dtype=c64)
assert (z + w).to_real() == FloatingValue(1.5, dtype=c64.component_dtype)
```

Multiplication and division also require matching complex dtypes. For stored
components `a + bi` and `c + di`, they compute exact rational intermediates:

```text
product  = (ac - bd) + (ad + bc)i
quotient = (ac + bd)/(c² + d²) + (bc - ad)/(c² + d²)i
```

Each final component is rounded once, nearest-even, to the component dtype.
Intermediate products, sums, and denominators are not narrowed or range-checked.
This permits finite results despite intermediate magnitudes that would overflow
or underflow fixed-width arithmetic. Final values outside the finite range raise
`OverflowError`; final inexact nonzero values below the smallest normal raise
`UnderflowError`, including rounding to zero or up to the normal boundary. Exact
subnormals remain valid. Division by a value with both components zero raises
`ZeroDivisionError`, regardless of their signs. No partial result is returned.

These operations use exact stored binary ratios and the floating converter to
produce NumPy components; they do not invoke NumPy complex multiplication or
division. This is a correctness-first scalar API with potentially substantial
intermediate integer sizes, not a vectorized fast path. Results can differ from
backend operations that round intermediates.

Both output components conservatively inherit known rounding from all four input
components, even if a term happens to vanish. Each also records whether its own
final conversion rounded. This tracks history, not an error estimate or a claim
that an exact intermediate recovers the intended source value.

Signed-zero results follow the displayed formulas: each zero product has the XOR
of its factors' signs; subtraction flips the right term's sign. An exact sum of
two negative zero terms yields `-0`; all other exact-zero sums, including
cancellation of nonzero terms, yield `+0`. The division denominator is strictly
positive, so the numerator's zero sign is retained. This is an explicit
formula-based convention; it does not promise identical zero signs to every
backend complex implementation.

```python
a = ComplexValue(1, 2, dtype=c64)
b = ComplexValue(3, -4, dtype=c64)
assert a * b == ComplexValue(11, 2, dtype=c64)
assert (a * b) / b == a
assert a / a == ComplexValue(1, 0, dtype=c64)
```

`.to(dtype)` preserves both components exactly in another supported complex dtype;
`.to(dtype, approximate=True)` explicitly permits rounding. It retains known
rounding history. `.to_real()` returns a `FloatingValue` in the component dtype
only when the stored imaginary component is zero (either sign); otherwise it
raises `ValueError`. The returned wrapper retains any known rounding from either
component. Further real dtype or integer/rational conversion uses that wrapper's
explicit methods. An approximately computed zero imaginary component permits
conversion based on the stored value, not on an inferred intended value.

Signed zeros are preserved through construction and conversion. Negation and
conjugation flip the corresponding signs. Signed-zero components compare equal
and have identical hashes. `bool(z)` is false only when both components are zero.

## Classification, equality, and limits

Classification uses stored values. Every value here is finite and algebraic,
with rational binary components. A nonzero imaginary component excludes `Real`,
`Rational`, and their subsets; it is not an irrational real number. When the
imaginary component is zero, classification follows the real component, including
`Integer`, `Whole`, and `Natural` where applicable.

Equality compares exact stored components across complex wrapper dtypes. A
zero-imaginary value can equal a floating, integer, rational, or fixed-point wrapper or a
supported raw integer. Dtype and rounding history do not affect equality. Hashes
combine exact component hashes using [Python's numeric hash convention](https://docs.python.org/3/library/stdtypes.html#hashing-of-numeric-types),
so supported equal values share hashes. Raw floating, rational, and complex
comparisons require explicit wrapping; unrelated nonnumeric objects use
`NotImplemented` fallback. When our comparison methods run, unsupported numeric
comparisons raise even when the imaginary component is nonzero. A foreign method
can answer first; see [comparison dispatch boundaries](integer.md#backend-boundaries).

NumPy ufunc arithmetic and implicit SymPy coercion are rejected. Equality and
inequality for the documented wrapper and integer operands work in either order,
including NumPy's zero-dimensional integer array transport, without options such
as `out` or `where`.
Powers, magnitude, ordering, mixed-type promotion,
implicit Python complex conversion, and exact symbolic complex storage are
outside this API.
