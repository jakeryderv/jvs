# Explicit Scalar Arithmetic

`add`, `subtract`, `multiply`, and `divide` combine existing numeric wrappers in
an explicitly selected operand representation:

```python
from jvs.numeric import (
    ExactDType, IntegerValue, NumPyDType, RationalValue,
    add, divide, multiply, subtract,
)

small = IntegerValue(120, dtype=NumPyDType("int8"))
other = IntegerValue(20, dtype=ExactDType.integer())
result = add(small, other, dtype=NumPyDType("int16"))
assert result == 140
assert result.dtype == NumPyDType("int16")

half = RationalValue(1, 2, dtype=ExactDType.rational())
assert multiply(small, half, dtype=ExactDType.rational()) == 60
assert subtract(half, small, dtype=ExactDType.rational()) == RationalValue(
    -239, 2, dtype=ExactDType.rational()
)

quotient = divide(small, other, dtype=NumPyDType("int16"))
assert quotient == 6
assert isinstance(quotient, RationalValue)
assert quotient.dtype == ExactDType.rational()
```

Each function has the signature `(a, b, *, dtype, approximate=False)`. Both
operands are converted with [`cast`](casting.md), then the corresponding checked
wrapper operator runs. Raw scalars and dtype strings are rejected. Existing
operators remain strict about matching dtypes; these functions do not introduce
automatic promotion.

## Result types

`dtype` selects the **operand representation**, not an unconditional result dtype.

| Requested operand dtype | `add`, `subtract` | `multiply` | `divide` |
| --- | --- | --- | --- |
| NumPy integer | `IntegerValue`, same dtype, checked range | Same as addition | `RationalValue`, `ExactDType.rational()` |
| Exact integer | `IntegerValue`, `ExactDType.integer()` | Same as addition | `RationalValue`, `ExactDType.rational()` |
| Exact rational | `RationalValue`, same dtype | Same as addition | Same as addition |
| NumPy floating | `FloatingValue`, same dtype | Same as addition | Same as addition |
| NumPy complex | `ComplexValue`, same dtype | Same as addition | Same as addition |
| Fixed-point | `FixedValue`, same dtype, checked coefficient range | `RationalValue`, `ExactDType.rational()` | `RationalValue`, `ExactDType.rational()` |

The selected wrapper's existing rules govern the operation. Integer arithmetic
uses exact intermediates and checks fixed-width results. Rational arithmetic is
exact. Floating arithmetic accepts normal nearest-even rounding; complex
multiplication/division use exact intermediates and round each final component
once. Overflow, inexact underflow, and zero division retain their established
exceptions. Results are never automatically widened or recast afterward.

Fixed-point addition/subtraction check the final coefficient range. Fixed-point
multiplication/division use exact rational intermediates and results, without
coefficient range limits on those results. Casting a result back onto a fixed
lattice is a separate request. Rational results discard known rounding history;
their exactness is relative to the stored operands, not their original intent.

## Conversion and approximation

`approximate=False` requires **exact operand conversion**. It does not prohibit
rounding during subsequent floating or complex arithmetic. To request exact
arithmetic on stored real values, explicitly select an exact integer or rational
operand dtype, subject to the corresponding conversion restrictions.

`approximate=True` permits nearest-even operand-conversion rounding only for
floating/complex/fixed-point dtypes. It is rejected for integer/rational dtypes.
It does not permit overflow, inexact floating underflow, truncation, or discarding
a nonzero imaginary part. Fixed-point quantization may explicitly round to zero.
Known rounding history follows the casts and the selected operators.

Both operands must individually fit the requested dtype before arithmetic. A
small final result does not authorize out-of-range operand conversion. For
example, two unbounded integers equal to 256 cannot be subtracted in `int8`, even
though their exact difference is zero. Select a representation that holds the
operands first, then explicitly cast the result if needed.

Complex inputs may enter real arithmetic only when their stored imaginary parts
are zero. Casting floating/complex values to exact types extracts their stored
values and discards signed-zero and rounding-history metadata, as documented by
`cast`. It cannot recover intended source values or prior precision.

Failures propagate from the existing conversion/arithmetic APIs. Operands are
immutable and no partial result is returned. No backend-default promotion,
permissive warning fallback, or alternate computation dtype is selected.
