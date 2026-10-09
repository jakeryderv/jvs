# Explicit Numeric Arithmetic

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

[`common_dtype(left, right)`](promotion.md) chooses an operand dtype from two
explicit descriptors, covering both complete finite input value sets. It is
exported from `jvs.numeric`. Inspect its result and pass it as the explicit
`dtype` argument to these functions. Selecting a common dtype does not guarantee
that an operation's result fits. Exact/approximate family mixtures and exhausted
NumPy integer ranges require the caller to choose an explicit target.

## Result types

For scalar operands, `dtype` selects the **operand representation**, not an
unconditional result dtype.

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

## Elementwise buffer arithmetic

`add`, `subtract`, and `multiply` also accept two [`NumericBuffer`](storage.md)
operands. They require identical shapes and an explicit native `NumPyDType`
target, and return a new buffer with that shape and dtype. Input dtypes may
differ: each pair of elements is cast to the target before its checked scalar
operation runs. The result owns independent, read-only data and rounding
metadata; neither input changes, even when the same buffer supplies both operands.

```python
import numpy as np
from jvs.numeric import NumericBuffer

left = NumericBuffer(np.array([[120, 3]], dtype=np.int8), dtype=NumPyDType("int8"))
right = NumericBuffer(np.array([[20, 4]], dtype=np.int16), dtype=NumPyDType("int16"))
total = add(left, right, dtype=NumPyDType("int16"))
assert total.shape == (1, 2) and total[0, 0] == 140 and total[0, 1] == 7
assert multiply(left, right, dtype=NumPyDType("int16"))[0, 1] == 12

try:
    add(left, right, dtype=NumPyDType("int8"))
except OverflowError as error:
    assert any("(0, 0)" in note for note in error.__notes__)
else:
    raise AssertionError("Integer overflow must fail")
assert left[0, 0] == 120 and right[0, 0] == 20
```

The scalar rules above also govern buffer operations:

- `approximate=False` requires exact operand conversion, while subsequent
  floating/complex arithmetic permits normal rounding. `approximate=True` permits
  operand rounding for floating/complex targets and is rejected for integers.
- Operands must individually fit before the operation. Integer results are
  range-checked before storage; they never wrap or automatically widen.
- Rounding history follows each cast and scalar operation. Floating/complex
  results retain known history and add arithmetic rounding; integer casts discard
  history. Complex component flags follow the scalar dependency rules.
- Overflow, inexact underflow, nonintegral integer conversion, precision loss,
  and nonzero imaginary parts entering real arithmetic retain their scalar
  exceptions. Failures include the element coordinate as an exception note.
  Traversal is in C order, and the first failed pair stops the operation;
  no partial buffer is returned.
- Zero-dimensional buffers operate with other zero-dimensional buffers. Empty
  buffers preserve shape and still validate operand formats, target format, and
  approximation policy. Shape mismatches raise `ValueError`, including shapes
  that NumPy could broadcast.

This is correctness-first scalar evaluation into owned output storage, not a
vectorized kernel. No intermediate converted operand buffers are allocated.
Mixed scalar/buffer arguments, raw ndarrays, exact/fixed-point buffer targets,
implicit promotion, broadcasting, reductions, operators, and ufuncs are outside
this API. Buffer division raises `TypeError` for all targets; exact rational
result storage must be settled before extending division consistently.
