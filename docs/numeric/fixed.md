# Fixed-point values

`FixedDType(coefficient_dtype, *, step)` describes the lattice
`value = coefficient × step`. `step` is a positive exact rational, supplied as a
Python/NumPy integer, `Fraction`, or SymPy rational and normalized to `Fraction`
metadata. Floats, booleans, strings, and symbolic expressions are rejected as
steps. Decimal steps such as `Fraction(1, 100)` and binary steps such as
`Fraction(1, 256)` use the same model; no implicit decimal scale is inferred.

The coefficient dtype is an existing NumPy integer descriptor or
`ExactDType.integer()`. Backend and storage width describe the **coefficient**;
the rational step is additional descriptor metadata. Unbounded coefficients
remove range limits but still represent only integer multiples of the step.
Non-native NumPy byte order can be described but cannot back a scalar value.

## Construction and conversion

`FixedValue(value, *, dtype)` accepts the same real scalar inputs as
`FloatingValue`, including fixed-point wrappers. Inputs are mathematical values,
not coefficients. `FixedValue.from_coefficient(coefficient, *, dtype)` accepts
raw Python/NumPy/SymPy integers instead. Both check the coefficient range before
storage; booleans are excluded. Values and descriptors are immutable.

- Exact construction and `.to(dtype)` require an integral `value / step` and
  raise `PrecisionLossError` otherwise.
- `.approx(value, dtype=...)` and `.to(dtype, approximate=True)` explicitly
  choose the nearest coefficient, with ties going to the even coefficient.
  There are no implicit rounding modes or truncation.
- Fixed-width bounds apply to the input value **before rounding**, and to
  arithmetic results before storage. Out-of-range inputs raise `OverflowError`
  even if rounding would bring them back into range. No saturation or wrapping.
- Fixed-point has no floating normal/subnormal distinction. Explicit rounding
  may produce zero; exact construction of an off-lattice value fails.
- `rounded` records known rounding within the wrapper API and propagates through
  rescaling, floating/complex conversion, addition, subtraction, and negation.
  `False` makes no claim about external accuracy. Routine requested rounding
  emits no warning.
- A float input means its stored binary value. For example, a Python `0.1`
  normally needs explicit approximation onto a step of `1/10`; it is never
  silently interpreted as an intended exact decimal tenth.
- Fixed-point has one zero. Conversion into this representation drops signed
  zero; conversion back to floating produces positive zero.

`.coefficient` is a checked `IntegerValue`; `.coefficient.value` exposes its
NumPy or SymPy backend scalar. `.value` reconstructs the mathematical value as
an exact SymPy rational, independent of coefficient storage.
`.to_rational()` extracts that value into `RationalValue`, discarding history.
`.to_integer(dtype)` additionally requires integrality and checks target range.
There are no implicit `int`, `float`, index, or complex conversions.

```python
from fractions import Fraction
from jvs.numeric import FixedDType, FixedValue, NumPyDType, PrecisionLossError

hundredths = FixedDType(NumPyDType("int16"), step=Fraction(1, 100))
value = FixedValue(Fraction(5, 4), dtype=hundredths)
assert int(value.coefficient) == 125
assert value == FixedValue.from_coefficient(125, dtype=hundredths)

tenths = FixedDType(NumPyDType("int16"), step=Fraction(1, 10))
try:
    value.to(tenths)
except PrecisionLossError:
    pass
else:
    raise AssertionError("Rescaling must not silently round")
rounded = value.to(tenths, approximate=True)
assert int(rounded.coefficient) == 12  # 12.5 ties to even coefficient 12
assert rounded.rounded
```

## Arithmetic and interoperability

All binary operators require matching fixed-point dtypes, including coefficient
storage and step. `+` and `-` retain that dtype, check coefficient bounds, and
propagate rounding history. Unary `-` checks bounds too; unary `+` returns self.
`*` and `/` return exact `RationalValue` results calculated from the stored
mathematical values, even if the result happens to fit the operand lattice.
They use unbounded exact intermediates. Division by zero raises
`ZeroDivisionError`. Rational results discard rounding history; exactness of
the stored result does not prove that its inputs were never rounded.

`cast` supports fixed-point sources and targets. The explicit arithmetic helpers
cast operands to the requested dtype before operating; fixed-point targets
permit `approximate=True` for operand conversion only. Requantizing a product
or quotient is a separate explicit cast.

```python
from fractions import Fraction
from jvs.numeric import (
    FixedDType, FixedValue, NumPyDType, RationalValue, cast, multiply,
)

tenths = FixedDType(NumPyDType("int16"), step=Fraction(1, 10))
a = FixedValue(Fraction(3, 10), dtype=tenths)
product = multiply(a, a, dtype=tenths)
assert isinstance(product, RationalValue)
assert product.numerator == 9 and product.denominator == 100
result = cast(product, tenths, approximate=True)
assert isinstance(result, FixedValue)
assert int(result.coefficient) == 1 and result.rounded
```

Classification, equality, and hashing use the represented rational value, not
the coefficient. Equality supports all numeric wrappers and raw supported
integer scalars. Other raw numeric comparisons require explicit wrapping.
Foreign comparison methods can return before wrapper dispatch; those methods
are outside this guarantee. NumPy equality/inequality dispatch is symmetric;
arithmetic ufuncs, arrays, and options cannot bypass wrapper checks. SymPy
conversion requires explicit extraction. Diagnostic representations abbreviate
huge coefficients and step components without changing Python's digit limit.

`<`, `<=`, `>`, and `>=` implement [exact ordering](ordering.md) by coefficient
times step against all real-valued wrappers and supported raw integer scalars.
Different steps do not need rescaling for comparison; history remains unchanged.
NumPy ordering ufuncs follow the same rules. Complex wrappers must be explicitly
converted with `.to_real()` before ordering.

Powers, automatic promotion, user-selected rounding modes, decimal
text parsing, and array/storage APIs remain deferred.
