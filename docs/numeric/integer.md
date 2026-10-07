# Integer Values

`IntegerValue` is an immutable integer wrapper with an explicit
[representation descriptor](dtype.md). It holds a NumPy integer scalar for a
fixed-width dtype or a SymPy integer for an unbounded exact dtype. It is independent
of `jvs.core` and does not inherit the mathematical `Integer` set.

## Construction and Membership

```python
from jvs.numeric import ExactDType, Integer, IntegerValue, Natural, NumPyDType, Whole

int16 = NumPyDType("int16")
x = IntegerValue(12, dtype=int16)
zero = IntegerValue(0, dtype=int16)

assert x in Integer
assert x in Natural
assert zero in Whole
assert zero not in Natural

large = IntegerValue(10**1000, dtype=ExactDType.integer())
assert int(large) == 10**1000
```

The keyword-only `dtype` argument is required and must be an integer `NumPyDType`
or `ExactDType`. Python integers, NumPy integer scalars, and SymPy integers are
accepted. Booleans, floats, rational/decimal representations, strings, arrays,
timedeltas, and symbolic variables are rejected, even when they appear to contain
an integer. Objects merely implementing `__int__` are not accepted implicitly.

NumPy range checks happen before conversion, using exact integers. Unrepresentable
values raise `OverflowError`; no wraparound or warning-only fallback is permitted.
NumPy scalar payloads require native byte order. Non-native dtype descriptors are
valid for the descriptor API but rejected here because a scalar cannot preserve
that storage byte order.

`value` exposes the immutable backend scalar, and `dtype` exposes its descriptor.
`int(x)` and the index protocol return the exact Python integer. `bool(x)` is false
only for zero. Membership uses the wrapped value, so sign-dependent sets such as
`Whole` and `Natural` are classified individually.

`repr(x)` displays integers of at most 256 bits in full decimal form. Larger
values use an explicit abbreviation with sign, bit length, and leading/trailing
32-bit hexadecimal chunks. This bounded diagnostic representation is not a
serialization format; the stored value is unchanged. It avoids Python's decimal
digit limit without changing global settings.

## Checked Arithmetic and Conversion

The shared [`cast(value, dtype)` API](casting.md) also converts integer wrappers
to any supported scalar family. Existing methods below remain available.
The [explicit arithmetic functions](arithmetic.md) cast both operands into a
chosen dtype before using these checked operations.

```python
y = IntegerValue(3, dtype=int16)
assert int(x + y) == 15
assert int(x - y) == 9
assert int(x * y) == 36
assert int(-x) == -12

exact = x.to(ExactDType.integer())
assert exact == x
assert exact.to(int16) == x
```

Addition, subtraction, and multiplication require two `IntegerValue` operands
whose dtype descriptors are equal. Results retain that dtype. Unary negation is
checked too: negating the minimum signed value or a nonzero unsigned value raises
`OverflowError`. Unary plus returns the same immutable value.

Computations use exact Python integer intermediates before constructing the
selected backend result. This avoids overflow during the calculation itself.
SymPy-backed results remain unbounded exact integers; NumPy-backed results must
fit their declared range.

True division (`/`) also requires matching integer dtypes, but always returns a
[RationalValue](rational.md) with `ExactDType.rational()`, including integral
results. It constructs an exact ratio directly from the integer operands without
floating conversion or fixed-width division. Division by zero raises
`ZeroDivisionError`. The result is not limited to the operands' integer range:
for example, `int8` values `-128 / -1` yield the exact rational `128/1`.
Converting that result back to `int8` with `to_integer(dtype)` raises
`OverflowError`.

```python
quotient = y / x  # 3 / 12
assert quotient.numerator == 1
assert quotient.denominator == 4
assert quotient.dtype == ExactDType.rational()
assert (x / y).to_integer(int16) == 4
```

Mixed dtypes and raw scalar arithmetic raise `TypeError`. Convert explicitly with
`.to(dtype)` before combining representations. Conversion applies the same input
and target checks as construction; it cannot turn an integer wrapper into a
floating or rational representation. Floor division (`//`), modulo, powers,
ordering, and mixed-type promotion are outside this initial API. NumPy division
ufuncs remain unsupported; use `/` for the checked exact operation.

## Equality and Hashing

Equality compares the exact integer value across wrapper dtypes and supported
Python, NumPy, and SymPy integer scalars. Equal integers have matching hashes, so
they behave consistently as set members and dictionary keys. Dtype identity is
checked separately with `x.dtype == y.dtype`.

Equality with a [RationalValue](rational.md) also compares exact mathematical
values: integral ratios compare equal to matching integers and share their hash;
nonintegral ratios compare unequal. Mixed arithmetic still requires explicit
conversion. To construct a rational from an integer wrapper, pass `int(x)` or
`x.value` as its numerator; use the rational's `to_integer(dtype)` to convert back.

Equality with a [FloatingValue](floating.md) likewise compares its stored binary
value exactly, with matching hashes when equal. Construct floating wrappers from
integer wrappers using `FloatingValue(x, dtype=...)`, or explicitly permit rounding
with `FloatingValue.approx`. Use `to_integer(dtype)` for checked conversion back.

[ComplexValue](complex.md) follows the same exact comparison rule when its stored
imaginary component is zero; non-real complex wrappers compare unequal. Pass an
integer wrapper as a component to construct a complex wrapper, and use
`to_real().to_integer(dtype)` for checked conversion back.

[FixedValue](fixed.md) compares its coefficient times step to the integer, with
matching hashes when equal. Use `cast` or its constructor for conversion into
fixed-point; `to_integer(dtype)` checks integrality and range on conversion back.

When the wrapper's comparison methods receive other numeric operands, including
booleans and raw integral-valued floats, they require explicit conversion and
raise rather than approximating or claiming non-equality. Comparisons with
unrelated nonnumeric objects follow Python's
`NotImplemented` fallback. Mathematical membership and accepted equality operands
are separate contracts.

NumPy sometimes transports scalar equality operands as zero-dimensional arrays.
Equality therefore also accepts zero-dimensional integer arrays through ufunc
dispatch, without rounding. Arrays with axes and arrays of other kinds remain
unsupported. This equality exception does not allow array construction inputs,
array membership, or array arithmetic.

## Backend Boundaries

NumPy ufunc arithmetic cannot bypass the wrapper's checks; use the operators above.
`np.equal` and `np.not_equal` support the documented wrapper and raw integer
comparisons in either operand order, without `out`, `where`, or other options.
Unresolved equality defers to the other operand through NumPy's dispatch protocol;
if all implementations defer, NumPy raises `TypeError`. Arithmetic and unsupported
array operations still raise directly. Implicit SymPy coercion is rejected; foreign
SymPy operations may raise `SympifyError` when they attempt that coercion.

A foreign object's comparison method can return before any wrapper method is
called. We cannot guarantee rejection of unsupported operands in that situation:
for example, `sympy.nan == x` returns `False`, while `x == sympy.nan` raises.
The strict rejection policy applies when our comparison method runs. Exact,
symmetric equality is guaranteed across the supported wrapper families.

Explicitly extracting `.value` or `int(x)` transfers control to the consumer.
Subsequent operations on those extracted values follow that backend's rules,
including NumPy overflow behavior, rather than the `IntegerValue` contract.
