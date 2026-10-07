# Exact ordering of stored real values

`IntegerValue`, `RationalValue`, `FloatingValue`, and `FixedValue` support `<`,
`<=`, `>`, and `>=` across representations. Comparisons use exact ratios of
stored mathematical values. They do not cast operands, select a common dtype,
round, or mutate values or their rounding history.

This supports sorting, `min`, `max`, and range checks. Large integers are not
narrowed through a float, floating values retain their original binary precision,
and fixed-point comparisons use coefficient times step. Positive and negative
floating zero have the same position in the ordering. For these real wrappers,
`a <= b and b <= a` agrees with `a == b`, independently of dtype and history.
Ordering an approximation compares its stored value, not an intended exact
input or an uncertainty interval.

```python
from fractions import Fraction
from jvs.numeric import (
    ExactDType, FixedDType, FixedValue, FloatingValue, IntegerValue,
    NumPyDType, RationalValue,
)

integer = IntegerValue(2**53 + 1, dtype=ExactDType.integer())
floating = FloatingValue(2**53, dtype=NumPyDType("float64"))
assert floating < integer  # No loss from promoting the integer to float64.

tenth = FixedValue(
    Fraction(1, 10),
    dtype=FixedDType(NumPyDType("int16"), step=Fraction(1, 100)),
)
binary_tenth = FloatingValue(0.1, dtype=NumPyDType("float64"))
assert tenth < binary_tenth
half = RationalValue(1, 2, dtype=ExactDType.rational())
assert sorted([half, binary_tenth, tenth]) == [tenth, binary_tenth, half]
assert min(half, tenth) is tenth
assert 0 <= tenth < 1
```

## Operand and backend boundaries

Raw Python, NumPy, and SymPy integer scalars are accepted in either order,
consistent with equality. Booleans and other raw numeric representations require
explicit wrapping, including integral-valued floats, `Fraction`, nonintegral
SymPy rationals, decimals, and raw complex values. Unsupported numeric operands
raise `TypeError` when wrapper comparison methods run. Unrelated objects return
`NotImplemented`, allowing reflected dispatch; if neither operand implements
the comparison, Python raises `TypeError`.

`ComplexValue` is deliberately unordered even with zero imaginary part. Extract
`.to_real()` explicitly before ordering; this fails for a nonzero imaginary
component. Equality between a real wrapper and a zero-imaginary complex wrapper
does not authorize ordering those wrappers.

NumPy `less`, `less_equal`, `greater`, and `greater_equal` use the same exact
scalar rules, including correct direction when the wrapper is on the right.
Like equality, they accept NumPy's zero-dimensional integer array transport.
Other arrays, reductions, `out`, `where`, and other options raise `TypeError`;
there is no broadcasting or object-array fallback in wrapper dispatch.
Arithmetic ufuncs remain unsupported.

Foreign comparison methods may produce their own result before dispatching to
the wrapper. Strict rejection applies when our methods run; it cannot override
foreign implementations. These comparisons concern concrete finite values;
symbolic uncertainty and set relations remain the responsibility of `number.py`.
