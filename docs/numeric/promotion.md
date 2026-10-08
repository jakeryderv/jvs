# Explicit Scalar Promotion Contract

**Status: implemented.** `common_dtype` selects a common scalar descriptor under
the rules below. Constructors, operators, casts, and arithmetic helpers retain
their existing explicit conversion and result policies.

## API and guarantee

The API is `common_dtype(left, right) -> ScalarDType`, exported from
`jvs.numeric` and implemented in `arithmetic.py`. Both arguments must be explicit
`NumPyDType`, `ExactDType`, or `FixedDType` descriptors. Values, dtype strings,
Python/NumPy scalar classes, arrays, and arbitrary `DType` subclasses are rejected.
There is no default dtype, approximation flag, or automatic backend fallback.

The result can represent **every finite stored mathematical value** allowed by
either input descriptor exactly. Selection depends on descriptors, never the
particular values later supplied. Comparisons do not need this helper: their
existing exact ordering and equality rules remain independent of promotion.

Calling the helper selects an operand representation; it does not cast values
or execute arithmetic. The intended use is to inspect the returned descriptor,
then supply it as `dtype` to `cast`, `add`, `subtract`, `multiply`, or `divide`.
The ordinary operators continue to require their existing operand contracts.
`//`, `%`, and `divmod` can operate on explicitly cast integer wrappers.

```python
from jvs.numeric import IntegerValue, NumPyDType, add, common_dtype

a = IntegerValue(-100, dtype=NumPyDType("int8"))
b = IntegerValue(200, dtype=NumPyDType("uint8"))
target = common_dtype(a.dtype, b.dtype)
assert target == NumPyDType("int16")
result = add(a, b, dtype=target)
assert result == 100 and result.dtype == target

try:
    common_dtype(NumPyDType("int64"), NumPyDType("uint64"))
except ValueError:
    pass  # Choose exact integer storage explicitly if this range is needed.
else:
    raise AssertionError("No implicit backend fallback is permitted")
```

This guarantee is about operand values, not subsequent results. For example,
the common dtype of two `int8` descriptors is still `int8`; adding two stored
values of 100 still raises `OverflowError`. Integer true division and fixed-point
multiplication/division still return rationals. Floating arithmetic still follows
its existing rounding and underflow rules.

Validation precedes selection, even for equal descriptors. This is a scalar API:
NumPy descriptors, including fixed-point coefficient descriptors, must use native
byte order. Floating formats must pass the same format checks as `FloatingValue`;
complex components must do so as well. Nonfinite capability in a NumPy descriptor
does not admit NaN or infinity into the finite wrappers.

## Family decision table

The table is symmetric; swapping the descriptors must not change the result.
Here, integer includes NumPy integers and `ExactDType.integer()`, rational means
`ExactDType.rational()`, and fixed means `FixedDType`.

| Input families | Common representation |
| --- | --- |
| Integer + integer | Cover both complete integer ranges using the integer rules below |
| Integer + rational | `ExactDType.rational()` |
| Rational + rational | `ExactDType.rational()` |
| Fixed + fixed | Fixed dtype with the common step and checked coefficient range below |
| Fixed + integer | Same fixed-point rule, treating the integer descriptor as step 1 |
| Fixed + rational | `ExactDType.rational()`; subsequent casts discard fixed-point history |
| Floating + floating | A supported NumPy floating format covering both finite value sets |
| Complex + complex | A supported NumPy complex format whose real component covers both input component formats |
| Floating + complex | A supported NumPy complex format whose component covers both the floating format and the complex component format |
| Integer/rational/fixed + floating/complex | Raise `TypeError`; the caller must select an explicit target |

The last restriction applies even to combinations such as `int8` + `float64`
where value-preserving conversion is possible. Selecting approximate arithmetic
is a separate policy decision; this first helper does not infer it from dtype
compatibility. A stored floating value with an integral value receives no special
treatment. Explicit `cast` remains available for all currently supported pairs.

## Integer range selection and backends

For two NumPy integer inputs, compute the union of their complete closed ranges.
Choose the smallest supported width from 8, 16, 32, and 64 bits that covers it.
If the union includes negative values, choose signed storage; otherwise choose
unsigned storage. Do not use a floating representation to extend an integer range.

If no such NumPy dtype exists, raise `ValueError` and direct the caller to choose
an explicit target such as `ExactDType.integer()`. In particular, `int64` plus
`uint64` does not silently become either `float64` or a SymPy integer.

If either integer input is already `ExactDType.integer()`, choose that descriptor.
Its explicitly supplied unbounded representation authorizes the common exact
integer backend. Similarly, a rational input explicitly selects the rational
backend in the family table. These are defined cross-backend combinations,
not failure-triggered fallback. Backend and kind remain inspectable on the result.

| Inputs | Required outcome |
| --- | --- |
| `int8`, `int8` | `int8` |
| `int8`, `uint8` | `int16` |
| `int16`, `uint8` | `int16` |
| `uint16`, `uint32` | `uint32` |
| `int32`, `uint32` | `int64` |
| `int64`, `uint64` | Raise `ValueError`; choose the target explicitly |
| `int64`, exact integer | Exact integer |
| `uint64`, exact rational | Exact rational |

## Floating and complex coverage

Evaluate coverage from actual validated binary-format metadata, not dtype names
or storage width alone. For the supported regular binary formats, a target must
have at least the source's significand precision, a finite maximum no smaller
than the source's, and a smallest subnormal no larger than the source's. Check
these conditions for both sources using exact ratios for the limits. This covers
the finite normal and subnormal values without rounding or narrowing through a
Python float. Floating signed zeros must remain representable too.

Search real candidates in the canonical order `float16`, `float32`, `float64`,
then the host's `longdouble`. Search complex candidates in the order `complex64`,
`complex128`, then `clongdouble`, applying coverage to their component formats.
Resolve aliases, deduplicate equivalent dtype descriptors, and skip candidates
unsupported by the existing scalar format validator. Choose the first covering
candidate. An identical validated input pair returns that descriptor unchanged.
An unsupported input is an error, even if another candidate could approximate it.

There is no assumed `float128` or `complex256` format. Extended precision is
platform-dependent. If no available supported candidate covers both inputs,
raise `ValueError`; do not select an mpmath context or another backend.

| Inputs | Required outcome |
| --- | --- |
| `float16`, `float32` | `float32` |
| `float32`, `float64` | `float64` |
| `complex64`, `complex128` | `complex128` |
| `float16`, `complex64` | `complex64` |
| `float64`, `complex64` | `complex128` |
| `float64`, supported `longdouble` | First covering format under the metadata rules; often `longdouble`, possibly an equivalent standard format |
| Supported `longdouble`, `complex128` | First complex format whose component covers both; otherwise raise `ValueError` |

Real-to-complex casts add an exact positive imaginary zero. Existing casts
preserve the real component's zero sign and known rounding history. This helper
does not change those rules or promise additional working precision for results.

## Fixed-point step and coefficient range

Write each positive step in reduced form as `s1 = n1/d1` and `s2 = n2/d2`.
Choose the largest common lattice step:

```text
s = gcd(n1, n2) / lcm(d1, d2)
k1 = s1 / s       # positive integer
k2 = s2 / s       # positive integer
```

An integer descriptor participates as a step-1 lattice with its full integer
range. An input coefficient `c` becomes `c*ki` exactly; no rounding is needed.
For bounded input ranges `[lo1, hi1]` and `[lo2, hi2]`, the required target
coefficient interval is:

```text
[min(lo1*k1, lo2*k2), max(hi1*k1, hi2*k2)]
```

Choose coefficient storage using the integer range rules above, then return
`FixedDType(storage, step=s)`. Scaling the full bounds is required: merely
promoting the original coefficient dtypes can lose representable input values.
The result stays fixed-point even when its step is 1 or an integer larger than 1.

If either source has an explicit unbounded integer representation (including a
fixed dtype's coefficient dtype), use exact unbounded integer coefficients.
Otherwise, failure to fit the rescaled ranges in a supported NumPy integer raises
`ValueError`. Do not silently switch to unbounded coefficients or rationals.
All step and range calculations use exact unbounded integer/rational arithmetic.

In these examples, `fixed(int8, 1/10)` abbreviates a `FixedDType` with `int8`
coefficient storage and step `Fraction(1, 10)`; it is not executable syntax.

| Inputs | Required outcome |
| --- | --- |
| `fixed(int8, 1/10)`, itself | Same descriptor |
| `fixed(int8, 1/10)`, `fixed(int8, 1/100)` | `fixed(int16, 1/100)`; required coefficients span -1280 through 1270 |
| `fixed(int8, 1/6)`, `fixed(int8, 1/10)` | `fixed(int16, 1/30)`; coefficient multipliers are 5 and 3 |
| `fixed(int8, 2)`, `int8` | `fixed(int16, 1)`; fixed input coefficients double |
| `fixed(uint8, 1/10)`, `uint8` | `fixed(uint16, 1/10)`; integer inputs require coefficients up to 2550 |
| `fixed(int64, 1/10)`, `fixed(int64, 1/100)` | Raise `ValueError`; scaled bounds exceed all supported NumPy integer ranges |
| `fixed(int64, 1/10)`, exact integer | `fixed(exact integer, 1/10)` |
| Any fixed descriptor, exact rational | Exact rational |

Fixed-to-fixed casts preserve known rounding history. When the family table
selects a rational target, the existing cast deliberately extracts stored values
and discards fixed-point history. The helper selects a descriptor without access
to history; callers must inspect that target before casting if provenance matters.
Here, “lossless” refers to mathematical value, not retention of all representation
metadata or recovery of intended inputs.

```python
from fractions import Fraction
from jvs.numeric import FixedDType, FixedValue, NumPyDType, cast, common_dtype

tenths = FixedDType(NumPyDType("int8"), step=Fraction(1, 10))
hundredths = FixedDType(NumPyDType("int8"), step=Fraction(1, 100))
target = common_dtype(tenths, hundredths)
assert target == FixedDType(NumPyDType("int16"), step=Fraction(1, 100))
value = FixedValue.from_coefficient(127, dtype=tenths)
rescaled = cast(value, target)
assert isinstance(rescaled, FixedValue)
assert rescaled.coefficient == 1270 and rescaled == value
assert not rescaled.rounded
```

## Errors and algebraic properties

- `TypeError`: non-descriptor input, unsupported descriptor class, or a prohibited
  exact/approximate family combination.
- `ValueError`: non-native scalar byte order, or no allowed common representation.
  Distinguish those causes in the message; name the input representations and
  suggest an explicit target where applicable, using bounded diagnostics.
- `NotImplementedError`: an input floating/component format fails the existing
  supported-format contract. An equal input pair does not bypass validation.

Range-selection failure is `ValueError`, not an arithmetic `OverflowError`: the
helper is choosing a descriptor and has not evaluated any actual values.
It never returns `None`, warns and continues, or returns a lossy best effort.

For valid supported inputs, selection must be commutative and idempotent by dtype
equality; object identity is not guaranteed. Every successful selection must
satisfy complete operand-value coverage. There is no associativity guarantee for
repeated pairwise calls: an intermediate descriptor may include values outside
the original input domains, and the next call must cover that wider domain too.
For example, combining `int8` and `uint8` first produces `int16`; adding
`fixed(int8, 1/100)` to that descriptor requires fixed `int32` coefficients.
Combining `uint8` with the fixed descriptor first, then `int8`, needs only fixed
`int16` coefficients. Both preserve all source values. This first API is binary;
a future multi-input API must define selection over all original descriptors.

## Validation coverage

The regression tests cover every family pair in both orders, equal inputs,
invalid descriptors, and the tables above as concrete fixtures. Coverage includes
full integer bounds, rational step divisibility and scaled coefficient bounds,
floating precision/range/subnormal coverage, native byte order, unsupported
formats, and the host's extended types and aliases.

Boundary casts use `approximate=False` and check value equality, zero signs,
and the documented history behavior. Tests also cover unchanged operator
strictness, a successful promotion followed by arithmetic overflow, and the
documented grouping difference. Unsupported-format and candidate-exhaustion
paths are simulated independently of the host's available formats.

Implicit operator promotion, array promotion, new backends, and configurable
fallback policies remain outside this API.
