# Explicit Scalar Casting

`cast(value, dtype, *, approximate=False)` converts an existing numeric wrapper
into the representation selected by an explicit `NumPyDType`, `ExactDType`, or
`FixedDType`.
It reuses the checked constructors and conversion methods; it does not introduce
automatic arithmetic promotion or new rounding algorithms.

For a conversion followed by an operation, [explicit arithmetic](arithmetic.md)
provides `add`, `subtract`, `multiply`, and `divide` with a required operand dtype.
They reuse `cast` and the checked wrapper operators.

```python
from jvs.numeric import (
    ComplexValue, ExactDType, FloatingValue, IntegerValue,
    NumPyDType, RationalValue, cast,
)

integer = IntegerValue(12, dtype=NumPyDType("int16"))
ratio = cast(integer, ExactDType.rational())
assert isinstance(ratio, RationalValue)
assert ratio == integer

third = RationalValue(1, 3, dtype=ExactDType.rational())
approximation = cast(third, NumPyDType("float32"), approximate=True)
assert isinstance(approximation, FloatingValue)
assert approximation.rounded

complex_value = cast(approximation, NumPyDType("complex128"))
assert isinstance(complex_value, ComplexValue)
assert complex_value.real.rounded
assert complex_value.imag == 0
```

## Conversion matrix

All successful conversions return a wrapper whose dtype equals the requested
one. Integer targets may be fixed-width NumPy or unbounded SymPy; rational targets
are `ExactDType.rational()`. Floating and complex targets are supported native
NumPy scalar formats. Fixed-point targets combine an integer coefficient dtype
and an exact rational step. The existing scalar methods remain available.

| Source wrapper | Integer target | Rational target | Floating target | Complex target |
| --- | --- | --- | --- | --- |
| `IntegerValue` | Exact value; target range checked | Exact ratio with denominator one | Exact by default; optional rounding | Convert real component; imaginary `+0` |
| `RationalValue` | Reduced denominator must be one; range checked | Preserve exact ratio | Exact by default; optional rounding | Convert real component; imaginary `+0` |
| `FloatingValue` | Stored value must be integral; range checked | Exact ratio of stored binary value | Exact by default; optional rounding | Convert stored real component; imaginary `+0` |
| `ComplexValue` | Imaginary component must be zero, then floating-to-integer rules | Imaginary component must be zero, then stored real binary ratio | Imaginary component must be zero, then floating conversion | Convert both components; exact by default, optional rounding |
| `FixedValue` | Coefficient times step must be integral; range checked | Exact coefficient times step | Exact by default; optional rounding; retain history | Convert real value; imaginary `+0`; retain history |

Every source wrapper can be cast to `FixedDType`: its stored real value must lie
on the target lattice, or `approximate=True` explicitly rounds to the nearest
coefficient (ties to even). A complex source must first have a zero imaginary
component. The input must be in the target range before rounding. Fixed-to-fixed
casts change storage and/or step and retain known rounding history. See
[fixed-point values](fixed.md).

An approximate source does not prevent an exact cast: exactness here means
preserving its **stored value**, not recovering an intended source value.

## Validation and failure policy

Only the five wrapper classes above are accepted. Raw Python/NumPy/SymPy scalars,
`Fraction`, arrays, and duck-typed objects raise `TypeError`; construct the desired
source wrapper explicitly first. Raw dtype strings/classes are rejected too.
The `approximate` argument must be an actual Python `bool`.

- Exact floating/complex/fixed-point conversion that needs rounding raises `PrecisionLossError`.
- `approximate=True` permits nearest-even rounding for floating, complex, and
  fixed-point targets. It does not permit overflow, inexact floating underflow,
  or discarding a nonzero imaginary component. Fixed-point has no normal/subnormal
  distinction and may explicitly round to zero. It emits no routine rounding warning.
- For integer/rational targets, `approximate=True` raises `ValueError`, even if the
  particular value would fit exactly. These targets have no approximate mode;
  the flag does not mean truncate, round to an integer, or limit a denominator.
- Nonintegral integer conversion and non-real conversion to a real representation
  raise `ValueError`.
- Out-of-range conversion raises `OverflowError`. Tiny inexact floating results
  raise `UnderflowError`, while exact subnormals remain valid.
- Native-byte-order and supported-format requirements are those of the target
  wrapper. Invalid combinations fail explicitly; there is no fallback backend.

Object identity is not part of the cast contract. Inputs remain immutable.

## Signed zero and rounding history

Floating-to-floating and complex-to-complex conversion preserve zero signs and
known component rounding history. Real-to-complex conversion retains the real
component's sign/history and adds an exact positive imaginary zero. Complex-to-
floating conversion drops the zero imaginary sign, preserves the real zero sign,
and conservatively retains known rounding from either input component.

Integer and rational wrappers have no signed zero or rounding-history field.
Explicit casts into those families discard both pieces of representation
metadata. A cast from an exact wrapper back to floating therefore starts with no
known prior rounding. This is intentional extraction of a stored value, not proof
that the earlier computation was exact. Keep the original wrapper when provenance
matters; no automatic warning or hidden provenance record is added.

Fixed-point wrappers retain rounding history through fixed, floating, and complex
conversions. They have no signed zero: conversion into fixed-point drops zero
signs, and conversion from a fixed-point zero produces positive zero. Integer or
rational extraction from fixed-point discards rounding history as above.

In particular, casting a floating approximation of `1/3` to a rational produces
its stored binary ratio, not `1/3`. Casting it back preserves those bits, but not
the discarded rounding history.

General promotion, permissive warning fallbacks, truncation, user-selected
rounding modes, exact symbolic complex targets, and array casts remain deferred.
