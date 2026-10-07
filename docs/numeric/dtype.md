# Numeric Representation Descriptors

`jvs.numeric.dtype` describes representations independently of mathematical sets,
concrete values, and storage ownership. It does not construct or cast numeric
values, infer their mathematical membership, or choose arithmetic promotion.

## Public API

```python
import numpy as np

from jvs.numeric import ExactDType, NumPyDType

integer = NumPyDType("int32")
assert integer == NumPyDType(np.int32)
assert integer == NumPyDType(np.dtype("int32"))
assert integer.storage_bits == 32
assert int(integer.integer_info.min) == -(2**31)
assert int(integer.integer_info.max) == 2**31 - 1

floating = NumPyDType("float64")
assert floating.storage_bits == 64
assert floating.floating_info.significand_bits == 53

complex_type = NumPyDType("complex128")
assert complex_type.storage_bits == 128
assert complex_type.component_dtype == floating

exact_integer = ExactDType.integer()
exact_rational = ExactDType.rational()
assert exact_integer.storage_bits is None
assert not exact_rational.is_fixed_width
```

`DType` is an abstract interface. Its current concrete descriptors are immutable
and hashable, with these shared properties:

| Property | Meaning |
| --- | --- |
| `backend` | `Backend.NUMPY` or `Backend.SYMPY` |
| `kind` | `NumericKind.INTEGER`, `FLOATING`, `COMPLEX`, `RATIONAL`, or `FIXED` |
| `name` | Representation name; diagnostic abbreviation for huge fixed-point steps |
| `storage_bits` | Fixed storage width including padding, or `None` for variable storage |
| `is_fixed_width` | Whether the descriptor specifies a fixed storage width |
| `supports_nonfinite` | Representation capability, independent of domain permission |

`NumericKind` describes representation. For example, a `FLOATING` scalar may store
an integer value, and a `COMPLEX` scalar may store a real value. Membership queries
belong to `number.py` and operate on values.

## NumPy Descriptors

`NumPyDType` accepts dtype strings, concrete NumPy scalar classes, and `np.dtype`
objects. It supports signed integers, unsigned integers, floating-point types,
and complex floating-point types. It preserves the supplied dtype and exposes it
as `numpy_dtype`.

There is no implicit dtype default. `None`, Python scalar classes such as `int`,
abstract NumPy scalar classes, scalar instances, and arrays are rejected. NumPy
aliases in explicit dtype strings resolve on the current platform; use width
names such as `int32` when a particular width is required.

Boolean, object, string, datetime, timedelta, structured, and subarray dtypes are
unsupported. Dtypes carrying metadata are rejected rather than silently dropping
that metadata or treating it as part of numeric equality.

The `byteorder` property resolves NumPy's native marker into `ByteOrder.LITTLE` or
`ByteOrder.BIG`, with `NOT_APPLICABLE` for single-byte types. Byte order participates
in descriptor equality. The `name` alone does not encode it; use `numpy_dtype` to
retain the full representation.

### Integer Metadata

`integer_info` returns an immutable `IntegerInfo` containing `signed`, `bits`,
`min`, and `max`. Bounds are exact NumPy integer scalars, including the full
`uint64` maximum; they are never routed through floating point.

### Floating Metadata

`floating_info` returns an immutable `FloatingInfo` for real floating types:

- `significand_bits`: reported normal-value binary precision, including the leading bit;
- `exponent_bits`: NumPy-reported exponent width;
- `min_exponent` and `max_exponent`: NumPy's exponent bounds, with an exclusive upper bound;
- `min` and `max`: finite representable bounds;
- `eps`: spacing above one;
- `smallest_normal` and `smallest_subnormal`: the distinct lower positive limits.

Floating metadata stays in the backend scalar format, preserving extended range
and precision. Metadata scalars use native scalar byte order; the descriptor
retains the requested storage byte order separately.

Storage bits, significand precision, and numerical accuracy are different.
Subnormal values have reduced precision. In particular, `longdouble` describes
the current platform's format and is not promised to be IEEE binary128. Metadata
comes from NumPy and is not a portable specification of an extended format's bit
layout.

### Complex Metadata

`component_dtype` returns the real floating descriptor for a complex dtype,
preserving byte order. Query its `floating_info` for component precision and
bounds; complex values have no ordered scalar minimum or maximum. The complex
descriptor's `storage_bits` counts both components.

Metadata requests for the wrong kind raise `TypeError` rather than returning a
misleading default. Floating and complex descriptors advertise nonfinite encoding
support, but finite mathematical value constructors must still reject nonfinite
values unless an explicit representation policy allows them.

## Exact Descriptors and Future Work

`ExactDType.integer()` and `ExactDType.rational()` select the respective SymPy
representation families. They have variable storage and no fixed-width range or
floating working precision. `None` for storage width means variable storage, not
an unknown fixed format. A rational representation also accommodates integers.

These descriptors do not themselves construct or convert values.
[`cast(value, dtype)`](casting.md) uses them to select checked target wrappers.
[IntegerValue](integer.md) supplies the integer wrapper and checked integer
conversions; [RationalValue](rational.md) supplies exact ratios, arithmetic, and
checked conversion to integer wrappers.
[FloatingValue](floating.md) supplies finite NumPy floating scalars with exact
conversion by default, explicit approximation, and checked arithmetic. A floating
descriptor's existence alone does not guarantee support by that wrapper; it
validates native byte order and the platform's binary format.
[ComplexValue](complex.md) uses the complex descriptor's `component_dtype` to
validate two floating components and preserve their precision and zero signs.
`FixedDType(coefficient_dtype, *, step)` describes integer coefficient storage
and an exact positive rational step. The coefficient dtype must be a NumPy
integer or `ExactDType.integer()`; `step` accepts integer scalars, `Fraction`,
and SymPy rationals, normalized to `Fraction` metadata. Storage dtype and step
both participate in equality and hashing. Its `backend` and `storage_bits`
describe coefficient storage, excluding descriptor metadata. A NumPy coefficient
does not imply floating-point rounding or native NumPy fixed-point support.
See [fixed-point values](fixed.md) for construction and rescaling.

Decimal contexts and arbitrary-precision floating computation contexts remain
deferred. They must not be squeezed into a NumPy dtype or assigned a fabricated
fixed width.

See the [initial representation decisions](outline.md#initial-representation-decisions)
for the scalar construction and arithmetic contracts.

## References

- [NumPy dtype metadata](https://numpy.org/doc/stable/reference/generated/numpy.dtype.html)
- [NumPy integer limits](https://numpy.org/doc/stable/reference/generated/numpy.iinfo.html)
- [NumPy floating limits and platform differences](https://numpy.org/doc/stable/reference/generated/numpy.finfo.html)
