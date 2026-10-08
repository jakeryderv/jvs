# Owned Numeric Buffers

`NumericBuffer(data, *, dtype, approximate=False)` stores an owned homogeneous
NumPy array of checked finite integer, floating, or complex values. This first
API provides storage and scalar access; it has no array arithmetic or promotion.

## Construction and ownership

`data` must be a plain `numpy.ndarray` with an integer, floating, or complex dtype.
Lists, raw scalars, object arrays, booleans, structured dtypes, dtype metadata,
masked arrays, and ndarray subclasses are rejected. Requiring an already typed
array makes NumPy's input inference a separate, explicit caller decision. Any
rounding before construction is outside this API's known history.

`dtype` must be an explicit native-byte-order `NumPyDType`. Exact and fixed-point
buffers are deferred. Source arrays may use non-native byte order: their numeric
values are decoded and checked without reinterpretation. Floating source and
target formats must satisfy the existing scalar format contract, even for empty
arrays. There is no fallback for an unsupported format.

Construction copies the source before checking its elements, then converts each
copied value using the existing scalar wrappers and `cast` rules. It never casts
the whole array to the target dtype before validation. The final data and
rounding metadata are privately owned, contiguous, and read-only. Mutating the
original array or an exported array cannot change the buffer. This is protection
through the public API, not a security boundary against deliberate access to
private attributes. A caller must synchronize concurrent writes during the input
copy; construction does not promise an atomic snapshot of shared mutable data.

- Conversion is exact by default. Explicit `approximate=True` permits normal
  nearest-even floating/component rounding and must be an actual Python `bool`.
- Integer targets reject approximation, require integral values, and check range.
- Complex-to-real conversion requires a stored zero imaginary component.
- Nonfinite values, overflow, and inexact floating underflow raise under the
  scalar policies. Exact subnormals remain valid. An element failure includes
  its multidimensional index as an exception note; no partial buffer is returned.
- The shape is preserved, including zero-dimensional arrays and empty dimensions.
  Transposed, strided, and read-only inputs are copied into owned C-order storage.
- Input format, target format, and option validation also apply to empty arrays.

These are correctness-first scalar checks over an array, not a vectorized fast
path. Construction allocates a detached source snapshot, target data, and one
byte of rounding metadata per element. Export allocates another data copy.

```python
import numpy as np
from jvs.numeric import IntegerValue, NumericBuffer, NumPyDType

source = np.array([[1, 2], [3, 4]], dtype=np.int16)
buffer = NumericBuffer(source, dtype=NumPyDType("int32"))
assert buffer.shape == (2, 2) and buffer.ndim == 2 and buffer.size == 4
assert len(buffer) == 2
assert isinstance(buffer[1, 0], IntegerValue)
assert buffer[1, 0] == 3
source[1, 0] = 99
exported = buffer.to_numpy()
exported[1, 0] = -99
assert buffer[1, 0] == 3
assert exported.dtype == np.dtype("int32")
```

## Scalar indexing and metadata

Public metadata consists of `dtype`, `shape`, `ndim`, `size`, and `rounded`.
`len(buffer)` is the first dimension's length; it raises `TypeError` for a
zero-dimensional buffer. `repr` displays metadata without dumping element data.

Indexing requires one Python/NumPy integer per dimension; negative indices are
supported. A one-dimensional buffer accepts `buffer[i]` or `buffer[i,]`, a
multidimensional buffer requires a complete tuple such as `buffer[i, j]`, and a
zero-dimensional buffer uses `buffer[()]`. Boolean indices, slices, partial
coordinates, masks, lists, ellipses, and new axes are unsupported. Invalid index
types raise `TypeError`; wrong coordinate counts or out-of-range positions raise
`IndexError`. An extracted element is an immutable `IntegerValue`, `FloatingValue`,
or `ComplexValue` whose dtype equals the buffer dtype.

`rounded` is true when any element carries known rounding from construction.
Per-element real/imaginary rounding flags are retained, so extraction restores
each component's own history. False is not a claim about external accuracy.
Floating zero signs are preserved according to scalar casting rules: conversion
to integer discards signs, and real-to-complex adds positive imaginary zero.

```python
from jvs.numeric import FloatingValue, PrecisionLossError

source = np.array([0.5, 0.1], dtype=np.float64)
try:
    NumericBuffer(source, dtype=NumPyDType("float32"))
except PrecisionLossError as error:
    assert any("(1,)" in note for note in error.__notes__)
else:
    raise AssertionError("Implicit rounding must fail")
rounded = NumericBuffer(source, dtype=NumPyDType("float32"), approximate=True)
assert rounded.rounded
half, tenth = rounded[0], rounded[1]
assert isinstance(half, FloatingValue) and not half.rounded
assert isinstance(tenth, FloatingValue) and tenth.rounded
```

## NumPy export and limits

`to_numpy()` returns a new writable, C-contiguous ndarray with the same shape,
dtype, and stored values. Every call makes an independent copy. It exposes no
view, base array, or rounding mask from the buffer. Export transfers numerical
control to NumPy and drops wrapper rounding history; reimporting that array cannot
recover it. Use scalar extraction when the flags matter.

Implicit NumPy conversion and ufuncs raise `TypeError` and direct callers to
`to_numpy()`. There is no mutation interface, implicit iteration, scalar truth
value, elementwise equality, ordering, broadcasting, reduction, or arithmetic.
Buffer equality uses ordinary object identity. Mathematical membership in
`number.py` remains scalar-only; query extracted values individually.

Borrowed views, shared memory, memory mapping, checked array arithmetic, exact
and fixed-point storage, slicing, buffer-to-buffer conversion, and aggregate
classification remain separate future work. `jvs.core` is not involved.
