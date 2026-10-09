# Numeric Subpackage Outline

This document specifies the intended `jvs.numeric` contracts. It builds on the
repository's [design philosophy](../../DESIGN.md); it does not describe completed
implementation of every planned module. The classification foundation lives in
`number.py` with isolated backend adapters. `dtype.py` supplies representation
descriptors. Checked integer, rational, floating, complex, and fixed-point wrappers are
implemented, together with explicit scalar casting and arithmetic dispatch.
Real-valued wrappers also support exact ordering across representations.
The [explicit promotion helper](promotion.md), `common_dtype`, selects a common
operand representation from two descriptors. [NumericBuffer](storage.md) provides
owned, checked NumPy storage, scalar extraction, and explicit dtype conversion.
Explicit addition, subtraction, and multiplication also support buffers with
identical shapes. Automatic promotion, additional precision backends, broadcasting,
reductions, buffer division, borrowed views, and shared or
memory-mapped storage remain future work.

## Goal

Build a `jvs.numeric` subpackage that separates:

1. **Mathematical meaning** — what set a number belongs to.
2. **Computational representation** — how the number is encoded.
3. **Storage** — where/how its data lives.
4. **Operations** — how values interact, convert, and preserve information.

The mathematical layer should model standard number-set relationships using Python
OOP and metaprogramming. Established backends implement representations and
computation; JVS defines domain meaning, validation, and explicit conversion and
failure policies.

---

## Mathematical Set Model

The fundamental number hierarchy is:

```text
ℕ ⊂ Whole ⊂ ℤ ⊂ ℚ ⊂ ℝ ⊂ ℂ
```

where:

- `ℕ` — Natural numbers, using `{1, 2, 3, ...}`
- `Whole` — Nonnegative integers, `{0, 1, 2, ...}`
- `ℤ` — Integers
- `ℚ` — Rational numbers
- `ℝ` — Real numbers
- `ℂ` — Complex numbers

This maps naturally to a primary inheritance chain:

```text
Number
└── Complex
    └── Real
        └── Rational
            └── Integer
                └── Whole
                    └── Natural
```

Inheritance between these set classes declares mathematical subset containment:

```python
Natural <= Integer
Integer <= Rational
Rational <= Real
Real <= Complex
```

The metaclass can also provide set-like syntax:

```python
from fractions import Fraction

3 in Integer
Fraction(1, 3) in Rational

Integer <= Rational
Natural < Real
```

Conceptually corresponding to:

```text
3 ∈ ℤ
ℤ ⊆ ℚ
```

### Sets and Values

The hierarchy describes mathematical sets. Set classes are not directly
instantiable scalar values: `Integer()` does not create an integer, and `Real()`
does not create a real value. Concrete representations hold backend values and
expose a separate value API. Use composition for those backend values rather than
mirroring NumPy's dtype hierarchy through mathematical inheritance.

Membership is value-dependent. A floating representation storing exactly `3.0`
belongs to `Integer`, while the same dtype storing `3.5` does not. A representation
class must not inherit a narrower mathematical set merely because some of its
values belong to that set.

For the initial ontology, `Number` denotes the same supported universe as
`Complex`: finite complex numbers. It is an organizational root, not an additional
proper superset of `Complex`. Future domains outside this universe require an
explicit extension to the model.

### Subset Relationships

- `A <= B` expresses established inclusion, including reflexivity and transitivity.
- Membership established in `A` implies membership in every established superset
  of `A`, whether the relationship uses inheritance or explicit registration.
- `A < B` requires established proper inclusion. Distinct Python class identities
  alone do not prove that their mathematical sets differ.
- The standard proper inclusions above are declared mathematical facts. Custom
  inclusion declarations do not automatically establish proper inclusion.
- Reject cycles in subset registration. Sets with the same meaning should use
  explicit aliases or declared equivalence, not opposing inclusion edges.
- Missing relationships mean unknown, not proven non-inclusion. Relationship
  queries preserve this distinction; comparison syntax raises when the requested
  relationship cannot be established or refuted.

These operators form a partial mathematical order, not a sorting order over
arbitrary Python classes. Runtime relationships do not imply static type narrowing
or change Python's `isinstance` / `issubclass` semantics.

---

## Intersecting Classifications

Not every mathematical classification forms a simple inheritance tree.

Examples include:

```text
Real
├── Rational
└── Irrational

Complex
├── Algebraic
└── Transcendental
```

For example:

```text
√2
├── Real
├── Irrational
└── Algebraic

π
├── Real
├── Irrational
└── Transcendental

i
├── Complex
└── Algebraic
```

`Algebraic` includes complex roots of nonzero polynomials with rational
coefficients. `Transcendental` is its complement within `Complex`. `Irrational`
means `Real` minus `Rational`; non-real transcendental numbers are not irrational.
Every rational number is algebraic.

Use explicit subset relationships and intersections rather than forcing these
classifications into one inheritance tree. `RealAlgebraic` and
`RealTranscendental` can name the real intersections when useful;
`AlgebraicIrrational` means `Algebraic ∩ Irrational`. Membership in an intersection
requires membership in both component sets, not merely a particular Python class.

Properties such as these should generally be predicates rather than major classes:

```text
positive
negative
zero
even
odd
prime
```

Sign and parity predicates need value-operation contracts. Defer their
implementation to concrete value APIs or capability protocols; the ontology must
not assume that its set classes implement comparison or modulo operations.

---

## Classification Contract

### Known, Refuted, and Unknown

A classification query returns an explicit result with one of three states:

| State | Meaning |
| --- | --- |
| `TRUE` | Membership is established for the supplied value and assumptions |
| `FALSE` | Non-membership is established |
| `UNKNOWN` | Available information or supported reasoning cannot decide |

Results should carry a reason and identify relevant backend facts or assumptions.
Their truth conversion must not silently treat `UNKNOWN` as false.

The `x in Set` shorthand returns a Python `bool` for established results and raises
a dedicated indeterminate-membership exception for `UNKNOWN`. Explicit queries
let callers inspect uncertainty without exceptions. A supported symbolic value
with unresolved assumptions is unknown; an unsupported input type raises
`TypeError`. Adapter bugs and backend failures propagate as errors rather than
becoming negative membership results.

The foundation API exposes `Set.query(value) -> Classification`, with a `Truth`
state and a reason, and `classify(value)` for results across all registered sets.
`Set.contains(value)` and `value in Set` are strict boolean queries.
`Set.subset_of(other, proper=False)` preserves uncertainty about relationships;
comparison operators require a decision. `Truth.UNKNOWN` and unknown classification
results raise on boolean conversion.

External rules register with `Set.register_member_test(test, value_type=SomeType)`.
They return `True`, `False`, `None` (unknown), or a `Classification` carrying a
reason. Registration is scoped to an explicit input type so unrelated inputs are
not mistaken for refuted membership. Backend failures and invalid callback results
remain errors. A custom subclass declares inclusion, not automatic membership of
all values accepted by its parent; proper inclusion requires separate evidence or
an explicit `register_subset(..., proper=True)` assertion. Custom equivalent sets
can use aliases of the same class.

### Input Boundaries

- Accept documented Python scalars and backend scalars through explicit adapters.
  Python `int`, `float`, `complex`, `Fraction`, and `Decimal` remain convenient
  inputs; backend-native inputs retain their representation during classification.
- Do not use arbitrary `numbers.Real` registration or a successful `float(value)`
  conversion as proof of rationality, integrality, or finiteness.
- Python `bool` and NumPy boolean scalars are explicitly outside the number sets
  in this API. Membership returns `FALSE`; numeric constructors reject them unless
  the caller explicitly converts them to a supported numeric value.
- Known infinities and NaNs are outside the finite number sets, including `Number`.
  Membership returns `FALSE`. Symbolic values whose finiteness is unresolved can
  instead produce `UNKNOWN`.
- Scalar membership does not implicitly reduce arrays, extract singleton values,
  or parse strings. Those inputs require a separate, explicit API.
- A finite complex value with exactly zero imaginary component is classified by
  its real component without reducing precision. An approximately zero imaginary
  component is not zero for exact membership.

### Adapter and Inference Rules

Adapters report supported facts about the original value. The ontology combines
those facts with subset, disjointness, and intersection relationships. Positive
membership propagates upward, never automatically downward; established
non-membership in a superset excludes its subsets. Disjointness can establish
negative results, but missing evidence cannot.

When an adapter establishes that an external value is irrational, the ontology
must infer membership in `Real`, `Complex`, and `Number` without requiring duplicate
adapter rules. Natural and whole-number queries use the same adapter and inference
path. An intersection is `TRUE` when all component memberships are established,
`FALSE` when any component is refuted, and otherwise `UNKNOWN`. Conflicting
established facts raise a dedicated classification-conflict error rather than
choosing whichever rule ran first.

---

## Exactness, Precision, and Accuracy

Classification describes the supplied value, not an inferred intention. A finite
binary floating-point value is rational as stored. That does not make it an exact
representation of an intended decimal, irrational constant, or measurement.

For example, exact symbolic `sqrt(2)` is irrational; a floating approximation to it
is a different stored value. Converting the approximation to higher precision or
an exact ratio does not recover `sqrt(2)`. Any API extracting a float's exact stored
ratio must be explicitly named and must not present it as recovered user intent.

Keep the following distinctions visible in the API:

- mathematical membership and assumptions;
- backend representation, dtype, and supported range;
- exact symbolic forms versus numerical approximations;
- working precision and rounding policy;
- known approximation provenance or error bounds, when actually available.

Working precision is not a guarantee of accuracy. Preserve known provenance across
conversions and report unavailable accuracy information as unknown. Do not infer
an error bound from a dtype or displayed digits. Exact constructors reject
floating approximations by default; numerical evaluation explicitly selects an
approximate representation and precision.

---

## Conversion and Failure Policies

Classification inspects values without canonicalizing or narrowing them.
Construction and casting perform explicit conversion under a declared contract.
Each concrete representation must document its defaults, accepted inputs, range,
rounding, overflow, and nonfinite-value behavior before implementation.

| Situation | Required behavior |
| --- | --- |
| Exact conversion would lose information | Raise a dedicated precision-loss error |
| Requested fixed-width representation or result overflows | Raise `OverflowError` by default |
| Input type or operation is unsupported | Raise `TypeError` or a specific unsupported-operation error |
| Value violates the requested domain | Raise `ValueError` or a specific domain error |
| Classification is unresolved | Preserve `UNKNOWN`; strict membership syntax raises |
| Explicit approximate evaluation or rounding | Apply the requested policy and expose representation and working precision |
| Explicit permissive loss or fallback | Emit a specific warning identifying the loss and resulting representation |

Routine rounding within an explicitly approximate computation does not warn on
every operation. Unexpected information loss must fail; a documented permissive
mode may warn and continue. Warnings must be filterable by category and actionable.

JVS owns arithmetic promotion and result rules. A backend's default cast or
promotion is not evidence that an operation is lossless. Fixed-width overflow
must be checked for both scalars and arrays; backend warning settings alone are
insufficient. Widening or switching backends requires a documented policy, and
mixed-backend operations must never silently downgrade exact values.

The explicit scalar helper follows the
[promotion decision table](promotion.md#family-decision-table):
`common_dtype(left, right)` selects a representation covering both complete
input value sets without inspecting values. It is opt-in; operators
keep their current strict contracts. The table defines integer range unions,
floating/component coverage, fixed-point common steps and scaled bounds, explicit
backend combinations, and errors when no permitted common representation exists.
In particular, exact/approximate family mixtures require an explicit target,
and two NumPy inputs never silently fall back to SymPy. The helper is exported
from `jvs.numeric`; selection does not guarantee that subsequent results fit.

Finite mathematical value constructors reject nonfinite values by default.
Representation APIs may explicitly support IEEE special values, but doing so
does not make them members of the finite mathematical sets. Precision and error
settings must be scoped to the operation without persistent global changes.

### Initial Representation Decisions

- **Ownership and names:** `jvs.numeric` is self-contained. `Integer`, `Real`, and
  `Complex` remain mathematical sets. Descriptors use `DType`, `NumPyDType`, and
  `ExactDType`, with `FixedDType` combining coefficient storage and step. The
  first concrete wrapper is `IntegerValue` and holds a
  backend scalar plus its descriptor. It does not inherit the mathematical set
  class. Specialized value names such as `Int8` can be added later if useful.
- **Representation:** require an explicit dtype for the first integer wrapper.
  Fixed-width signed and unsigned representations use NumPy; unbounded exact
  integers use SymPy. Exact rationals also have a SymPy descriptor. Dtype metadata
  reports actual backend capabilities and does not imply mathematical membership,
  numerical accuracy, or permission to lose information.
- **Construction and arithmetic:** the first integer wrapper accepts Python
  integers, NumPy integer scalars, and SymPy integers, rejecting booleans and
  implicit float/rational conversions. Check bounds before converting. Initial
  addition, subtraction, multiplication, and unary negation preserve an explicitly
  shared operand dtype and raise on fixed-width overflow. Mixed representations
  require explicit conversion; automatic operator promotion remains deferred. True
  division of matching integer dtypes always produces `RationalValue` with
  `ExactDType.rational()`, including integral results, without floating conversion
  or fixed-width overflow. Zero divisors raise `ZeroDivisionError`. Exact integer
  intermediates may be used to detect overflow in integer-valued operations.
  Matching-dtype `//`, `%`, and `divmod` return integer wrappers: the quotient
  floors toward negative infinity and a nonzero remainder has the divisor's sign.
  Check each requested result before storage. In particular, signed minimum
  divided by `-1` overflows the quotient, but its remainder is valid zero;
  `divmod` requires both results to fit. See
  [integer quotient and remainder](integer.md#integer-quotient-and-remainder).
- **Exact rationals:** `RationalValue` requires `ExactDType.rational()` and exact
  integer numerator/denominator inputs. Reject zero denominators and implicit
  float conversions; reduce fractions and normalize signs with SymPy. Arithmetic
  between rational wrappers supports `+`, `-`, `*`, and `/`, preserving the
  rational wrapper even for integral results. Explicit `to_integer(dtype)` checks
  integrality and the target range. Equality accepts numeric wrappers and
  supported integer scalars; raw rational representations require explicit
  wrapping to avoid conflicting backend equality/hash conventions.
- **Finite floating values:** `FloatingValue` requires an explicit native NumPy
  floating dtype. Construction and `.to(dtype)` preserve stored/exact input values
  or raise `PrecisionLossError`; `approx` and `.to(dtype, approximate=True)` permit
  nearest-even rounding. Matching-dtype arithmetic permits normal rounding but
  raises on overflow, zero division, and inexact values below the smallest normal
  (including rounding up to that boundary). Exact subnormals are valid. The
  `rounded` flag propagates known rounding, never a claim about external accuracy.
  Exact integer ratios validate conversion and NumPy results without narrowing
  through Python float. Only validated regular binary formats are supported.
- **Finite complex values:** `ComplexValue(real, imag=0, dtype=...)` uses an explicit
  native NumPy complex dtype and checked `FloatingValue` components. Construction
  and `.to(dtype)` are exact by default; `approx` and explicit approximate casts
  apply the floating policy componentwise. Addition, subtraction, negation, and
  conjugation preserve the dtype and component rounding history. `.to_real()`
  requires a stored zero imaginary part and retains known rounding history.
  Classification follows the stored real value when the imaginary part is zero;
  otherwise these rational-component values are algebraic and non-real. Complex
  multiplication and division use exact stored-ratio intermediates, then round
  each final component once (nearest-even), checking final range and inexact
  underflow only. Division by complex zero raises. Both output components inherit
  known input rounding; zero signs follow the explicit product/sum formula rules
  documented in [complex values](complex.md).
- **Fixed-point values:** `FixedDType(coefficient_dtype, step=...)` describes an
  integer coefficient multiplied by a positive exact rational step. NumPy integer
  and unbounded exact integer coefficients are supported. `FixedValue(value,
  dtype=...)` requires exact lattice membership by default; `from_coefficient`
  explicitly accepts storage units. Approximation rounds the coefficient
  nearest-even and records known rounding. Check range before rounding or
  storage; explicit rounding to zero is allowed. Matching-dtype addition and
  subtraction retain the representation and history; multiplication and division
  produce exact rational results and discard history. Rescaling is explicit.
  Classification and equality use coefficient times step, never the coefficient
  alone. See [fixed-point values](fixed.md).
- **Exact real ordering:** `<`, `<=`, `>`, and `>=` compare integer, rational,
  floating, and fixed-point wrappers by their stored mathematical values, with
  no casting, rounding, or mutation of provenance. Supported raw integer scalars
  are accepted in either order; other raw numeric inputs require wrapping.
  Complex wrappers are unordered, even with zero imaginary part; require explicit
  `.to_real()` extraction. NumPy scalar ordering ufuncs preserve these rules and
  operand direction. See [ordering](ordering.md) for dispatch boundaries.
- **Explicit casting:** `cast(value, dtype, *, approximate=False)` accepts numeric
  wrappers and explicit descriptors, reusing the checked scalar APIs. Integer
  targets require integral values and target range checks; real targets require
  a zero imaginary component. Approximation is opt-in for floating, complex, or
  fixed-point targets. Integer/rational targets explicitly extract stored values and discard
  signed-zero and rounding-history metadata. See the
  [conversion matrix](casting.md#conversion-matrix); automatic promotion remains
  deferred.
- **Explicit arithmetic dispatch:** `add`, `subtract`, `multiply`, and `divide`
  require a keyword-only operand dtype, cast both wrappers under the requested
  conversion policy, then invoke the existing checked operators. `approximate`
  controls operand conversion only, not normal floating arithmetic rounding.
  Integer division and fixed-point multiplication/division return exact rationals;
  other results retain the operand dtype. See the [result-type table](arithmetic.md#result-types). Existing
  operators remain strict and automatic promotion is deferred. `add`, `subtract`,
  and `multiply` also accept two buffers with identical shapes and an explicit
  native NumPy target. Each pair uses checked scalar casts and arithmetic, with
  independent result storage and coordinate-specific failures. Approximation
  controls operand conversion only; scalar rounding-history rules still apply.
  Empty buffers validate formats and policy. Broadcasting, reductions, mixed
  scalar/buffer arguments, and buffer division are deferred.
- **Owned numeric storage:** `NumericBuffer(data, dtype=..., approximate=False)`
  copies a plain typed NumPy array, then applies the scalar conversion rules to
  each copied value before storing it. The target is an explicit native NumPy
  integer, floating, or complex dtype. Shape is preserved, including empty and
  zero-dimensional arrays. Complete integer coordinates extract checked scalar
  wrappers with per-component rounding history. `to(dtype, approximate=False)`
  creates independent storage, preserving shape and applying scalar casting's
  exactness and history rules to every element. Failures report the coordinate
  and leave the source unchanged. `to_numpy()` returns an
  independent writable copy and explicitly drops wrapper history. The private
  data and metadata are read-only; the public API exposes no mutation or views.
  Source non-native byte order is decoded, while non-native target storage is
  deferred. See [owned buffers](storage.md) for ownership and failure boundaries.

`IntegerValue` implements these initial checked operations and exact integer
conversions. Its NumPy scalar payload requires native byte order; non-native dtype
descriptors remain available for future storage outputs. See
[integer values](integer.md), [rational values](rational.md),
[floating values](floating.md), [complex values](complex.md), [fixed-point values](fixed.md),
[dtype descriptors](dtype.md), [explicit casting](casting.md),
[explicit arithmetic](arithmetic.md), [ordering](ordering.md), and
[owned buffers](storage.md) for the implemented
APIs and their current limits. Arbitrary-precision computation contexts will be
specified with their corresponding modules.

---

## Overall Architecture

```text
                       Numeric value
                    JVS domain interface
                            │
        ┌───────────────────┼────────────────────┐
        │                   │                    │
      domain              dtype                storage
        │                   │                    │
   ℕ / ℤ / ℚ / ℝ / ℂ   representation       memory model
                            │                    │
                       ├─ width              ├─ owned
                       ├─ signedness         ├─ view
                       ├─ precision          ├─ shared
                       ├─ exponent           └─ mmap
                       └─ scale
                            │
                            ▼
                   operations between values
                     ┌──────┴──────┐
                  arithmetic    casting
```

The important separation is:

```text
mathematical number
        ≠
numeric representation
        ≠
memory/storage representation
```

For example, the mathematical integer `42` could be represented as:

```text
int8
int32
uint64
arbitrary-precision integer
```

while still belonging to:

```text
42 ∈ ℕ
42 ∈ ℤ
42 ∈ ℚ
42 ∈ ℝ
42 ∈ ℂ
```

---

## Package Layout

```text
jvs/
└── numeric/
    ├── __init__.py
    ├── number.py
    ├── dtype.py
    ├── integer.py
    ├── floating.py
    ├── fixed.py
    ├── rational.py
    ├── complex.py
    ├── arithmetic.py
    ├── casting.py
    └── storage.py
```

| Module | Responsibility |
|---|---|
| `number.py` | Mathematical number sets, subset relationships, classification, and set-like metaprogramming |
| `dtype.py` | Numeric representation descriptors: width, signedness, precision, scale, exponent/significand |
| `integer.py` | Concrete integer representations and integer-specific behavior |
| `floating.py` | Floating-point representations and behavior |
| `fixed.py` | Fixed-point representations |
| `rational.py` | Exact numerator/denominator representations |
| `complex.py` | Complex-number representations composed from real components |
| `arithmetic.py` | Arithmetic dispatch, type promotion, and result-type rules |
| `casting.py` | Explicit conversion, loss detection, rounding, truncation, and overflow policy |
| `storage.py` | Owned buffers, borrowed views, shared memory, memory mapping, and related storage concepts |
| `__init__.py` | Public `jvs.numeric` API |

Backend adapters should be isolated from ontology rules. Their concrete module
layout can follow implementation needs; adding a backend must not require
duplicating the subset and classification logic.

---

## Design Principle

`number.py` should remain the **mathematical ontology**.

It answers:

```text
What kind of number is this?
What mathematical sets does it belong to?
How are those sets related?
```

The remaining modules answer different questions:

```text
dtype.py       → How is it represented?
storage.py     → Where/how is it stored?
casting.py     → How can representation change?
arithmetic.py  → How do representations interact?
```

This keeps mathematical identity independent from implementation details.
Runtime set relationships and Python static typing have separate responsibilities;
type annotations do not prove value membership.

---

## Supporting Numerical Tooling

The subpackage uses established libraries as implementation backends behind JVS
contracts. Python remains the vocabulary for semantic structure and control.

### Backend Responsibilities

| Tool | Primary Role |
|---|---|
| **Python stdlib** | Semantic objects, protocols, enums, and explicit scalar adapters; `int` / `Fraction` for appropriate exact values and `Decimal` for decimal arithmetic |
| **NumPy** | Default backend for fixed-width scalars, dtype descriptors, homogeneous arrays, computation, and buffer interoperability |
| **SymPy** | Preferred backend for exact integers and rationals, symbolic expressions and constants, and supported classification facts; preserve unresolved results |
| **mpmath** | Arbitrary-precision real and complex approximation with an explicit working-precision contract |
| **SciPy** | Algorithms such as integration, optimization, interpolation, and sparse computation when a concrete module requires them |

A useful conceptual division is:

```text
NumPy  → machine-oriented numeric representation
SymPy  → exact / symbolic mathematics
mpmath → arbitrary-precision numerical approximation
SciPy  → numerical algorithms
```

NumPy and SymPy are the initial implementation backends for their respective
concerns. Add direct mpmath integration when configurable-precision numerical APIs
are needed; SymPy numerical evaluation can serve symbolic evaluation needs.
Introduce SciPy only for an actual algorithm. None of these choices permits
implicit approximation, overflow, or a claim that all classifications are decidable.

Wrap backend values through composition. Preserve extended-precision inputs and
inspect the backend's actual capabilities rather than assuming a dtype name
guarantees the same precision on every platform. General dtype descriptors must
also accommodate exact and arbitrary-precision representations that cannot be
described by a NumPy dtype alone.

### Optional Structured Input Validation

Consider Pydantic when computation configuration, external records, or serialized
numeric values need structured validation. Defer adoption until such a boundary
exists; it is not a prerequisite for refining `number.py`.

Keep classification results and internal policies in ordinary enums, dataclasses,
and protocols. JVS owns membership inference, dtype and shape checks, precision,
overflow, and domain invariants. Any Pydantic integration must reuse those
validators, preserve unknown results, and honor explicit conversion policies.

`arbitrary_types_allowed=True` does not validate the contents of NumPy or SymPy
objects, and strict mode alone does not prove a numerical conversion is lossless.
Serialization must explicitly preserve the required representation and exactness
information. See [Pydantic's custom-type integration](https://docs.pydantic.dev/latest/concepts/types/#handling-third-party-types)
and the repository's [runtime validation guidance](../../DESIGN.md#runtime-validation).

### Performance / Acceleration

| Tool | Potential Role |
|---|---|
| **Numba** | JIT compilation of numerical kernels operating on compatible representations |
| **NumExpr** | Efficient evaluation of large array expressions with reduced temporary allocations |
| **JAX** | Accelerated and differentiable array computation; useful as an interoperability target rather than a foundational dependency |
| **Cython / native extensions** | Possible future path for performance-critical low-level operations |
| **gmpy2** | High-performance arbitrary-precision integer, rational, and floating-point arithmetic |
| **python-flint** | High-performance exact arithmetic, number theory, polynomials, and algebraic-number computation |

Optimization should remain separable from semantics:

```text
correct mathematical model
        ↓
correct representation
        ↓
correct operations
        ↓
optimized implementation
```

Performance tooling should therefore be optional rather than defining mathematical behavior.

### Data / Tabular Ecosystem

| Tool | Role |
|---|---|
| **pandas** | Interoperability with conventional scientific tabular data |
| **Polars** | Interoperability with typed columnar/Arrow-oriented data |
| **Apache Arrow** | Useful future target for explicit typed memory and zero-copy data interchange |

These are consumers of the numeric system rather than core mathematical dependencies.

### Visualization

| Tool | Role |
|---|---|
| **Matplotlib** | Primary static numerical/mathematical visualization backend |
| **Seaborn** | Statistical visualization layered on Matplotlib |
| **Plotly** | Interactive visualization |
| **Manim** | Mathematical animation and explanatory visualization |

Visualization should remain downstream:

```text
jvs.numeric
    ↓
mathematical/data objects
    ↓
visualization adapters
    ↓
Matplotlib / Plotly / Manim / ...
```

### Design Principle

External libraries may serve several roles:

```text
Representation / computation / reference
    NumPy, SymPy, mpmath

Acceleration
    Numba, NumExpr, JAX, native code

Interoperability
    NumPy, pandas, Polars, Arrow

Visualization
    Matplotlib, Seaborn, Plotly, Manim
```

The core `jvs.numeric` API should remain mathematically defined and
backend-independent wherever practical. Libraries implement representations and
algorithms; JVS defines the mathematical ontology and checks its contracts.

---

## Implementation Sequence

1. Establish the contracts in this outline and keep repository-wide principles in
   `DESIGN.md`.
2. Refine `number.py` and the required classification adapters. Cover uncertainty,
   subset inference, disjointness, intersections, and strict failures before adding
   value arithmetic.
3. Add focused regression tests for membership and relationship invariants; run
   pytest, Ruff lint/format checks, and ty for the implementation changes.
4. Implement `dtype.py` and one concrete integer representation, including checked
   construction and the minimal operations needed to exercise its contract.
5. Expand representations and dispatch incrementally. Add storage, additional
   backends, and acceleration when a concrete use requires them.

`jvs.numeric` remains self-contained and does not depend on `jvs.core`. The older
core wrappers may serve as reference material. Removing them or reorganizing the
packages is a separate future decision and does not block numeric development.

Foundation regression cases must include:

- the standard subset chain, real/complex algebraic classifications, and proper
  inclusion versus equivalence;
- external membership propagating to supersets, intersection reasoning, cycle
  rejection, and contradictory facts;
- unknown symbolic membership and unsupported input errors;
- natural-number zero exclusion, whole-number zero inclusion, booleans, NaNs,
  infinities, and real-valued complex scalars;
- large exact integers and extended-precision floating values without narrowing;
- exact symbolic irrationals versus their floating approximations.

## Backend References

- [NumPy scalar representations](https://numpy.org/doc/stable/reference/arrays.scalars.html)
- [NumPy promotion and overflow behavior](https://numpy.org/doc/stable/reference/arrays.promotion.html)
- [SymPy assumptions and three-valued classification](https://docs.sympy.org/latest/guides/assumptions.html)
- [SymPy mathematical predicates](https://docs.sympy.org/latest/modules/assumptions/predicates.html)
- [mpmath input and working precision](https://mpmath.readthedocs.io/en/latest/basics.html)
