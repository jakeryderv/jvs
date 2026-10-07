# Design Philosophy

## Overview

This project separates three complementary concerns:

1. **Semantic and object structure**
2. **Exact and symbolic mathematical representation**
3. **Concrete numerical representation and computation**

The core rules are:

> Use native Python for structure, semantics, relationships, and control.

> Preserve exact values and symbolic structure where the domain requires them.

> Use NumPy as the default vocabulary for concrete numerical values and homogeneous numerical data.

These are complementary concerns, not a required processing pipeline. A project may use only the concerns it needs, and a domain object may combine them.

SymPy is the preferred backend for symbolic manipulation and exact mathematical expressions. NumPy is the preferred default backend for concrete numerical representation. Other numerical backends may be used when their precision, arithmetic model, or algorithms are required.

The goal is to make numerical APIs explicit, strongly typed, representation-aware, and consistent while keeping semantic Python code idiomatic and composable.

Package-specific contracts belong in their design documents. For `jvs.numeric`,
see the [numeric outline](docs/numeric/outline.md) for mathematical membership,
backend responsibilities, and strict conversion and failure policies.

---

## Native Python Semantic Layer

Native Python defines the program's semantic and structural model.

Prefer:

- `str` for names, labels, and text
- `bool` for program state and control
- `None` for absence
- `dict` for mappings and relationships
- `list` for heterogeneous or semantic sequences
- `tuple` for structural grouping
- `set` / `frozenset` for membership
- `Enum` / `Literal` for symbolic choices
- classes and dataclasses for domain objects
- protocols, generics, and type aliases for API structure

These types answer questions such as:

- What is this object?
- What does it mean?
- How does it relate to other objects?
- What state is the program in?
- How is the API organized?

Native Python is therefore the default **semantic vocabulary**, not the default numerical vocabulary.

---

## Exact and Symbolic Representation

Preserve exact values and symbolic structure when they are part of the domain's meaning.

Examples include:

- arbitrary-size integers
- exact fractions
- symbolic constants and variables
- expressions and their structure
- mathematical domains and assumptions

Native Python types such as `int`, and standard-library types such as `fractions.Fraction`, may be appropriate when exact values themselves are the desired representation.

Use SymPy when the domain requires symbolic manipulation or exact mathematical expressions.

For example:

- `sp.Integer` and `sp.Rational` represent exact integers and fractions
- `sp.pi` retains an exact symbolic constant
- `sp.Symbol` represents a symbolic variable and can carry assumptions
- `sp.Expr` provides symbolic expression structure
- SymPy matrices and sets represent mathematical structure that need not be reduced to numerical arrays

SymPy supplies representation and manipulation capabilities; domain objects still define meaning and valid operations.

Exactness and precision are distinct.

An arbitrary-precision approximation is still an approximation. Increasing precision does not recover symbolic structure that has already been discarded.

Keep exact or symbolic representations until concrete numerical evaluation is needed. Make evaluation policy and any loss of exactness explicit at that boundary.

---

## NumPy Numerical Layer

NumPy is the project's **default numerical vocabulary**.

Concrete numerical values should generally use NumPy scalar types, and homogeneous numerical collections should use NumPy arrays.

Prefer:

- `np.int8` ... `np.int64`
- `np.uint8` ... `np.uint64`
- `np.float16` ... `np.float64`
- NumPy complex types
- `np.bool_` for numerical boolean values and masks
- `ndarray` for vectors, matrices, tensors, images, samples, and other homogeneous numerical data
- `np.datetime64` and `np.timedelta64` where time is represented numerically
- structured dtypes where fixed numerical records are appropriate

These types make properties such as the following explicit:

- width
- precision
- signedness
- dtype
- shape
- memory representation

### Why NumPy Is the Default Numerical Vocabulary

The purpose of this convention is broader than computational performance.

Using NumPy consistently for concrete numerical values provides:

- **explicit representation** instead of Python's implicit numerical representation
- **consistent width and precision** across scalar and array APIs
- **signed and unsigned integer choices**
- **clearer numerical contracts**
- **stronger integration with static and runtime typing tools**
- **natural promotion from scalar values to arrays**
- **predictable interoperability with binary, native, scientific, and accelerator-backed systems**
- **one coherent numerical model across a package**

For example, an API that consistently uses:

```text
np.uint32
np.int64
np.float32
np.float64
NDArray[np.float64]
```

has a clearer representation model than one that mixes:

```text
int
float
np.int64
np.float32
NDArray[np.float64]
```

without an explicit reason.

The convention therefore favors:

> **NumPy by default for concrete numerics, with deviations made intentionally.**

---

## Numerical Backends and Precision

NumPy is the default numerical representation, but not the only numerical backend.

Choose another backend when its arithmetic model, precision, or algorithms are part of the contract.

| Backend | Role |
| --- | --- |
| NumPy | Default concrete numerical scalars, homogeneous arrays, explicit dtypes, and array computation |
| mpmath | Arbitrary-precision real and complex numerical evaluation |
| SciPy | Numerical algorithms built around NumPy data, including integration, optimization, interpolation, and sparse computation |
| `decimal.Decimal` | Decimal arithmetic with explicit precision and rounding rules |

Mpmath provides numerical approximations at configurable working precision. It belongs to the numerical concern alongside NumPy.

Decimal provides a distinct decimal arithmetic model and configured rounding behavior.

SymPy also provides configurable-precision numerical evaluation through `evalf()` and `N()`. Use this when evaluation of a symbolic expression is sufficient; use a numerical backend directly when the computation requires its numerical API.

Make precision part of the computation's contract:

- choose and document working precision where it matters
- scope changes to precision and rounding to the relevant computation
- preserve accurate inputs; converting an already-rounded Python float to a higher-precision representation cannot recover lost information
- make conversions between backends explicit
- make reductions in precision explicit
- assess numerical accuracy from inputs and algorithms, not displayed digits alone

Keep backend choices behind the project's domain API where practical.

---

## Scalar Convention

Concrete numerical scalars should default to NumPy scalar types.

This convention applies to the domain's concrete numerical model. Native Python integers remain appropriate for ordinary implementation and control purposes, such as loop counters and configuration values, and for external interfaces that require them. Convert deliberately at boundaries with the numerical model.

Examples:

```text
mass          → np.float64
temperature   → np.float32
index         → np.int64
count         → np.uint32
object_id     → np.uint64
sample        → np.float32
timestamp     → np.datetime64
```

Choose width, signedness, and precision according to the domain contract.

Representation width and exactness are separate properties. NumPy integer types represent integers exactly within their supported ranges; unbounded integers require a representation without a fixed-width limit.

For fixed-point representations, separate integer coefficient storage from the
exact rational step: the mathematical value is their product. Changing the step
is an explicit conversion. Preserve the value exactly by default, or require an
explicit rounding request; check range before rounding and storage. In
`jvs.numeric`, matching-representation addition and subtraction retain the
fixed-point dtype, while multiplication and division return exact rational
values for subsequent explicit rescaling. Extracting rational results discards
rounding history and does not recover the original input's intended value.

Use a non-NumPy representation when the numerical semantics specifically require it:

```text
unbounded_integer → int or sp.Integer
exact_ratio       → Fraction or sp.Rational
precise_value     → mpmath.mpf
decimal_value     → decimal.Decimal
```

This makes NumPy the normal concrete representation while keeping exact, symbolic, arbitrary-precision, and decimal arithmetic as explicit alternatives.

### Default Dtype Policy

Each domain must declare its canonical dtypes for concrete numerical values, including floating-point precision and integer width and signedness where applicable. The examples above illustrate possible choices; they do not establish universal defaults.

Use these declared defaults consistently across the domain's scalar and array APIs. Choose defaults from the domain's accuracy, range, storage, and interoperability requirements, and apply them explicitly during canonicalization.

A project may establish shared defaults such as `float64` and `int64` when its numerical model is standardized around those choices. Domains must document which shared defaults they adopt and any exceptions.

---

## Boolean Convention

Use native `bool` when the value represents semantic program state:

```text
enabled
visible
debug
```

Use `np.bool_` when the value belongs to the numerical/data model:

```text
valid_sample
collision_state
numeric_flag
```

Use boolean arrays for masks and homogeneous boolean data:

```text
selection_mask
collision_mask
valid_samples
```

The distinction is semantic:

> Python `bool` controls the program.

> NumPy boolean values represent numerical/data state.

---

## Collections

Container choice follows semantic meaning.

Use Python containers for program structure:

```text
dict[str, Body]
list[Body]
set[Entity]
tuple[Frame, Frame]
```

Use NumPy arrays for homogeneous numerical collections:

```text
float64[3]
float64[4, 4]
float32[N, 3]
uint8[H, W, 3]
```

The array notation above describes dtype and shape conceptually; it is not literal Python syntax.

A domain object may therefore combine both layers:

```text
Body
├── name: str
├── active: bool
├── id: np.uint64
├── mass: np.float64
├── position: float64[3]
├── velocity: float64[3]
└── vertices: float32[N, 3]
```

The outer object expresses semantic meaning.

The contained NumPy values express concrete numerical representation.

The same object may also retain exact or symbolic forms when the domain requires them.

---

## Meaning and Representation

Representation describes how a value is stored and computed.

Domain semantics describe what the value means and which operations are valid.

A `float64[3]` array may represent:

- a position
- a direction
- a velocity
- an axis
- a measurement

Its dtype and shape do not establish:

- units
- coordinate frame
- ownership
- identity
- semantic validity

Use domain classes where identity, relationships, behavior, or invariants require additional meaning.

Use numerical aliases directly when representation alone is sufficient.

---

## Typing and Enforcement

Separate four responsibilities:

| Responsibility | Purpose | Possible tools |
| --- | --- | --- |
| Static typing | Describe API structure, domain types, and supported numerical types | Python typing, NumPy typing, static type checker |
| Representation annotations | Express dtype, rank, dimensions, and dimension relationships | NumPy annotations, jaxtyping |
| Runtime validation | Check actual inputs and outputs against representation contracts | Explicit validation or compatible runtime checker |
| Domain invariants | Establish value constraints and semantic rules | Domain constructors, operations, validators |

Annotations express contracts; they do not automatically enforce them at runtime.

Static checking and runtime validation provide different guarantees.

### Static Numerical Typing

Use `numpy.typing` and NumPy annotations for the concrete numerical vocabulary.

For example:

```text
np.float64
np.uint32
NDArray[np.float64]
NDArray[np.uint8]
```

`NDArray[np.float64]` specifies a dtype with unspecified shape.

Current NumPy typing does not fully express exact axis lengths or shared dimension relationships, so additional tooling may be appropriate.

### Shape Contracts

Jaxtyping may be used to express richer numerical representation contracts.

Conceptually:

```text
Position → float64[3]
Matrix4  → float64[4, 4]
Vertices → float32[N, 3]
Image    → uint8[H, W, 3]
Mask     → bool[N]
```

For example:

```text
Float64[np.ndarray, "N 3"]
```

describes a two-dimensional `float64` array with three columns.

Shared symbolic dimensions may express relationships between arguments.

Jaxtyping's dtype and shape annotations require compatible runtime checks for enforcement; do not assume a static checker verifies those contracts. For example, `@jaxtyped(typechecker=beartype)` can check function arguments and returns at runtime. See the [jaxtyping runtime documentation](https://docs.kidger.site/jaxtyping/api/runtime-type-checking/) and [static typing limitations](https://docs.kidger.site/jaxtyping/faq/).

### Runtime Validation

Runtime checking may be added where API boundaries require stronger guarantees.

Possible checks include:

- scalar dtype
- array dtype
- shape
- rank
- shared dimensions
- allowed conversions

Representation validation remains distinct from domain validation.

Pydantic may be used at structured input and serialization boundaries, such as
configuration files or external records, when a concrete need justifies it. Keep
domain validation reusable independently of Pydantic; model validators should call
the same validation functions used by domain constructors and operations.

Use explicit validation and conversion policies. Pydantic's
[strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/) reduces
coercion but does not establish numerical exactness or mathematical validity.
Allowing arbitrary backend types only checks their instance type; validating their
contents requires custom rules. Pydantic complements static checking and tests.

### Domain Validation

Dtype and shape do not establish properties such as:

- positivity
- finiteness
- normalization
- units
- valid coordinate frames
- allowable ranges
- semantic relationships

Validate these separately when required.

Avoid repeatedly validating trusted internal data inside numerical hot loops.

---

## Conversion Boundaries

Distinguish accepted inputs from canonical internal representation.

A public API may accept convenient Python or array-like inputs.

### Canonicalization Policy

Domain and API boundaries should normalize accepted concrete numerical values into the domain's declared canonical representation, usually NumPy with an explicit dtype. This keeps the internal numerical model predictable and consistent regardless of the accepted input form.

Canonicalization must honor the declared precision and information-loss policy.
A default dtype is not permission to narrow a wider input, overflow an exact
integer, or approximate an exact value silently. Validate whether conversion is
permitted before discarding information. Queries that only inspect or classify a
value should preserve its original representation.

Apply the same representation contract to numerical outputs. Preserve exact and symbolic values until numerical evaluation is requested; use another backend's declared canonical representation when its arithmetic model, precision, or algorithms are required.

For example:

```text
accepted input
    ↓
conversion + validation
    ↓
canonical representation for the selected backend
(NumPy by default)
```

Make conversion deliberate:

- document accepted input forms
- document the resulting representation, including NumPy dtype or the selected backend's precision and arithmetic model
- distinguish checking from conversion
- define precision-loss policy
- define overflow policy
- define rounding policy
- define nonfinite-value policy
- preserve exact/symbolic forms until numerical evaluation is requested

A successful dtype cast does not prove that the original value was valid or that semantic meaning was preserved.

Backend defaults do not replace domain policy. Operations that violate a declared
contract must fail explicitly. A documented permissive policy may emit a specific,
actionable warning and continue; routine rounding within an explicitly requested
approximate computation does not require repeated warnings. Preserve unresolved
mathematical facts as unknown rather than interpreting them as false.

---

## Ownership and Mutation

Numerical representation and ownership are separate concerns.

Specify whether NumPy data is:

- owned
- copied
- borrowed
- shared
- exposed as a view
- mutable
- read-only

A frozen dataclass does not make contained NumPy arrays immutable.

Validation at construction establishes a contract only for the validated state. Later mutation may invalidate it.

Choose copying, restricted mutation, read-only access, revalidation, or controlled mutation according to the object's invariants.

Account for aliases and views that share storage.

---

## Design Boundary

### Use Native Python When the Value Represents

- semantics
- object structure
- relationships
- configuration
- program state
- ownership
- control flow
- heterogeneous organization

Examples:

```text
str
bool
None
dict
list
tuple
set
Enum
class
dataclass
Protocol
```

### Use Exact or Symbolic Representation When the Domain Requires

- unbounded or symbolic integer values
- exact fractions
- symbolic constants
- symbolic variables
- expressions
- mathematical assumptions
- deferred numerical evaluation

Examples:

```text
int
Fraction
sp.Integer
sp.Rational
sp.Symbol
sp.Expr
```

### Use NumPy by Default When the Value Is Concrete Numerical Data

- numerical scalars
- counts
- indices
- numerical identifiers
- magnitudes
- coordinates
- numerical state
- vectors
- matrices
- tensors
- images
- signals
- measurements
- timestamps
- numerical datasets

Examples:

```text
np.uint32
np.int64
np.float32
np.float64
np.datetime64
NDArray[np.float64]
```

### Use Another Numerical Backend When Its Semantics Are Required

Examples:

```text
mpmath → arbitrary-precision approximation
Decimal → decimal arithmetic and rounding
SciPy   → higher-level numerical algorithms
```

The object's name alone does not determine its representation.

A vector may be:

- symbolic
- exact
- NumPy-backed
- arbitrary precision

Choose representation according to the contract.

---

## Rationale

This design provides:

- a **single default vocabulary for concrete numerical values**
- consistent width, precision, and signedness
- stronger and more expressive numerical type contracts
- predictable scalar-to-array behavior
- preservation of exact and symbolic structure where required
- a clear distinction between annotation and enforcement
- explicit conversion boundaries
- cleaner interoperability with native and scientific systems
- compatibility with NumPy's optimized computational ecosystem
- clear separation between meaning and representation

Most importantly, it avoids an ambiguous numerical API where Python and NumPy scalar types are mixed without a defined reason.

Instead, the default is explicit:

> **Python defines what things are and how they relate.**

> **Exact and symbolic representations preserve mathematical meaning where concrete approximation would lose information.**

> **NumPy defines the canonical concrete numerical representation by default.**

> **Alternative numerical backends are used when their arithmetic model, precision, or algorithms are part of the contract.**
