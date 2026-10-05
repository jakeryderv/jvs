# jvs.math

`jvs.math` is a math-native layer for representing and working with mathematics in Python.

The core idea is simple:

> **Model mathematics according to its mathematical meaning first, and treat symbolic, numerical, visualization, and other libraries as implementation backends.**

Instead of building directly around library-specific objects such as SymPy expressions, NumPy arrays, or plotting primitives, `jvs.math` defines its own mathematical concepts and relationships:

```text
Scalar
Variable
Expression
Set
Function
Polynomial
Point
Root
...
```

These objects should preserve the structure and semantics of the mathematics they represent, including concepts such as exactness, assumptions, domains, codomains, dimensions, and algebraic relationships.

SymPy is the current symbolic backend. NumPy, SciPy, or mpmath may support future computational needs, while the public mathematical model remains defined by `jvs.math`.

The guiding principles are:

- **Mathematical meaning before implementation details**
- **Exact mathematics where possible; approximation should be explicit**
- **Objects represent mathematical concepts**
- **Operations transform mathematical objects**
- **Mathematical structure and assumptions should be explicit**
- **Backend-specific behavior should remain behind deliberate boundaries**
- **Downstream systems consume the common math model**

## Current behavior and usage

- `Variable("x")` represents a real variable by default; equal names represent the same variable.
- `Scalar` preserves exact integers, fractions, and decimal strings. `Scalar.exact(...)` rejects approximate inputs; `Scalar.approx(...)` explicitly requests approximation. Passing a Python float directly to `Scalar` emits an `ApproximateScalarWarning` with both alternatives.
- `Scalar.is_exact` describes the current representation, rather than the provenance of previous arithmetic.
- Function domains and codomains are optional, descriptive metadata. Construction validates expression and argument structure, and evaluation checks argument count. Set membership is available explicitly through `Set.contains()`; Cartesian products use `Set.product()`.
- General derivatives leave domain and codomain unspecified unless supplied. Polynomial derivatives remain polynomials with real default sets.
- Expression equality is structural. `real_roots()` returns exact real roots with multiplicities; `limit()` defaults to a two-sided approach and accepts an explicit direction.
- `to_sympy()` provides a deliberate backend escape hatch, including SymPy lambdas for functions and SymPy polynomials for polynomials.

```python
from fractions import Fraction

from jvs.math.core import Scalar, Variable
from jvs.math.operations import derivative
from jvs.math.polynomial import Polynomial

x = Variable("x")
exact_decimal = Scalar.exact("0.1")  # Exact 1/10
exact_fraction = Scalar(Fraction(1, 3))
approximate = Scalar.approx(0.1)  # Explicit approximation, without a warning

p = Polynomial(x**2 - exact_decimal, x)
print(p(Scalar(2)))  # 39/10
print(derivative(p))  # 2*x
backend_polynomial = p.to_sympy()
```

Package tests live in [`tests/math/`](../../tests/math/). See the
[development guide](../development.md) for commands and directory conventions.

## Future direction

Other parts of `jvs` can build on this common mathematical foundation for visualization, simulation, or analysis. One possible future visualization layout is:

```text
jvs.math
├── core mathematical objects
├── symbolic / numeric operations
└── viz/
    └── visualization of mathematical objects
```

Visualization is therefore not the purpose of the math layer itself; it is one consumer of it.

The long-term goal is for code using `jvs.math` to read and behave more like the mathematics being expressed, rather than like the particular libraries used to compute it.
