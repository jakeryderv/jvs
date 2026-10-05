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

Underlying libraries such as SymPy, NumPy, SciPy, or mpmath may provide the computational machinery, but they should not define the public mathematical model.

The guiding principles are:

- **Mathematical meaning before implementation details**
- **Exact mathematics where possible; approximation should be explicit**
- **Objects represent mathematical concepts**
- **Operations transform mathematical objects**
- **Mathematical structure and assumptions should be explicit**
- **Backend-specific behavior should remain behind deliberate boundaries**
- **Downstream systems consume the common math model**

This allows other parts of `jvs` to build on the same mathematical foundation, for example:

```text
jvs.math
├── core mathematical objects
├── symbolic / numeric operations
└── viz/
    └── visualization of mathematical objects
```

Visualization is therefore not the purpose of the math layer itself; it is one consumer of it.

The long-term goal is for code using `jvs.math` to read and behave more like the mathematics being expressed, rather than like the particular libraries used to compute it.
