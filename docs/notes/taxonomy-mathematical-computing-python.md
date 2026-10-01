## NumPy

**Purpose:** Core numerical computing in Python. It provides efficient multidimensional arrays and the basic operations needed for numerical mathematics.

**Best for:**
- Vectors, matrices, tensors
- Linear algebra
- Elementwise math
- Numerical geometry
- Simulation state
- Bulk floating-point computation

| Built around NumPy | What it adds |
|---|---|
| SciPy | Scientific numerical methods |
| pandas | Labeled tabular data |
| TensorLy | Tensor algebra/decompositions |
| SfePy | Finite-element methods |
| many simulation libraries | Numerical state, geometry, transforms |

---

## SciPy

**Purpose:** Higher-level numerical methods built on NumPy.

**Best for:**
- Optimization
- Numerical integration
- ODE solving
- Root finding
- Interpolation
- Sparse linear algebra
- Signal processing
- Spatial algorithms
- Special functions

| Built around SciPy / commonly uses it | What it adds |
|---|---|
| PyDy | Numerical mechanics/dynamics |
| SfePy | Finite-element PDE solving |
| pyodesys | ODE-system solving |
| pyneqsys | Nonlinear equation solving |
| statsmodels | Statistical modeling |

---

## SymPy

**Purpose:** Symbolic mathematics. It lets Python manipulate mathematical expressions directly instead of immediately evaluating everything numerically.

**Best for:**
- Algebra
- Exact arithmetic
- Symbolic calculus
- Equation solving
- Simplification/factoring
- Symbolic matrices
- Jacobians
- Deriving formulas

| Built around SymPy | What it adds |
|---|---|
| PyDy | Symbolic multibody mechanics |
| EinsteinPy | Relativity/tensor calculus |
| galgebra | Geometric algebra |
| Devito | Symbolic PDEs → generated numerical code |
| Lcapy | Circuit analysis |
| pyneqsys | Symbolic nonlinear systems |
| pyodesys | Symbolic ODE definitions |
| unyt | Unit algebra |

---

## mpmath

**Purpose:** Arbitrary-precision numerical mathematics.

**Best for:**
- High-precision arithmetic
- Special functions
- Numerical integration
- Root finding
- Precision-sensitive calculations
- Reference calculations for validating other methods

| Built around / uses mpmath | What it adds |
|---|---|
| SymPy | High-precision numerical evaluation |
| research math code | Precision beyond `float64` |
| validation tooling | Accurate reference values |

---

# Libraries built across the math stack

These often combine symbolic representation with numerical execution.

| Library | Built on / uses | Purpose |
|---|---|---|
| **PyDy** | SymPy + NumPy + SciPy | Mechanics, equations of motion, simulation |
| **EinsteinPy** | SymPy + NumPy | Tensor calculus and general relativity |
| **Devito** | Symbolic math + NumPy-style arrays | PDE DSL and generated high-performance kernels |
| **SfePy** | NumPy + SciPy | Finite-element PDE solving |
| **pyodesys** | SymPy + SciPy | Symbolic ODE definitions → numerical integration |
| **pyneqsys** | SymPy + SciPy | Symbolic nonlinear systems → numerical solving |
| **galgebra** | SymPy | Geometric algebra |
| **CVXPY** | NumPy + numerical solvers | Mathematical optimization DSL |
| **FEniCS** | symbolic DSL + numerical linear algebra | Finite-element PDE formulation |
| **TensorLy** | NumPy and other tensor backends | Tensor algebra/decompositions |
| **unyt** | NumPy + SymPy | Quantities with physical units |
| **Pint** | Python numeric ecosystem | Unit-aware quantities |
| **python-flint** | FLINT | Exact algebra, polynomials, number theory |

---

# Visualization / plotting stack

This is the layer for **turning mathematical or numerical objects into visual representations**.

## Matplotlib

**Purpose:** The foundational general-purpose plotting library for scientific Python.

**Best for:**
- Line/scatter/bar plots
- Mathematical curves
- 2D scientific figures
- Basic 3D plotting
- Publication-style figures
- Fine-grained control over axes, labels, layouts

Commonly works directly with:
- NumPy
- SciPy results
- SymPy numerical output
- pandas
- most scientific libraries

Think of it as the **NumPy of plotting**: many other visualization libraries build on it.

---

## Seaborn

**Purpose:** Higher-level statistical visualization built on Matplotlib.

**Best for:**
- Distributions
- Correlation plots
- Statistical summaries
- Heatmaps
- grouped/category plots
- cleaner defaults with less setup

**Built on:**
- Matplotlib
- pandas / NumPy

Typical relationship:

```text
Seaborn
   ↓
Matplotlib
   ↓
rendering backend
```

---

## Plotly

**Purpose:** Interactive plotting and browser-based visualization.

**Best for:**
- Interactive zoom/pan
- Hover tooltips
- Dashboards
- 3D plots
- Web output
- Exploratory scientific visualization

Works well with:
- NumPy
- pandas
- SciPy
- scientific datasets

Unlike Seaborn, Plotly is **not primarily built on Matplotlib**. It has its own rendering model.

---

## Manim

**Purpose:** Mathematical animation rather than ordinary plotting.

**Best for:**
- Visualizing equations
- Geometry
- Linear algebra transformations
- Calculus concepts
- Animated mathematical explanations
- Educational videos

For example, instead of simply plotting:

\[
y=x^2
\]

Manim can animate:
- the equation appearing
- axes being drawn
- the curve forming
- tangent lines moving
- transformations occurring

It is closer to a **mathematical animation engine** than a plotting library.

---

## Bokeh

**Purpose:** Interactive browser-based visualization.

**Best for:**
- Interactive scientific plots
- dashboards
- streaming data
- web applications

Similar space to Plotly, though with a somewhat different API and ecosystem.

---

## Altair

**Purpose:** Declarative statistical visualization.

You describe:

> x = time, y = temperature, color = category

rather than manually constructing all plot elements.

**Best for:**
- Statistical/data visualization
- concise declarative charts
- exploratory analysis

Built around the **Vega-Lite** visualization grammar.

---

## PyVista

**Purpose:** Scientific 3D visualization.

**Best for:**
- Meshes
- point clouds
- volumetric data
- finite-element results
- simulation geometry
- vector/scalar fields

Built on:
- VTK

This is much stronger than Matplotlib's basic 3D support for actual scientific geometry.

---

## Mayavi

**Purpose:** Scientific 3D visualization.

**Best for:**
- Vector fields
- surfaces
- volumetric scientific data
- simulation results

Also built around VTK.

---

## VisPy

**Purpose:** High-performance GPU-accelerated scientific visualization.

**Best for:**
- Large datasets
- fast interactive plots
- custom OpenGL/GPU rendering
- large point clouds or images

---

## SymPy plotting

SymPy itself also has plotting helpers:

```python
from sympy import symbols, sin
from sympy.plotting import plot

x = symbols("x")
plot(sin(x))
```

This is convenient for quick symbolic math visualization, but it generally delegates rendering to plotting backends rather than replacing Matplotlib/Plotly-class tools.

---

# Combined picture

```text
                     MATHEMATICS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
     symbolic          numerical       high precision
        │                 │                 │
      SymPy             NumPy             mpmath
        │                 │
        │               SciPy
        │                 │
        └──────────┬──────┘
                   │
       domain-specific math
                   │
   ┌────────┬──────┼──────┬─────────┐
   │        │      │      │         │
  PyDy   Devito  CVXPY  FEniCS   galgebra
   │
   └───────────────────────┐
                           │
                     VISUALIZATION
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
     general           interactive         animation
        │                  │                  │
   Matplotlib          Plotly/Bokeh         Manim
        │
      Seaborn

        scientific 3D
             │
      PyVista / Mayavi
```

## Simplest taxonomy

| Layer | Main tools |
|---|---|
| Numerical math | NumPy |
| Numerical methods | SciPy |
| Symbolic math | SymPy |
| High precision | mpmath |
| Domain-specific math | PyDy, Devito, FEniCS, CVXPY, galgebra, etc. |
| General plotting | Matplotlib |
| Statistical plotting | Seaborn |
| Interactive plotting | Plotly, Bokeh, Altair |
| Mathematical animation | Manim |
| Scientific 3D visualization | PyVista, Mayavi, VisPy |

So if you're thinking of this as a **“taxonomy of mathematical computing in Python,”** visualization is naturally another layer alongside the math engines: the math libraries represent/solve the problem, while visualization libraries turn those mathematical objects and results into something you can inspect.
