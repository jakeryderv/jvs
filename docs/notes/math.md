## Math Reference Stack

| Resource | Best use |
|---|---|
| **MSC2020** | Map/taxonomy of mathematics |
| **Paul’s Online Math Notes** | Quick reference for algebra, calculus, ODEs |
| **MIT OpenCourseWare** | Deep, structured learning |
| **OpenStax Math** | Free textbooks + exercises |

## Core Progression

```text
Algebra
  ↓
Linear Algebra
  ↓
Calculus
  ↓
Multivariable Calculus
  ↓
Differential Equations
  ↓
Probability & Statistics
  ↓
Numerical Analysis
  ↓
Optimization
```

Then branch into:

```text
Dynamical Systems
Fourier Analysis
Geometry
Control Theory
Physics / Simulation
Machine Learning
Scientific Computing
```

### Simple usage model

```text
MSC2020   → where does this topic fit?
Paul's    → quick concept/formula refresher
MIT OCW   → learn the subject deeply
OpenStax  → textbook-style study + practice
```

---

The most **core and universally applicable** areas are:

| Area | Core idea | Builds into |
|---|---|---|
| **Linear algebra** | Vectors, matrices, transformations, systems | ML, physics, graphics, optimization, simulation |
| **Calculus** | Change and accumulation | gradients, motion, continuous systems |
| **Differential equations** | How systems evolve | physics, control, robotics, simulation |
| **Numerical analysis** | How computers approximate mathematical solutions | root finding, integration, ODE solvers, interpolation |
| **Optimization** | Finding best parameters or states | ML, engineering, control, fitting |
| **Probability & statistics** | Uncertainty and inference | data science, ML, estimation |

### How they connect

```text
Linear Algebra
      +
Calculus
      ↓
Differential Equations
      ↓
Numerical Analysis
      ↓
Optimization
```

Probability/statistics sits alongside this and uses **linear algebra, calculus, optimization, and numerical methods** heavily.

Other topics are mostly important techniques built on these foundations:

- **Root finding** → calculus + numerical analysis
- **Eigenvalues** → linear algebra
- **Interpolation** → linear algebra + numerical analysis
- **Fourier analysis** → linear algebra + calculus
- **Special functions** → often arise from differential equations

And the four Python libraries map well to this stack:

```text
SymPy   → symbolic formulation/derivation
NumPy   → vectors, matrices, arrays, computation
SciPy   → numerical solvers and algorithms
mpmath  → high-precision/reference computation
```
