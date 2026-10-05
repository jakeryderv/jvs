# Math-Native Interactive Visualization System

## Concept

Build a Python visualization system where **mathematical and data objects are the source of truth**, and visualizations are persistent objects that describe how those objects should be viewed.

Instead of immediately drawing with commands such as:

```python
plt.plot(x, y)
plt.show()
```

the system would separate:

1. **Mathematical objects**
2. **Visualization specifications**
3. **Rendering**
4. **Interactive viewing or export**

This allows the same visualization to be opened interactively, embedded in a notebook, exported as a static figure, or rendered as an animation without redefining it.

---

## Core Idea

```text
Mathematical Model
        │
        ▼
Visualization Object
        │
        ▼
Renderer / Viewer
        │
        ├── Interactive GUI
        ├── Notebook
        ├── Static Figure
        ├── Animation
        └── Web
```

The visualization itself is therefore a reusable object rather than a one-time plotting operation.

---

# 1. Mathematical Layer

The mathematical layer represents the actual objects being studied.

Possible objects include:

```text
Function
Equation
Variable
Parameter
Point
Vector
Matrix
Curve
Surface
Scalar Field
Vector Field
Trajectory
Dataset
Geometry
```

Python's scientific ecosystem could provide the underlying capabilities:

| Library | Primary role |
|---|---|
| NumPy | Arrays, vectors, matrices, numerical evaluation |
| SymPy | Symbolic expressions and exact mathematics |
| SciPy | Numerical algorithms, optimization, interpolation, integration, ODEs |
| mpmath | Arbitrary-precision numerical mathematics |

For example:

```python
x = Variable("x")
a = Parameter("a", 1)

f = sin(a * x)
```

`f` remains a mathematical object independent of how it is eventually visualized.

---

# 2. Visualization Layer

A visualization object describes **how mathematical objects should be viewed**.

For example:

```python
viz = Plot(f)

viz.range(x, -10, 10)
viz.slider(a, -5, 5)
```

Creating `viz` does not necessarily display anything.

It is a persistent specification containing information such as:

```text
Visualization
├── mathematical objects
├── coordinate system
├── ranges
├── axes
├── labels
├── camera/view
├── styles
├── parameters
├── interactions
├── animation state
└── timeline
```

This separates the mathematical meaning from the presentation.

---

# 3. Scene as the General Visualization Primitive

A general `Scene` abstraction could sit at the center of the visualization system.

```python
scene = Scene()

scene.add(surface)
scene.add(vector_field)
scene.add(trajectory)
```

A scene could contain:

```text
Scene
├── Objects
├── Coordinate system
├── Camera
├── Axes
├── Lighting
├── Style
├── Interaction
└── Timeline
```

Specialized visualization types could be convenient wrappers around scenes:

```text
Scene
├── Plot2D
├── Plot3D
├── FieldView
├── MatrixView
├── GeometryView
└── SimulationView
```

This would allow the system to grow beyond conventional plotting into geometry, physics, simulation, and 3D visualization.

---

# 4. Renderer Layer

Visualization objects should remain independent of any particular graphics library.

A renderer translates the visualization specification into an actual output.

```text
Visualization
      │
      ▼
Renderer
├── Matplotlib
├── GUI
├── Web
├── Notebook
└── Headless renderer
```

For example:

```python
render(viz, backend="matplotlib")
```

or:

```python
viz.show()
```

The system could choose an appropriate renderer automatically.

Matplotlib would therefore be a **backend**, rather than defining the architecture of the library.

---

# 5. Visualization vs Figure

It is useful to distinguish a visualization specification from its rendered output.

## Visualization

A backend-independent description.

```python
viz = Plot(sin(x))
```

## Figure

A concrete rendered artifact.

```python
fig = viz.render()
fig.save("plot.svg")
```

Conceptually:

```text
Visualization = what should be shown
Renderer      = how it should be drawn
Figure        = resulting rendered artifact
Viewer        = environment for interacting with it
```

This avoids coupling the system to concepts such as Matplotlib's `Figure`.

---

# 6. Interactive Viewing

A visualization could be launched into an interactive viewer:

```python
viz.show()
```

The viewer could support:

```text
Pan
Zoom
Rotate
Inspect coordinates
Toggle objects
Change parameters
Move points
Play/pause animation
Step through time
Modify camera
Change visualization modes
```

This would create a Desmos-like exploratory environment, but centered around general mathematical objects rather than only equations.

---

# 7. Reactive Parameters

Parameters should be first-class objects.

```python
a = Parameter("a", 1, range=(-5, 5))

f = sin(a * x)

viz = Plot(f)
viz.slider(a)
```

Internally:

```text
Parameter a changes
        │
        ▼
Dependent mathematics invalidated
        │
        ▼
Numerical representation recomputed
        │
        ▼
Affected visualization updated
```

This dependency system is what would provide the immediate interactive behavior associated with tools such as Desmos.

---

# 8. Static, Interactive, and Dynamic Are Rendering Modes

Static plots, interactive viewers, and animations should not require completely separate visualization definitions.

The same visualization could support:

```python
viz.show()
viz.save("graph.svg")
viz.save("graph.png")
viz.animate("graph.mp4")
```

For an animated visualization:

```text
GUI
→ timeline + playback + stepping

Notebook
→ interactive controls

MP4/GIF
→ rendered sequence of frames

PNG/SVG
→ current state or selected frame
```

Therefore:

> Static, interactive, and animated outputs are different renderings of the same visualization model.

---

# 9. Multiple Views of the Same Mathematics

One mathematical object could simultaneously have multiple visualization objects.

For example:

```text
Vector Field
     │
     ├── Arrow View
     ├── Streamline View
     ├── Magnitude Heatmap
     └── Animated Particle View
```

The underlying field remains the same.

This establishes a useful separation:

```text
Object
  │
  ├── View A
  ├── View B
  └── View C
```

A visualization is therefore a **view of mathematics**, not the mathematics itself.

---

# 10. Example User Experience

A simple API might eventually look like:

```python
x = Variable("x")
a = Parameter("a", 1)

f = sin(a * x)

viz = Plot(f)

viz.range(x, -10, 10)
viz.slider(a, -5, 5)
```

Interactive use:

```python
viz.show()
```

Static output:

```python
viz.save("sine.svg")
```

Animation:

```python
viz.animate("sine.mp4")
```

The mathematical definition is written once.

---

# 11. Possible Architecture

```text
┌──────────────────────────────────────┐
│          Mathematical Layer          │
│                                      │
│ Functions, vectors, matrices, fields │
│ curves, surfaces, trajectories, etc. │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Visualization Object Model    │
│                                      │
│ scenes, views, axes, cameras,        │
│ styles, parameters, interactions     │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             Renderers                │
│                                      │
│ Matplotlib / GUI / Web / Notebook    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│               Outputs                │
│                                      │
│ GUI / PNG / SVG / PDF / HTML / video│
└──────────────────────────────────────┘
```

---

# 12. Initial Implementation Scope

The first version should remain small.

A useful vertical slice would be:

```text
SymPy expression
      ↓
NumPy numerical evaluation
      ↓
Visualization object
      ↓
Matplotlib renderer
      ↓
Interactive viewer
```

Initial capabilities:

1. Define a symbolic function.
2. Define one or more parameters.
3. Create a `Plot` visualization object.
4. Evaluate the function numerically using NumPy.
5. Render it using Matplotlib.
6. Change parameters interactively.
7. Automatically recompute the plot.
8. Save the same visualization as PNG or SVG.

This would test the important architecture without requiring a custom rendering engine.

---

# Long-Term Direction

The goal is not simply to recreate Desmos in Python.

The larger idea is:

> **A math-native visualization system where mathematical objects exist independently of their visual representations, visualization objects define persistent views of those objects, and interchangeable renderers determine how those views are explored or exported.**

Desmos-like graphing would become one use case.

The same architecture could eventually support:

```text
Analytic mathematics
Linear algebra
Geometry
Calculus
Differential equations
Vector fields
Scientific data
3D geometry
Physics simulations
Trajectories
Time-dependent systems
Engineering visualization
```

The central principle remains:

```text
mathematics → visualization → rendering
```

rather than:

```text
plotting commands → pixels
```
