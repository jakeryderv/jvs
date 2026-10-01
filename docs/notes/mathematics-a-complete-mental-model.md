# Mathematics: A Complete Mental Model

**Mental model:** **Language** describes **Objects**. Objects have **Properties**, are linked by **Relations**, transformed by **Operations/Maps**, and grouped into **Structures**. **Statements** make claims about all of these, and **Proofs** justify them. A **Foundation** determines what all of these are ultimately made of.

---

## The Layered Picture

```
Logic                    → the rules of reasoning
  └─ Foundations         → set theory / type theory / category theory
       │                    (each builds sections 1–8 its own way)
       └─ Components     → Model theory:   1 Objects · 2 Relations · 3 Operations
            │                               4 Structures · 5 Properties
            │              Proof theory:   6 Statements · 7 Proofs
            │              Formal syntax:  8 Language
            └─ Mother structures (Bourbaki) → algebraic · order · topological
                 └─ Fields of math (MSC2020) → linear algebra, analysis, geometry, …
```

## Two Perpendicular Axes

| Question | Answered By | Axis |
| --- | --- | --- |
| **What is everything ultimately *made of*?** | Set theory, type theory, category theory | Foundations |
| **What *components* does math have?** | Model theory, proof theory, formal syntax | Sections 1–8 |

## The Foundations

| Framework | Core Idea | What's Primitive |
| --- | --- | --- |
| **Set Theory (ZFC)** | Everything is a set, and even numbers are built from sets | Membership (∈) |
| **Type Theory** | Every object has exactly one type (`x : ℝ`) | Types and terms |
| **Category Theory** | Objects are defined by their relationships to other objects | Objects and morphisms (maps) |

## The Component Frameworks

| Framework | Covers | What It Formally Defines |
| --- | --- | --- |
| **Model Theory** | Sections 1–5 | A *structure* = domain (objects) + relations + functions, with predicates interpreted on it |
| **Proof Theory** | Sections 6–7 | Formulas, axioms, theorems, and the inference rules connecting them |
| **Formal Syntax** (first-order logic) | Section 8 | Signature, symbols, terms, formulas, free/bound variables |

---

## 1. Objects

*The "things" math is about.*

**Basis:** Model theory's **domain of discourse**.

**Foundations:** ZFC: sets · Type theory: terms of a type (`x : ℝ`) · Category theory: objects, defined by their arrows

| Family | Description | Members |
| --- | --- | --- |
| **Numbers** | Quantities and magnitudes | natural, integer, rational, real, complex, quaternion, octonion, p-adic, ordinal, cardinal, infinitesimal |
| **Collections** | Groupings of other objects | set, class, multiset, tuple, ordered pair, sequence, list, family, partition |
| **Linear / Algebraic** | Objects manipulated by algebraic rules | scalar, vector, matrix, tensor, polynomial, form, ideal, coset |
| **Functional** | Rules assigning inputs to outputs | function, map, operator, functional, transformation, morphism, functor |
| **Geometric** | Shapes and spatial entities | point, line, plane, angle, curve, surface, polygon, polytope, manifold, space |
| **Discrete** | Countable, separated arrangements | graph, tree, permutation, combination, word/string, lattice point |
| **Analytic** | Objects arising from limits and change | limit, series, derivative, integral, differential, measure, distribution |
| **Probabilistic** | Objects modeling chance | event, outcome, random variable, probability distribution, stochastic process |
| **Logical** | Objects of reasoning itself | proposition, truth value, formula, model |

## 2. Relations

*How objects connect or compare.*

**Basis:** Model theory's **relation symbols**, plus **set theory** and **order theory**.

**Foundations:** ZFC: sets of ordered pairs · Type theory: predicates (`A → B → Prop`) · Category theory: subobjects / spans

| Group | Description | Members |
| --- | --- | --- |
| **Identity / Comparison** | Sameness, exact or in some respect | equality, equivalence, congruence, similarity, isomorphism |
| **Order** | Ranking or precedence | ≤, \<, preorder, partial order, total order, well-order, dominance |
| **Set-based** | Containment and overlap | membership (∈), subset (⊆), disjointness |
| **Number-theoretic** | Relationships between integers | divisibility, coprimality, congruence mod n |
| **Geometric** | Spatial arrangement | parallel, perpendicular/orthogonal, incidence, tangency, intersection |
| **Graph** | Links within networks | adjacency, connectivity, reachability |
| **Properties of relations** | Classifying traits of relations themselves | reflexive, irreflexive, symmetric, antisymmetric, asymmetric, transitive, total |

## 3. Operations & Maps

*How objects combine or transform.*

**Basis:** Model theory's **function symbols**; **universal algebra** (Birkhoff) for operations and arity; **category theory** for morphism types.

**Foundations:** ZFC: sets of input–output pairs · Type theory: function types (`A → B`), primitive · Category theory: morphisms, primitive

| Group | Description | Members |
| --- | --- | --- |
| **By arity** | Number of inputs taken | nullary (constants), unary, binary, ternary, n-ary |
| **Unary** | Act on a single object | negation, inverse, transpose, complement, absolute value, norm, conjugate, derivative, integral |
| **Binary** | Combine two objects | addition, multiplication, subtraction, division, exponentiation, composition, union, intersection, dot product, cross product, tensor product, convolution |
| **Operation properties** | Rules an operation may obey | associative, commutative, distributive, idempotent, has identity, has inverses, closed |
| **Map types (by behavior)** | How inputs correspond to outputs | injective (one-to-one), surjective (onto), bijective (both) |
| **Map types (by structure preserved)** | What a map keeps intact | homomorphism, isomorphism, endomorphism, automorphism, embedding, homeomorphism, diffeomorphism, isometry, linear map |

## 4. Structures

*Objects bundled with operations and relations.*

**Basis:** Model theory's **σ-structures**; **Bourbaki's mother structures**; **universal algebra** for the algebraic hierarchy; **category theory** for meta-structures.

**Foundations:** ZFC: tuples of (set, operations, relations) · Type theory: records / typeclasses · Category theory: objects in a category of structures

| Type | Description | Members |
| --- | --- | --- |
| **Algebraic** ⭐ | Defined by operations ("combine") | magma → semigroup → monoid → group → abelian group; ring → commutative ring → integral domain → field; module, vector space, algebra, Lie algebra |
| **Order** ⭐ | Defined by ordering ("compare") | preorder, poset, total order, well-order, lattice, Boolean algebra |
| **Topological / Metric** ⭐ | Defined by nearness or distance | topological space → metric space → normed space → inner product space; Banach space, Hilbert space |
| **Geometric** | Spaces with shape, curvature, or coordinates | manifold, smooth manifold, Riemannian manifold, affine space, projective space |
| **Measure** | Defined by size, volume, or probability | σ-algebra, measure space, probability space |
| **Combinatorial** | Finite arrangements with rules | graph, hypergraph, matroid, design, simplicial complex |
| **Hybrid** | Combine two or more of the types above | ordered field, topological group, Lie group, Banach algebra, topological vector space |
| **Meta** | Structures of structures | category, functor category, topos |

⭐ = Bourbaki's three mother structures

## 5. Properties

*Traits an object may or may not have.*

**Basis:** First-order logic's **predicates**, interpreted on objects in model theory.

**Foundations:** ZFC: subsets · Type theory: propositions (`Prop`) · Category theory: subobjects

| Domain | Description | Members |
| --- | --- | --- |
| **Size** | How many or how large | finite, infinite, countable, uncountable, bounded, unbounded |
| **Analytic** | Behavior under limits and change | continuous, differentiable, smooth, integrable, convergent, monotonic |
| **Topological** | Shape-related traits that survive stretching | open, closed, compact, connected, dense, complete |
| **Algebraic** | Behavior under operations | invertible, singular, linear, nilpotent, prime, irreducible, commutative |
| **Geometric** | Shape and spatial traits | convex, symmetric, orthogonal, regular, orientable |
| **Logical** | Status of statements or systems | true, false, consistent, decidable, provable, independent |

## 6. Statements

*Claims made about objects.*

**Basis:** **Proof theory** and **first-order logic** fix the logical forms. The status terms are long-standing writing convention.

**Foundations:** ZFC: first-order formulas about sets · Type theory: types (*propositions as types*) · Category theory: internal logic of a topos

| Group | Description | Members |
| --- | --- | --- |
| **By status** | Role in the body of math | axiom, postulate, definition, lemma, proposition, theorem, corollary, conjecture, hypothesis, claim, paradox |
| **By logical form** | Grammatical shape of the claim | atomic, negation, conjunction, disjunction, conditional, biconditional, universal (∀), existential (∃), uniqueness (∃!) |
| **By content** | What kind of fact is asserted | identity, equation, inequality, existence claim, classification, characterization, impossibility result |

## 7. Proofs

*How statements are justified.*

**Basis:** **Proof theory**, especially Gentzen's **natural deduction**, formalizes the rules. The techniques are standard conventions, taught in texts like Velleman's *How to Prove It*.

**Foundations:** ZFC: derivations in first-order logic, *outside* the theory · Type theory: terms of a proposition's type (*Curry–Howard*), *inside* the theory · Category theory: morphisms in the internal logic

| Group | Description | Members |
| --- | --- | --- |
| **Techniques** | Overall argument strategies | direct, contrapositive, contradiction, induction (weak, strong, structural, transfinite), construction, non-constructive existence, counterexample, exhaustion/cases, diagonalization, pigeonhole, probabilistic, combinatorial/double counting, infinite descent |
| **Logical rules** | Individual valid reasoning steps | modus ponens, modus tollens, universal instantiation/generalization, existential instantiation/generalization |

## 8. Language

*How all of the above is written.*

**Basis:** **Formal syntax of first-order logic**. The roles are informal teaching vocabulary.

**Foundations:** ZFC: first-order logic plus the symbol `∈` · Type theory: the type theory is its own language · Category theory: diagrams and the internal language

| Group | Description | Members |
| --- | --- | --- |
| **Symbols** | Basic written units | constants, variables, function symbols, relation symbols, connectives, quantifiers, punctuation |
| **Constructions** | Combinations of symbols | term, expression, equation, inequality, formula, sentence |
| **Roles** | Position a symbol fills in an expression | operand, argument, coefficient, term, factor, index, parameter, exponent, base |
| **Variable status** | How a variable behaves | free, bound, dummy, unknown, parameter |

---

## Related Full-Scale Layouts of Math

| Resource | Organizes Math By |
| --- | --- |
| **MSC2020** (Mathematics Subject Classification) | Fields, such as algebra, analysis, and geometry |
| **Bourbaki, *Éléments de mathématique*** | Structures, built up from sets |
| **Lean's Mathlib** | Formal type and structure hierarchies |
| **nLab** | Concepts and their relationships, from a category-theory view |
| **OntoMathPRO** | A formal ontology of mathematical concepts |

---

## TL;DR

- **Full stack:** logic → foundation → components (1–8) → mother structures → fields.
- **Foundations** (set, type, category theory) say what everything is *made of*. **Component frameworks** (model theory, proof theory, formal syntax) say what math *contains*.
- **Semantics vs. syntax:** sections 1–5 concern objects themselves, while sections 6–8 concern claims, justifications, and notation about them.
- **Key contrast:** ZFC builds maps and proofs *from sets*. Type theory makes functions primitive and proofs into objects. Category theory makes maps primitive and defines objects only through them.

---

# Appendix: "Real Math in Programming" — A Python Library Based on This Model

**Goal:** A library whose *interface* follows this mental model, with existing libraries acting as *engines underneath*. It serves both **learning** (you build and understand the core) and **practical use** (one consistent API instead of many sets of docs).

## Approach: Your Own Core + Library Backends

| Layer | Who Writes It | Role |
| --- | --- | --- |
| **Interface** | You | Classes and names that follow sections 1–8, written how you think |
| **Core abstractions** | You | `Structure`, `Element`, `Map`, `Relation`, properties, axioms |
| **Adapters** | You (thin) | Translate your objects to and from each backend |
| **Backends** | Existing libraries | SymPy, NumPy, SciPy, mpmath, Sage, etc. do the actual computation |

> **Prior art:** **SageMath** already wraps many of these libraries under a math-first object model (**Parent / Element / Category / Morphism / Coercion**). Study its design, and optionally use it as one backend.

## Mapping the Mental Model to Python

| # | Section | Python Construct | Backing Libraries |
| --- | --- | --- | --- |
| **Foundation** | Type theory | Type hints, `Protocol`, `Generic[T]` | `typing`, `mypy` |
| **1** | Objects | Classes; each element holds a reference to its parent structure | NumPy, SymPy, mpmath |
| **2** | Relations | Operator overloads (`__eq__`, `__le__`, `__contains__`) plus explicit `Relation` classes | SymPy `Relational` |
| **3** | Operations & Maps | Dunder methods; a `Map` class with `domain`, `codomain`, `__call__`, composition | — |
| **4** | Structures | Abstract base classes forming the hierarchy (`Group → Ring → Field`) | `abc` |
| **5** | Properties | Predicate methods (`is_invertible()`) or a cached assumptions system | SymPy assumptions |
| **6** | Statements | Symbolic expressions and relations | SymPy |
| **7** | Proofs | Python can't verify proofs: check axioms with property-based tests, or bridge to Lean | `hypothesis`, Lean |
| **8** | Language | Operator overloading and pretty-printing | SymPy printing, LaTeX |

## Core Design Decisions

| Decision | Recommendation |
| --- | --- |
| **Structure-first** | Elements know their parent, so `3` in ℤ and `3` in ℤ/5ℤ behave differently |
| **Swappable backends** | Exact (SymPy), fast float (NumPy), arbitrary precision (mpmath), same math object |
| **Axioms as data** | Declare axioms on each structure; auto-generate `hypothesis` tests to verify instances |
| **Maps as first-class objects** | Domain, codomain, composition, inverses, and "structure preserved" become checkable |
| **Distinct sameness** | Separate identity (`is`), equality (`==`), and isomorphism (`.is_isomorphic()`) |
| **Coercion rules** | Define how elements move between structures (ℤ → ℚ → ℝ → ℂ) |
| **Grow incrementally** | Start with one area (e.g. linear algebra), then generalize the core as patterns repeat |

## Relevant Libraries

| Library | Role |
| --- | --- |
| **SymPy** | Exact symbolic math, expressions, assumptions |
| **NumPy** | Fast numerical arrays and linear algebra |
| **SciPy** | Numerical analysis, optimization, sparse matrices, special functions |
| **mpmath** | Arbitrary-precision arithmetic |
| **SageMath** | Unified algebra/number theory/geometry system; design reference and optional backend |
| **python-flint** | Fast exact number theory and polynomials |
| **NetworkX** | Graphs and combinatorial structures |
| **galgebra / clifford** | Geometric and Clifford algebra |
| **hypothesis** | Property-based testing of axioms and properties |
| **mypy** | Static type checking for the type-theory layer |