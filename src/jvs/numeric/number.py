"""Finite number sets with explicit evidence and set-like syntax.

``Set.query(value)`` preserves uncertainty; ``value in Set`` requires a decision.
Set classes describe domains and cannot be instantiated as numeric values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Never


class IndeterminateError(ValueError):
    """Available evidence cannot decide a mathematical question."""


class IndeterminateMembershipError(IndeterminateError):
    """Membership cannot be established or refuted."""


class IndeterminateRelationError(IndeterminateError):
    """A set relationship cannot be established or refuted."""


class ClassificationConflictError(ValueError):
    """Rules supply contradictory established facts."""


class Truth(Enum):
    """Three-valued knowledge; unknown is never silently false."""

    FALSE = 0
    TRUE = 1
    UNKNOWN = 2

    def __bool__(self) -> bool:
        if self is Truth.UNKNOWN:
            raise IndeterminateError("Cannot interpret UNKNOWN as a boolean")
        return self is Truth.TRUE


@dataclass(frozen=True, slots=True)
class Classification:
    """A decision with evidence or a reason it remains unresolved."""

    state: Truth
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.state, Truth) or not isinstance(self.reason, str):
            raise TypeError("Classification requires a Truth state and a reason string")
        if not self.reason:
            raise ValueError("Classification requires a nonempty reason")

    def __bool__(self) -> bool:
        if self.state is Truth.UNKNOWN:
            raise IndeterminateError(self.reason)
        return bool(self.state)


type MembershipTest[T] = Callable[[T], Classification | bool | None]
type Facts = dict[NumberSetMeta, Classification]

# Inclusion edges carry a flag for explicitly established proper inclusion.
_PARENTS: dict[NumberSetMeta, dict[NumberSetMeta, bool]] = {}
_EQUIVALENT: dict[NumberSetMeta, NumberSetMeta] = {}
_INTERSECTIONS: dict[NumberSetMeta, tuple[NumberSetMeta, ...]] = {}
_DISJOINT: list[tuple[NumberSetMeta, NumberSetMeta]] = []
_PARTITIONS: list[tuple[NumberSetMeta, NumberSetMeta, NumberSetMeta]] = []
_TESTS: list[tuple[NumberSetMeta, type, MembershipTest[Any]]] = []


class NumberSetMeta(type):
    """Runtime set syntax independent of backend dtype inheritance."""

    def __new__(
        mcls,
        name: str,
        bases: tuple[type, ...],
        namespace: dict[str, Any],
        **kwargs: Any,
    ) -> NumberSetMeta:
        cls = super().__new__(mcls, name, bases, namespace, **kwargs)
        _PARENTS[cls] = {
            base: False for base in bases if isinstance(base, NumberSetMeta)
        }
        return cls

    def __call__(cls, *args: Any, **kwargs: Any) -> Never:
        raise TypeError(f"{cls!r} is a mathematical set, not a scalar constructor")

    def __repr__(cls) -> str:
        return cls.__dict__.get("symbol", cls.__name__)

    def query(cls, value: object) -> Classification:
        """Classify without coercion or collapsing unknown results."""
        return classify(value)[cls]

    def contains(cls, value: object) -> bool:
        """Return established membership, raising when it is unknown."""
        result = cls.query(value)
        if result.state is Truth.UNKNOWN:
            raise IndeterminateMembershipError(result.reason)
        return bool(result)

    def __contains__(cls, value: object) -> bool:
        return cls.contains(value)

    def subset_of(cls, other: NumberSetMeta, *, proper: bool = False) -> Classification:
        """Query inclusion or proper inclusion with evidence."""
        _require_set(other)
        included, strict = _path(cls, other)
        reverse, _ = _path(other, cls)
        if included and (not proper or strict):
            return Classification(
                Truth.TRUE, f"Declared inclusion: {cls!r} <= {other!r}"
            )
        if proper and included and reverse:
            return Classification(Truth.FALSE, "Equivalent sets are not proper subsets")
        for label, facts in _WITNESSES:
            inferred = _infer(_evidence(facts, f"Known witness {label}"))
            if (
                _state(inferred, cls) is Truth.TRUE
                and _state(inferred, other) is Truth.FALSE
            ):
                return Classification(
                    Truth.FALSE,
                    f"Counterexample {label} belongs to {cls!r}, not {other!r}",
                )
            if (
                included
                and proper
                and _state(inferred, other) is Truth.TRUE
                and _state(inferred, cls) is Truth.FALSE
            ):
                return Classification(
                    Truth.TRUE, f"Inclusion with witness {label} outside {cls!r}"
                )
        if _path(other, cls)[1]:
            return Classification(
                Truth.FALSE, f"{other!r} is a proper subset of {cls!r}"
            )
        return Classification(
            Truth.UNKNOWN, f"Insufficient evidence relating {cls!r} and {other!r}"
        )

    def _compare(cls, other: NumberSetMeta, *, proper: bool = False) -> bool:
        result = cls.subset_of(other, proper=proper)
        if result.state is Truth.UNKNOWN:
            raise IndeterminateRelationError(result.reason)
        return bool(result)

    def __le__(cls, other: NumberSetMeta) -> bool:
        return cls._compare(other)

    def __lt__(cls, other: NumberSetMeta) -> bool:
        return cls._compare(other, proper=True)

    def __ge__(cls, other: NumberSetMeta) -> bool:
        _require_set(other)
        return other._compare(cls)

    def __gt__(cls, other: NumberSetMeta) -> bool:
        _require_set(other)
        return other._compare(cls, proper=True)

    def register_member_test[T](
        cls, test: MembershipTest[T], *, value_type: type[T]
    ) -> MembershipTest[T]:
        """Register a rule for an explicit input type.

        Return True/False for established facts, None for unknown, or a
        Classification with specific evidence. All applicable rules contribute;
        exceptions and conflicting facts are errors. Return the original rule.
        """
        if not isinstance(value_type, type) or not callable(test):
            raise TypeError("A membership rule requires an input type and a callable")
        _TESTS.append((cls, value_type, test))
        return test


def _require_set(value: object) -> None:
    if not isinstance(value, NumberSetMeta):
        raise TypeError("Expected a mathematical number set")


def _canonical(cls: NumberSetMeta) -> NumberSetMeta:
    return _EQUIVALENT.get(cls, cls)


def _path(start: NumberSetMeta, end: NumberSetMeta) -> tuple[bool, bool]:
    """Close inclusions under transitivity and intersection introduction."""
    reachable = {_canonical(start): False}

    def add(target: NumberSetMeta, strict: bool) -> bool:
        target = _canonical(target)
        if target not in reachable or (strict and not reachable[target]):
            reachable[target] = strict
            return True
        return False

    changed = True
    while changed:
        changed = False
        for node, parents in _PARENTS.items():
            current = _canonical(node)
            if current in reachable:
                for parent, proper in parents.items():
                    if _canonical(parent) is not current:
                        changed |= add(parent, reachable[current] or proper)
        for intersection, components in _INTERSECTIONS.items():
            if all(_canonical(part) in reachable for part in components):
                changed |= add(intersection, False)
    target = _canonical(end)
    return target in reachable, reachable.get(target, False)


def register_subset(
    subset: NumberSetMeta, superset: NumberSetMeta, *, proper: bool = False
) -> None:
    """Declare inclusion; proper=True is an explicit mathematical assertion.

    Reject cycles and declarations refuted by known witnesses. Arbitrary user
    assertions cannot be proved here: callers are responsible for them.
    Equivalent custom sets should use aliases of the same set class.
    """
    _require_set(subset)
    _require_set(superset)
    if not isinstance(proper, bool):
        raise TypeError("proper must be a bool")
    if _canonical(subset) is _canonical(superset):
        if proper:
            raise ValueError("Equivalent sets cannot be proper subsets")
        return
    if _path(superset, subset)[0]:
        raise ValueError("Subset registration would create a cycle")
    if subset.subset_of(superset).state is Truth.FALSE:
        raise ValueError("Subset declaration contradicts established relationships")
    _PARENTS[subset][superset] = proper or _PARENTS[subset].get(superset, False)


class Number(metaclass=NumberSetMeta):
    """Organizational root: the same finite universe as Complex."""

    symbol = "Number"


class Complex(Number):
    """ℂ: finite complex numbers."""

    symbol = "ℂ"


class Real(Complex):
    """ℝ: finite real numbers."""

    symbol = "ℝ"


class Rational(Real):
    """ℚ: ratios of integers, including finite stored binary floats."""

    symbol = "ℚ"


class Integer(Rational):
    """ℤ: integers of any sign."""

    symbol = "ℤ"


class Whole(Integer):
    """Nonnegative integers: {0, 1, 2, ...}."""

    symbol = "Whole"


class Natural(Whole):
    """ℕ: positive integers, excluding zero."""

    symbol = "ℕ"


class Irrational(Real):
    """Real numbers outside Rational."""

    symbol = "Irrational"


class Algebraic(Complex):
    """Complex roots of nonzero polynomials with rational coefficients."""

    symbol = "Algebraic"


class Transcendental(Complex):
    """Complex numbers outside Algebraic, including non-real values."""

    symbol = "Transcendental"


class RealAlgebraic(Real, Algebraic):
    """The intersection of Real and Algebraic."""


class RealTranscendental(Real, Transcendental):
    """The intersection of Real and Transcendental."""


class AlgebraicIrrational(Algebraic, Irrational):
    """The intersection of Algebraic and Irrational."""


def _state(facts: Facts, domain: NumberSetMeta) -> Truth:
    fact = facts.get(domain)
    return Truth.UNKNOWN if fact is None else fact.state


def _put(facts: Facts, domain: NumberSetMeta, fact: Classification) -> bool:
    old = facts.get(domain)
    if old is not None and old.state is not Truth.UNKNOWN:
        if fact.state is not Truth.UNKNOWN and old.state is not fact.state:
            raise ClassificationConflictError(
                f"Conflicting facts for {domain!r}: {old.reason}; {fact.reason}"
            )
        return False
    facts[domain] = fact
    return fact.state is not Truth.UNKNOWN


def _infer(facts: Facts) -> Facts:
    """Close evidence under inclusion, partitions, and intersections."""
    inclusions = [
        (a, b) for a in _PARENTS for b in _PARENTS if a is not b and _path(a, b)[0]
    ]
    changed = True
    while changed:
        changed = False
        for subset, superset in inclusions:
            if _state(facts, subset) is Truth.TRUE:
                changed |= _put(
                    facts,
                    superset,
                    Classification(
                        Truth.TRUE,
                        f"Via {subset!r} <= {superset!r}: {facts[subset].reason}",
                    ),
                )
            if _state(facts, superset) is Truth.FALSE:
                changed |= _put(
                    facts,
                    subset,
                    Classification(
                        Truth.FALSE,
                        f"Excluded by {superset!r}: {facts[superset].reason}",
                    ),
                )
        for left, right in _DISJOINT:
            for source, target in ((left, right), (right, left)):
                if _state(facts, source) is Truth.TRUE:
                    changed |= _put(
                        facts,
                        target,
                        Classification(
                            Truth.FALSE,
                            f"Disjoint from {source!r}: {facts[source].reason}",
                        ),
                    )
        for universe, left, right in _PARTITIONS:
            if _state(facts, universe) is Truth.TRUE:
                for excluded, remaining in ((left, right), (right, left)):
                    if _state(facts, excluded) is Truth.FALSE:
                        changed |= _put(
                            facts,
                            remaining,
                            Classification(
                                Truth.TRUE,
                                f"In {universe!r} and outside {excluded!r}: {facts[excluded].reason}",
                            ),
                        )
        for intersection, components in _INTERSECTIONS.items():
            if all(_state(facts, part) is Truth.TRUE for part in components):
                changed |= _put(
                    facts,
                    intersection,
                    Classification(
                        Truth.TRUE,
                        "Intersection: "
                        + "; ".join(facts[p].reason for p in components),
                    ),
                )
            if _state(facts, intersection) is Truth.FALSE:
                for part in components:
                    if all(
                        _state(facts, p) is Truth.TRUE
                        for p in components
                        if p is not part
                    ):
                        changed |= _put(
                            facts,
                            part,
                            Classification(
                                Truth.FALSE,
                                f"Excluded by intersection {intersection!r}: {facts[intersection].reason}",
                            ),
                        )
    return facts


def _evidence(values: dict[NumberSetMeta, bool], reason: str) -> Facts:
    return {
        domain: Classification(Truth.TRUE if value else Truth.FALSE, reason)
        for domain, value in values.items()
    }


def classify(value: object) -> Facts:
    """Return results for all registered sets without canonicalizing the value.

    Unsupported inputs raise TypeError. Backend and extension failures propagate.
    """
    from ._adapters import builtin_facts

    facts = builtin_facts(value)
    supported = facts is not None
    if facts is None:
        facts = {}
    for domain, value_type, test in tuple(_TESTS):
        if isinstance(value, value_type):
            supported = True
            answer = test(value)
            if isinstance(answer, Classification):
                fact = answer
            elif answer is None or type(answer) is bool:
                state = (
                    Truth.UNKNOWN
                    if answer is None
                    else Truth.TRUE
                    if answer
                    else Truth.FALSE
                )
                fact = Classification(
                    state,
                    f"Registered rule for {value_type.__qualname__} in {domain!r}",
                )
            else:
                raise TypeError(
                    "Membership rules must return Classification, bool, or None"
                )
            _put(facts, domain, fact)
    if not supported:
        raise TypeError(f"Unsupported numeric input type: {type(value).__qualname__}")
    _infer(facts)
    return {
        domain: facts.get(
            domain,
            Classification(
                Truth.UNKNOWN,
                f"No conclusive evidence for {domain!r} on {type(value).__qualname__}",
            ),
        )
        for domain in _PARENTS
    }


# Number and Complex denote the same universe, despite different Python roles.
_EQUIVALENT[Number] = Complex
for _child in (
    Real,
    Rational,
    Integer,
    Whole,
    Natural,
    Irrational,
    Algebraic,
    Transcendental,
):
    for _parent in _PARENTS[_child]:
        _PARENTS[_child][_parent] = True
_PARENTS[Rational][Algebraic] = True
_PARENTS[RealTranscendental][Irrational] = True
_PARENTS[AlgebraicIrrational][RealAlgebraic] = True
_INTERSECTIONS.update(
    {
        RealAlgebraic: (Real, Algebraic),
        RealTranscendental: (Real, Transcendental),
        AlgebraicIrrational: (Algebraic, Irrational),
    }
)
_DISJOINT.extend(((Rational, Irrational), (Algebraic, Transcendental)))
_PARTITIONS.extend(((Real, Rational, Irrational), (Complex, Algebraic, Transcendental)))
_WITNESSES: tuple[tuple[str, dict[NumberSetMeta, bool]], ...] = (
    ("1", {Natural: True}),
    ("0", {Whole: True, Natural: False}),
    ("-1", {Integer: True, Whole: False}),
    ("1/2", {Rational: True, Integer: False}),
    ("sqrt(2)", {Irrational: True, Algebraic: True}),
    ("pi", {Real: True, Transcendental: True}),
    ("i", {Algebraic: True, Real: False}),
    ("i*pi", {Transcendental: True, Real: False}),
)
