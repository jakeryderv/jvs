"""Public mathematical contracts and regressions for precision and inference."""

import math
import numbers
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    Algebraic,
    AlgebraicIrrational,
    Classification,
    ClassificationConflictError,
    Complex,
    IndeterminateError,
    IndeterminateMembershipError,
    IndeterminateRelationError,
    Integer,
    Irrational,
    Natural,
    Number,
    Rational,
    Real,
    RealAlgebraic,
    RealTranscendental,
    Transcendental,
    Truth,
    Whole,
    classify,
    register_subset,
)
from jvs.numeric import number as ontology


@pytest.fixture(autouse=True)
def isolate_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    # Extensions are process-wide declarations; each test gets a fresh registry.
    monkeypatch.setattr(
        ontology, "_PARENTS", {k: dict(v) for k, v in ontology._PARENTS.items()}
    )
    monkeypatch.setattr(ontology, "_TESTS", list(ontology._TESTS))


def test_standard_relationships() -> None:
    chain = [Natural, Whole, Integer, Rational, Real, Complex]
    for i, subset in enumerate(chain):
        assert subset <= subset
        assert not subset < subset
        for superset in chain[i + 1 :]:
            assert subset < superset
            assert superset > subset
            assert superset >= subset
            assert not superset <= subset
    assert Rational < Algebraic
    assert Irrational < Real
    assert Algebraic < Complex
    assert Transcendental < Complex
    assert not Algebraic <= Real
    assert not Real <= Algebraic
    assert not Transcendental <= Irrational
    assert RealTranscendental < Irrational
    assert AlgebraicIrrational <= RealAlgebraic
    assert RealAlgebraic < Real
    assert Rational < RealAlgebraic
    assert not Natural <= Irrational
    assert not Natural <= Transcendental


def test_number_and_complex_are_equivalent() -> None:
    assert Number <= Complex
    assert Complex <= Number
    assert not Number < Complex
    assert not Complex < Number
    register_subset(Number, Complex)
    with pytest.raises(ValueError, match="Equivalent"):
        register_subset(Complex, Number, proper=True)


@pytest.mark.parametrize(
    "domain",
    [Number, Complex, Real, Rational, Integer, Whole, Natural, Irrational, Algebraic],
)
def test_sets_cannot_be_instantiated(domain: ontology.NumberSetMeta) -> None:
    with pytest.raises(TypeError, match="not a scalar constructor"):
        domain()


@pytest.mark.parametrize(
    "value",
    [
        3,
        3.0,
        3 + 0j,
        Fraction(3),
        Decimal(3),
        np.int8(3),
        np.uint64(3),
        np.float32(3),
        np.clongdouble(3),
        sp.Integer(3),
        sp.Float(3),
    ],
)
def test_positive_integer_representations(value: object) -> None:
    facts = classify(value)
    for domain in (
        Natural,
        Whole,
        Integer,
        Rational,
        Real,
        Complex,
        Number,
        Algebraic,
        RealAlgebraic,
    ):
        assert facts[domain].state is Truth.TRUE
    for domain in (Irrational, Transcendental, AlgebraicIrrational):
        assert facts[domain].state is Truth.FALSE


@pytest.mark.parametrize(
    "value", [0, -0.0, 0j, Fraction(0), Decimal("-0"), np.float64(-0.0), sp.S.Zero]
)
def test_zero_convention(value: object) -> None:
    assert value in Whole
    assert value not in Natural


@pytest.mark.parametrize(
    "value",
    [-3, -3.0, -3 + 0j, Fraction(-3), Decimal(-3), np.int64(-3), sp.Integer(-3)],
)
def test_negative_integer_convention(value: object) -> None:
    assert value in Integer
    assert value not in Whole
    assert value not in Natural


@pytest.mark.parametrize(
    "value",
    [
        Fraction(1, 3),
        0.1,
        Decimal("0.1"),
        np.float32(0.1),
        sp.Rational(1, 3),
        sp.Float("0.1"),
    ],
)
def test_fractional_values(value: object) -> None:
    assert value in Rational
    assert value not in Integer
    assert value not in Irrational


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        sp.true,
        float("nan"),
        float("inf"),
        -float("inf"),
        complex(0, float("inf")),
        np.float32("nan"),
        np.clongdouble("inf"),
        Decimal("NaN"),
        Decimal("sNaN"),
        Decimal("Infinity"),
        sp.nan,
        sp.oo,
        -sp.oo,
        sp.zoo,
    ],
)
def test_values_outside_the_finite_universe(value: object) -> None:
    assert all(result.state is Truth.FALSE for result in classify(value).values())


@pytest.mark.parametrize(
    "value",
    [
        "3",
        [3],
        np.array(3),
        np.array([3]),
        np.datetime64("2026-01-01"),
        np.timedelta64(3, "D"),
        object(),
        sp.Matrix([3]),
        sp.Symbol("A", commutative=False),
    ],
)
def test_unsupported_inputs_raise(value: object) -> None:
    with pytest.raises(TypeError):
        Real.query(value)


def test_large_exact_values_and_decimal_context() -> None:
    assert 10**1000 in Natural
    assert Fraction(10**1000, 3) in Rational
    assert Fraction(10**1000, 3) not in Integer
    with localcontext() as context:
        context.prec = 2
        assert Decimal("1e10000") in Natural
        assert Decimal("1.0000000000000000001") not in Integer


def test_extended_precision_fraction_is_not_rounded_to_integer() -> None:
    if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
        pytest.skip("This platform has no wider longdouble significand")
    value = np.nextafter(np.longdouble(1), np.longdouble(2))
    assert float(value) == 1.0  # The old narrowing implementation lost this bit.
    assert value not in Integer
    assert value in Rational
    assert np.clongdouble(value) not in Integer


def test_extended_precision_range_is_not_narrowed() -> None:
    if np.finfo(np.longdouble).maxexp <= np.finfo(np.float64).maxexp:
        pytest.skip("This platform has no wider longdouble exponent")
    value = np.longdouble("1e400")
    assert bool(np.isfinite(value))
    assert value in Real
    assert value in Natural


@pytest.mark.parametrize("value", [1j, np.complex64(1j), sp.I])
def test_nonreal_algebraic_values(value: object) -> None:
    assert value in Algebraic
    assert value not in Real
    assert value not in Irrational
    assert value not in Transcendental


def test_exact_symbolic_classification_and_approximation() -> None:
    assert sp.sqrt(2) in AlgebraicIrrational
    assert sp.sqrt(2) in RealAlgebraic
    assert sp.pi in RealTranscendental
    assert sp.I * sp.pi in Transcendental
    assert sp.I * sp.pi not in Irrational
    assert math.sqrt(2) in Rational
    assert sp.sqrt(2).evalf(80) in Rational
    assert math.pi not in Transcendental


def test_symbolic_unknown_and_assumptions() -> None:
    x = sp.Symbol("x")
    result = Real.query(x)
    assert result.state is Truth.UNKNOWN
    assert "SymPy" in result.reason
    with pytest.raises(IndeterminateMembershipError):
        _ = x in Real
    with pytest.raises(IndeterminateError):
        bool(result)
    with pytest.raises(IndeterminateError):
        bool(Truth.UNKNOWN)
    assert sp.Symbol("n", positive=True, integer=True) in Natural
    assert sp.Symbol("n", integer=True) in Integer
    assert Whole.query(sp.Symbol("n", integer=True)).state is Truth.UNKNOWN


def test_custom_real_abc_is_not_guessed_rational() -> None:
    @numbers.Real.register
    class ExactIrrational:
        def __float__(self) -> float:
            raise AssertionError("Classification must not call float()")

    value = ExactIrrational()
    with pytest.raises(TypeError):
        Rational.query(value)
    Irrational.register_member_test(lambda _: True, value_type=ExactIrrational)
    for domain in (Irrational, Real, Complex, Number):
        assert value in domain
    assert value not in Rational


def test_natural_hook_and_intersection_inference() -> None:
    class Token:
        pass

    Natural.register_member_test(lambda _: True, value_type=Token)
    value = Token()
    assert value in Natural
    assert value in Whole
    assert value in RealAlgebraic
    assert value not in AlgebraicIrrational


def test_external_intersection_and_partition_reasoning() -> None:
    class Token:
        pass

    Real.register_member_test(lambda _: True, value_type=Token)
    Rational.register_member_test(lambda _: False, value_type=Token)
    Algebraic.register_member_test(
        lambda _: Classification(Truth.TRUE, "Exact polynomial witness"),
        value_type=Token,
    )
    value = Token()
    assert value in Irrational
    result = AlgebraicIrrational.query(value)
    assert result.state is Truth.TRUE
    assert "Exact polynomial witness" in result.reason


def test_unknown_rule_supports_input_without_fabricating_membership() -> None:
    class Token:
        pass

    Real.register_member_test(lambda _: None, value_type=Token)
    assert Real.query(Token()).state is Truth.UNKNOWN
    assert Number.query(Token()).state is Truth.UNKNOWN


def test_conflicting_rules_raise() -> None:
    class Token:
        pass

    Rational.register_member_test(lambda _: True, value_type=Token)
    Irrational.register_member_test(lambda _: True, value_type=Token)
    with pytest.raises(ClassificationConflictError):
        Number.query(Token())


def test_backend_or_rule_failures_propagate() -> None:
    class Token:
        pass

    def broken(_: Token) -> bool:
        raise RuntimeError("adapter failure")

    Real.register_member_test(broken, value_type=Token)
    with pytest.raises(RuntimeError, match="adapter failure"):
        Real.query(Token())


def test_custom_subsets_do_not_inherit_parent_membership() -> None:
    class PositiveInteger(Integer):
        pass

    assert PositiveInteger <= Integer
    assert PositiveInteger.query(-3).state is Truth.UNKNOWN
    assert PositiveInteger.query(Fraction(1, 2)).state is Truth.FALSE
    assert PositiveInteger.subset_of(Integer, proper=True).state is Truth.UNKNOWN
    with pytest.raises(IndeterminateRelationError):
        _ = PositiveInteger < Integer


def test_subset_registration_transitivity_and_cycles() -> None:
    class A(Number):
        pass

    class B(Number):
        pass

    class C(Number):
        pass

    assert A.subset_of(B).state is Truth.UNKNOWN
    with pytest.raises(IndeterminateRelationError):
        _ = A <= B
    register_subset(A, B)
    register_subset(B, C, proper=True)
    assert A < C
    assert not C <= A
    with pytest.raises(ValueError, match="cycle"):
        register_subset(C, A)
    with pytest.raises(ValueError):
        register_subset(Real, Integer)
    with pytest.raises(ValueError, match="contradicts"):
        register_subset(Real, Algebraic)


def test_extension_fact_follows_registered_edges() -> None:
    class A(Number):
        pass

    class B(Number):
        pass

    class Token:
        pass

    register_subset(A, B)
    A.register_member_test(lambda _: True, value_type=Token)
    assert Token() in B
    assert Token() in Number


def test_custom_subset_of_both_components_is_subset_of_intersection() -> None:
    class A(Number):
        pass

    register_subset(A, Real)
    register_subset(A, Algebraic)
    assert RealAlgebraic >= A


def test_all_established_inclusions_preserve_membership() -> None:
    domains = (
        Number,
        Complex,
        Real,
        Rational,
        Integer,
        Whole,
        Natural,
        Irrational,
        Algebraic,
        Transcendental,
        RealAlgebraic,
        RealTranscendental,
        AlgebraicIrrational,
    )
    inclusions = [
        (a, b) for a in domains for b in domains if a.subset_of(b).state is Truth.TRUE
    ]
    for value in (
        0,
        1,
        -1,
        Fraction(1, 3),
        sp.sqrt(2),
        sp.pi,
        sp.I,
        sp.I * sp.pi,
        sp.Symbol("x"),
        sp.Symbol("n", integer=True),
        np.longdouble("0.1"),
    ):
        facts = classify(value)
        for subset, superset in inclusions:
            if facts[subset].state is Truth.TRUE:
                assert facts[superset].state is Truth.TRUE
            if facts[superset].state is Truth.FALSE:
                assert facts[subset].state is Truth.FALSE


def test_invalid_extension_answer_is_not_coerced_to_bool() -> None:
    class Token:
        pass

    # Simulate an untyped third-party callback violating the documented contract.
    def invalid(_: Token):
        return "yes"

    Real.register_member_test(invalid, value_type=Token)
    with pytest.raises(TypeError, match="must return"):
        Real.query(Token())


def test_negative_hook_conflicting_with_builtin_evidence_raises() -> None:
    Rational.register_member_test(lambda _: False, value_type=int)
    with pytest.raises(ClassificationConflictError):
        Rational.query(3)
