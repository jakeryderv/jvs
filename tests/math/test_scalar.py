import warnings
from fractions import Fraction

import pytest
import sympy as sp

from jvs.math.core import ApproximateScalarWarning, Scalar, Variable
from jvs.math.function import Function
from jvs.math.polynomial import Polynomial


@pytest.mark.parametrize("value", [0, 3, -7, 2**80 + 1])
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_integers_are_exact(value, constructor):
    scalar = constructor(value)

    assert isinstance(scalar.to_sympy(), sp.Integer)
    assert scalar.to_sympy() == sp.Integer(value)
    assert scalar.is_exact


@pytest.mark.parametrize("value", [Fraction(1, 3), Fraction(-7, 12), Fraction(2, 4)])
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_fractions_are_exact(value, constructor):
    scalar = constructor(value)

    assert scalar.to_sympy() == sp.Rational(value.numerator, value.denominator)
    assert scalar == value
    assert scalar.is_exact


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.1", sp.Rational(1, 10)),
        ("-2.50", sp.Rational(-5, 2)),
        ("1e-3", sp.Rational(1, 1000)),
        ("0.123456789012345678901", sp.Rational(123456789012345678901, 10**21)),
    ],
)
@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact])
def test_decimal_strings_are_exact(text, expected, constructor):
    scalar = constructor(text)

    assert scalar.to_sympy() == expected
    assert scalar.is_exact


def test_exact_decimal_arithmetic_stays_exact():
    result = Scalar("0.1") + Scalar.exact("0.2")

    assert isinstance(result, Scalar)
    assert result.to_sympy() == sp.Rational(3, 10)
    assert result.is_exact


@pytest.mark.parametrize("value", [2, Fraction(1, 3), "0.1", sp.pi, sp.Float("0.1")])
def test_unambiguous_or_existing_backend_inputs_do_not_warn(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Scalar(value)


def test_implicit_float_warns_at_the_call_site():
    with pytest.warns(ApproximateScalarWarning) as captured:
        scalar = Scalar(0.1)

    assert len(captured) == 1
    warning = captured[0]
    assert warning.filename == __file__
    assert "Python floats are approximate" in str(warning.message)
    assert 'Scalar.exact("0.1")' in str(warning.message)
    assert "Scalar.approx(0.1)" in str(warning.message)
    assert isinstance(scalar.to_sympy(), sp.Float)
    assert scalar.to_sympy() == sp.Float(0.1)
    assert scalar.to_sympy() != sp.Rational(1, 10)
    assert not scalar.is_exact


@pytest.mark.parametrize("value", [0.0, 0.1, -2.5])
def test_explicit_float_approximation_preserves_input_without_warning(value):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scalar = Scalar.approx(value)

    assert isinstance(scalar.to_sympy(), sp.Float)
    assert scalar.to_sympy() == sp.Float(value)
    assert not scalar.is_exact


@pytest.mark.parametrize("value", [0, 2, Fraction(1, 3), "0.1", sp.sqrt(2)])
def test_exact_values_can_be_explicitly_approximated(value):
    scalar = Scalar.approx(value)

    assert isinstance(scalar.to_sympy(), sp.Float)
    assert not scalar.is_exact
    assert float(scalar.to_sympy()) == pytest.approx(float(Scalar(value).to_sympy()))


@pytest.mark.parametrize("value", [0.1, 1.0, 1 + 2j])
def test_exact_rejects_python_approximate_inputs(value):
    with pytest.raises(TypeError, match="cannot infer intent"):
        Scalar.exact(value)


@pytest.mark.parametrize(
    "value",
    [sp.Float("0.1"), Scalar.approx(0.1), sp.sqrt(2) + sp.Float("0.1")],
)
def test_exact_rejects_existing_approximate_expressions(value):
    with pytest.raises(ValueError, match="approximate SymPy Floats"):
        Scalar.exact(value)


def test_exact_symbolic_constants_remain_exact():
    assert Scalar.exact(sp.pi).to_sympy() == sp.pi
    assert Scalar.exact(sp.sqrt(2)).is_exact
    assert Scalar.exact("sqrt(2) + 0.1").to_sympy() == sp.sqrt(2) + sp.Rational(1, 10)


@pytest.mark.parametrize("constructor", [Scalar, Scalar.exact, Scalar.approx])
def test_free_variables_are_rejected(constructor):
    with pytest.raises(ValueError, match="free variables"):
        constructor(Variable("x"))


def test_explicit_approximation_does_not_repeat_warnings_in_math_operations():
    x = Variable("x")
    coefficient = Scalar.approx(0.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        function_result = Function(coefficient * x, (x,))(Scalar(2))
        polynomial = Polynomial(coefficient * x, x)
        polynomial_result = polynomial(Scalar(2))
        arithmetic_result = coefficient + Scalar(1)

    for result in (function_result, polynomial_result, arithmetic_result):
        assert isinstance(result, Scalar)
        assert not result.is_exact
    assert not polynomial.coefficients[0].is_exact
