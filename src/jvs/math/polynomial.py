from __future__ import annotations

import sympy as sp
from sympy.polys.polyerrors import PolynomialError

from jvs.math.core import (
    Expression,
    Scalar,
    Set,
    Variable,
    _to_sympy,
    _wrap_sympy,
)

from .function import Function


class Polynomial(Function):
    """A univariate polynomial with real coefficients and descriptive sets.

    Domain and codomain default to the reals and can be overridden as metadata.
    The zero polynomial is valid and has degree negative infinity.
    """

    def __init__(
        self,
        expression: Expression | int | float | complex | sp.Expr,
        variable: Variable,
        *,
        domain: Set | None = None,
        codomain: Set | None = None,
    ) -> None:
        if not isinstance(variable, Variable):
            raise TypeError("Polynomial variable must be a Variable")

        try:
            poly = sp.Poly(
                _to_sympy(expression),
                variable.to_sympy(),
            )
        except PolynomialError as exc:
            raise ValueError(
                "expression must be a polynomial in the given variable"
            ) from exc

        if any(coefficient.is_real is not True for coefficient in poly.all_coeffs()):
            raise ValueError("Polynomial coefficients must be real")

        self._poly = poly
        self._variable = variable

        super().__init__(
            expression=_wrap_sympy(poly.as_expr()),
            variables=(variable,),
            domain=Set.reals() if domain is None else domain,
            codomain=Set.reals() if codomain is None else codomain,
        )

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def coefficients(self) -> tuple[Scalar, ...]:
        return tuple(Scalar(coefficient) for coefficient in self._poly.all_coeffs())

    @property
    def terms(self) -> tuple[Expression, ...]:
        symbol = self._variable.to_sympy()

        return tuple(
            _wrap_sympy(coefficient * symbol ** monomial[0])
            for monomial, coefficient in self._poly.terms()
        )

    @property
    def degree(self) -> Scalar:
        return Scalar(self._poly.degree())

    @property
    def leading_coefficient(self) -> Scalar:
        return Scalar(self._poly.LC())

    def to_sympy(self) -> sp.Poly:
        """Return the underlying SymPy polynomial, preserving its generator."""
        return self._poly

    def __repr__(self) -> str:
        return (
            f"Polynomial("
            f"expression={self.expression!r}, "
            f"variable={self.variable!r}, "
            f"domain={self.domain!r}, "
            f"codomain={self.codomain!r}"
            f")"
        )
