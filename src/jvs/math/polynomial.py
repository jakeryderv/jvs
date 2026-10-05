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
    """A univariate polynomial with real coefficients, domain, and codomain.

    Concrete calls enforce membership in the reals; symbolic calls defer it.
    The zero polynomial is valid and has degree negative infinity.
    """

    def __init__(
        self,
        expression: Expression | int | float | complex,
        variable: Variable,
    ) -> None:
        try:
            poly = sp.Poly(
                _to_sympy(expression),
                variable._as_sympy(),
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
            expression=Expression(poly.as_expr()),
            variables=(variable,),
            domain=Set.reals(),
            codomain=Set.reals(),
        )

    @property
    def variable(self) -> Variable:
        return self._variable

    @property
    def coefficients(self) -> tuple[Scalar, ...]:
        return tuple(Scalar(coefficient) for coefficient in self._poly.all_coeffs())

    @property
    def terms(self) -> tuple[Expression, ...]:
        symbol = self._variable._as_sympy()

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

    def _as_sympy_poly(self) -> sp.Poly:
        return self._poly

    def __repr__(self) -> str:
        return f"Polynomial(expression={self.expression!r}, variable={self.variable!r})"
