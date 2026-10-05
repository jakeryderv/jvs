from __future__ import annotations

from typing import Literal

import sympy as sp
from sympy.core.function import PoleError
from sympy.polys.polyerrors import PolynomialError

from jvs.math.core import (
    Expression,
    Root,
    Scalar,
    Set,
    Variable,
    _to_sympy,
    _wrap_sympy,
)

from .function import Function
from .polynomial import Polynomial


def derivative(
    function: Function,
    variable: Variable | None = None,
    *,
    domain: Set | None = None,
    codomain: Set | None = None,
) -> Function:
    """Differentiate while preserving known structure and descriptive sets.

    Polynomial derivatives remain polynomials with real default sets. General
    derivatives leave their domain and codomain unspecified. Either set may be
    supplied explicitly; the caller chooses a domain of differentiability.
    """
    variable = _resolve_variable(function, variable)

    result = _wrap_sympy(sp.diff(function.expression.to_sympy(), variable.to_sympy()))

    if isinstance(function, Polynomial):
        return Polynomial(
            result,
            function.variable,
            domain=domain,
            codomain=codomain,
        )

    return Function(
        expression=result,
        variables=function.variables,
        domain=domain,
        codomain=codomain,
    )


def limit(
    function: Function,
    point: Scalar | int | float,
    variable: Variable | None = None,
    *,
    direction: Literal["-", "+", "+-"] = "+-",
) -> Expression:
    """Compute an expression limit, defaulting to a two-sided approach.

    Use '-' for the left-hand limit and '+' for the right-hand limit. At
    infinity, SymPy determines the approach direction. Nonexistent or
    unresolved limits raise ValueError; valid signed infinite limits remain
    expressions. Other function variables are held constant. Choose an
    approach direction compatible with the function's domain.
    """
    if direction not in ("-", "+", "+-"):
        raise ValueError("direction must be '-', '+', or '+-'")

    variable = _resolve_variable(function, variable)
    approach_point = _to_sympy(point)

    try:
        result = sp.limit(
            function.expression.to_sympy(),
            variable.to_sympy(),
            approach_point,
            dir=direction,
        )
    except (ValueError, NotImplementedError, PoleError) as exc:
        raise ValueError(
            f"Could not determine a valid limit at {approach_point} "
            f"with direction {direction!r}: {exc}"
        ) from exc

    if result.has(sp.nan, sp.zoo, sp.AccumBounds, sp.Limit):
        raise ValueError(
            f"The limit at {approach_point} with direction {direction!r} "
            "does not exist or could not be determined"
        )

    return _wrap_sympy(result)


def real_roots(
    polynomial: Polynomial,
) -> tuple[Root, ...]:
    """Return sorted exact real roots with their positive multiplicities.

    Nonzero constant polynomials have no roots. The zero polynomial raises
    ValueError because every real number is a root. Unsupported exact root
    computations raise NotImplementedError.
    """
    if not isinstance(polynomial, Polynomial):
        raise TypeError("polynomial must be a Polynomial")

    poly = polynomial.to_sympy()

    if poly.is_zero:
        raise ValueError("The zero polynomial has infinitely many real roots")
    if poly.degree() == 0:
        return ()

    try:
        result = sp.real_roots(
            poly,
            multiple=False,
            extension=True,
        )
    except (NotImplementedError, PolynomialError) as exc:
        raise NotImplementedError(
            "Exact real roots are unavailable for this polynomial"
        ) from exc

    return tuple(
        Root(
            value=_wrap_sympy(root),
            multiplicity=int(multiplicity),
        )
        for root, multiplicity in result
    )


def _resolve_variable(
    function: Function,
    variable: Variable | None,
) -> Variable:
    if not isinstance(function, Function):
        raise TypeError("function must be a Function")

    if variable is None:
        if len(function.variables) != 1:
            raise ValueError("variable is required for multivariable functions")

        return function.variables[0]

    if not isinstance(variable, Variable):
        raise TypeError("variable must be a Variable or None")

    if variable not in function.variables:
        raise ValueError("variable does not belong to this function")

    return variable
