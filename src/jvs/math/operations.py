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
    """Differentiate with explicitly declared derivative sets.

    Polynomial derivatives default to polynomials over the reals. Other
    functions require both domain and codomain. Supplying both sets for a
    polynomial returns a general Function with those sets instead.
    The caller is responsible for choosing a domain of differentiability.
    """
    variable = _resolve_variable(function, variable)

    if isinstance(function, Polynomial) and domain is None and codomain is None:
        return Polynomial(
            Expression(sp.diff(function.expression._as_sympy(), variable._as_sympy())),
            function.variable,
        )

    if domain is None or codomain is None:
        raise ValueError("Derivative domain and codomain must both be supplied")

    return Function(
        expression=Expression(
            sp.diff(function.expression._as_sympy(), variable._as_sympy())
        ),
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
            function.expression._as_sympy(),
            variable._as_sympy(),
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
    poly = polynomial._as_sympy_poly()

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
    if variable is None:
        if len(function.variables) != 1:
            raise ValueError("variable is required for multivariable functions")

        return function.variables[0]

    if not any(
        variable._as_sympy() == candidate._as_sympy()
        for candidate in function.variables
    ):
        raise ValueError("variable does not belong to this function")

    return variable
