from __future__ import annotations

import sympy as sp

from jvs.math.core import Expression, Set, Variable, _to_sympy, _wrap_sympy


class Function:
    """A scalar expression with distinct arguments and declared sets.

    Multivariable domains contain argument tuples, typically constructed with
    Set.product(). Concrete inputs and outputs must have definite membership
    in their declared sets; violations or unresolved membership raise ValueError.
    Symbolic inputs or outputs defer the corresponding membership check.
    """

    def __init__(
        self,
        expression: Expression,
        variables: tuple[Variable, ...],
        domain: Set,
        codomain: Set,
    ) -> None:
        allowed_symbols = {variable._as_sympy() for variable in variables}

        if len(allowed_symbols) != len(variables):
            raise ValueError("Function variables must have distinct symbols")

        unknown_symbols = expression._as_sympy().free_symbols - allowed_symbols

        if unknown_symbols:
            raise ValueError(
                f"Expression contains undeclared variables: {unknown_symbols}"
            )

        self._expression = expression
        self._variables = variables
        self._domain = domain
        self._codomain = codomain

    @property
    def expression(self) -> Expression:
        return self._expression

    @property
    def variables(self) -> tuple[Variable, ...]:
        return self._variables

    @property
    def domain(self) -> Set:
        return self._domain

    @property
    def codomain(self) -> Set:
        return self._codomain

    def __call__(self, *values: object) -> Expression:
        """Evaluate with simultaneous substitution and check concrete membership."""
        if len(values) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} arguments, got {len(values)}"
            )

        arguments = tuple(_to_sympy(value) for value in values)

        if not any(argument.free_symbols for argument in arguments):
            input_value = arguments[0] if len(arguments) == 1 else sp.Tuple(*arguments)
            _require_membership(input_value, self._domain, "Input")

        substitutions = {
            variable._as_sympy(): argument
            for variable, argument in zip(
                self._variables,
                arguments,
                strict=True,
            )
        }

        result = self._expression._as_sympy().subs(
            substitutions.items(), simultaneous=True
        )

        if not result.free_symbols:
            _require_membership(result, self._codomain, "Result")

        return _wrap_sympy(result)

    def __str__(self) -> str:
        return str(self.expression)

    def __repr__(self) -> str:
        return (
            f"Function("
            f"expression={self.expression!r}, "
            f"variables={self.variables!r}, "
            f"domain={self.domain!r}, "
            f"codomain={self.codomain!r}"
            f")"
        )


def _require_membership(value: object, allowed_set: Set, label: str) -> None:
    membership = allowed_set.contains(value)

    if membership is False:
        raise ValueError(f"{label} {value} is outside the declared set {allowed_set}")
    if membership is None:
        raise ValueError(
            f"Cannot determine whether {label.lower()} {value} "
            f"belongs to the declared set {allowed_set}"
        )
