from __future__ import annotations

import sympy as sp

from jvs.math.core import Expression, Set, Variable, _to_sympy, _wrap_sympy


class Function:
    """A scalar expression with distinct arguments and descriptive sets.

    Multivariable domains contain argument tuples, typically constructed with
    Set.product(). None means the domain or codomain is unspecified. Declared
    sets express an intended mathematical contract, not a proof of validity.
    Evaluation validates argument structure; membership can be checked explicitly
    using Set.contains().
    """

    def __init__(
        self,
        expression: Expression,
        variables: tuple[Variable, ...],
        domain: Set | None = None,
        codomain: Set | None = None,
    ) -> None:
        if not isinstance(expression, Expression):
            raise TypeError("Function expression must be an Expression")
        if not isinstance(variables, tuple) or not all(
            isinstance(variable, Variable) for variable in variables
        ):
            raise TypeError("Function variables must be a tuple of Variables")
        if domain is not None and not isinstance(domain, Set):
            raise TypeError("Function domain must be a Set or None")
        if codomain is not None and not isinstance(codomain, Set):
            raise TypeError("Function codomain must be a Set or None")

        allowed_symbols = {variable.to_sympy() for variable in variables}

        if len(allowed_symbols) != len(variables):
            raise ValueError("Function variables must have distinct symbols")

        unknown_symbols = expression.to_sympy().free_symbols - allowed_symbols

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
    def domain(self) -> Set | None:
        return self._domain

    @property
    def codomain(self) -> Set | None:
        return self._codomain

    def __call__(self, *values: object) -> Expression:
        """Evaluate with simultaneous substitution and descriptive set metadata."""
        if len(values) != len(self._variables):
            raise ValueError(
                f"Expected {len(self._variables)} arguments, got {len(values)}"
            )

        arguments = tuple(_to_sympy(value) for value in values)

        substitutions = {
            variable.to_sympy(): argument
            for variable, argument in zip(
                self._variables,
                arguments,
                strict=True,
            )
        }

        result = self._expression.to_sympy().subs(
            substitutions.items(), simultaneous=True
        )

        return _wrap_sympy(result)

    def to_sympy(self) -> sp.Basic:
        """Return a SymPy Lambda; declared sets remain metadata on this object.

        Subclasses may return a more specific mathematical representation.
        """
        return sp.Lambda(
            tuple(variable.to_sympy() for variable in self._variables),
            self._expression.to_sympy(),
        )

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
