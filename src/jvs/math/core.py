from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


class Expression:
    """A scalar symbolic expression with structural equality.

    Equality compares SymPy representations without algebraic simplification.
    Expressions are unhashable, including Scalar and Variable instances.
    """

    __slots__ = ("_expr",)

    def __init__(
        self,
        value: Expression | int | float | complex | sp.Expr,
    ) -> None:
        self._expr = _to_sympy(value)

    def _as_sympy(self) -> sp.Expr:
        return self._expr

    def __add__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr + _to_sympy(other))

    def __radd__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) + self._expr)

    def __sub__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr - _to_sympy(other))

    def __rsub__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) - self._expr)

    def __mul__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr * _to_sympy(other))

    def __rmul__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) * self._expr)

    def __truediv__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr / _to_sympy(other))

    def __rtruediv__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) / self._expr)

    def __pow__(self, other: object) -> Expression:
        return _wrap_sympy(self._expr ** _to_sympy(other))

    def __rpow__(self, other: object) -> Expression:
        return _wrap_sympy(_to_sympy(other) ** self._expr)

    def __neg__(self) -> Expression:
        return _wrap_sympy(-self._expr)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, bool) or not isinstance(
            other, (Expression, int, float, complex, sp.Expr)
        ):
            return NotImplemented

        return self._expr == _to_sympy(other)

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"Expression({self._expr!r})"


class Scalar(Expression):
    def __init__(
        self,
        value: Expression | int | float | complex | sp.Expr,
    ) -> None:
        super().__init__(value)

        if self._expr.free_symbols:
            raise ValueError("Scalar cannot contain free variables")

    def __repr__(self) -> str:
        return f"Scalar({self._expr!r})"


class Variable(Expression):
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not name:
            raise ValueError("Variable name cannot be empty")

        self._name = name
        super().__init__(sp.Symbol(name))

    @property
    def name(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"Variable({self._name!r})"


class Set:
    """A mathematical set with definite or unresolved membership."""

    __slots__ = ("_set",)

    def __init__(self, value: sp.Set) -> None:
        if not isinstance(value, sp.Set):
            raise TypeError("value must be a SymPy Set")

        self._set = value

    @classmethod
    def reals(cls) -> Set:
        return cls(sp.S.Reals)

    @classmethod
    def integers(cls) -> Set:
        return cls(sp.S.Integers)

    @classmethod
    def interval(
        cls,
        start: object,
        end: object,
        *,
        left_open: bool = False,
        right_open: bool = False,
    ) -> Set:
        return cls(
            sp.Interval(
                _to_sympy(start),
                _to_sympy(end),
                left_open=left_open,
                right_open=right_open,
            )
        )

    @classmethod
    def product(cls, *sets: Set) -> Set:
        """Build a Cartesian product whose elements are coordinate tuples."""
        return cls(sp.ProductSet(*(value._as_sympy() for value in sets)))

    def contains(self, value: object) -> bool | None:
        """Return True or False for known membership, or None if unresolved.

        Tuples are interpreted as elements of a Cartesian product.
        """
        if isinstance(value, (tuple, sp.Tuple)):
            element = sp.Tuple(*(_to_sympy(coordinate) for coordinate in value))
        else:
            element = _to_sympy(value)

        membership = self._set.contains(element)

        if membership is sp.S.true:
            return True
        if membership is sp.S.false:
            return False

        return None

    def _as_sympy(self) -> sp.Set:
        return self._set

    def __str__(self) -> str:
        return str(self._set)

    def __repr__(self) -> str:
        return f"Set({self._set!r})"


class Point:
    __slots__ = ("_coordinates",)

    def __init__(self, *coordinates: object) -> None:
        if not coordinates:
            raise ValueError("Point requires at least one coordinate")

        self._coordinates = tuple(
            _wrap_sympy(_to_sympy(coordinate)) for coordinate in coordinates
        )

    @property
    def coordinates(self) -> tuple[Expression, ...]:
        return self._coordinates

    @property
    def dimension(self) -> int:
        return len(self._coordinates)

    def __repr__(self) -> str:
        return f"Point({', '.join(map(str, self._coordinates))})"


@dataclass(frozen=True, slots=True)
class Root:
    value: Expression
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if self.multiplicity < 1:
            raise ValueError("Root multiplicity must be at least 1")


def _to_sympy(value: object) -> sp.Expr:
    if isinstance(value, Expression):
        return value._as_sympy()

    result = sp.sympify(value)

    if not isinstance(result, sp.Expr):
        raise TypeError(f"Cannot represent {value!r} as a scalar expression")

    return result


def _wrap_sympy(value: sp.Expr) -> Expression:
    if value.free_symbols:
        return Expression(value)

    return Scalar(value)
