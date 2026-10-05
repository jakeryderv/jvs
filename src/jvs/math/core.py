from __future__ import annotations

import warnings
from dataclasses import dataclass
from fractions import Fraction

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    rationalize,
    standard_transformations,
)


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

    def to_sympy(self) -> sp.Expr:
        """Return the underlying SymPy expression for backend-specific work."""
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
            other, (Expression, int, Fraction, float, complex, sp.Expr)
        ):
            return NotImplemented

        return self._expr == _to_sympy(other)

    def __str__(self) -> str:
        return str(self._expr)

    def __repr__(self) -> str:
        return f"Expression({self._expr!r})"


class ApproximateScalarWarning(UserWarning):
    """A Python float was passed to Scalar without explicit approximate intent."""


class Scalar(Expression):
    """A scalar whose numeric representation distinguishes exact and approximate.

    Integers and Fractions remain exact. Decimal strings are parsed as exact
    rationals. Python floats remain approximate and warn unless passed through
    approx(). Existing SymPy expressions retain their representation.
    """

    def __init__(
        self,
        value: Expression | int | Fraction | str | float | complex | sp.Expr,
    ) -> None:
        super().__init__(_to_scalar_sympy(value))

        if self._expr.free_symbols:
            raise ValueError("Scalar cannot contain free variables")

        if isinstance(value, float):
            warnings.warn(
                "Python floats are approximate. "
                f'Use Scalar.exact("{value}") for exact decimal intent or '
                f"Scalar.approx({value!r}) to explicitly accept approximation.",
                ApproximateScalarWarning,
                stacklevel=2,
            )

    @classmethod
    def exact(
        cls,
        value: Expression | int | Fraction | str | float | complex | sp.Expr,
    ) -> Scalar:
        """Construct an exact scalar without guessing intent from approximate input.

        Python floats/complex values and expressions containing SymPy Floats
        are rejected. Use decimal strings or Fractions for exact rational intent.
        """
        if isinstance(value, (float, complex)):
            raise TypeError(
                "Exact scalars cannot infer intent from Python floats or complex "
                "values; use a string, integer, Fraction, or exact expression"
            )

        scalar = cls(value)

        if not scalar.is_exact:
            raise ValueError("Exact scalars cannot contain approximate SymPy Floats")

        return scalar

    @classmethod
    def approx(
        cls,
        value: Expression | int | Fraction | str | float | complex | sp.Expr,
    ) -> Scalar:
        """Explicitly approximate a numeric scalar using SymPy's default precision.

        Python floats keep their existing approximation. Increasing evaluation
        precision later cannot recover information absent from a float input.
        """
        scalar = cls(_to_scalar_sympy(value))

        if not scalar._expr.is_number:
            raise ValueError("Approximation requires a numeric scalar")

        result = scalar._expr.evalf()

        # evalf() can keep exact zero; an explicit approximation should be Float.
        if result.is_Rational:
            result = sp.Float(result)

        return cls(result)

    @property
    def is_exact(self) -> bool:
        """Whether the current representation contains no approximate Float atoms.

        This describes the representation, not the provenance of earlier input
        or arithmetic. Nonfinite symbolic values retain their representation.
        """
        return not self._expr.has(sp.Float)

    def __repr__(self) -> str:
        return f"Scalar({self._expr!r})"


class Variable(Expression):
    """A named real variable; equal names represent the same variable."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Variable name must be a string")
        if not name:
            raise ValueError("Variable name cannot be empty")

        self._name = name
        super().__init__(sp.Symbol(name, real=True))

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
        if not all(isinstance(value, Set) for value in sets):
            raise TypeError("Cartesian product factors must be Sets")

        return cls(sp.ProductSet(*(value.to_sympy() for value in sets)))

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

    def to_sympy(self) -> sp.Set:
        """Return the underlying SymPy set for backend-specific operations."""
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

    def to_sympy(self) -> sp.Tuple:
        """Return the ordered coordinates as a SymPy tuple."""
        return sp.Tuple(*(coordinate.to_sympy() for coordinate in self._coordinates))

    def __repr__(self) -> str:
        return f"Point({', '.join(map(str, self._coordinates))})"


@dataclass(frozen=True, slots=True)
class Root:
    value: Expression
    multiplicity: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.value, Expression):
            raise TypeError("Root value must be an Expression")
        if isinstance(self.multiplicity, bool) or not isinstance(
            self.multiplicity, int
        ):
            raise TypeError("Root multiplicity must be an integer")
        if self.multiplicity < 1:
            raise ValueError("Root multiplicity must be at least 1")

    def to_sympy(self) -> sp.Tuple:
        """Return the root value and its multiplicity as a SymPy tuple."""
        return sp.Tuple(self.value.to_sympy(), self.multiplicity)


def _to_sympy(value: object) -> sp.Expr:
    if isinstance(value, Expression):
        return value.to_sympy()

    result = sp.sympify(value)

    if not isinstance(result, sp.Expr):
        raise TypeError(f"Cannot represent {value!r} as a scalar expression")

    return result


def _wrap_sympy(value: sp.Expr) -> Expression:
    if value.free_symbols:
        return Expression(value)

    return Scalar(value)


def _to_scalar_sympy(value: object) -> sp.Expr:
    if isinstance(value, Fraction):
        return sp.Rational(value.numerator, value.denominator)
    if isinstance(value, str):
        return _to_sympy(
            parse_expr(value, transformations=(*standard_transformations, rationalize))
        )

    return _to_sympy(value)
