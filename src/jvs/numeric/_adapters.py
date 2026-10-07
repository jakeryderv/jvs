"""Backend facts for scalar classification without lossy canonicalization."""

from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import sympy as sp

from .complex import ComplexValue
from .fixed import FixedValue
from .floating import FloatingValue
from .integer import IntegerValue
from .number import (
    Algebraic,
    Classification,
    Complex,
    Facts,
    Integer,
    Irrational,
    Natural,
    Number,
    Rational,
    Real,
    Transcendental,
    Truth,
    Whole,
    _evidence,
)
from .rational import RationalValue


def _rational_facts(value: Any, *, finite: bool, integral: bool, reason: str) -> Facts:
    if not finite:
        return _evidence({Number: False}, reason)
    return _evidence(
        {
            Rational: True,
            Integer: integral,
            Whole: integral and bool(value >= 0),
            Natural: integral and bool(value >= 1),
        },
        reason,
    )


def _sympy_facts(value: sp.Expr) -> Facts:
    reason = "SymPy assumptions about the supplied expression"
    if value.is_commutative is not True:
        raise TypeError("Only commutative scalar SymPy expressions are supported")
    if value is sp.nan or value.is_finite is False:
        return _evidence({Number: False}, "SymPy nonfinite scalar")
    if isinstance(value, sp.Float):
        # The ratio supplies classification evidence only. The original value
        # and its approximation provenance are not replaced or reinterpreted.
        stored = sp.Rational(value)
        return _rational_facts(
            stored,
            finite=True,
            integral=stored.q == 1,
            reason="SymPy Float classified by its stored binary value, not intended exactness",
        )
    facts: Facts = {}
    for domain, answer in (
        (Complex, value.is_complex),
        (Real, value.is_real),
        (Rational, value.is_rational),
        (Integer, value.is_integer),
        (Irrational, value.is_irrational),
        (Algebraic, value.is_algebraic),
        (Transcendental, value.is_transcendental),
    ):
        facts[domain] = Classification(
            Truth.UNKNOWN if answer is None else Truth.TRUE if answer else Truth.FALSE,
            reason,
        )
    if value.is_integer is True:
        for domain, answer in (
            (Whole, value.is_nonnegative),
            (Natural, value.is_positive),
        ):
            facts[domain] = Classification(
                Truth.UNKNOWN
                if answer is None
                else Truth.TRUE
                if answer
                else Truth.FALSE,
                reason,
            )
    return facts


def builtin_facts(value: object) -> Facts | None:
    """Return facts, or None for an unsupported input type."""
    if isinstance(value, FixedValue):
        ratio = value._ratio()
        return _rational_facts(
            ratio,
            finite=True,
            integral=ratio.denominator == 1,
            reason="FixedValue classified by coefficient times step, not intended exactness",
        )
    if isinstance(value, ComplexValue):
        if not value.imag:
            p, q = value.real.value.as_integer_ratio()
            return _rational_facts(
                p,
                finite=True,
                integral=q == 1,
                reason="ComplexValue with zero imaginary part, classified by stored real value",
            )
        return _evidence(
            {Algebraic: True, Real: False},
            "Finite ComplexValue with rational binary components and nonzero imaginary part",
        )
    if isinstance(value, FloatingValue):
        p, q = value.value.as_integer_ratio()
        return _rational_facts(
            p,
            finite=True,
            integral=q == 1,
            reason="FloatingValue classified by stored binary value, not intended exactness",
        )
    if isinstance(value, RationalValue):
        return _rational_facts(
            value.value,
            finite=True,
            integral=value.value.q == 1,
            reason="Exact RationalValue in sympy/Rational",
        )
    if isinstance(value, IntegerValue):
        return _rational_facts(
            int(value),
            finite=True,
            integral=True,
            reason=f"Checked IntegerValue in {value.dtype.backend.value}/{value.dtype.name}",
        )
    if isinstance(value, (bool, np.bool_)) or value is sp.true or value is sp.false:
        return _evidence(
            {Number: False}, "Boolean values are outside the numeric domains"
        )
    if isinstance(value, np.integer) and value.dtype.kind in "iu":
        return _rational_facts(
            value, finite=True, integral=True, reason="NumPy integer value"
        )
    if isinstance(value, np.floating):
        finite = bool(np.isfinite(value))
        return _rational_facts(
            value,
            finite=finite,
            integral=finite and bool(value == np.trunc(value)),
            reason="NumPy floating scalar inspected at its original precision",
        )
    if isinstance(value, np.complexfloating):
        if not bool(np.isfinite(value)):
            return _evidence({Number: False}, "NumPy nonfinite complex value")
        if value.imag == 0:
            return builtin_facts(value.real)
        return _evidence(
            {Algebraic: True, Real: False},
            "NumPy complex scalar with rational components",
        )
    if isinstance(value, int):
        return _rational_facts(
            value, finite=True, integral=True, reason="Python exact integer"
        )
    if isinstance(value, Fraction):
        return _rational_facts(
            value,
            finite=True,
            integral=value.denominator == 1,
            reason="Python exact Fraction",
        )
    if isinstance(value, Decimal):
        finite = value.is_finite()
        return _rational_facts(
            value,
            finite=finite,
            integral=finite and value == value.to_integral_value(),
            reason="Decimal value inspected without binary conversion",
        )
    if isinstance(value, float):
        return _rational_facts(
            value,
            finite=math.isfinite(value),
            integral=value.is_integer(),
            reason="Python float classified by its stored binary value, not intended exactness",
        )
    if isinstance(value, complex):
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            return _evidence({Number: False}, "Python nonfinite complex value")
        if value.imag == 0:
            return builtin_facts(value.real)
        return _evidence(
            {Algebraic: True, Real: False},
            "Python complex value with rational components",
        )
    if isinstance(value, sp.Expr):
        return _sympy_facts(value)
    return None
