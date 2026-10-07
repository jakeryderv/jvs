"""Cross-wrapper equality, bounded diagnostics, and runtime typing contracts."""

import inspect
import os
import subprocess
import sys
from fractions import Fraction
from itertools import product
from typing import Any, get_type_hints

import numpy as np
import pytest
import sympy as sp

from jvs.numeric import (
    ComplexValue,
    ExactDType,
    FixedDType,
    FixedValue,
    FloatingValue,
    IntegerValue,
    NumPyDType,
    RationalValue,
)

INTEGER = ExactDType.integer()
RATIONAL = ExactDType.rational()
WRAPPERS = (IntegerValue, RationalValue, FloatingValue, ComplexValue, FixedValue)

# NumPy stubs do not model third-party dispatch or deliberately invalid inputs.
_equal: Any = np.equal
_not_equal: Any = np.not_equal


def representations() -> list[tuple[Any, tuple[Fraction, Fraction]]]:
    result = []
    for p, q in [(0, 1), (1, 1), (-1, 1), (3, 2)]:
        oracle = (Fraction(p, q), Fraction(0))
        if q == 1:
            result.extend(
                [
                    (IntegerValue(p, dtype=INTEGER), oracle),
                    (IntegerValue(p, dtype=NumPyDType("int16")), oracle),
                ]
            )
        result.extend(
            [
                (RationalValue(p, q, dtype=RATIONAL), oracle),
                (
                    FixedValue(
                        Fraction(p, q),
                        dtype=FixedDType(NumPyDType("int16"), step=Fraction(1, 2)),
                    ),
                    oracle,
                ),
                (
                    FixedValue(
                        Fraction(p, q), dtype=FixedDType(INTEGER, step=Fraction(1, 10))
                    ),
                    oracle,
                ),
                (FloatingValue(Fraction(p, q), dtype=NumPyDType("float32")), oracle),
                (
                    ComplexValue(Fraction(p, q), -0.0, dtype=NumPyDType("complex128")),
                    oracle,
                ),
            ]
        )
    result.append(
        (ComplexValue(1, 2, dtype=NumPyDType("complex64")), (Fraction(1), Fraction(2)))
    )
    return result


def test_numpy_equality_matches_exact_component_oracle_in_every_order() -> None:
    for (a, expected_a), (b, expected_b) in product(representations(), repeat=2):
        expected = expected_a == expected_b
        assert (a == b) is expected
        assert _equal(a, b) is expected
        assert _not_equal(a, b) is (not expected)
        if expected:
            assert hash(a) == hash(b)


@pytest.mark.parametrize("index", range(5))
def test_ufunc_guards_still_reject_arithmetic_arrays_and_options(index: int) -> None:
    values: list[Any] = [
        IntegerValue(1, dtype=INTEGER),
        RationalValue(1, dtype=RATIONAL),
        FloatingValue(1, dtype=NumPyDType("float64")),
        ComplexValue(1, dtype=NumPyDType("complex128")),
        FixedValue(1, dtype=FixedDType(INTEGER, step=Fraction(1, 10))),
    ]
    value = values[index]
    assert _equal(np.array(1, dtype=np.int64), value)
    assert _equal(value, np.array(1, dtype=np.int64))
    for ufunc in (np.add, np.subtract, np.multiply, np.divide):
        with pytest.raises(TypeError, match="ufunc"):
            ufunc(value, value)
    with pytest.raises(TypeError):
        _equal.reduce(value)
    for options in ({"where": True}, {"out": np.array(False)}, {"dtype": bool}):
        with pytest.raises(TypeError):
            _equal(value, value, **options)
    for other in (
        np.array([1]),
        np.array(1.0),
        np.array(1 + 0j),
        np.array(True),
        Fraction(1),
        1.0,
    ):
        for a, b in ((value, other), (other, value)):
            with pytest.raises(TypeError):
                _equal(a, b)
    for a, b in ((value, object()), (object(), value)):
        # An unresolved comparison must remain unresolved, not become False.
        with pytest.raises(TypeError):
            _equal(a, b)


def test_integer_ufunc_defers_when_other_wrapper_owns_comparison() -> None:
    a = IntegerValue(1, dtype=INTEGER)
    b = RationalValue(1, dtype=RATIONAL)
    assert a.__array_ufunc__(np.equal, "__call__", a, b) is NotImplemented
    assert a.__array_ufunc__(np.not_equal, "__call__", a, b) is NotImplemented
    assert _equal(a, b)
    assert not _not_equal(a, b)


@pytest.mark.parametrize("sign", [-1, 1])
def test_huge_integer_and_rational_reprs_are_bounded_without_value_changes(
    sign: int,
) -> None:
    huge = 10**5000
    before = sys.get_int_max_str_digits()
    integer = IntegerValue(sign * huge, dtype=INTEGER)
    numerator = RationalValue(sign * huge, 3, dtype=RATIONAL)
    denominator = RationalValue(sign, huge, dtype=RATIONAL)
    both = RationalValue(sign * (huge + 1), huge, dtype=RATIONAL)
    for value in (integer, numerator, denominator, both):
        rendered = repr(value)
        assert len(rendered) < 350
        assert "..." in rendered and "bits=" in rendered
        assert "dtype=" in rendered
        assert str(value) == rendered
    assert int(integer) == sign * huge
    assert numerator.numerator == sign * huge
    assert numerator.denominator == 3
    assert denominator.denominator == huge
    assert both.numerator == sign * (huge + 1)
    assert both.denominator == huge
    assert sys.get_int_max_str_digits() == before
    # NumPy's own TypeError diagnostic includes the operand's repr.
    with pytest.raises(TypeError):
        _equal(integer, object())


def test_ordinary_integer_and_rational_reprs_keep_full_decimal_values() -> None:
    for value in (0, 1, -123, 2**256 - 1, -(2**256 - 1)):
        rendered = repr(IntegerValue(value, dtype=INTEGER))
        assert rendered.startswith(f"IntegerValue({value}, dtype=")
        assert "..." not in rendered
    assert repr(RationalValue(-7, 3, dtype=RATIONAL)).startswith(
        "RationalValue(-7, 3, dtype="
    )


def test_reprs_respect_lowest_python_digit_limit_in_fresh_process() -> None:
    code = """
import sys
from fractions import Fraction
from jvs.numeric import ExactDType, FixedDType, FixedValue, IntegerValue, RationalValue
assert sys.get_int_max_str_digits() == 640
n = 10**5000
for v in (IntegerValue(n, dtype=ExactDType.integer()), RationalValue(-n, n+1, dtype=ExactDType.rational())):
    assert len(repr(v)) < 350
assert sys.get_int_max_str_digits() == 640
n = 10**5000
dtype = FixedDType(ExactDType.integer(), step=Fraction(n+1, n))
assert len(repr(FixedValue.from_coefficient(n, dtype=dtype))) < 650
assert sys.get_int_max_str_digits() == 640
"""
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONINTMAXSTRDIGITS": "640"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_runtime_annotations_resolve_without_custom_namespaces() -> None:
    assert get_type_hints(IntegerValue.__truediv__)["return"] is RationalValue
    assert get_type_hints(IntegerValue.__rtruediv__)["return"] is RationalValue
    assert get_type_hints(RationalValue.to_integer)["return"] is IntegerValue
    for cls in (*WRAPPERS, FixedDType):
        get_type_hints(cls)
        for member in vars(cls).values():
            if isinstance(member, classmethod):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if inspect.isfunction(member):
                get_type_hints(member)


@pytest.mark.parametrize(
    "module",
    [
        "jvs.numeric.integer",
        "jvs.numeric.rational",
        "jvs.numeric.fixed",
        "jvs.numeric.floating",
    ],
)
def test_import_and_annotations_in_fresh_process(module: str) -> None:
    code = f"""
from importlib import import_module
from typing import get_type_hints
import_module({module!r})
from jvs.numeric.integer import IntegerValue
from jvs.numeric.rational import RationalValue
from jvs.numeric.dtype import ExactDType
assert get_type_hints(IntegerValue.__truediv__)["return"] is RationalValue
assert get_type_hints(RationalValue.to_integer)["return"] is IntegerValue
one = IntegerValue(1, dtype=ExactDType.integer())
assert (one / one).to_integer(ExactDType.integer()) == one
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_foreign_comparison_can_answer_before_wrapper_dispatch() -> None:
    value = IntegerValue(1, dtype=INTEGER)
    with pytest.raises(TypeError):
        value.__eq__(sp.nan)
    assert (sp.nan == value) is False
