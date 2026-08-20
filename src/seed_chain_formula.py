#!/usr/bin/env python3
r"""
Exact checks for the ordered broken-line chain formula of Lemma 7.9 and
Theorem 7.10 in ``paper/main.tex``.

The script checks the two coefficients that distinguish the formula:
  (1) the factorial-weighted double bend in the pure t_(3,3) tower;
  (2) the inverse-wall sign in the first off-ray far-side correction.

No floating-point arithmetic or external dependency is used.
"""

from fractions import Fraction
from math import factorial


def omega(exponent, lie_degree):
    """Coefficient of X_k(z^a), with boundary direction k + (1,1)."""
    a1, a2 = exponent
    k1, k2 = lie_degree
    return (k2 + 1) * a1 - (k1 + 1) * a2


def repeated_bend(start, coefficient, lie_degree, count, orientation=1):
    """Weight and final exponent for repeated use of one exponential summand."""
    exponent = start
    weight = Fraction(1, factorial(count))
    for _ in range(count):
        weight *= orientation * coefficient * omega(exponent, lie_degree)
        exponent = (exponent[0] + lie_degree[0], exponent[1] + lie_degree[1])
    return weight, exponent


def main():
    print("== Ordered broken-line chain formula ==")

    single, exp_single = repeated_bend(
        (0, 5), Fraction(1, 50), (1, 1), 1, orientation=-1
    )
    double, exp_double = repeated_bend(
        (0, 5), Fraction(1, 50), (1, 1), 2, orientation=-1
    )
    assert (single, exp_single) == (Fraction(1, 5), (1, 6))
    assert (double, exp_double) == (Fraction(1, 50), (2, 7))
    print("[1] pure t_(3,3): single bend = 1/5; factorial-weighted double bend = 1/50")

    far_side, exp_far_side = repeated_bend(
        (2, 1), Fraction(5, 6), (5, 0), 1, orientation=-1
    )
    assert (far_side, exp_far_side) == (Fraction(10, 3), (7, 1))
    assert -far_side == Fraction(-10, 3)
    print("[2] off-ray case: transported lower seed = 10/3; residual seed = -10/3")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
