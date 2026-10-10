#!/usr/bin/env python3
"""
mirror_periods.py -- exact oscillatory-integral periods for the mirror theorem
[GKT, Theorem 0.3],
    int_{Xi_{a,b}} e^{W/hbar} dx dy = delta_{a,0} delta_{b,0} + (closed extended FJRW invariants),
computed for x^5+y^5, to see how a term x^{r-1} y^{s-1} of the potential W enters the
good-basis periods.

The 1-variable periods of x^5, P_a(p) = int_{Xi_a} x^p e^{x^5/hbar} dx (good basis a in 0..r-2),
satisfy (integration by parts, boundary vanishing on the good cycles; Paper 2, Lemma 4.1):
    P_a(p) = delta_{a,p}           (0 <= p <= r-2)
    P_a(r-1) = 0                   (the x^{r-1} moment VANISHES)   <-- the key fact
    P_a(p) = -(hbar*(p-(r-1))/r) * P_a(p-r)    (p >= r)
The two-variable moment is M_{a,b}(p,q) = P_a(p) P_b(q).

RESULT.  A term c x^4 y^4 = c x^{r-1} y^{s-1} of the potential W contributes linearly
(1/hbar) c M_{a,b}(r-1,s-1) = (1/hbar) c P_a(r-1) P_b(s-1) = 0 for every (a,b): it is
INVISIBLE to the good-basis periods at linear order. It first enters through the
quadratic cross-terms x^{r-1}y^{s-1} * x^alpha y^beta = x^{r-1+alpha} y^{s-1+beta}, whose
moment M_{a,b}(r-1+alpha, s-1+beta) is nonzero (= alpha*beta*hbar^2/r^2 at
(a,b)=(alpha-1,beta-1)).  This is why the coefficient of t_{3,3}^3 x^4 y^4 of x^5+y^5
(r(J) = 4 = r-1) is not fixed by a period condition (Paper 2, Remark 4.2).  The
boundary-dependent transport is checked in mixed_sector.py and the other transport scripts.

Run:  python3 mirror_periods.py
"""
import sympy as sp

h = sp.symbols('hbar')


def period(a, p, r=5):
    """P_a(p) = int_{Xi_a} x^p e^{x^r/hbar} dx, good basis a in {0,...,r-2}."""
    if p < 0:
        return sp.Integer(0)
    if p <= r - 2:
        return sp.Integer(1) if p == a else sp.Integer(0)
    if p == r - 1:
        return sp.Integer(0)
    return -sp.Rational(p - (r - 1), r) * h * period(a, p - r, r)


def moment(a, b, p, q, r=5, s=5):
    return period(a, p, r) * period(b, q, s)


def _check(name, cond):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}")
    if not cond:
        raise AssertionError(name)


def run():
    r = s = 5
    print("=" * 74)
    print("Mirror-theorem periods for x^5+y^5 and the x^{r-1}y^{s-1} term")
    print("=" * 74)
    print("\n1-variable periods P_a(p) = int_{Xi_a} x^p e^{x^5/hbar}:")
    print("   p:   " + "  ".join(f"{p:>8}" for p in range(9)))
    for a in range(r - 1):
        print(f"  a={a}: " + "  ".join(f"{str(period(a, p)):>8}" for p in range(9)))

    _check("good-basis normalization P_a(p)=delta_{a,p} for p<=r-2",
           all(period(a, p) == (1 if a == p else 0) for a in range(r - 1) for p in range(r - 1)))
    _check("the x^{r-1} moment vanishes: P_a(4)=0 for all a",
           all(period(a, r - 1) == 0 for a in range(r - 1)))

    print("\nA term ~ x^4 y^4 of W; its linear contribution")
    print("to int_{Xi_{a,b}} e^{W/hbar} is (1/hbar) P_a(4) P_b(4):")
    lead = {(a, b): moment(a, b, r - 1, s - 1) for a in range(r - 1) for b in range(s - 1)}
    _check("x^4 y^4 is INVISIBLE at linear order (all (a,b) give 0)",
           all(v == 0 for v in lead.values()))

    print("\nQuadratic order: x^4 y^4 crossed with x^alpha y^beta -> nonzero moment M(4+a,4+b):")
    ok_pattern = True
    for (al, be) in [(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (3, 1)]:
        nz = {(a, b): sp.simplify(moment(a, b, r - 1 + al, s - 1 + be))
              for a in range(r - 1) for b in range(s - 1)}
        nz = {k: v for k, v in nz.items() if v != 0}
        print(f"     x^4y^4 * x^{al}y^{be}:  {nz}")
        # pattern: nonzero only at (al-1, be-1), value = al*be*hbar^2/25
        exp = {(al - 1, be - 1): sp.Rational(al * be, 25) * h**2}
        if nz != exp:
            ok_pattern = False
    _check("quadratic cross-term moments follow the pattern  M -> alpha*beta*hbar^2/25 at (alpha-1,beta-1)",
           ok_pattern)

    print("\n" + "=" * 74)
    print("RESULT: a term x^{r-1}y^{s-1} of W is invisible to the good-basis periods at linear")
    print("order, because the x^{r-1} moment vanishes; it enters only through the quadratic")
    print("cross-terms, whose moments are listed above (cf. Paper 2, Remark 4.2).")
    print("Boundary-dependent transport is checked in mixed_sector.py and the other transport scripts.")
    print("=" * 74)


if __name__ == "__main__":
    run()
