#!/usr/bin/env python3
r"""
transport_engine.py -- the first NONLINEAR test of the broken-line/transport mechanism:
the pure-t33 sector of x^5+y^5 through order t^4.

Objects (all previously verified): the t33-tower normal forms
    nu^min: nu_(1,6) = -2/5 (t^2), nu_(4,4) = 1/5 (t^3, invariant), nu_(2,7) = -12/25 (t^4)
    nu^max: the x<->y mirror,
potential dressing (-1)^{l-1} nu/|Aut(A)|:
    W^min = x^5 + y^5 + t x^3y^3 + (1/5) t^2 x y^6 + (1/30) t^3 x^4y^4 + (1/50) t^4 x^2y^7
    W^max = x^5 + y^5 + t x^3y^3 + (1/5) t^2 x^6 y + (1/30) t^3 x^4y^4 + (1/50) t^4 x^7y^2
and the order-2 wall function theta_(1,1) = exp((1/50) t^2 X_{1,1}).

Certified here:
 [1] the general ansatz g = exp((1/50) t^2 X_11 + c4 t^4 X_22) (the only graded pieces of
     the algebra in this sector) satisfies g(W^min) = W^max through t^4 iff c4 = 0 --
     BOTH diagonal slots (x^7y^2 and x^2y^7) give c4 = 0 independently: the
     overdetermined nonlinear system is consistent, the entire order-4 motion being
     produced by the SECOND-ORDER term of the order-2 wall plus its action on the
     order-2 coefficients.  In particular the canonical diagram has NO X_22-wall at
     t33^4, and the transport independently confirms nu_(2,7)^min = -12/25 from the
     A-engine (a genuinely nonlinear cross-validation of the corrected framework).
 [2] the transport identity (paper Thm 1.5) holds in this family through t^4:
     with seeds {x^5 (home: bottom), y^5 (home: top), t x^3y^3 and (1/30)t^3 x^4y^4
     (transport-invariant, on-ray)}, the transport potential in the bottom chamber
       T(bottom) = x^5 + t x^3y^3 + (1/30) t^3 x^4y^4 + theta^{-1}(y^5)
     equals W^min EXACTLY: the single bend of y^5 gives the t^2 coefficient (+1/5 xy^6)
     and the DOUBLE BEND (second-order exp term) gives the t^4 coefficient
     (+1/50 x^2y^7), i.e. the broken-line count computes
       <tau0^(3,3)^4 sigma_1^2 sigma_2^7 sigma_12>^min = -12/25
     from the wall function alone.  Symmetrically T(top) = W^max.
 [3] the invariant t^3 slot (x^4y^4) is fixed by the wall exactly (X_11(x^3y^3) = 0,
     X_11(x^4y^4) = 0): it rides as a seed, as required by the transport theorem.

Run: ./venv/bin/python src/transport_engine.py
"""

import sympy as sp

x, y, t, c4 = sp.symbols("x y t c4")
ORDER = 5  # truncate t-degree >= 5


def X(k1, k2, f):
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def trunc(f):
    f = sp.expand(f)
    return sum(term for term in f.as_ordered_terms()
               if sp.degree(term, t) < ORDER)


def exp_action(vfun, f):
    """exp(v)(f) with v(g) := vfun(g) carrying t-degree >= 2; truncated at ORDER."""
    out = sp.Integer(0)
    term = f
    fact = 1
    for n in range(0, ORDER // 2 + 1):
        out += term / fact
        term = trunc(vfun(term))
        fact *= (n + 1)
        if term == 0:
            break
    return trunc(sp.expand(out))


def v(f):  # the sector wall element: (1/50) t^2 X_11 + c4 t^4 X_22
    return trunc(sp.Rational(1, 50) * t**2 * X(1, 1, f) + c4 * t**4 * X(2, 2, f))


def v_inv(f):
    return trunc(-v(f))


W0 = x**5 + y**5
Wmin = (W0 + t * x**3 * y**3 + sp.Rational(1, 5) * t**2 * x * y**6
        + sp.Rational(1, 30) * t**3 * x**4 * y**4
        + sp.Rational(1, 50) * t**4 * x**2 * y**7)
Wmax = (W0 + t * x**3 * y**3 + sp.Rational(1, 5) * t**2 * x**6 * y
        + sp.Rational(1, 30) * t**3 * x**4 * y**4
        + sp.Rational(1, 50) * t**4 * x**7 * y**2)


def main():
    print("== Nonlinear transport test: the pure-t33 sector through t^4 ==")
    # [1] solve g(Wmin) = Wmax for c4, slot by slot
    diff = sp.expand(exp_action(v, Wmin) - Wmax)
    poly = sp.Poly(diff, x, y, t)
    eqs = set()
    for coeff in poly.coeffs():
        c = sp.simplify(coeff)
        if c != 0:
            eqs.add(sp.simplify(c))
    sols = {sp.solve(sp.Eq(e, 0), c4)[0] for e in eqs}
    assert sols == {0}, (eqs, sols)
    print("[1] g(W^min) = W^max through t^4  <=>  c4 = 0 (both slots independently);")
    print("    the X_22-wall coefficient of the canonical diagram at t33^4 VANISHES;")
    print("    the order-4 motion = second-order term of the t^2 wall + its action on")
    print("    the t^2 coefficients -- independent nonlinear confirmation of nu_(2,7) = -12/25")

    # [2] the anchor: T(bottom) = x^5 + invariant seeds + theta^{-1}(y^5)
    def v0(f):  # the wall with c4 = 0
        return trunc(sp.Rational(1, 50) * t**2 * X(1, 1, f))

    def v0_inv(f):
        return trunc(-v0(f))

    T_bottom = trunc(x**5 + t * x**3 * y**3 + sp.Rational(1, 30) * t**3 * x**4 * y**4
                     + exp_action(v0_inv, y**5))
    assert sp.expand(T_bottom - Wmin) == 0, sp.expand(T_bottom - Wmin)
    T_top = trunc(y**5 + t * x**3 * y**3 + sp.Rational(1, 30) * t**3 * x**4 * y**4
                  + exp_action(v0, x**5))
    assert sp.expand(T_top - Wmax) == 0
    print("[2] ANCHOR holds through t^4: T(bottom) = W^min and T(top) = W^max exactly;")
    print("    single bend of y^5 -> the t^2 coefficient; DOUBLE bend -> the t^4")
    print("    coefficient: the broken-line count computes nu^min_(2,7) = -12/25")

    # [3] invariance of the on-ray seeds
    assert X(1, 1, x**3 * y**3) == 0 and X(1, 1, x**4 * y**4) == 0
    print("[3] on-ray seeds x^3y^3, x^4y^4 fixed by the wall exactly (ride as seeds)")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
