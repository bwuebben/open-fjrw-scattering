#!/usr/bin/env python3
r"""
farside_test.py -- the smallest far-side residual case, solved completely:
D = {(0,0,d=2), (1,1,0)} at r = s = 5.

Contents:
 [1] the unit-twist tower t_{0,0,2} (singleton, N=2, exact by paper Thm 7.7): slots
     (0,10),(5,5),(10,0), A = +1; chain gives nu_0 = 25/6 and NONZERO near-axis wall
     functions c = 5/6 on both rays (1,6) and (6,1).  The far-side gate is OPEN.
 [2] the D-diagonal (slots (1,6),(6,1), N=1, own wall X_{1,1} on ray (2,2)):
     building W(c0) from the A-conditions and transporting up through the three
     rays, the max normal form forces the own wall function c_D = 1, and the
     A-condition A(D) = 0 holds in all four chambers.  Chamber vectors:
       c0: (35/6, 0), c1: (35/6, -25/6), c2: (-25/6, 35/6), c3: (0, 35/6).
 [3] the far-side bend is real: the t_{1,1}-seed (xy, nu = 1) bending down at the
     near-axis wall u X_{5,0} contributes +25/6 to slot (6,1) in the bottom chamber.
     With the STRICTLY-BELOW on-ray convention the anchor FAILS (constant defect
     -25/6 at slot (6,1) -- the chamber-independence of Thm 1.5 confirmed on a genuinely
     composite case).  With the DIAGONAL-SIDE convention (an on-ray slot takes
     the chamber on the primitive-form-diagonal side of its ray) the anchor
     HOLDS EXACTLY in all four chambers: the convention is forced, and the
     smallest far-side case closes.

Run: ./venv/bin/python src/farside_test.py
"""

import sympy as sp

x, y, u, v = sp.symbols("x y u v")  # u = t_{0,0,2}, v = t_{1,1,0}


def X(k1, k2, f):
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def trunc(f):
    f = sp.expand(f)
    out = sp.Integer(0)
    for t in f.as_ordered_terms():
        if sp.degree(t, u) <= 1 and sp.degree(t, v) <= 1:
            out += t
    return out


def exp_action(vfun, f):
    out, term, fact = sp.Integer(0), f, 1
    for n in range(3):
        out += term / fact
        term = trunc(vfun(term))
        fact *= (n + 1)
        if term == 0:
            break
    return trunc(sp.expand(out))


def wall(c, mono, k1, k2):
    return lambda f: exp_action(lambda g: trunc(c * mono * X(k1, k2, g)), f)


def coeff(W, mono):
    return sp.expand(W).coeff(mono) if sp.expand(W).coeff(mono) != sp.nan else 0


def main():
    print("== The smallest far-side case: D = {(0,0,2),(1,1,0)}, r = s = 5 ==")

    # [1] unit-twist tower (exact chain): nu_0 = 25/6, walls 5/6 on rays (1,6),(6,1)
    nu0_ax = sp.Rational(25, 6)
    c_ax = sp.Rational(5, 6)
    # weights (6/25,1/25,6/25); A = +1 with min form: (6/25)(25/6) = 1
    assert sp.Rational(6, 25) * nu0_ax == 1
    # chain: c1 = nu0/b1 = (25/6)/5 = 5/6; a1 = 30, b2 = 30 -> c2 = c1; c2*a2 = 25/6 ok
    assert nu0_ax / 5 == c_ax and c_ax * 5 == nu0_ax
    print(f"[1] unit-twist tower t_(0,0,2): nu_0 = {nu0_ax}; wall functions {c_ax} on BOTH")
    print("    near-axis rays (1,6),(6,1) -- NONZERO: the far-side gate is open")

    # [2] build W(c0) and transport up; solve c_D from the max normal form
    cD = sp.Symbol("cD")
    W0 = (x**5 + y**5 + v * x * y
          + nu0_ax * u * y**10                      # axis bal_0 (min form)
          + sp.Rational(35, 6) * u * v * x * y**6)  # D slot (1,6); (6,1) = 0
    th_low = wall(c_ax, u, 5, 0)     # ray (6,1)
    th_own = wall(cD, u * v, 1, 1)   # ray (2,2)
    th_hi = wall(c_ax, u, 0, 5)      # ray (1,6)
    W1 = th_low(W0)
    W2 = th_own(W1)
    W3 = th_hi(W2)
    # max normal form on D: slot (1,6) coefficient of W3 must vanish
    e16 = sp.expand(W3).coeff(u * v * x * y**6)
    sol = sp.solve(sp.Eq(e16, 0), cD)
    assert sol == [1], sol
    W1, W2, W3 = [sp.expand(Wc.subs(cD, 1)) for Wc in (W1, W2, W3)]
    Wc = [sp.expand(W0), W1, W2, W3]
    vec = [(Wcs.coeff(u * v * x * y**6), Wcs.coeff(u * v * x**6 * y)) for Wcs in Wc]
    expect = [(sp.Rational(35, 6), 0), (sp.Rational(35, 6), sp.Rational(-25, 6)),
              (sp.Rational(-25, 6), sp.Rational(35, 6)), (0, sp.Rational(35, 6))]
    assert vec == expect, vec
    # A-condition A(D) = 0 in all chambers: -(2/5)(w16+w61) + h2(axis)*1 = 0
    for i, Wcs in enumerate(Wc):
        ax = (Wcs.coeff(u * y**10), Wcs.coeff(u * x**5 * y**5), Wcs.coeff(u * x**10))
        h2 = sp.Rational(14, 25) * ax[0] + sp.Rational(4, 25) * ax[1] \
            + sp.Rational(14, 25) * ax[2]
        A = -sp.Rational(2, 5) * (vec[i][0] + vec[i][1]) + h2
        assert sp.simplify(A) == 0, (i, A)
    print(f"[2] own wall function FORCED: c_D = 1; chamber D-vectors {expect};")
    print("    A(D) = 0 verified in ALL FOUR chambers (axis middle seed -25 included)")

    # [3] the anchor, both conventions
    def mkinv(c, mono, k1, k2):
        return lambda f: exp_action(lambda g: trunc(-c * mono * X(k1, k2, g)), f)

    i_low = mkinv(c_ax, u, 5, 0)
    i_own = mkinv(1, u * v, 1, 1)
    i_hi = mkinv(c_ax, u, 0, 5)
    up = [th_low, lambda f: wall(1, u * v, 1, 1)(f), th_hi]
    dn = [i_low, i_own, i_hi]

    def transport(term, h, c):
        if c > h:
            for i in range(h, c):
                term = up[i](term)
        else:
            for i in range(h - 1, c - 1, -1):
                term = dn[i](term)
        return trunc(term)

    def T_of(seed61_home, seed61_val):
        seeds = [
            (x**5, 0), (y**5, 3), (v * x * y, 1),
            (-25 * u * x**5 * y**5, 1),                       # axis middle seed
            (sp.Rational(-25, 6) * u * v * x * y**6, 2),      # (1,6): diag-side c2
            (seed61_val * u * v * x**6 * y, seed61_home),     # (6,1): convention
        ]
        return [sum(transport(s, h, c) for s, h in seeds) for c in range(4)]

    # strictly-below convention: (6,1) on ray (6,1) -> home c0, seed = W(c0)-slot = 0
    T_bad = T_of(0, sp.Integer(0))
    defects = [sp.expand(Wc[c] - T_bad[c]).coeff(u * v * x**6 * y) for c in range(4)]
    assert defects == [sp.Rational(-25, 6)] * 4, defects
    others_ok = all(sp.expand(Wc[c] - T_bad[c]
                              + sp.Rational(25, 6) * u * v * x**6 * y) == 0
                    for c in range(4))
    assert others_ok
    print("[3] STRICTLY-BELOW convention FAILS: constant defect -25/6 at slot (6,1)")
    print("    (Thm 1.5 chamber-independence confirmed on a composite case with foreign bends)")

    # diagonal-side convention: (6,1) below the diagonal -> chamber ABOVE its ray: c1,
    # seed = W(c1)-slot = -25/6
    T_good = T_of(1, sp.Rational(-25, 6))
    for c in range(4):
        assert sp.expand(Wc[c] - T_good[c]) == 0, (c, sp.expand(Wc[c] - T_good[c]))
    print("    DIAGONAL-SIDE convention HOLDS: T(c) = W(c) EXACTLY in all four chambers")
    print("    -> the on-ray convention is FORCED: own chamber = the side of the ray")
    print("       toward the primitive-form diagonal; the smallest far-side case CLOSES")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
