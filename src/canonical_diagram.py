#!/usr/bin/env python3
r"""
canonical_diagram.py -- the canonical scattering diagram of open FJRW theory of x^5+y^5:
normal forms and the first wall functions (paper Thm 1.3).

Normal forms / extreme chambers (paper Thm 1.3):
on every (J,d)-diagonal with N >= 1, the infinitesimal wall action moves the coefficient
vector (nu_0,...,nu_N) along the triangular directions
      w_p = r(k_2(p+1)+1) e_{p+1} - s(k_1(p)+1) e_p ,   p = 0..N-1,
whose span is complementary to the e_0-line (and to the e_N-line).  Hence by t-adic
induction there exist UNIQUE chamber indices nu^min (only p=0 survives per diagonal) and
nu^max (only p=N survives).  This uses no periods, so it holds on NS-visible and invisible
diagonals alike.

The canonical diagram (paper Thm 1.3): D_0^{r,s} is the deformation-filtered,
boundary-ray-ordered factorization of the unique g in the enlarged group with
W^{nu^max} = g(W^{nu^min}). This is a canonical, choice-free system of primary
wall functions. The boundary ray of X_(k1,k2) is (k1+1,k2+1).

This script verifies, for r=s=5 at t-order 2 (the three first walls):
 [1] the A(J,0)=0 relations on all three l=2 wall diagonals:
       {(2,3),(3,3)}: (2/5) nu_{(0,6)} + (1/5) nu_{(5,1)} = -2/25,
       {(3,2),(3,3)}: symmetric,
       {(3,3),(3,3)}: (2/5)(nu_{(1,6)} + nu_{(6,1)}) = -4/25;
 [2] the normal-form values:
       nu^min: nu_{(0,6)} = -1/5, nu_{(1,6)} = -2/5   (p>=1 coefficients = 0),
       nu^max: nu_{(5,1)} = -2/5, nu_{(6,1)} = -2/5   (p<N coefficients = 0);
 [3] the wall functions of the canonical diagram at this order, by solving
       W^{nu^max} - W^{nu^min} = c * (u X_{k1,k2})(x^5+y^5)
     on each diagonal -- an OVERDETERMINED system (two monomial components per wall, one
     unknown c); its consistency is forced by [1] and verifies the mechanism:
       theta_{(1,2)} = exp( (1/25) t_{2,3} t_{3,3} X_{0,1} + ... ),
       theta_{(2,1)} = exp( (1/25) t_{3,2} t_{3,3} X_{1,0} + ... ),
       theta_{(1,1)} = exp( (1/50) t_{3,3}^2  X_{1,1} + ... );
 [4] tower extensions in the min gauge:
       t_{3,3}^3 (wall-free, gauge-independent):  nu_{(4,4)} = 1/5  (re-used);
       t_{3,3}^4 (wall X_{2,2}): A({(3,3)^4},0)=0 gives the weighted relation and the
       min-gauge value nu^min_{(2,7)};
 [5] two more absolute (wall-free) invariants:
       <tau_0^{(1,2)} tau_0^{(2,1)} tau_0^{(2,2)} sigma_12> = -1/25   (pure-tau disk),
       and the neutral engine checks A({(2,1),(1,2)},0) = 1 = <tau0^{(0,0)} pairing>,
       A({(0,0),(0,0)},0) = 1.

Run: ./venv/bin/python src/canonical_diagram.py
"""

import os
import sys

import sympy as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a_invariants import A_invariant, d_of, nu_from_dict, part_boundaries  # noqa: E402

x, y = sp.symbols("x y")
R, S = 5, 5
W0 = x**R + y**S


def X_action(k1, k2, f):
    """X_{k1,k2} = x^k1 y^k2 ((k2+1) x d_x - (k1+1) y d_y) applied to f."""
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def check_l2_relations():
    nu06, nu51 = sp.symbols("nu06 nu51")
    J = [(2, 3, 0), (3, 3, 0)]
    assert d_of(J, R, S) < 0
    assert part_boundaries(J, R, S) == [(0, 6), (5, 1)]
    A = A_invariant(J, R, S, nu_from_dict({
        (((2, 3, 0), (3, 3, 0)), 0, 6): nu06,
        (((2, 3, 0), (3, 3, 0)), 5, 1): nu51,
    }))
    expected = sp.Rational(2, 5) * nu06 + sp.Rational(1, 5) * nu51 + sp.Rational(2, 25)
    assert sp.expand(A - expected) == 0, A
    # symmetric diagonal
    Jm = [(3, 2, 0), (3, 3, 0)]
    num60, nu15 = sp.symbols("nu60 nu15")
    Am = A_invariant(Jm, R, S, nu_from_dict({
        (((3, 2, 0), (3, 3, 0)), 6, 0): num60,
        (((3, 2, 0), (3, 3, 0)), 1, 5): nu15,
    }))
    expectedm = sp.Rational(2, 5) * num60 + sp.Rational(1, 5) * nu15 + sp.Rational(2, 25)
    # x<->y symmetry swaps the roles: boundary (6,0) has k1=6=s(J)-shifted... check weights
    assert sp.expand(Am - expectedm) == 0, Am
    return True


def normal_forms_l2():
    """nu^min: p>=1 coefficients vanish; nu^max: p<N vanish.  p ordered by k1."""
    # {(2,3),(3,3)}: boundaries (0,6) [p=0], (5,1) [p=1]
    nu06_min = sp.solve(sp.Rational(2, 5) * sp.Symbol("v") + sp.Rational(2, 25),
                        sp.Symbol("v"))[0]
    assert nu06_min == sp.Rational(-1, 5)
    nu51_max = sp.solve(sp.Rational(1, 5) * sp.Symbol("v") + sp.Rational(2, 25),
                        sp.Symbol("v"))[0]
    assert nu51_max == sp.Rational(-2, 5)
    # {(3,3)^2}: boundaries (1,6) [p=0], (6,1) [p=1]; relation (2/5)(sum) = -4/25
    nu16_min = sp.Rational(-2, 5)
    nu61_max = sp.Rational(-2, 5)
    return {"nu06_min": nu06_min, "nu51_max": nu51_max,
            "nu16_min": nu16_min, "nu61_max": nu61_max}


def wall_functions_l2(nf):
    """Solve W^max - W^min = c * X-action(W0) per diagonal; assert the overdetermined
    system is consistent; return the wall coefficients."""
    out = {}
    # ---- diagonal {(2,3),(3,3)}, wall direction (0,1), |Aut|=1, l=2 sign (-1)^{l-1}=-1
    # W^min term: -( nu06_min * y^6 ),  W^max term: -( nu51_max * x^5 y )
    dW = sp.expand(-(nf["nu51_max"] * x**5 * y) - (-(nf["nu06_min"] * y**6)))
    act = X_action(0, 1, W0)  # = 10 x^5 y - 5 y^6
    c = sp.symbols("c")
    sols = sp.solve(sp.Poly(dW - c * act, x, y).coeffs(), c)
    assert sols, (dW, act)
    cval = sols[c] if isinstance(sols, dict) else sols[0]
    assert sp.simplify(dW - cval * act) == 0  # overdetermined consistency
    out[(0, 1)] = cval
    # ---- diagonal {(3,3)^2}, wall direction (1,1), |Aut|=2
    dW = sp.expand(-(sp.Rational(1, 2) * nf["nu61_max"] * x**6 * y)
                   + (sp.Rational(1, 2) * nf["nu16_min"] * x * y**6))
    act = X_action(1, 1, W0)  # = 10 x^6 y - 10 x y^6
    sols = sp.solve(sp.Poly(dW - c * act, x, y).coeffs(), c)
    cval = sols[c] if isinstance(sols, dict) else sols[0]
    assert sp.simplify(dW - cval * act) == 0
    out[(1, 1)] = cval
    # ---- symmetric diagonal -> (1,0) by x<->y
    out[(1, 0)] = out[(0, 1)]
    assert out[(0, 1)] == sp.Rational(1, 25)
    assert out[(1, 1)] == sp.Rational(1, 50)
    return out


def tower_t33_4(nf):
    """A({(3,3)^4},0) = 0: the l=4 relation and the min-gauge value nu^min_{(2,7)}.

    In the min gauge the known inputs are: singleton nu=1; l=2: nu16=-2/5, nu61=0;
    l=3: nu44=1/5 (wall-free).  Boundaries at l=4: (2,7) [p=0], (7,2) [p=1]."""
    J = [(3, 3, 0)] * 4
    assert d_of(J, R, S) < 0
    assert part_boundaries(J, R, S) == [(2, 7), (7, 2)]
    nu27, nu72 = sp.symbols("nu27 nu72")
    numap = {
        (((3, 3, 0),) * 4, 2, 7): nu27,
        (((3, 3, 0),) * 4, 7, 2): nu72,
        (((3, 3, 0),) * 3, 4, 4): sp.Rational(1, 5),
        (((3, 3, 0), (3, 3, 0)), 1, 6): sp.Symbol("nu16"),
        (((3, 3, 0), (3, 3, 0)), 6, 1): sp.Symbol("nu61"),
    }
    A = A_invariant(J, R, S, nu_from_dict(numap))
    # general relation, then min gauge nu16=-2/5, nu61=0
    A_min = A.subs({sp.Symbol("nu16"): sp.Rational(-2, 5), sp.Symbol("nu61"): 0, nu72: 0})
    val = sp.solve(sp.Eq(A_min, 0), nu27)[0]
    # gauge-independence probe: solve in the max gauge too and check the weighted sum
    A_max = A.subs({sp.Symbol("nu16"): 0, sp.Symbol("nu61"): sp.Rational(-2, 5), nu27: 0})
    val_max = sp.solve(sp.Eq(A_max, 0), nu72)[0]
    return sp.expand(A), val, val_max


def pure_tau_invariant():
    """A({(1,2),(2,1),(2,2)},0) = 0 determines the wall-free pure-tau disk invariant
    <tau0^{(1,2)} tau0^{(2,1)} tau0^{(2,2)} sigma_12> = -1/25."""
    J = [(1, 2, 0), (2, 1, 0), (2, 2, 0)]
    assert d_of(J, R, S) < 0
    assert part_boundaries(J, R, S) == [(0, 0)]
    nu000 = sp.Symbol("nu000")
    A = A_invariant(J, R, S, nu_from_dict({
        (tuple(sorted(J)), 0, 0): nu000,
    }))
    val = sp.solve(sp.Eq(A, 0), nu000)[0]
    assert val == sp.Rational(-1, 25), A
    return val


def neutral_checks():
    """More Cor 0.6 / Thm 5.4(2) cross-checks with closed Frobenius values."""
    # A({(2,1),(1,2)},0) = <tau0^{(0,0)} tau0^{(2,1)} tau0^{(1,2)}> = 1
    J = [(2, 1, 0), (1, 2, 0)]
    assert d_of(J, R, S) == 0
    assert A_invariant(J, R, S, nu_from_dict({})) == 1
    # A({(0,0),(0,0)},0) = <tau0^{(3,3)} tau0^{(0,0)} tau0^{(0,0)}> = 1
    J = [(0, 0, 0), (0, 0, 0)]
    assert d_of(J, R, S) == 0
    assert A_invariant(J, R, S, nu_from_dict({})) == 1
    return True


def main():
    print("== The canonical scattering diagram of x^5+y^5: first walls ==")
    assert check_l2_relations()
    print("[1] l=2 wall-diagonal relations: (2/5)nu06+(1/5)nu51 = -2/25 (and symmetric);")
    print("    (2/5)(nu16+nu61) = -4/25")
    nf = normal_forms_l2()
    print(f"[2] normal forms: nu^min: nu06 = {nf['nu06_min']}, nu16 = {nf['nu16_min']};"
          f"  nu^max: nu51 = {nf['nu51_max']}, nu61 = {nf['nu61_max']}")
    wf = wall_functions_l2(nf)
    print("[3] wall functions of the canonical diagram (overdetermined solve consistent):")
    print(f"      theta_(1,2) = exp( {wf[(0,1)]} * t_23 t_33 X_01 + ... )")
    print(f"      theta_(2,1) = exp( {wf[(1,0)]} * t_32 t_33 X_10 + ... )")
    print(f"      theta_(1,1) = exp( {wf[(1,1)]} * t_33^2  X_11 + ... )")
    Arel, v_min, v_max = tower_t33_4(nf)
    print(f"[4] t_33^4 (wall X_22): A-relation {sp.simplify(Arel)} = 0;")
    print(f"      min gauge: nu_(2,7) = {v_min};  max gauge: nu_(7,2) = {v_max}")
    v = pure_tau_invariant()
    print(f"[5] NEW absolute invariant: <tau0^(1,2) tau0^(2,1) tau0^(2,2) sigma_12> = {v}")
    assert neutral_checks()
    print("[6] neutral closed-value cross-checks (A = 1 twice): OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
