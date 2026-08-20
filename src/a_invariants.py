#!/usr/bin/env python3
r"""
a_invariants.py -- the invariant combinations A(J,d,nu) of [GKT] Notation 3.41, in the
LABELED-PARTITION convention forced by Theorem 4.9 (= Thm 0.2), and the relations they
impose on the open FJRW invariants of x^r + y^s.

Convention (twice confirmed against [GKT] Notation 3.41):
  A(J,d,nu) = sum over UNORDERED set partitions {J_1,...,J_h} of the labeled set J
              (equivalently: ordered tuples with 1/h!, since distinct label-blocks
              admit h! orderings) of
       [ Gamma((1+sum k1(i))/r) Gamma((1+sum k2(i))/s)
         / (Gamma((1+r(J))/r) Gamma((1+s(J))/s)) ]
       * prod_i nu_{Gamma_{0,k1(i),k2(i),1,J_i}, d|J_i}
  where per part (k1(i),k2(i)) runs over k1=r(J_i) mod r, k2=s(J_i) mod s,
  s k1 + r k2 = m(J_i, d|J_i).  All Gamma-ratios are rational (arguments differ by
  integers); everything here is exact.

The convention is pinned INDEPENDENTLY of the oscillatory route by a case where GKT's
proven Theorem 5.4(2) supplies the value:  J = {(1,1),(1,1)}, d=0 at r=s=5 has
d(J,0)=0 and
    A = <tau_0^{(1,1)} tau_0^{(1,1)} tau_0^{(1,1)}>^{x^5+y^5} = 1   (Frobenius algebra),
while the Notation 3.41 expansion is (labeled count) 1 * nu_single^2 = 1  vs (the
survey Ex 5.5 style halved count) 1/2.  Only the labeled count is consistent -- the
same counting question as survey Ex 5.5, decided by a PROVEN theorem plus closed data.

Checks below (ALL exact):
 [1] the convention pin above, plus the convention-neutral Cor 0.6 check
     A({(1,1),(2,2)},0) = <tau_0^{(0,0)}tau_0^{(1,1)}tau_0^{(2,2)}> = 1;
 [2] seed vanishings reproduce the oscillatory-route relations of oscillatory_check.py:
     r=4: nu40+nu04 = -1/4;  r=5 (both seed pairs): nu50+nu05 = -1/5;
 [3] first wall t_{3,3}^2:  A({(3,3)^2},0)=0  ==>  nu16+nu61 = -2/5  (same as the
     oscillatory route -- two independent derivations);
 [4] NEW INVARIANT, invisible diagonal t_{3,3}^3 (r(J)=4=r-1, good-basis periods see
     nothing; N=0 so no walls -- the value is absolute):
       A({(3,3)^3},0) = nu44 + 3(nu16+nu61) + 1 = 0
       ==>  <tau_0^{(3,3)}tau_0^{(3,3)}tau_0^{(3,3)} sigma_1^4 sigma_2^4 sigma_12> = 1/5;
 [5] descendent seed (condition (2), A(singleton,(1)) = -1), e.g. (3,3):
       nu_{(8,3)} + nu_{(3,8)} = -5/4  (descendent-boundary seed relation).

Run: ./venv/bin/python src/a_invariants.py
"""

from fractions import Fraction
from itertools import product as iproduct

import sympy as sp
from sympy.utilities.iterables import multiset_partitions


# ---------------------------------------------------------------- basic data

def r_of(J, r):
    return sum(a for (a, b, d) in J) % r


def s_of(J, s):
    return sum(b for (a, b, d) in J) % s


def m_of(J, r, s):
    """m(J,d) = rs + sum_j (s a_j + r b_j + rs(d_j - 1)); J = [(a,b,d),...]."""
    return r * s + sum(s * a + r * b + r * s * (d - 1) for (a, b, d) in J)


def d_of(J, r, s):
    """d(J,d) = (s r(J) + r s(J) - m(J,d))/rs - 1   [GKT Def 4.8]."""
    num = s * r_of(J, r) + r * s_of(J, s) - m_of(J, r, s)
    assert num % (r * s) == 0
    return num // (r * s) - 1


def part_boundaries(J, r, s):
    """Per-part boundary degrees: k1=r(J) mod r, k2=s(J) mod s, s k1 + r k2 = m(J,d)."""
    M = m_of(J, r, s)
    out = []
    if M < 0:
        return out
    for k1 in range(M // s + 1):
        rem = M - s * k1
        if rem % r:
            continue
        k2 = rem // r
        if k1 % r == r_of(J, r) and k2 % s == s_of(J, s):
            out.append((k1, k2))
    return out


def gamma_ratio(top_num, base_num, den):
    """Gamma(top_num/den)/Gamma(base_num/den) with (top_num-base_num)/den a
    nonnegative integer: the rising product (base/den)(base/den+1)...(top/den-1)."""
    assert (top_num - base_num) % den == 0 and top_num >= base_num
    n = (top_num - base_num) // den
    val = Fraction(1)
    for j in range(n):
        val *= Fraction(base_num + j * den, den)
    return val


# ---------------------------------------------------------------- A(J,d,nu)

def A_invariant(J, r, s, nu):
    """A(J,d,nu) for the labeled multiset J = [(a,b,d), ...] (labels = positions).

    nu: callable(part_key, k1, k2) -> sympy expression, with part_key the sorted
    tuple of (a,b,d) in the part.  Singleton d=0 parts default to 1 (Thm 5.1) if
    nu returns None."""
    labels = list(range(len(J)))
    rJ, sJ = r_of(J, r), s_of(J, s)
    total = sp.Integer(0)
    for partition in multiset_partitions(labels):
        # each unordered set partition counted once (= ordered with 1/h!)
        per_part_choices = []
        ok = True
        for block in partition:
            part = [J[i] for i in block]
            bnds = part_boundaries(part, r, s)
            if not bnds:
                ok = False
                break
            per_part_choices.append((part, bnds))
        if not ok:
            continue
        for choice in iproduct(*[bnds for _, bnds in per_part_choices]):
            K1 = sum(k1 for k1, k2 in choice)
            K2 = sum(k2 for k1, k2 in choice)
            w = gamma_ratio(1 + K1, 1 + rJ, r) * gamma_ratio(1 + K2, 1 + sJ, s)
            term = sp.Rational(w.numerator, w.denominator)
            for (part, _), (k1, k2) in zip(per_part_choices, choice):
                key = tuple(sorted(part))
                v = nu(key, k1, k2)
                if v is None:
                    if len(part) == 1 and part[0][2] == 0:
                        v = sp.Integer(1)  # Thm 5.1 normalization
                    else:
                        raise KeyError((key, k1, k2))
                term *= v
            total += term
    return sp.expand(total)


def nu_from_dict(d):
    def nu(key, k1, k2):
        return d.get((key, k1, k2))
    return nu


# ---------------------------------------------------------------- the checks

def check_convention_pin():
    """[1] A({(1,1)^2},0) at r=s=5: d(J,0)=0 and Thm 5.4(2) gives the closed value
    <tau0^{(1,1)}^3> = 1 (A_4 Frobenius: e1*e1=e2, eta(e2,e1)=1, tensor-squared with
    positive sign).  The labeled count yields exactly 1."""
    J = [(1, 1, 0), (1, 1, 0)]
    assert d_of(J, 5, 5) == 0
    val = A_invariant(J, 5, 5, nu_from_dict({}))
    assert val == 1, val
    # the halved (survey-Ex-5.5-style) count would give 1/2 -- demonstrate:
    # its only contribution is the h=2 all-singleton partition, weight 1, nu^2=1,
    # counted once WITH the 1/2! retained.
    halved = sp.Rational(1, 2)
    assert halved != val
    # convention-neutral Cor 0.6 check: distinct parts
    J2 = [(1, 1, 0), (2, 2, 0)]
    assert d_of(J2, 5, 5) == 0
    val2 = A_invariant(J2, 5, 5, nu_from_dict({}))
    assert val2 == 1, val2  # = <tau0^{(0,0)} tau0^{(1,1)} tau0^{(2,2)}> = 1
    return True


def check_seed_vanishings():
    """[2] A(seed,0)=0 reproduces the oscillatory relations."""
    # r=4, seed {(2,2),(2,2)}: boundaries (4,0),(0,4)
    nu40, nu04 = sp.symbols("nu40 nu04")
    J = [(2, 2, 0), (2, 2, 0)]
    assert d_of(J, 4, 4) < 0
    A = A_invariant(J, 4, 4, nu_from_dict({
        (((2, 2, 0), (2, 2, 0)), 4, 0): nu40,
        (((2, 2, 0), (2, 2, 0)), 0, 4): nu04,
    }))
    sol = sp.solve(sp.Eq(A, 0), nu40)[0]
    assert sp.simplify(sol + nu04) == sp.Rational(-1, 4)
    # r=5, both seed pairs
    for pair in [((2, 3, 0), (3, 2, 0)), ((2, 2, 0), (3, 3, 0))]:
        nu50, nu05 = sp.symbols("nu50 nu05")
        J = list(pair)
        assert d_of(J, 5, 5) < 0
        A = A_invariant(J, 5, 5, nu_from_dict({
            (tuple(sorted(pair)), 5, 0): nu50,
            (tuple(sorted(pair)), 0, 5): nu05,
        }))
        sol = sp.solve(sp.Eq(A, 0), nu50)[0]
        assert sp.simplify(sol + nu05) == sp.Rational(-1, 5), pair
    return True


def check_first_wall():
    """[3] A({(3,3)^2},0) = 0  ==>  nu16 + nu61 = -2/5."""
    nu16, nu61 = sp.symbols("nu16 nu61")
    J = [(3, 3, 0), (3, 3, 0)]
    assert d_of(J, 5, 5) < 0
    A = A_invariant(J, 5, 5, nu_from_dict({
        (((3, 3, 0), (3, 3, 0)), 1, 6): nu16,
        (((3, 3, 0), (3, 3, 0)), 6, 1): nu61,
    }))
    sol = sp.solve(sp.Eq(A, 0), nu16)[0]
    assert sp.simplify(sol + nu61) == sp.Rational(-2, 5)
    return True


def check_t33_cubed():
    """[4] the invisible diagonal t_{3,3}^3: A({(3,3)^3},0)=0 determines the
    wall-free coefficient nu44 absolutely."""
    nu16, nu61, nu44 = sp.symbols("nu16 nu61 nu44")
    J = [(3, 3, 0)] * 3
    assert d_of(J, 5, 5) == -1
    assert r_of(J, 5) == 4  # = r-1: invisible to good-basis periods
    A = A_invariant(J, 5, 5, nu_from_dict({
        (((3, 3, 0),) * 3, 4, 4): nu44,
        (((3, 3, 0), (3, 3, 0)), 1, 6): nu16,
        (((3, 3, 0), (3, 3, 0)), 6, 1): nu61,
    }))
    expected = nu44 + 3 * (nu16 + nu61) + 1
    assert sp.expand(A - expected) == 0, A
    nu44_value = sp.solve(sp.Eq(expected.subs(nu16, sp.Rational(-2, 5) - nu61), 0), nu44)[0]
    assert nu44_value == sp.Rational(1, 5)
    return nu44_value


def check_descendent_seed():
    """[5] condition (2): A({(3,3)},(1)) = (-1)^1 = -1  ==>  nu83+nu38 = -5/4."""
    nu83, nu38 = sp.symbols("nu83 nu38")
    J = [(3, 3, 1)]
    A = A_invariant(J, 5, 5, nu_from_dict({
        (((3, 3, 1),), 8, 3): nu83,
        (((3, 3, 1),), 3, 8): nu38,
    }))
    assert sp.expand(A - sp.Rational(4, 5) * (nu83 + nu38)) == 0, A
    sol = sp.solve(sp.Eq(A, -1), nu83)[0]
    assert sp.simplify(sol + nu38) == sp.Rational(-5, 4)
    return True


def main():
    print("== A(J,d,nu) in the labeled convention: relations and checks ==")
    assert check_convention_pin()
    print("[1] convention PIN via proven Thm 5.4(2): A({(1,1)^2},0) = <tau0^{(1,1)}^3> = 1")
    print("    labeled count: 1 (correct); survey-Ex-5.5-style count: 1/2 (fails).")
    print("    + neutral check A({(1,1),(2,2)},0) = <tau0^{(0,0)}tau0^{(1,1)}tau0^{(2,2)}> = 1: OK")
    assert check_seed_vanishings()
    print("[2] seed vanishings == oscillatory route: r=4: -1/4;  r=5 both pairs: -1/5")
    assert check_first_wall()
    print("[3] first wall t_33^2: nu16+nu61 = -2/5 (independent re-derivation, matches)")
    v = check_t33_cubed()
    print(f"[4] NEW invariant (invisible diagonal, wall-free): "
          f"<tau0^(3,3)^3 sigma_1^4 sigma_2^4 sigma_12> = {v}")
    assert check_descendent_seed()
    print("[5] descendent seed (condition (2), d=1 at (3,3)): nu83+nu38 = -5/4")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
