"""Exact checks for the two-label truncations of Section 7.

Two labels of twists (1,0) and (0,1), one descendent each, over
Q[t1,t2]/(t1^2,t2^2).  The script verifies:

1. the derivatives of Section 7 and the bracket [X_(1,0), X_(0,1)] = -3 X_(1,1);
2. the commutation relation S2 S1 = U_(3/4) S1 S2 on polynomials;
3. the crossing points, orders and signs of the two local paths;
4. the transported potential W^tr = V(8,4), the GKT transition and U_1;
5. the residual-factor comparison of the remark (mixed profile (4,8));
6. the maximal normal form N^max and its dependence on the incoming coefficient m;
7. the relative primitive beta = t1 t2 d(x^2 y^2) of the discrepancy;
8. the t1 t2 period class (6 - (a+b)/2)[xy] of V(a,b) in the Brieskorn module;
9. the truncation of two labels of twist (1,0): the transported potential, the
   flow exp(tt' X_(2,0)/2) on a ray without walls, the normal form, the
   nabla_0-relations for x^2 y^4, x^2 y^8, x^6, x^10, the tt'-parts of both
   extremal families, and the vanishing of their tt' period classes.

Run with Python 3 and SymPy:
    python3 fjrw-annular-scattering/src/check_two_label_endpoint.py
"""

from fractions import Fraction as Fr
import itertools

import sympy as sp

x, y, t1, t2, m, a, b, c = sp.symbols("x y t1 t2 m a b c")
checks = 0


def ok(cond, label):
    global checks
    if not cond:
        raise AssertionError(label)
    checks += 1


def trunc(expr):
    """Reduce modulo (t1^2, t2^2)."""
    p = sp.Poly(sp.expand(expr), t1, t2)
    out = 0
    for (i, j), coeff in p.terms():
        if i < 2 and j < 2:
            out += coeff * t1**i * t2**j
    return sp.expand(out)


def X(k1, k2, f):
    """GKT vector field X_k = x^k1 y^k2 ((k2+1) x d/dx - (k1+1) y d/dy)."""
    return sp.expand(x**k1 * y**k2 * ((k2 + 1) * x * sp.diff(f, x)
                                      - (k1 + 1) * y * sp.diff(f, y)))


def S1(f):
    return trunc(f - sp.Rational(1, 2) * t1 * X(1, 0, f))


def S2(f):
    return trunc(f - sp.Rational(1, 2) * t2 * X(0, 1, f))


def U(q, f):
    return trunc(f + q * t1 * t2 * X(1, 1, f))


def S1inv(f):
    return trunc(f + sp.Rational(1, 2) * t1 * X(1, 0, f))


def S2inv(f):
    return trunc(f + sp.Rational(1, 2) * t2 * X(0, 1, f))


W0 = x**4 + y**4


def V(p, q):
    return sp.expand(W0 - 2 * t1 * x**5 - 4 * t2 * x**4 * y
                     + t1 * t2 * (p * x * y**5 + q * x**5 * y))


Wmin = sp.expand(W0 - 4 * t1 * x * y**4 - 2 * t2 * y**5 + 12 * t1 * t2 * x * y**5)
Wmax = sp.expand(W0 - 2 * t1 * x**5 - 4 * t2 * x**4 * y + 12 * t1 * t2 * x**5 * y)

# 1. Derivatives and bracket.
ok(X(1, 0, W0) == sp.expand(4 * x**5 - 8 * x * y**4), "X10 W0")
ok(X(1, 0, y**5) == -10 * x * y**5, "X10 y^5")
ok(X(0, 1, W0) == sp.expand(8 * x**4 * y - 4 * y**5), "X01 W0")
ok(X(0, 1, x**5) == 10 * x**5 * y, "X01 x^5")
ok(X(1, 1, W0) == sp.expand(8 * x**5 * y - 8 * x * y**5), "X11 W0")
for i, j in itertools.product(range(7), repeat=2):
    f = x**i * y**j
    comm = sp.expand(X(1, 0, X(0, 1, f)) - X(0, 1, X(1, 0, f)))
    ok(comm == sp.expand(-3 * X(1, 1, f)), f"bracket on x^{i} y^{j}")

# 2. S2 S1 = U_{3/4} S1 S2 on monomials, with generic coefficients.
for i, j in itertools.product(range(7), repeat=2):
    f = x**i * y**j * (1 + t1 + t2 + t1 * t2)
    ok(S2(S1(f)) == U(sp.Rational(3, 4), S1(S2(f))), f"commutation on x^{i} y^{j}")

# 3. Geometry of the local paths (scale eps = 1).
supports = {
    "S1": ((Fr(3), Fr(4)), "line"),
    "S2": ((Fr(3), Fr(2)), "line"),
    "T": ((Fr(1), Fr(1)), "ray"),
}


def crossings(level):
    hits = []
    for name, (normal, kind) in supports.items():
        n1, n2 = normal
        s1 = -n2 * level / n1          # n1 s1 + n2 s2 = 0 on s2 = level
        if kind == "ray" and level < 0:
            continue                   # the mixed ray has s2 >= 0
        if abs(s1) <= 1:
            sign = 1 if n1 > 0 else -1  # direction of travel (1,0)
            hits.append((s1, name, sign))
    return sorted(hits)


up = crossings(Fr(1, 10))
down = crossings(Fr(-1, 10))
ok(up == [(Fr(-2, 15), "S1", 1), (Fr(-1, 10), "T", 1), (Fr(-1, 15), "S2", 1)], "upper path")
ok(down == [(Fr(1, 15), "S2", 1), (Fr(2, 15), "S1", 1)], "lower path")
for s1 in (Fr(-1), Fr(1)):             # vertical segments, |s2| <= 1/10
    for name, ((n1, n2), kind) in supports.items():
        s2 = -n1 * s1 / n2
        if kind == "ray" and s2 < 0:
            continue
        ok(abs(s2) > Fr(1, 10), f"vertical segment at {s1} misses {name}")

# 4. Transport, GKT transition and U_1.
Wtr = V(8, 4)
up_word = S2(U(-sp.Rational(3, 4), S1(Wmin)))
down_word = S1(S2(Wmin))
ok(up_word == Wtr, "upper wall word")
ok(down_word == Wtr, "lower wall word")
ok(sp.expand(S1(Wmin) - (W0 - 2 * t1 * x**5 - 2 * t2 * y**5 + 2 * t1 * t2 * x * y**5)) == 0,
   "S1 W^min")
gkt = S2(U(sp.Rational(1, 4), S1(Wmin)))
ok(gkt == Wmax, "GKT transition")
ok(U(1, Wtr) == Wmax, "U_1 W^tr = W^max")
for i, j in itertools.product(range(7), repeat=2):
    f = x**i * y**j * (1 + t1 + t2)
    ok(S2(U(sp.Rational(1, 4), S1(f))) == U(1, S2(U(-sp.Rational(3, 4), S1(f)))),
       f"S2 U_1/4 S1 = U_1 Theta_+ on x^{i} y^{j}")

# 5. Replacing T by G_mix.
ok(S2(U(-sp.Rational(1, 4), S1(Wmin))) == V(4, 8), "residual comparison")
ok(S2(U(-sp.Rational(1, 4), S1(Wmin))) == U(sp.Rational(1, 2), down_word), "R_mix holonomy")

# 6. Normal form.
ok(sp.expand(U(c, V(a, b)) - V(a - 8 * c, b + 8 * c)) == 0, "U_c V(a,b)")
ok(sp.expand(U(a / 8, V(a, b)) - V(0, a + b)) == 0, "N^max V(a,b)")
ok(sp.expand(U(0, V(0, a + b)) - V(0, a + b)) == 0, "N^max idempotent")
Wmin_m = sp.expand(W0 - 4 * t1 * x * y**4 - 2 * t2 * y**5 + m * t1 * t2 * x * y**5)
tr_m = S2(U(-sp.Rational(3, 4), S1(Wmin_m)))
ok(sp.expand(tr_m - V(m - 4, 4)) == 0, "transport with unknown m")
ok(sp.expand(U((m - 4) / 8, tr_m) - V(0, m)) == 0, "normal form with unknown m")

# 7. Relative primitive.
beta_x = t1 * t2 * sp.diff(x**2 * y**2, x)   # coefficient of dx
beta_y = t1 * t2 * sp.diff(x**2 * y**2, y)   # coefficient of dy
ok(sp.simplify(sp.diff(beta_y, x) - sp.diff(beta_x, y)) == 0, "d beta = 0")
ok(beta_x.subs(y, 0) == 0 and beta_y.subs(x, 0) == 0, "axis pullbacks vanish")
dW = (sp.diff(W0, x), sp.diff(W0, y))
D0beta = sp.expand(dW[0] * beta_y - dW[1] * beta_x)   # coefficient of dx^dy
ok(sp.expand(D0beta - (Wmax - Wtr)) == 0, "D0 beta = (W^max - W^tr) Omega")


# 8. Period classes in the Brieskorn module of W0 = x^4 + y^4.
#    With f dy: [4 x^3 f + hbar f_x] = 0; with g dx: [4 y^3 g + hbar g_y] = 0.
hb = sp.symbols("hbar")


def reduce_class(mono):
    """Express [x^i y^j] as c * hbar^k * [x^a y^b] with a, b <= 2."""
    i, j = mono
    coeff = sp.Integer(1)
    while i >= 3:
        # x^i y^j = x^3 * x^(i-3) y^j ;  f = x^(i-3) y^j
        coeff *= -hb * (i - 3) / 4
        i -= 4
        if i < 0:
            return 0, None
    while j >= 3:
        coeff *= -hb * (j - 3) / 4
        j -= 4
        if j < 0:
            return 0, None
    return sp.simplify(coeff), (i, j)


ok(reduce_class((5, 1)) == (-hb / 2, (1, 1)), "[x^5 y] = -(hbar/2)[xy]")
ok(reduce_class((1, 5)) == (-hb / 2, (1, 1)), "[x y^5] = -(hbar/2)[xy]")
ok(reduce_class((9, 1)) == (sp.Rational(3, 4) * hb**2, (1, 1)), "[x^9 y] = (3/4) hbar^2 [xy]")


def mixed_period_class(W, u, v):
    """Coefficient of [basis] in the u*v part of exp((W - W0)/hbar) Omega."""
    d = sp.expand(W - W0)
    part = sp.expand(d / hb + d**2 / (2 * hb**2))
    part = sp.Poly(part, u, v).as_dict().get((1, 1), 0)
    total = {}
    for (i, j), cf in sp.Poly(sp.expand(part), x, y).as_dict().items():
        r, basis = reduce_class((i, j))
        if basis is not None:
            total[basis] = sp.simplify(total.get(basis, 0) + cf * r)
    return {k: v_ for k, v_ in total.items() if v_ != 0}


ok(mixed_period_class(V(a, b), t1, t2) == {(1, 1): 6 - (a + b) / 2}, "period class of V(a,b)")
ok(mixed_period_class(Wmin, t1, t2) == {} and mixed_period_class(Wmax, t1, t2) == {},
   "extremal potentials have vanishing mixed period class")

# 9. Two labels t, t' of twist (1,0), descendent one.
t, tp, p_, q_, n = sp.symbols("t tp p q n")


def trunc2(expr):
    poly = sp.Poly(sp.expand(expr), t, tp)
    return sp.expand(sum(cf * t**i * tp**j for (i, j), cf in poly.terms() if i < 2 and j < 2))


def flow(D, f, order=3):
    out, term = f, f
    for k in range(1, order):
        term = trunc2(D(term) / k)
        out = trunc2(out + term)
    return out


D10 = lambda f: sp.expand(-sp.Rational(1, 2) * (t + tp) * X(1, 0, f))
Wmin2 = sp.expand(W0 - 4 * (t + tp) * x * y**4 + 20 * t * tp * x**2 * y**4)
Wmax2 = sp.expand(W0 - 2 * (t + tp) * x**5 + 7 * t * tp * x**6)


def V2(pp, qq):
    return sp.expand(W0 - 2 * (t + tp) * x**5 + t * tp * (pp * x**2 * y**4 + qq * x**6))


theta2 = flow(D10, Wmin2)
ok(theta2 == V2(6, 5), "same-twist wall word gives V'(6,5)")
ok(X(2, 0, W0) == sp.expand(4 * x**6 - 12 * x**2 * y**4), "X20 W0")
Z20 = lambda cc, f: trunc2(f + cc * t * tp * X(2, 0, f))
ok(Z20(sp.Rational(1, 2), theta2) == Wmax2, "GKT factor exp(tt' X20/2)")
ok(sp.expand(Z20(p_ / 12, V2(p_, q_)) - V2(0, q_ + p_ / 3)) == 0, "same-twist normal form")
Wmin2_n = sp.expand(W0 - 4 * (t + tp) * x * y**4 + n * t * tp * x**2 * y**4)
ok(sp.expand(flow(D10, Wmin2_n) - V2(n - 14, 5)) == 0, "same-twist transport with unknown n")
ok(mixed_period_class(Wmin2, t, tp) == {} and mixed_period_class(Wmax2, t, tp) == {},
   "same-twist extremal potentials have vanishing tt' period class")
ok(mixed_period_class(V2(p_, q_), t, tp) == {(2, 0): sp.Rational(21, 4) - p_ / 4 - 3 * q_ / 4},
   "same-twist period class")
Wmin2_p = sp.expand(W0 - 4 * (t + tp) * x * y**4 + p_ * t * tp * x**2 * y**4)
ok(mixed_period_class(Wmin2_p, t, tp) == {(2, 0): 5 - p_ / 4},
   "same-twist period class, least-exponent singleton terms")
# the four nabla_0-relations quoted in Section 7.3 and the tt'-parts they reduce
ok(reduce_class((2, 4)) == (-hb / 4, (2, 0)), "[x^2 y^4] = -(hbar/4)[x^2]")
ok(reduce_class((2, 8)) == (sp.Rational(5, 16) * hb**2, (2, 0)), "[x^2 y^8] = (5/16) hbar^2 [x^2]")
ok(reduce_class((6, 0)) == (-sp.Rational(3, 4) * hb, (2, 0)), "[x^6] = -(3 hbar/4)[x^2]")
ok(reduce_class((10, 0)) == (sp.Rational(21, 16) * hb**2, (2, 0)), "[x^10] = (21/16) hbar^2 [x^2]")


def ttp_part(W):
    d = sp.expand(W - W0)
    part = sp.expand(d / hb + d**2 / (2 * hb**2))
    return sp.expand(sp.Poly(part, t, tp).as_dict().get((1, 1), 0))


Wmax2_q = sp.expand(W0 - 2 * (t + tp) * x**5 + q_ * t * tp * x**6)
ok(sp.expand(ttp_part(Wmin2_p) - (p_ * x**2 * y**4 / hb + 16 * x**2 * y**8 / hb**2)) == 0,
   "tt'-part of the least-exponent family")
ok(sp.expand(ttp_part(Wmax2_q) - (q_ * x**6 / hb + 4 * x**10 / hb**2)) == 0,
   "tt'-part of the greatest-exponent family")
ok(mixed_period_class(Wmax2_q, t, tp) == {(2, 0): sp.Rational(21, 4) - 3 * q_ / 4},
   "same-twist period class, greatest-exponent singleton terms")
# the ray of shifted exponent (3,1) is outside the cone of (2,1),(1,2) and is met by gamma_+ first
lam, mu = sp.symbols("lam mu")
sol = sp.solve([2 * lam + mu - 3, lam + 2 * mu - 1], [lam, mu])
ok(sol[mu] < 0, "(3,1) outside the scattering cone")
L = sp.Matrix([[-2, 0], [1, 1]])
dirn = L * sp.Matrix([3, 1])              # cofactor direction of the shifted exponent (3,1)
s1_hit = dirn[0] / dirn[1] * sp.Rational(1, 10)
ok(s1_hit == -sp.Rational(3, 10) / 2 and s1_hit < -sp.Rational(2, 15),
   "gamma_+ meets the (3,1)-ray before the (2,1)-chord")

print(f"PASS: {checks} exact checks for the two-label truncations.")
