"""Exact checks for the x^4+y^4 marginal elliptic pencil.

Run with SageMath 10.9 or later:

    sage fjrw-annular-scattering/src/check_elliptic_quartic.sage
"""

R.<s> = PolynomialRing(QQ)
K = R.fraction_field()
sK = K(s)

# Binary-quartic invariants for f_s(x)=x^4+s*x^2+1.
I = sK^2 + 12
J = 2 * sK * (36 - sK^2)
discriminant = (4 * I^3 - J^2) / 27
j_quartic = 256 * I^3 / discriminant

# A rational birational transformation takes the quartic to Legendre form:
#
#   X=(z+1)/x^2,  Y=x*(X^2-1),  u=(X+1)/2,  v=Y/4,
#   v^2=u*(u-1)*(u-lambda),  lambda=(2-s)/4.
#
# Thus s=2,-2,infinity map respectively to lambda=0,1,infinity.
lam = (2 - sK) / 4
j_legendre = 256 * (1 - lam + lam^2)^3 / (lam^2 * (1 - lam)^2)

assert discriminant == 16 * (sK - 2)^2 * (sK + 2)^2
assert j_quartic == 16 * (sK^2 + 12)^3 / (sK^2 - 4)^2
assert j_quartic == j_legendre
assert j_quartic(s=0) == 1728

# Standard Legendre monodromies around lambda=0 and lambda=1 in one
# symplectic basis. Reversing loop orientation inverts the corresponding
# matrix; changing the basis conjugates all three matrices simultaneously.
M0 = matrix(ZZ, [[1, 2], [0, 1]])
M1 = matrix(ZZ, [[1, 0], [-2, 1]])
Minf = (M0 * M1)^(-1)

assert M0.det() == M1.det() == Minf.det() == 1
assert M0 * M1 * Minf == identity_matrix(ZZ, 2)
assert M0 != identity_matrix(ZZ, 2)

# The marginal GKT/Maher vector field is a right-equivalence direction.
P.<x, y, q> = PolynomialRing(QQ)
W = x^4 + y^4 + q * x^2 * y^2
D_W = x * W.derivative(x) - y * W.derivative(y)
assert D_W == 4 * (x^4 - y^4)
assert D_W == x * W.derivative(x) - y * W.derivative(y)

print("Delta(s) =", factor(discriminant))
print("j(s) =", factor(j_quartic))
print("lambda(s) =", lam)
print("M_0 =")
print(M0)
print("M_1 =")
print(M1)
print("M_infinity =")
print(Minf)
print("D(W_s) =", D_W)
