"""Exact checks for the degree-weighted residual associator (Appendix D).

Appendix D interpolates residual transports with

    g #_c h = g * (g^-1*h)^c.

At a vertex of a product tree T with subtrees T_1, T_2 the interpolant is
H_T = H_{T_1} #_c H_{T_2} with c = |T_2|/|T|, the degree of the right subtree
over the degree of T (Definition D.3); for two inputs of equal degree this is
the midpoint H = Q_1 #_{1/2} Q_2 of Lemma D.2.  Iterated unweighted midpoints
give three inputs the leaf weights (1/4,1/4,1/2) and (1/2,1/4,1/4) in the two
parenthesizations already for commuting inputs; the degree-weighted
recursion gives each leaf of equal degree the weight 1/3.  The remaining
defect is the tree transition V_{T<-T'} = H_T*H_T'^-1.

The computation takes place in the quotient U of the free associative algebra
by the words containing some letter twice (Proposition D.4).  It checks:
  - the symmetry of the midpoint (Lemma D.2(1));
  - the leaf weights of the unweighted and of the weighted recursion, and the
    absence of quadratic terms in log H_T for the two three-leaf trees;
  - log V_{T_l<-T_r} = (1/72)[[x,z],y] for T_l = ((1,2),3), T_r = (1,(2,3))
    (Proposition D.4), and that V_{T_l<-T_r} carries the right-parenthesized
    corrected product to the left-parenthesized one;
  - the equivariance (Psi g Psi'^-1) #_c (Psi h Psi'^-1) = Psi (g #_c h) Psi'^-1
    under a common continuation (Lemma D.2(3)), under which the associator is
    conjugated by Psi;
  - the pentagon of tree transitions for four inputs (Proposition D.4).
"""


# Work in the completed tensor algebra on square-zero deformation labels.
# A word with a repeated label vanishes.  Six labels allow four residual
# leaves and independent left/right terminal transports.
labels = ("x", "y", "z", "u", "t", "w")
max_degree = len(labels)
one = {(): QQ.one()}


def clean(value):
    return {word: QQ(coefficient) for word, coefficient in value.items() if coefficient}


def add(left, right):
    output = dict(left)
    for word, coefficient in right.items():
        output[word] = output.get(word, QQ.zero()) + coefficient
    return clean(output)


def scale(value, scalar):
    return clean({word: QQ(scalar) * coefficient for word, coefficient in value.items()})


def multiply(left, right):
    output = {}
    for left_word, left_coefficient in left.items():
        left_support = set(left_word)
        for right_word, right_coefficient in right.items():
            if left_support.intersection(right_word):
                continue
            word = left_word + right_word
            output[word] = (
                output.get(word, QQ.zero())
                + left_coefficient * right_coefficient
            )
    return clean(output)


def bracket(left, right):
    return add(multiply(left, right), scale(multiply(right, left), -1))


def power(value, exponent):
    output = one
    for _ in range(exponent):
        output = multiply(output, value)
    return output


def exponential(value):
    output = one
    factorial = QQ.one()
    for degree in range(1, max_degree + 1):
        factorial *= degree
        output = add(output, scale(power(value, degree), 1 / factorial))
    return output


def logarithm(group_value):
    nilpotent = add(group_value, scale(one, -1))
    output = {}
    for degree in range(1, max_degree + 1):
        output = add(
            output,
            scale(power(nilpotent, degree), QQ((-1) ** (degree + 1)) / degree),
        )
    return output


def inverse(group_value):
    return exponential(scale(logarithm(group_value), -1))


def group_power(group_value, exponent):
    return exponential(scale(logarithm(group_value), exponent))


def weighted_midpoint(left, right, left_weight, right_weight):
    relative = multiply(inverse(left), right)
    fraction = QQ(right_weight) / (left_weight + right_weight)
    return multiply(left, group_power(relative, fraction))


def transition(target_center, source_center):
    """Output automorphism carrying the source tree to the target tree."""

    return multiply(target_center, inverse(source_center))


def generator(label):
    return {(label,): QQ.one()}


Qx = exponential(generator("x"))
Qy = exponential(generator("y"))
Qz = exponential(generator("z"))
Qu = exponential(generator("u"))
T = exponential(generator("t"))
W = exponential(generator("w"))


# The ordinary midpoint is symmetric and is the equal-weight specialization.
assert weighted_midpoint(Qx, Qy, 1, 1) == weighted_midpoint(Qy, Qx, 1, 1)


# Unweighted recursion is already wrong in degree one: the left tree gives
# weights (1/4,1/4,1/2), while the right gives (1/2,1/4,1/4).
unweighted_left = weighted_midpoint(
    weighted_midpoint(Qx, Qy, 1, 1), Qz, 1, 1
)
unweighted_right = weighted_midpoint(
    Qx, weighted_midpoint(Qy, Qz, 1, 1), 1, 1
)
unweighted_left_log = logarithm(unweighted_left)
unweighted_right_log = logarithm(unweighted_right)
assert tuple(unweighted_left_log[(label,)] for label in ("x", "y", "z")) == (
    QQ(1) / 4,
    QQ(1) / 4,
    QQ(1) / 2,
)
assert tuple(unweighted_right_log[(label,)] for label in ("x", "y", "z")) == (
    QQ(1) / 2,
    QQ(1) / 4,
    QQ(1) / 4,
)


# Degree-weighted recursion gives each equal-degree leaf weight 1/3 in the
# abelian quotient.  Its remaining defect is purely noncommutative.
H_left = weighted_midpoint(weighted_midpoint(Qx, Qy, 1, 1), Qz, 2, 1)
H_right = weighted_midpoint(Qx, weighted_midpoint(Qy, Qz, 1, 1), 1, 2)
for center in (H_left, H_right):
    center_log = logarithm(center)
    assert tuple(center_log[(label,)] for label in ("x", "y", "z")) == (
        QQ(1) / 3,
        QQ(1) / 3,
        QQ(1) / 3,
    )
    assert all(
        coefficient == 0
        for word, coefficient in center_log.items()
        if len(word) == 2
    )

associator = transition(H_left, H_right)
associator_log = logarithm(associator)
assert associator != one
assert all(len(word) == 3 for word in associator_log)
expected_associator_log = scale(
    bracket(bracket(generator("x"), generator("z")), generator("y")),
    QQ(1) / 72,
)
assert associator_log == expected_associator_log


# The associator exactly carries the right-parenthesized corrected product
# to the left-parenthesized corrected product for every decorated output P.
P = exponential(add(generator("x"), add(generator("y"), generator("z"))))
assert multiply(associator, multiply(H_right, P)) == multiply(H_left, P)


# Weighted midpoints satisfy M(T P W^-1, T Q W^-1) = T M(P,Q) W^-1, where T
# is the transport of a common continuation through D and W its GKT
# transport (Psi and Psi' in Lemma D.2).  The associator
# consequently transforms by conjugation with T.
def bitorsor_transform(value):
    return multiply(T, multiply(value, inverse(W)))


transformed_left = weighted_midpoint(
    weighted_midpoint(bitorsor_transform(Qx), bitorsor_transform(Qy), 1, 1),
    bitorsor_transform(Qz),
    2,
    1,
)
transformed_right = weighted_midpoint(
    bitorsor_transform(Qx),
    weighted_midpoint(bitorsor_transform(Qy), bitorsor_transform(Qz), 1, 1),
    1,
    2,
)
assert transformed_left == bitorsor_transform(H_left)
assert transformed_right == bitorsor_transform(H_right)
transformed_associator = transition(transformed_left, transformed_right)
assert transformed_associator == multiply(
    T, multiply(associator, inverse(T))
)


# Five centers for the four equal-weight binary trees around the Stasheff
# pentagon.  The transition maps form a thin groupoid, so the boundary
# composite is exactly the identity rather than merely infinitesimally so.
H_1 = weighted_midpoint(
    weighted_midpoint(weighted_midpoint(Qx, Qy, 1, 1), Qz, 2, 1),
    Qu,
    3,
    1,
)  # (((xy)z)u)
H_2 = weighted_midpoint(
    weighted_midpoint(Qx, Qy, 1, 1),
    weighted_midpoint(Qz, Qu, 1, 1),
    2,
    2,
)  # ((xy)(zu))
H_3 = weighted_midpoint(
    Qx,
    weighted_midpoint(Qy, weighted_midpoint(Qz, Qu, 1, 1), 1, 2),
    1,
    3,
)  # (x(y(zu)))
H_4 = weighted_midpoint(
    Qx,
    weighted_midpoint(weighted_midpoint(Qy, Qz, 1, 1), Qu, 2, 1),
    1,
    3,
)  # (x((yz)u))
H_5 = weighted_midpoint(
    weighted_midpoint(Qx, weighted_midpoint(Qy, Qz, 1, 1), 1, 2),
    Qu,
    3,
    1,
)  # ((x(yz))u)

pentagon_boundary = one
for target, source in (
    (H_2, H_1),
    (H_3, H_2),
    (H_4, H_3),
    (H_5, H_4),
    (H_1, H_5),
):
    pentagon_boundary = multiply(transition(target, source), pentagon_boundary)
assert pentagon_boundary == one


print("unweighted leaf weights:", (QQ(1) / 4, QQ(1) / 4, QQ(1) / 2), (QQ(1) / 2, QQ(1) / 4, QQ(1) / 4))
print("weighted leaf weights:", (QQ(1) / 3, QQ(1) / 3, QQ(1) / 3))
print("weighted associator logarithm:", associator_log)
print("equivariance under common continuation: PASS")
print("four-input pentagon: PASS")
print("DEGREE-WEIGHTED RESIDUAL ASSOCIATOR: PASS")
