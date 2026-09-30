#!/usr/bin/env sage
"""Exact development of the two boundary circles and separation of the exits.

Develops the three facets along each boundary circle of the annulus in one
invariant tangent-normal chart, checks the lengths, widths, heights and the
common apex of Appendix A, computes the slopes of the exiting supports in
both cofactor representatives, and checks the slope envelopes and the
separation inequalities used for the proposition on separation of the exits
in Section 4.

Usage: sage check_horizontal_cap_atlas.sage
"""

# Bottom endpoints/slopes of the four negative standard chords.
bottom = {
    "alpha": QQ(4)/3,
    "delta": QQ(8)/7,
    "gamma": QQ(6)/7,
    "beta": QQ(2)/3,
}
bottom_values = list(bottom.values())
bottom_spread = max(bottom_values) - min(bottom_values)
assert bottom_spread == QQ(2)/3

# Every nonzero scattering descendant has
# q(a,b)=a(2,1)+b(1,2), a,b >= 0.  With z=b/a, its bottom endpoint and
# tangential/outward slope have the following common coordinate.  The
# limiting values also cover a=0 or b=0, so all descendants lie in the same
# envelope as the four standard chords.
z = var("z")
x_bottom = 2*(2 + z)/(3*(1 + z))
assert x_bottom.subs(z=0) == QQ(4)/3
assert limit(x_bottom, z=infinity) == QQ(2)/3
assert diff(x_bottom, z) == -2/(3*(z + 1)^2)

# On either positive oblique facet, the endpoint coordinate and the
# tangential/outward slope are the same function of the direction ratio.
h = 3*(1 + z)/(4 + 2*z)
assert h.subs(z=0) == QQ(3)/4
assert limit(h, z=1, dir='minus') == 1
assert diff(h, z) == 3/(2*(z + 2)^2)
positive_spread = 1 - QQ(3)/4
assert positive_spread == QQ(1)/4

# Crosswise gluing places the positive family from one lift and the negative
# family from the other lift on the same boundary circle.  Develop all three
# facets in one invariant tangent-normal coordinate, including translations.
p_A = vector(QQ, (-1, -1))
p_B = vector(QQ, (-1, 1))
p_C = vector(QQ, (3, -1))
origin = vector(QQ, (0, 0))

T_A = matrix(ZZ, [[0, -1], [1, 2]])
T_B = matrix(ZZ, [[3, -2], [2, -1]])
T_C = matrix(ZZ, [[4, -1], [9, -2]])
support_transition_A = T_A.inverse().transpose()
support_transition_B = T_B.inverse().transpose()
support_transition_C = T_C.inverse().transpose()

cofactor_plus = matrix(QQ, [[-2, 0], [1, 1]])
cofactor_minus = matrix(QQ, [[-4, 2], [3, -1]])
assert cofactor_minus == matrix(ZZ, [[3, 2], [-2, -1]]) * cofactor_plus

first_base_frame = cofactor_plus
second_base_frame = T_A.transpose() * cofactor_plus
boundary_frames = []
for base_frame in (first_base_frame, second_base_frame):
    boundary_frames.append({
        "bottom": base_frame,
        "slope": support_transition_C * base_frame,
        "left": support_transition_B * support_transition_C * base_frame,
    })

# The first columns are the invariant primitive tangents, oriented along
# A -> B -> C -> A.  In these bases the boundary holonomies are precisely
# T^(-4),T^(-16), as in Section 4 and Appendix A.
collar_bases = (
    matrix(ZZ, [(1, -1), (0, 1)]).transpose(),
    matrix(ZZ, [(1, -2), (0, 1)]).transpose(),
)
boundary_widths = (4, 16)
boundary_heights = (QQ(1), QQ(1)/2)
boundary_lengths = (4, 8)
boundary_phases = (QQ(7)/2, QQ(15)/2)
facet_path = (
    ("left", p_A, p_B),
    ("slope", p_B, p_C),
    ("bottom", p_C, p_A),
)


def develop_boundary(boundary_index):
    """Return affine facet charts with consecutive boundary translations."""
    frames = boundary_frames[boundary_index]
    collar_basis = collar_bases[boundary_index]
    charts = {}
    current_x = QQ(0)
    for facet, start, end in facet_path:
        linear = collar_basis.inverse() * frames[facet].inverse()
        translation = vector(QQ, (current_x, 0)) - linear * start
        charts[facet] = (linear, translation)
        assert linear * start + translation == vector(QQ, (current_x, 0))
        developed_end = linear * end + translation
        assert developed_end[1] == 0
        current_x = developed_end[0]
    return charts, current_x


developed_boundaries = []
for boundary_index in (0, 1):
    charts, affine_length = develop_boundary(boundary_index)
    developed_boundaries.append(charts)
    assert affine_length == boundary_lengths[boundary_index]
    common_apex = vector(QQ, (
        boundary_phases[boundary_index],
        boundary_heights[boundary_index],
    ))
    for linear, translation in charts.values():
        assert linear * origin + translation == common_apex

    closing_frame = (
        support_transition_A
        * support_transition_B
        * support_transition_C
        * (first_base_frame if boundary_index == 0 else second_base_frame)
    )
    tangent_holonomy = (
        (first_base_frame if boundary_index == 0 else second_base_frame).inverse()
        * closing_frame
    )
    collar_holonomy = (
        collar_bases[boundary_index].inverse()
        * tangent_holonomy
        * collar_bases[boundary_index]
    )
    assert collar_holonomy == matrix(
        ZZ, [[1, -boundary_widths[boundary_index]], [0, 1]]
    )
    assert affine_length == (
        boundary_widths[boundary_index] * boundary_heights[boundary_index]
    )

# Every descendant exponent is q=a(2,1)+b(1,2).  The plus and minus
# sectorial cofactor representatives cover the possible direction change
# induced by crossing the B-focus cut.  For either representative, a>b exits
# through the left facet and b>a through the slope facet; the negative half
# exits through the bottom facet.
SlopeRing.<a,b> = PolynomialRing(QQ, 2)
q = vector(SlopeRing, (2*a + b, a + 2*b))
directions = {
    "plus": cofactor_plus * q,
    "minus": cofactor_minus * q,
}
assert directions == {
    "plus": vector(SlopeRing, (-4*a - 2*b, 3*a + 3*b)),
    "minus": vector(SlopeRing, (-6*a, 5*a + b)),
}


def outward_slope(boundary_index, facet, direction, positive):
    linear, _ = developed_boundaries[boundary_index][facet]
    outward_direction = direction if positive else -direction
    developed_direction = linear * outward_direction
    # The developed interior has second coordinate >0.  Section 4
    # attaches the flare with r=-y.  We record xi=-dx/dr, so every ray is
    # x=phase-xi*(r+h_i) and the useful envelopes stay positive.
    return (-developed_direction[0] / (-developed_direction[1])).factor()


computed_slopes = {}
for boundary_index in (0, 1):
    computed_slopes[boundary_index] = {}
    for representative, direction in directions.items():
        computed_slopes[boundary_index][representative] = {
            "negative": outward_slope(
                boundary_index, "bottom", direction, False
            ),
            "positive_left": outward_slope(
                boundary_index, "left", direction, True
            ),
            "positive_slope": outward_slope(
                boundary_index, "slope", direction, True
            ),
        }

expected_slopes = {
    0: {
        "plus": {
            "negative": (2*a + b)/(3*(a + b)),
            "positive_left": 3*(7*a + 3*b)/(4*(2*a + b)),
            "positive_slope": (11*a + 19*b)/(4*(a + 2*b)),
        },
        "minus": {
            "negative": 3*a/(5*a + b),
            "positive_left": (31*a - b)/(12*a),
            "positive_slope": (21*a + 9*b)/(4*(2*a + b)),
        },
    },
    1: {
        "plus": {
            "negative": (11*a + 7*b)/(3*(a + b)),
            "positive_left": (23*a + 10*b)/(2*a + b),
            "positive_slope": (12*a + 21*b)/(a + 2*b),
        },
        "minus": {
            "negative": (17*a + b)/(5*a + b),
            "positive_left": (34*a - b)/(3*a),
            "positive_slope": (23*a + 10*b)/(2*a + b),
        },
    },
}
for boundary_index in (0, 1):
    for representative in ("plus", "minus"):
        for family in ("negative", "positive_left", "positive_slope"):
            assert (
                computed_slopes[boundary_index][representative][family]
                - expected_slopes[boundary_index][representative][family]
            ) == 0

# Exact closed slope envelopes.  Positive intervals combine the left branch
# 0<=b/a<=1 and the slope branch 1<=b/a<=infinity.  The diagonal ray ends at
# the B focus and is obtained as the common limiting value; it creates no
# distinct outer support.
slope_envelopes = {
    0: {
        "plus": {
            "negative": (QQ(1)/3, QQ(2)/3),
            "positive": (QQ(19)/8, QQ(21)/8),
        },
        "minus": {
            "negative": (QQ(0), QQ(3)/5),
            "positive": (QQ(9)/4, QQ(31)/12),
        },
    },
    1: {
        "plus": {
            "negative": (QQ(7)/3, QQ(11)/3),
            "positive": (QQ(21)/2, QQ(23)/2),
        },
        "minus": {
            "negative": (QQ(1), QQ(17)/5),
            "positive": (QQ(10), QQ(34)/3),
        },
    },
}

# The crosswise branch gluing is the identity in physical affine
# coordinates.  Hence the positive and negative families on one boundary
# have the same developed apex and no relative phase.  A ray of oriented
# slope sigma is x=phase_i-sigma*(r+h_i), while one deck period is
# w_i*(r+h_i).  Cross-family collision is therefore equivalent to
# sigma_+-sigma_- lying in w_i*ZZ.  The
# following strict inequalities cover both cofactor representatives on both
# sides, hence all four possible representative pairings.
for boundary_index in (0, 1):
    width = QQ(boundary_widths[boundary_index])
    envelopes = slope_envelopes[boundary_index]
    for positive_representative in ("plus", "minus"):
        positive_range = envelopes[positive_representative]["positive"]
        assert positive_range[1] - positive_range[0] < width
        for negative_representative in ("plus", "minus"):
            negative_range = envelopes[negative_representative]["negative"]
            assert negative_range[1] - negative_range[0] < width
            smallest_difference = positive_range[0] - negative_range[1]
            largest_difference = positive_range[1] - negative_range[0]
            assert 0 < smallest_difference
            assert largest_difference < width

# Check the phase relation directly at all three possible endpoints for both
# cofactor representatives.  This checks that the two sheet families share
# one affine phase, so that all exiting supports are rays from one apex.
for boundary_index in (0, 1):
    charts = developed_boundaries[boundary_index]
    phase = boundary_phases[boundary_index]
    height = boundary_heights[boundary_index]
    for representative, direction in directions.items():
        endpoints = {
            "negative": -direction / direction[1],
            "positive_left": direction / (-direction[0]),
            "positive_slope": direction / (direction[0] + 2*direction[1]),
        }
        facets = {
            "negative": "bottom",
            "positive_left": "left",
            "positive_slope": "slope",
        }
        for family, endpoint in endpoints.items():
            linear, translation = charts[facets[family]]
            attaching_x = (linear * endpoint + translation)[0]
            sigma = computed_slopes[boundary_index][representative][family]
            assert attaching_x - (phase - sigma * height) == 0

# The two boundary slope systems are related by the exact affine rescaling
# mu_1=4*mu_0+1 for every physical direction.  Differences therefore scale
# by four, matching the width change 4 -> 16.
for representative in ("plus", "minus"):
    for family in ("negative", "positive_left", "positive_slope"):
        assert (
            computed_slopes[1][representative][family]
            - 4 * computed_slopes[0][representative][family] - 1
        ) == 0

print("boundary development and exit separation: PASS")
print("  bottom endpoint/slope data:", bottom)
print("  maximal bottom spread:", bottom_spread)
print("  maximal positive-facet spread:", positive_spread)
print("  developed boundary lengths:", boundary_lengths)
print("  developed boundary widths:", boundary_widths)
print("  developed apex heights:", boundary_heights)
print("  developed apex phases:", boundary_phases)
print("  global slope envelopes:", slope_envelopes)
print("  mixed-family deck collisions: none")
