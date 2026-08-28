"""Exact cubic labelled-window certificate for quartic GKT/Jacobi operations.

The four cases distribute three distinct descendent-one labels over the two
standard singleton twists alpha=(1,0) and beta=(0,1).  For each case the
script derives every endpoint invariant from the GKT set-partition chamber
equation, reconstructs the complete GKT slope factorization recursively by
label support, factors the exact inverse commutator of the two aggregate
singleton lines, and computes the cubic residual.  All arithmetic is exact.
"""


from math import gcd


S.<x, y> = PolynomialRing(QQ, 2)


def bits(mask):
    """Increasing tuple of label indices in a bit mask."""

    output = []
    index = 0
    while mask:
        if mask & 1:
            output.append(index)
        mask >>= 1
        index += 1
    return tuple(output)


def mask_of(indices):
    output = 0
    for index in indices:
        output |= 1 << index
    return output


def set_partitions(items):
    """Unordered set partitions, with blocks and entries canonically ordered."""

    items = tuple(items)
    if not items:
        yield tuple()
        return
    first = items[0]
    for partition in set_partitions(items[1:]):
        yield ((first,),) + partition
        for position in range(len(partition)):
            enlarged = list(partition)
            enlarged[position] = (first,) + enlarged[position]
            yield tuple(enlarged)


def gamma_ratio(top_numerator, base_numerator, denominator=4):
    assert top_numerator >= base_numerator
    assert (top_numerator - base_numerator) % denominator == 0
    output = QQ.one()
    for step in range((top_numerator - base_numerator) // denominator):
        output *= QQ(base_numerator + step * denominator) / denominator
    return output


def gamma_weight(exponent, residue):
    return gamma_ratio(exponent[0] + 1, residue[0] + 1) * gamma_ratio(
        exponent[1] + 1, residue[1] + 1
    )


def monomial(exponent):
    return x**exponent[0] * y**exponent[1]


def slope(k):
    return QQ(k[1] + 1) / QQ(k[0] + 1)


def shifted_ray(k):
    divisor = gcd(k[0] + 1, k[1] + 1)
    return ((k[0] + 1) // divisor, (k[1] + 1) // divisor)


def jacobi_derivative(poly, k):
    a, b = k
    return S(
        x**a
        * y**b
        * ((b + 1) * x * poly.derivative(x) - (a + 1) * y * poly.derivative(y))
    )


def add_state(left, right):
    output = dict(left)
    for support, coefficient in right.items():
        output[support] = output.get(support, S.zero()) + coefficient
        if output[support] == 0:
            del output[support]
    return output


def apply_generator(state, generator):
    """Apply one square-zero labelled generator (support,k,coefficient)."""

    support, k, coefficient = generator
    output = {}
    for old_support, poly in state.items():
        if old_support & support:
            continue
        new_support = old_support | support
        value = coefficient * jacobi_derivative(poly, k)
        if value:
            output[new_support] = output.get(new_support, S.zero()) + value
    return output


def flow(state, generator):
    return add_state(state, apply_generator(state, generator))


def chronological_product(state, generators):
    output = state
    for generator in generators:
        output = flow(output, generator)
    return output


def negate(generator):
    support, k, coefficient = generator
    return (support, k, -coefficient)


def block_geometry(labels, mask):
    indices = bits(mask)
    A = sum(labels[index][0] for index in indices)
    B = sum(labels[index][1] for index in indices)
    residue = (A % 4, B % 4)
    carry = ((A - residue[0]) // 4, (B - residue[1]) // 4)
    # Every selected label has descendent degree one, so Notation 2.38 gives
    # N_balanced=carry_1+carry_2+1 independently of the support size.
    N_balanced = carry[0] + carry[1] + 1
    profiles = tuple(
        (residue[0] + 4 * p, residue[1] + 4 * (N_balanced - p))
        for p in range(N_balanced + 1)
    )
    critical = tuple(
        (residue[0] + 4 * p, residue[1] + 4 * (N_balanced - 1 - p))
        for p in range(N_balanced)
    )
    return {
        "sum": (A, B),
        "residue": residue,
        "profiles": profiles,
        "critical": critical,
    }


def endpoint_system(labels, endpoint):
    """Solve all GKT chamber equations at the minimum or maximum endpoint."""

    assert endpoint in ("minimum", "maximum")
    count = len(labels)
    solved = {}
    for size in range(1, count + 1):
        for mask in range(1, 1 << count):
            if len(bits(mask)) != size:
                continue
            geometry = block_geometry(labels, mask)
            profile = (
                geometry["profiles"][0]
                if endpoint == "minimum"
                else geometry["profiles"][-1]
            )
            target = -QQ.one() if size == 1 else QQ.zero()
            proper_partition_sum = QQ.zero()
            for partition in set_partitions(bits(mask)):
                if len(partition) == 1:
                    continue
                block_masks = tuple(mask_of(block) for block in partition)
                total_exponent = (
                    sum(solved[block_mask]["exponent"][0] for block_mask in block_masks),
                    sum(solved[block_mask]["exponent"][1] for block_mask in block_masks),
                )
                product = prod(
                    solved[block_mask]["invariant"] for block_mask in block_masks
                )
                proper_partition_sum += gamma_weight(
                    total_exponent, geometry["residue"]
                ) * product
            invariant = (target - proper_partition_sum) / gamma_weight(
                profile, geometry["residue"]
            )
            solved[mask] = {
                "invariant": invariant,
                "exponent": profile,
                "geometry": geometry,
            }
    return solved


def endpoint_potential(endpoint_data):
    state = {0: x**4 + y**4}
    for mask, data in endpoint_data.items():
        sign = (-1) ** (len(bits(mask)) - 1)
        state[mask] = sign * data["invariant"] * monomial(data["exponent"])
    return state


def polynomial_vector(poly, profiles):
    return vector(
        QQ,
        [poly.monomial_coefficient(monomial(profile)) for profile in profiles],
    )


def solve_gkt_factorization(labels, W_min, W_max):
    """Recover every GKT factor recursively in increasing support size."""

    count = len(labels)
    generators = []
    coefficients = {}
    for size in range(1, count + 1):
        for mask in range(1, 1 << count):
            if len(bits(mask)) != size:
                continue
            geometry = block_geometry(labels, mask)
            if size == 1:
                index = bits(mask)[0]
                a, b = labels[index]
                values = (-QQ.one() / ((a + 1) * (b + 1)),)
            else:
                baseline = chronological_product(
                    W_min, sorted(generators, key=lambda generator: slope(generator[1]))
                )
                profiles = geometry["profiles"]
                columns = tuple(
                    polynomial_vector(jacobi_derivative(x**4 + y**4, k), profiles)
                    for k in geometry["critical"]
                )
                incidence = matrix(QQ, columns).transpose()
                target = polynomial_vector(
                    W_max[mask] - baseline.get(mask, S.zero()), profiles
                )
                values = tuple(incidence.solve_right(target))
            coefficients[mask] = dict(zip(geometry["critical"], values))
            for k, coefficient in coefficients[mask].items():
                generators.append((mask, k, coefficient))

    ordered = tuple(sorted(generators, key=lambda generator: slope(generator[1])))
    assert chronological_product(W_min, ordered) == W_max
    return coefficients, ordered


def inverse_aggregate_commutator(labels):
    """Exact inverse commutator of the aggregate alpha and beta line factors."""

    alpha = []
    beta = []
    for index, twist in enumerate(labels):
        generator = (1 << index, twist, -QQ(1) / 2)
        if twist == (1, 0):
            alpha.append(generator)
        else:
            assert twist == (0, 1)
            beta.append(generator)
    # Displayed operator product e^B e^A e^-B e^-A acts chronologically as
    # e^-A, e^-B, e^A, e^B.  Factors within either aggregate line commute.
    word = tuple(map(negate, alpha)) + tuple(map(negate, beta)) + tuple(alpha) + tuple(beta)
    return {
        "x": chronological_product({0: x}, word),
        "y": chronological_product({0: y}, word),
    }


def solve_scattering_factorization(labels, target):
    """Factor the aggregate inverse commutator recursively by label support."""

    count = len(labels)
    generators = []
    coefficients = {}
    for size in range(2, count + 1):
        for mask in range(1, 1 << count):
            if len(bits(mask)) != size:
                continue
            baseline_x = chronological_product(
                {0: x}, sorted(generators, key=lambda generator: slope(generator[1]))
            )
            baseline_y = chronological_product(
                {0: y}, sorted(generators, key=lambda generator: slope(generator[1]))
            )
            difference_x = target["x"].get(mask, S.zero()) - baseline_x.get(
                mask, S.zero()
            )
            difference_y = target["y"].get(mask, S.zero()) - baseline_y.get(
                mask, S.zero()
            )
            k = block_geometry(labels, mask)["sum"]
            coefficient = difference_x.monomial_coefficient(
                x ** (k[0] + 1) * y**k[1]
            ) / (k[1] + 1)
            assert difference_x == coefficient * jacobi_derivative(x, k)
            assert difference_y == coefficient * jacobi_derivative(y, k)
            coefficients[mask] = coefficient
            if coefficient:
                generators.append((mask, k, coefficient))

    ordered = tuple(sorted(generators, key=lambda generator: slope(generator[1])))
    assert chronological_product({0: x}, ordered) == target["x"]
    assert chronological_product({0: y}, ordered) == target["y"]
    return coefficients, ordered


def run_case(name, labels):
    minimum = endpoint_system(labels, "minimum")
    maximum = endpoint_system(labels, "maximum")
    W_min = endpoint_potential(minimum)
    W_max = endpoint_potential(maximum)
    gkt, gkt_generators = solve_gkt_factorization(labels, W_min, W_max)
    commutator = inverse_aggregate_commutator(labels)
    scattering, scattering_generators = solve_scattering_factorization(labels, commutator)

    full_mask = (1 << len(labels)) - 1
    geometry = block_geometry(labels, full_mask)
    gkt_full = gkt[full_mask]
    assert len(geometry["critical"]) == 1
    k = geometry["critical"][0]
    assert k == geometry["sum"]
    scattering_full = scattering[full_mask]
    residual_full = scattering_full - gkt_full[k]

    # Pair-support checks recover Proposition 6.21 inside every three-label
    # window and ensure the aggregate commutator orientation is fixed.
    for mask in range(1, 1 << len(labels)):
        if len(bits(mask)) != 2:
            continue
        twists = {labels[index] for index in bits(mask)}
        expected = QQ(3) / 4 if len(twists) == 2 else QQ.zero()
        assert scattering[mask] == expected

    return {
        "name": name,
        "labels": labels,
        "minimum_invariant": minimum[full_mask]["invariant"],
        "maximum_invariant": maximum[full_mask]["invariant"],
        "gkt": gkt_full[k],
        "scattering": scattering_full,
        "residual": residual_full,
        "k": k,
        "gkt_generators": gkt_generators,
        "scattering_generators": scattering_generators,
    }


alpha = (1, 0)
beta = (0, 1)
cases = (
    run_case("alpha^3", (alpha, alpha, alpha)),
    run_case("alpha^2 beta", (alpha, alpha, beta)),
    run_case("alpha beta^2", (alpha, beta, beta)),
    run_case("beta^3", (beta, beta, beta)),
)

expected = {
    "alpha^3": {
        "k": (3, 0),
        "invariants": (-QQ(120), -QQ(36)),
        "gkt": -QQ(3) / 4,
        "scattering": QQ.zero(),
        "residual": QQ(3) / 4,
    },
    "alpha^2 beta": {
        "k": (2, 1),
        "invariants": (-QQ(84), -QQ(56)),
        "gkt": -QQ(1) / 2,
        "scattering": QQ(3) / 4,
        "residual": QQ(5) / 4,
    },
    "alpha beta^2": {
        "k": (1, 2),
        "invariants": (-QQ(56), -QQ(84)),
        "gkt": -QQ(1) / 2,
        "scattering": -QQ(3) / 4,
        "residual": -QQ(1) / 4,
    },
    "beta^3": {
        "k": (0, 3),
        "invariants": (-QQ(36), -QQ(120)),
        "gkt": -QQ(3) / 4,
        "scattering": QQ.zero(),
        "residual": QQ(3) / 4,
    },
}

assert {case["name"] for case in cases} == set(expected)
for case in cases:
    target = expected[case["name"]]
    assert case["k"] == target["k"]
    assert (
        case["minimum_invariant"],
        case["maximum_invariant"],
    ) == target["invariants"]
    assert case["gkt"] == target["gkt"]
    assert case["scattering"] == target["scattering"]
    assert case["residual"] == target["residual"]

# Polarizing the unlabelled cubic factors of check_singleton_scattering.sage
# into two distinct labels of the repeated type multiplies their coefficients
# by 2.  This independently fixes the signs and normalization of the
# aggregate-line commutator calculation.
assert expected["alpha^2 beta"]["scattering"] / 2 == QQ(3) / 8
assert expected["alpha beta^2"]["scattering"] / 2 == -QQ(3) / 8

for case in cases:
    print("CASE", case["name"])
    print("  cubic exponent:", case["k"], "shifted ray:", shifted_ray(case["k"]))
    print(
        "  endpoint invariants:",
        case["minimum_invariant"],
        case["maximum_invariant"],
    )
    print("  cubic GKT coefficient:", case["gkt"])
    print("  cubic scattering coefficient:", case["scattering"])
    print("  cubic residual coefficient:", case["residual"])

print("ALL THREE-LABEL GKT/JACOBI OPERATION CHECKS PASSED")
