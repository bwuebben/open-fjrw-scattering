# The scattering structure of open FJRW theory of x^r + y^s

**Bernd J. Wuebben** — papers, computations, and replication code.

Gross, Kelly and Tessler ([arXiv:2203.02435](https://arxiv.org/abs/2203.02435))
constructed a genus-zero open enumerative theory for the Landau–Ginzburg model
(x^r + y^s, μ_r × μ_s), in which the open FJRW invariants depend on boundary
conditions and the systems of invariants form a torsor under a wall-crossing
group that they observe is "very similar in spirit" to the tropical vertex group
of Kontsevich–Soibelman. This paper turns the observation into a structure
theory. Main results:

1. **Weight trichotomy and the threshold** (Thm 1.1). Pro-nilpotent walls exist
   exactly in positive Euler weight; primary (d = 0) wall-crossing occurs if and
   only if the central charge exceeds 1, with the marginal weight carried by a
   torus direction.
2. **The relative scattering presentation** (Thm 1.2). Around any loop of
   boundary conditions the ordered product of the critical-graph jump
   automorphisms is the identity, and every element of the enlarged
   wall-crossing group factors uniquely as a slope-ordered product of
   exponentials supported on the **shifted boundary rays**
   ℝ₊(k₁+1, k₂+1) — the marginal torus terms lying on the (1,1)-ray. The
   resulting boundary-to-boundary transport is a *relative* scattering
   structure: unlike a consistent diagram on a complete affine base, its total
   transport need not be the identity.
3. **Normal forms and the canonical diagram** (Thm 1.3). On every wall-bearing
   diagonal the wall action is triangular, so there are unique extreme chamber
   indices; the boundary-slope-ordered factorization of the group element
   joining them is a canonical, choice-free primary diagram D₀^{r,s}, computed
   order by order. For x⁵ + y⁵ the first wall functions are computed exactly
   (coefficients 1/25, 1/25, 1/50 on the rays (1,2), (2,1), (1,1)).
4. **Effective determination** (Thm 1.4). The open topological recursion is the
   direction transverse to the walls: consistency and the closed theory
   determine all invariants up to the torsor action.
5. **The transport representation** (Thm 1.5). Anchoring each potential
   coefficient at the one-sided slope cut through its own direction, the
   potential family is the coefficientwise broken-line transport of a unique
   cut-independent seed system over D^{r,s}. The seeds have a finite triangular
   inversion as alternating ordered bend chains (Thm 7.10), a rational oriented
   zero-bordism cascade class (Thm 7.11), and a realization as the virtual Euler
   cycle of one compact finite Kuranishi bar-cascade object (Thm 7.14). Seeds
   vanish at the extreme coefficients reached by no foreign bends; interlacing
   and a monotone-bending mechanism characterize this purely tropical part of
   the theory. Their nonzero values are non-tropical input beyond the tropical
   limit.
6. **Finite-support and pro-geometric realization** (Thm 1.6 = Thm 7.20).
   With descendents the walls accumulate densely already at degree one in the
   deformation parameters — for every (r, s), including the simple singularities,
   which carry no primary wall at all — so open chambers are replaced by
   coefficientwise slope cuts. Nevertheless, on every finite divisor-closed
   primary or descendent coefficient set, the diagram **is realized by an actual
   family of boundary conditions**: each boundary-ray factor is realized in its
   own stage, and the stages are concatenated in increasing slope, so that every
   jump is supported on a single ray and the total jump on each ray is its wall
   automorphism. Relative endpoint and homotopy extensions make these
   realizations compatible coefficientwise.

   The staging is necessary, not an artifact. One *global* normal-form family
   gives every marking subset a single jump time, and so cannot separate the
   walls of a diagonal that carries two of them (Prop 7.23, Cor 7.25); the
   smallest instance, J = {(3,3)⁵} for x⁵ + y⁵, is computed in Ex 7.24, where the
   transition between the extreme chamber indices is shown to force both wall
   coefficients. Such a family therefore suffices exactly below N₂(r, s), the
   first degree admitting a diagonal with two walls, for which Prop 7.21 gives
   the exact count

   > N_max(r, s; m) = m + 1 − ⌈2m/r⌉ − ⌈2m/s⌉,

   so that N₂ is infinite precisely for the simple and parabolic pairs and, in
   the hyperbolic range, is at most 21 — attained only at x³ + y⁷ (Rem 7.26 gives
   the complete classification; N₂ = 5 for x⁵ + y⁵).
7. **Polar extreme representatives** (Prop 7.29). The two extreme invariant
   systems admit symmetric representatives for which every nonextreme balanced
   section is nowhere zero, compatibly coefficientwise. The statement is an
   existence theorem; no preferred polar family is asserted.
8. **Projective coherence and the position-space obstruction**
   (Thm 1.7 = Thm 7.31). The staged families extend to homotopy-coherent path
   and disk transport on the punctured boundary-charge sector, but the transport
   factors through slope and radial paths act trivially. Since the punctured
   sector is contractible, the extreme-to-extreme product is boundary-to-boundary
   transport, not origin monodromy. A genuine singular-affine position space and
   theta theory therefore require new valuation, affine-gluing, and binary
   enumerative input beyond the present GKT continuation data.

Among the newly computed invariant values for x⁵ + y⁵:

| invariant | value |
|---|---|
| ⟨τ₀^(3,3) τ₀^(3,3) σ₁σ₂⁶σ₁₂⟩ + ⟨τ₀^(3,3) τ₀^(3,3) σ₁⁶σ₂σ₁₂⟩ | −2/5 |
| ⟨τ₀^(3,3) τ₀^(3,3) τ₀^(3,3) σ₁⁴σ₂⁴σ₁₂⟩ | 1/5 |
| ⟨τ₀^(1,2) τ₀^(2,1) τ₀^(2,2) σ₁₂⟩ | −1/25 |
| ⟨τ₂^(3,3) σ₁³σ₂¹³σ₁₂⟩ | 25/36 |

The general two-insertion (|J| = 2) seed relation is
ν₍ᵣ,₀₎/r + ν₍₀,ₛ₎/s = −1/(rs); the paper also records three corrections to the
literature (Remarks 2.3 and 4.3).

## Contents

```
paper/            the 33-page paper (main.tex, fjrw-scattering.pdf)
companion/        the companion note on the central-charge threshold
                  (main.tex, fjrw-threshold.pdf)
src/              the verification scripts (see the map below)
requirements.txt  Python dependencies (sympy)
```

## Replication

All claims are verified by exact polynomial and rational arithmetic — no
floating point. Every script is self-contained, prints what it verifies, and
ends with `ALL CHECKS PASSED` (or a per-check report).

```sh
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
for f in src/*.py; do ./venv/bin/python "$f"; done
```

The one exception is `src/verify_thm05_full.sage`, which needs
[SageMath](https://www.sagemath.org) with the
[admcycles](https://pypi.org/project/admcycles/) package
(`sage -pip install admcycles`); run it as `sage src/verify_thm05_full.sage`.

### Script map

| script | verifies |
|---|---|
| `gkt_algebra.py` | the wall-crossing algebra (GKT Def 4.22), graph classification, first walls, period-invisibility — Thms 1.1, 1.4 |
| `oscillatory_check.py` | seed relations directly from the open mirror theorem — Remark 4.3 |
| `a_invariants.py` | the invariant combinations 𝒜(J,d,ν), convention pin, the t₃₃-tower — Thm 1.4, Remark 4.3 |
| `canonical_diagram.py` | normal forms and the first wall functions of D^{5,5} — Thm 1.3 |
| `transport_engine.py` | the nonlinear transport identity in the pure t₃₃-family through t⁴ |
| `seed_chain_formula.py` | the ordered bend-chain weights and triangular seed inversion — Lemma 7.9, Thm 7.10 |
| `mixed_sector.py` | the mixed t₂₃–t₃₃ family through t⁴: the first scattering vertex, the transport identity in all five chambers |
| `anchor_induction.py` | the linear-order mechanism (randomized), singleton towers, middle seeds — Thm 7.7, Ex 7.17 |
| `backscatter.py` | the no-backwards-bending theorem (randomized) and the far-side census — Thm 7.5 |
| `farside_test.py`, `canonical_seeds.py` | the two smallest far-side cases and the canonical-seed mechanism — Thm 1.5, Ex 7.18 |
| `ray_accumulation.py`, `ray_density_exact.py` | accumulation of the descendent walls and local finiteness of the primary diagram — Prop 7.19 |
| `diagonal_wall_count.py` | walls per primary diagonal: the exact maximal-twist count, the equivalence "at most one wall everywhere ⟺ central charge at most 1", and the complete piecewise classification of N₂ against multiset enumeration — Prop 7.21, Rem 7.26 |
| `first_two_ray_diagonal.py` | the first primary diagonal of x⁵+y⁵ with two walls on distinct rays, its chamber-index relation and extreme value, with the lower orders re-derived as a self-check — Ex 7.24 |
| `two_ray_forced.py` | that the transition between the extreme chamber indices forces both wall coefficients on that diagonal, so the two walls are genuinely crossed — Ex 7.24, Cor 7.25 |
| `mirror_periods.py` | the good-basis periods — Lemma 4.1 |
| `rspin.py`, `taut_m0n.py`, `closed_fjrw.py` | the closed FJRW data of x⁵+y⁵ (Saito primitive form + a self-contained genus-0 tautological-intersection calculus), with internal consistency checks |
| `verify_thm05.py`, `verify_thm05_full.sage` | the GKT open topological recursion verified in full (Neveu–Schwarz + Ramond; all 72 two-insertion and all 105 three-insertion instances) |
| `scattering.py` | the abstract Hamiltonian bracket and leading BCH cancellation, retained as a negative control for naive ungraded seeding — Prop 2.2 |

## References

- M. Gross, T. L. Kelly, R. J. Tessler, *Open FJRW theory and mirror symmetry*,
  [arXiv:2203.02435](https://arxiv.org/abs/2203.02435).
- M. Gross, T. L. Kelly, R. J. Tessler, *Open enumerative geometries for
  Landau–Ginzburg models*, [arXiv:2602.12707](https://arxiv.org/abs/2602.12707).
- M. Carl, M. Pumperla, B. Siebert, *A tropical view on Landau–Ginzburg models*,
  [arXiv:2205.07753](https://arxiv.org/abs/2205.07753).
- R. Maher, *Predictions in open Fan–Jarvis–Ruan–Witten theory via mirror
  symmetry, modularity, and wall-crossing*, Ph.D. thesis, University of
  Birmingham, 2024.

## License

MIT — see [LICENSE](LICENSE).
