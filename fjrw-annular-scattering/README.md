# Annular scattering in quartic open FJRW theory

Source, PDF and exact computations for Paper 3.

- [`fjrw-annular-scattering.pdf`](fjrw-annular-scattering.pdf) is the compiled
  manuscript.
- `main.tex`, `references.tex`, `sections/`, and `figures/` are the complete
  LaTeX source. Build with `latexmk -pdf main.tex` from this directory.
- `src/` contains exact checks of the finite algebraic and combinatorial
  identities used in the paper.

## Running the checks

Run from the repository root. The Python programs need SymPy; two of them
import the shared engine `src/a_invariants.py`.

```sh
python3 fjrw-annular-scattering/src/check_support_rule.py
python3 fjrw-annular-scattering/src/check_two_label_endpoint.py
python3 fjrw-annular-scattering/src/check_marginal_saito_series.py
python3 fjrw-annular-scattering/src/check_mixed_primary_saito_slice.py
for f in fjrw-annular-scattering/src/*.sage; do sage "$f"; done
```

| Program | Section | What it checks |
|---|---|---|
| `check_p112_degeneration.sage` | 2, 3, App. A | the degeneration of the quartic pencil, the cofactor lattice, the parity cover, the monodromies and the core holonomy |
| `check_support_rule.py` | 2 | the cofactor as the Legendre transform of a quadratic function, its equivariance under exchanging $x$ and $y$, and the annihilator rule |
| `check_stopped_pro_log_charts.sage` | 2 to 4 | root degrees and local root data at the singular points |
| `check_singleton_scattering.sage` | 3 | the ordered factorization at the joint through total degree ten, and a loop test at the joint |
| `check_outward_cap_valuations.sage` | 4 | exit data of the walls at the boundary of the disk |
| `check_outward_affine_caps.sage` | 4 | the outward flares on the two boundary circles |
| `check_horizontal_cap_atlas.sage` | 4, App. A | development of the two boundary circles and separation of the exits |
| `check_regular_log_refinement.sage` | 4 | the nonregular root stage and the regular presentation |
| `check_gkt_labelled_transport.sage` | 5 | the labelled critical-algebra embedding and the retwisting obstruction |
| `check_gkt_wall_operation.sage` | 5 | the first mixed truncation: GKT transition and wall factors |
| `check_gkt_three_label_operation.sage` | 5 | three-label truncations of the all-order factorization |
| `check_gkt_four_label_operation.sage` | 5 | four-label truncations of the all-order factorization |
| `check_residual_associator.sage` | 5 | the degree-weighted residual associator |
| `check_period_module_intertwiner.sage` | 6 | transport of the twisted de Rham complex |
| `check_marginal_calibration.sage` | 6 | the marginal calibration |
| `check_marginal_saito_series.py` | 6 | the primary marginal Saito series |
| `check_mixed_primary_saito_slice.py` | 6 | the first mixed primary slice |
| `check_two_label_endpoint.py` | 7 | the two-label truncations, the maximal normal form and the period classes |

The programs verify finite algebraic identities. The geometric statements of
the paper are proved in the manuscript.
