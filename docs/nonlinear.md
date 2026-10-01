# Nonlinear RVE solve: cohesive interfaces and plasticity

The `nonlinear` config section runs a small-strain, rate-independent RVE analysis with

- **J2 (von Mises) plasticity** with linear isotropic hardening in the matrix and/or the
  fibres (omit `yield_stress` for an elastic phase);
- **zero-thickness cohesive elements** on every fibre/matrix interface, with the
  traction-separation laws of [`diffcohesive`](https://pypi.org/project/diffcohesive/)
  (mixed-mode bilinear with Benzeggagh-Kenane or power-law closure, or the Alfano shape
  library: bilinear, linear-parabolic, exponential, trapezoidal);
- **periodic** (or affine "dirichlet") boundary conditions with **mixed macro control**:
  one macro strain component follows the load path while the other macro stresses are held
  at zero (uniaxial stress) or the other strains are held at zero (uniaxial strain);
- 2D **plane strain** or **generalized plane strain** (uniform out-of-plane strain with zero
  out-of-plane macro stress, the natural setting for a UD cross-section) and 3D solids.

Two interchangeable engines implement the same formulation:

- `engine: python` (default, `src/rve2d/engines/python/nonlinear/`): PyTorch, vectorised
  over elements, with the traction-separation laws of diffcohesive; the sparse tangent is
  solved with PARDISO (if `pypardiso` is installed), SciPy's SuperLU, or TensorMesh's
  `SparseMatrix` / torch-sla (which also runs on CUDA with `device: cuda`).
- `engine: julia`: Ferrite.jl ([FerriteRVE](../src/rve2d/engines/julia/FerriteRVE)) with
  [DiffCohesive.jl](../src/rve2d/engines/julia/DiffCohesive), the Julia counterpart of
  diffcohesive in this repository. It needs Julia 1.11+ but not PyTorch, and reproduces
  the Python engine to round-off (see [Julia engine](#julia-engine)).

`--engine python|julia` on the command line overrides the configured engine.

```bash
pip install -e ".[gmsh,nonlinear]"          # torch + diffcohesive (+ tensormesh-fem, torch-sla)
pip install pypardiso                        # optional: ~6x faster sparse solves on CPU
rve2d build-and-solve examples/2d/synthetic/nonlinear_cohesive_plastic.yaml
rve2d solve-nonlinear CONFIG.yaml MESH.msh --output-dir out/   # on an existing mesh
```

For a GPU install, get the CUDA build of PyTorch from pytorch.org first and add
`torch-sla[cudss]` for direct sparse solves on the GPU.

## Configuration

```yaml
nonlinear:
  enabled: true
  engine: python                         # or julia (Ferrite.jl + DiffCohesive.jl)
  kinematics: generalized_plane_strain   # 2D: plane_strain | generalized_plane_strain; 3D: solid
  boundary_condition: periodic           # or dirichlet (zero boundary fluctuation)
  matrix: {youngs_modulus: 3500.0, poisson_ratio: 0.35, yield_stress: 60.0, hardening_modulus: 300.0}
  fibre:  {youngs_modulus: 70000.0, poisson_ratio: 0.2}          # elastic (no yield_stress)
  interface:
    law: bilinear_mixed_mode   # or bilinear | linear-parabolic | exponential | trapezoidal
    penalty_stiffness: 1.0e8   # K       [stress / length]
    normal_strength: 50.0      # T_n     [stress]
    shear_strength: 75.0       # T_s     [stress]   (defaults to T_n)
    mode_i_toughness: 0.002    # G_Ic    [stress * length]
    mode_ii_toughness: 0.006   # G_IIc   (defaults to G_Ic)
    bk_exponent: 1.45
    viscosity: 1.0e-4          # damage relaxation time as a fraction of the load path (0 = off)
    integration: nodal         # nodal (Newton-Cotes, default) or gauss
  load:
    type: uniaxial_stress      # or uniaxial_strain
    component: xx              # xx yy zz yz xz xy (active components depend on kinematics)
    max_strain: 0.02
    steps: 40
    unload: false              # true: load to max_strain, then back to zero
  device: cpu                  # or cuda (Python engine)
  linear_solver: auto          # auto | pardiso | scipy | tensormesh (Python engine)
  output_every: 0              # write VTU fields every N steps (0 = final state only)
```

Units are free but must be consistent. With lengths in mm and stresses in MPa, `K` is in
N/mm^3 and toughness in N/mm (1 J/m^2 = 0.001 N/mm). Validation rejects interface parameters
whose final opening would precede damage onset (`G_c <= T^2 / 2K`).

Outputs (in the output directory): `nonlinear_response.csv` (macro strain and stress history,
damage and plasticity measures, work density), `nonlinear_summary.json`,
`nonlinear_final.vtu` (displacement, equivalent plastic strain, von Mises and stress
components) and `nonlinear_final_interface.vtu` (damage, normal and shear opening per
cohesive facet).

## Formulation

Unknowns are the displacement fluctuations `w` (total displacement `E x + w`) and the
stress-controlled macro strain components. Element strains are `eps = B w + E`; the macro
stress is the volume average `sigma_bar = (1/V) sum_e V_e sigma_e`, which is the
Hill-Mandel-consistent conjugate of `E` for periodic fluctuations. Cohesive separations are
`u_fibre - u_matrix` in the reference facet frame (normal into the fibre, so positive normal
separation means opening); they involve only `w` because the duplicated interface nodes
share coordinates. Periodicity ties every image-face node to its mirror partner (matched
per side of the interface) and one interior node carries the rigid-body anchor.

Newton-Raphson uses the consistent tangent of both nonlinearities: the J2 algorithmic
tangent (Simo & Hughes 1998) and the traction-separation tangent obtained by forward-mode
autograd through the diffcohesive law, so any law (analytic or neural) gets an exact
tangent. History variables are committed only after a converged increment; failed
increments are halved and grown back (x1.5) after easy convergence.

Two numerical details matter for real RVEs:

- **Unit safety.** diffcohesive's laws contain small absolute regularisation constants
  (e.g. `sqrt(shear^2 + 1e-12)`), which with realistic stiffnesses (onset opening
  `T/K = 5e-7 mm` in mm/MPa, `5e-10 m` in SI) produce non-zero damage at zero separation. The
  laws are therefore evaluated in units of their own onset opening (exactly covariant for the
  bilinear and shape laws), which makes the results independent of the unit system.
- **Debonding instabilities.** When a fibre debonds, load redistributes suddenly and the
  strain-controlled response can snap back. A small Duvaut-Lions damage viscosity
  (`viscosity`, a relaxation time in units of the load path, applied consistently whatever
  the increment size) regularises this; nodal integration of the cohesive elements avoids
  spurious traction oscillations.

## Verification

All of these run in the test suite (`tests/test_nonlinear.py`, structured meshes, no gmsh):

| Check | Result |
|---|---|
| J2 consistent tangent vs finite differences (yielding points) | agreement to ~1e-10 |
| J2 uniaxial stress path vs analytical bilinear curve | exact (machine precision) |
| Homogeneous RVE with cohesive interfaces (stiff), plane strain / GPS / 3D | `sigma/eps = E/(1-nu^2)`, `E`, `E` to 1e-6 |
| Homogeneous plastic RVE, uniaxial stress, 2D GPS and 3D | analytical bilinear curve to 1e-8 |
| Full bordered tangent (plasticity + interface softening) vs finite differences | ~1e-9 relative |
| Scaled cohesive law: damage at zero opening in mm/MPa and SI; dissipated energy | 0 and `G_Ic` to 1e-3 |
| Weak interface under transverse tension | full debonding and softening vs the bonded RVE |

### Cross-check against the linear homogenization

With elastic phases the nonlinear solver must reproduce the effective stiffness of the
linear homogenization (`solver` section) on the same periodic mesh. Stiffness columns from
uniaxial-strain loads, compared with the linear Ferrite.jl homogenization:

| Case | max abs(C - C_linear) / max abs(C_linear) |
|---|---|
| 2D plane strain, perfectly bonded (no interface) | 1.2e-14 |
| 2D plane strain, stiff cohesive interfaces (`K h / E_matrix` about 6000) | 9.3e-5 |
| 3D solid, perfectly bonded | 1.0e-14 |
| 3D solid, stiff cohesive interfaces | 3.5e-5 |

Perfect bonding agrees to round-off; the small difference with interfaces is the compliance
of a finite penalty stiffness.

### Example runs

`examples/2d/synthetic/nonlinear_cohesive_plastic.yaml` (9 glass fibres, Vf 0.40, epoxy
matrix, transverse uniaxial tension to 2 %, 6,501 unknowns): initial modulus 7.88 GPa,
peak 45.0 MPa at 0.70 % strain, then softening to 38.7 MPa with 24 % of the interface fully
debonded and 13 % of the RVE yielded; 52 increments (one cut), 293 Newton iterations, 21 s
on 4 CPU cores with PARDISO.

Debonding robustness on the same RVE (40 initial increments):

| Cohesive integration, `viscosity` | Outcome |
|---|---|
| gauss, 0 | stops at 0.82 % strain (increment cuts exhausted after the first debond) |
| nodal, 0 (default) | completes; peak 45.0 MPa at 0.70 % |
| nodal, 1e-4 | completes; peak 45.4 MPa at 0.74 % |
| nodal, 1e-3 | completes; peak 46.8 MPa at 0.75 % |

A viscosity up to about 1e-4 regularises without visibly changing the response; larger
values delay and raise the peak.

`examples/3d/synthetic/nonlinear_cohesive_plastic.yaml` (3D solid, 776 cohesive triangles,
4,961 unknowns, uniaxial stress to 1.2 %): initial modulus 6.86 GPa, peak 44.0 MPa at
0.98 % strain; 27 increments, 137 Newton iterations, 37 s.

## Julia engine

`engine: julia` runs the same solve on Ferrite.jl. Two Julia packages ship inside the
Python package, in `src/rve2d/engines/julia/`:

- [`DiffCohesive`](../src/rve2d/engines/julia/DiffCohesive): traction-separation laws
  with ForwardDiff tangents, a port of diffcohesive's mixed-mode bilinear law and Alfano
  shape library with unit-safe regularisation. Its tests check dissipated energies, unit
  independence, the AD tangent and parity with diffcohesive 0.1.2: tractions agree to
  1e-14 relative.
- [`FerriteRVE`](../src/rve2d/engines/julia/FerriteRVE): the RVE solvers. For the
  nonlinear solve: interface insertion, periodic constraints, J2 plasticity on Tensors.jl,
  cohesive elements on DiffCohesive, bordered Newton with the same stepping rules, UMFPACK
  solves, VTU output via Ferrite. It also holds the Julia engine's linear homogenization.

```bash
rve2d doctor --setup-julia                 # once (Julia 1.11+); otherwise done on first use
rve2d build-and-solve config.yaml --engine julia
```

The Python side writes the mesh arrays and `julia_nonlinear_input.toml` to the output
directory and runs `julia --project=<FerriteRVE> run.jl julia_nonlinear_input.toml` in the
bundled environment (set up on first use; copied to `~/.cache/rve2d` first if the install
directory is read-only). Outputs and the summary JSON are the same as for the Python
engine, plus the Julia log `nonlinear_log.txt`. `RVE2D_JULIA` selects the Julia executable
and `RVE2D_JULIA_TIMEOUT` (seconds, default one day) bounds a run. The package precompiles
a small workload, so a solve in a fresh process costs about 10 s more than in a warm
session.

Both engines take identical increments and Newton iterations on the examples:

| Example | Increments / cuts / iterations | max abs(stress difference) / max stress | Runtime (Python with PARDISO / Julia) |
|---|---|---|---|
| 2D, `nonlinear_cohesive_plastic.yaml` | 52 / 1 / 293 (both) | 1.3e-14 | 21 s / 30 s (fresh process) |
| 3D, `nonlinear_cohesive_plastic.yaml` | 27 / 1 / 137 (both) | 4.5e-15 | 37 s / 39 s (fresh process) |

Damage, debonded and yielded fractions, equivalent plastic strain and work density agree
to 1e-15 as well; `tests/test_julia_nonlinear.py` checks the same parity on a structured
mesh whenever the Julia environment is instantiated.

## Limitations

- Small strains and rotations: facet frames and strain measures use the reference
  configuration.
- Linear triangles and tetrahedra only (`mesh_order: 1`, no `recombine`).
- The load is strain-controlled, so a macroscopic snap-back (sudden debonding of many
  fibres at once) cannot be followed; the increment is cut until `max_step_cuts` is
  exhausted and the solve stops with a message, keeping the results up to that point.
  A small `viscosity` usually carries the solve through.
- There is no friction or contact in shear. A fibre that debonds around its whole
  perimeter can rotate freely (a zero-energy mode); the normal penalty still prevents
  interpenetration.
