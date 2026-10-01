# Nonlinear RVE solve: cohesive interfaces and plasticity

The `nonlinear` config section runs a small-strain, rate-independent RVE analysis with

- **J2 (von Mises) plasticity** with linear isotropic hardening in the matrix and/or the
  fibres (omit `yield_stress` for an elastic phase), or the **pressure-dependent
  paraboloidal plasticity** of epoxies (different yield stresses in tension and
  compression, non-associated flow), optionally with **ductile damage** regularised by the
  element size (crack band), see [Matrix plasticity and damage](#matrix-plasticity-and-damage);
- **zero-thickness cohesive elements** on every fibre/matrix interface, with the
  traction-separation laws of [`diffcohesive`](https://pypi.org/project/diffcohesive/)
  (mixed-mode bilinear with Benzeggagh-Kenane or power-law closure, or the Alfano shape
  library: bilinear, linear-parabolic, exponential, trapezoidal);
- **periodic** (or affine "dirichlet") boundary conditions with **mixed macro control**:
  one macro strain component follows the load path while the other macro stresses are held
  at zero (uniaxial stress) or the other strains are held at zero (uniaxial strain);
- 2D **plane strain** or **generalized plane strain** (uniform out-of-plane strain with zero
  out-of-plane macro stress, the natural setting for a UD cross-section) and 3D solids.

The laminate pipeline ([laminate.md](laminate.md)) runs this solve for the transverse
tension, transverse compression and shear curves of a ply.

Two interchangeable engines implement the same formulation:

- `engine: tensormesh` (default, `src/rve2d/engines/tensormesh/nonlinear/`): PyTorch,
  vectorised over elements, with the traction-separation laws of diffcohesive; the sparse
  tangent is solved with SciPy's SuperLU on the CPU, or with TensorMesh's `SparseMatrix` /
  torch-sla, which also runs on CUDA with `device: cuda`. (`python`, its former name, is
  still accepted.)
- `engine: julia`: Ferrite.jl ([FerriteRVE](../src/rve2d/engines/julia/FerriteRVE)) with
  [DiffCohesive.jl](../src/rve2d/engines/julia/DiffCohesive), the Julia counterpart of
  diffcohesive in this repository. It needs Julia 1.11+ but not PyTorch, and reproduces
  the TensorMesh engine to round-off (see [Julia engine](#julia-engine)).

`--engine tensormesh|julia` on the command line overrides the configured engine.

```bash
pip install -e ".[gmsh,nonlinear]"          # torch + diffcohesive (+ tensormesh-fem, torch-sla)
rve2d build-and-solve examples/2d/synthetic/nonlinear_cohesive_plastic.yaml
rve2d solve-nonlinear CONFIG.yaml MESH.msh --output-dir out/   # on an existing mesh
```

For a GPU install, get the CUDA build of PyTorch from pytorch.org first and add
`torch-sla[cudss]` for direct sparse solves on the GPU.

## Configuration

```yaml
nonlinear:
  enabled: true
  engine: tensormesh                     # or julia (Ferrite.jl + DiffCohesive.jl)
  kinematics: generalized_plane_strain   # 2D: plane_strain | generalized_plane_strain; 3D: solid
  boundary_condition: periodic           # or dirichlet (zero boundary fluctuation)
  matrix: {youngs_modulus: 3500.0, poisson_ratio: 0.35, yield_stress: 60.0, hardening_modulus: 300.0}
  fibre:  {youngs_modulus: 70000.0, poisson_ratio: 0.2}          # elastic (no yield_stress)
  # or a pressure-dependent epoxy with damage (all keys below are optional):
  # matrix:
  #   youngs_modulus: 3350.0
  #   poisson_ratio: 0.35
  #   yield_stress: 50.0                # yield in uniaxial tension
  #   compressive_yield_stress: 75.0    # yield in uniaxial compression (default: yield_stress)
  #   plastic_poisson_ratio: 0.3        # lateral / axial plastic strain (default 0.5)
  #   hardening_modulus: 3000.0         # in tension; compression hardens in proportion
  #   saturation_stress: 80.0           # Voce hardening towards this stress (default: linear)
  #   damage_onset_strain: 0.1          # equivalent plastic strain at damage onset
  #   fracture_energy: 0.002            # G_f [stress * length], dissipated per crack area
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
  device: cpu                  # or cuda (TensorMesh engine)
  linear_solver: auto          # auto (SciPy on CPU, TensorMesh on CUDA) | scipy | tensormesh
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
per side of the interface) and one matrix node (an interior one when the mesh has any)
carries the rigid-body anchor.

Newton-Raphson uses the consistent tangent of both nonlinearities: the algorithmic
tangent of the plasticity model (J2: Simo & Hughes 1998; paraboloidal model and damage:
below) and the traction-separation tangent obtained by forward-mode
autograd through the diffcohesive law, so any law (analytic or neural) gets an exact
tangent. History variables are committed only after a converged increment; failed
increments are halved and grown back (x1.5) after easy convergence.

### Matrix plasticity and damage

With `compressive_yield_stress` (σc0) different from `yield_stress` (σt0), or
`plastic_poisson_ratio` (νp) below 0.5, a phase follows the paraboloidal yield criterion
used for epoxy matrices (Tschoegl 1971; Melro et al. 2013),

    f = q² + 3 p (σc − σt) − σc σt,    p = tr(σ) / 3,  q = von Mises stress,

which reaches the yield stress σt in uniaxial tension and σc in uniaxial compression, and
a non-associated flow potential `g = q² + α p²` with `α = 9 (1 − 2νp) / (2 (1 + νp))`, so
that the plastic strain in uniaxial loading has the lateral-to-axial ratio νp. Hardening
is homothetic, `σc = σc0 σt / σt0`, and in tension either linear, `σt = σt0 + H ε̄p`, or,
with `saturation_stress` σs, of Voce's saturating form

    σt = σs − (σs − σt0) exp(−H ε̄p / (σs − σt0))      (initial slope H),

which lets a polymer matrix level off at its strength in tension, compression
(`σs σc0 / σt0`) and shear (`sqrt(σc σt / 3)` on the paraboloid). `ε̄p` is the equivalent
plastic strain with increment `dε̄p = sqrt(dεp : dεp / (1 + 2νp²))`, which is the axial
plastic strain in uniaxial tension or compression. The return mapping solves one scalar
equation for the plastic multiplier (safeguarded Newton, to round-off) and gives the exact,
non-symmetric consistent tangent. With σc0 = σt0 and νp = 0.5 it is J2 plasticity; such a
phase with linear hardening and no damage keeps the closed-form J2 return mapping.

With `damage_onset_strain` and `fracture_energy`, damage starts when ε̄p reaches the onset
value ε̄p0 and grows linearly with it,

    d = min(0.99, (ε̄p − ε̄p0) l_e σt(ε̄p0) / (2 G_f)),    σ = (1 − d) σ̃,

where σ̃ is the stress of the plasticity model and `l_e` the element's crack-band length
(`sqrt(2A)` for a triangle, `(6V)^(1/3)` for a tetrahedron; Bažant & Oh 1983). Without
hardening after onset, a localised band of one element dissipates `G_f` per unit crack
area, whatever the element size (with hardening a little more). The damage term is part of
the consistent tangent. Damage turns the strain-controlled response into a softening one;
when it localises in a band the macro response can snap back, and the solve then stops
after `max_step_cuts` (a peak and the start of the softening are what the ply curves of
the laminate pipeline need).

### Numerical details

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
| Paraboloidal model with σc0 = σt0, νp = 0.5 against the J2 return mapping | stresses to 1e-11 |
| Paraboloidal model, uniaxial tension and compression vs the bilinear curves (σt0, σc0, proportional hardening); plastic Poisson ratio | to 1e-9 |
| Voce hardening, uniaxial tension and compression (paraboloidal and J2) vs the saturating curve | to 1e-9 |
| Paraboloidal model with damage: non-symmetric consistent tangent vs finite differences | ~1e-6 relative |
| Damage: work of a softening point = `G_f / l_e` | to 2e-3 |
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
matrix, transverse uniaxial tension to 2 %, 6,507 unknowns): initial modulus 7.88 GPa,
peak 45.0 MPa at 0.70 % strain, then softening to 38.6 MPa with 23 % of the interface fully
debonded and 13 % of the RVE yielded; 56 increments (three cuts), 331 Newton iterations,
42 s on 4 CPU cores with the TensorMesh engine.

Debonding robustness on the same RVE (40 initial increments):

| Cohesive integration, `viscosity` | Outcome |
|---|---|
| gauss, 0 | completes, but needs 12 increment cuts after the first debond (87 increments); peak 45.2 MPa at 0.73 % (on an earlier mesh of this example it stopped at 0.82 % strain) |
| nodal, 0 (default) | completes with 3 cuts (56 increments); peak 45.0 MPa at 0.70 % |
| nodal, 1e-4 | completes (67 increments); peak 45.4 MPa at 0.73 % |
| nodal, 1e-3 | completes (53 increments, 1 cut); peak 46.8 MPa at 0.75 % |

A viscosity up to about 1e-4 regularises without visibly changing the response; larger
values delay and raise the peak.

`examples/3d/synthetic/nonlinear_cohesive_plastic.yaml` (3D solid, 776 cohesive triangles,
4,961 unknowns, uniaxial stress to 1.2 %): initial modulus 6.86 GPa, peak 44.0 MPa at
0.98 % strain; 27 increments, 137 Newton iterations, 66 s.

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
directory is read-only). Outputs and the summary JSON are the same as for the TensorMesh
engine, plus the Julia log `nonlinear_log.txt`. `RVE2D_JULIA` selects the Julia executable
and `RVE2D_JULIA_TIMEOUT` (seconds, default one day) bounds a run. The package precompiles
its workload when the environment is set up; starting Julia and loading the engine still
adds about 4 s to every run.

Both engines take identical increments and Newton iterations on the examples:

| Example | Increments / cuts / iterations | max abs(stress difference) / max stress |
|---|---|---|
| 2D, `nonlinear_cohesive_plastic.yaml` | 56 / 3 / 331 (both) | 1.3e-14 |
| 3D, `nonlinear_cohesive_plastic.yaml` | 27 / 1 / 137 (both) | 6.3e-15 |

Damage, debonded and yielded fractions, equivalent plastic strain and work density agree
to about 1e-14 as well; `tests/test_julia_nonlinear.py` checks the same parity on a
structured mesh whenever the Julia environment is set up, also with a pressure-dependent,
damaging matrix and for longitudinal shear on an extruded layer.

## Performance

Wall time of `rve2d solve-nonlinear` on the example meshes (median of two runs, 4-core
Xeon at 2.8 GHz, CPU only). Every variant takes the same increments and Newton iterations
and gives the same results; only the speed differs.

| Example | Julia | TensorMesh (default: SciPy SuperLU) | TensorMesh, `linear_solver: tensormesh` |
|---|---|---|---|
| 2D (6,507 unknowns, 331 Newton iterations) | 42 s | 42 s | 54 s |
| 3D (4,961 unknowns, 137 Newton iterations) | 50 s | 66 s | 101 s |

- The TensorMesh engine evaluates all elements at once in PyTorch (on all cores) and
  factorizes with SciPy's SuperLU, which is single-threaded and has more fill-in on 3D
  meshes.
- The Julia engine runs its own code on one thread (UMFPACK uses 2 BLAS threads) and
  spends about 4 s starting up.
- With pressure-dependent plasticity or damage the material tangent is not symmetric and
  each Newton iteration costs a little more on both engines; the laminate pipelines
  ([laminate.md](laminate.md#performance)) give timings for such matrices.
- `linear_solver: tensormesh` hands the factorization to SciPy on the CPU and adds
  conversions, so it is the slowest CPU choice; it exists for `device: cuda`, where the
  whole Newton loop stays on the GPU (not measured here).
- Intel MKL PARDISO (through `pypardiso`) took 30–50 % less time than SciPy here, but MKL
  only exists for x86-64 Linux and Windows (not macOS or ARM), so it is not offered.

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
