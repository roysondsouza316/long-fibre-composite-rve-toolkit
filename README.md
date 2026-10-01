# rve2d-fibre

`rve2d-fibre` is an open-source Python package for building **2D and 3D
representative volume elements (RVEs)** of long-fibre composite plies and solving them:

- **linear homogenization**: the effective stiffness and engineering constants of the RVE
  under periodic or affine boundary conditions;
- **nonlinear RVE solves**: J2 plasticity in the fibres and matrix and cohesive-zone
  fibre/matrix interfaces (debonding) under mixed macro strain/stress control;
- **laminates**: ply properties from the RVE, then for any stacking sequence the ABD
  matrix, engineering constants and 3D effective stiffness, and a laminate tensile test
  (stress–strain curve, elongation and force).

The solves run on either of **two interchangeable engines** that give the same results
to round-off:

| Engine   | Linear homogenization | Nonlinear solve                                                                                   | Needs                          |
| -------- | --------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------ |
| `tensormesh` (default) | NumPy / SciPy | PyTorch with the laws of [diffcohesive](https://pypi.org/project/diffcohesive/); SciPy (CPU) or TensorMesh (CPU, CUDA) sparse solvers | nothing extra / the `nonlinear` extra |
| `julia`  | [Ferrite.jl](https://ferrite-fem.github.io/) | Ferrite.jl with the bundled DiffCohesive.jl laws                                    | Julia 1.11+                    |

> Despite the historical `rve2d` name, this package supports both 2D and 3D RVEs.

![Synthetic example](docs/images/synthetic_rve.svg)
![Mask import example](docs/images/mask_import.svg)

---

## What can it do?

You pick **one of three input modes** and **a dimension**, and the package
takes care of geometry, meshing, periodic-boundary metadata, the solves and
post-processing exports.

| Input mode         | Description                                                              | 2D | 3D |
| ------------------ | ------------------------------------------------------------------------ | -- | -- |
| `synthetic`        | Random circular (2D) / cylindrical (3D) fibres in a rectangle / box; fibre volume fractions up to about 0.65 with the relaxation packing | ✅ | ✅ |
| `image`            | Segmented image (any bit depth, grey or colour) → polygonal fibres (2D) or extruded volumes (3D) | ✅ | ✅ |
| `sem_to_synthetic` | Raw SEM micrograph → segment + fit circles/ellipses → clean RVE          | ✅ | —  |

Pipeline stages:

1. **Geometry**: synthetic generator (random sequential or relaxation packing, optional
   periodic wrapping), mask importer, or SEM extractor
2. **Quality validation**: spacing (periodic minimum-image distances), clipped, wrapped and
   outside fibres, gaps thinner than the mesh size
3. **Meshing**: gmsh OpenCASCADE fragments (conforming fibre/matrix interfaces, fibres
   clipped to the domain), periodic node matching
4. **Export**: `.msh`, `.xdmf` (or any meshio format), JSON metadata bundle
5. **Linear homogenization** (optional, `solver` section): effective stiffness,
   engineering constants, stress–strain and traction CSVs, `.vtu` fields for ParaView
6. **Nonlinear RVE solve** (optional, `nonlinear` section): J2 plasticity + cohesive
   interfaces; macro stress–strain curve, damage and plasticity histories, `.vtu` fields
7. **Laminate pipeline** (optional, `laminate` section, `rve2d laminate`): ply properties
   from stages 5–6, then stiffness and tensile test of every stacking sequence

---

## Installation

```bash
python -m pip install -e ".[gmsh,dev]"
rve2d doctor                      # what is installed, and what each part is for
```

Required Python: **3.11+**. This is enough for geometry, meshing, linear
homogenization with the TensorMesh engine and the laminate stiffness pipeline. On Linux,
gmsh also needs the system OpenGL/X libraries (e.g. `apt install libglu1-mesa libxcursor1
libxinerama1 libxft2`). Without gmsh, geometry generation and validation still work, but
meshing fails with a clear error.

The **TensorMesh engine's nonlinear solve** needs the `nonlinear` extra (PyTorch,
diffcohesive, TensorMesh):

```bash
python -m pip install -e ".[gmsh,nonlinear]"
```

The **Julia engine** needs [Julia](https://julialang.org/downloads) 1.11 or newer on
`PATH` (or `RVE2D_JULIA=/path/to/julia`). Its Julia packages ship inside the Python
package and use their own environment, so nothing is installed into your global Julia
environment. The first Julia solve sets it up automatically (downloads Ferrite.jl and
precompiles, a few minutes); to do that ahead of time:

```bash
rve2d doctor --setup-julia
```

---

## Choosing an engine

```yaml
solver:              # linear homogenization
  enabled: true
  engine: tensormesh # or julia
nonlinear:           # plasticity + cohesive interfaces
  enabled: true
  engine: tensormesh # or julia
```

or override both from the command line with `--engine julia`. The result files are the
same whichever engine runs; the summary JSON records which one did. (`python`, the
TensorMesh engine's former name, is still accepted.)

Which is faster depends on the job (4-core CPU): the TensorMesh engine is faster for small
linear problems (Julia spends about 4 s starting up) and as fast as Julia for the 2D
nonlinear example (40 s against 38 s); the Julia engine is faster for the 3D nonlinear
example (41 s against 59 s) and about 5× faster for large 3D linear homogenization
(150,000 unknowns: 43 s against 208 s). Only the TensorMesh engine runs on a GPU.
See the performance sections of [`docs/nonlinear.md`](docs/nonlinear.md#performance) and
[`docs/homogenization.md`](docs/homogenization.md#performance).

---

## Quick start: pick your starting point

The [`examples/`](examples/) tree mirrors the two choices you have to make:
**dimension** and **input mode**.

```
examples/
├── 2d/
│   ├── synthetic/   # random circular fibres in a rectangle
│   ├── image/       # binary mask → polygonal fibres
│   └── sem/         # raw SEM micrograph → mask or fitted circles/ellipses
├── 3d/
│   ├── synthetic/   # random cylindrical fibres in a box
│   └── image/       # 2D mask extruded into a 3D box
└── pipelines/       # RVE → ply → laminates (stacking sequences, tensile test)
    ├── tensormesh/  # the same pipelines on the TensorMesh engine ...
    └── julia/       # ... and on the Julia engine
```

### Common commands (run from the repo root)

```bash
# 1. Validate any config first
rve2d validate-config examples/2d/synthetic/basic.yaml

# 2. Build geometry + mesh + metadata
rve2d build           examples/2d/synthetic/basic.yaml

# 3. Build + linear homogenization in one step (TensorMesh engine; --engine julia for Ferrite.jl)
rve2d build-and-solve examples/2d/synthetic/periodic_solve.yaml
rve2d build-and-solve examples/3d/synthetic/periodic_solve.yaml --engine julia

# 4. Build a 2D RVE from a binary segmented mask
rve2d build-and-solve examples/2d/image/basic_solve.yaml

# 5. Build a 3D RVE by extruding a 2D mask
rve2d build           examples/3d/image/extruded.yaml

# 6. Convert a raw SEM micrograph into circular/elliptical fibres
#    (drop your own image at examples/2d/sem/sem_sample.png first)
rve2d build           examples/2d/sem/to_synthetic.yaml

# 7. A periodic RVE at fibre volume fraction 0.60 (relaxation packing, periodic wrapping)
rve2d build-and-solve examples/2d/synthetic/high_vf_periodic.yaml

# 8. Run a batch study and collect the engineering constants of every case
rve2d batch-study \
  examples/2d/synthetic/periodic_solve.yaml \
  examples/2d/image/basic_solve.yaml \
  --output-dir outputs/batch_demo

# 9. Nonlinear RVE: matrix plasticity + fibre/matrix debonding
rve2d build-and-solve examples/2d/synthetic/nonlinear_cohesive_plastic.yaml
rve2d build-and-solve examples/3d/synthetic/nonlinear_cohesive_plastic.yaml --engine julia

# 10. Laminates: stiffness of several stacking sequences (seconds), then the tensile test
rve2d laminate examples/pipelines/tensormesh/laminate_stiffness_2d.yaml
rve2d laminate examples/pipelines/julia/laminate_tensile_2d.yaml
```

The full picker table is in [`examples/README.md`](examples/README.md).

---

## CLI reference

```text
rve2d [--traceback] COMMAND ...

rve2d validate-config CONFIG_PATH
rve2d build           CONFIG_PATH [--output-dir PATH] [--basename NAME]
rve2d solve           CONFIG_PATH MESH_PATH --output-dir PATH [--engine tensormesh|julia]
rve2d solve-nonlinear CONFIG_PATH MESH_PATH --output-dir PATH [--engine tensormesh|julia]
rve2d build-and-solve CONFIG_PATH [--output-dir PATH] [--basename NAME] [--engine tensormesh|julia]
rve2d batch-study     CONFIG_PATH [CONFIG_PATH ...] --output-dir PATH [--engine ...] [--fail-fast]
rve2d laminate        CONFIG_PATH [--output-dir PATH] [--engine tensormesh|julia] [--ply PLY_JSON]
rve2d doctor          [--setup-julia]
rve2d solve-ferrite   CONFIG_PATH MESH_PATH --output-dir PATH   # same as solve --engine julia
```

Errors in the configuration or the input files are reported in one line; `--traceback`
shows the full Python traceback instead.

`build` writes:

- mesh files (`.msh`, `.xdmf`, …)
- `geometry_summary.json`
- `phase_tags.json`
- `quality_report.json`: `valid` plus the defects behind it (fibres closer than
  `min_spacing`, fibres cut by the boundary that are not complete periodic wraps) and
  informational counts and notes (removed specks, wrapped and outside fibres, gaps
  thinner than the mesh size)
- `periodic_pairs.json` (when periodic metadata is enabled)

`solve` and `build-and-solve` (when `solver.enabled: true`) write:

- `homogenization_summary.json`: effective stiffness, engineering constants, average
  stresses, boundary tractions, volume fractions, material rotations, engine and timing
- `homogenized_stiffness.csv`
- `engineering_constants.csv`: `ex, ey, ez, gyz, gxz, gxy, nuxy, nuyx, nuxz, nuzx, nuyz, nuzy`
  for 3D and generalized plane strain; `ex, ey, gxy, nuxy, nuyx` for plane stress;
  `ex_plane_strain, …` for plane strain (plane-strain moduli, see
  [homogenization notes](docs/homogenization.md#engineering-constants))
- `stress_strain_response.csv`: the unit macro strain of each load case and the
  resulting average stress
- `traction_response.csv`: average traction on each boundary face per load case
- `homogenization.vtu` (when `solver.write_vtk: true`): displacement of every load case
  (macro strain × position + periodic fluctuation) at the nodes, stress per cell, phase id
- Julia engine only: `julia_homogenization_input.toml`, the mesh arrays it reads,
  `julia_homogenization_result.toml` and the Julia log `homogenization_log.txt`

`solve-nonlinear` (and `build-and-solve` when `nonlinear.enabled: true`) writes:

- `nonlinear_response.csv`: macro strain/stress history, interface damage,
  plasticity measures, work density
- `nonlinear_summary.json`: peak stress, initial modulus, convergence statistics
- `nonlinear_final.vtu` and `nonlinear_final_interface.vtu`: displacement,
  stresses, equivalent plastic strain, interface damage and openings
- Julia engine only: `julia_nonlinear_input.toml`, the mesh arrays it reads and the
  Julia log `nonlinear_log.txt`

`batch-study` writes one folder per case and `study_summary.csv` with one row per case
(status and error message for failed cases, which do not stop the study unless
`--fail-fast` is given).

`laminate` writes `rve/` (the RVE build), `ply/` (the RVE solves and
`ply_properties.json`), `laminates/summary.csv` (one row per stacking sequence),
`laminates/<sequence>/` (`abd.csv`, `effective_3d_stiffness.csv`, `constants.json` and,
with the tensile test, `tensile_test.csv` and `tensile_test_summary.json`),
`laminates/tensile_tests.png` and `pipeline_summary.json`; see
[`docs/laminate.md`](docs/laminate.md).

---

## Configuration cheat-sheet

Top-level keys:

| Key                | Type     | Notes                                                        |
| ------------------ | -------- | ------------------------------------------------------------ |
| `mode`             | string   | `synthetic`, `image`, or `sem_to_synthetic`                  |
| `dimension`        | int      | `2` or `3`                                                   |
| `synthetic`        | object   | Required when `mode: synthetic`                              |
| `image`            | object   | Required when `mode: image`                                  |
| `sem_to_synthetic` | object   | Required when `mode: sem_to_synthetic` (2D only)             |
| `mesh`             | object   | gmsh meshing parameters                                      |
| `export`           | object   | Output dir, basename, formats                                |
| `solver`           | object   | Optional linear homogenization: engine, kinematics, BCs, materials |
| `nonlinear`        | object   | Optional plasticity + cohesive-interface RVE solve           |
| `laminate`         | object   | Optional laminate pipeline: ply thickness, stacking sequences, tensile test |

YAML and JSON are both supported. See [`docs/configuration.md`](docs/configuration.md)
for every field, default, and example value.

---

## Package layout

The two engines live side by side under `engines/`; everything they share (material
matrices, result files, load paths) is in `engines/common/`.

```
src/rve2d/
├── cli.py                  # `rve2d` entry point
├── workflow.py             # build_rve / solve_homogenization / solve_nonlinear / build_and_solve_rve
├── study.py                # batch-study driver
├── doctor.py               # `rve2d doctor` installation checks
├── config.py               # dataclass schema + YAML/JSON loader + validation
├── models.py               # Domain / Fibre / GeometryModel dataclasses
├── mesh_io.py              # quiet mesh reading, 2D mesh extrusion into tetrahedra
├── exceptions.py
├── synthetic_generation/   # 2D circular, 3D cylindrical fibres; SEM → fitted circles/ellipses
├── image_import/           # binary mask → 2D polygons / extruded 3D volumes
├── geometry_cleanup/
├── meshing/gmsh_builder.py
├── export/writers.py
├── validation/checks.py
├── laminate/               # engine-independent: works on the RVE results
│   ├── stacking.py         # stacking-sequence notation: [0/±45/90]2s, [0_2/90]T, ...
│   ├── ply.py              # ply stiffness and stress–strain curves from the RVE results
│   ├── clt.py              # ABD matrix, engineering constants, 3D effective stiffness
│   ├── tensile.py          # laminate tensile test: stress, strain, elongation, force
│   └── pipeline.py         # `rve2d laminate`: RVE -> ply -> laminates
└── engines/
    ├── __init__.py         # homogenize() / solve_nonlinear(): dispatch to the chosen engine
    ├── common/             # shared by both engines
    │   ├── materials.py        # phase stiffness matrices, rotations
    │   ├── homogenization.py   # problem/result types, engineering constants, result files
    │   └── records.py          # nonlinear load paths, step records, result files
    ├── tensormesh/         # TensorMesh engine
    │   ├── mesh.py             # mesh reading, interface node duplication, cohesive elements
    │   ├── constraints.py      # periodic / affine fluctuation constraints
    │   ├── homogenization.py   # linear homogenization (SciPy sparse LU, one factorization)
    │   └── nonlinear/          # PyTorch: J2 plasticity, diffcohesive cohesive elements,
    │                           # bordered Newton, SciPy SuperLU / TensorMesh solvers
    └── julia/              # Julia engine
        ├── runner.py           # find Julia, set up the bundled environment, run a task
        ├── homogenization.py   # writes the TOML input, reads the TOML result
        ├── nonlinear.py
        ├── run.jl              # entry script: julia --project=FerriteRVE run.jl INPUT.toml
        ├── FerriteRVE/         # Julia package: linear homogenization + nonlinear RVE on Ferrite.jl
        └── DiffCohesive/       # Julia package: cohesive laws with ForwardDiff tangents

tools/make_diffcohesive_reference.py   # regenerates DiffCohesive.jl's parity data from diffcohesive
```

---

## Linear homogenization

For each unit macro strain (3 cases for plane stress/strain, 6 for generalized plane
strain and 3D) the RVE is solved for the displacement fluctuation; the volume-averaged
stresses are the columns of the effective stiffness. One sparse factorization serves all
load cases.

- **Kinematics**: `plane_stress` (thin lamina), `plane_strain` (`eps_zz = 0`, gives
  plane-strain moduli), `generalized_plane_strain` (a UD cross-section with fibres along
  z: uniform out-of-plane strains, full 6×6 stiffness and true engineering constants from
  a 2D mesh) and `solid` (3D).
- **Boundary conditions**: `periodic` (node-matched periodic mesh, needs
  `periodic_compatible: true`) or `dirichlet` (zero boundary fluctuation, i.e. affine
  displacement on the boundary: an upper bound of the periodic result).
- **Materials**: isotropic or orthotropic phases from gmsh physical ids; orthotropic
  phases need the in-plane constants for plane stress and all nine 3D constants
  otherwise, and are checked for positive-definite compliance.
- **Material rotations**: `material_angle_deg` (about z) or `material_angle_x/y/z_deg`
  (`Rz Ry Rx`); plane stress allows in-plane rotations only. For generalized plane strain
  with fibres along z, rotate the fibre's axis 1 onto z with `fibre_material_angle_y_deg: -90`.
- **Rotation precedence** (highest to lowest): explicit `solver` angles; geometry metadata
  `phase_orientation_rotations_deg` (3D generators); the 2D geometry's `orientation_deg`
  (fibres only). `build-and-solve`, `solve` and `solve-ferrite` all apply the same rules
  (`solve` reads `geometry_summary.json` next to the mesh).

See [`docs/homogenization.md`](docs/homogenization.md) for the formulation, conventions and
verification.

---

## Nonlinear RVE solve: plasticity + cohesive interfaces

The `nonlinear` config section runs a small-strain, rate-independent RVE solve on
the generated mesh:

- J2 plasticity with linear isotropic hardening in the matrix and/or fibres
- zero-thickness cohesive elements on every fibre/matrix interface, with
  diffcohesive's traction-separation laws (mixed-mode bilinear with
  Benzeggagh–Kenane or power-law closure; bilinear, linear-parabolic,
  exponential and trapezoidal shape laws)
- periodic (or affine) boundary conditions with uniaxial-stress or
  uniaxial-strain macro loading, optional unloading
- 2D plane strain or generalized plane strain, and 3D solids
- consistent Newton tangents (the cohesive tangent comes from automatic
  differentiation through the law), adaptive load stepping
- TensorMesh engine: PyTorch with SciPy's SuperLU on CPU and TensorMesh's solvers (CPU or
  CUDA); Julia engine: Ferrite.jl with
  [DiffCohesive.jl](src/rve2d/engines/julia/DiffCohesive), a Julia port of
  diffcohesive's laws with ForwardDiff tangents. Both engines take the same increments
  and Newton iterations and agree to round-off.

See [`docs/nonlinear.md`](docs/nonlinear.md) for the configuration, the
formulation and the verification results.

---

## Laminates: stacking sequences, stiffness and tensile test

`rve2d laminate CONFIG` turns the RVE into plies and the plies into laminates. It runs on
either engine (only the RVE solves depend on it) for 2D and 3D RVEs:

1. **Ply stiffness**: the linear homogenization of the RVE (generalized plane strain in 2D,
   solid in 3D) with the fibres along the ply's axis 1; optionally averaged to transverse
   isotropy.
2. **Ply curves** (for the tensile test): nonlinear RVE solves under uniaxial stress give the
   transverse (RVE `xx`) and in-plane shear (RVE `xz`) stress–strain curves. For a 2D RVE the
   shear runs on one periodic layer of tetrahedra extruded from its mesh: the exact
   z-invariant 3D problem at a few times the cost of the 2D solve.
3. **Laminates**: for every entry of `stacking_sequences` (`[0/90]s`, `[0/±45/90]2s`,
   `[0_2/90]T`, `[(±45)2/0]`, ...): the ABD matrix, membrane and flexural engineering
   constants, coupling flags and the 3D effective stiffness of the stack.
4. **Tensile test** (optional): strain-controlled in x, y or xy with the other resultants and
   moments zero; plies follow the RVE curves (secant law with damage memory) and fail in the
   fibre direction at the given strengths. Output: stress–strain curve, damage events,
   elongation over `gauge_length` and force over `width`.

```yaml
laminate:
  enabled: true
  ply_thickness: 0.125
  stacking_sequences: ["[0/90]2s", "[±45]2s", "[0/±45/90]s"]
  longitudinal_tensile_strength: 750.0      # fibre failure is an input
  tensile_test: {direction: x, max_strain: 0.03, steps: 300, gauge_length: 150.0, width: 25.0}
```

To try other stacking sequences, edit the list and rerun with `--ply
OUTPUT/ply/ply_properties.json`: the RVE is not solved again and the laminates take
seconds. Ready-made pipelines for both engines are in
[`examples/pipelines/`](examples/pipelines/); the formulation and its checks are in
[`docs/laminate.md`](docs/laminate.md).

---

## Development

```bash
pytest                  # run the test suite
ruff check .            # lint
mypy                    # type-check (strict; configured in pyproject.toml)
```

The physics tests (single-material recovery, bounds, periodicity of the solution,
generalized plane strain against a 3D extrusion, laminate theory against closed forms and
a layered finite-element RVE) run on structured meshes and need no gmsh; end-to-end tests
build the examples with gmsh and are skipped without it. Tests of the TensorMesh engine's
nonlinear solve are skipped without PyTorch and diffcohesive; tests comparing the two
engines run when Julia is installed and the engine environment is set up
(`rve2d doctor --setup-julia`).

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs ruff, mypy and the
tests (with gmsh) on Python 3.11 / 3.12 / 3.13, a job with the `nonlinear` extra (CPU
PyTorch), the tests of the two Julia packages, and the engine-parity tests with Julia.
The Julia packages can also be tested directly:

```bash
julia --project=src/rve2d/engines/julia/DiffCohesive -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=src/rve2d/engines/julia/FerriteRVE   -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

---

## Roadmap

- elliptical and polygonal fibre cross-sections in synthetic mode
- waviness and tow-scale path generation
- graded or multi-material matrix regions
- direct export helpers for additional FEM solvers
- true volumetric 3D reconstruction from serial-section / micro-CT slice stacks
- richer 3D orientation input from segmentation metadata

---

## License

MIT; see [LICENSE](LICENSE).
