# rve2d-fibre

`rve2d-fibre` is an open-source Python package for building **2D and 3D
representative volume elements (RVEs)** of long-fibre composite plies and solving them:

- **linear homogenization**: the effective stiffness and engineering constants of the RVE
  under periodic or affine boundary conditions;
- **nonlinear RVE solves**: J2 plasticity in the fibres and matrix and cohesive-zone
  fibre/matrix interfaces (debonding) under mixed macro strain/stress control.

Both solves run on either of **two interchangeable engines** that give the same results
to round-off:

| Engine   | Linear homogenization | Nonlinear solve                                                                                   | Needs                          |
| -------- | --------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------ |
| `python` (default) | NumPy / SciPy | PyTorch with the laws of [diffcohesive](https://pypi.org/project/diffcohesive/), TensorMesh sparse solvers (CPU and CUDA) | nothing extra / the `nonlinear` extra |
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
| `synthetic`        | Random circular (2D) / cylindrical (3D) fibres in a rectangle / box      | ✅ | ✅ |
| `image`            | Binary segmented mask → polygonal fibres (2D) or extruded volumes (3D)   | ✅ | ✅ |
| `sem_to_synthetic` | Raw SEM micrograph → segment + fit circles/ellipses → clean RVE          | ✅ | —  |

Pipeline stages:

1. **Geometry**: synthetic generator, mask importer, or SEM extractor
2. **Quality validation**: spacing, clipping, disconnected-region checks
3. **Meshing**: gmsh OpenCASCADE Boolean cuts, periodic node matching
4. **Export**: `.msh`, `.xdmf` (or any meshio format), JSON metadata bundle
5. **Linear homogenization** (optional, `solver` section): effective stiffness,
   engineering constants, stress–strain and traction CSVs, `.vtu` fields for ParaView
6. **Nonlinear RVE solve** (optional, `nonlinear` section): J2 plasticity + cohesive
   interfaces; macro stress–strain curve, damage and plasticity histories, `.vtu` fields

---

## Installation

```bash
python -m pip install -e ".[gmsh,dev]"
rve2d doctor                      # what is installed, and what each part is for
```

Required Python: **3.11+**. This is enough for geometry, meshing and linear
homogenization with the Python engine. On Linux, gmsh also needs the system
OpenGL/X libraries (e.g. `apt install libglu1-mesa libxcursor1 libxinerama1 libxft2`).
Without gmsh, geometry generation and validation still work, but meshing fails with a
clear error.

The **Python engine's nonlinear solve** needs the `nonlinear` extra (PyTorch,
diffcohesive, TensorMesh):

```bash
python -m pip install -e ".[gmsh,nonlinear]"
python -m pip install pypardiso   # optional, x86 CPUs: several times faster sparse solves
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
  engine: python     # or julia
nonlinear:           # plasticity + cohesive interfaces
  enabled: true
  engine: python     # or julia
```

or override both from the command line with `--engine julia`. The result files are the
same whichever engine runs; the summary JSON records which one did.

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
└── 3d/
    ├── synthetic/   # random cylindrical fibres in a box
    └── image/       # 2D mask extruded into a 3D box
```

### Common commands (run from the repo root)

```bash
# 1. Validate any config first
rve2d validate-config examples/2d/synthetic/basic.yaml

# 2. Build geometry + mesh + metadata
rve2d build           examples/2d/synthetic/basic.yaml

# 3. Build + linear homogenization in one step (Python engine; add --engine julia for Ferrite.jl)
rve2d build-and-solve examples/2d/synthetic/periodic_solve.yaml
rve2d build-and-solve examples/3d/synthetic/periodic_solve.yaml --engine julia

# 4. Build a 2D RVE from a binary segmented mask
rve2d build-and-solve examples/2d/image/basic_solve.yaml

# 5. Build a 3D RVE by extruding a 2D mask
rve2d build           examples/3d/image/extruded.yaml

# 6. Convert a raw SEM micrograph into circular/elliptical fibres
#    (drop your own image at examples/2d/sem/sem_sample.png first)
rve2d build           examples/2d/sem/to_synthetic.yaml

# 7. Run a batch study and collect the engineering constants of every case
rve2d batch-study \
  examples/2d/synthetic/periodic_solve.yaml \
  examples/2d/image/basic_solve.yaml \
  --output-dir outputs/batch_demo

# 8. Nonlinear RVE: matrix plasticity + fibre/matrix debonding
rve2d build-and-solve examples/2d/synthetic/nonlinear_cohesive_plastic.yaml
rve2d build-and-solve examples/3d/synthetic/nonlinear_cohesive_plastic.yaml --engine julia
```

The full picker table is in [`examples/README.md`](examples/README.md).

---

## CLI reference

```text
rve2d [--traceback] COMMAND ...

rve2d validate-config CONFIG_PATH
rve2d build           CONFIG_PATH [--output-dir PATH] [--basename NAME]
rve2d solve           CONFIG_PATH MESH_PATH --output-dir PATH [--engine python|julia]
rve2d solve-nonlinear CONFIG_PATH MESH_PATH --output-dir PATH [--engine python|julia]
rve2d build-and-solve CONFIG_PATH [--output-dir PATH] [--basename NAME] [--engine python|julia]
rve2d batch-study     CONFIG_PATH [CONFIG_PATH ...] --output-dir PATH [--engine ...] [--fail-fast]
rve2d doctor          [--setup-julia]
rve2d solve-ferrite   CONFIG_PATH MESH_PATH --output-dir PATH   # same as solve --engine julia
```

Errors in the configuration or the input files are reported in one line; `--traceback`
shows the full Python traceback instead.

`build` writes:

- mesh files (`.msh`, `.xdmf`, …)
- `geometry_summary.json`
- `phase_tags.json`
- `quality_report.json`
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
├── exceptions.py
├── synthetic_generation/   # 2D circular, 3D cylindrical fibres; SEM → fitted circles/ellipses
├── image_import/           # binary mask → 2D polygons / extruded 3D volumes
├── geometry_cleanup/
├── meshing/gmsh_builder.py
├── export/writers.py
├── validation/checks.py
└── engines/
    ├── __init__.py         # homogenize() / solve_nonlinear(): dispatch to the chosen engine
    ├── common/             # shared by both engines
    │   ├── materials.py        # phase stiffness matrices, rotations
    │   ├── homogenization.py   # problem/result types, engineering constants, result files
    │   └── records.py          # nonlinear load paths, step records, result files
    ├── python/             # Python engine
    │   ├── mesh.py             # mesh reading, interface node duplication, cohesive elements
    │   ├── constraints.py      # periodic / affine fluctuation constraints
    │   ├── homogenization.py   # linear homogenization (SciPy sparse LU, one factorization)
    │   └── nonlinear/          # PyTorch: J2 plasticity, diffcohesive cohesive elements,
    │                           # bordered Newton, PARDISO / SuperLU / TensorMesh solvers
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
- Python engine: PyTorch with PARDISO / SuperLU / TensorMesh solvers on CPU and
  TensorMesh on CUDA; Julia engine: Ferrite.jl with
  [DiffCohesive.jl](src/rve2d/engines/julia/DiffCohesive), a Julia port of
  diffcohesive's laws with ForwardDiff tangents. Both engines take the same increments
  and Newton iterations and agree to round-off.

See [`docs/nonlinear.md`](docs/nonlinear.md) for the configuration, the
formulation and the verification results.

---

## Development

```bash
pytest                  # run the test suite
ruff check .            # lint
mypy                    # type-check (strict; configured in pyproject.toml)
```

The physics tests (single-material recovery, bounds, periodicity of the solution,
generalized plane strain against a 3D extrusion) run on structured meshes and need no
gmsh; end-to-end tests build the examples with gmsh and are skipped without it. Tests of
the Python nonlinear solve are skipped without PyTorch and diffcohesive; tests comparing
the two engines run when Julia is installed and the engine environment is set up
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
- denser periodic packing and edge-wrapping inclusions
- direct export helpers for additional FEM solvers
- true volumetric 3D reconstruction from serial-section / micro-CT slice stacks
- richer 3D orientation input from segmentation metadata

---

## License

MIT; see [LICENSE](LICENSE).
