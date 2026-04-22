# rve2d-fibre

`rve2d-fibre` is an open-source Python package for building **2D and 3D
representative volume elements (RVEs)** of long-fibre composite plies and
running linear-elastic homogenization on them via [Ferrite.jl](https://ferrite-fem.github.io/).

> Despite the historical `rve2d` name, this package now supports both 2D and
> 3D RVEs.

![Synthetic example](docs/images/synthetic_rve.svg)
![Mask import example](docs/images/mask_import.svg)

---

## What can it do?

You pick **one of three input modes** and **a dimension**, and the package
takes care of geometry, meshing, periodic-boundary metadata, solver inputs,
the homogenization solve, and post-processing exports.

| Input mode         | Description                                                              | 2D | 3D |
| ------------------ | ------------------------------------------------------------------------ | -- | -- |
| `synthetic`        | Random circular (2D) / cylindrical (3D) fibres in a rectangle / box      | ✅ | ✅ |
| `image`            | Binary segmented mask → polygonal fibres (2D) or extruded volumes (3D)   | ✅ | ✅ |
| `sem_to_synthetic` | Raw SEM micrograph → segment + fit circles/ellipses → clean RVE          | ✅ | —  |

Pipeline stages:

1. **Geometry** — synthetic generator, mask importer, or SEM extractor
2. **Quality validation** — spacing, clipping, disconnected-region checks
3. **Meshing** — gmsh OpenCASCADE Boolean cuts, periodic node matching
4. **Export** — `.msh`, `.xdmf` (or any meshio format), JSON metadata bundle
5. **Homogenization** — Ferrite.jl 2D-triangle / 3D-tetrahedral solve
6. **Post-processing** — homogenized stiffness, engineering constants,
   stress–strain CSVs, `.vtu` for ParaView

---

## Installation

```bash
python -m pip install -e ".[gmsh,dev]"
```

Required Python: **3.11+**.

The Julia solve stage additionally requires:

- A working [Julia](https://julialang.org) installation (Julia 1.10+).
- The Julia packages declared in [`julia/Project.toml`](julia/Project.toml).
  Install them once with:

  ```bash
  cd julia && julia --project=. -e 'using Pkg; Pkg.instantiate()'
  ```

If `gmsh` is not installed, geometry generation and validation still work, but
meshing commands will fail with a clear error.

---

## Quick start — pick your starting point

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

# 3. Build + Ferrite.jl homogenization in one step
rve2d build-and-solve examples/2d/synthetic/periodic_solve.yaml
rve2d build-and-solve examples/3d/synthetic/periodic_solve.yaml

# 4. Build a 2D RVE from a binary segmented mask
rve2d build-and-solve examples/2d/image/basic_solve.yaml

# 5. Build a 3D RVE by extruding a 2D mask
rve2d build           examples/3d/image/extruded.yaml

# 6. Convert a raw SEM micrograph into circular/elliptical fibres
#    (drop your own image at examples/2d/sem/sem_sample.png first)
rve2d build           examples/2d/sem/to_synthetic.yaml

# 7. Run a batch parametric study and aggregate engineering constants
rve2d batch-study \
  examples/2d/synthetic/periodic_solve.yaml \
  examples/2d/image/basic_solve.yaml \
  --output-dir outputs/batch_demo
```

The full picker table is in [`examples/README.md`](examples/README.md).

---

## CLI reference

```text
rve2d validate-config CONFIG_PATH
rve2d build           CONFIG_PATH [--output-dir PATH] [--basename NAME]
rve2d solve-ferrite   CONFIG_PATH MESH_PATH --output-dir PATH
rve2d build-and-solve CONFIG_PATH [--output-dir PATH] [--basename NAME]
rve2d batch-study     CONFIG_PATH [CONFIG_PATH ...] --output-dir PATH
```

`build` writes:

- mesh files (`.msh`, `.xdmf`, …)
- `geometry_summary.json`
- `phase_tags.json`
- `quality_report.json`
- `periodic_pairs.json` (when periodic metadata is enabled)

`solve-ferrite` and `build-and-solve` additionally write:

- `ferrite_homogenization_summary.json`
- `ferrite_homogenization_stdout.txt`
- `homogenized_stiffness.csv`
- `engineering_constants.csv` — `ex, ey, gxy, nuxy, nuyx` for 2D;
  `ex, ey, ez, gyz, gxz, gxy, nuxy, nuyx, nuxz, nuzx, nuyz, nuzy` for 3D
- `stress_strain_response.csv`
- `traction_response.csv`
- `ferrite_homogenization.vtu` (when `solver.write_vtk: true`)

`batch-study` additionally writes `study_summary.csv`.

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
| `solver`           | object   | Optional Ferrite.jl material + solve settings                |

YAML and JSON are both supported. See [`docs/configuration.md`](docs/configuration.md)
for every field, default, and example value.

---

## Package layout

```
src/rve2d/
├── cli.py                  # `rve2d` entry point
├── workflow.py             # build_rve / build_and_solve_rve
├── study.py                # batch-study driver
├── config.py               # dataclass schema + YAML/JSON loader
├── models.py               # Domain / Fibre / GeometryModel dataclasses
├── exceptions.py
├── synthetic_generation/
│   ├── circular.py         # 2D circular fibres
│   ├── cylindrical.py      # 3D cylindrical fibres
│   └── sem_image.py        # SEM → fitted circles/ellipses
├── image_import/
│   ├── mask_to_geometry.py     # 2D mask → polygons
│   └── mask_to_geometry_3d.py  # 2D mask → extruded volumes
├── geometry_cleanup/cleanup.py
├── meshing/gmsh_builder.py
├── export/writers.py
├── validation/checks.py
└── ferrite_bridge.py       # subprocess driver + constitutive matrices

julia/
├── ferrite_homogenization.jl       # 2D triangle solver
└── ferrite_homogenization_3d.jl    # 3D tetrahedral solver
```

---

## Homogenization-ready metadata

When `periodic_compatible: true`, the exported metadata includes:

- left/right and bottom/top boundary pairs in 2D
- left/right, front/back, and bottom/top pairs in 3D
- translation vectors and domain size
- boundary physical tag ids

This is solver-agnostic metadata; downstream solvers can use it to construct
periodic constraints. The Ferrite path consumes it automatically.

---

## Ferrite.jl solve stage

- 2D triangle and 3D tetrahedral constant-strain formulations
- Material assignment from gmsh physical phase ids
- Strong periodic constraints on a node-matched periodic mesh
- VTK visualization output (`.vtu`) for ParaView
- 2D phases: isotropic or orthotropic (orthotropic only for `plane_stress`)
- 3D phases: isotropic or orthotropic, with `material_angle_x_deg`,
  `material_angle_y_deg`, `material_angle_z_deg`
- 3D workflows can also propagate phase rotations through geometry metadata
  (`phase_orientation_rotations_deg`)
- Image-import solves accept explicit `matrix_material_angle_deg` and
  `fibre_material_angle_deg`
- Summaries include derived engineering constants and full `material_rotations_deg`
- 3D solves require `kinematics: solid`

**Rotation precedence** (highest to lowest):

1. Explicit solver rotation inputs
2. Geometry metadata `phase_orientation_rotations_deg`
3. Legacy fallback (e.g. 2D `orientation_deg`)

---

## Development

```bash
pytest                  # run the test suite
ruff check .            # lint
mypy src/rve2d          # type-check (strict)
```

CI runs lint + tests on Python 3.11 / 3.12 / 3.13 — see
[`.github/workflows/ci.yml`](.github/workflows/ci.yml).

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

MIT — see [LICENSE](LICENSE).
