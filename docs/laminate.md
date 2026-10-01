# Laminates: stacking sequences, stiffness and tensile test

`rve2d laminate CONFIG` takes the RVE of a unidirectional ply through to laminates:

1. **RVE**: geometry and mesh from the config (2D or 3D, any input mode).
2. **Ply stiffness**: linear homogenization of the RVE (`solver` section).
3. **Ply curves** (only for the tensile test): nonlinear RVE solves (`nonlinear` section)
   under uniaxial stress, transverse and in-plane shear.
4. **Laminates**: for every stacking sequence, the ABD matrix, membrane and flexural
   engineering constants and the 3D effective stiffness and, optionally, a tensile test:
   stress–strain curve, damage events, elongation and force.

Steps 2 and 3 run on the engine of the config (`engine: tensormesh` or `julia`, or
`--engine`); step 4 is engine-independent and takes seconds. Laminates are computed from
the saved ply, so other stacking sequences do not need the RVE again:

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_tensile_2d.yaml
# edit stacking_sequences (or tensile_test), then:
rve2d laminate examples/pipelines/tensormesh/laminate_tensile_2d.yaml \
    --ply outputs/pipelines/tensormesh/laminate_tensile_2d/ply/ply_properties.json
```

With `--ply` the strengths in the config replace those stored with the ply.

## Ready-made pipelines

[`examples/pipelines/`](../examples/pipelines/) has the same four pipelines for each engine:

| Config                     | What it computes                                         |
| -------------------------- | -------------------------------------------------------- |
| `laminate_stiffness_2d.yaml` | 2D RVE → ply stiffness → stiffness of 7 laminates      |
| `laminate_stiffness_3d.yaml` | the same from a 3D RVE                                  |
| `laminate_tensile_2d.yaml`   | 2D RVE → ply stiffness and curves → stiffness and tensile test of 5 laminates |
| `laminate_tensile_3d.yaml`   | the same from a 3D RVE                                  |

Use the `tensormesh/` or the `julia/` copy; they differ only in `engine` and the output
folder, and give the same results. Measured run times are in [Performance](#performance).

## Configuration

```yaml
solver:                     # ply stiffness: generalized_plane_strain (2D) or solid (3D)
  enabled: true
  kinematics: generalized_plane_strain
  boundary_condition: periodic
  ...
nonlinear:                  # ply curves (only needed for the tensile test)
  enabled: true
  matrix: {...}
  fibre: {...}
  interface: {..., viscosity: 1.0e-4}   # a small viscosity carries the curves past debonding
laminate:
  enabled: true
  ply_thickness: 0.125
  stacking_sequences: ["[0]8", "[0/90]2s", "[±45]2s", "[0/±45/90]s"]
  longitudinal_tensile_strength: 750.0
  longitudinal_compressive_strength: 500.0
  transverse_max_strain: 0.03
  shear_max_strain: 0.08
  curve_steps: 60
  tensile_test:
    direction: x
    max_strain: 0.03
    steps: 300
    gauge_length: 150.0
    width: 25.0
```

| Field                                | Default     | Notes |
| ------------------------------------ | ----------- | ----- |
| `enabled`                            | `false`     | `rve2d laminate` needs `true` |
| `ply_thickness`                      | `0.125`     | thickness of every ply (length unit of the config) |
| `stacking_sequences`                 | `["[0/90]s"]` | laminates to analyse; notation below |
| `transversely_isotropic`             | `true`      | average the RVE stiffness about the fibre axis (removes the in-plane anisotropy of a finite random RVE) |
| `longitudinal_tensile_strength`      | none        | fibre-direction strengths (inputs: fibre failure is not modelled in the RVE); without them the fibres never fail |
| `longitudinal_compressive_strength`  | none        | |
| `ply_curves`                         | `true`      | run the nonlinear RVE solves for the tensile test; `false` gives linear plies with fibre failure only |
| `transverse_compression_curve`       | `false`     | also solve transverse compression (otherwise linear in compression) |
| `transverse_max_strain`              | `0.03`      | strain range of the transverse curve |
| `shear_max_strain`                   | `0.06`      | strain range of the shear curve (engineering) |
| `curve_steps`                        | `60`        | load steps of each curve |
| `tensile_test.enabled`               | `true`      | `false` for stiffness only |
| `tensile_test.direction`             | `x`         | `x`, `y` or `xy` (in-plane shear) |
| `tensile_test.max_strain`            | `0.02`      | negative for compression |
| `tensile_test.steps`                 | `200`       | |
| `tensile_test.gauge_length`          | none        | gives the elongation column (strain × gauge length) |
| `tensile_test.width`                 | none        | gives the force column (stress × laminate thickness × width) |

Units follow the rest of the config (e.g. mm and MPa, giving forces in N). The `solver`
section must use `generalized_plane_strain` (2D) or `solid` (3D), the kinematics that give
the full 6×6 ply stiffness, and the `nonlinear` section is required when the tensile test
uses ply curves. The `load` of the `nonlinear` section is replaced by the curve loads.

### Stacking-sequence notation

Angles in degrees from the laminate x axis, listed from the bottom ply to the top ply.

| Notation          | Plies                                  |
| ----------------- | -------------------------------------- |
| `[0/90/90/0]`, `[0/90/90/0]T` | as listed (`T`: total)   |
| `[0/90]s`         | symmetric: `0/90/90/0`                 |
| `[0/90]2s`        | repeated, then mirrored: `0/90/0/90/90/0/90/0` |
| `[0/90]3`         | repeated: `0/90/0/90/0/90`             |
| `[0/±45/90]s`     | `±45` = `45/-45`, `∓45` = `-45/45` (also `+-`, `-+`) |
| `[0_2/90]s`, `[0₂/90]s` | a subscript repeats one ply: `0/0/90/90/0/0` |
| `[(0/90)_2/45]`, `[(±45)2/0]` | a group repeated: `0/90/0/90/45` |
| `[0/90/45]sb`, `[0/90/45]s̄` | odd symmetric (the last ply is the middle one): `0/90/45/90/0` |
| `[0, 90, -45]`, `0, 90, -45`, a YAML list | explicit angles |

Each laminate gets a folder named after its sequence (`[0/±45/90]2s` → `0_pm45_90-2s`).

## Ply properties from the RVE

The RVE has its fibres along z. The ply axes are 1 = fibre (RVE z), 2 = transverse in the
ply plane (RVE x) and 3 = through the ply thickness (RVE y), so the ply stiffness is the RVE
stiffness with its Voigt rows and columns reordered. A finite random RVE is slightly
anisotropic in its cross-section; `transversely_isotropic: true` averages the stiffness over
rotations about the fibre axis, which gives the transversely isotropic ply of an infinite
random microstructure (`C22 = C33`, `C44 = (C22 − C23)/2`, `C55 = C66`). The ply
engineering constants are in `ply/ply_properties.json` and `pipeline_summary.json`.

The ply curves are the macro stress–strain curves of nonlinear RVE solves under uniaxial
stress (the other macro stresses zero):

- **transverse** (ply σ22–ε22): RVE `xx`, on the RVE mesh (generalized plane strain in 2D,
  solid in 3D); optionally also in compression;
- **in-plane shear** (ply τ12–γ12): RVE `xz`, the shear of planes along the fibres. A 2D
  mesh has no z displacement for it, so the solve runs on the 2D mesh **extruded into one
  periodic layer of tetrahedra** (`ply/shear/rve_layer.msh`). With periodic top and bottom
  faces, the fluctuations of a node and of the node above it are equal, so every
  tetrahedron carries the strain of its triangle: this is the exact z-invariant 3D problem
  of the cross-section, longitudinal shears included, at a few times the cost of the 2D
  solve (three tetrahedra per triangle, 1.5 times the unknowns) instead of the cost of a
  meshed 3D slab. The linear homogenization on such a layer equals the generalized plane
  strain result to round-off on both engines (`tests/test_homogenization.py`).

A curve ends where its RVE solve ends. When debonding makes the macro response snap back,
the strain-controlled solve can stop before the requested strain; the curve then ends at
the last converged step and is marked `completed: false` in `ply/ply_properties.json` and
`pipeline_summary.json`. A small damage viscosity in `nonlinear.interface` (`1e-4`, see
[nonlinear.md](nonlinear.md#example-runs)) usually carries the solve through, as in the
example pipelines.

## Laminate stiffness

Two homogenizations of the stack of perfectly bonded plies, both in `constants.json`
(laminate axes: x the reference direction, z through the thickness):

- **Classical lamination theory** (plies in plane stress, Kirchhoff plate kinematics). Each
  ply's plane-stress stiffness `Q̄` is the inverse of the in-plane block of its rotated 3D
  compliance, so it is consistent with the full ply stiffness. `A`, `B` and `D` integrate
  `Q̄` over the thickness (z from −h/2 at the bottom), `abd.csv` holds the 6×6 matrix with
  engineering shear. The membrane constants (`ex`, `ey`, `gxy`, `nuxy`, `nuyx`) come from
  the inverse of the ABD matrix, so the laminate is free to bend (unsymmetric laminates
  included), and the flexural constants (`flexural_ex = 12 / (h³ d*₁₁)`, ...) from its
  bending block. `eta_xy_x` is the shear strain per axial strain under `Nx` (zero for a
  balanced laminate); `symmetric` and `balanced` flag `B = 0` and `A16 = A26 = 0`.
- **3D effective stiffness** (`effective_3d_stiffness.csv`, `constants.json` →
  `effective_3d`): the exact 6×6 stiffness of the layered medium, with in-plane strains and
  out-of-plane stresses equal in every ply; it adds `ez`, `gxz`, `gyz`, `nuxz` and `nuyz`
  for 3D models of the laminate. Its in-plane constants equal the CLT membrane constants
  of a symmetric laminate, and an FE RVE of a two-ply stack reproduces it to round-off on
  both engines (`tests/test_laminate.py`).

## Tensile test

The coupon test of each laminate is strain controlled in `x`, `y` or `xy`; the other force
resultants and all moments are zero, so unsymmetric laminates curve and unbalanced ones
shear as in a real test. The kinematics are those of CLT (mid-plane strains and
curvatures); every ply is integrated with two points through its thickness, and each load
step is solved by Newton iterations with step cutting.

Each ply, in ply axes:

- **fibre direction**: linear elastic up to the longitudinal strength; when the fibres of a
  ply fail it carries no more load and the step is solved again (load redistribution);
- **transverse**: the RVE transverse curve, driven by the transverse mechanical strain
  (linear in compression unless a compression curve is given);
- **in-plane shear**: the RVE shear curve;
- beyond the largest strain reached so far, a curve is followed; below it the ply unloads
  along the secant to the origin (damage-like memory). Past the end of a curve the stress
  stays at its last value, so set the curve strains to cover the expected ply strains.

The transverse and shear responses are independent of each other (no interaction), which
is exact when each ply sees one mode, e.g. a `[90]` laminate in transverse tension, whose
curve is the RVE curve. With linear curves the test reproduces the CLT modulus exactly
(`tests/test_laminate_tensile.py`). The test stops at `max_strain`, or once the stress has
dropped below 5 % of its peak (all load-carrying plies failed).

Events are recorded when a ply first passes the peak of its transverse or shear curve
("transverse peak passed", "shear peak passed"), when its fibres fail ("fibre failure")
and when it is strained past the end of one of its curves ("transverse curve exceeded",
"shear curve exceeded": its stress is then held at the last value of the curve, so extend
`transverse_max_strain` or `shear_max_strain`). Per laminate the folder holds
`tensile_test.csv` (`strain`, `stress`, mid-plane strains, curvatures, failed plies,
`elongation`, `force`) and `tensile_test_summary.json` (initial modulus, peak stress and its
strain, first events); `laminates/summary.csv` collects the key numbers of every laminate
(including `curve_exceeded_strain`) and `laminates/tensile_tests.png` plots the curves, with
the elongation on the top axis when `gauge_length` is set.

## Checks

All of these run in the test suite:

| Check | Result |
|---|---|
| Stacking notation: `s`, `2s`, `T`, `±`/`∓`, subscripts, groups, `sb`, lists; malformed input rejected | `tests/test_laminate.py` |
| `[0]n` laminate: CLT and 3D constants equal the ply constants | to 1e-12 |
| Off-axis ply: `ex(θ)` against the transformation formula | to 1e-12 |
| Cross-ply `[0/90]s`: `A`, `B = 0` and `D` against hand calculation | to 1e-12 |
| Isotropic plies: every laminate is isotropic with the ply constants | to 1e-12 |
| 3D effective stiffness against the CLT membrane constants (symmetric laminates) | to 1e-10 |
| 3D effective stiffness of a two-ply stack against a periodic FE RVE of it, both engines | to 1e-9 |
| One-layer extruded mesh against generalized plane strain (linear, both engines) | to 1e-9 |
| Longitudinal shear on a one-layer extruded mesh: TensorMesh against Julia (nonlinear) | same iterations, stresses to 1e-9 |
| Tensile test with linear plies against the CLT moduli (x, y, xy; unbalanced, unsymmetric) | to 1e-8 |
| `[90]` laminate reproduces the transverse RVE curve; fibre failure at `Xt / E1` | to 1e-8 / one step |
| Pipeline end to end: ply reuse with `--ply`, engines give the same ply | to 1e-10 |

On the example pipelines the two engines give the same ply stiffness to 4e-16 and the same
laminate constants to 3e-15.

## Performance

Wall time of `rve2d laminate` on the example pipelines (4-core Xeon at 2.1 GHz, one run at a
time; Julia times include about 4 s of start-up per solve):

| Config | TensorMesh | Julia |
|---|---|---|
| `laminate_stiffness_2d.yaml` (linear solve with 8,676 unknowns, 7 laminates) | 1.6 s | 4.3 s |
| `laminate_stiffness_3d.yaml` (linear solve with 3,792 unknowns, 7 laminates) | 2.1 s | 4.9 s |
| `laminate_tensile_2d.yaml` | 284 s | 286 s |
| — transverse curve, 2D (6,507 unknowns, 108 increments, 609 Newton iterations) | 79 s | 73 s |
| — shear curve, extruded layer (9,761 unknowns, 86 increments, 436 iterations) | 189 s | 197 s |
| `laminate_tensile_3d.yaml` | 381 s | 269 s |
| — transverse curve (4,961 unknowns, 68 increments, 392 iterations) | 202 s | 138 s |
| — shear curve (61 increments, 315 iterations) | 162 s | 114 s |

The laminates themselves (five tensile tests of 300 steps and the plot) take about 10 s,
which is also the time of a rerun with `--ply`. Both engines take the same increments and
Newton iterations; the ply curves agree to 5e-13 and the laminate results to 4e-15.

Meshing the 2D example's shear problem as a 3D slab with gmsh instead (28,000 tetrahedra,
37,000 unknowns) had not finished its shear curve after 17 minutes, which is why the
pipeline uses the extruded layer.

## Outputs

```
OUTPUT/
├── rve/                         RVE mesh and metadata (as rve2d build)
├── ply/
│   ├── elastic/                 linear homogenization of the RVE
│   ├── transverse_tension/      nonlinear RVE solve, xx (tensile test only)
│   ├── shear/                   nonlinear RVE solve, xz (2D: on rve_layer.msh)
│   └── ply_properties.json      ply stiffness, curves, strengths (reuse with --ply)
├── laminates/
│   ├── summary.csv              one row per stacking sequence
│   ├── tensile_tests.png
│   └── <sequence>/              abd.csv, effective_3d_stiffness.csv, constants.json,
│                                tensile_test.csv, tensile_test_summary.json
└── pipeline_summary.json
```

## Limitations

- One ply material and thickness per laminate; perfectly bonded plies (no delamination).
- Fibre failure is an input strength; the ply curves come from the RVE (matrix plasticity
  and fibre/matrix debonding) without transverse–shear interaction in the laminate test.
- The tensile test is quasi-static and strain controlled up to the peak and the first
  load drops; post-peak softening of a whole laminate is mesh- and size-dependent in reality
  and is reported only until the load falls below 5 % of the peak.
- Laminate kinematics are those of thin plates (CLT); free-edge stresses are not modelled.
