# Configuration reference

Configs are YAML or JSON. Top-level keys:

```yaml
mode: synthetic | image | sem_to_synthetic
dimension: 2 | 3            # default: 2
synthetic: { ... }          # required if mode == synthetic
image: { ... }              # required if mode == image
sem_to_synthetic: { ... }   # required if mode == sem_to_synthetic (2D only)
mesh: { ... }
export: { ... }
solver: { ... }             # optional; linear homogenization when enabled: true
nonlinear: { ... }          # optional; plasticity + cohesive interfaces when enabled: true
laminate: { ... }           # optional; laminate pipeline (rve2d laminate) when enabled: true
```

Unknown keys in any section are rejected with a `ConfigError` naming the key.
Numeric fields also accept strings such as `1.0e8` (PyYAML reads exponents
without a sign as strings).

---

## `synthetic`

Randomly placed circular (2D) or cylindrical (3D, along z) fibres of equal radius in a
rectangular / box domain.

| Field                               | Type     | Default  | Notes                                                            |
| ----------------------------------- | -------- | -------- | ---------------------------------------------------------------- |
| `domain_width`                      | float    | required | Domain extent in x                                               |
| `domain_height`                     | float    | required | Domain extent in y                                               |
| `domain_depth`                      | float    | `None`   | Required for 3D                                                  |
| `fibre_radius`                      | float    | required | Fibre cross-section radius                                       |
| `target_volume_fraction`            | float    | required | In `(0, 1)`. Used when `fibre_count` is unset                    |
| `fibre_count`                       | int?     | `None`   | Overrides `target_volume_fraction` when set                      |
| `min_spacing`                       | float    | `0.0`    | Minimum gap between fibre surfaces                               |
| `edge_clearance`                    | float    | `0.0`    | Minimum gap between fibre and domain edge (ignored with `periodic_wrapping`) |
| `random_seed`                       | int      | `0`      | RNG seed for reproducibility                                     |
| `max_attempts`                      | int      | `100000` | Random-placement attempt cap                                     |
| `periodic_compatible`               | bool     | `false`  | Emit periodic boundary metadata + node-matched mesh              |
| `packing_algorithm`                 | enum     | `random_sequential` | `random_sequential` (one fibre at a time, reproduces earlier layouts; saturates near Vf 0.45-0.5) or `relaxation` (push overlapping fibres apart, then Monte Carlo shaking; Vf 0.60-0.65 and above; needs `min_spacing > 0`) |
| `periodic_wrapping`                 | bool     | `false`  | Fibres may cross the domain boundary and are stored with their periodic images; needs `periodic_compatible: true` |
| `boundary_clearance`                | float?   | `None`   | Minimum distance between a fibre surface and a face/corner it does not cross (or the minimum crossing depth); default `max(min_spacing, 0.1 r)` for the new modes; about one mesh element avoids slivers |
| `max_relaxation_iterations`         | int      | `20000`  | Iteration cap of the relaxation packing                          |
| `shake_sweeps`                      | int      | `20`     | Monte Carlo sweeps after relaxation (0 disables them)            |
| `orientation_deg` *(2D)*            | float    | `0.0`    | Fibre material angle about z, recorded in the geometry metadata; the solve uses it when no fibre angle is set in `solver` |
| `matrix_orientation_angle_*_deg`    | float?   | `None`   | 3D matrix rotation, passed to the solve via the geometry metadata |
| `fibre_orientation_angle_*_deg`     | float?   | `None`   | 3D fibre rotation, passed to the solve via the geometry metadata  |

For fibre volume fractions above about 0.45, use `packing_algorithm: relaxation` (and, for
periodic RVEs, `periodic_wrapping: true`, which removes the matrix-only band along the
edges); see `examples/{2d,3d}/synthetic/high_vf_periodic.yaml`. Fibre positions do not
depend on the mesh settings; when `min_spacing` or `boundary_clearance` is smaller than
`mesh.element_size_min`, the quality report notes that the mesh will contain slivers.

---

## `image`

Segmented image (fibres light, or dark with `invert: true`) → 2D polygons or 3D extruded
volumes.

| Field                  | Type     | Default | Notes                                                           |
| ---------------------- | -------- | ------- | --------------------------------------------------------------- |
| `image_path`           | string   | req.    | Any image `skimage.io.imread` reads: 8/16-bit grey, boolean, float in [0, 1], RGB(A) |
| `pixel_size`           | float    | `1.0`   | Physical edge length of one pixel                               |
| `extrusion_depth`      | float    | `None`  | Required for 3D                                                 |
| `threshold`            | float    | `0.5`   | Foreground = grey level above this, on a 0-1 scale whatever the image format |
| `invert`               | bool     | `false` | Treat dark pixels as foreground when true                       |
| `min_artifact_area_px` | int      | `16`    | Drop connected components smaller than this (reported as `removed_artifacts`) |
| `clear_border`         | bool     | `false` | Remove regions touching the image border                        |
| `simplify_tolerance`   | float    | `1.0`   | Polygon simplification tolerance, in pixels (vertices on the domain boundary are kept) |
| `domain_width`         | float?   | `None`  | Domain width; default the image width. Smaller crops the image   |
| `domain_height`        | float?   | `None`  | Domain height; default the image height. Smaller crops the image |
| `periodic_compatible`  | bool     | `false` | Emit periodic boundary metadata                                 |
| `matrix/fibre_orientation_angle_*_deg` | float? | `None` | Per-phase rotations passed to the solve via the geometry metadata |

Images are read as grey levels in [0, 1] (colour images by luminance, alpha ignored;
integer images with only 0 and 1 are binary masks). Pixel `(row, col)` covers
`[col s, (col+1) s] x [(rows-row-1) s, (rows-row) s]` with `s = pixel_size`: x to the right,
y up, origin at the bottom-left corner. Fibres cut by the image border end exactly on the
domain boundary, and fibres outside a cropped domain are dropped.

---

## `sem_to_synthetic` (2D only)

Raw SEM micrograph → segment + watershed-split + circle/ellipse fit → clean
synthetic-style RVE.

Key fields (see [`src/rve2d/config.py`](../src/rve2d/config.py) for the full
list and validation rules):

- `image_path`, `pixel_size`
- `smoothing_sigma`, `threshold`, `invert`, `clear_border`
- `separate_touching_fibres`, `separation_min_distance_px`,
  `separation_peak_threshold_px`
- `min_region_area_px`, `min_artifact_area_px`
- `ellipse_axis_ratio_threshold`, `ellipse_sample_points`
- `max_candidate_overlap_fraction`, `drop_boundary_fibres`
- `domain_width`, `domain_height`, `periodic_compatible`

---

## `mesh`

| Field              | Type   | Default |
| ------------------ | ------ | ------- |
| `element_size_min` | float  | `0.01`  |
| `element_size_max` | float  | `0.05`  |
| `algorithm`        | int    | `6`     |
| `mesh_order`       | int    | `1`     |
| `recombine`        | bool   | `false` |
| `verbosity`        | int    | `2`     |
| `elements_per_circle` | int | `0`     |

`elements_per_circle > 0` refines the mesh along curved fibre boundaries
(gmsh `Mesh.MeshSizeFromCurvature`), still bounded by `element_size_min`/`max`.
Useful for cohesive interfaces, which need a well-resolved fibre perimeter, and
whenever the meshed fibre volume fraction should match the nominal one: circles are
meshed as polygons, so a coarse mesh loses fibre area. The linear homogenization reports
the meshed fraction (`fibre_volume_fraction` in `homogenization_summary.json`).

---

## `export`

| Field                 | Type      | Default               |
| --------------------- | --------- | --------------------- |
| `output_dir`          | string    | `outputs`             |
| `basename`            | string    | `rve2d`               |
| `formats`             | string[]  | `["msh", "xdmf"]`     |
| `write_phase_json`    | bool      | `true`                |
| `write_quality_json`  | bool      | `true`                |
| `write_periodic_json` | bool      | `true`                |

---

## `solver` (linear homogenization)

Runs when `enabled: true`: `build-and-solve` and `batch-study` then include it, and
`solve` / `solve-ferrite` require it. Materials default to a polymer matrix (3.5 GPa,
0.35) and glass fibres (70 GPa, 0.2) in SI units; override per phase. See
[homogenization.md](homogenization.md) for the formulation.

| Field                                      | Type    | Default        | Notes                                                  |
| ------------------------------------------ | ------- | -------------- | ------------------------------------------------------ |
| `enabled`                                  | bool    | `false`        | Master switch                                          |
| `engine`                                   | enum    | `tensormesh`   | `tensormesh` (NumPy/SciPy) or `julia` (Ferrite.jl, needs Julia 1.11+); `--engine` overrides it; `python` is an alias of `tensormesh` |
| `boundary_condition`                       | enum    | `dirichlet`    | `dirichlet` (zero boundary fluctuation, upper bound) or `periodic` |
| `kinematics`                               | enum    | `plane_strain` | 2D: `plane_stress`, `plane_strain`, `generalized_plane_strain`; 3D: `solid` |
| `matrix_material_model` / `fibre_material_model` | enum | `isotropic` | `isotropic` or `orthotropic`                          |
| `matrix_youngs_modulus` / `matrix_poisson_ratio` | float | 3.5e9 / 0.35 | Isotropic matrix                                     |
| `fibre_youngs_modulus`  / `fibre_poisson_ratio`  | float | 70e9  / 0.20 | Isotropic fibre                                      |
| `matrix_e1, e2, e3, g12, g13, g23, nu12, nu13, nu23` | float? | `None` | Orthotropic matrix: `e1, e2, g12, nu12` for plane stress, all nine otherwise |
| `fibre_e1, e2, e3, g12, g13, g23, nu12, nu13, nu23`  | float? | `None` | Orthotropic fibre: same rule                          |
| `matrix_material_angle_deg` / `fibre_material_angle_deg`         | float? | `None` | Rotation about z                     |
| `matrix_material_angle_x/y/z_deg` / `fibre_material_angle_x/y/z_deg` | float? | `None` | Rotation `Rz Ry Rx`; cannot be mixed with `material_angle_deg`; plane stress allows z only |
| `matrix_cellset`, `fibre_cellset`          | string  | `matrix`/`fibre` | Phase names used in the summary                     |
| `matrix_phase_id`, `fibre_phase_id`        | int     | `1`/`2`        | gmsh physical ids                                     |
| `write_vtk`                                | bool    | `true`         | Write `homogenization.vtu` for ParaView               |

Which kinematics to use:

- `plane_stress`: a thin lamina loaded in its plane.
- `plane_strain`: a long body with `eps_zz = 0`; the constants are plane-strain moduli
  and are reported as `ex_plane_strain`, ... .
- `generalized_plane_strain`: the cross-section of a UD ply with fibres along z; gives the
  full 6×6 stiffness and the true engineering constants (longitudinal and transverse)
  from a 2D mesh. For orthotropic fibres with axis 1 along the fibre, set
  `fibre_material_angle_y_deg: -90`.
- `solid`: 3D meshes.

### Validation rules (enforced by `load_config`)

- `engine` must be `tensormesh` (or its alias `python`) or `julia`.
- Periodic solves require `periodic_compatible: true` in the section of the active
  `mode`.
- 2D solves use `plane_stress`, `plane_strain` or `generalized_plane_strain`; 3D solves
  require `solid`.
- Orthotropic phases need the constants listed above, positive moduli and a
  positive-definite compliance (e.g. `nu12^2 < e1/e2`).
- Mixing `material_angle_deg` with `material_angle_x/y/z_deg` is rejected; plane stress
  rejects x/y rotations.
- The mesh must be linear (`mesh_order: 1`, no `recombine`).
- Moduli must be positive and Poisson ratios in (-1, 0.5).

---

## `nonlinear` (plasticity + cohesive interfaces)

Runs only when `enabled: true`. The TensorMesh engine needs `pip install -e ".[nonlinear]"`;
the Julia engine needs Julia 1.11+ (see [nonlinear.md](nonlinear.md#julia-engine)). Full
description, units and outputs in [nonlinear.md](nonlinear.md).

| Field                | Type   | Default            | Notes                                                        |
| -------------------- | ------ | ------------------ | ------------------------------------------------------------ |
| `enabled`            | bool   | `false`            | Master switch                                                |
| `engine`             | enum   | `tensormesh`       | `tensormesh` (PyTorch) or `julia` (Ferrite.jl + DiffCohesive.jl, CPU); `--engine` overrides it; `python` is an alias of `tensormesh` |
| `kinematics`         | enum   | GPS (2D) / `solid` (3D) | 2D: `plane_strain`, `generalized_plane_strain`; 3D: `solid` |
| `boundary_condition` | enum   | `periodic`         | `periodic` (needs `periodic_compatible: true`) or `dirichlet` |
| `matrix`, `fibre`    | object | required           | `youngs_modulus`, `poisson_ratio`, optional `yield_stress` (omit: elastic), `hardening_modulus` (default 0) and the pressure-dependent plasticity and damage keys below |
| `interface`          | object | none               | Cohesive fibre/matrix interfaces; omit or `enabled: false` for perfect bonding |
| `load`               | object | see below          | Macro load path                                              |
| `device`             | enum   | `cpu`              | `cpu` or `cuda` (TensorMesh engine)                          |
| `linear_solver`      | enum   | `auto`             | `auto` (SciPy on CPU, TensorMesh on CUDA), `scipy`, `tensormesh` (TensorMesh engine; the Julia engine uses UMFPACK) |
| `newton_max_iterations` / `newton_tolerance` | int / float | `25` / `1e-8` | Relative residual tolerance                     |
| `max_step_cuts`      | int    | `10`               | Max halvings of an increment before the solve stops          |
| `output_every`       | int    | `0`                | Write VTU fields every N steps (0: final state only)         |
| `matrix_phase_id`, `fibre_phase_id` | int | `1` / `2` | gmsh physical ids                                         |

Optional keys of `matrix` and `fibre` (pressure-dependent plasticity and damage, see
[nonlinear.md](nonlinear.md#matrix-plasticity-and-damage)):

| Field                      | Default        | Notes |
| -------------------------- | -------------- | ----- |
| `compressive_yield_stress` | `yield_stress` | Yield stress in uniaxial compression, at least `yield_stress` (paraboloidal criterion when larger) |
| `plastic_poisson_ratio`    | `0.5`          | Lateral-to-axial plastic strain ratio in [0, 0.5]; 0.5 is isochoric flow (J2 with equal yield stresses); a larger compressive yield stress needs a value below 0.5, e.g. 0.3 |
| `saturation_stress`        | none           | Voce hardening: the tensile yield stress levels off at this value (above `yield_stress`; `hardening_modulus` is the initial slope); none: linear hardening |
| `damage_onset_strain`      | none           | Equivalent plastic strain at damage onset; needs `fracture_energy` |
| `fracture_energy`          | none           | `G_f` per unit crack area [stress x length], regularised with the element size; needs `damage_onset_strain` |

`interface`:

| Field                   | Type   | Default               | Notes                                                  |
| ----------------------- | ------ | --------------------- | ------------------------------------------------------ |
| `law`                   | enum   | `bilinear_mixed_mode` | or `bilinear`, `linear-parabolic`, `exponential`, `trapezoidal` (mode I shape laws) |
| `penalty_stiffness`     | float  | required              | `K`, stress / length                                   |
| `normal_strength`       | float  | required              | `T_n`                                                  |
| `shear_strength`        | float  | `normal_strength`     | `T_s`                                                  |
| `mode_i_toughness`      | float  | required              | `G_Ic`, stress x length                                |
| `mode_ii_toughness`     | float  | `mode_i_toughness`    | `G_IIc`                                                |
| `bk_exponent`           | float  | `1.45`                | Benzeggagh–Kenane exponent                             |
| `mixed_mode_criterion`  | enum   | `bk`                  | `bk` or `power`                                        |
| `viscosity`             | float  | `0.0`                 | Duvaut–Lions relaxation time (fraction of the load path) |
| `shear_penalty_stiffness` | float | `penalty_stiffness`  | Separate shear stiffness                               |
| `integration`           | enum   | `nodal`               | `nodal` (Newton–Cotes) or `gauss`                      |

`load`: `type` (`uniaxial_stress` or `uniaxial_strain`), `component`
(`xx`, `yy`, `zz`, `yz`, `xz`, `xy`; must be active for the kinematics),
`max_strain` (default `0.02`), `steps` (default `40`), `unload` (default `false`).

Validation also checks that each toughness exceeds the elastic energy at damage
onset (`G_c > T^2 / 2K`), that periodic solves use a periodic-compatible
geometry, and that the mesh is linear (`mesh_order: 1`, no `recombine`).

---

## `laminate` (laminate pipeline)

Used by `rve2d laminate`: ply properties from the RVE (`solver` section, and `nonlinear`
for the ply curves and strengths), then the stiffness and, optionally, coupon tests of each
stacking sequence with one or more ply damage models. Description, notation, models and
outputs in [laminate.md](laminate.md).

| Field                     | Type    | Default        | Notes |
| ------------------------- | ------- | -------------- | ----- |
| `enabled`                 | bool    | `false`        | Master switch (`rve2d laminate` needs it) |
| `ply_thickness`           | float   | `0.125`        | Thickness of every ply |
| `stacking_sequences`      | list    | `["[0/90]s"]`  | Strings such as `"[0/±45/90]2s"` or lists of angles (degrees, bottom to top) |
| `transversely_isotropic`  | bool    | `true`         | Average the ply stiffness about the fibre axis |
| `damage_models`           | list    | `[rve_curves]` | Any of `rve_curves`, `max_stress`, `hashin`, `continuum_damage` |
| `strengths`               | object  | none           | `longitudinal_tension`, `longitudinal_compression` (Xt, Xc: inputs, needed by the strength-based models), `transverse_tension`, `transverse_compression`, `in_plane_shear` (Yt, Yc, S12: default to the peaks of the RVE curves), `transverse_shear` (S23, default Yc / 2) |
| `fracture_energies`       | object  | none           | `fibre_tension`, `fibre_compression`, `matrix_tension`, `matrix_compression` (energy per unit crack area); needed by `continuum_damage` |
| `characteristic_length`   | float   | `1.0`          | Crack-band width of `continuum_damage` |
| `ply_curves`              | bool    | `true`         | Nonlinear RVE solves for the ply curves; `false`: no curves (give Yt, Yc, S12) |
| `transverse_max_strain`   | float   | `0.03`         | Strain range of the transverse curves (tension and compression) |
| `shear_max_strain`        | float   | `0.06`         | Strain range of the shear curve (engineering) |
| `curve_steps`             | int     | `60`           | Load steps of each curve |
| `coupon_tests`            | mapping | none           | Named tests (letters, digits, `_`, `-`); none: stiffness only |

Each entry of `coupon_tests`: `direction` (`x`, `y` or `xy`, default `x`), `max_strain`
(default `0.02`, negative for compression), `steps` (default `200`), `gauge_length`
(optional: elongation = strain × gauge length) and `width` (optional: force = stress ×
laminate thickness × width). Every test runs with every damage model.

Validation requires `solver.enabled: true` with `kinematics: generalized_plane_strain`
(2D) or `solid` (3D), readable stacking sequences, a positive thickness, known damage
models, positive strengths and fracture energies, Xt and Xc for the strength-based models
(and Yt, Yc, S12 when `ply_curves` is false), fracture energies for `continuum_damage`,
valid test settings, and `nonlinear.enabled: true` when a curve is needed (the curves of
`rve_curves`, and those whose peak gives a strength the config does not set).
