# Laminates: stacking sequences, stiffness, damage and coupon tests

`rve2d laminate CONFIG` takes the RVE of a unidirectional ply through to laminates:

1. **RVE**: geometry and mesh from the config (2D or 3D, any input mode).
2. **Ply stiffness**: linear homogenization of the RVE (`solver` section).
3. **Ply curves and strengths** (for the coupon tests): nonlinear RVE solves (`nonlinear`
   section) under uniaxial stress: transverse tension, transverse compression and in-plane
   shear. Their peaks are the ply strengths `Yt`, `Yc` and `S12` unless the config gives
   them; the fibre-direction strengths `Xt` and `Xc` are inputs, since fibre failure is not
   part of the RVE model.
4. **Laminates**: for every stacking sequence, the ABD matrix, membrane and flexural
   engineering constants and the 3D effective stiffness; then every coupon test (tension,
   compression or shear, strain controlled) with every ply damage model: stress–strain
   curve, elongation, force, first ply failure, first fibre failure and peak stress.

Steps 2 and 3 run on the engine of the config (`engine: tensormesh` or `julia`, or
`--engine`); step 4 is engine-independent and takes seconds to a minute. Laminates are
computed from the saved ply, so other stacking sequences, damage models or tests do not need
the RVE again:

```bash
rve2d laminate examples/pipelines/tensormesh/laminate_damage_2d.yaml
# edit stacking_sequences, damage_models or coupon_tests, then:
rve2d laminate examples/pipelines/tensormesh/laminate_damage_2d.yaml \
    --ply outputs/pipelines/tensormesh/laminate_damage_2d/ply/ply_properties.json
```

With `--ply` the strengths and fracture energies in the config replace those stored with
the ply.

## Ready-made pipelines

[`examples/pipelines/`](../examples/pipelines/) has the same four pipelines for each engine:

| Config                       | What it computes                                         |
| ---------------------------- | -------------------------------------------------------- |
| `laminate_stiffness_2d.yaml` | 2D RVE → ply stiffness → stiffness of 7 laminates        |
| `laminate_stiffness_3d.yaml` | the same from a 3D RVE                                   |
| `laminate_damage_2d.yaml`    | E-glass/epoxy (WWFE-I data), 2D RVE at Vf 0.62 → ply stiffness, curves and strengths → stiffness, tension and compression tests of 6 laminates with the four damage models |
| `laminate_damage_3d.yaml`    | the same from a 3D RVE                                   |

Use the `tensormesh/` or the `julia/` copy; they differ only in `engine` and the output
folder, and give the same results. Measured run times are in [Performance](#performance).

## Configuration

```yaml
solver:                     # ply stiffness: generalized_plane_strain (2D) or solid (3D)
  enabled: true
  kinematics: generalized_plane_strain
  boundary_condition: periodic
  ...
nonlinear:                  # ply curves (only needed for the coupon tests)
  enabled: true
  matrix: {...}             # e.g. pressure-dependent epoxy with damage (nonlinear.md)
  fibre: {...}
  interface: {..., viscosity: 1.0e-4}   # a small viscosity carries the curves past debonding
laminate:
  enabled: true
  ply_thickness: 0.125
  stacking_sequences: ["[0]8", "[0/90]2s", "[±45]2s", "[0/±45/90]s"]
  damage_models: [rve_curves, max_stress, hashin, continuum_damage]
  strengths:                # MPa; Yt, Yc and S12 default to the peaks of the RVE curves
    longitudinal_tension: 1140.0
    longitudinal_compression: 570.0
  fracture_energies:        # N/mm, continuum_damage only
    fibre_tension: 40.0
    fibre_compression: 20.0
    matrix_tension: 0.3
    matrix_compression: 1.0
  characteristic_length: 0.5
  transverse_max_strain: 0.03
  shear_max_strain: 0.08
  curve_steps: 60
  coupon_tests:             # any names; each runs with every damage model
    tension:     {direction: x, max_strain: 0.03, steps: 300, gauge_length: 150.0, width: 25.0}
    compression: {direction: x, max_strain: -0.02, steps: 200}
```

| Field                     | Default       | Notes |
| ------------------------- | ------------- | ----- |
| `enabled`                 | `false`       | `rve2d laminate` needs `true` |
| `ply_thickness`           | `0.125`       | thickness of every ply (length unit of the config) |
| `stacking_sequences`      | `["[0/90]s"]` | laminates to analyse; notation below |
| `transversely_isotropic`  | `true`        | average the RVE stiffness about the fibre axis (removes the in-plane anisotropy of a finite random RVE) |
| `damage_models`           | `[rve_curves]` | any of `rve_curves`, `max_stress`, `hashin`, `continuum_damage` ([Ply damage models](#ply-damage-models)) |
| `strengths`               | none          | `longitudinal_tension` (Xt), `longitudinal_compression` (Xc), `transverse_tension` (Yt), `transverse_compression` (Yc), `in_plane_shear` (S12), `transverse_shear` (S23, Hashin's matrix compression; default Yc / 2). Positive values. Xt and Xc are required by the strength-based models; Yt, Yc and S12 default to the RVE curve peaks |
| `fracture_energies`       | none          | `fibre_tension`, `fibre_compression`, `matrix_tension`, `matrix_compression`: energy per unit crack area (N/mm with mm and MPa); required by `continuum_damage` |
| `characteristic_length`   | `1.0`         | crack-band width of `continuum_damage` |
| `ply_curves`              | `true`        | run the nonlinear RVE solves; `false`: no curves (give Yt, Yc and S12; `rve_curves` is then linear up to fibre failure) |
| `transverse_max_strain`   | `0.03`        | strain range of the transverse curves (tension and compression) |
| `shear_max_strain`        | `0.06`        | strain range of the shear curve (engineering) |
| `curve_steps`             | `60`          | load steps of each curve |
| `coupon_tests`            | none          | named tests; none: stiffness only |
| `coupon_tests.<name>.direction` | `x`     | `x`, `y` or `xy` (in-plane shear) |
| `coupon_tests.<name>.max_strain` | `0.02` | negative for compression |
| `coupon_tests.<name>.steps` | `200`       | |
| `coupon_tests.<name>.gauge_length` | none | gives the elongation column (strain × gauge length) |
| `coupon_tests.<name>.width` | none        | gives the force column (stress × laminate thickness × width) |

Units follow the rest of the config (e.g. mm and MPa, giving forces in N and fracture
energies in N/mm = kJ/m²). The `solver` section must use `generalized_plane_strain` (2D) or
`solid` (3D), the kinematics that give the full 6×6 ply stiffness. Only the curves the
coupon tests need are solved: all three for `rve_curves` (the compression curve only when a
test has a negative `max_strain`), and for the strength-based models the curves of the
strengths the config does not give; the `nonlinear` section is required when any curve is
needed. The `load` of the `nonlinear` section is replaced by the curve loads.

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

- **transverse tension and compression** (ply σ22–ε22): RVE `xx`, on the RVE mesh
  (generalized plane strain in 2D, solid in 3D);
- **in-plane shear** (ply τ12–γ12): RVE `xz`, the shear of planes along the fibres. A 2D
  mesh has no z displacement for it, so the solve runs on the 2D mesh **extruded into one
  periodic layer of tetrahedra** (`ply/shear/rve_layer.msh`). With periodic top and bottom
  faces, the fluctuations of a node and of the node above it are equal, so every
  tetrahedron carries the strain of its triangle: this is the exact z-invariant 3D problem
  of the cross-section, longitudinal shears included, at a few times the cost of the 2D
  solve (three tetrahedra per triangle, 1.5 times the unknowns) instead of the cost of a
  meshed 3D slab. The linear homogenization on such a layer equals the generalized plane
  strain result to round-off on both engines (`tests/test_homogenization.py`).

A curve ends where its RVE solve ends. When damage localises or fibres debond, the macro
response can snap back and the strain-controlled solve stops before the requested strain;
the curve then ends at the last converged step and is marked `completed: false` in
`ply/ply_properties.json` and `pipeline_summary.json`, which also lists a warning (with the
`rve_curves` model). The peak, and so the strength, is normally reached before that. A small
damage viscosity in `nonlinear.interface` (`1e-4`, see
[nonlinear.md](nonlinear.md#example-runs)) carries the solves past most debonding events.

### Strengths

| Strength | Symbol | Source |
|---|---|---|
| longitudinal tension, compression | Xt, Xc | input (`strengths.longitudinal_tension`, `longitudinal_compression`) |
| transverse tension | Yt | peak of the transverse tension curve, or input |
| transverse compression | Yc | peak of the transverse compression curve, or input |
| in-plane shear | S12 | peak of the shear curve, or input |
| transverse shear | S23 | input, default Yc / 2 (Hashin's matrix compression only) |

`pipeline_summary.json` lists the strengths used (`ply_strengths`) and where each comes
from (`ply_strength_sources`: `input` or `rve`). A peak at the very end of a curve (a curve
still rising) is not a strength: extend `transverse_max_strain` or `shear_max_strain`, or
give the strength.

## Ply damage models

Each model maps the strain at a point of a ply (ply axes 1, 2 and engineering shear 12,
plane stress) and its history to the ply stress. From the simplest to the most elaborate:

| Model | Failure initiation | After initiation | Needs |
|---|---|---|---|
| `max_stress` | a stress component reaches its strength | ply discount: fibre failure removes the ply, matrix or shear failure its E2 and G12 | Xt, Xc, Yt, Yc, S12 |
| `hashin` | Hashin's interactive criteria of four modes | each failed mode keeps a fraction of the stiffness | + S23 (optional) |
| `continuum_damage` | Hashin's criteria on the effective stress | one damage variable per mode, linear softening that dissipates the mode's fracture energy | + fracture energies, characteristic length |
| `rve_curves` | the RVE curves (their peaks) | the RVE curves with secant unloading; fibre failure at Xt / Xc | the curves, Xt, Xc |

The first three models share the damaged plane-stress stiffness of Matzenmiller, Lubliner
and Taylor (1995):

    Q11 = (1 − d1) E1 / D,   Q22 = (1 − d2) E2 / D,   Q12 = (1 − d1)(1 − d2) ν12 E2 / D,
    Q66 = (1 − d6) G12,       D = 1 − (1 − d1)(1 − d2) ν12 ν21.

**Maximum stress** (`max_stress`). A mode fails when `σ1 ≥ Xt` (fibre tension), `−σ1 ≥ Xc`
(fibre compression), `σ2 ≥ Yt` (matrix tension), `−σ2 ≥ Yc` (matrix compression) or
`|τ12| ≥ S12` (shear). Fibre failure removes the ply (`d1 = d2 = d6 = 1`); matrix or shear
failure removes its transverse and shear stiffness (`d2 = d6 = 1`), the classical ply
discount. No interaction between the stresses.

**Hashin** (`hashin`, Hashin 1980). Four modes, chosen by the signs of σ1 and σ2:

| Mode | Criterion (failure at ≥ 1) |
|---|---|
| fibre tension, σ1 ≥ 0 | (σ1/Xt)² + α (τ12/S12)² |
| fibre compression, σ1 < 0 | (σ1/Xc)² |
| matrix tension, σ2 ≥ 0 | (σ2/Yt)² + (τ12/S12)² |
| matrix compression, σ2 < 0 | (σ2/2S23)² + ((Yc/2S23)² − 1) σ2/Yc + (τ12/S12)² |

with α = 0 (the Hashin–Rotem form, which does not predict fibre failure from shear alone).
A failed mode keeps a fraction of the ply moduli (progressive degradation after Tserpes et
al. 2001; fractions multiply when several modes fail):

| Failed mode | E1 | E2 | G12 |
|---|---|---|---|
| fibre tension | 0.07 | 0.07 | 0.07 |
| fibre compression | 0.14 | 0.14 | 0.14 |
| matrix tension | 1 | 0.2 | 0.2 |
| matrix compression | 1 | 0.4 | 0.4 |

**Continuum damage** (`continuum_damage`, after Lapczyk & Hurtado 2007 and Maimí et al.
2007). Each mode `m` has a damage variable `d_m`. With `F_m` its Hashin index of the
effective (undamaged) stress `Q0 ε`, `r = sqrt(F_m)` grows with the load; damage starts at
`r = 1`, where the mode's work density is `w0` (σ̃·ε of its components), and then follows

    d_m = r_f (r − 1) / (r (r_f − 1)),     r_f = 2 G_m / (L w0),

with `r` the largest value reached (no healing), `G_m` the mode's fracture energy and `L`
the `characteristic_length`. In uniaxial loading this is linear softening from the strength
at `ε0 = X/E` to zero stress at `ε_f = 2 G_m / (L X)`: a band of width L dissipates G_m per
unit area. `d1` is the fibre tension or compression damage (sign of σ1), `d2` the matrix
tension or compression damage (sign of σ2) and `d6 = 1 − Π (1 − d_m)`; damage is capped at
0.999. When `L > 2 G_m E / X²` the softening would snap back: it is then instantaneous and
the test reports a warning. The coupon test is one material point, so `L` stands for the
size of the region in which damage localises (the element size of a finite element model);
the strengths do not depend on it, the post-peak curves do.

**RVE curves** (`rve_curves`). The ply responses are the RVE curves themselves:

- **fibre direction**: linear elastic up to Xt / Xc; a ply whose fibres fail carries no more
  load;
- **transverse**: the RVE transverse tension or compression curve, driven by the transverse
  strain caused by the transverse stress (linear in compression when no compression curve
  was solved);
- **in-plane shear**: the RVE shear curve;
- beyond the largest strain reached so far, a curve is followed; below it the ply unloads
  along the secant to the origin (damage-like memory). Past the end of a curve the stress
  stays at its last value.

The transverse and shear responses are independent of each other (no interaction), which
is exact when each ply sees one mode, e.g. a `[90]` laminate in transverse tension, whose
curve is the RVE curve.

Choosing a model: `max_stress` and `hashin` give first ply failure and a simple picture of
the load redistribution after it (ply discount, residual stiffness); `continuum_damage`
adds energy-consistent progressive failure (its post-peak part depends on L); `rve_curves`
carries the micromechanics into the laminate (matrix plasticity, debonding, the nonlinear
shear response of the RVE) but has no interaction between modes. Running several and
comparing them is the point of `damage_models` being a list.

## Coupon tests

Each test is strain controlled in `x`, `y` or `xy` (a negative `max_strain` is a
compression test); the other force resultants and all moments are zero, so unsymmetric
laminates curve and unbalanced ones shear as in a real test. The kinematics are those of
classical lamination theory (mid-plane strains and curvatures); every ply is integrated with
two Gauss points through its thickness (exact for the elastic ABD response), and each load
step is solved by Newton iterations with a line search and step cutting. A mode that fails
suddenly (maximum stress, Hashin, fibre failure in `rve_curves`) is located by cutting the
step until it is 2 % of the nominal one, so the strengths do not depend on the step size;
the failed ply is then degraded and the step solved again until no new failure appears
(load redistribution). A test stops at `max_strain`, when all load-carrying plies have
failed, or once the stress has dropped below 5 % of its peak.

Events recorded per ply (first occurrence):

| Model | Events |
|---|---|
| `max_stress`, `hashin` | `fibre tension failure`, `fibre compression failure`, `matrix tension failure`, `matrix compression failure`, `shear failure` (maximum stress only) |
| `continuum_damage` | `<mode> damage onset`, `<mode> failure` (damage at the cap) |
| `rve_curves` | `transverse tension peak passed`, `transverse compression peak passed`, `shear peak passed`, `<curve> curve exceeded`, `fibre tension failure`, `fibre compression failure` |

The first ply failure is the first event other than "curve exceeded"; the first fibre
failure the first fibre event (for `continuum_damage`, the fibre damage onset). A "curve
exceeded" event means a ply went past the end of its RVE curve, whose last stress was then
held: extend the curve strains, or note that the RVE solve stopped there (see the
pipeline warnings).

Per laminate, `laminates/<sequence>/<model>/<test>.csv` holds the curve (`strain`,
`stress`, mid-plane strains, curvatures, damaged and fibre-failed plies, `elongation`,
`force`) and `<test>_summary.json` the initial modulus, peak stress and its strain, the
events and warnings. `laminates/coupon_tests.csv` has one row per laminate, model and test:

| Column | Meaning |
|---|---|
| `initial_modulus` | slope of the first step |
| `peak_stress`, `strain_at_peak` | largest stress magnitude and its strain |
| `first_ply_failure`, `first_ply_failure_strain`, `first_ply_failure_stress` | first damage event |
| `first_fibre_failure_strain` | first fibre failure (or fibre damage onset) |
| `curve_exceeded_strain` | first ply past the end of an RVE curve |
| `elongation_at_peak`, `force_at_peak` | with `gauge_length` and `width` |
| `final_strain`, `warnings`, `csv` | where the test ended, model warnings, the curve file |

`laminates/coupon_tests_<model>.png` plots every laminate and test of one model and
`laminates/<sequence>/coupon_tests.png` every model and test of one laminate.

## Glass/epoxy reference set

The `laminate_damage_*` pipelines use the E-glass/LY556/HT907/DY063 epoxy system of the
first World-Wide Failure Exercise (WWFE-I; Soden, Hinton & Kaddour, Compos. Sci. Technol.
58, 1998, 1011–1022), whose lamina and constituent data are published together:

| | Value | Source |
|---|---|---|
| Fibre: E-glass (Gevetex) | E = 80 GPa, ν = 0.2 | WWFE-I |
| Matrix: LY556/HT907/DY063 | E = 3.35 GPa, ν = 0.35; strengths 80 MPa (tension), 120 MPa (compression), 54 MPa (shear); failure strain 5 % | WWFE-I |
| Lamina | Vf = 0.62; E1 = 53.48 GPa, E2 = 17.7 GPa, G12 = 5.83 GPa, ν12 = 0.278 | WWFE-I |
| Lamina strengths | Xt = 1140, Xc = 570, Yt = 35, Yc = 114, S12 = 72 MPa | WWFE-I |

Model parameters chosen from these data:

- **Matrix** (pressure-dependent plasticity, [nonlinear.md](nonlinear.md#matrix-plasticity-and-damage)):
  yield at 50 MPa in tension and 75 MPa in compression (the ratio 1.5 of the strengths),
  Voce hardening (initial slope 3000 MPa) that levels off at the strengths: 80 MPa in
  tension, 120 MPa in compression and `sqrt(80 · 120 / 3)` = 57 MPa in shear (measured:
  54 MPa); plastic Poisson ratio 0.3. Damage starts at an equivalent plastic strain of 0.3
  and dissipates 0.002 N/mm over one element: the matrix in the narrow gaps between fibres
  must shear far more than a tensile specimen can stretch. The onset is one value for all
  stress states, so the matrix is too ductile in tension; this only affects the transverse
  tension curve after debonding (see below).
- **Interface** (bilinear mixed-mode cohesive law): strengths 50 MPa normal and 75 MPa
  shear, toughness 2 J/m² (mode I) and 6 J/m² (mode II), penalty stiffness 1e8 N/mm³,
  Benzeggagh–Kenane exponent 1.45. These are assumed values of the kind used in
  micromechanical studies of glass- and carbon-epoxy, not measurements for this system
  (apparent interfacial shear strengths reported for glass/epoxy are typically 25–60 MPa).
- **Fibre-direction strengths**: the WWFE-I values Xt = 1140 and Xc = 570 MPa (inputs).
- **Fracture energies** (continuum damage): 40 N/mm (fibre tension), 20 N/mm (fibre
  compression), 0.3 N/mm (matrix tension), 1.0 N/mm (matrix compression), with a crack band
  of 0.5 mm. Assumed, typical orders of magnitude for glass/epoxy laminates.
- **RVE**: 15 fibres of 14 µm diameter (a typical E-glass filament; the diameter is not part
  of the WWFE-I data) in a periodic 61 µm window (2D) or 7 fibres in a
  41.7 × 41.7 × 5 µm box (3D), random periodic packing at a target Vf of 0.62 (the meshed
  circles are polygons, so the meshes have Vf 0.613 and 0.608).

The RVE against the WWFE-I lamina data:

| Property | WWFE-I | RVE, 2D | RVE, 3D |
|---|---|---|---|
| Vf | 0.62 | 0.613 | 0.608 |
| E1 (GPa) | 53.48 | 50.3 (−6 %) | 50.0 (−7 %) |
| E2 (GPa) | 17.7 | 14.6 (−17 %) | 15.0 (−15 %) |
| G12 (GPa) | 5.83 | 5.51 (−5 %) | 5.55 (−5 %) |
| ν12 | 0.278 | 0.243 | 0.244 |
| Yt (MPa) | 35 | 37.2 (+6 %) | 41.2 (+18 %) |
| Yc (MPa) | 114 | 135.1 (+18 %) | 127.5 (+12 %) |
| S12 (MPa) | 72 | 55.6 (−23 %) | 58.5 (−19 %) |
| Xt, Xc (MPa) | 1140, 570 | inputs | inputs |

- **Stiffness.** E1 is the rule of mixtures of the published constituents (50.3 GPa at Vf
  0.613); the measured 53.48 GPa is above it even at Vf 0.62 (50.9 GPa). The RVE's E2 is
  15–17 % below the measured value but close to the Halpin–Tsai estimate from the same
  constituents (15.5 GPa at Vf 0.62, ξ = 2), so most of that gap lies between the
  published constituent and lamina data rather than in the RVE.
- **Strengths.** Yt comes from fibre/matrix debonding (interface strength 50 MPa) and Yc
  from matrix yielding with interface shear failure; both are within 6–18 % of the
  measured values. S12 is 19–23 % low: in longitudinal shear the matrix levels off at
  `sqrt(σc σt / 3)` = 57 MPa (measured matrix shear strength 54 MPa) and the RVE's shear
  strength cannot exceed that by much, whereas the measured lamina S12 (72 MPa) is a third
  above the matrix's. Set `strengths.in_plane_shear: 72.0` to use the measured value.
- **Scatter.** With 15 (2D) or 7 (3D) fibres the RVE is small: another random packing (14
  fibres, Vf 0.59) gave Yt = 35.4, Yc = 119.9 and S12 = 55.0 MPa. Expect differences of
  about 10 % between realisations; use a larger RVE or several seeds for converged
  strengths.
- **After the peak.** The transverse tension curve drops to about 35 % of its peak after
  debonding and the RVE solves stop where damage localises (snap-back), at 1.3–2.7 %
  transverse and 2.5–5.3 % shear strain; `pipeline_summary.json` lists these as warnings.
  The `rve_curves` model holds the last stress beyond them.

First ply failure / peak stress (MPa) of the coupon tests of the 2D example, by damage
model:

| Laminate | Test | `rve_curves` | `max_stress` | `hashin` | `continuum_damage` |
|---|---|---|---|---|---|
| `[0]8` | tension | 1140 / 1140 | 1140 / 1140 | 1140 / 1140 | 1139 / 1139 |
| `[0]8` | compression | 570 / 570 | 570 / 570 | 570 / 570 | 570 / 570 |
| `[90]8` | tension | 37 / 37 | 37 / 37 | 37 / 88 | 37 / 37 |
| `[90]8` | compression | 135 / 135 | 135 / 135 | 135 / 135 | 135 / 135 |
| `[0/90]2s` | tension | 108 / 577 | 84 / 570 | 84 / 603 | 85 / 572 |
| `[0/90]2s` | compression | 327 / 351 | 305 / 305 | 305 / 317 | 306 / 346 |
| `[±45]2s` | tension | 110 / 111 | 111 / 111 | 89 / 122 | 89 / 89 |
| `[±45]2s` | compression | 110 / 111 | 111 / 111 | 109 / 154 | 109 / 109 |
| `[0/±45/90]s` | tension | 90 / 425 | 70 / 383 | 70 / 424 | 70 / 398 |
| `[0/±45/90]s` | compression | 260 / 271 | 204 / 204 | 191 / 233 | 192 / 265 |
| `[90/±30]s` | tension | 90 / 553 | 70 / 503 | 70 / 559 | 70 / 505 |
| `[90/±30]s` | compression | 260 / 362 | 236 / 285 | 231 / 342 | 232 / 342 |

The unidirectional laminates fail at the ply strengths in every model. First ply failure is
the same for the three strength-based models when one stress dominates, and earlier for
Hashin and continuum damage when the transverse and shear stresses interact (`[±45]2s`,
compression of the quasi-isotropic laminates); `rve_curves` reports it later, where a ply
passes the peak of its curve. The peaks differ mostly through what each model does after
first ply failure. Note the `[90]8` tension test with `hashin`: once every ply has failed in
matrix tension, the residual 20 % of E2 keeps carrying load, so the curve rises past the
strength up to `max_strain`. For a laminate whose plies all fail in the same matrix mode,
read the first ply failure as its strength. (`[0/±45/90]s` and `[90/±30]s` are both
quasi-isotropic, so their stiffnesses are equal and only their strengths differ.)

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
| Coupon test with linear plies against the CLT moduli (x, y, xy; tension and compression; unbalanced, unsymmetric) | to 1e-8 |
| `rve_curves`: a `[90]` laminate reproduces the transverse tension and compression curves; secant unloading; curve ends reported | to 1e-8 |
| `max_stress`, `hashin`: unidirectional tension and compression fail at Xt and Xc | stress to 1e-3, strain to 5e-6 |
| `max_stress`: first ply failure of a cross-ply at the closed-form CLT strain where the 90 plies reach Yt | to 2e-6 |
| `hashin`: the interaction lowers the off-axis strength; 20 % of E2 kept after matrix tension failure | slope to 2 % |
| `continuum_damage`: peak at the strength, linear softening, dissipated energy G / L (matrix tension, fibre tension and compression); snap-back warning | to 1 %, 5 %, 2 % |
| Missing strengths or fracture energies reported per model | `tests/test_laminate_coupon.py` |
| Pipeline end to end with all four models, tension and compression; strengths from the RVE curves; ply reuse with `--ply`; config validation; engines give the same ply | `tests/test_laminate_pipeline.py` |

## Performance

Wall time of `rve2d laminate` on the example pipelines (4-core Xeon at 2.8 GHz, one run at a
time; Julia times include about 4 s of start-up per solve):

| Config | TensorMesh | Julia |
|---|---|---|
| `laminate_stiffness_2d.yaml` (7 laminates) | 2.0 s | 4.8 s |
| `laminate_stiffness_3d.yaml` (7 laminates) | 2.5 s | 4.9 s |
| `laminate_damage_2d.yaml` | 4.6 min | 3.7 min |
| — transverse tension (3,375 unknowns, 121 increments, 889 Newton iterations) | 114 s | 55 s |
| — transverse compression (68 increments, 506 iterations) | 44 s | 33 s |
| — shear, extruded layer (5,063 unknowns, 29 increments, 229 iterations) | 62 s | 76 s |
| `laminate_damage_3d.yaml` | 7.1 min | 5.4 min |
| — transverse tension (2,501 unknowns, 75 increments, 572 iterations) | 165 s | 87 s |
| — transverse compression (110 increments, 789 iterations) | 126 s | 111 s |
| — shear (58 increments, 423 iterations) | 73 s | 68 s |

The rest of a damage pipeline (RVE build, linear homogenization, 48 coupon tests and the
plots) takes about a minute; the coupon tests alone, i.e. a rerun with `--ply`, take 55 s.
Both engines take the same increments and Newton iterations. Between the engines the ply
stiffness agrees to 3e-16, the RVE curves to 3e-11 and the 48 coupon test curves to 2e-11
(2D example; 3D: 1.5e-13 and 4e-14).

Meshing the shear problem of a 2D RVE as a 3D slab with gmsh instead of extruding one layer
(28,000 tetrahedra, 37,000 unknowns, on an earlier example) had not finished its shear curve
after 17 minutes, which is why the pipeline uses the extruded layer.

## Outputs

```
OUTPUT/
├── rve/                         RVE mesh and metadata (as rve2d build)
├── ply/
│   ├── elastic/                 linear homogenization of the RVE
│   ├── transverse_tension/      nonlinear RVE solve, xx > 0
│   ├── transverse_compression/  nonlinear RVE solve, xx < 0
│   ├── shear/                   nonlinear RVE solve, xz (2D: on rve_layer.msh)
│   └── ply_properties.json      ply stiffness, curves, strengths (reuse with --ply)
├── laminates/
│   ├── summary.csv              stiffness, one row per stacking sequence
│   ├── coupon_tests.csv         one row per laminate, damage model and test
│   ├── coupon_tests_<model>.png
│   └── <sequence>/              abd.csv, effective_3d_stiffness.csv, constants.json,
│       │                        coupon_tests.png
│       └── <model>/             <test>.csv, <test>_summary.json
└── pipeline_summary.json        ply constants, strengths and their sources, curves,
                                 laminates, coupon test summaries, warnings
```

## Limitations

- One ply material and thickness per laminate; perfectly bonded plies (no delamination);
  thin-plate (CLT) kinematics, so no free-edge stresses.
- Fibre failure is not part of the RVE: Xt and Xc are inputs.
- The ply damage models act at one material point per Gauss point: no localisation, so the
  post-peak branch of `continuum_damage` depends on the characteristic length, and that of
  the sudden-failure models on their degradation rules. The coupon tests are quasi-static and
  strain controlled.
- `rve_curves` has no interaction between the transverse and shear responses; past the end
  of an RVE curve the stress is held at its last value.
- The RVE's damage onset is one equivalent plastic strain for all stress states, and the
  strength of an RVE curve is limited by its constituents: the in-plane shear strength
  cannot exceed the matrix's shear plateau by much (see the reference set above).
