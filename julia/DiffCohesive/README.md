# DiffCohesive.jl

Cohesive-zone traction-separation laws with automatic-differentiation tangents: a Julia
counterpart of the Python package [diffcohesive](https://pypi.org/project/diffcohesive/),
used by the Ferrite.jl nonlinear RVE solver in [`../nonlinear`](../nonlinear).

- `BilinearMixedMode`: mixed-mode bilinear law (Turon et al. onset/final-opening closure)
  with Benzeggagh-Kenane or power-law toughness, optional independent shear stiffness and
  Duvaut-Lions viscous regularisation.
- `BilinearShape`, `LinearParabolic`, `ExponentialShape`, `Trapezoidal`: the Alfano (2006)
  shape library with irreversible secant damage (`shape_law("linear-parabolic"; ...)`
  accepts diffcohesive's names).
- `traction(law, δ, state; Δt)` returns the traction, updated history and damage;
  `traction_tangent` adds `∂t/∂δ` from ForwardDiff, so new laws written as plain Julia
  functions get exact tangents too.

```julia
using DiffCohesive, StaticArrays

law = BilinearMixedMode(; stiffness = 1e8, normal_strength = 50.0, shear_strength = 75.0,
                        mode_i_toughness = 0.002, mode_ii_toughness = 0.006, exponent = 1.45)
state = initial_state(law)
t, dt_dδ, new_state, damage = traction_tangent(law, SVector(4e-7, 1e-7), state)
```

Separations are local to the interface: normal first, then one (2D) or two (3D) shear
components. States returned for a trial separation are committed by the caller after a
converged increment.

**Differences from diffcohesive.** The formulas, smoothing and history update are the same;
diffcohesive's absolute regularisation constants (`1e-12` under the shear norm, `1e-15` as
a history floor) are relative to the onset opening here. The results are therefore
independent of the unit system (with `K = 1e8 N/mm^3`, diffcohesive's constants produce
damage at zero separation).

**Tests.** `julia --project=. -e 'using Pkg; Pkg.test()'` checks dissipated energies (pure
and mixed mode, both criteria, all shapes), unit independence, AD tangents against finite
differences, unloading, viscosity and parity with diffcohesive 0.1.2
(`test/reference_diffcohesive.csv`, generated through the Python rve2d solver's unit-safe
wrapper): tractions agree to 1e-14 relative and tangents to 1e-15.
