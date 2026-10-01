"""
    DiffCohesive

Traction-separation laws for zero-thickness cohesive elements with automatic-differentiation
tangents: a Julia counterpart of the Python package
[diffcohesive](https://pypi.org/project/diffcohesive/).

A law maps the local separation of an integration point, `δ = [δn, δs1(, δs2)]` (normal
first, then the shear components), and its history state to a traction in the same frame:

    t, new_state, damage = traction(law, δ, state; Δt = 1.0)
    t, dt_dδ, new_state, damage = traction_tangent(law, δ, state; Δt = 1.0)

`dt_dδ` is the algorithmic tangent (it includes the history update) computed with
ForwardDiff, so a law written as a plain Julia function gets an exact tangent. Separations,
tractions and states are `SVector`s; a state has `state_length(law)` entries and starts from
`initial_state(law)`. The state returned for a trial separation is committed by the caller
once its load increment has converged.

The formulas and the damage/history formalism are those of diffcohesive, with one
difference: diffcohesive's small absolute regularisation constants are expressed here
relative to the onset opening, so results do not depend on the unit system.
"""
module DiffCohesive

using ForwardDiff: ForwardDiff
using StaticArrays: SMatrix, SVector, popfirst, pushfirst

export TractionSeparationLaw, BilinearMixedMode, ShapeLaw, BilinearShape, LinearParabolic,
    ExponentialShape, Trapezoidal, shape_law, state_length, initial_state, onset_opening,
    traction, traction_tangent

"""
Supertype of all laws. A law implements `traction(law, δ, state; Δt)`, `state_length(law)`
and `onset_opening(law)`.
"""
abstract type TractionSeparationLaw end

"C¹ approximation of `max(x, 0)`, exact as `ε → 0`."
smooth_macaulay(x, ε) = (x + sqrt(x * x + ε * ε)) / 2

"C¹ approximation of `max(a, b)`, used to advance the history variable."
smooth_max(a, b, ε) = (a + b + sqrt((a - b) * (a - b) + ε * ε)) / 2

# diffcohesive regularises the shear norm with an absolute 1e-12 (length²) and floors the
# history at an absolute 1e-15 (length); both are taken relative to the onset opening here.
const SHEAR_REGULARISATION = 1.0e-6
const HISTORY_FLOOR = 1.0e-15

"""
    initial_state(law) -> SVector

History of an undamaged integration point.
"""
initial_state(law::TractionSeparationLaw) = zeros(SVector{state_length(law), Float64})

"""
    traction_tangent(law, δ, state; Δt = 1.0) -> (t, dt_dδ, new_state, damage)

Traction, its algorithmic tangent `∂t/∂δ` (an `SMatrix`, by forward-mode automatic
differentiation), the updated state and the damage.
"""
function traction_tangent(
    law::TractionSeparationLaw,
    δ::SVector{D, Float64},
    state::SVector{N, Float64};
    Δt::Real = 1.0,
) where {D, N}
    t, new_state, damage = traction(law, δ, state; Δt)
    jacobian = ForwardDiff.jacobian(d -> first(traction(law, d, state; Δt)), δ)
    return t, jacobian, new_state, damage
end

function _check_state(law::TractionSeparationLaw, ::SVector{N}) where {N}
    N == state_length(law) || throw(
        DimensionMismatch("state has $N entries, the law needs $(state_length(law))"),
    )
    return nothing
end

include("bilinear.jl")
include("shapes.jl")

end # module
