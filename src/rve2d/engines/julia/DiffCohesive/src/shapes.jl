"""
    ShapeLaw{S}(; stiffness, strength, toughness, smoothing_fraction = 1e-3)

Pure-mode envelope shapes of Alfano, "On the influence of the shape of the interface law on
the application of cohesive-zone models", Compos. Sci. Technol. 66 (2006) 723-730, all
parametrised by the initial stiffness `K0`, the peak traction `σ0` and the fracture energy
`Gc`: [`BilinearShape`](@ref), [`LinearParabolic`](@ref), [`ExponentialShape`](@ref) and
[`Trapezoidal`](@ref). The envelope drives an irreversible secant damage
`D = 1 - σ_env(κ) / (K0 κ)` of the effective opening `λ = sqrt(<δn>² + |δs|²)`; compression
keeps the full penalty stiffness. One parameter set serves all modes (use
[`BilinearMixedMode`](@ref) for mode-dependent toughness).

Counterparts of diffcohesive's `SHAPE_LAWS` (`K0`, `sigma0`, `Gc`).
"""
struct ShapeLaw{S} <: TractionSeparationLaw
    stiffness::Float64
    strength::Float64
    toughness::Float64
    smoothing_fraction::Float64
end

"Bilinear envelope: linear to `σ0`, then linear softening to zero at `2 Gc / σ0`."
const BilinearShape = ShapeLaw{:bilinear}
"Linear to `σ0/2`, then a parabola through the peak `σ0` down to zero."
const LinearParabolic = ShapeLaw{:linear_parabolic}
"Exponential rise to `σ0` followed by an exponential tail."
const ExponentialShape = ShapeLaw{:exponential}
"Trapezoidal (Tvergaard-Hutchinson type): rise, plateau at `σ0`, linear softening."
const Trapezoidal = ShapeLaw{:trapezoidal}

const SHAPES = (:bilinear, :linear_parabolic, :exponential, :trapezoidal)
const LP_ENERGY_COEFF = (5 + 4 * sqrt(2)) / 6

function ShapeLaw{S}(;
    stiffness::Real,
    strength::Real,
    toughness::Real,
    smoothing_fraction::Real = 1.0e-3,
) where {S}
    S in SHAPES || throw(ArgumentError("unknown shape $S; use one of $SHAPES"))
    all(>(0), (stiffness, strength, toughness, smoothing_fraction)) ||
        throw(ArgumentError("stiffness, strength, toughness and smoothing_fraction must be positive"))
    lower = minimum_toughness(ShapeLaw{S}, stiffness, strength)
    toughness > lower || throw(
        ArgumentError(
            "toughness $toughness is too small for the $S shape with this stiffness and " *
            "strength (it must exceed $lower)",
        ),
    )
    return ShapeLaw{S}(stiffness, strength, toughness, smoothing_fraction)
end

"""
    shape_law(name; stiffness, strength, toughness, smoothing_fraction = 1e-3)

Shape law by its diffcohesive name: `"bilinear"`, `"linear-parabolic"`, `"exponential"` or
`"trapezoidal"`.
"""
function shape_law(name::AbstractString; kwargs...)
    shape = Symbol(replace(name, "-" => "_"))
    shape in SHAPES || throw(ArgumentError("unknown shape law \"$name\""))
    return ShapeLaw{shape}(; kwargs...)
end

# Smallest toughness for which each envelope is well defined.
minimum_toughness(::Type{BilinearShape}, K0, σ0) = σ0^2 / (2 * K0)
minimum_toughness(::Type{LinearParabolic}, K0, σ0) = σ0^2 / (8 * K0)
minimum_toughness(::Type{ExponentialShape}, K0, σ0) = σ0^2 * (ℯ^2 - 2ℯ) / K0
minimum_toughness(::Type{Trapezoidal}, K0, σ0) = σ0^2 / K0 * (1 - 1.0e-12)

state_length(::ShapeLaw) = 1

"Onset (peak) opening scale `strength / stiffness`."
onset_opening(law::ShapeLaw) = law.strength / law.stiffness

function traction(
    law::ShapeLaw,
    δ::SVector{D, T},
    state::SVector{1, Float64};
    Δt::Real = 1.0,
) where {D, T}
    K0 = law.stiffness
    δ0 = law.strength / K0
    ε = law.smoothing_fraction * δ0
    εs = (SHEAR_REGULARISATION * δ0)^2

    δn = δ[1]
    shear = popfirst(δ)
    δs = sqrt(sum(abs2, shear) + εs)
    mn = smooth_macaulay(δn, ε)
    λ = sqrt(mn * mn + δs * δs)
    κ = smooth_max(state[1], λ, ε)

    κsafe = max(κ, HISTORY_FLOOR * δ0)
    secant = envelope(law, κsafe) / (K0 * κsafe)
    damage = clamp(1 - secant, zero(secant), one(secant))

    tn = K0 * δn - damage * K0 * mn
    ts = (1 - damage) * K0 * shear
    return pushfirst(ts, tn), SVector(κ), damage
end

"""
    envelope(law, δ)

Traction of the monotonic pure-mode envelope at effective opening `δ`.
"""
function envelope(law::BilinearShape, δ)
    K0, σ0, Gc = law.stiffness, law.strength, law.toughness
    a0 = σ0 / K0
    a1 = 2 * Gc / σ0
    env = δ <= a0 ? K0 * δ : σ0 * (a1 - δ) / (a1 - a0)
    return max(env, zero(env))
end

function envelope(law::LinearParabolic, δ)
    K0, σ0, Gc = law.stiffness, law.strength, law.toughness
    b0 = σ0 / (2 * K0)
    b1 = b0 + (Gc - σ0^2 / (8 * K0)) / (LP_ENERGY_COEFF * σ0)
    x = (δ - b0) / (b1 - b0)
    parabola = σ0 * (0.5 + x - 0.5 * x * x)
    return δ <= b0 ? K0 * δ : max(parabola, zero(parabola))
end

function envelope(law::ExponentialShape, δ)
    K0, σ0, Gc = law.stiffness, law.strength, law.toughness
    c0 = σ0 * ℯ / K0
    β = 2 * σ0 / (Gc - K0 * c0^2 * (1 - 2 / ℯ))
    if δ <= c0
        return K0 * δ * exp(-δ / c0)
    end
    x = β * (δ - c0)
    return σ0 * (1 + x) * exp(-x)
end

function envelope(law::Trapezoidal, δ)
    K0, σ0, Gc = law.stiffness, law.strength, law.toughness
    d0 = σ0 / K0
    d1 = Gc / σ0
    d2 = d0 + d1
    δ <= d0 && return K0 * δ
    δ <= d1 && return σ0 * one(δ)
    softening = σ0 * (d2 - δ) / (d2 - d1)
    return max(softening, zero(softening))
end
