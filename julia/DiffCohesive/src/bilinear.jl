"""
    BilinearMixedMode(; stiffness, normal_strength, mode_i_toughness,
                      shear_strength = normal_strength, mode_ii_toughness = mode_i_toughness,
                      shear_stiffness = stiffness, exponent = 1.0, criterion = :bk,
                      viscosity = 0.0, smoothing_fraction = 1e-3)

Mixed-mode bilinear law (Camanho-Davila / Alfano-Crisfield) with the onset and
final-opening closure of Turon et al. (2006). The mixed-mode toughness follows
Benzeggagh-Kenane, `G_c = G_Ic + (G_IIc - G_Ic) B^η` (`criterion = :bk`), or the power law
`(G_I/G_Ic)^α + (G_II/G_IIc)^α = 1` (`criterion = :power`); `exponent` is `η` or `α`, and
`B` is the energy mode ratio. With `shear_stiffness != stiffness` the final opening uses the
path stiffness, so pure modes dissipate `G_Ic`/`G_IIc` and proportional mixed-mode paths
dissipate `G_c(B)`.

`viscosity > 0` enables Duvaut-Lions regularisation of the damage: the damage that degrades
the traction relaxes towards the rate-independent value with relaxation time `viscosity`,
in the units of the `Δt` passed to [`traction`](@ref) (`Δt = 1` means per load step, as in
diffcohesive). The state is then `[κ, D_v]` instead of `[κ]`.

Counterpart of `diffcohesive.laws.BilinearMixedModeTSL(T_max_n, T_max_s, G_c1, G_c2, eta,
K, K_s, mixed_mode_criterion, viscosity)`.
"""
struct BilinearMixedMode <: TractionSeparationLaw
    stiffness::Float64
    shear_stiffness::Float64
    normal_strength::Float64
    shear_strength::Float64
    mode_i_toughness::Float64
    mode_ii_toughness::Float64
    exponent::Float64
    criterion::Symbol
    viscosity::Float64
    smoothing_fraction::Float64
end

function BilinearMixedMode(;
    stiffness::Real,
    normal_strength::Real,
    mode_i_toughness::Real,
    shear_strength::Real = normal_strength,
    mode_ii_toughness::Real = mode_i_toughness,
    shear_stiffness::Real = stiffness,
    exponent::Real = 1.0,
    criterion::Symbol = :bk,
    viscosity::Real = 0.0,
    smoothing_fraction::Real = 1.0e-3,
)
    criterion in (:bk, :power) ||
        throw(ArgumentError("criterion must be :bk or :power, got :$criterion"))
    positive = (stiffness, shear_stiffness, normal_strength, shear_strength, mode_i_toughness,
        mode_ii_toughness, exponent, smoothing_fraction)
    all(>(0), positive) || throw(
        ArgumentError(
            "stiffnesses, strengths, toughnesses, exponent and smoothing_fraction must be " *
            "positive",
        ),
    )
    viscosity >= 0 || throw(ArgumentError("viscosity must be non-negative"))
    _check_toughness(mode_i_toughness, normal_strength^2 / (2 * stiffness), "mode I")
    _check_toughness(mode_ii_toughness, shear_strength^2 / (2 * shear_stiffness), "mode II")
    return BilinearMixedMode(
        stiffness, shear_stiffness, normal_strength, shear_strength, mode_i_toughness,
        mode_ii_toughness, exponent, criterion, viscosity, smoothing_fraction,
    )
end

function _check_toughness(toughness, minimum, label)
    toughness > minimum || throw(
        ArgumentError(
            "$label toughness $toughness must exceed $minimum, the elastic energy at damage " *
            "onset (otherwise the final opening precedes the onset opening)",
        ),
    )
    return nothing
end

state_length(law::BilinearMixedMode) = law.viscosity > 0 ? 2 : 1

"Normal onset opening `normal_strength / stiffness`."
onset_opening(law::BilinearMixedMode) = law.normal_strength / law.stiffness

"""
    traction(law, δ, state; Δt = 1.0) -> (t, new_state, damage)

Traction for the local separation `δ` (normal first) given the committed `state`.
"""
function traction(
    law::BilinearMixedMode,
    δ::SVector{D, T},
    state::SVector{N, Float64};
    Δt::Real = 1.0,
) where {D, T, N}
    _check_state(law, state)
    K = law.stiffness
    Ks = law.shear_stiffness
    δ0n = law.normal_strength / K
    δ0s = law.shear_strength / Ks
    ε = law.smoothing_fraction * δ0n
    εs = (SHEAR_REGULARISATION * δ0n)^2

    δn = δ[1]
    shear = popfirst(δ)
    δs = sqrt(sum(abs2, shear) + εs)
    mn = smooth_macaulay(δn, ε)
    λ = sqrt(mn * mn + δs * δs)

    # Displacement mode ratio (onset interpolation, path stiffness) and energy mode ratio
    # (toughness criterion); they coincide when shear_stiffness == stiffness.
    mode_mix = δs * δs / (δs * δs + mn * mn + εs)
    B = Ks * δs * δs / (Ks * δs * δs + K * mn * mn + K * εs)
    δ0m = sqrt(δ0n * δ0n + (δ0s * δ0s - δ0n * δ0n) * mode_mix)
    GIc, GIIc, η = law.mode_i_toughness, law.mode_ii_toughness, law.exponent
    Gc = if law.criterion === :power
        Bc = clamp(B, 1.0e-12, 1.0 - 1.0e-12)
        (((1 - Bc) / GIc)^η + (Bc / GIIc)^η)^(-1 / η)
    else
        GIc + (GIIc - GIc) * B^η
    end
    Kpath = (1 - mode_mix) * K + mode_mix * Ks
    δfm = 2 * Gc / (Kpath * δ0m)

    κ = smooth_max(state[1], λ, ε)
    raw = δfm * (κ - δ0m) / (max(κ, ε) * max(δfm - δ0m, ε))
    new_state, damage = _relax(law, state, κ, clamp(raw, zero(raw), one(raw)), Δt)

    tn = K * δn - damage * K * mn
    ts = (1 - damage) * Ks * shear
    return pushfirst(ts, tn), new_state, damage
end

_relax(::BilinearMixedMode, ::SVector{1, Float64}, κ, damage, Δt) = (SVector(κ), damage)

function _relax(law::BilinearMixedMode, state::SVector{2, Float64}, κ, damage, Δt)
    μ = law.viscosity / Δt
    relaxed = (μ * state[2] + damage) / (μ + 1)
    return SVector(κ, relaxed), relaxed
end
