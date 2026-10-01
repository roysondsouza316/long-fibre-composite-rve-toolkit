# Small-strain plasticity and ductile damage of the RVE phases on 3D symmetric tensors, so that
# plane strain and generalized plane strain carry their out-of-plane stress (Julia counterpart
# of rve2d.engines.tensormesh.nonlinear.material, same algorithms and tolerances).
#
# * J2 (von Mises) with linear isotropic hardening (Simo & Hughes, Computational Inelasticity,
#   1998, Box 3.1/3.2): yield function `||s|| - sqrt(2/3) (σy + H α)`, `α` the equivalent
#   (uniaxial) plastic strain.
# * Pressure-dependent paraboloidal plasticity for polymer matrices (Tschoegl 1971; Melro et
#   al. 2013): `f = q² + 3p(σc - σt) - σc σt` with the current tensile and compressive yield
#   stresses (compression scaled with tension) and the non-associative potential
#   `g = q² + α_g p²`, `α_g = 9/2 (1 - 2ν_p) / (1 + ν_p)`.
# * Hardening: linear, or Voce's saturation `σt = σs - (σs - σt0) exp(-H ε̄ / (σs - σt0))`
#   towards a finite `saturation_stress` σs (paraboloidal return mapping).
# * Ductile damage: past `damage_onset` the stress is `(1 - d)` times the effective stress,
#   `d` linear in the plastic displacement over a crack band one element wide (capped at
#   MAX_DAMAGE); `d` is a function of the equivalent plastic strain.
# Phases with `σy = Inf` are linear elastic.

const MAX_DAMAGE = 0.99
const ROOT_TOLERANCE = 1.0e-14
const ROOT_ITERATIONS = 100

struct PhaseMaterial
    shear_modulus::Float64
    bulk_modulus::Float64
    yield_stress::Float64
    hardening_modulus::Float64
    compressive_yield_stress::Float64
    plastic_poisson_ratio::Float64
    damage_onset::Float64     # equivalent plastic strain at damage onset (Inf: no damage)
    fracture_energy::Float64
    damage_rate::Float64      # d(damage)/d(equivalent plastic strain), per element
    saturation_stress::Float64  # Voce hardening (Inf: linear hardening)
end

function PhaseMaterial(; youngs_modulus, poisson_ratio, yield_stress = Inf,
        hardening_modulus = 0.0, compressive_yield_stress = yield_stress,
        plastic_poisson_ratio = 0.5, damage_onset = Inf, fracture_energy = Inf,
        saturation_stress = Inf)
    G = youngs_modulus / (2 * (1 + poisson_ratio))
    K = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    return PhaseMaterial(G, K, yield_stress, hardening_modulus, compressive_yield_stress,
        plastic_poisson_ratio, damage_onset, fracture_energy, 0.0, saturation_stress)
end

"""
    tensile_yield_stress(m, eqps) -> (stress, slope)

Current tensile yield stress and its slope: linear hardening, or Voce's saturation towards
a finite `saturation_stress` with initial slope `hardening_modulus`.
"""
function tensile_yield_stress(m::PhaseMaterial, eqps::Float64)
    H = m.hardening_modulus
    isfinite(m.saturation_stress) || return m.yield_stress + H * eqps, H
    span = m.saturation_stress - m.yield_stress
    decay = exp(-H * eqps / span)
    return m.saturation_stress - span * decay, H * decay
end

"""
    with_characteristic_length(m, length)

`m` for an element of crack-band width `length`: the damage reaches 1 at the plastic
displacement `2 G_f / σ_onset` (tensile yield stress at onset).
"""
function with_characteristic_length(m::PhaseMaterial, length::Float64)
    isfinite(m.damage_onset) || return m
    onset_stress = tensile_yield_stress(m, m.damage_onset)[1]
    rate = length * onset_stress / (2 * m.fracture_energy)
    return PhaseMaterial(m.shear_modulus, m.bulk_modulus, m.yield_stress, m.hardening_modulus,
        m.compressive_yield_stress, m.plastic_poisson_ratio, m.damage_onset,
        m.fracture_energy, rate, m.saturation_stress)
end

"Crack-band width: `sqrt(2A)` of a triangle, `(6V)^(1/3)` of a tetrahedron."
characteristic_length(volume::Float64, dim::Int) = dim == 2 ? sqrt(2 * volume) : cbrt(6 * volume)

is_general(m::PhaseMaterial) = isfinite(m.yield_stress) &&
    (m.compressive_yield_stress != m.yield_stress || m.plastic_poisson_ratio != 0.5 ||
     isfinite(m.damage_onset) || isfinite(m.saturation_stress))

function bulk_damage(eqps::Float64, m::PhaseMaterial)
    isfinite(m.damage_onset) || return 0.0
    return clamp(m.damage_rate * (eqps - m.damage_onset), 0.0, MAX_DAMAGE)
end

struct PlasticState
    plastic_strain::SymmetricTensor{2, 3, Float64, 6}
    equivalent_plastic_strain::Float64
end

PlasticState() = PlasticState(zero(SymmetricTensor{2, 3, Float64}), 0.0)

"Thrown when a return mapping does not converge; the Newton solver then cuts the increment."
struct MaterialUpdateError <: Exception end

const IDENTITY = one(SymmetricTensor{2, 3, Float64})
const VOLUMETRIC = IDENTITY ⊗ IDENTITY
const DEVIATORIC = one(SymmetricTensor{4, 3, Float64}) - VOLUMETRIC / 3

"""
    material_update(strain, state, material) -> (stress, tangent, new_state, yielding)

J2 radial return (or elastic) for von Mises phases; paraboloidal return mapping with ductile
damage otherwise. The consistent tangent need not be major-symmetric.
"""
function material_update(strain::SymmetricTensor{2, 3}, state::PlasticState, m::PhaseMaterial)
    is_general(m) || return j2_update(strain, state, m)
    σ̃, C̃, new_state, yielding, deqps = paraboloid_update(strain, state, m)
    ε̄ = new_state.equivalent_plastic_strain
    d = bulk_damage(ε̄, m)
    raw = isfinite(m.damage_onset) ? m.damage_rate * (ε̄ - m.damage_onset) : 0.0
    rate = yielding && 0 < raw < MAX_DAMAGE ? m.damage_rate : 0.0
    return (1 - d) * σ̃, (1 - d) * C̃ - rate * (σ̃ ⊗ deqps), new_state, yielding
end

"""
    j2_update(strain, state, material) -> (stress, tangent, new_state, yielding)

Radial return mapping and consistent (algorithmic) tangent.
"""
function j2_update(strain::SymmetricTensor{2, 3}, state::PlasticState, m::PhaseMaterial)
    G, K, H = m.shear_modulus, m.bulk_modulus, m.hardening_modulus
    elastic = strain - state.plastic_strain
    volumetric = tr(elastic)
    s_trial = 2G * dev(elastic)
    s_norm = norm(s_trial)
    radius = sqrt(2 / 3) * (m.yield_stress + H * state.equivalent_plastic_strain)
    f_trial = s_norm - radius
    if !(f_trial > 0)
        stress = s_trial + K * volumetric * IDENTITY
        return stress, K * VOLUMETRIC + 2G * DEVIATORIC, state, false
    end
    direction = s_trial / s_norm
    Δγ = f_trial / (2G * (1 + H / (3G)))
    stress = s_trial - 2G * Δγ * direction + K * volumetric * IDENTITY
    θ = 1 - 2G * Δγ / s_norm
    θbar = 1 / (1 + H / (3G)) - (1 - θ)
    tangent = K * VOLUMETRIC + 2G * θ * DEVIATORIC - 2G * θbar * (direction ⊗ direction)
    new_state = PlasticState(state.plastic_strain + Δγ * direction,
        state.equivalent_plastic_strain + sqrt(2 / 3) * Δγ)
    return stress, tangent, new_state, true
end

"""
    paraboloid_update(strain, state, material)
        -> (effective_stress, tangent, new_state, yielding, d_eqps_d_strain)

Return mapping of the paraboloidal surface (no damage): a safeguarded Newton iteration on the
plastic multiplier, then the consistent tangent; the last output is the derivative of the
equivalent plastic strain with respect to the strain (for the damage tangent).
"""
function paraboloid_update(strain::SymmetricTensor{2, 3}, state::PlasticState, m::PhaseMaterial)
    G, K = m.shear_modulus, m.bulk_modulus
    ratio = m.compressive_yield_stress / m.yield_stress  # σc follows σt
    νp = m.plastic_poisson_ratio
    α = 4.5 * (1 - 2νp) / (1 + νp)
    keq = 1 / (1 + 2νp^2)
    elastic = strain - state.plastic_strain
    s_trial = 2G * dev(elastic)
    p_trial = K * tr(elastic)
    q_trial = sqrt(1.5 * (s_trial ⊡ s_trial))
    ε̄n = state.equivalent_plastic_strain
    st_n = tensile_yield_stress(m, ε̄n)[1]
    sc_n = ratio * st_n
    f_trial = q_trial^2 + 3 * (sc_n - st_n) * p_trial - sc_n * st_n
    if !(f_trial > 0)
        stress = s_trial + p_trial * IDENTITY
        return stress, K * VOLUMETRIC + 2G * DEVIATORIC, state, false, zero(s_trial)
    end
    function residual(Δλ)
        # `local`: the enclosing function has its own a, b, q and p, which a closure would
        # otherwise share (boxed variables make the update type-unstable and slow)
        local a, b, q, p
        a = 1 + 6G * Δλ
        b = 1 + 2K * α * Δλ
        q, p = q_trial / a, p_trial / b
        mm = sqrt(keq * (6q^2 + (4 / 3) * α^2 * p^2))
        Δε̄ = Δλ * mm
        st, Ht = tensile_yield_stress(m, ε̄n + Δε̄)
        sc, Hc = ratio * st, ratio * Ht
        f = q^2 + 3 * (sc - st) * p - sc * st
        dq, dp = -6G * q / a, -2K * α * p / b
        dm = mm > 0 ? keq * (6q * dq + (4 / 3) * α^2 * p * dp) / mm : 0.0
        dΔε̄ = mm + Δλ * dm
        dst, dsc = Ht * dΔε̄, Hc * dΔε̄
        df = 2q * dq + 3 * (dsc - dst) * p + 3 * (sc - st) * dp - dsc * st - sc * dst
        return (; a, b, q, p, mm, Δε̄, st, sc, Ht, Hc, f, df, dΔε̄)
    end
    # Safeguarded Newton: f(0) > 0 and f -> -σc σt < 0 as Δλ grows.
    Δλ, low, high, first = 0.0, 0.0, Inf, Inf
    stalled, converged = false, false
    for _ in 1:ROOT_ITERATIONS
        r = residual(Δλ)
        if stalled || abs(r.f) <= ROOT_TOLERANCE * r.sc * r.st
            converged = true
            break
        end
        r.f > 0 && (low = Δλ)
        r.f < 0 && (high = Δλ)
        newton = Δλ - r.f / (r.df != 0 ? r.df : -1.0)
        isinf(first) && r.df < 0 && (first = newton)
        valid = r.df < 0 && low < newton < high
        grow = Δλ > 0 ? 2Δλ : (isfinite(first) ? first : 1 / (6G))
        updated = valid ? newton : (isinf(high) ? grow : (low + high) / 2)
        stalled = abs(updated - Δλ) <= 4 * eps(Float64) * abs(updated)
        Δλ = updated
    end
    converged || throw(MaterialUpdateError())
    r = residual(Δλ)
    a, b, q, p = r.a, r.b, r.q, r.p
    s = s_trial / a
    effective = s + p * IDENTITY
    plastic = state.plastic_strain + Δλ * (3s + (2α * p / 3) * IDENTITY)
    new_state = PlasticState(plastic, ε̄n + r.Δε̄)
    # Consistent tangent: dσ̃ = C_iso dε - v dΔλ with dΔλ = -(F_ε ⊡ dε) / f'(Δλ).
    φ = 3 * (r.Hc - r.Ht) * p - r.Hc * r.st - r.sc * r.Ht
    safe_m = r.mm > 0 ? r.mm : 1.0
    dε̄_dq = Δλ * keq * 6q / safe_m / a
    dε̄_dp = Δλ * keq * (4 / 3) * α^2 * p / safe_m / b
    f_q = 2q / a + φ * dε̄_dq
    f_p = 3 * (r.sc - r.st) / b + φ * dε̄_dp
    direction = q > 0 ? (3G / q) * s : zero(s)   # d q_trial / dε
    f_ε = f_q * direction + (f_p * K) * IDENTITY
    v = (6G / a) * s + (2K * α * p / b) * IDENTITY
    tangent = (K / b) * VOLUMETRIC + (2G / a) * DEVIATORIC + (v ⊗ f_ε) / r.df
    dε̄ = r.dΔε̄ * (-f_ε / r.df) + dε̄_dq * direction + (dε̄_dp * K) * IDENTITY
    return effective, tangent, new_state, true, dε̄
end

function von_mises(stress::SymmetricTensor{2, 3})
    s = dev(stress)
    return sqrt(1.5 * (s ⊡ s))
end
