# Small-strain J2 (von Mises) plasticity with linear isotropic hardening (Simo & Hughes,
# Computational Inelasticity, 1998, Box 3.1/3.2), on 3D symmetric tensors so that plane
# strain and generalized plane strain carry their out-of-plane stress. Yield function
# `||s|| - sqrt(2/3) (σy + H α)`, with `α` the equivalent (uniaxial) plastic strain; phases
# with `σy = Inf` are linear elastic.

struct PhaseMaterial
    shear_modulus::Float64
    bulk_modulus::Float64
    yield_stress::Float64
    hardening_modulus::Float64
end

function PhaseMaterial(; youngs_modulus, poisson_ratio, yield_stress = Inf,
        hardening_modulus = 0.0)
    G = youngs_modulus / (2 * (1 + poisson_ratio))
    K = youngs_modulus / (3 * (1 - 2 * poisson_ratio))
    return PhaseMaterial(G, K, yield_stress, hardening_modulus)
end

struct PlasticState
    plastic_strain::SymmetricTensor{2, 3, Float64, 6}
    equivalent_plastic_strain::Float64
end

PlasticState() = PlasticState(zero(SymmetricTensor{2, 3, Float64}), 0.0)

const IDENTITY = one(SymmetricTensor{2, 3, Float64})
const VOLUMETRIC = IDENTITY ⊗ IDENTITY
const DEVIATORIC = one(SymmetricTensor{4, 3, Float64}) - VOLUMETRIC / 3

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

function von_mises(stress::SymmetricTensor{2, 3})
    s = dev(stress)
    return sqrt(1.5 * (s ⊡ s))
end
