using DiffCohesive
using StaticArrays
using Test

# Lengths in mm, stresses in MPa: a glass/epoxy fibre-matrix interface.
const K, TN, TS, GIc, GIIc = 1.0e8, 50.0, 75.0, 0.002, 0.006
const δ0 = TN / K

mixed_mode(; kw...) = BilinearMixedMode(;
    stiffness = K, normal_strength = TN, shear_strength = TS, mode_i_toughness = GIc,
    mode_ii_toughness = GIIc, exponent = 1.45, kw...,
)
shapes(; stiffness = K, strength = TN, toughness = GIc) = [
    S(; stiffness, strength, toughness) for
    S in (BilinearShape, LinearParabolic, ExponentialShape, Trapezoidal)
]

unit(v) = v / sqrt(sum(abs2, v))

"Work of the traction along the monotonic proportional path δ = s * direction."
function dissipated_energy(law, direction::SVector; smax = 4.0e-4, n = 40_000)
    direction = unit(direction)
    state = initial_state(law)
    t_prev = zero(direction)
    work = 0.0
    for i in 1:n
        δ = (smax * i / n) * direction
        t, state, _ = traction(law, δ, state)
        work += 0.5 * sum((t + t_prev) .* direction) * smax / n
        t_prev = t
    end
    return work
end

@testset "zero opening and unit independence" begin
    # SI units (Pa, m): K = 1e17 Pa/m, strengths in Pa, toughness in J/m^2.
    si_mixed = BilinearMixedMode(;
        stiffness = 1.0e17, normal_strength = TN * 1.0e6, shear_strength = TS * 1.0e6,
        mode_i_toughness = GIc * 1.0e3, mode_ii_toughness = GIIc * 1.0e3, exponent = 1.45,
    )
    si_shapes = shapes(; stiffness = 1.0e17, strength = TN * 1.0e6, toughness = GIc * 1.0e3)
    pairs = [(mixed_mode(), si_mixed); collect(zip(shapes(), si_shapes))]
    for (mm, si) in pairs, dim in (2, 3)
        zero_opening = zeros(SVector{dim, Float64})
        _, _, damage = traction(mm, zero_opening, initial_state(mm))
        if mm isa ExponentialShape
            @test damage < 1.0e-3  # the exponential envelope softens from the start
        else
            @test damage == 0.0
        end
        for δ in (0.4, 1.7, 9.0, 60.0) .* δ0
            direction = unit(SVector{dim}(ntuple(i -> sin(3.0 * i + δ / δ0), dim)))
            state_mm = SVector(0.3 * δ)
            t_mm, s_mm, d_mm = traction(mm, δ * direction, state_mm)
            t_si, s_si, d_si = traction(si, 1.0e-3 * δ * direction, 1.0e-3 * state_mm)
            @test isapprox(t_si / 1.0e6, t_mm; rtol = 1.0e-12, atol = 1.0e-12 * TN)
            @test isapprox(s_si[1] * 1.0e3, s_mm[1]; rtol = 1.0e-12)
            @test isapprox(d_si, d_mm; atol = 1.0e-12)
        end
    end
end

@testset "dissipated energy" begin
    law = mixed_mode()
    @test isapprox(dissipated_energy(law, SVector(1.0, 0.0)), GIc; rtol = 2.0e-3)
    @test isapprox(dissipated_energy(law, SVector(0.0, 1.0)), GIIc; rtol = 2.0e-3)
    @test isapprox(dissipated_energy(law, SVector(0.0, 0.6, 0.8)), GIIc; rtol = 2.0e-3)
    bk = GIc + (GIIc - GIc) * 0.5^1.45  # proportional path with mode ratio B = 1/2
    @test isapprox(dissipated_energy(law, SVector(1.0, 1.0)), bk; rtol = 2.0e-3)
    power = mixed_mode(; exponent = 2.0, criterion = :power)
    gc_power = ((0.5 / GIc)^2 + (0.5 / GIIc)^2)^(-1 / 2)
    @test isapprox(dissipated_energy(power, SVector(1.0, 1.0)), gc_power; rtol = 2.0e-3)
    soft_shear = mixed_mode(; shear_stiffness = 0.5 * K)
    @test isapprox(dissipated_energy(soft_shear, SVector(0.0, 1.0)), GIIc; rtol = 2.0e-3)
    for law in shapes()
        @test isapprox(dissipated_energy(law, SVector(1.0, 0.0)), GIc; rtol = 2.0e-3)
    end
end

@testset "automatic-differentiation tangent" begin
    laws = [mixed_mode(); mixed_mode(; criterion = :power, exponent = 2.0);
        mixed_mode(; viscosity = 0.3); shapes()]
    for law in laws, dim in (2, 3), magnitude in (0.5, 3.0, 25.0), sign in (1.0, -1.0)
        direction = unit(SVector{dim}(ntuple(i -> cos(2.0 * i + magnitude), dim)))
        δ = magnitude * δ0 * direction .* SVector{dim}(ntuple(i -> i == 1 ? sign : 1.0, dim))
        state = state_length(law) == 2 ? SVector(0.8 * magnitude * δ0, 0.1) :
            SVector(0.8 * magnitude * δ0)
        t, J, _, _ = traction_tangent(law, δ, state; Δt = 0.5)
        h = 1.0e-6 * magnitude * δ0
        for j in 1:dim
            e = SVector{dim}(ntuple(i -> i == j ? h : 0.0, dim))
            fd = (first(traction(law, δ + e, state; Δt = 0.5)) -
                  first(traction(law, δ - e, state; Δt = 0.5))) / (2h)
            @test isapprox(J[:, j], fd; rtol = 1.0e-5, atol = 1.0e-6 * K)
        end
        @test t == first(traction(law, δ, state; Δt = 0.5))
    end
end

@testset "irreversible unloading" begin
    law = mixed_mode()
    loaded, state, damage = traction(law, SVector(20δ0, 0.0), initial_state(law))
    @test 0.5 < damage < 1.0
    unloaded, state2, damage2 = traction(law, SVector(10δ0, 0.0), state)
    @test isapprox(damage2, damage; rtol = 1.0e-6)
    @test isapprox(unloaded[1], (1 - damage) * K * 10δ0; rtol = 1.0e-5)
    @test state2[1] >= state[1]
    compressed, _, _ = traction(law, SVector(-2δ0, 0.0), state)
    @test isapprox(compressed[1], -2 * K * δ0; rtol = 1.0e-3)  # contact keeps the penalty
end

@testset "viscous regularisation" begin
    law = mixed_mode(; viscosity = 0.2)
    @test state_length(law) == 2
    δ = SVector(20δ0, 0.0)
    _, _, rate_independent = traction(mixed_mode(), δ, initial_state(mixed_mode()))
    _, slow, d_slow = traction(law, δ, initial_state(law); Δt = 1.0e6)
    @test isapprox(d_slow, rate_independent; rtol = 1.0e-6)
    _, fast, d_fast = traction(law, δ, SVector(0.0, 0.1); Δt = 1.0e-6)
    @test isapprox(d_fast, 0.1; atol = 1.0e-5)
    @test slow[2] == d_slow && fast[2] == d_fast
    @test_throws DimensionMismatch traction(law, δ, SVector(0.0))
end

@testset "parameter validation" begin
    @test_throws ArgumentError mixed_mode(; criterion = :linear)
    @test_throws ArgumentError BilinearMixedMode(;
        stiffness = K, normal_strength = TN, mode_i_toughness = TN^2 / (2K) / 2,
    )
    @test_throws ArgumentError ExponentialShape(; stiffness = K, strength = TN, toughness = 1.5 * TN^2 / K)
    @test shape_law("linear-parabolic"; stiffness = K, strength = TN, toughness = GIc) isa LinearParabolic
    @test_throws ArgumentError shape_law("cubic"; stiffness = K, strength = TN, toughness = GIc)
    @test onset_opening(mixed_mode()) == δ0
end

@testset "parity with diffcohesive" begin
    # test/reference_diffcohesive.csv: diffcohesive 0.1.2 laws evaluated in the Python rve2d
    # solver (unit-safe onset-opening scaling), same units and parameters as above.
    lines = readlines(joinpath(@__DIR__, "reference_diffcohesive.csv"))
    header = split(lines[1], ',')
    column(name) = findfirst(==(name), header)
    count = 0
    for line in lines[2:end]
        fields = split(line, ',')
        value(name) = parse(Float64, fields[column(name)])
        name, criterion = fields[column("law")], fields[column("criterion")]
        dim = parse(Int, fields[column("dim")])
        viscosity, Δt = value("viscosity"), value("dt")
        law = if name == "bilinear_mixed_mode"
            ks = value("shear_stiffness")
            mixed_mode(;
                exponent = criterion == "power" ? 2.0 : 1.45, criterion = Symbol(criterion),
                shear_stiffness = ks > 0 ? ks : K, viscosity,
            )
        else
            shape_law(name; stiffness = K, strength = TN, toughness = GIc)
        end
        δ = SVector{dim}(ntuple(i -> value("delta$(i - 1)"), dim))
        state = state_length(law) == 2 ? SVector(value("kappa_prev"), value("dv_prev")) :
            SVector(value("kappa_prev"))
        t, J, new_state, damage = traction_tangent(law, δ, state; Δt)
        t_ref = SVector{dim}(ntuple(i -> value("t$(i - 1)"), dim))
        J_ref = SMatrix{dim, dim}(ntuple(k -> value("J$((k - 1) % dim)$((k - 1) ÷ dim)"), dim^2))
        @test isapprox(t, t_ref; rtol = 1.0e-9, atol = 1.0e-9 * TN)
        @test isapprox(damage, value("damage"); atol = 1.0e-10)
        @test isapprox(new_state[1], value("kappa_new"); rtol = 1.0e-10)
        state_length(law) == 2 && @test isapprox(new_state[2], value("dv_new"); atol = 1.0e-10)
        @test isapprox(J, J_ref; rtol = 1.0e-7, atol = 1.0e-7 * K)
        count += 1
    end
    @test count == 160
end
