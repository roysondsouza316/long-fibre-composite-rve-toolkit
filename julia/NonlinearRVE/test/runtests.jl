# Structured meshes built in code (no gmsh): the same checks as the Python suite
# (tests/test_nonlinear.py) for interface insertion, patch tests, plasticity and the tangent.
using DiffCohesive: BilinearMixedMode
using Ferrite: Vec
using LinearAlgebra: dot, norm
using NonlinearRVE
using NonlinearRVE: LoadPath, NewtonSettings, PhaseMaterial, RVESystem, build_rve_mesh,
    evaluate, facet_normal, initial_state, n_cohesive, n_reduced, solve_load_path
using SparseArrays: sparse
using Test

const E_MATRIX, NU_MATRIX = 3500.0, 0.35

function structured_2d(n; fibre = (0.3, 0.7))
    xs = range(0.0, 1.0; length = n + 1)
    points = [Vec{2}((x, y)) for y in xs for x in xs]
    cells = Vector{NTuple{3, Int}}()
    for j in 0:(n - 1), i in 0:(n - 1)
        a = j * (n + 1) + i + 1
        b, c, d = a + 1, a + n + 2, a + n + 1
        push!(cells, (a, b, c), (a, c, d))
    end
    return points, phases(points, cells, fibre, 2)...
end

function structured_3d(n; fibre = (0.3, 0.7))
    xs = range(0.0, 1.0; length = n + 1)
    points = [Vec{3}((x, y, z)) for z in xs for y in xs for x in xs]
    node(i, j, k) = (k * (n + 1) + j) * (n + 1) + i + 1
    cells = Vector{NTuple{4, Int}}()
    for k in 0:(n - 1), j in 0:(n - 1), i in 0:(n - 1)
        v = [node(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in 0:7]
        for (first, second) in ((1, 2), (2, 1), (1, 4), (4, 1), (2, 4), (4, 2))
            push!(cells, (v[1], v[first + 1], v[(first | second) + 1], v[8]))
        end
    end
    return points, phases(points, cells, fibre, 2)...  # square prism fibre along z
end

function phases(points, cells, fibre, in_plane)
    connectivity = reduce(hcat, [collect(c) for c in cells])
    phase = ones(Int, length(cells))
    if fibre !== nothing
        lo, hi = fibre
        for (index, cell) in enumerate(cells)
            centre = sum(points[n] for n in cell) / length(cell)
            all(lo < centre[i] < hi for i in 1:in_plane) && (phase[index] = 2)
        end
    end
    return connectivity, phase
end

function rve(dim, n; fibre = (0.3, 0.7), cohesive = true)
    points, cells, phase = dim == 2 ? structured_2d(n; fibre) : structured_3d(n; fibre)
    return build_rve_mesh(points, cells, phase, 1, 2; cohesive = cohesive && fibre !== nothing)
end

function system_for(mesh; matrix = (E_MATRIX, NU_MATRIX, Inf, 0.0),
        fibre = (70000.0, 0.2, Inf, 0.0), interface = (K = 1.0e7, T = 30.0, G = 0.05),
        bc = "periodic")
    material(p) = PhaseMaterial(; youngs_modulus = p[1], poisson_ratio = p[2],
        yield_stress = p[3], hardening_modulus = p[4])
    materials = [phase == 2 ? material(fibre) : material(matrix) for phase in mesh.phase]
    law = n_cohesive(mesh) == 0 ? nothing :
        BilinearMixedMode(; stiffness = interface.K, normal_strength = interface.T,
            shear_strength = 1.5 * interface.T, mode_i_toughness = interface.G,
            mode_ii_toughness = 2 * interface.G, exponent = 1.45)
    return RVESystem(mesh, materials, law, bc, "nodal")
end

function uniaxial(kinematics, strain)
    active = Dict("plane_strain" => [1, 2, 6], "generalized_plane_strain" => [1, 2, 3, 6],
        "solid" => collect(1:6))[kinematics]
    return LoadPath([1 => strain], setdiff(1:6, active), [k => 0.0 for k in active if k != 1],
        false)
end

solve(sys, load; steps = 4) =
    solve_load_path(sys, load, NewtonSettings(; steps), initial_state(sys), E_MATRIX)

@testset "interface insertion" begin
    for dim in (2, 3)
        mesh = rve(dim, dim == 3 ? 6 : 10)
        fibre_nodes = Set(vec(mesh.cells[:, mesh.phase .== 2]))
        matrix_nodes = Set(vec(mesh.cells[:, mesh.phase .== 1]))
        @test isempty(intersect(fibre_nodes, matrix_nodes))
        bottom, top = mesh.cohesive[1:dim, :], mesh.cohesive[(dim + 1):end, :]
        @test all(mesh.points[bottom] .== mesh.points[top])
        @test issubset(Set(bottom), matrix_nodes) && issubset(Set(top), fibre_nodes)
        for e in axes(bottom, 2)  # normals point towards the fibre (centred at 0.5)
            to_centre = Vec{dim}(ntuple(i -> i <= 2 ? 0.5 : 0.0, dim)) -
                sum(mesh.points[n] for n in bottom[:, e]) / dim
            dim == 3 && (to_centre = Vec{3}((to_centre[1], to_centre[2], 0.0)))
            @test dot(facet_normal(mesh.points, bottom[:, e]), to_centre) > 0
        end
    end
end

@testset "homogeneous patch test with stiff interfaces" begin
    cases = [(2, "plane_strain", E_MATRIX / (1 - NU_MATRIX^2)),
        (2, "generalized_plane_strain", E_MATRIX), (3, "solid", E_MATRIX)]
    same = (E_MATRIX, NU_MATRIX, Inf, 0.0)
    for (dim, kinematics, expected) in cases
        mesh = rve(dim, dim == 3 ? 6 : 10)
        sys = system_for(mesh; matrix = same, fibre = same,
            interface = (K = 1.0e12, T = 1.0e9, G = 1.0e9))
        out = solve(sys, uniaxial(kinematics, 0.01); steps = 2)
        @test out.completed
        last = out.records[end]
        @test isapprox(last.macro_stress[1] / last.macro_strain[1], expected; rtol = 1.0e-6)
        free = kinematics == "plane_strain" ? [2, 4, 5, 6] : collect(2:6)
        @test maximum(abs, last.macro_stress[free]) < 1.0e-8 * expected
    end
end

@testset "homogeneous plastic RVE follows the uniaxial curve" begin
    sy, H = 60.0, 300.0
    for (dim, kinematics) in ((2, "generalized_plane_strain"), (3, "solid"))
        mesh = rve(dim, dim == 3 ? 4 : 6; fibre = nothing)
        same = (E_MATRIX, NU_MATRIX, sy, H)
        out = solve(system_for(mesh; matrix = same, fibre = same), uniaxial(kinematics, 0.04);
            steps = 8)
        @test out.completed
        for r in out.records[2:end]
            e = r.macro_strain[1]
            expected = e <= sy / E_MATRIX ? E_MATRIX * e :
                sy + E_MATRIX * H / (E_MATRIX + H) * (e - sy / E_MATRIX)
            @test isapprox(r.macro_stress[1], expected; rtol = 1.0e-8)
        end
    end
end

@testset "global tangent matches finite differences" begin
    mesh = rve(2, 12)
    sys = system_for(mesh; matrix = (E_MATRIX, NU_MATRIX, 40.0, 300.0),
        interface = (K = 1.0e6, T = 20.0, G = 0.05))
    load = uniaxial("generalized_plane_strain", 0.012)
    out = solve(sys, load; steps = 10)
    @test out.completed && out.records[end].max_damage > 0.1
    state = out.state
    free = [2, 3, 6]
    E = copy(state.macro_strain)
    E[1] += 5.0e-4  # beyond the converged state: plastic flow and damage growth are active
    target = zeros(6)
    ev = evaluate(sys, state.w, E, state.plastic, state.history, free, target, 0.1)
    direction = sin.(1:(n_reduced(sys) + 3))
    direction[1:n_reduced(sys)] .*= maximum(abs, state.w)
    direction[(n_reduced(sys) + 1):end] .*= 1.0e-3
    h = 1.0e-7
    function residual(sign)
        w = state.w .+ sign * h .* direction[1:n_reduced(sys)]
        Ei = copy(E)
        Ei[free] .+= sign * h .* direction[(n_reduced(sys) + 1):end]
        return evaluate(sys, w, Ei, state.plastic, state.history, free, target, 0.1;
            with_tangent = false).residual
    end
    fd = (residual(1.0) - residual(-1.0)) / (2h)
    @test norm(ev.matrix * direction - fd) / norm(fd) < 1.0e-6
end

@testset "interface debonding softens the response" begin
    mesh = rve(2, 16)
    load = uniaxial("generalized_plane_strain", 0.01)
    bonded = solve(system_for(mesh; interface = (K = 1.0e7, T = 1.0e6, G = 1.0e6)), load;
        steps = 10)
    weak = solve(system_for(mesh; interface = (K = 1.0e7, T = 15.0, G = 0.02)), load; steps = 20)
    @test weak.completed
    @test weak.records[end].max_damage > 0.99
    @test weak.records[end].macro_stress[1] < 0.9 * bonded.records[end].macro_stress[1]
end
