# Incremental Newton solution under mixed macro control (Julia counterpart of
# rve2d.nonlinear.solver: same error measures, line search, increment cutting and growth).

"""
    LoadPath

Each macro strain component (Voigt index 1-6) is prescribed (`prescribed[k]` is its value at
load factor 1), held at zero (`fixed_zero`), or stress-controlled (`stress_controlled[k]` is
the target macro stress at load factor 1). Pseudo-time runs to 1, or to 2 with `unload`.
"""
struct LoadPath
    prescribed::Vector{Pair{Int, Float64}}
    fixed_zero::Vector{Int}
    stress_controlled::Vector{Pair{Int, Float64}}
    unload::Bool
end

end_time(load::LoadPath) = load.unload ? 2.0 : 1.0
load_factor(::LoadPath, t) = t <= 1 ? t : 2 - t
free_components(load::LoadPath) = sort!(first.(load.stress_controlled))

Base.@kwdef struct NewtonSettings
    max_iterations::Int = 25
    tolerance::Float64 = 1.0e-8
    steps::Int = 40
    max_cuts::Int = 8
    growth::Float64 = 1.5
    line_search::Bool = true
end

mutable struct SolverState{N}
    w::Vector{Float64}
    macro_strain::Vector{Float64}
    plastic::Vector{PlasticState}
    history::Matrix{SVector{N, Float64}}
end

function initial_state(sys::RVESystem{dim, L, N, NQ}) where {dim, L, N, NQ}
    return SolverState{N}(zeros(n_reduced(sys)), zeros(6),
        [PlasticState() for _ in 1:n_cells(sys.mesh)],
        fill(zero(SVector{N, Float64}), NQ, n_cohesive(sys.mesh)))
end

struct StepRecord
    time::Float64
    load_factor::Float64
    macro_strain::Vector{Float64}
    macro_stress::Vector{Float64}
    iterations::Int
    max_damage::Float64
    damaged_fraction::Float64
    mean_damage::Float64
    yielded_fraction::Float64
    mean_eqps_matrix::Float64
    mean_eqps_fibre::Float64
    work_density::Float64
end

struct SolveOutcome{S, E}
    records::Vector{StepRecord}
    state::S
    last_evaluation::E
    completed::Bool
    reached_time::Float64
    total_iterations::Int
    cuts::Int
    messages::Vector{String}
end

"""
    solve_load_path(sys, load, settings, state, stress_scale; on_step = nothing)

Follow the load path with adaptive increments; `stress_scale` (the matrix Young's modulus)
sets the floors of the convergence measures. History variables are committed only after a
converged increment; failed increments are halved, easy ones grow back by `growth`.
"""
function solve_load_path(sys::RVESystem, load::LoadPath, settings::NewtonSettings,
        state::SolverState, stress_scale::Float64; on_step = nothing)
    free = free_components(load)
    target_final = zeros(6)
    for (k, value) in load.stress_controlled
        target_final[k] = value
    end
    dim = sys.mesh isa RVEMesh{2} ? 2 : 3
    force_floor = 1.0e-6 * stress_scale * sys.volume^((dim - 1) / dim)
    stress_floor = 1.0e-6 * stress_scale

    first_evaluation = evaluate_state(sys, state, free, zeros(6), 1.0; with_tangent = false)
    records = [record(sys, 0.0, 0.0, state, first_evaluation, 0, 0.0)]
    last_evaluation = first_evaluation
    t = 0.0
    dt0 = end_time(load) / settings.steps
    dt = dt0
    min_dt = dt0 / 2^settings.max_cuts
    total_iterations = cuts = 0
    messages = String[]
    work = 0.0
    while t < end_time(load) - 1.0e-12
        dt = min(dt, end_time(load) - t)
        t_new = t + dt
        λ = load_factor(load, t_new)
        trial = typeof(state)(copy(state.w), copy(state.macro_strain), state.plastic,
            state.history)
        for (k, value) in load.prescribed
            trial.macro_strain[k] = λ * value
        end
        for k in load.fixed_zero
            trial.macro_strain[k] = 0.0
        end
        converged, evaluation, iterations = newton!(sys, trial, free, λ * target_final, dt,
            settings, force_floor, stress_floor)
        total_iterations += iterations
        if !converged
            cuts += 1
            dt /= 2
            if dt < min_dt
                push!(messages, @sprintf(
                    "Stopped at t=%.6g: no convergence after %d increment cuts.", t,
                    settings.max_cuts))
                break
            end
            continue
        end
        committed = typeof(state)(trial.w, trial.macro_strain, evaluation.plastic,
            evaluation.history)
        work += work_increment(records[end], committed, evaluation)
        state, t, last_evaluation = committed, t_new, evaluation
        step = record(sys, t, λ, state, evaluation, iterations, work)
        push!(records, step)
        on_step === nothing || on_step(step, state, evaluation)
        iterations <= 4 && (dt = min(dt * settings.growth, dt0))
    end
    return SolveOutcome(records, state, last_evaluation, t >= end_time(load) - 1.0e-12, t,
        total_iterations, cuts, messages)
end

function evaluate_state(sys, state, free, target, dt; with_tangent = true)
    return evaluate(sys, state.w, state.macro_strain, state.plastic, state.history, free,
        target, dt; with_tangent)
end

function newton!(sys, trial, free, target, dt, settings, force_floor, stress_floor)
    nr = n_reduced(sys)
    evaluation = evaluate_state(sys, trial, free, target, dt)
    history = Float64[]
    for iteration in 1:settings.max_iterations
        error = residual_error(sys, evaluation, free, force_floor, stress_floor)
        isfinite(error) || return false, evaluation, iteration
        error <= settings.tolerance && return true, evaluation, iteration - 1
        push!(history, error)
        # Give up early on a stalled or diverging iteration: cutting the increment is cheaper.
        length(history) >= 8 && error > 0.5 * history[end - 4] &&
            return false, evaluation, iteration
        step = try
            lu(evaluation.matrix) \ (-evaluation.residual)
        catch exception
            exception isa LinearAlgebra.LAPACKException ||
                exception isa LinearAlgebra.SingularException || rethrow()
            return false, evaluation, iteration
        end
        all(isfinite, step) || return false, evaluation, iteration
        α = 1.0
        base_w, base_e = copy(trial.w), copy(trial.macro_strain)
        candidate = evaluation
        for _ in 1:(settings.line_search ? 4 : 1)
            trial.w = base_w .+ α .* view(step, 1:nr)
            trial.macro_strain = copy(base_e)
            for (j, k) in enumerate(free)
                trial.macro_strain[k] = base_e[k] + α * step[nr + j]
            end
            candidate = evaluate_state(sys, trial, free, target, dt)
            candidate_error = residual_error(sys, candidate, free, force_floor, stress_floor)
            isfinite(candidate_error) && candidate_error < error && break
            α /= 2
        end
        evaluation = candidate
    end
    final_error = residual_error(sys, evaluation, free, force_floor, stress_floor)
    return final_error <= settings.tolerance, evaluation, settings.max_iterations
end

function residual_error(sys, evaluation, free, force_floor, stress_floor)
    nr = n_reduced(sys)
    force_error = norm(view(evaluation.residual, 1:nr)) / max(evaluation.force_scale, force_floor)
    isempty(free) && return force_error
    stress_ref = max(maximum(abs, evaluation.macro_stress), stress_floor) * sys.volume
    stress_error = maximum(abs, view(evaluation.residual, (nr + 1):length(evaluation.residual))) /
        stress_ref
    return max(force_error, stress_error)
end

function work_increment(previous::StepRecord, state, evaluation)
    return sum(0.5 * (previous.macro_stress[i] + evaluation.macro_stress[i]) *
               (state.macro_strain[i] - previous.macro_strain[i]) for i in 1:6)
end

function record(sys, t, λ, state, evaluation, iterations, work)
    volumes = sys.volumes
    matrix = sys.mesh.phase .== sys.mesh.matrix_phase_id
    eqps = [p.equivalent_plastic_strain for p in state.plastic]
    function phase_mean(mask)
        weight = sum(volumes[mask])
        return weight > 0 ? sum(eqps[mask] .* volumes[mask]) / weight : 0.0
    end
    max_damage = damaged = mean_damage = 0.0
    if n_cohesive(sys.mesh) > 0
        weights = sys.cohesive.weights
        damage = evaluation.damage
        max_damage = maximum(damage)
        damaged = sum(weights .* (damage .>= 0.99)) / sum(weights)
        mean_damage = sum(weights .* damage) / sum(weights)
    end
    return StepRecord(t, λ, copy(state.macro_strain), collect(evaluation.macro_stress),
        iterations, max_damage, damaged, mean_damage, sum(volumes[eqps .> 0]) / sum(volumes),
        phase_mean(matrix), phase_mean(.!matrix), work)
end
