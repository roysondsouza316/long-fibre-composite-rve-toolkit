# Residual and consistent tangent of the RVE problem (Julia counterpart of
# rve2d.engines.tensormesh.nonlinear.assembly / cohesive).
#
# Unknowns: reduced displacement fluctuations `w` and the stress-controlled macro strain
# components `E_f`. Bulk strain `ε = sym(∇w) + E`; with `V` the RVE volume
#   fluctuation residual  r_w = Σ_e ∫ B_e^T σ + Σ_c f_c
#   macro residual        r_E = Σ_e V_e σ_e - V σ_target  (Hill-Mandel average)
# and the bordered tangent [K_ww K_wE; K_Ew K_EE]. Cohesive separations involve only `w`
# because the duplicated interface nodes share coordinates.

const VOIGT_INDEX = ((1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2))

"Macro strain tensor from Voigt components `(xx, yy, zz, yz, xz, xy)` with engineering shear."
function macro_tensor(E::AbstractVector)
    return SymmetricTensor{2, 3}((E[1], E[6] / 2, E[5] / 2, E[2], E[4] / 2, E[3]))
end

"Basis tensor `∂ε/∂E_k` of Voigt component `k` (engineering shear)."
function voigt_basis(k::Int)
    i, j = VOIGT_INDEX[k]
    return SymmetricTensor{2, 3}((a, b) -> ((a, b) == (i, j) || (a, b) == (j, i)) ?
        (i == j ? 1.0 : 0.5) : 0.0)
end

voigt(σ::SymmetricTensor{2, 3}) = SVector{6}(ntuple(k -> σ[VOIGT_INDEX[k]...], 6))

"3D symmetric gradient of a shape function from its in-plane symmetric gradient."
embed(ε::SymmetricTensor{2, 3}) = ε
embed(ε::SymmetricTensor{2, 2}) =
    SymmetricTensor{2, 3}((a, b) -> a <= 2 && b <= 2 ? ε[a, b] : 0.0)

"""
    CohesiveGeometry

Reference frames (`local = R * global`, rows `[n, t(, t2)]`, `n` into the fibre), quadrature
weights times facet measure (`n_q × n_cohesive`) and shape values `N[q, a]` of the cohesive
elements. Nodal (Newton-Cotes) integration puts the points at the node pairs.
"""
struct CohesiveGeometry{dim, NQ, M}
    rotation::Vector{SMatrix{dim, dim, Float64, M}}
    weights::Matrix{Float64}
    shape::SMatrix{NQ, dim, Float64, M}  # every rule here has NQ == dim points
end

function cohesive_geometry(mesh::RVEMesh{dim}, integration::AbstractString) where {dim}
    integration in ("nodal", "gauss") ||
        error("cohesive integration must be \"nodal\" or \"gauss\", got \"$integration\"")
    n = n_cohesive(mesh)
    rotation = Vector{SMatrix{dim, dim, Float64, dim * dim}}(undef, n)
    measure = zeros(n)
    for e in 1:n
        p = ntuple(a -> SVector{dim}(Tuple(mesh.points[mesh.cohesive[a, e]])), dim)
        if dim == 2
            edge = p[2] - p[1]
            measure[e] = norm(edge)
            t = edge / measure[e]
            rotation[e] = SMatrix{2, 2}(-t[2], t[1], t[1], t[2])
        else
            a, b = p[2] - p[1], p[3] - p[1]
            normal = cross(a, b)
            measure[e] = norm(normal) / 2
            nvec = normal / norm(normal)
            t1 = a / norm(a)
            t2 = cross(nvec, t1)
            rotation[e] = SMatrix{3, 3}(nvec[1], t1[1], t2[1], nvec[2], t1[2], t2[2],
                nvec[3], t1[3], t2[3])
        end
    end
    shape, w = if integration == "nodal"
        SMatrix{dim, dim}(ntuple(k -> (k - 1) % dim == (k - 1) ÷ dim ? 1.0 : 0.0, dim^2)),
        fill(1 / dim, dim)
    elseif dim == 2
        g = 1 / sqrt(3)
        SMatrix{2, 2}((1 - g) / 2, (1 + g) / 2, (1 + g) / 2, (1 - g) / 2), [0.5, 0.5]
    else
        SMatrix{3, 3}(2 / 3, 1 / 6, 1 / 6, 1 / 6, 2 / 3, 1 / 6, 1 / 6, 1 / 6, 2 / 3), fill(1 / 3, 3)
    end
    return CohesiveGeometry(rotation, w * measure', shape)
end

"""
    RVESystem

Mesh, Ferrite DOF handler, precomputed element data, materials, interface law and the
reduced-DOF map; `evaluate` returns residual and tangent for given unknowns.
"""
struct RVESystem{dim, L, N, NQ, M, G <: Grid, H <: DofHandler}
    mesh::RVEMesh{dim}
    grid::G
    dh::H
    node_dofs::Matrix{Int}
    cell_dofs::Matrix{Int}
    strain_basis::Matrix{SymmetricTensor{2, 3, Float64, 6}}
    volumes::Vector{Float64}
    volume::Float64
    materials::Vector{PhaseMaterial}
    law::L
    cohesive::CohesiveGeometry{dim, NQ, M}
    dofmap::DofMap
end

n_reduced(sys::RVESystem) = sys.dofmap.n_reduced
history_length(::RVESystem{dim, L, N}) where {dim, L, N} = N

function ferrite_grid(mesh::RVEMesh{2})
    cells = [Triangle(Tuple(mesh.cells[:, c])) for c in 1:n_cells(mesh)]
    return Grid(cells, [Node(Tuple(p)) for p in mesh.points])
end

function ferrite_grid(mesh::RVEMesh{3})
    cells = [Tetrahedron(Tuple(mesh.cells[:, c])) for c in 1:n_cells(mesh)]
    return Grid(cells, [Node(Tuple(p)) for p in mesh.points])
end

reference_shape(::RVEMesh{2}) = RefTriangle
reference_shape(::RVEMesh{3}) = RefTetrahedron

function RVESystem(mesh::RVEMesh{dim}, materials::Vector{PhaseMaterial}, law,
        boundary_condition::AbstractString, integration::AbstractString) where {dim}
    grid = ferrite_grid(mesh)
    shape = reference_shape(mesh)
    ip = Lagrange{shape, 1}()^dim
    dh = DofHandler(grid)
    add!(dh, :u, ip)
    close!(dh)
    cellvalues = CellValues(QuadratureRule{shape}(1), ip)  # linear simplices: constant strain

    nb = getnbasefunctions(cellvalues)
    cell_dofs = zeros(Int, nb, n_cells(mesh))
    node_dofs = zeros(Int, dim, n_nodes(mesh))
    strain_basis = Matrix{SymmetricTensor{2, 3, Float64, 6}}(undef, nb, n_cells(mesh))
    volumes = zeros(n_cells(mesh))
    for cell in 1:n_cells(mesh)
        dofs = celldofs(dh, cell)
        cell_dofs[:, cell] .= dofs
        # Vector Lagrange DOFs are node-major: (node 1: x, y(, z)), (node 2: ...), ...
        for (a, node) in enumerate(mesh.cells[:, cell]), c in 1:dim
            dof = dofs[(a - 1) * dim + c]
            node_dofs[c, node] == 0 || node_dofs[c, node] == dof ||
                error("unexpected Ferrite DOF layout")
            node_dofs[c, node] = dof
        end
        reinit!(cellvalues, getcoordinates(grid, cell))
        volumes[cell] = getdetJdV(cellvalues, 1)
        for i in 1:nb
            strain_basis[i, cell] = embed(shape_symmetric_gradient(cellvalues, 1, i))
        end
    end
    all(>(0), node_dofs) || error("mesh has nodes without degrees of freedom")
    dofmap = build_dof_map(mesh, node_dofs, ndofs(dh), boundary_condition)
    cohesive = cohesive_geometry(mesh, integration)
    N = law === nothing ? 1 : state_length(law)
    NQ = size(cohesive.weights, 1)
    return RVESystem{dim, typeof(law), N, NQ, dim * dim, typeof(grid), typeof(dh)}(
        mesh, grid, dh, node_dofs, cell_dofs, strain_basis, volumes, sum(volumes), materials,
        law, cohesive, dofmap,
    )
end

"Full fluctuation vector (Ferrite DOF order) from the reduced unknowns."
function expand(sys::RVESystem, w::AbstractVector)
    full = zeros(length(sys.dofmap.full_to_reduced))
    for (dof, r) in enumerate(sys.dofmap.full_to_reduced)
        r > 0 && (full[dof] = w[r])
    end
    return full
end

function reduce_vector(sys::RVESystem, full::AbstractVector)
    out = zeros(n_reduced(sys))
    for (dof, r) in enumerate(sys.dofmap.full_to_reduced)
        r > 0 && (out[r] += full[dof])
    end
    return out
end

"Everything a Newton iteration needs, reduced to the free unknowns."
struct Evaluation{dim, N}
    residual::Vector{Float64}
    matrix::Union{Nothing, SparseMatrixCSC{Float64, Int}}
    macro_stress::SVector{6, Float64}
    force_scale::Float64
    stress::Vector{SymmetricTensor{2, 3, Float64, 6}}
    plastic::Vector{PlasticState}
    history::Matrix{SVector{N, Float64}}
    damage::Matrix{Float64}
    delta_local::Matrix{SVector{dim, Float64}}
end

# COO triplets of the reduced matrix; entries touching a fixed DOF are dropped.
struct Triplets
    rows::Vector{Int}
    cols::Vector{Int}
    vals::Vector{Float64}
end
Triplets() = Triplets(Int[], Int[], Float64[])

@inline function add_entry!(t::Triplets, r::Int, c::Int, v::Float64)
    if r > 0 && c > 0
        push!(t.rows, r)
        push!(t.cols, c)
        push!(t.vals, v)
    end
    return t
end

"""
    evaluate(sys, w, E, plastic, history, free, target, Δt; with_tangent = true)

Residual (and bordered tangent) for reduced fluctuations `w`, macro strain `E` (Voigt),
committed plastic states and interface histories; `free` lists the stress-controlled macro
components and `target` the macro stress they must reach. `Δt` is the pseudo-time increment
(it sets the viscous relaxation of the interface law).
"""
function evaluate(sys::RVESystem{dim, L, N, NQ}, w::AbstractVector, E::AbstractVector,
        plastic::Vector{PlasticState}, history::Matrix{SVector{N, Float64}},
        free::Vector{Int}, target::AbstractVector, Δt::Float64;
        with_tangent::Bool = true) where {dim, L, N, NQ}
    f2r = sys.dofmap.full_to_reduced
    w_full = expand(sys, w)
    n_full = length(w_full)
    residual_full = zeros(n_full)
    triplets = Triplets()
    nfree = length(free)
    basis = [voigt_basis(k) for k in free]
    k_we_full = zeros(n_full, nfree)
    k_ee = zeros(nfree, nfree)

    # Bulk: J2 plasticity on constant-strain simplices.
    Emacro = macro_tensor(E)
    n_cell = n_cells(sys.mesh)
    stress = Vector{SymmetricTensor{2, 3, Float64, 6}}(undef, n_cell)
    new_plastic = Vector{PlasticState}(undef, n_cell)
    macro_sum = zero(SymmetricTensor{2, 3, Float64})
    bulk_sq = 0.0
    nb = size(sys.cell_dofs, 1)
    if with_tangent
        n_entries = nb^2 * n_cell + (2 * dim^2)^2 * n_cohesive(sys.mesh)
        foreach(v -> sizehint!(v, n_entries), (triplets.rows, triplets.cols, triplets.vals))
    end
    c_eps = Vector{SymmetricTensor{2, 3, Float64, 6}}(undef, nb)
    for cell in 1:n_cell
        strain = Emacro
        for i in 1:nb
            strain += w_full[sys.cell_dofs[i, cell]] * sys.strain_basis[i, cell]
        end
        σ, C, state, _ = j2_update(strain, plastic[cell], sys.materials[cell])
        stress[cell] = σ
        new_plastic[cell] = state
        V = sys.volumes[cell]
        macro_sum += V * σ
        for i in 1:nb
            f = V * (sys.strain_basis[i, cell] ⊡ σ)
            residual_full[sys.cell_dofs[i, cell]] += f
            bulk_sq += f^2
        end
        with_tangent || continue
        for j in 1:nb
            c_eps[j] = C ⊡ sys.strain_basis[j, cell]
        end
        for i in 1:nb
            ri = f2r[sys.cell_dofs[i, cell]]
            ri == 0 && continue
            for j in 1:nb
                add_entry!(triplets, ri, f2r[sys.cell_dofs[j, cell]],
                    V * (sys.strain_basis[i, cell] ⊡ c_eps[j]))
            end
        end
        for (k, Bk) in enumerate(basis)
            CB = C ⊡ Bk
            for i in 1:nb
                k_we_full[sys.cell_dofs[i, cell], k] += V * (sys.strain_basis[i, cell] ⊡ CB)
            end
            for (l, Bl) in enumerate(basis)
                k_ee[l, k] += V * (Bl ⊡ CB)
            end
        end
    end

    # Cohesive interfaces.
    n_coh = n_cohesive(sys.mesh)
    new_history = similar(history)
    damage = zeros(NQ, n_coh)
    delta_local = Matrix{SVector{dim, Float64}}(undef, NQ, n_coh)
    coh_sq = 0.0
    if n_coh > 0
        coh_sq = assemble_cohesive!(residual_full, triplets, new_history, damage, delta_local,
            sys, w_full, history, Δt, with_tangent)
    end

    macro_voigt = voigt(macro_sum)
    residual = vcat(reduce_vector(sys, residual_full),
        [macro_voigt[k] - sys.volume * target[k] for k in free])
    matrix = nothing
    if with_tangent
        nr = n_reduced(sys)
        k_we = zeros(nr, nfree)
        for (dof, r) in enumerate(f2r)
            r > 0 && (k_we[r, :] .+= view(k_we_full, dof, :))
        end
        for k in 1:nfree, r in 1:nr
            add_entry!(triplets, r, nr + k, k_we[r, k])
            add_entry!(triplets, nr + k, r, k_we[r, k])
        end
        for k in 1:nfree, l in 1:nfree
            add_entry!(triplets, nr + l, nr + k, k_ee[l, k])
        end
        matrix = sparse(triplets.rows, triplets.cols, triplets.vals, nr + nfree, nr + nfree)
    end
    return Evaluation{dim, N}(residual, matrix, macro_voigt / sys.volume, sqrt(bulk_sq + coh_sq),
        stress, new_plastic, new_history, damage, delta_local)
end

function nodal_displacement(sys::RVESystem{dim}, w_full, node) where {dim}
    return SVector{dim}(ntuple(c -> w_full[sys.node_dofs[c, node]], dim))
end

function assemble_cohesive!(residual_full, triplets, new_history, damage, delta_local,
        sys::RVESystem{dim, L, N, NQ}, w_full, history, Δt, with_tangent) where {dim, L, N, NQ}
    geometry = sys.cohesive
    f2r = sys.dofmap.full_to_reduced
    mesh = sys.mesh
    nd = dim * dim  # DOFs per element side (dim nodes × dim components)
    sq = 0.0
    dofs = zeros(Int, 2nd)
    for e in 1:n_cohesive(mesh)
        R = geometry.rotation[e]
        bottom = ntuple(a -> nodal_displacement(sys, w_full, mesh.cohesive[a, e]), dim)
        top = ntuple(a -> nodal_displacement(sys, w_full, mesh.cohesive[dim + a, e]), dim)
        f_top = zero(MMatrix{dim, dim, Float64})           # component × node
        k_tt = zero(MMatrix{nd, nd, Float64})
        for q in 1:NQ
            jump = sum(geometry.shape[q, a] * (top[a] - bottom[a]) for a in 1:dim)
            δ = R * jump
            if with_tangent
                t, D, state, d = traction_tangent(sys.law, δ, history[q, e]; Δt)
            else
                t, state, d = traction(sys.law, δ, history[q, e]; Δt)
            end
            new_history[q, e] = state
            damage[q, e] = d
            delta_local[q, e] = δ
            weight = geometry.weights[q, e]
            t_global = R' * t
            for a in 1:dim
                f_top[:, a] .+= (weight * geometry.shape[q, a]) .* t_global
            end
            if with_tangent
                D_global = R' * D * R
                for a in 1:dim, b in 1:dim
                    scale = weight * geometry.shape[q, a] * geometry.shape[q, b]
                    for i in 1:dim, k in 1:dim
                        k_tt[(a - 1) * dim + i, (b - 1) * dim + k] += scale * D_global[i, k]
                    end
                end
            end
        end
        for side in 1:2, a in 1:dim, c in 1:dim
            dofs[(side - 1) * nd + (a - 1) * dim + c] =
                sys.node_dofs[c, mesh.cohesive[(side - 1) * dim + a, e]]
        end
        for a in 1:dim, c in 1:dim
            f = f_top[c, a]
            residual_full[dofs[(a - 1) * dim + c]] -= f
            residual_full[dofs[nd + (a - 1) * dim + c]] += f
            sq += 2 * f^2
        end
        with_tangent || continue
        for i in 1:2nd
            ri = f2r[dofs[i]]
            ri == 0 && continue
            for j in 1:2nd
                sign = (i <= nd) == (j <= nd) ? 1.0 : -1.0
                add_entry!(triplets, ri, f2r[dofs[j]],
                    sign * k_tt[(i - 1) % nd + 1, (j - 1) % nd + 1])
            end
        end
    end
    return sq
end
