# Linear-elastic, first-order homogenization: the effective stiffness of the RVE from unit
# macro strains, with periodic (or affine) boundary fluctuations. Same formulation as the
# Python engine (rve2d.engines.python.homogenization).
#
# Unknowns are the nodal fluctuations `w`; the strain of an element is `B w + E` for the macro
# strain `E` (Voigt, engineering shear). Equilibrium `K w = -F E` with `K = Σ V Bᵀ D B` and
# `F = Σ V Bᵀ D` is factorised once and solved for every unit macro strain; column `j` of the
# effective stiffness is the volume-averaged stress of load case `j`.

const VOIGT_NAMES = ("xx", "yy", "zz", "yz", "xz", "xy")

"""
    HomogenizationKinematics

Displacement components per node and active Voigt strain components (indices into
`(xx, yy, zz, yz, xz, xy)`) of each kinematic model. Generalized plane strain solves the 2D
cross-section with all three displacement components, so the out-of-plane strain and the two
longitudinal shears are included and the full 6x6 stiffness of a unidirectional ply (fibres
along z) is obtained.
"""
const HOMOGENIZATION_KINEMATICS = Dict(
    "plane_stress" => (components = 2, voigt = [1, 2, 6]),
    "plane_strain" => (components = 2, voigt = [1, 2, 6]),
    "generalized_plane_strain" => (components = 3, voigt = [1, 2, 3, 4, 5, 6]),
    "solid" => (components = 3, voigt = [1, 2, 3, 4, 5, 6]),
)

"Faces of the RVE box and their outward normals."
face_normals(::Val{2}) = (left = (1, -1), right = (1, 1), bottom = (2, -1), top = (2, 1))
face_normals(::Val{3}) = (left = (1, -1), right = (1, 1), front = (2, -1), back = (2, 1),
    bottom = (3, -1), top = (3, 1))

"""
Strain-displacement matrix (all six Voigt rows) of a linear simplex for shape-function
gradients `grads` (one `Vec` per node) and `ncomp` displacement components per node.
"""
function strain_matrix(grads, ncomp::Int)
    nnodes = length(grads)
    B = zeros(6, nnodes * ncomp)
    for (a, g) in enumerate(grads)
        gx, gy = g[1], g[2]
        gz = length(g) == 3 ? g[3] : 0.0
        col = (a - 1) * ncomp
        # u_x
        B[1, col + 1] = gx
        B[6, col + 1] = gy
        B[5, col + 1] = gz
        # u_y
        B[2, col + 2] = gy
        B[6, col + 2] = gx
        B[4, col + 2] = gz
        if ncomp == 3  # u_z
            B[3, col + 3] = gz
            B[4, col + 3] = gy
            B[5, col + 3] = gx
        end
    end
    return B
end

"""
    homogenize(mesh, stiffness, kinematics, boundary_condition) -> NamedTuple

Effective stiffness of `mesh` with per-cell constitutive matrices `stiffness[cell]` (in the
active Voigt basis of `kinematics`). Returns the stiffness, the per-case average stresses,
nodal fluctuations, cell stresses, face-averaged tractions and bookkeeping.
"""
function homogenize(mesh::RVEMesh{dim}, stiffness::Vector{Matrix{Float64}},
        kinematics::AbstractString, boundary_condition::AbstractString) where {dim}
    haskey(HOMOGENIZATION_KINEMATICS, kinematics) ||
        error("unknown kinematics \"$kinematics\"")
    model = HOMOGENIZATION_KINEMATICS[kinematics]
    dim == 3 && kinematics != "solid" && error("3D meshes need kinematics \"solid\"")
    dim == 2 && kinematics == "solid" && error("2D meshes cannot use kinematics \"solid\"")
    ncomp, voigt = model.components, model.voigt
    ne = length(voigt)

    grid = ferrite_grid(mesh)
    shape = reference_shape(mesh)
    dh = DofHandler(grid)
    add!(dh, :u, Lagrange{shape, 1}()^ncomp)
    close!(dh)
    scalar_values = CellValues(QuadratureRule{shape}(1), Lagrange{shape, 1}())

    nnodes_cell = dim + 1
    node_dofs = zeros(Int, ncomp, n_nodes(mesh))
    cell_dofs = zeros(Int, nnodes_cell * ncomp, n_cells(mesh))
    strain = Vector{Matrix{Float64}}(undef, n_cells(mesh))
    volumes = zeros(n_cells(mesh))
    for cell in 1:n_cells(mesh)
        dofs = celldofs(dh, cell)
        cell_dofs[:, cell] .= dofs
        for (a, node) in enumerate(view(mesh.cells, :, cell)), c in 1:ncomp
            node_dofs[c, node] = dofs[(a - 1) * ncomp + c]
        end
        reinit!(scalar_values, getcoordinates(grid, cell))
        volumes[cell] = getdetJdV(scalar_values, 1)
        grads = [shape_gradient(scalar_values, 1, a) for a in 1:nnodes_cell]
        strain[cell] = strain_matrix(grads, ncomp)[voigt, :]
    end
    dofmap = build_dof_map(mesh, node_dofs, ndofs(dh), boundary_condition)
    f2r = dofmap.full_to_reduced
    nr = dofmap.n_reduced

    rows, cols, vals = Int[], Int[], Float64[]
    load = zeros(nr, ne)
    for cell in 1:n_cells(mesh)
        B, D, V = strain[cell], stiffness[cell], volumes[cell]
        DB = D * B
        Ke = V * (B' * DB)
        Fe = V * (B' * D)
        dofs = view(cell_dofs, :, cell)
        for i in eachindex(dofs)
            ri = f2r[dofs[i]]
            ri == 0 && continue
            for j in eachindex(dofs)
                rj = f2r[dofs[j]]
                rj == 0 && continue
                push!(rows, ri)
                push!(cols, rj)
                push!(vals, Ke[i, j])
            end
            load[ri, :] .+= view(Fe, i, :)
        end
    end
    K = sparse(rows, cols, vals, nr, nr)
    factor = try
        cholesky(Symmetric(K))
    catch exception
        exception isa LinearAlgebra.PosDefException || rethrow()
        error("the RVE stiffness matrix is singular: check the boundary condition and the mesh")
    end
    reduced = factor \ (-load)  # one factorisation, one solve per unit macro strain

    total_volume = sum(volumes)
    fluctuations = [expand_reduced(f2r, view(reduced, :, j)) for j in 1:ne]
    cell_stress = [Matrix{Float64}(undef, ne, n_cells(mesh)) for _ in 1:ne]
    average = zeros(ne, ne)
    for j in 1:ne, cell in 1:n_cells(mesh)
        e = strain[cell] * fluctuations[j][view(cell_dofs, :, cell)]
        e[j] += 1.0
        σ = stiffness[cell] * e
        cell_stress[j][:, cell] .= σ
        average[:, j] .+= volumes[cell] .* σ
    end
    average ./= total_volume
    fibre_volume = sum(volumes[mesh.phase .== mesh.fibre_phase_id])
    return (
        grid = grid, dh = dh, node_dofs = node_dofs, voigt = voigt, components = ncomp,
        stiffness = average, fluctuations = fluctuations, cell_stress = cell_stress,
        tractions = face_tractions(mesh, cell_stress, voigt),
        volume = total_volume, fibre_volume_fraction = fibre_volume / total_volume,
        unknowns = nr, anchor = dofmap.anchor,
    )
end

function expand_reduced(f2r::Vector{Int}, w::AbstractVector)
    full = zeros(length(f2r))
    for (dof, r) in enumerate(f2r)
        r > 0 && (full[dof] = w[r])
    end
    return full
end

"Full 3x3 stress tensor from the active Voigt components."
function stress_tensor(σ::AbstractVector, voigt::Vector{Int})
    full = zeros(6)
    full[voigt] .= σ
    return [full[1] full[6] full[5]; full[6] full[2] full[4]; full[5] full[4] full[3]]
end

"Strain tensor from Voigt components with engineering shear."
strain_tensor(E::AbstractVector) =
    [E[1] E[6]/2 E[5]/2; E[6]/2 E[2] E[4]/2; E[5]/2 E[4]/2 E[3]]

"""
Facet-averaged traction `σ n` on every face of the RVE box, for every load case: the stress
of the cell owning each boundary facet, weighted by the facet measure.
"""
function face_tractions(mesh::RVEMesh{dim}, cell_stress, voigt) where {dim}
    tol = boundary_tolerance(mesh)
    faces = face_normals(Val(dim))
    ncomp_traction = length(voigt) == 6 ? 3 : 2
    result = [Dict{String, Vector{Float64}}() for _ in cell_stress]
    for (name, (axis, sign)) in pairs(faces)
        plane = sign < 0 ? mesh.lower[axis] : mesh.upper[axis]
        normal = zeros(3)
        normal[axis] = sign
        sums = [zeros(3) for _ in cell_stress]
        measure = 0.0
        for cell in 1:n_cells(mesh), facet in cell_facets(Val(dim))
            nodes = [mesh.cells[a, cell] for a in facet]
            all(abs(mesh.points[n][axis] - plane) < tol for n in nodes) || continue
            m = norm(facet_normal(mesh.points, nodes))
            dim == 3 && (m /= 2)
            measure += m
            for (j, σ) in enumerate(cell_stress)
                sums[j] .+= m .* (stress_tensor(view(σ, :, cell), voigt) * normal)
            end
        end
        for j in eachindex(cell_stress)
            result[j][String(name)] = measure > 0 ? sums[j][1:ncomp_traction] ./ measure :
                fill(NaN, ncomp_traction)
        end
    end
    return result
end

"Write `homogenization.vtu`: per-case total displacement and cell stresses, and the phases."
function write_homogenization_vtk(path::AbstractString, mesh::RVEMesh{dim}, result) where {dim}
    names = [VOIGT_NAMES[k] for k in result.voigt]
    VTKGridFile(path, result.grid) do vtk
        for (j, w) in enumerate(result.fluctuations)
            E = zeros(6)
            E[result.voigt[j]] = 1.0
            macro_gradient = strain_tensor(E)
            u = zeros(3, n_nodes(mesh))
            for node in 1:n_nodes(mesh)
                x = zeros(3)
                x[1:dim] .= mesh.points[node]
                affine = macro_gradient * x
                for c in 1:result.components
                    u[c, node] = affine[c] + w[result.node_dofs[c, node]]
                end
            end
            write_node_data(vtk, u, "u_case_$(j)_$(names[j])")
            WriteVTK.vtk_cell_data(vtk.vtk, result.cell_stress[j],
                "stress_case_$(j)_$(names[j])"; component_names = ["s$(n)" for n in names])
        end
        write_cell_data(vtk, Float64.(mesh.phase), "phase")
    end
    return path * ".vtu"
end

"Run the homogenization described by an input table (written by the rve2d bridge)."
function run_homogenization(input::AbstractDict; verbose::Bool = true)
    started = time()
    dim = Int(input["dimension"])
    output_dir = input["output_dir"]
    mkpath(output_dir)
    matrix_id = Int(get(input, "matrix_phase_id", 1))
    fibre_id = Int(get(input, "fibre_phase_id", 2))
    points, cells, tags = read_rve_arrays(input["nodes"], input["cells"], input["cell_tags"], dim)
    mesh = build_rve_mesh(points, cells, tags, matrix_id, fibre_id; cohesive = false)
    as_matrix(rows) = Matrix{Float64}(permutedims(reduce(hcat, rows)))
    matrix_d = as_matrix(input["matrix_stiffness"])
    fibre_d = as_matrix(input["fibre_stiffness"])
    stiffness = [phase == fibre_id ? fibre_d : matrix_d for phase in mesh.phase]
    kinematics = input["kinematics"]
    result = homogenize(mesh, stiffness, kinematics, input["boundary_condition"])
    vtk_path = ""
    if Bool(get(input, "write_vtk", true))
        vtk_path = write_homogenization_vtk(joinpath(output_dir, "homogenization"), mesh, result)
    end
    names = [VOIGT_NAMES[k] for k in result.voigt]
    output = Dict{String, Any}(
        "engine" => "julia",
        "julia_version" => string(VERSION),
        "kinematics" => kinematics,
        "boundary_condition" => input["boundary_condition"],
        "voigt_components" => names,
        "stiffness" => [result.stiffness[i, :] for i in axes(result.stiffness, 1)],
        "average_stresses" => [result.stiffness[:, j] for j in axes(result.stiffness, 2)],
        "traction_faces" => collect(keys(result.tractions[1])),
        "tractions" => [[t[face] for face in keys(result.tractions[1])] for t in result.tractions],
        "volume" => result.volume,
        "fibre_volume_fraction" => result.fibre_volume_fraction,
        "nodes" => n_nodes(mesh),
        "cells" => n_cells(mesh),
        "unknowns" => result.unknowns,
        "anchor_node" => result.anchor,
        "vtk_path" => vtk_path,
        "runtime_seconds" => round(time() - started; digits = 3),
    )
    open(joinpath(output_dir, "julia_homogenization_result.toml"), "w") do io
        TOML.print(io, output)
    end
    verbose && println("HOMOGENIZATION done: ", n_cells(mesh), " cells, ", result.unknowns,
        " unknowns")
    return output
end
