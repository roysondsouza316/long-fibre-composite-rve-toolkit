# Fluctuation constraints: map every Ferrite DOF to an independent (reduced) unknown, or to 0
# for a DOF fixed at zero (same rules as the Python rve2d.engines.tensormesh.constraints module).
#
# periodic:  image-face nodes are tied to their mirror-face partners, matched per interface
#            side, with edge and corner chains resolved to one master; one interior matrix node
#            is fixed to remove the rigid translation.
# dirichlet: zero fluctuation on the whole boundary (affine boundary displacement, KUBC).

boundary_tolerance(mesh::RVEMesh) = 1.0e-8 * maximum(mesh.upper - mesh.lower)

function on_boundary(mesh::RVEMesh{dim}, point, tol) where {dim}
    return any(i -> abs(point[i] - mesh.lower[i]) < tol || abs(point[i] - mesh.upper[i]) < tol,
        1:dim)
end

boundary_nodes(mesh::RVEMesh) =
    findall(p -> on_boundary(mesh, p, boundary_tolerance(mesh)), mesh.points)

function match_key(mesh::RVEMesh, node::Int, others, tol)
    return (map(a -> round(Int, mesh.points[node][a] / tol), others)..., mesh.node_side[node])
end

"Master node of every node under periodicity (a node is its own master when untied)."
function periodic_masters(mesh::RVEMesh{dim}) where {dim}
    tol = boundary_tolerance(mesh)
    master = collect(1:n_nodes(mesh))
    for axis in 1:dim
        others = Tuple(a for a in 1:dim if a != axis)
        lookup = Dict{NTuple{dim, Int}, Int}()
        for node in 1:n_nodes(mesh)
            if abs(mesh.points[node][axis] - mesh.lower[axis]) < tol
                lookup[match_key(mesh, node, others, tol)] = node
            end
        end
        for node in 1:n_nodes(mesh)
            abs(mesh.points[node][axis] - mesh.upper[axis]) < tol || continue
            partner = get(lookup, match_key(mesh, node, others, tol), 0)
            partner == 0 && error(
                "periodic boundary conditions need a node-matched periodic mesh: no partner " *
                "for the node at $(Tuple(mesh.points[node])); build the geometry with " *
                "periodic_compatible: true",
            )
            master[node] = partner
        end
    end
    for _ in 1:(dim + 1)
        master = master[master]
    end
    return master
end

"Matrix-side node closest to the RVE centre that lies off the boundary and off the interface."
function interior_anchor(mesh::RVEMesh{dim}) where {dim}
    candidate = trues(n_nodes(mesh))
    candidate[boundary_nodes(mesh)] .= false
    candidate .&= mesh.node_side .== 0
    if n_cohesive(mesh) > 0
        candidate[unique(vec(mesh.cohesive[1:dim, :]))] .= false
    end
    pool = findall(candidate)
    isempty(pool) && error("could not find an interior node to anchor the periodic fluctuation")
    centre = (mesh.lower + mesh.upper) / 2
    return pool[argmin([norm(mesh.points[node] - centre) for node in pool])]
end

"""
    DofMap

`full_to_reduced[d]` is the reduced unknown of Ferrite DOF `d`, or 0 when the DOF is fixed.
Reduced unknowns are numbered node by node like the Python engine, so both engines assemble
the same reduced system.
"""
struct DofMap
    full_to_reduced::Vector{Int}
    n_reduced::Int
    anchor::Int
end

"""
    build_dof_map(mesh, node_dofs, n_full, boundary_condition)

`node_dofs[c, n]` is the Ferrite DOF of displacement component `c` at node `n`; there may be
more components than space dimensions (three on a 2D mesh for generalized plane strain).
"""
function build_dof_map(mesh::RVEMesh, node_dofs::Matrix{Int}, n_full::Int,
        boundary_condition::AbstractString)
    if boundary_condition == "periodic"
        master = periodic_masters(mesh)
        anchor = interior_anchor(mesh)
        independent = filter!(!=(anchor), sort!(unique(master)))
        reduced_node = zeros(Int, n_nodes(mesh))
        reduced_node[independent] .= 1:length(independent)
        node_index = reduced_node[master]
        n_independent = length(independent)
    elseif boundary_condition == "dirichlet"
        fixed = falses(n_nodes(mesh))
        fixed[boundary_nodes(mesh)] .= true
        free = findall(!, fixed)
        node_index = zeros(Int, n_nodes(mesh))
        node_index[free] .= 1:length(free)
        n_independent = length(free)
        anchor = 0
    else
        error("unsupported boundary condition \"$boundary_condition\"")
    end
    ncomp = size(node_dofs, 1)
    full_to_reduced = zeros(Int, n_full)
    for node in 1:n_nodes(mesh), component in 1:ncomp
        if node_index[node] > 0
            full_to_reduced[node_dofs[component, node]] = (node_index[node] - 1) * ncomp + component
        end
    end
    return DofMap(full_to_reduced, n_independent * ncomp, anchor)
end
