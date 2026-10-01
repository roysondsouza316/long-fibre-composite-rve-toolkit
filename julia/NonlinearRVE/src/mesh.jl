# RVE mesh preparation: read the bulk simplex mesh and insert zero-thickness cohesive
# elements on every fibre/matrix interface facet (same algorithm and ordering as the Python
# rve2d.nonlinear.mesh module).

"""
    RVEMesh{dim}

Bulk linear simplex mesh plus cohesive interface elements. Interface insertion duplicates
every interface node: matrix cells keep the original node and fibre cells use the duplicate.
`cohesive[:, e]` lists the matrix-side ("bottom") nodes and then the coincident fibre-side
("top") nodes, ordered so that the facet normal built from the bottom nodes points into the
fibre; the separation `u_top - u_bottom = u_fibre - u_matrix` is then positive in opening.
"""
struct RVEMesh{dim}
    points::Vector{Vec{dim, Float64}}
    cells::Matrix{Int}          # (dim + 1) × n_cells
    phase::Vector{Int}
    cohesive::Matrix{Int}       # 2dim × n_cohesive: bottom nodes, then top nodes
    node_side::Vector{Int}      # 0: original node, 1: fibre-side duplicate
    lower::Vec{dim, Float64}
    upper::Vec{dim, Float64}
    matrix_phase_id::Int
    fibre_phase_id::Int
end

n_nodes(mesh::RVEMesh) = length(mesh.points)
n_cells(mesh::RVEMesh) = size(mesh.cells, 2)
n_cohesive(mesh::RVEMesh) = size(mesh.cohesive, 2)

# Local facets of linear triangles and tetrahedra (node positions within the cell).
cell_facets(::Val{2}) = ((1, 2), (2, 3), (3, 1))
cell_facets(::Val{3}) = ((1, 2, 3), (1, 2, 4), (2, 3, 4), (1, 3, 4))

"""
    read_rve_arrays(nodes_path, cells_path, tags_path, dim)

Read the CSV mesh arrays written by the rve2d bridge (node coordinates, 1-based cell
connectivity, one physical tag per cell). Nodes not used by any cell are dropped and the
remaining ones renumbered in their original order.
"""
function read_rve_arrays(nodes_path::AbstractString, cells_path::AbstractString,
        tags_path::AbstractString, dim::Int)
    node_table = readdlm(nodes_path, ',', Float64)
    cell_table = readdlm(cells_path, ',', Int)
    tags = vec(readdlm(tags_path, ',', Int))
    size(cell_table, 2) == dim + 1 ||
        error("expected $(dim + 1) nodes per cell for a $(dim)D mesh, got $(size(cell_table, 2))")
    size(cell_table, 1) == length(tags) || error("cell and tag counts differ")
    used = sort!(unique(vec(cell_table)))
    new_id = zeros(Int, size(node_table, 1))
    new_id[used] .= 1:length(used)
    points = [Vec{dim}(ntuple(i -> node_table[node, i], dim)) for node in used]
    cells = permutedims(new_id[cell_table])
    return points, cells, tags
end

"""
    build_rve_mesh(points, cells, phase, matrix_phase_id, fibre_phase_id; cohesive = true)

`RVEMesh` with cohesive elements on every facet shared by a matrix and a fibre cell, or
without interfaces (perfect bonding) when `cohesive = false`.
"""
function build_rve_mesh(points::Vector{Vec{dim, Float64}}, cells::Matrix{Int},
        phase::Vector{Int}, matrix_phase_id::Int, fibre_phase_id::Int;
        cohesive::Bool = true) where {dim}
    unknown = setdiff(unique(phase), (matrix_phase_id, fibre_phase_id))
    isempty(unknown) || error("bulk cells carry unexpected physical tags $(sort(unknown))")
    cells = orient_cells(points, cells)
    cohesive && return insert_cohesive_interfaces(points, cells, phase, matrix_phase_id,
        fibre_phase_id)
    return RVEMesh{dim}(copy(points), copy(cells), copy(phase), zeros(Int, 2dim, 0),
        zeros(Int, length(points)), bounds(points)..., matrix_phase_id, fibre_phase_id)
end

"""
    orient_cells(points, cells)

Copy of `cells` with every simplex positively oriented (Ferrite requires `det(J) > 0`); an
inverted cell has its second and third nodes swapped.
"""
function orient_cells(points::Vector{Vec{dim, Float64}}, cells::Matrix{Int}) where {dim}
    oriented = copy(cells)
    for cell in axes(cells, 2)
        x = [points[n] for n in view(cells, :, cell)]
        signed = dim == 2 ?
            (x[2][1] - x[1][1]) * (x[3][2] - x[1][2]) - (x[3][1] - x[1][1]) * (x[2][2] - x[1][2]) :
            dot(x[2] - x[1], cross(x[3] - x[1], x[4] - x[1]))
        if signed < 0
            oriented[2, cell], oriented[3, cell] = cells[3, cell], cells[2, cell]
        end
    end
    return oriented
end

function bounds(points::Vector{Vec{dim, Float64}}) where {dim}
    lower = Vec{dim}(ntuple(i -> minimum(p[i] for p in points), dim))
    upper = Vec{dim}(ntuple(i -> maximum(p[i] for p in points), dim))
    return lower, upper
end

function insert_cohesive_interfaces(points::Vector{Vec{dim, Float64}}, cells::Matrix{Int},
        phase::Vector{Int}, matrix_phase_id::Int, fibre_phase_id::Int) where {dim}
    n_original = length(points)
    owners = Dict{NTuple{dim, Int}, Vector{Int}}()
    for cell in axes(cells, 2), facet in cell_facets(Val(dim))
        key = Tuple(sort(SVector{dim, Int}(ntuple(i -> cells[facet[i], cell], dim))))
        push!(get!(() -> Int[], owners, key), cell)
    end
    phases = (matrix_phase_id, fibre_phase_id)
    keys = NTuple{dim, Int}[]
    fibre_cell = Int[]
    for (key, owner) in owners
        length(owner) == 2 || continue
        pa, pb = phase[owner[1]], phase[owner[2]]
        (pa != pb && pa in phases && pb in phases) || continue
        push!(keys, key)
        push!(fibre_cell, pa == fibre_phase_id ? owner[1] : owner[2])
    end
    order = sortperm(keys)  # lexicographic, like numpy.unique on the sorted facet rows
    keys, fibre_cell = keys[order], fibre_cell[order]

    bottom = zeros(Int, dim, length(keys))
    for (facet, key) in enumerate(keys)
        bottom[:, facet] .= key
    end
    interface_nodes = sort!(unique(vec(bottom)))
    duplicate_of = zeros(Int, n_original)
    duplicate_of[interface_nodes] .= n_original .+ (1:length(interface_nodes))

    new_cells = copy(cells)
    for cell in axes(cells, 2)
        phase[cell] == fibre_phase_id || continue
        for a in axes(cells, 1)
            duplicate = duplicate_of[cells[a, cell]]
            duplicate > 0 && (new_cells[a, cell] = duplicate)
        end
    end

    # Orient every facet so that the normal built from its bottom nodes points into the fibre.
    extent = maximum(maximum, points) - minimum(minimum, points)
    for facet in axes(bottom, 2)
        nodes = view(bottom, :, facet)
        centre = sum(points[n] for n in nodes) / dim
        fibre_centre = sum(points[n] for n in view(cells, :, fibre_cell[facet])) / (dim + 1)
        if dot(facet_normal(points, nodes), fibre_centre - centre) < 0
            bottom[1, facet], bottom[2, facet] = bottom[2, facet], bottom[1, facet]
        end
        norm(facet_normal(points, nodes)) > 1.0e-14 * max(extent, 1.0)^(dim - 1) ||
            error("found a degenerate (zero-size) fibre/matrix interface facet")
    end
    top = duplicate_of[bottom]

    lower, upper = bounds(points)
    return RVEMesh{dim}(
        vcat(points, points[interface_nodes]),
        new_cells,
        copy(phase),
        vcat(bottom, top),
        vcat(zeros(Int, n_original), ones(Int, length(interface_nodes))),
        lower,
        upper,
        matrix_phase_id,
        fibre_phase_id,
    )
end

"Unnormalised facet normal: left normal of an edge (2D) or the triangle normal (3D)."
function facet_normal(points::Vector{Vec{2, Float64}}, facet)
    edge = points[facet[2]] - points[facet[1]]
    return Vec{2}((-edge[2], edge[1]))
end

function facet_normal(points::Vector{Vec{3, Float64}}, facet)
    return cross(points[facet[2]] - points[facet[1]], points[facet[3]] - points[facet[1]])
end
