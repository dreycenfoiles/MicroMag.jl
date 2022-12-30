

check_normalize!(m) = m ./= sqrt.(sum(m .^ 2, dims=1))

function Init_m(mesh::Mesh, init::Vector{Float64})

    m0 = CUDA.zeros(Float32, 3, mesh.nx, mesh.ny, mesh.nz)

    m0[1, :, :, :] .= init[1]
    m0[2, :, :, :] .= init[2]
    m0[3, :, :, :] .= init[3]

    check_normalize!(m0)

    return m0
end