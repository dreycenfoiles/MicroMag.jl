function neumann_bc(x, n)

    if x == n + 1
        return x - 1
    elseif x == 0
        return x + 1
    else
        return x
    end

end

# TODO: Periodic boundary conditions


function Exchange_kernel(H_exch, M, dx, dy, dz)

    nc, nx, ny, nz = size(M)

    c = blockIdx().x 
    j = blockIdx().y
    i = threadIdx().x

    ip1 = neumann_bc(i + 1, nx)
    im1 = neumann_bc(i - 1, nx)
    jp1 = neumann_bc(j + 1, ny)
    jm1 = neumann_bc(j - 1, ny)

   H_exch[c, i, j, 1] = (M[c, ip1, j, 1] - 2 * M[c, i, j, 1] + M[c, im1, j, 1]) / dx / dx +
                        (M[c, i, jp1, 1] - 2 * M[c, i, j, 1] + M[c, i, jm1, 1]) / dy / dy

    nothing
end 

function Exchange!(H_exch, M, exch, dx, dy, dz)

    nc, nx, ny, nz = size(M)

    # FIXME: Will error if nx >= 1024
    @cuda blocks=(nc, ny) threads=nx Exchange_kernel(H_exch, M, dx, dy, dz)

    H_exch .*= exch

    return
end


# function Exchange!(H_exch, M, exch, H0, H1, H2, H3, dd)
#     # calculation of exchange field

#     fill!(H0, 0)
#     fill!(H1, 0)
#     fill!(H2, 0)
#     fill!(H3, 0)

#     H0[:, 2:end, :, :] .= @views M[:, 1:end-1, :, :]
#     H0[:, 1, :, :] .= @views H0[:, 2, :, :]
#     H1[:, 1:end-1, :, :] .= @views M[:, 2:end, :, :]
#     H1[:, end, :, :] .= @views H1[:, end-1, :, :]

#     H2[:, :, 2:end, :] .= @views M[:, :, 1:end-1, :]
#     H2[:, :, 1, :] .= @views H2[:, :, 2, :]
#     H3[:, :, 1:end-1, :] .= @views M[:, :, 2:end, :]
#     H3[:, :, end, :] .= @views H3[:, :, end-1, :]

#     @. @views H_exch .= exch / dd / dd * (H0 + H1 + H2 + H3 - 4 * M)

# end


    
