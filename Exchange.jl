


# function δx!(H_exch, M, dx)
#     @views @. H_exch[:, 2:end-1, :, :] += M[:, 3:end, :, :] - 2 * M[:, 2:end-1, :, :] + M[:, 1:end-2, :, :]
#     @views @. H_exch[:, 1, :, :] += M[:, 2, :, :] - 2 * M[:, 1, :, :] + M[:, 1, :, :]
#     @views @. H_exch[:, end, :, :] += M[:, end, :, :] - 2 * M[:, end, :, :] + M[:, end-1, :, :]
#     H_exch ./ dx^2
#     nothing
# end

# function δy!(H_exch, M, dy)
#     @views @. H_exch[:, :, 2:end-1, :] += M[:, :, 3:end, :] - 2 * M[:, :, 2:end-1, :] + M[:, :, 1:end-2, :]
#     @views @. H_exch[:, :, 1, :] += M[:, :, 2, :] - 2 * M[:, :, 1, :] + M[:, :, 1, :]
#     @views @. H_exch[:, :, end, :] += M[:, :, end, :] - 2 * M[:, :, end, :] + M[:, :, end-1, :]
#     H_exch ./ dy^2
#     nothing
# end

# function δz!(H_exch, M, dz)
#     @views @. H_exch[:, :, :, 2:end-1] += M[:, :, :, 3:end] - 2 * M[:, :, :, 2:end-1] + M[:, :, :, 1:end-2]
#     @views @. H_exch[:, :, :, 1] += M[:, :, :, 2] - 2 * M[:, :, :, 1] + M[:, :, :, 1]
#     @views @. H_exch[:, :, :, end] += M[:, :, :, end] - 2 * M[:, :, :, end] + M[:, :, :, end-1]
#     H_exch ./ dz^2
#     nothing
# end

# function Exchange!(H_exch, M, exch, dx, dy, dz)

#     δx!(H_exch, M, dx)
#     δy!(H_exch, M, dy)
#     # δz!(H_exch, M, dz)

#     H_exch .*= exch
#     nothing
# end

function Exchange!(H_exch, M, exch, H0, H1, H2, H3, dd)
    # calculation of exchange field

    fill!(H0, 0)
    fill!(H1, 0)
    fill!(H2, 0)
    fill!(H3, 0)

    H0[:, 2:end, :, :] .= @views M[:, 1:end-1, :, :]
    H0[:, 1, :, :] .= @views H0[:, 2, :, :]
    H1[:, 1:end-1, :, :] .= @views M[:, 2:end, :, :]
    H1[:, end, :, :] .= @views H1[:, end-1, :, :]

    H2[:, :, 2:end, :] .= @views M[:, :, 1:end-1, :]
    H2[:, :, 1, :] .= @views H2[:, :, 2, :]
    H3[:, :, 1:end-1, :] .= @views M[:, :, 2:end, :]
    H3[:, :, end, :] .= @views H3[:, :, end-1, :]

    H_exch .+= @views @. exch / dd / dd * (H0 + H1 + H2 + H3 - 4 * M)

end


    
