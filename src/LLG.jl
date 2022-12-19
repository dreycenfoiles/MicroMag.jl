
function LLG_loop!(dm, m0, p, t)

    mesh, fields, demag, param, B_ext, relax = p

    Ms, A, α = param

    if relax
        α = 0.5
    end

    prefactor1 = -γ / (1 + α * α)
    prefactor2 = prefactor1 * α
    exch = 2 * A / μ₀ / Ms

    #################
    ## Demag Field ##
    #################

    fill!(fields.M_pad, 0)
    @inbounds fields.M_pad[mesh.in] = m0 .* Ms

    Demag!(fields, demag, mesh)

    ####################
    ## Exchange Field ##
    ####################

    Exchange!(fields.H_eff, m0, exch, mesh)

    if !relax
        Zeeman!(fields.H_eff, B_ext(t))
    end

    # apply LLG equation
    cross!(fields.M_x_H, m0, fields.H_eff)
    cross!(dm, m0, fields.M_x_H)

    dm .*= prefactor2

    if !relax 
    @. dm += prefactor1 * fields.M_x_H
    end

    nothing

end


function cross!(product, A, B)

    Ax = @views A[1, :, :, :]
    Ay = @views A[2, :, :, :]
    Az = @views A[3, :, :, :]

    Bx = @views B[1, :, :, :]
    By = @views B[2, :, :, :]
    Bz = @views B[3, :, :, :]

    @. @views product[1, :, :, :] = Ay * Bz - Az * By
    @. @views product[2, :, :, :] = Az * Bx - Ax * Bz
    @. @views product[3, :, :, :] = Ax * By - Ay * Bx

    nothing
end