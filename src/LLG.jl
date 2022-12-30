
function LLG_loop!(dm, m0, p, t)

    mesh, fields, demag, params, relax = p

    #################
    ## Demag Field ##
    #################

    fill!(fields.M_pad, 0)
    @inbounds fields.M_pad[mesh.in] = m0 .* params.Ms

    Demag!(fields, demag, mesh)

    ####################
    ## Exchange Field ##
    ####################

    Exchange!(fields.H_eff, m0, params.exch, mesh)

    if !relax
        Zeeman!(fields.H_eff, params.B_ext(t))
    end

    # apply LLG equation
    cross!(fields.M_x_H, m0, fields.H_eff)
    cross!(dm, m0, fields.M_x_H)

    if relax
        dm .*= params.relax_prefactor2
    else
        dm .*= params.prefactor2
        @. dm += params.prefactor1 * fields.M_x_H
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