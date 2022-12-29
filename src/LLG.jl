
function LLG_loop!(dm, m, sim, t)

    Demag!(sim.Heff, m, sim.demag, sim.mesh, sim.params.Ms)
    Exchange!(sim.Heff, m, sim.mesh, sim.params.exch, t)
    Zeeman!(sim.Heff, sim.params.Bext, t)

    # Anisotropy_Field
    # DMI_Field

    # apply LLG equation
    cross!(sim.M_x_H, m, sim.Heff)
    cross!(dm, m, sim.M_x_H)

    dm .*= sim.params.prefactor2

    if !sim.relax 
    @. dm += sim.params.prefactor1 * fields.M_x_H
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