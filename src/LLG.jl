
function LLG_loop!(dm::T, m0::T, p::Tuple{Sim,Bool}, t::Float64) where {T<:CuArray{Float32}}

    sim, relax = p

    fill!(sim.H_eff, 0.)

    Demag!(sim.H_eff, m0, sim.demag, sim.params.Ms)
    Exchange!(sim.H_eff, m0, sim.params.exch, sim.mesh)

    if !relax
        Zeeman!(sim.H_eff, sim.params.B_ext, t)
    end

    # apply LLG equation
    cross!(sim.M_x_H, m0, sim.H_eff)
    cross!(dm, m0, sim.M_x_H)

    if relax
        dm .*= sim.params.relax_prefactor2
    else
        dm .*= sim.params.prefactor2
        @. dm += sim.params.prefactor1 * sim.M_x_H
    end

    nothing

end


function cross!(product::T, A::T, B::T) where {T<:CuArray{Float32}}

    Ax = @views A[1, ..]
    Ay = @views A[2, ..]
    Az = @views A[3, ..]

    Bx = @views B[1, ..]
    By = @views B[2, ..]
    Bz = @views B[3, ..]

    @. @views product[1, ..] = Ay * Bz - Az * By
    @. @views product[2, ..] = Az * Bx - Ax * Bz
    @. @views product[3, ..] = Ax * By - Ay * Bx

    nothing
end