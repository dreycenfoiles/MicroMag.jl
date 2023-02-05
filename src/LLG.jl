
function LLG_relax!(dm::T, m::T, sim::Sim, t) where {T<:CuArray{Float32}}

    t = 0. # time is irrelevant for relaxation

    fill!(sim.H_eff, 0.)

    for interaction in sim.Interactions
        interaction(sim.H_eff, m, t)
    end

    # apply LLG equation
    cross!(sim.M_x_H, m, sim.H_eff)
    cross!(dm, m, sim.M_x_H)

    dm .*= sim.relax_prefactor2

    nothing
end


function LLG_run!(dm::T, m::T, sim::Sim, t) where {T<:CuArray{Float32}}

    fill!(sim.H_eff, 0.)

    for interaction in sim.Interactions
        interaction(sim.H_eff, m, t)
    end

    # apply LLG equation
    cross!(sim.M_x_H, m, sim.H_eff)
    cross!(dm, m, sim.M_x_H)

    dm .*= sim.prefactor2
    @. dm += sim.prefactor1 * sim.M_x_H

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