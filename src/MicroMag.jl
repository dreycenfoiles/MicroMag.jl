module MicroMag

using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using LinearAlgebra
using PreallocationTools
using Memoize

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

export LLG_loop!
export Demag_Kernel
export check_normalize!

include("Demag.jl")
include("Exchange.jl")

const mu_0 = pi * 4e-7 / 1e9 # vacuum permeability, = 4 * pi / 10
const gamma = 2.221e5


function LLG_loop!(dm, m0, p, t)

    arrays, fft_plans, indices, parameters, Mx_kernels, My_kernels, Mz_kernels, spacing = p

    M_pad, M_fft, H_demag, H_demag_fft, H_eff, M_x_H = arrays

    plan, iplan = fft_plans

    input_indices, output_indices = indices

    Ms, A, alpha = parameters

    Kxx_fft, Kxy_fft, Kxz_fft = Mx_kernels
    Kxy_fft, Kyy_fft, Kyz_fft = My_kernels
    Kxz_fft, Kyz_fft, Kzz_fft = Mz_kernels

    dx, dy, dz = spacing

    Hx_eff = @view H_eff[1, :, :, :]
    Hy_eff = @view H_eff[2, :, :, :]

    prefactor1 = -gamma / (1 + alpha * alpha)
    prefactor2 = prefactor1 * alpha
    exch = 2 * A / mu_0 / Ms

    #################
    ## Demag Field ##
    #################

    fill!(M_pad, 0)

    @inbounds M_pad[input_indices] = m0 .* Ms

    mul!(M_fft, plan, M_pad)

    Mx_fft = @view M_fft[1, :, :, :]
    My_fft = @view M_fft[2, :, :, :]
    Mz_fft = @view M_fft[3, :, :, :]

    @. H_demag_fft[1, :, :, :] = Mx_fft * Kxx_fft + My_fft * Kxy_fft + Mz_fft * Kxz_fft
    @. H_demag_fft[2, :, :, :] = Mx_fft * Kxy_fft + My_fft * Kyy_fft + Mz_fft * Kyz_fft
    @. H_demag_fft[3, :, :, :] = Mx_fft * Kxz_fft + My_fft * Kyz_fft + Mz_fft * Kzz_fft

    mul!(H_demag, iplan, H_demag_fft)

    @inbounds H_eff .= real(H_demag[output_indices]) # truncation of demag field

    ####################
    ## Exchange Field ##
    ####################

    Exchange!(H_eff, m0, exch, dx, dy, dz)

    if t < 0.05
        H_eff .+= 0.1 / 1e18 / mu_0 # apply a saturation field to get S-state
    elseif t > 0.25
        Hx_eff .+= -24.6e-3 / 1e18 / mu_0 # apply the reverse field
        Hy_eff .+= +4.3e-3 / 1e18 / mu_0
        alpha = 0.02
        prefactor1 = -gamma / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha
    end

    # apply LLG equation
    cross!(M_x_H, m0, H_eff)
    cross!(dm, m0, M_x_H)

    dm .*= prefactor2
    @. dm += prefactor1 * M_x_H

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

check_normalize!(m) = m ./= sqrt.(sum(m .^ 2, dims=1))
normalize_llg!(integrator) = check_normalize!(integrator.u)
condition(u, t, integrator) = true


function Relax(m0)
    prob = SteadyStateProblem(LLG_loop!, m0, p)
    # saveat=2000 if memory becomes an issue
    sol = solve(prob, DynamicSS(OwrenZen3()), abstol=1e-2, reltol=1e-2)
    return sol
end

end # module MicroMag.jl
