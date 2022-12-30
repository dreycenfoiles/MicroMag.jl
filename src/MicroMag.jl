module MicroMag

using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using LinearAlgebra
using PreallocationTools
using Memoize
using Parameters

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

export Relax
export Init_sim
export Run

const μ₀ = pi * 4e-7 / 1e9 # vacuum permeability, = 4 * pi / 10
const γ = 2.221e5

 struct Mesh
    nx::Int
    ny::Int
    nz::Int
    dx::Float32
    dy::Float32
    dz::Float32
end

struct Demag{T<:CuArray{ComplexF32, 3}}
    Kxx_fft::T
    Kyy_fft::T
    Kzz_fft::T
    Kxy_fft::T
    Kxz_fft::T
    Kyz_fft::T
    fft::CUDA.CUFFT.rCuFFTPlan{Float32,-1,false,4}
    M_pad::CuArray{Float32, 4}
    M_fft::CuArray{ComplexF32, 4}
    H_demag::CuArray{Float32, 4}
    H_demag_fft::CuArray{ComplexF32, 4}
    in::CartesianIndices{4}
    out::CartesianIndices{4}
end

@with_kw struct Params
    Ms::Float64
    A::Float64
    α::Float64
    B_ext::Vector{Float64}
    exch::Float64 = 2 * A / μ₀ / Ms
    prefactor1::Float64 = -γ / (1 + α * α)
    prefactor2::Float64 = prefactor1 * α
    relax_α::Float64 = 0.5 
    relax_prefactor1::Float64 = -γ / (1 + relax_α * relax_α)
    relax_prefactor2::Float64 = relax_prefactor1 * relax_α
end

include("Demag.jl")
include("Exchange.jl")
include("LLG.jl")
include("Zeeman.jl")

struct Sim 
    mesh::Mesh
    params::Params
    demag::Demag
    H_eff::CuArray{Float32, 4}
    M_x_H::CuArray{Float32, 4}
end


check_normalize!(m) = m ./= sqrt.(sum(m .^ 2, dims=1))

# FIXME: Does converting to Float32 here improve performance?
# TODO: Make α spatially dependent
function Init_sim(m0::CuArray{Float32,4}, dx::Float64, dy::Float64, dz::Float64, Ms::Float64, A::Float64, α::Float64, B_ext)
    

    ### Initialize Mesh ###

    nc, nx, ny, nz = size(m0)

    check_normalize!(m0)

    input_indices = CartesianIndices((3, nx, ny, nz))
    output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)))
   
    mesh = Mesh(nx, ny, nz, dx, dy, dz)

    ########################

    ### Initialize Empty Arrays ###

    M_pad = CUDA.zeros(Float32, nc, nx * 2, ny * 2, nz * 2)
    M_fft = CUDA.zeros(ComplexF32, nc, nx + 1, ny * 2, nz * 2)

    H_demag = CUDA.zeros(Float32, nc, nx * 2, ny * 2, nz * 2)
    H_demag_fft = CUDA.zeros(ComplexF32, nc, nx + 1, ny * 2, nz * 2)

    H_eff = CUDA.zeros(Float32, nc, nx, ny, nz)
    M_x_H = CUDA.zeros(Float32, nc, nx, ny, nz)

    ################################

    ### Initialize Demag Kernel ###

    plan = plan_rfft(M_pad, [2, 3, 4])

    demag = Demag(Demag_Kernel(nx, ny, nz, dx, dy, dz)..., plan, M_pad, M_fft, H_demag, H_demag_fft, input_indices, output_indices)

    ################################

    ### Initialize Parameters ###

    params = Params(Ms=Ms, A=A, α=α, B_ext=B_ext)

    ##############################

    return Sim(mesh, params, demag, H_eff, M_x_H)
end


function Relax(m0, p)
    # prob = SteadyStateProblem(LLG_loop!, m0, p)
    new_p = (p, true)
    cb = TerminateSteadyState(1, 1)
    end_point = 4
    tspan = (0, end_point)
    t_points = range(0, end_point, length=600)
    prob = ODEProblem(LLG_loop!, m0, tspan, new_p)
    # saveat=2000 if memory becomes an issue
    sol = solve(prob, OwrenZen3(), progress=true, abstol=1e-3, reltol=1e-3, callback=cb, saveat=t_points, dt=1e-3)

    # cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

    return sol.u[end]
end

function Run(m0,t,p)

    new_p = (p, false)
    
    # TODO: Convert to nice units
    end_point = t
    tspan = (0, end_point)
    t_points = range(0, end_point, length=300)

    prob = ODEProblem(LLG_loop!, m0, tspan, new_p)
    sol = solve(prob, OwrenZen3(), progress=true, progress_steps=1000, abstol=1e-3, reltol=1e-3, saveat=t_points, dt=1e-3)

    cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

    return sol.t, cpu_sol
end

# normalize_llg!(integrator) = check_normalize!(integrator.u)
# condition(u, t, integrator) = true


end # module MicroMag.jl
