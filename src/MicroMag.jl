module MicroMag

using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using LinearAlgebra
using PreallocationTools
using Memoize
using Parameters
using BenchmarkTools
using EllipsisNotation

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

export Relax
export InitSim 
export Run
export Mesh 
export Params

const μ₀::Float32 = pi * 4e-7 / 1e9 # vacuum permeability, = 4 * pi / 10
const γ::Float32 = 2.221e5

 struct Mesh
    nx::Int
    ny::Int
    nz::Int
    dx::Float32
    dy::Float32
    dz::Float32
end


struct Demag2D{T<:CuArray{ComplexF32,2}}
    Kxx_fft::T
    Kyy_fft::T
    Kzz_fft::T
    Kxy_fft::T
    fft::CUDA.CUFFT.rCuFFTPlan{Float32,-1,false,3}
    M_pad::CuArray{Float32,3}
    M_fft::CuArray{ComplexF32,3}
    H_demag::CuArray{Float32,3}
    H_demag_fft::CuArray{ComplexF32,3}
    in::CartesianIndices{3}
    out::CartesianIndices{3}
end

struct Demag3D{T<:CuArray{ComplexF32, 3}}
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
    Ms::Float32
    A::Float32
    α::Float32
    B_ext::Vector{Float32}
    exch::Float32 = 2 * A / μ₀ / Ms
    prefactor1::Float32 = -γ / (1 + α * α)
    prefactor2::Float32 = prefactor1 * α
    relax_α::Float32 = 0.5 
    relax_prefactor1::Float32 = -γ / (1 + relax_α * relax_α)
    relax_prefactor2::Float32 = relax_prefactor1 * relax_α
end

struct Sim
    m::CuArray{Float32}
    mesh::Mesh
    params::Params
    demag::Union{Demag2D,Demag3D}
    H_eff::CuArray{Float32}
    M_x_H::CuArray{Float32}
end

include("Demag.jl")
include("Exchange.jl")
include("LLG.jl")
include("Zeeman.jl")
include("Init_m.jl")


# TODO: Make α spatially dependent
function InitSim(init_m, mesh::Mesh, params::Params)::Sim
    
    ### Initialize Mesh ###

    nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
    dx, dy, dz = mesh.dx, mesh.dy, mesh.dz

    m0 = Init_m(mesh, init_m)

    if nz == 1

        input_indices = CartesianIndices((3, nx, ny))
        output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1)))

        ########################

        ### Initialize Empty Arrays ###

        M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2)
        M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2)

        H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2)
        H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2)

        H_eff = CUDA.zeros(Float32, 3, nx, ny)
        M_x_H = CUDA.zeros(Float32, 3, nx, ny)

        ################################

        ### Initialize Demag Kernel ###

        plan = plan_rfft(M_pad, [2, 3])

        demag = Demag2D(Demag_Kernel(nx, ny, nz, dx, dy, dz)..., plan, M_pad, M_fft, H_demag, H_demag_fft, input_indices, output_indices)

        ################################


        return Sim(m0, mesh, params, demag, H_eff, M_x_H)

    else 

        input_indices = CartesianIndices((3, nx, ny, nz))
        output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)))
    
        ########################

        ### Initialize Empty Arrays ###

        M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
        M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

        H_demag = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
        H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

        H_eff = CUDA.zeros(Float32, 3, nx, ny, nz)
        M_x_H = CUDA.zeros(Float32, 3, nx, ny, nz)

        ################################

        ### Initialize Demag Kernel ###

        plan = plan_rfft(M_pad, [2, 3, 4])

        demag = Demag3D(Demag_Kernel(nx, ny, nz, dx, dy, dz)..., plan, M_pad, M_fft, H_demag, H_demag_fft, input_indices, output_indices)

        ################################


        return Sim(m0, mesh, params, demag, H_eff, M_x_H)
    end
end


function Relax(sim::Sim)
    # prob = SteadyStateProblem(LLG_loop!, m0, p)
    p = (sim, true)
    cb = TerminateSteadyState(1, 1)
    end_point = 4
    tspan = (0, end_point)
    t_points = range(0, end_point, length=600)
    prob = ODEProblem(LLG_loop!, sim.m, tspan, p)
    # saveat=2000 if memory becomes an issue
    sol = solve(prob, OwrenZen3(), progress=true, abstol=1e-3, reltol=1e-3, callback=cb, saveat=t_points, dt=1e-3)

    # cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

    sim.m .= sol.u[end]

    nothing 
end

function Run(sim::Sim, t)

    new_p = (sim, false)
    
    # TODO: Convert to nice units
    end_point = t
    tspan = (0, end_point)
    t_points = range(0, end_point, length=300)

    prob = ODEProblem(LLG_loop!, sim.m, tspan, new_p)
    sol = solve(prob, OwrenZen3(), progress=true, progress_steps=1000, abstol=1e-3, reltol=1e-3, saveat=t_points, dt=1e-3)

    last_dim = length(size(sim.m))

    cpu_sol = cat([Array(x) for x in sol.u]..., dims=last_dim+1)

    return sol.t, cpu_sol
end

end # module MicroMag.jl
