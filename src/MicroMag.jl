module MicroMag

using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using LinearAlgebra
using PreallocationTools
using Memoize
using EllipsisNotation
using Parameters

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(false)

export Relax
export InitSim
export Run
export Params
export Mesh

const μ₀ = π * 4e-7 / 1e9 # vacuum permeability, = 4 * pi / 10
const γ = 2.221e5

@with_kw struct Mesh{T<:Int,R<:Float64}
    nx::T
    ny::T
    nz::T = 1
    dx::R
    dy::R
    dz::R
end

ScalarOrArrayOrFunction = Union{Float64,CuArray{Float32,3},Function}
VectorOrArrayOrFunction = Union{Vector{Float64},CuArray{Float32,4},Function}

@with_kw mutable struct Params{T<:ScalarOrArrayOrFunction}
    Ms::T
    α::T
    A::T = 0.0
    # Ku::T = 0.
    # D::T = 0.
    Bext::VectorOrArrayOrFunction = [0.0, 0.0, 0.0]
    exch = 2 * A / μ₀ / Ms
    prefactor1 = -γ / (1 + α * α)
    prefactor2 = prefactor1 * α
end

struct Interactions{T<:Bool}
    demag::T
    exchange::T
    zeeman::T
    # anistropy::T = false
    # dmi::T = false
end


include("Demag.jl")
include("Exchange.jl")
include("LLG.jl")
include("Zeeman.jl")

mutable struct Sim
    mesh::Mesh
    params::Params
    Heff::CuArray{Float32,4}
    M_x_H::CuArray{Float32,4}
    demag::Demag
    interactions::Interactions
    relax::Bool
end


check_normalize!(m::CuArray{Float32,4}) = m ./= sqrt.(sum(m .^ 2, dims=1))

# FIXME: Does converting to Float32 here improve performance?
# TODO: Make α spatially dependent
function InitSim(m0, mesh::Mesh, params::Params; calc_demag=true)

    check_normalize!(m0)

    nx = mesh.nx 
    ny = mesh.ny
    nz = mesh.nz

    Heff = similar(m0)
    M_x_H = similar(m0)

    # FIXME: Will probably break with dual numbers
    M_pad = CUDA.zeros(Float32, 3, nx * 2, ny * 2, nz * 2)
    M_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)
    H_demag_fft = CUDA.zeros(ComplexF32, 3, nx + 1, ny * 2, nz * 2)

    plan = plan_rfft(M_pad, [2, 3, 4])

    input_indices = CartesianIndices((3, nx, ny, nz))
    output_indices = CartesianIndices((3, nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)))

    demag = Demag(M_pad, M_fft, H_demag_fft, Demag_Kernel(mesh)...,
        plan, input_indices, output_indices)

    interactions = Interactions(calc_demag,
        params.A != 0.,
        params.Bext != [0.0, 0.0, 0.0]
    )

    sim = Sim(mesh, params, Heff, M_x_H, demag, interactions, false)

    return sim

end


function Relax(m0, sim::Sim)
    # prob = SteadyStateProblem(LLG_loop!, m0, p)

    sim.relax = true
    true_α = sim.params.α
    true_Bext = sim.params.Bext

    sim.params.α = 0.5
    sim.params.Bext = [0.0, 0.0, 0.0]

    cb = TerminateSteadyState(1, 1)
    end_point = 4
    tspan = (0, end_point)
    t_points = range(0, end_point, length=600)
    prob = ODEProblem(LLG_loop!, m0, tspan, sim)


    # saveat=2000 if memory becomes an issue
    sol = solve(prob, OwrenZen3(), progress=true, abstol=1e-3, reltol=1e-3, callback=cb, saveat=t_points, dt=1e-3)

    # cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

    sim.params.α = true_α  
    sim.params.Bext = true_Bext

    return sol.u[end]
end

function Run(m0, sim::Sim, t::Float64)

    # TODO: Convert to nice units
    end_point = t
    tspan = (0, end_point)
    t_points = range(0, end_point, length=300)

    prob = ODEProblem(LLG_loop!, m0, tspan, sim)
    sol = solve(prob, OwrenZen3(), progress=true, progress_steps=1000, abstol=1e-3, reltol=1e-3, saveat=t_points, dt=1e-3)

    cpu_sol = cat([Array(x) for x in sol.u]..., dims=5)

    return sol.t, cpu_sol
end

end # module MicroMag.jl
