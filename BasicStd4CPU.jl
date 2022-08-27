using DifferentialEquations
using LinearAlgebra # Used for cross product
using Plots
using Statistics # Used for mean 
# using DSP # Used for convolution
using Images
using ImageView
using FFTW
# using CUDA

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

nx = 10; # number of cells on x direction
ny = 10; # number of cells on y direction
nz = 1; # number of cells on z direction

dx = 6e-9; # cell size on x direction (nm)
dy = 3e-9; # cell size on y direction (nm)
dz = 3e-9; # cell size on z direction (nm)

alpha = 0.5
A = 1.3e-11 # (J/m)
const mu0 = pi * 4e-7 # (H/m)
Ms = 8e5 # (A/m)
const gamma = 2.211e5
# const gamma = -1.761e11

# include("Demag.jl")


Hx_demag = zeros(2 * nx, 2 * ny, 2 * nz)
Hy_demag = zeros(2 * nx, 2 * ny, 2 * nz)
Hz_demag = zeros(2 * nx, 2 * ny, 2 * nz)

mx_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)
my_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)
mz_demag_buffer = zeros(2 * nx, 2 * ny, 2 * nz)

Kxx = zeros(nx * 2, ny * 2, nz * 2); # Initialization of demagnetization tensor
Kxy = zeros(nx * 2, ny * 2, nz * 2);
Kxz = zeros(nx * 2, ny * 2, nz * 2);
Kyy = zeros(nx * 2, ny * 2, nz * 2);
Kyz = zeros(nx * 2, ny * 2, nz * 2);
Kzz = zeros(nx * 2, ny * 2, nz * 2);

prefactor = 1 / 4 / 3.14159265;
for K = -nz+1:nz-1 # Calculation of Demag tensor
    for J = -ny+1:ny-1
        for I = -nx+1:nx-1
            if I == 0 && J == 0 && K == 0
                continue
            end
            L = I + nx # shift the indices, b/c no negative index allowed in MATLAB
            M = J + ny
            N = K + nz
            for i = 0:1 #helper indices
                for j = 0:1
                    for k = 0:1
                        r = sqrt((I + i - 0.5) * (I + i - 0.5) * dx * dx + (J + j - 0.5) * (J + j - 0.5) * dy * dy + (K + k - 0.5) * (K + k - 0.5) * dz * dz)
                        Kxx[L, M, N] += (-1) .^ (i + j + k) * atan((K + k - 0.5) * (J + j - 0.5) * dx / r / (I + i - 0.5))
                        Kxy[L, M, N] += (-1) .^ (i + j + k) * log((K + k - 0.5) * dz + r)
                        Kxz[L, M, N] += (-1) .^ (i + j + k) * log((J + j - 0.5) * dy + r)
                        Kyy[L, M, N] += (-1) .^ (i + j + k) * atan((I + i - 0.5) * (K + k - 0.5) * dy / r / (J + j - 0.5))
                        Kyz[L, M, N] += (-1) .^ (i + j + k) * log((I + i - 0.5) * dx + r)
                        Kzz[L, M, N] += (-1) .^ (i + j + k) * atan((J + j - 0.5) * (I + i - 0.5) * dz / r / (K + k - 0.5))
                    end
                end
            end
            Kxx[L, M, N] = Kxx[L, M, N] * prefactor
            Kxy[L, M, N] = Kxy[L, M, N] * -prefactor
            Kxz[L, M, N] = Kxz[L, M, N] * -prefactor
            Kyy[L, M, N] = Kyy[L, M, N] * prefactor
            Kyz[L, M, N] = Kyz[L, M, N] * -prefactor
            Kzz[L, M, N] = Kzz[L, M, N] * prefactor
        end
    end
end # calculation of demag tensor done

fft_Kxx = rfft(Kxx)
fft_Kxy = rfft(Kxy)
fft_Kxz = rfft(Kxz)
fft_Kyy = rfft(Kyy)
fft_Kyz = rfft(Kyz)
fft_Kzz = rfft(Kzz)


# TODO: Periodic boundary conditions

function neumann_bc(x, n)

    if x == n + 1
        return x - 1
    elseif x == 0
        return x + 1
    else
        return x
    end

end

# plan = plan_fft(Hx_demag);
# inverse_plan = plan_ifft(Hx_demag);


function calc_demag_field(mx, my, mz, fft_kernel1, fft_kernel2, fft_kernel3)

    # Calculate demagnetizing field
    H_demag_buffer = irfft(
                rfft(mx) .* fft_kernel1 +
                rfft(my) .* fft_kernel2 +
                rfft(mz) .* fft_kernel3,
            2 * nx)

    return H_demag_buffer[nx:2 * nx - 1, ny:2 * ny - 1, nz:2 * nz - 1]
end


function LLG_loop!(dm, m0, p, t)

    A, Ms, dx, dy, dz = p

    if t < 4e-9
        H_ext = [0, 0, 0]
        precess = 0
        alpha = 0.5
    elseif t >= 4e-9
        H_ext = [-24.6e-3, 4.3e-3, 0] / mu0
        precess = 1
        alpha = 0.02
    end

    prefactor1 = -gamma / (1 + alpha^2)
    prefactor2 = alpha / Ms

    mx = @view m0[1, :, :, :]
    my = @view m0[2, :, :, :]
    mz = @view m0[3, :, :, :]

    mx_demag_buffer[1:nx, 1:ny, 1:nz] .= (Ms .* mx)
    my_demag_buffer[1:nx, 1:ny, 1:nz] .= (Ms .* my)
    mz_demag_buffer[1:nx, 1:ny, 1:nz] .= (Ms .* mz)

    Hx_demag = calc_demag_field(mx_demag_buffer, my_demag_buffer, mz_demag_buffer, fft_Kxx, fft_Kxy, fft_Kxz)
    Hy_demag = calc_demag_field(mx_demag_buffer, my_demag_buffer, mz_demag_buffer, fft_Kxy, fft_Kyy, fft_Kyz)
    Hz_demag = calc_demag_field(mx_demag_buffer, my_demag_buffer, mz_demag_buffer, fft_Kxz, fft_Kyz, fft_Kzz)

    for index in CartesianIndices((nx, ny, nz))

        i, j, k = Tuple(index)

        #TODO: Add 3rd dimsion to exchange field calulation
        #TODO: Use ParallelStencil.jl to speed up the calculation

        # Neumann boundary conditions

        ip1 = neumann_bc(i + 1, nx)
        im1 = neumann_bc(i - 1, nx)
        jp1 = neumann_bc(j + 1, ny)
        jm1 = neumann_bc(j - 1, ny)

        # Laplacian operator
        m_diff = [
            (m0[comp, ip1, j, k] - 2 * m0[comp, i, j, k] + m0[comp, im1, j, k]) / dx / dx +
            (m0[comp, i, jp1, k] - 2 * m0[comp, i, j, k] + m0[comp, i, jm1, k]) / dy / dy for comp in 1:3
        ]

        #TODO: Use harmonic mean for spatially dependent exchange field
        H_exch = 2 * A / (mu0 * Ms) .* m_diff

        #TODO: Make more concise vector expression
        H_effx = Hx_demag[i, j, k] + H_exch[1] + H_ext[1] # effective field
        H_effy = Hy_demag[i, j, k] + H_exch[2] + H_ext[2]
        H_effz = Hz_demag[i, j, k] + H_exch[3] + H_ext[3]

        H_eff = [H_effx, H_effy, H_effz]

        # TODO: Add anisotropy term 
        # TODO: Add DMI term

        current_m = [mx[i, j, k], my[i, j, k], mz[i, j, k]]

        # TODO: Spatially dependent damping
        dm[:, i, j, k] = prefactor1 .* (precess * cross(current_m, H_eff) +
                                        prefactor2 .* cross(current_m, cross(current_m, H_eff)))

    end # end of LLG loop
end




end_point = 10e-9

tspan = (0, end_point)
t_range = range(0, end_point, length=300)

m0 = zeros(3, nx, ny, nz)
m0[1, :, :, :] .= 1


function H_ext(t)
    if t < 0.5
        return [-2 * t + 1, -2 * t + 1, -2 * t + 1]
    elseif t < 1
        return [0, 0, 0]
    elseif t >= 1
        return [-24.6e-3, 4.3e-3, 0]
    end
end


p = (A, Ms, dx, dy, dz)
prob = ODEProblem(LLG_loop!, m0, tspan, p)
sol = solve(prob, BS3(), progress=true, progress_steps=100)


mx_vals = sol(t_range)[1, 1:nx, 1:ny, 1:nz, :]
my_vals = sol(t_range)[2, 1:nx, 1:ny, 1:nz, :]
mz_vals = sol(t_range)[3, 1:nx, 1:ny, 1:nz, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(t_range, mx_avg, label="mx")
plot!(t_range, my_avg, label="my", color="black")
plot!(t_range, mz_avg, label="mz")
plot!(t_range, m_norm, label="norm")

# norm_mx_vals = sol(t_range)[1, :, :, :, :] ./ 2 .+ .5
# norm_my_vals = sol(t_range)[2, :, :, :, :] ./ 2 .+ .5
# norm_mz_vals = sol(t_range)[3, :, :, :, :] ./ 2 .+ .5

# time_slice = 1
# HSV_img_array = zeros(3, nx, ny)
# xy_angle = atan.(norm_mx_vals, norm_my_vals)[:, :, :, time_slice]
# xy_length = sqrt.(norm_mx_vals .^ 2 + norm_my_vals .^ 2)[:, :, :, time_slice]
# z_height = norm_mz_vals[:, :, :, time_slice]

# HSV_img_array[1, :, :] = xy_angle
# HSV_img_array[2, :, :] = xy_length
# HSV_img_array[3, :, :] = z_height

# @profview solve(prob, OwrenZen3(), progress=true, progress_steps=50)
# @profview solve(prob, OwrenZen3(), progress=true, progress_steps=50)

# colorview(HSV, HSV_img_array)

