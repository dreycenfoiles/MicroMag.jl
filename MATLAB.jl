# using FFTW
using Statistics
using CUDA
using CUDA.CUFFT
using DifferentialEquations
using Plots
# using ProfileView

using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())

CUDA.allowscalar(true)

nx = 166; # number of cells on x direction
ny = 42;
nz = 1;
dd = 3; # cell volume = dd x dd x dd
dt = 5E-6; # timestep in nanoseconds
# dt = CuArray(dt); # copy data from CPU to GPU
timesteps = 150000;
alpha = 0.5; # damping constant to relax system to S-state
# alpha = CuArray(alpha);
exchConstant = 1.3E-11 * 1E18; # nanometer/nanosecond units
mu_0 = 1.256636; # vacuum permeability, = 4 * pi / 10
Ms = 800; # saturation magnetization
# Ms = CuArray(Ms);
exch = 2 * exchConstant / mu_0 / Ms / Ms;
# exch = CuArray(exch);
prefactor1 = (-0.221) * dt / (1 + alpha * alpha);
prefactor2 = prefactor1 * alpha / Ms;
# prefactor1 = CuArray(prefactor1);
# prefactor2 = CuArray(prefactor2);
Mx = ones(nx, ny, nz) .* Ms; # magnetization on x direction 
My = zeros(nx, ny, nz); # magnetization on y direction
Mz = zeros(nx, ny, nz); # magnetization on z direction

Mx = CuArray(Mx);
My = CuArray(My);
Mz = CuArray(Mz);

deltaMx = CUDA.zeros(nx, ny, nz);
deltaMy = CUDA.zeros(nx, ny, nz);
deltaMz = CUDA.zeros(nx, ny, nz);
mag = CUDA.zeros(nx, ny, nz);

Mx_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);
My_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);
Mz_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);

Hx_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);
Hy_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);
Hz_pad = CUDA.zeros(nx * 2, ny * 2, nz * 2);

# p = plan_rfft(Mx_pad)
# ip = plan_irfft(Mx_pad,)


mx_avg = CUDA.zeros(floor(Int, timesteps / 1000))
my_avg = CUDA.zeros(floor(Int, timesteps / 1000))
mz_avg = CUDA.zeros(floor(Int, timesteps / 1000))

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
            for i = 0:1 # helper indices
                for j = 0:1
                    for k = 0:1
                        r = sqrt((I + i - 0.5) * (I + i - 0.5) * dd * dd + (J + j - 0.5) * (J + j - 0.5) * dd * dd + (K + k - 0.5) * (K + k - 0.5) * dd * dd)
                        Kxx[L, M, N] += (-1) .^ (i + j + k) * atan((K + k - 0.5) * (J + j - 0.5) * dd / r / (I + i - 0.5))
                        Kxy[L, M, N] += (-1) .^ (i + j + k) * log((K + k - 0.5) * dd + r)
                        Kxz[L, M, N] += (-1) .^ (i + j + k) * log((J + j - 0.5) * dd + r)
                        Kyy[L, M, N] += (-1) .^ (i + j + k) * atan((I + i - 0.5) * (K + k - 0.5) * dd / r / (J + j - 0.5))
                        Kyz[L, M, N] += (-1) .^ (i + j + k) * log((I + i - 0.5) * dd + r)
                        Kzz[L, M, N] += (-1) .^ (i + j + k) * atan((J + j - 0.5) * (I + i - 0.5) * dd / r / (K + k - 0.5))
                    end
                end
            end
            Kxx[L, M, N] *= prefactor
            Kxy[L, M, N] *= -prefactor
            Kxz[L, M, N] *= -prefactor
            Kyy[L, M, N] *= prefactor
            Kyz[L, M, N] *= -prefactor
            Kzz[L, M, N] *= prefactor
        end
    end
end # calculation of demag tensor done

Kxx = CuArray(Kxx);
Kxy = CuArray(Kxy);
Kxz = CuArray(Kxz);
Kyy = CuArray(Kyy);
Kyz = CuArray(Kyz);
Kzz = CuArray(Kzz);

Kxx_fft = rfft(Kxx); # fast fourier transform of demag tensor
Kxy_fft = rfft(Kxy); # need to be done only one time
Kxz_fft = rfft(Kxz);
Kyy_fft = rfft(Kyy);
Kyz_fft = rfft(Kyz);
Kzz_fft = rfft(Kzz);

plan = plan_rfft(Kxx);
iplan = plan_irfft(Kxx_fft, 2 * nx);


Hx_exch = CUDA.zeros(nx, ny, nz);
Hy_exch = CUDA.zeros(nx, ny, nz);
Hz_exch = CUDA.zeros(nx, ny, nz);

# outFile = open("Mdata.txt", "w");

Hx0 = CUDA.zeros(nx, ny, nz);
Hx1 = CUDA.zeros(nx, ny, nz);
Hx2 = CUDA.zeros(nx, ny, nz);
Hx3 = CUDA.zeros(nx, ny, nz);
Hy0 = CUDA.zeros(nx, ny, nz);
Hy1 = CUDA.zeros(nx, ny, nz);
Hy2 = CUDA.zeros(nx, ny, nz);
Hy3 = CUDA.zeros(nx, ny, nz);
Hz0 = CUDA.zeros(nx, ny, nz);
Hz1 = CUDA.zeros(nx, ny, nz);
Hz2 = CUDA.zeros(nx, ny, nz);
Hz3 = CUDA.zeros(nx, ny, nz);

function LLG_loop!(dm, m0, p, t)

    global alpha
    global prefactor1
    global prefactor2
    global mag

    Mx = m0[1, :, :, :]
    My = m0[2, :, :, :]
    Mz = m0[3, :, :, :]

    fill!(Mx_pad, 0)
    fill!(My_pad, 0)
    fill!(Mz_pad, 0)

    Mx_pad[1:nx, 1:ny, 1:nz] = Mx
    My_pad[1:nx, 1:ny, 1:nz] = My
    Mz_pad[1:nx, 1:ny, 1:nz] = Mz

    # Hx_pad = irfft(rfft(Mx_pad) .* Kxx_fft + rfft(My_pad) .* Kxy_fft + rfft(Mz_pad) .* Kxz_fft, 2 * nx) # calc demag field with fft
    Hx_pad = iplan * (
        (plan * Mx_pad) .* Kxx_fft +
        (plan * My_pad) .* Kxy_fft +
        (plan * Mz_pad) .* Kxz_fft
    ) # calc demag field with fft

    Hy_pad = iplan * (
        (plan * Mx_pad) .* Kxy_fft +
        (plan * My_pad) .* Kyy_fft +
        (plan * Mz_pad) .* Kyz_fft
    )

    Hz_pad = iplan * (
        (plan * Mx_pad) .* Kxz_fft +
        (plan * My_pad) .* Kyz_fft +
        (plan * Mz_pad) .* Kzz_fft
    )
    # Hy_pad = irfft(rfft(Mx_pad) .* Kxy_fft + rfft(My_pad) .* Kyy_fft + rfft(Mz_pad) .* Kyz_fft, 2 * nx)
    # Hz_pad = irfft(rfft(Mx_pad) .* Kxz_fft + rfft(My_pad) .* Kyz_fft + rfft(Mz_pad) .* Kzz_fft, 2 * nx)
    Hx = real(Hx_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)]) # truncation of demag field
    Hy = real(Hy_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])
    Hz = real(Hz_pad[nx:(2*nx-1), ny:(2*ny-1), nz:(2*nz-1)])
    # Mx = Mx_pad[1:nx, 1:ny, 1:nz] # truncation of Mx, remove zero padding
    # My = My_pad[1:nx, 1:ny, 1:nz]
    # Mz = Mz_pad[1:nx, 1:ny, 1:nz]
    # calculation of exchange field
    Hx0[2:end, :, :] = Mx[1:end-1, :, :]
    Hx0[1, :, :] = Hx0[2, :, :]
    Hx1[1:end-1, :, :] = Mx[2:end, :, :]
    Hx1[end, :, :] = Hx1[end-1, :, :]

    Hx2[:, 2:end, :] = Mx[:, 1:end-1, :]
    Hx2[:, 1, :] = Hx2[:, 2, :]
    Hx3[:, 1:end-1, :] = Mx[:, 2:end, :]
    Hx3[:, end, :] = Hx3[:, end-1, :]

    Hy0[2:end, :, :] = My[1:end-1, :, :]
    Hy0[1, :, :] = Hy0[2, :, :]
    Hy1[1:end-1, :, :] = My[2:end, :, :]
    Hy1[end, :, :] = Hy1[end-1, :, :]

    Hy2[:, 2:end, :] = My[:, 1:end-1, :]
    Hy2[:, 1, :] = Hy2[:, 2, :]
    Hy3[:, 1:end-1, :] = My[:, 2:end, :]
    Hy3[:, end, :] = Hy3[:, end-1, :]

    Hz0[2:end, :, :] = Mz[1:end-1, :, :]
    Hz0[1, :, :] = Hz0[2, :, :]
    Hz1[1:end-1, :, :] = Mz[2:end, :, :]
    Hz1[end, :, :] = Hz1[end-1, :, :]

    Hz2[:, 2:end, :] = Mz[:, 1:end-1, :]
    Hz2[:, 1, :] = Hz2[:, 2, :]
    Hz3[:, 1:end-1, :] = Mz[:, 2:end, :]
    Hz3[:, end, :] = Hz3[:, end-1, :]

    Hx += exch / dd / dd * (Hx0 + Hx1 + Hx2 + Hx3 - 4 * Mx)
    Hy += exch / dd / dd * (Hy0 + Hy1 + Hy2 + Hy3 - 4 * My)
    Hz += exch / dd / dd * (Hz0 + Hz1 + Hz2 + Hz3 - 4 * Mz)

    if t < 4000
        Hx .+= 100 # apply a saturation field to get S-state
        Hy .+= 100
        Hz .+= 100
    elseif t < 6000
        Hx .+= (6000 - t) / 20 # gradually diminish the field
        Hx .+= (6000 - t) / 20
        Hx .+= (6000 - t) / 20
    elseif t > 50000
        Hx .+= -19.576 # apply the reverse field
        Hy .+= +3.422
        alpha = 0.02
        prefactor1 = (-0.221) * dt / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha / Ms
    end
    # apply LLG equation
    MxHx = My .* Hz - Mz .* Hy # = M cross H
    MxHy = Mz .* Hx - Mx .* Hz
    MxHz = Mx .* Hy - My .* Hx

    deltaMx = prefactor1 .* MxHx + prefactor2 .* (My .* MxHz - Mz .* MxHy)
    deltaMy = prefactor1 .* MxHy + prefactor2 .* (Mz .* MxHx - Mx .* MxHz)
    deltaMz = prefactor1 .* MxHz + prefactor2 .* (Mx .* MxHy - My .* MxHx)

    dm[1, :, :, :] = deltaMx
    dm[2, :, :, :] = deltaMy
    dm[3, :, :, :] = deltaMz

    # Mx += deltaMx
    # My += deltaMy
    # Mz += deltaMz

    # mag = sqrt.(Mx .* Mx + My .* My + Mz .* Mz)
    # Mx = Mx ./ mag * Ms
    # My = My ./ mag * Ms
    # Mz = Mz ./ mag * Ms

    # if t % 1000 == 0 # recod the average magnetization
    #     print(t)
    #     MxMean = mean(Mx)
    #     MyMean = mean(My)
    #     MzMean = mean(Mz)
    #     mx_avg[floor(Int,t/1000)] = MxMean
    #     my_avg[floor(Int,t/1000)] = MyMean
    #     mz_avg[floor(Int,t/1000)] = MzMean
    # write(outFile, "#d\t#f\t#f\t#f\r\n", t, MxMean / Ms, MyMean / Ms, MzMean / Ms)
end
# close(outFile);

end_point = 200000
tspan = (0, end_point)
t_range = range(0, end_point, length=300)

m0 = CUDA.zeros(3, nx, ny, nz)
m0[1, :, :, :] = Mx
m0[2, :, :, :] = My
m0[3, :, :, :] = Mz

p = ()


prob = ODEProblem(LLG_loop!, m0, tspan, p)
# @profview sol = solve(prob, BS3(), progress=true, progress_steps=100)
sol = solve(prob, BS3(), progress=true, progress_steps=100)



# The '...' is absolutely necessary here. It's called splatting and I don't know 
# how it works.
cpu_sol = cat([Array(x) for x in sol.u]...,dims=5)

mx_vals = cpu_sol[1, 1:nx, 1:ny, 1:nz, :]
my_vals = cpu_sol[2, 1:nx, 1:ny, 1:nz, :]
mz_vals = cpu_sol[3, 1:nx, 1:ny, 1:nz, :]

mx_avg = mean(mx_vals, dims=[1, 2, 3])[1, 1, 1, :]
my_avg = mean(my_vals, dims=[1, 2, 3])[1, 1, 1, :]
mz_avg = mean(mz_vals, dims=[1, 2, 3])[1, 1, 1, :]

m_norm = sqrt.(mx_avg .^ 2 + my_avg .^ 2 + mz_avg .^ 2)

plot(sol.t, mx_avg, label="mx")
plot!(sol.t, my_avg, label="my", color="black")
plot!(sol.t, mz_avg, label="mz")
plot!(sol.t, m_norm, label="norm")