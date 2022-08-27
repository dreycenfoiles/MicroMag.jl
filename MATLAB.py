import numpy as np
# from numpy.fft import fftn, ifftn
import cupy as cp
from cupy.fft import fftn, ifftn

nx = 166  # number of cells on x direction
ny = 42
nz = 1
dd = 3  # cell volume = dd x dd x dd
dt = 5E-6  # timestep in nanoseconds
timesteps = 150000
alpha = 0.5  # damping constant to relax system to S-state
exchConstant = 1.3E-11 * 1E18  # nanometer/nanosecond units
mu_0 = 1.256636  # vacuum permeability, = 4 * pi / 10
Ms = 800  # saturation magnetization
exch = 2 * exchConstant / mu_0 / Ms / Ms
prefactor1 = (-0.221) * dt / (1 + alpha * alpha)
prefactor2 = prefactor1 * alpha / Ms

Mx = cp.ones((nx, ny, nz)) * Ms  # magnetization on x direction
My = cp.zeros((nx, ny, nz))
Mz = cp.zeros((nx, ny, nz))

Mx_pad = cp.zeros((2*nx, 2*ny, 2*nz))
My_pad = cp.zeros((2*nx, 2*ny, 2*nz))
Mz_pad = cp.zeros((2*nx, 2*ny, 2*nz))

deltaMx = cp.zeros((nx, ny, nz))
deltaMy = cp.zeros((nx, ny, nz))
deltaMz = cp.zeros((nx, ny, nz))
mag = cp.zeros((nx, ny, nz))

# Initialization of demagnetization tensor
Kxx = cp.zeros((nx * 2, ny * 2, nz * 2))
Kxy = cp.zeros((nx * 2, ny * 2, nz * 2))
Kxz = cp.zeros((nx * 2, ny * 2, nz * 2))
Kyy = cp.zeros((nx * 2, ny * 2, nz * 2))
Kyz = cp.zeros((nx * 2, ny * 2, nz * 2))
Kzz = cp.zeros((nx * 2, ny * 2, nz * 2))
prefactor = 1 / 4 / 3.14159265

for K in range(-nz, nz):  # Calculation of Demag tensor
    for J in range(-ny, ny):
        for I in range(-nx, nx):
            if I == 0 & J == 0 & K == 0:
                continue
            L = I + nx 
            M = J + ny
            N = K + nz

            for i in (-0.5, 0.5):
                for j in (-0.5, 0.5):
                    for k in (-0.5, 0.5):
                        sgn = (-1)**(i+j+k+1.5)/(4*np.pi)
                        r = np.sqrt(((I+i)*dd)**2 + ((J+j)*dd)**2 + ((K+k)*dd)**2)
                        Kxx[L, M, N] += sgn * \
                            np.arctan((K+k)*(J+j)*dd*dd/(r*(I+i)*dd))
                        Kyy[L, M, N] += sgn * \
                            np.arctan((I+i)*(K+k)*dd*dd/(r*(J+j)*dd))
                        Kzz[L, M, N] += sgn * \
                            np.arctan((J+j)*(I+i)*dd*dd/(r*(K+k)*dd))
                        Kxy[L, M, N] -= sgn * np.log(abs((K+k)*dd + r))
                        Kxz[L, M, N] -= sgn * np.log(abs((J+j)*dd + r))
                        Kyz[L, M, N] -= sgn * np.log(abs((I+i)*dd + r))

            Kxx[L, M, N] *= prefactor
            Kxy[L, M, N] *= - prefactor
            Kxz[L, M, N] *= - prefactor
            Kyy[L, M, N] *= prefactor
            Kyz[L, M, N] *= - prefactor
            Kzz[L, M, N] *= prefactor


Kxx_fft = fftn(Kxx)
# fast fourier transform of demag tensor
Kxy_fft = fftn(Kxy)
# need to be done only one time
Kxz_fft = fftn(Kxz)
Kyy_fft = fftn(Kyy)
Kyz_fft = fftn(Kyz)
Kzz_fft = fftn(Kzz)

Hx_exch = cp.zeros((nx, ny, nz))
Hy_exch = cp.zeros((nx, ny, nz))
Hz_exch = cp.zeros((nx, ny, nz))

outFile = open('Mdata.txt', 'w')

Hx0 = cp.zeros((nx, ny, nz))
Hx1 = cp.zeros((nx, ny, nz))
Hx2 = cp.zeros((nx, ny, nz))
Hx3 = cp.zeros((nx, ny, nz))
Hy0 = cp.zeros((nx, ny, nz))
Hy1 = cp.zeros((nx, ny, nz))
Hy2 = cp.zeros((nx, ny, nz))
Hy3 = cp.zeros((nx, ny, nz))
Hz0 = cp.zeros((nx, ny, nz))
Hz1 = cp.zeros((nx, ny, nz))
Hz2 = cp.zeros((nx, ny, nz))
Hz3 = cp.zeros((nx, ny, nz))

for t in range(timesteps):
    # Mx[nx:, ny:, nz:] = 0  # zero padding
    # My[nx:, ny:, nz:] = 0
    # Mz[nx:, ny:, nz:] = 0

    Mx_pad = cp.pad(Mx, (0, nx, 0, ny, 0, nz))
    My_pad = cp.pad(My, (0, nx, 0, ny, 0, nz))
    Mz_pad = cp.pad(Mz, (0, nx, 0, ny, 0, nz))

    Hx = ifftn(fftn(Mx_pad) * Kxx_fft + fftn(My_pad)
               * Kxy_fft + fftn(Mz_pad) * Kxz_fft)
    # calc demag field with fft
    Hy = ifftn(fftn(Mx_pad) * Kxy_fft + fftn(My_pad)
               * Kyy_fft + fftn(Mz_pad) * Kyz_fft)
    Hz = ifftn(fftn(Mx_pad) * Kxz_fft + fftn(My_pad)
               * Kyz_fft + fftn(Mz_pad) * Kzz_fft)
    # truncation of demag field
    Hx = cp.real(Hx[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])
    Hy = cp.real(Hy[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])
    Hz = cp.real(Hz[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])

    # Mx = Mx_pad[nx-1:2*nx-1,ny-1:2*ny-1,nz-1:2*nz-1]  # truncation of Mx, remove zero padding
    # My = My_pad[nx-1:2*nx-1,ny-1:2*ny-1,nz-1:2*nz-1]
    # Mz = Mz_pad[nx-1:2*nx-1,ny-1:2*ny-1,nz-1:2*nz-1]
    # calculation of exchange field

    Hx0[1:, :, :] = Mx[:-1, :, :]
    Hx0[0, :, :] = Hx0[1, :, :]
    Hx1[:-1, :, :] = Mx[1:, :, :]
    Hx1[-1, :, :] = Hx1[-2, :, :]

    Hx2[:, 1:, :] = Mx[:, :-1, :]
    Hx2[:, 0, :] = Hx2[:, 1, :]
    Hx3[:, :-1, :] = Mx[:, 1:, :]
    Hx3[:, -1, :] = Hx3[:, -2, :]

    Hy0[1:, :, :] = My[:-1, :, :]
    Hy0[0, :, :] = Hy0[1, :, :]
    Hy1[:-1, :, :] = My[1:, :, :]
    Hy1[-1, :, :] = Hy1[-2, :, :]

    Hy2[:, 1:, :] = My[:, :-1, :]
    Hy2[:, 0, :] = Hy2[:, 1, :]
    Hy3[:, :-1, :] = My[:, 1:, :]
    Hy3[:, -1, :] = Hy3[:, -2, :]

    Hz0[1:, :, :] = Mz[:-1, :, :]
    Hz0[0, :, :] = Hz0[1, :, :]
    Hz1[:-1, :, :] = Mz[1:, :, :]
    Hz1[-1, :, :] = Hz1[-2, :, :]

    Hz2[:, 1:, :] = Mz[:, :-1, :]
    Hz2[:, 0, :] = Hz2[:, 1, :]
    Hz3[:, :-1, :] = Mz[:, 1:, :]
    Hz3[:, -1, :] = Hz3[:, -2, :]

    Hx += exch / dd / dd * (Hx0 + Hx1 + Hx2 + Hx3 - 4 * Mx)
    Hy += exch / dd / dd * (Hy0 + Hy1 + Hy2 + Hy3 - 4 * My)
    Hz += exch / dd / dd * (Hz0 + Hz1 + Hz2 + Hz3 - 4 * Mz)

    if t < 4000:
        Hx = Hx + 100
        # apply a saturation field to get S-state
        Hy = Hy + 100
        Hz = Hz + 100
    elif t < 6000:
        Hx = Hx + (6000 - t) / 20
        # gradually diminish the field
        Hx = Hx + (6000 - t) / 20
        Hx = Hx + (6000 - t) / 20
    elif t > 50000:
        Hx = Hx - 19.576
        # apply the reverse field
        Hy = Hy + 3.422
        alpha = 0.02
        prefactor1 = (-0.221) * dt / (1 + alpha * alpha)
        prefactor2 = prefactor1 * alpha / Ms

    # apply LLG equation
    MxHx = My * Hz - Mz * Hy  # = M cross H
    MxHy = Mz * Hx - Mx * Hz
    MxHz = Mx * Hy - My * Hx

    deltaMx = prefactor1 * MxHx + prefactor2 * (My * MxHz - Mz * MxHy)
    deltaMy = prefactor1 * MxHy + prefactor2 * (Mz * MxHx - Mx * MxHz)
    deltaMz = prefactor1 * MxHz + prefactor2 * (Mx * MxHy - My * MxHx)

    Mx = Mx + deltaMx
    My = My + deltaMy
    Mz = Mz + deltaMz

    mag = np.sqrt(Mx * Mx + My * My + Mz * Mz)
    Mx = Mx / mag * Ms
    My = My / mag * Ms
    Mz = Mz / mag * Ms

    if t % 1000 == 0:  # recod the average magnetization
        print(t)
        MxMean = cp.mean(Mx)
        MyMean = cp.mean(My)
        MzMean = cp.mean(Mz)
        outFile.write(f"{t}\t{MxMean/Ms}\t{MyMean/Ms}\t{MzMean/Ms}\r\n")

outFile.close()
