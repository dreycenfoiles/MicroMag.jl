# %%
from tkinter import Widget
# import numpy as np
from scipy.integrate import solve_ivp, odeint
import Demag
# from numpy.fft import rfftn, irfftn
from cupy.fft import rfftn, irfftn
from numba import njit, prange, cuda
import matplotlib.pyplot as plt
import cupy as cp
from scipy.integrate._ivp.rk import OdeSolver
import math

old_init = OdeSolver.__init__


def new_init(self, fun, t0, y0, t_bound, vectorized, support_complex=False):
    y_copy = cp.asnumpy(y0)
    old_init(self, fun, t0, y_copy, t_bound, vectorized, support_complex)


OdeSolver.__init__ = new_init



nx = 10
ny = 5
nz = 1

dx = 5e-9
dy = 5e-9
dz = 5e-9

Ms = 8e5
A = 1.3e-11
alpha = 0.02
gamma = 2.211e5
mu0 = 4 * cp.pi * 1e-7

H_exch = cp.zeros((3, nx, ny, nz))
H_demag = cp.zeros((3, nx, ny, nz))

# I copied the four lines below from the Numba docs
threadsperblock = (16, 16, 1)
blockspergrid_x = math.ceil(nx / threadsperblock[0])
blockspergrid_y = math.ceil(ny / threadsperblock[1])
blockspergrid_z = math.ceil(nz / threadsperblock[2])
blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

m = cp.zeros((3, nx, ny, nz))
# m_pad = cp.zeros((3, 2*nx, 2*ny, 2*nz))

Kxx = cp.zeros((nx*2, ny*2, nz*2))
Kyy = cp.zeros((nx*2, ny*2, nz*2))
Kzz = cp.zeros((nx*2, ny*2, nz*2))
Kxy = cp.zeros((nx*2, ny*2, nz*2))
Kxz = cp.zeros((nx*2, ny*2, nz*2))
Kyz = cp.zeros((nx*2, ny*2, nz*2))

Demag.demag_tensor[blockspergrid,threadsperblock](nx, ny, nz, dx, dy, dz, Kxx, Kyy, Kzz, Kxy, Kxz, Kyz)

Kxx_fft = rfftn(Kxx)
Kyy_fft = rfftn(Kyy)
Kzz_fft = rfftn(Kzz)
Kxy_fft = rfftn(Kxy)
Kxz_fft = rfftn(Kxz)
Kyz_fft = rfftn(Kyz)  

# TODO: Make this non-allocating
def demag_field_component(mx_pad, my_pad, mz_pad, fft_kernel1, fft_kernel2, fft_kernel3):

    return irfftn(
        rfftn(mx_pad) * fft_kernel1 +
        rfftn(my_pad) * fft_kernel2 +
        rfftn(mz_pad) * fft_kernel3,
        mx_pad.shape
    )


def demag_field(mx, my, mz, H_demag):

    mx = cp.array(mx)
    my = cp.array(my)
    mz = cp.array(mz)

    nx, ny, nz = cp.shape(mx)

    mx_pad = cp.pad(mx, (0, nx, 0, ny, 0, nz))
    my_pad = cp.pad(my, (0, nx, 0, ny, 0, nz))
    mz_pad = cp.pad(mz, (0, nx, 0, ny, 0, nz))

    Hx_demag = demag_field_component(
        mx_pad, my_pad, mz_pad, Kxx_fft, Kxy_fft, Kxz_fft)
    Hy_demag = demag_field_component(
        mx_pad, my_pad, mz_pad, Kxy_fft, Kyy_fft, Kyz_fft)
    Hz_demag = demag_field_component(
        mx_pad, my_pad, mz_pad, Kxz_fft, Kyz_fft, Kzz_fft)

    H_demag[0] = cp.real(Hx_demag[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])
    H_demag[1] = cp.real(Hy_demag[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])
    H_demag[2] = cp.real(Hz_demag[nx-1:2*nx, ny-1:2*ny, nz-1:2*nz])

    # return H_demag


@cuda.jit
def neumann_bc(x, n):

    if x == n:
        return x - 1
    elif x == -1:
        return x + 1
    else:
        return x


@cuda.jit
def exchange_field(mx, my, mz, H_exch):

    nx, ny, nz = mx.shape

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                ip1 = neumann_bc(i + 1, nx)
                im1 = neumann_bc(i - 1, nx)
                jp1 = neumann_bc(j + 1, ny)
                jm1 = neumann_bc(j - 1, ny)
                kp1 = neumann_bc(k + 1, nz)
                km1 = neumann_bc(k - 1, nz)

                mx_diff = (mx[ip1, j, k] - 2*mx[i, j, k] + mx[im1, j, k]) / dx**2 + \
                          (mx[i, jp1, k] - 2*mx[i, j, k] + mx[i, jm1, k]) / dy**2
                #   (mx[i, j, kp1] - 2*mx[i, j, k] + mx[i, j, km1]) / dz**2

                my_diff = (my[ip1, j, k] - 2*my[i, j, k] + my[im1, j, k]) / dx**2 + \
                          (my[i, jp1, k] - 2*my[i, j, k] + my[i, jm1, k]) / dy**2
                #   (my[i, j, kp1] - 2*my[i, j, k] + my[i, j, km1]) / dz**2

                mz_diff = (mz[ip1, j, k] - 2*mz[i, j, k] + mz[im1, j, k]) / dx**2 + \
                          (mz[i, jp1, k] - 2*mz[i, j, k] + mz[i, jm1, k]) / dy**2
                #   (mz[i, j, kp1] - 2*mz[i, j, k] + mz[i, j, km1]) / dz**2

                H_exch[0, i, j, k] = 2*A/(mu0*Ms**2) * mx_diff
                H_exch[1, i, j, k] = 2*A/(mu0*Ms**2) * my_diff
                H_exch[2, i, j, k] = 2*A/(mu0*Ms**2) * mz_diff

    # return H_exch


def LLG(t, m, H_exch, H_demag):

    print(type(m))

    m = m.reshape((3, nx, ny, nz))

    prefactor1 = -gamma/(1+alpha**2)
    prefactor2 = alpha/Ms

    mx = m[0]
    my = m[1]
    mz = m[2]

    exchange_field[blockspergrid,threadsperblock](mx, my, mz, H_exch)
    demag_field(mx, my, mz, H_demag)

    if t < 2e-9:
        H_ext = cp.array([1, 1, 1]) / mu0
    elif t < 4e-9:
        H_ext = cp.array([0, 0, 0])
    else:
        H_ext = cp.array([-24.6e-3, 4.3e-3, 0]) / mu0

    H_eff = H_exch + H_demag

    H_eff[0] += H_ext[0]
    H_eff[1] += H_ext[1]
    H_eff[2] += H_ext[2]

    dmdt = prefactor1*(cp.cross(m, H_eff, axis=0) + prefactor2 *
                       cp.cross(m, cp.cross(m, H_eff, axis=0), axis=0))

    dmdt = dmdt.reshape((3*nx*ny*nz))

    return dmdt


t_final = 6e-9

m[1, :, :, :] = Ms
m = m.reshape((3*nx*ny*nz))


sol = solve_ivp(LLG, [0, t_final], m, args=(
    H_exch, H_demag), vectorized=False, dense_output=True)


# %%

m = sol.y[:, -1].reshape((3, nx, ny, nz))

# %%
# %matplotlib widget
t_range = cp.linspace(0, t_final, 1000)

mx_avg = []
my_avg = []
mz_avg = []

for t in t_range:
    m_flat = sol.sol(t)
    m_shaped = cp.reshape(m_flat, ((3, nx, ny, nz)))
    mx = m_shaped[0, :, :, :]
    my = m_shaped[1, :, :, :]
    mz = m_shaped[2, :, :, :]
    mx_avg.append(cp.mean(mx))
    my_avg.append(cp.mean(my))
    mz_avg.append(cp.mean(mz))

plt.plot(t_range, mx_avg, label='mx')
plt.plot(t_range, my_avg, label='my')
plt.plot(t_range, mz_avg, label='mz')
plt.plot(t_range, cp.sqrt(cp.array(mx_avg)**2 +
         cp.array(my_avg)**2 + cp.array(mz_avg)**2), label='m')
plt.legend()

# %%
