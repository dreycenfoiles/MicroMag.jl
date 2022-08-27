
from numba import cuda
from cupyx.scipy.fft import rfftn, irfftn
import cupy as cp

@cuda.jit
def demag_tensor(nx, ny, nz, dx, dy, dz, Kxx, Kyy, Kzz, Kxy, Kxz, Kyz):
    """
        Calculate the demagnetization tensor.
        Numba is used to accelerate the calculation.
        Inputs: nx, ny, nz: number of cells in x/y/z,
                dx, dy, dz: cellsizes in x/y/z,
                z_off: optional offset in z direction (integer with units of dz)
        Outputs: demag tensor elements (numpy.array)
        """
    # Initialization of demagnetization tensor
    

    for K in range(-nz+1, nz):
        for J in range(-ny+1, ny):
            for I in range(-nx+1, nx):
                # non-negative indices
                L, M, N = (I+nx-1), (J+ny-1), (K+nz-1)
                for i in (-0.5, 0.5):
                    for j in (-0.5, 0.5):
                        for k in (-0.5, 0.5):
                            sgn = (-1)**(i+j+k+1.5)/(4*cp.pi)
                            r = cp.sqrt(((I+i)*dx)**2 + ((J+j)*dy)
                                        ** 2 + ((K+k)*dz)**2)
                            Kxx[L, M, N] += sgn * \
                                cp.arctan((K+k)*(J+j)*dz*dy/(r*(I+i)*dx))
                            Kyy[L, M, N] += sgn * \
                                cp.arctan((I+i)*(K+k)*dx*dz/(r*(J+j)*dy))
                            Kzz[L, M, N] += sgn * \
                                cp.arctan((J+j)*(I+i)*dy*dx/(r*(K+k)*dz))
                            Kxy[L, M, N] -= sgn * \
                                cp.log(cp.abs((K+k)*dz + r))
                            Kxz[L, M, N] -= sgn * \
                                cp.log(cp.abs((J+j)*dy + r))
                            Kyz[L, M, N] -= sgn * \
                                cp.log(cp.abs((I+i)*dx + r))
  

