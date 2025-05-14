import numpy as np
import cupy as cp
from pyxalign.api.types import r_type
from pyxalign.gpu_utils import memory_releasing_error_handler
from pyxalign.timing.timer_utils import timer

timerOn = False

@memory_releasing_error_handler
def div(P):
    xp = cp.get_array_module(P)

    Px = P[0, :, :, :]
    Py = P[1, :, :, :]
    Pz = P[2, :, :, :]

    L, M, N = P.shape[1:]

    idx = xp.array([0] + [i for i in range(P.shape[1] - 1)])
    fx = Px - Px[idx, :, :]
    # fx = np.diff(Px, axis=0)
    # fx = np.append(np.zeros((1, M, N)), fx, axis=0)

    idx = xp.array([0] + [i for i in range(P.shape[2] - 1)])
    fy = Py - Py[:, idx, :]
    # fy = np.diff(Py, axis=1)
    # fy = np.append(np.zeros((L, 1, N)), fy, axis=1)

    idx = xp.array([0] + [i for i in range(P.shape[3] - 1)])
    fz = Pz - Pz[:, :, idx]
    # fz = np.diff(Pz, axis=2)
    # fz = np.append(np.zeros((L, M, 1)), fz, axis=2)

    fd = fx + fy + fz

    return fd

@memory_releasing_error_handler
def grad(M):
    xp = cp.get_array_module(M)

    idx = np.array([i + 1 for i in range(M.shape[0] - 1)] + [M.shape[0] - 1])
    fx = M[idx, :, :] - M
    idx = np.array([i + 1 for i in range(M.shape[1] - 1)] + [M.shape[1] - 1])
    fy = M[:, idx, :] - M
    idx = np.array([i + 1 for i in range(M.shape[2] - 1)] + [M.shape[2] - 1])
    fz = M[:, :, idx] - M

    f = np.stack([fx, fy, fz])
    return f

@memory_releasing_error_handler
@timer()
def chambolleLocalTV3D(x, alpha, Niter):
    if alpha == 0:
        return x

    xp = cp.get_array_module(x)

    x0 = x
    tau = 1 / 4
    (L, M, N) = x.shape
    xi = xp.zeros((3, L, M, N), dtype=r_type)

    for i in range(Niter):
        # Chambolle step
        gdv = grad(div(xi) - x / alpha)
        # lam.utils.timerEnd(t0, "Chambolle Step", False)

        # Anisotropic

        # d = xp.abs(gdv.sum(axis=0))
        d = xp.abs(gdv).sum(axis=0)
        xi = (xi + tau * gdv) / (1 + tau * d)
        # lam.utils.timerEnd(t0, "Anisotropic", False)

        # Reconstruct

        x = x - alpha * div(xi)
        # print("iteration", i)

    # prevent pushing values to zero by the TV regularization
    x = xp.sum(x0 * x) / xp.sum(x**2) * x
    return x
