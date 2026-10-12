"""Strict compiled acoustic-memory RHS; no equilibrium or transport laws."""
import numpy as np
from numba import njit, prange

from core.collision_numba import _parallel_field


@njit(cache=True, fastmath=False, inline='always')
def _dx(v, channel, y, x, nx):
    if x == 0:
        return (-3*v[channel, y, 0]+4*v[channel, y, 1]-v[channel, y, 2])/2
    if x == nx-1:
        return (3*v[channel, y, x]-4*v[channel, y, x-1]+v[channel, y, x-2])/2
    if x == 1 or x == nx-2:
        return (v[channel, y, x+1]-v[channel, y, x-1])/2
    return (v[channel, y, x-2]-8*v[channel, y, x-1]
            +8*v[channel, y, x+1]-v[channel, y, x+2])/12


@njit(cache=True, fastmath=False, inline='always')
def _dy(v, channel, y, x, ny):
    if y == 0:
        return (-3*v[channel, 0, x]+4*v[channel, 1, x]-v[channel, 2, x])/2
    if y == ny-1:
        return (3*v[channel, y, x]-4*v[channel, y-1, x]+v[channel, y-2, x])/2
    if y == 1 or y == ny-2:
        return (v[channel, y+1, x]-v[channel, y-1, x])/2
    return (v[channel, y-2, x]-8*v[channel, y-1, x]
            +8*v[channel, y+1, x]-v[channel, y+2, x])/12


@njit(cache=True, fastmath=False, nogil=True, error_model='numpy')
def _rhs_cells(v, rho, c2, sigma_x, sigma_y, decay_x, decay_y):
    _, ny, nx = v.shape
    out = np.empty_like(v)
    for cell in prange(ny*nx):
        y, x = cell//nx, cell % nx
        px, py = v[3, y, x], v[4, y, x]
        vx, vy = v[5, y, x], v[6, y, x]
        out[0, y, x] = -rho[y, x]*c2[y, x]*(vx+vy)
        out[1, y, x] = -px/rho[y, x]
        out[2, y, x] = -py/rho[y, x]
        out[3, y, x] = -decay_x[y, x]*px-sigma_x[y, x]*_dx(v, 0, y, x, nx)
        out[4, y, x] = -decay_y[y, x]*py-sigma_y[y, x]*_dy(v, 0, y, x, ny)
        out[5, y, x] = -decay_x[y, x]*vx-sigma_x[y, x]*_dx(v, 1, y, x, nx)
        out[6, y, x] = -decay_y[y, x]*vy-sigma_y[y, x]*_dy(v, 2, y, x, ny)
    return out


_rhs_cells_parallel = njit(fastmath=False, nogil=True, parallel=True,
    error_model='numpy')(_rhs_cells.py_func)


def acoustic_rhs(v, rho, c2, sigma_x, sigma_y, decay_x, decay_y):
    """Match AcousticPML._rhs on unit-grid coordinates, including every edge.

    Scalars or broadcastable cell arrays supply the background and damping.
    Inputs are read-only; output is a newly allocated float64 (7,ny,nx) array.
    The shared CPU dispatch threshold selects serial or independent-cell work.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 3 or v.shape[0] != 7 or min(v.shape[1:]) < 3:
        raise ValueError('acoustic RHS requires (7,ny,nx) with ny,nx>=3')
    shape = v.shape[1:]
    coefficients = tuple(np.ascontiguousarray(np.broadcast_to(
        np.asarray(value, dtype=np.float64), shape))
        for value in (rho, c2, sigma_x, sigma_y, decay_x, decay_y))
    evaluate = _rhs_cells_parallel if _parallel_field(shape[0]*shape[1]) else _rhs_cells
    return evaluate(np.ascontiguousarray(v), *coefficients)
