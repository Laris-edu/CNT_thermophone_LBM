"""Strict compiled memory passes for finite-strip populations, without physics laws."""
import math

import numpy as np
from numba import njit, prange

from core.collision_numba import _parallel_field


@njit(cache=True, fastmath=False, nogil=True)
def _stream_pair(f, g, c):
    ny, nx, q = f.shape
    fn, gn = np.empty_like(f), np.empty_like(g)
    for cell in prange(ny*nx):
        y, x = cell//nx, cell % nx
        for a in range(q):
            sy, sx = y-c[a, 1], x-c[a, 0]
            if 0 <= sy < ny and 0 <= sx < nx:
                fn[y, x, a], gn[y, x, a] = f[sy, sx, a], g[sy, sx, a]
            else:
                fn[y, x, a], gn[y, x, a] = np.nan, np.nan
    return fn, gn


@njit(cache=True, fastmath=False, nogil=True)
def _cell_statistics(f, g, kinetic):
    cells, q = f.shape
    mass, energy = np.empty(cells), np.empty(cells)
    minimum_f, minimum_g = np.empty(cells), np.empty(cells)
    finite = np.ones(cells, dtype=np.bool_)
    for cell in prange(cells):
        total_mass = total_energy = 0.
        low_f = low_g = math.inf
        for a in range(q):
            fa, ga = f[cell, a], g[cell, a]
            total_mass += fa
            total_energy += fa*kinetic[a]+ga
            low_f, low_g = min(low_f, fa), min(low_g, ga)
            if not math.isfinite(fa) or not math.isfinite(ga):
                finite[cell] = False
        mass[cell], energy[cell] = total_mass, total_energy
        minimum_f[cell], minimum_g[cell] = low_f, low_g
    return mass, energy, minimum_f, minimum_g, finite


_stream_pair_parallel = njit(fastmath=False, nogil=True, parallel=True)(_stream_pair.py_func)
_cell_statistics_parallel = njit(fastmath=False, nogil=True, parallel=True)(_cell_statistics.py_func)


def stream_open_xy_pair(f, g, c):
    """Exactly the open pull permutation; exterior source links remain NaN."""
    if f.shape != g.shape or f.ndim != 3 or f.shape[-1] != len(c):
        raise ValueError('paired open streaming requires equal (ny,nx,Q) populations')
    stream = _stream_pair_parallel if _parallel_field(f.size//f.shape[-1]) else _stream_pair
    return stream(np.ascontiguousarray(f), np.ascontiguousarray(g), np.asarray(c, dtype=np.int64))


def population_statistics(f, g, kinetic):
    """Read actual mass, raw total energy, extrema and finiteness together.

    Reduce the contiguous per-cell sums with NumPy's pairwise summation rather
    than a many-million-term sequential sum or nondeterministic thread reduction.
    """
    q = len(kinetic)
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != q:
        raise ValueError('population statistics require equal shapes ending in Q')
    measure = _cell_statistics_parallel if _parallel_field(f.size//q) else _cell_statistics
    mass, energy, low_f, low_g, finite = measure(np.ascontiguousarray(f).reshape(-1, q),
        np.ascontiguousarray(g).reshape(-1, q), np.asarray(kinetic, dtype=float))
    return float(mass.sum()), float(energy.sum()), float(low_f.min()), float(low_g.min()), bool(finite.all())
