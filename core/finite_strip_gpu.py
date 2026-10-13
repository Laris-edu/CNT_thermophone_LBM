"""CUDA memory passes for the finite strip; no transport or boundary laws."""
from functools import lru_cache

import numpy as np


_SOURCE = r'''
extern "C" __global__ void stream_pair(const double* f, const double* g,
        const int* c, const long long nx, const long long ny, double* fn, double* gn) {
    const long long i = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= nx*ny*37) return;
    const int a = i%37;
    const long long cell = i/37, x = cell%nx, y = cell/nx;
    const long long sx = x-c[2*a], sy = y-c[2*a+1];
    if (sx >= 0 && sx < nx && sy >= 0 && sy < ny) {
        const long long source = (sy*nx+sx)*37+a;
        fn[i] = f[source]; gn[i] = g[source];
    } else { fn[i] = nan(""); gn[i] = nan(""); }
}
extern "C" __global__ void statistics(const double* f, const double* g,
        const double* kinetic, const long long cells, double* out) {
    const long long i = (long long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i >= cells) return;
    double mass=0., energy=0., minf=f[i*37], ming=g[i*37], valid=1.;
    for (int a=0; a<37; ++a) {
        const double fa=f[i*37+a], ga=g[i*37+a];
        mass+=fa; energy+=kinetic[a]*fa+ga;
        minf=fmin(minf,fa); ming=fmin(ming,ga);
        if (!isfinite(fa) || !isfinite(ga)) valid=0.;
    }
    out[i]=mass; out[cells+i]=energy; out[2*cells+i]=minf;
    out[3*cells+i]=ming; out[4*cells+i]=valid;
}
'''


@lru_cache(maxsize=8)
def _kernels(device_id):
    import cupy as cp
    return tuple(cp.RawKernel(_SOURCE, name, options=('--fmad=false',))
                 for name in ('stream_pair', 'statistics'))


@lru_cache(maxsize=8)
def _directions(device_id, values):
    import cupy as cp
    return cp.asarray(values, dtype=cp.int32)


def stream_open_xy_pair(f, g, c):
    import cupy as cp
    if f.shape != g.shape or f.ndim != 3 or f.shape[-1] != 37:
        raise ValueError('CUDA strip streaming requires matching (ny,nx,37) arrays')
    f, g = cp.ascontiguousarray(f, dtype=cp.float64), cp.ascontiguousarray(g, dtype=cp.float64)
    ny, nx = f.shape[:2]
    directions = _directions(cp.cuda.Device().id, tuple(map(tuple, c)))
    fn, gn = cp.empty_like(f), cp.empty_like(g)
    _kernels(cp.cuda.Device().id)[0](((f.size+255)//256,), (256,),
                                   (f, g, directions, np.int64(nx), np.int64(ny), fn, gn))
    return fn, gn


def population_statistics(f, g, kinetic):
    """One scalar download; sums read actual populations, never equation RHS."""
    import cupy as cp
    if f.shape != g.shape or f.shape[-1] != 37:
        raise ValueError('CUDA strip statistics require matching D2Q37 populations')
    f, g = cp.ascontiguousarray(f, dtype=cp.float64), cp.ascontiguousarray(g, dtype=cp.float64)
    cells = f.size//37
    out = cp.empty((5, cells), dtype=cp.float64)
    _kernels(cp.cuda.Device().id)[1](((cells+127)//128,), (128,),
        (f, g, cp.asarray(kinetic, dtype=cp.float64), np.int64(cells), out))
    values = cp.asnumpy(cp.stack((out[0].sum(), out[1].sum(),
                                out[2].min(), out[3].min(), out[4].min())))
    return float(values[0]), float(values[1]), float(values[2]), float(values[3]), bool(values[4])
