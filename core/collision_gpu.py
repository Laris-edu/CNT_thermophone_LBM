"""Optional bounded-memory float64 CUDA collision and macro preparation.

The per-cell arithmetic mirrors collision_numba; transport and fixed projectors
remain owned by unit_mapping and collision_smrt. CUDA is imported only on use.
No fast math, positivity clipping, host fallback, or full-grid equilibrium
scratch is used by the fused collision kernel.
"""
from functools import lru_cache

import numpy as np

from core.equilibrium import EquilibriumDomainError
from core.macroscopic import MacroState


_CUDA_SOURCE = r'''
__device__ int state_status(double rho, double ux, double uy, double theta) {
    if (!(isfinite(rho) && isfinite(ux) && isfinite(uy) && isfinite(theta))) return 1;
    if (rho < 0. || theta <= 0.) return 2;
    return 0;
}

__device__ void equilibrium_direction(double rho, double extra, double ux, double uy,
    double u2, double delta, double cx, double cy, double weight, double tq,
    double &fa, double &ga) {
    double dot = (ux*cx+uy*cy)/tq;
    double c2 = (cx*cx+cy*cy)/tq;
    double dot2 = dot*dot;
    double h2 = 1.+dot+.5*(dot2-u2+delta*(c2-2.));
    double h4 = h2+dot*(dot2-3.*u2+3.*delta*(c2-4.))/6.;
    h4 += (dot2*dot2-6.*dot2*u2+3.*u2*u2
        +6.*delta*(dot2*(c2-6.)-u2*(c2-4.))
        +3.*delta*delta*(c2*c2-8.*c2+8.))/24.;
    fa = rho*weight*h4;
    ga = extra*weight*h2;
}

__device__ int population_status(double fa, double ga) {
    if (!(isfinite(fa) && isfinite(ga))) return 3;
    if (fa < 0. || ga < 0.) return 4;
    return 0;
}

extern "C" __global__ void recover_cells(const double *f, const double *g,
    const double *c, long long cells, double degrees, double *rho, double *u,
    double *theta, double *pressure, double *specific, double *mach,
    double *kinetic, double *internal, double *energy) {
    long long cell = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (cell >= cells) return;
    const double *fc = f+37*cell, *gc = g+37*cell;
    double mass = 0., px = 0., py = 0.;
    for (int a=0; a<37; ++a) {
        mass += fc[a]; px += fc[a]*c[2*a]; py += fc[a]*c[2*a+1];
    }
    double ux = px/mass, uy = py/mass, k = 0., extra = 0.;
    for (int a=0; a<37; ++a) {
        double x = c[2*a]-ux, y = c[2*a+1]-uy;
        k += fc[a]*(x*x+y*y); extra += gc[a];
    }
    k *= .5;
    double total = k+extra, temperature = 2.*total/(degrees*mass);
    double speed = sqrt(ux*ux+uy*uy), sound = sqrt((1.+2./degrees)*temperature);
    rho[cell] = mass; u[2*cell] = ux; u[2*cell+1] = uy;
    theta[cell] = temperature; pressure[cell] = mass*temperature;
    specific[cell] = total/mass; mach[cell] = sound != 0. ? speed/sound : 0.;
    kinetic[cell] = k; internal[cell] = extra;
    energy[cell] = .5*mass*speed*speed+total;
}

extern "C" __global__ void equilibrium_cells(const double *rho, const double *u,
    const double *theta, const double *c, const double *w, long long cells,
    double S, double tq, double *f, double *g, int *invalid) {
    long long cell = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (cell >= cells) return;
    double r = rho[cell], ux = u[2*cell], uy = u[2*cell+1], t = theta[cell];
    int status = state_status(r, ux, uy, t);
    if (status) { atomicMax(invalid, status); return; }
    double u2 = (ux*ux+uy*uy)/tq, delta = t/tq-1., extra = .5*S*r*t;
    for (int a=0; a<37; ++a) {
        double fa, ga;
        equilibrium_direction(r, extra, ux, uy, u2, delta, c[2*a], c[2*a+1], w[a], tq, fa, ga);
        status = population_status(fa, ga);
        if (status) atomicMax(invalid, status);
        f[37*cell+a] = fa; g[37*cell+a] = ga;
    }
}

extern "C" __global__ void increment_cells(const double *f, const double *g,
    const double *old_rho, const double *old_u, const double *old_theta,
    const double *new_rho, const double *new_u, const double *new_theta,
    const bool *mask, const double *c, const double *w, long long cells,
    double S, double tq, double *fn, double *gn, double *mass_change,
    double *energy_change, int *invalid) {
    long long cell = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (cell >= cells) return;
    mass_change[cell] = 0.; energy_change[cell] = 0.;
    if (!mask[cell]) {
        for (int a=0; a<37; ++a) {
            fn[37*cell+a] = f[37*cell+a]; gn[37*cell+a] = g[37*cell+a];
        }
        return;
    }
    double ro = old_rho[cell], rn = new_rho[cell], to = old_theta[cell], tn = new_theta[cell];
    double uxo = old_u[2*cell], uyo = old_u[2*cell+1];
    double uxn = new_u[2*cell], uyn = new_u[2*cell+1];
    int so = state_status(ro, uxo, uyo, to), sn = state_status(rn, uxn, uyn, tn);
    if (so || sn) { atomicMax(invalid, so > sn ? so : sn); return; }
    double u2o = (uxo*uxo+uyo*uyo)/tq, u2n = (uxn*uxn+uyn*uyn)/tq;
    double deltao = to/tq-1., deltan = tn/tq-1.;
    double extrao = .5*S*ro*to, extran = .5*S*rn*tn, mass = 0., energy = 0.;
    for (int a=0; a<37; ++a) {
        double fo, go, fa, ga;
        equilibrium_direction(ro, extrao, uxo, uyo, u2o, deltao, c[2*a], c[2*a+1], w[a], tq, fo, go);
        equilibrium_direction(rn, extran, uxn, uyn, u2n, deltan, c[2*a], c[2*a+1], w[a], tq, fa, ga);
        so = population_status(fo, go); sn = population_status(fa, ga);
        if (so || sn) atomicMax(invalid, so > sn ? so : sn);
        fn[37*cell+a] = f[37*cell+a]+(fa-fo);
        gn[37*cell+a] = g[37*cell+a]+(ga-go);
        double df = fn[37*cell+a]-f[37*cell+a], dg = gn[37*cell+a]-g[37*cell+a];
        mass += df;
        energy += .5*(c[2*a]*c[2*a]+c[2*a+1]*c[2*a+1])*df+dg;
    }
    mass_change[cell] = mass; energy_change[cell] = energy;
}

__device__ void correct_mass_momentum(double *fp, const double *c, const double *w,
    double rho, double px, double py, double denom_x, double denom_y) {
    double mass = 0.;
    for (int a=0; a<37; ++a) mass += fp[a];
    for (int a=0; a<37; ++a) fp[a] += (rho-mass)*w[a];
    double current_x = 0., current_y = 0.;
    for (int a=0; a<37; ++a) {
        current_x += fp[a]*c[2*a]; current_y += fp[a]*c[2*a+1];
    }
    double dx = px-current_x, dy = py-current_y;
    for (int a=0; a<37; ++a) {
        fp[a] += dx*w[a]*c[2*a]/denom_x;
        fp[a] += dy*w[a]*c[2*a+1]/denom_y;
    }
}

extern "C" __global__ void collide_cells(const double *f, const double *g,
    const double *rho, const double *u, const double *theta, const double *energy,
    const double *tau21, const double *tau22, const double *retention,
    const double *c, const double *w, const double *second, const double *heat,
    const double *first, long long cells, double S, double tq, double trace_scale,
    double f_fraction, double denom_x, double denom_y, double *result_f,
    double *result_g, int *invalid) {
    long long cell = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (cell >= cells) return;
    double ux = u[2*cell], uy = u[2*cell+1], r = rho[cell], t = theta[cell];
    int status = state_status(r, ux, uy, t);
    if (status) { atomicMax(invalid, status); return; }
    double fe[37], ge[37], fp[37], gp[37];
    double u2 = (ux*ux+uy*uy)/tq, delta = t/tq-1., extra = .5*S*r*t;
    double xx = 0., xy = 0., yy = 0., px = 0., py = 0.;
    double qfx = 0., qfy = 0., qgx = 0., qgy = 0.;
    for (int a=0; a<37; ++a) {
        equilibrium_direction(r, extra, ux, uy, u2, delta, c[2*a], c[2*a+1], w[a], tq, fe[a], ge[a]);
        status = population_status(fe[a], ge[a]);
        if (status) { atomicMax(invalid, status); return; }
        double fa = f[37*cell+a], ga = g[37*cell+a];
        double x = c[2*a]-ux, y = c[2*a+1]-uy, speed2 = x*x+y*y;
        double neq = fa-fe[a];
        xx += neq*x*x; xy += neq*x*y; yy += neq*y*y;
        px += fa*c[2*a]; py += fa*c[2*a+1];
        qfx += fa*speed2*x; qfy += fa*speed2*y; qgx += ga*x; qgy += ga*y;
    }
    double trace = trace_scale*(1.-1./tau22[cell])*(xx+yy);
    double dev = (1.-1./tau21[cell])*(xx-yy), shear = (1.-1./tau21[cell])*xy;
    double dxx = .5*(trace+dev), dyy = .5*(trace-dev);
    for (int a=0; a<37; ++a) fp[a] = fe[a]+dyy*second[a]+shear*second[37+a]+dxx*second[74+a];
    correct_mass_momentum(fp, c, w, r, px, py, denom_x, denom_y);
    double present_fx = 0., present_fy = 0., present_gx = 0., present_gy = 0.;
    for (int a=0; a<37; ++a) {
        double x = c[2*a]-ux, y = c[2*a+1]-uy, speed2 = x*x+y*y;
        present_fx += fp[a]*speed2*x; present_fy += fp[a]*speed2*y;
        present_gx += ge[a]*x; present_gy += ge[a]*y;
    }
    double target_x = retention[cell]*(.5*qfx+qgx), target_y = retention[cell]*(.5*qfy+qgy);
    double dfx = f_fraction*target_x-.5*present_fx, dfy = f_fraction*target_y-.5*present_fy;
    double dgx = (1.-f_fraction)*target_x-present_gx, dgy = (1.-f_fraction)*target_y-present_gy;
    for (int a=0; a<37; ++a) {
        fp[a] += dfy*heat[a]+dfx*heat[37+a];
        gp[a] = ge[a]+dgy*first[a]+dgx*first[37+a];
    }
    correct_mass_momentum(fp, c, w, r, px, py, denom_x, denom_y);
    double mass = 0., momentum_x = 0., momentum_y = 0.;
    for (int a=0; a<37; ++a) {
        mass += fp[a]; momentum_x += fp[a]*c[2*a]; momentum_y += fp[a]*c[2*a+1];
    }
    double vx = momentum_x/mass, vy = momentum_y/mass, kinetic = 0., internal = 0.;
    for (int a=0; a<37; ++a) {
        double x = c[2*a]-vx, y = c[2*a+1]-vy;
        kinetic += fp[a]*(x*x+y*y); internal += gp[a];
    }
    double speed = sqrt(vx*vx+vy*vy);
    double correction = energy[cell]-(.5*mass*speed*speed+(.5*kinetic+internal));
    for (int a=0; a<37; ++a) {
        gp[a] += correction*(fe[a]/r);
        result_f[37*cell+a] = fp[a]; result_g[37*cell+a] = gp[a];
    }
}
'''


@lru_cache(maxsize=8)
def _module(device):
    import cupy as cp
    return cp.RawModule(code=_CUDA_SOURCE, options=('--std=c++11', '--fmad=false'))


@lru_cache(maxsize=24)
def _constants(device, c_values, weights):
    import cupy as cp
    from core.collision_smrt import _fixed_zero_lower_projector
    c = cp.asarray(c_values, dtype=cp.float64)
    w = cp.asarray(weights, dtype=cp.float64)
    projectors = tuple(cp.asarray(_fixed_zero_lower_projector(c_values, weights, kind))
                       for kind in ('second', 'heat', 'first'))
    return c, w, projectors


def _lattice_data(lattice):
    import cupy as cp
    if lattice.d != 2 or lattice.q != 37:
        raise ValueError('unsupported equilibrium quadrature')
    if not np.isfinite(lattice.theta_q) or lattice.theta_q <= 0:
        raise ValueError('quadrature temperature must be positive and finite')
    return _constants(cp.cuda.runtime.getDevice(), tuple(map(tuple, lattice.c)), tuple(lattice.w))


def _validate_degrees(S):
    if not np.isfinite(S) or S < 0:
        raise ValueError('internal degrees S must be finite and nonnegative')


def _launch(name, cells, args):
    import cupy as cp
    if cells:
        kernel = _module(cp.cuda.runtime.getDevice()).get_function(name)
        kernel(((cells+127)//128,), (128,), args)


def _check_equilibrium(invalid):
    status = int(invalid[0])  # Required error boundary, never a population transfer.
    if status:
        messages = {1: 'equilibrium inputs must be finite',
                    2: 'equilibrium density must be nonnegative and temperature positive',
                    3: 'equilibrium expansion produced non-finite populations',
                    4: 'equilibrium H4/H2 is not nonnegative; populations are not clipped'}
        raise EquilibriumDomainError(messages[status])


def _populations(f, g):
    import cupy as cp
    f, g = cp.asarray(f, dtype=cp.float64), cp.asarray(g, dtype=cp.float64)
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != 37:
        raise ValueError('GPU D2Q37 requires equal f/g arrays ending in 37')
    return cp.ascontiguousarray(f), cp.ascontiguousarray(g)


def _scalar(value, shape):
    import cupy as cp
    return cp.ascontiguousarray(cp.broadcast_to(cp.asarray(value, dtype=cp.float64), shape))


def _vector(value, shape):
    import cupy as cp
    value = cp.asarray(value, dtype=cp.float64)
    if value.ndim == 0 or value.shape[-1] != 2:
        raise ValueError('equilibrium velocity must have a final axis of length 2')
    return cp.ascontiguousarray(cp.broadcast_to(value, (*shape, 2)))


def recover_macro_gpu(f, g, *, D, S, lattice):
    """Recover public MacroState fields without leaving device memory."""
    import cupy as cp
    if D != 2:
        raise ValueError('GPU macro recovery requires D2Q37')
    _validate_degrees(S)
    f, g = _populations(f, g)
    c, _, _ = _lattice_data(lattice)
    shape, cells = f.shape[:-1], f.size//37
    names = ('rho', 'u', 'theta', 'p', 'e', 'mach', 'K_tr', 'G_int', 'E_tot')
    values = [cp.empty((*shape, 2) if name == 'u' else shape, dtype=cp.float64) for name in names]
    _launch('recover_cells', cells, (f, g, c, np.int64(cells), np.float64(D+S), *values))
    return MacroState(**dict(zip(names, values)), gamma=1+2/(D+S))


def equilibrium_gpu(rho, u, theta, S, lattice):
    """H4/H2 equilibrium with the same positivity/domain rejection as CPU."""
    import cupy as cp
    _validate_degrees(S)
    c, w, _ = _lattice_data(lattice)
    rho, theta = cp.broadcast_arrays(cp.asarray(rho, dtype=cp.float64), cp.asarray(theta, dtype=cp.float64))
    shape = rho.shape
    rho, theta = cp.ascontiguousarray(rho), cp.ascontiguousarray(theta)
    u = _vector(u, shape)
    fe, ge = (cp.empty((*shape, 37), dtype=cp.float64) for _ in range(2))
    invalid = cp.zeros(1, dtype=cp.int32)
    _launch('equilibrium_cells', rho.size, (rho, u, theta, c, w, np.int64(rho.size),
        np.float64(S), np.float64(lattice.theta_q), fe, ge, invalid))
    _check_equilibrium(invalid)
    return fe, ge


def apply_equilibrium_increment(f, g, old_rho, old_u, old_theta,
        new_rho, new_u, new_theta, mask, lattice, S):
    """Copy inactive cells; add equilibrium increments and audit actual writes."""
    import cupy as cp
    _validate_degrees(S)
    c, w, _ = _lattice_data(lattice)
    f, g = _populations(f, g)
    shape, cells = f.shape[:-1], f.size//37
    macro = (_scalar(old_rho, shape), _vector(old_u, shape), _scalar(old_theta, shape),
             _scalar(new_rho, shape), _vector(new_u, shape), _scalar(new_theta, shape))
    active = cp.ascontiguousarray(cp.broadcast_to(cp.asarray(mask, dtype=cp.bool_), shape))
    fn, gn = cp.empty_like(f), cp.empty_like(g)
    mass, energy = cp.empty(shape, dtype=cp.float64), cp.empty(shape, dtype=cp.float64)
    invalid = cp.zeros(1, dtype=cp.int32)
    _launch('increment_cells', cells, (f, g, *macro, active, c, w, np.int64(cells),
        np.float64(S), np.float64(lattice.theta_q), fn, gn, mass, energy, invalid))
    _check_equilibrium(invalid)
    return fn, gn, cp.stack((cp.sum(mass), cp.sum(energy)))


def collide_gpu(f, g, mapping, *, lattice, return_diagnostics=False):
    """Fused conservative collision; coefficients come from unit_mapping only."""
    import cupy as cp
    from core.collision_smrt import CollisionDiagnostics
    from core.unit_mapping import local_transport_coefficients
    f, g = _populations(f, g)
    c, w, projectors = _lattice_data(lattice)
    macro = recover_macro_gpu(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    transport = local_transport_coefficients(mapping, macro.rho, macro.theta)
    fp, gp = cp.empty_like(f), cp.empty_like(g)
    invalid = cp.zeros(1, dtype=cp.int32)
    denom = np.sum(lattice.w[:, None]*lattice.c*lattice.c, axis=0)
    cells = f.size//37
    _launch('collide_cells', cells, (f, g, macro.rho, macro.u, macro.theta, macro.E_tot,
        transport.tau21, transport.tau22, transport.heat_flux_retention,
        c, w, *projectors, np.int64(cells), np.float64(mapping.lattice.S),
        np.float64(lattice.theta_q), np.float64(mapping.collision.trace_bulk_scale),
        np.float64(mapping.collision.regularized_heat_flux_f_fraction),
        np.float64(denom[0]), np.float64(denom[1]), fp, gp, invalid))
    _check_equilibrium(invalid)
    if not return_diagnostics:
        return fp, gp
    after = recover_macro_gpu(fp, gp, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    difference = fp-f
    diagnostics = CollisionDiagnostics(mass_residual=cp.sum(difference, axis=-1),
        momentum_residual=cp.einsum('...a,ai->...i', difference, c),
        energy_residual=after.E_tot-macro.E_tot, min_f_post=float(cp.min(fp)),
        min_g_post=float(cp.min(gp)), clipping_used=False)
    return fp, gp, diagnostics
