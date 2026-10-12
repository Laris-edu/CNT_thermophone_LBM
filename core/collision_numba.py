"""Compiled collision and bounded-memory preparation arithmetic.

No transport law or SI/LU conversion is implemented here. The public mapping supplies
local relaxation coefficients and the same fixed moment projectors. All kernels
use float64 and strict floating-point arithmetic (no fastmath). Large fields
parallelize independent cells; small fields retain the serial kernels.
"""
import math
import os

import numpy as np
from numba import get_num_threads, njit, prange

from core.equilibrium import EquilibriumDomainError
from core.macroscopic import MacroState


# Thread overhead did not pay off in short <=65k-cell checks; the r2 field did.
# This is a conservative dispatch threshold, not a change to collision physics.
def _parallel_threshold():
    value = os.environ.get('CNT_LBM_PARALLEL_MIN_CELLS', '262144')
    try:
        threshold = int(value)
    except ValueError as exc:
        raise ValueError('CNT_LBM_PARALLEL_MIN_CELLS must be a positive integer') from exc
    if threshold < 1:
        raise ValueError('CNT_LBM_PARALLEL_MIN_CELLS must be a positive integer')
    return threshold


# Explicit process-level override; it changes dispatch, never arithmetic.
PARALLEL_MIN_CELLS = _parallel_threshold()


@njit(cache=True, fastmath=False, nogil=True, error_model='numpy')
def _recover_cells(f, g, c, degrees):
    cells, q = f.shape
    rho = np.empty(cells)
    u = np.empty((cells, 2))
    theta, pressure, specific = np.empty(cells), np.empty(cells), np.empty(cells)
    mach, kinetic, internal, energy = (np.empty(cells), np.empty(cells),
                                      np.empty(cells), np.empty(cells))
    gamma = 1 + 2/degrees
    for cell in prange(cells):
        mass = px = py = 0.
        for a in range(q):
            mass += f[cell, a]
            px += f[cell, a]*c[a, 0]
            py += f[cell, a]*c[a, 1]
        ux, uy = px/mass, py/mass
        k = extra = 0.
        for a in range(q):
            cx, cy = c[a, 0]-ux, c[a, 1]-uy
            k += f[cell, a]*(cx*cx+cy*cy)
            extra += g[cell, a]
        k *= .5
        total = k+extra
        temperature = 2*total/(degrees*mass)
        speed = math.sqrt(ux*ux+uy*uy)
        sound_speed = math.sqrt(gamma*temperature)
        rho[cell] = mass
        u[cell, 0], u[cell, 1] = ux, uy
        theta[cell], pressure[cell] = temperature, mass*temperature
        specific[cell] = total/mass
        mach[cell] = speed/sound_speed if sound_speed != 0. else 0.
        kinetic[cell], internal[cell] = k, extra
        energy[cell] = .5*mass*speed*speed+total
    return rho, u, theta, pressure, specific, mach, kinetic, internal, energy


@njit(cache=True, fastmath=False, inline='always')
def _equilibrium_direction(rho, extra, ux, uy, u2, delta, cx, cy, weight, tq):
    dot = (ux*cx+uy*cy)/tq
    c2 = (cx*cx+cy*cy)/tq
    dot2 = dot*dot
    h2 = 1+dot+.5*(dot2-u2+delta*(c2-2))
    h4 = h2+dot*(dot2-3*u2+3*delta*(c2-4))/6
    h4 += (dot2*dot2-6*dot2*u2+3*u2*u2
        +6*delta*(dot2*(c2-6)-u2*(c2-4))
        +3*delta*delta*(c2*c2-8*c2+8))/24
    return rho*weight*h4, extra*weight*h2


@njit(cache=True, fastmath=False, nogil=True)
def _equilibrium_cells(rho, u, theta, S, c, w, tq):
    cells, q = rho.size, w.size
    fe, ge = np.empty((cells, q)), np.empty((cells, q))
    invalid = np.zeros(cells, dtype=np.uint8)
    for cell in prange(cells):
        ux, uy = u[cell, 0], u[cell, 1]
        u2 = (ux*ux+uy*uy)/tq
        delta = theta[cell]/tq-1
        extra = .5*S*rho[cell]*theta[cell]
        for a in range(q):
            fa, ga = _equilibrium_direction(rho[cell], extra, ux, uy, u2,
                delta, c[a, 0], c[a, 1], w[a], tq)
            fe[cell, a], ge[cell, a] = fa, ga
            if not math.isfinite(fa) or not math.isfinite(ga):
                invalid[cell] |= 1
            if fa < 0 or ga < 0:
                invalid[cell] |= 2
    return fe, ge, invalid


@njit(cache=True, fastmath=False, nogil=True)
def _equilibrium_increment_cells(f, g, old_rho, old_u, old_theta,
        new_rho, new_u, new_theta, mask, c, w, tq, S):
    cells, q = f.shape
    fn, gn = np.empty_like(f), np.empty_like(g)
    mass_change, energy_change = np.zeros(cells), np.zeros(cells)
    invalid = np.zeros(cells, dtype=np.uint8)
    for cell in prange(cells):
        if not mask[cell]:
            for a in range(q):
                fn[cell, a], gn[cell, a] = f[cell, a], g[cell, a]
            continue
        ro, rn = old_rho[cell], new_rho[cell]
        to, tn = old_theta[cell], new_theta[cell]
        uxo, uyo = old_u[cell, 0], old_u[cell, 1]
        uxn, uyn = new_u[cell, 0], new_u[cell, 1]
        if not (math.isfinite(ro) and math.isfinite(rn)
                and math.isfinite(to) and math.isfinite(tn)
                and math.isfinite(uxo) and math.isfinite(uyo)
                and math.isfinite(uxn) and math.isfinite(uyn)):
            invalid[cell] = 1
            continue
        if ro < 0 or rn < 0 or to <= 0 or tn <= 0:
            invalid[cell] = 2
            continue
        u2o, u2n = (uxo*uxo+uyo*uyo)/tq, (uxn*uxn+uyn*uyn)/tq
        deltao, deltan = to/tq-1, tn/tq-1
        extrao, extran = .5*S*ro*to, .5*S*rn*tn
        mass = energy = 0.
        for a in range(q):
            fo, go = _equilibrium_direction(ro, extrao, uxo, uyo, u2o,
                deltao, c[a, 0], c[a, 1], w[a], tq)
            fa, ga = _equilibrium_direction(rn, extran, uxn, uyn, u2n,
                deltan, c[a, 0], c[a, 1], w[a], tq)
            if not (math.isfinite(fo) and math.isfinite(go)
                    and math.isfinite(fa) and math.isfinite(ga)):
                invalid[cell] |= 4
            if fo < 0 or go < 0 or fa < 0 or ga < 0:
                invalid[cell] |= 8
            fn[cell, a] = f[cell, a]+(fa-fo)
            gn[cell, a] = g[cell, a]+(ga-go)
            # Independent accounting uses the rounded writes, not target macros.
            df, dg = fn[cell, a]-f[cell, a], gn[cell, a]-g[cell, a]
            mass += df
            energy += .5*(c[a, 0]*c[a, 0]+c[a, 1]*c[a, 1])*df+dg
        mass_change[cell], energy_change[cell] = mass, energy
    return fn, gn, mass_change, energy_change, invalid


_recover_cells_parallel = njit(fastmath=False, nogil=True, parallel=True,
    error_model='numpy')(_recover_cells.py_func)
_equilibrium_cells_parallel = njit(fastmath=False, nogil=True, parallel=True)(_equilibrium_cells.py_func)
_equilibrium_increment_cells_parallel = njit(fastmath=False, nogil=True,
    parallel=True)(_equilibrium_increment_cells.py_func)


def recover_macro_compiled(f, g, *, D, S, lattice):
    """Same central energy and fields as recover_macro, with scalar cell work."""
    f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != 37 or D != 2:
        raise ValueError('compiled macro recovery requires equal D2Q37 f/g arrays')
    shape = f.shape[:-1]
    recover = _recover_cells_parallel if _parallel_field(f.size//37) else _recover_cells
    values = recover(np.ascontiguousarray(f).reshape(-1, 37),
        np.ascontiguousarray(g).reshape(-1, 37), lattice.c, D+S)
    names = ('rho', 'u', 'theta', 'p', 'e', 'mach', 'K_tr', 'G_int', 'E_tot')
    fields = {name: value.reshape((*shape, 2) if name == 'u' else shape)
              for name, value in zip(names, values)}
    return MacroState(**fields, gamma=1+2/(D+S))


def equilibrium_compiled(rho, u, theta, S, lattice):
    """Public H4/H2 equilibrium algebra, without population-sized temporaries."""
    rho, theta = np.broadcast_arrays(np.asarray(rho, dtype=float), np.asarray(theta, dtype=float))
    u = np.broadcast_to(np.asarray(u, dtype=float), (*rho.shape, 2))
    if lattice.d != 2 or lattice.q != 37:
        raise ValueError('unsupported equilibrium quadrature')
    if not np.isfinite(S) or S < 0:
        raise ValueError('internal degrees S must be finite and nonnegative')
    if not all(np.isfinite(value).all() for value in (rho, u, theta)):
        raise EquilibriumDomainError('equilibrium inputs must be finite')
    if np.any(rho < 0) or np.any(theta <= 0):
        raise EquilibriumDomainError('equilibrium density must be nonnegative and temperature positive')
    if not np.isfinite(lattice.theta_q) or lattice.theta_q <= 0:
        raise ValueError('quadrature temperature must be positive and finite')
    evaluate = _equilibrium_cells_parallel if _parallel_field(rho.size) else _equilibrium_cells
    fe, ge, invalid = evaluate(np.ascontiguousarray(rho).reshape(-1),
        np.ascontiguousarray(u).reshape(-1, 2), np.ascontiguousarray(theta).reshape(-1),
        S, lattice.c, lattice.w, lattice.theta_q)
    if np.any(invalid & 1):
        raise EquilibriumDomainError('equilibrium expansion produced non-finite populations')
    if np.any(invalid & 2):
        raise EquilibriumDomainError('equilibrium H4/H2 is not nonnegative; populations are not clipped')
    return fe.reshape((*rho.shape, 37)), ge.reshape((*rho.shape, 37))


def apply_equilibrium_increment(f, g, old_rho, old_u, old_theta,
        new_rho, new_u, new_theta, mask, lattice, S):
    """Return f/g + (new eq - old eq) and actual mass/energy changes in LU.

    Macro fields cover the full population grid. Only mask cells evaluate or
    validate equilibrium; all other populations are copied bit for bit. Inputs
    remain unchanged, including when an active equilibrium is invalid.
    """
    f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != 37:
        raise ValueError('compiled equilibrium increment requires equal D2Q37 f/g arrays')
    if lattice.d != 2 or lattice.q != 37:
        raise ValueError('unsupported equilibrium quadrature')
    if not np.isfinite(S) or S < 0:
        raise ValueError('internal degrees S must be finite and nonnegative')
    if not np.isfinite(lattice.theta_q) or lattice.theta_q <= 0:
        raise ValueError('quadrature temperature must be positive and finite')
    shape = f.shape[:-1]
    def scalar(value):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=float), shape)).reshape(-1)
    def vector(value):
        return np.ascontiguousarray(np.broadcast_to(np.asarray(value, dtype=float), (*shape, 2))).reshape(-1, 2)
    active = np.ascontiguousarray(np.broadcast_to(np.asarray(mask, dtype=bool), shape)).reshape(-1)
    increment = (_equilibrium_increment_cells_parallel if _parallel_field(active.size)
                 else _equilibrium_increment_cells)
    fn, gn, mass, energy, invalid = increment(np.ascontiguousarray(f).reshape(-1, 37),
        np.ascontiguousarray(g).reshape(-1, 37), scalar(old_rho), vector(old_u),
        scalar(old_theta), scalar(new_rho), vector(new_u), scalar(new_theta), active,
        lattice.c, lattice.w, lattice.theta_q, S)
    if np.any(invalid & 1):
        raise EquilibriumDomainError('equilibrium inputs must be finite')
    if np.any(invalid & 2):
        raise EquilibriumDomainError('equilibrium density must be nonnegative and temperature positive')
    if np.any(invalid & 4):
        raise EquilibriumDomainError('equilibrium expansion produced non-finite populations')
    if np.any(invalid & 8):
        raise EquilibriumDomainError('equilibrium H4/H2 is not nonnegative; populations are not clipped')
    change = np.array([np.sum(mass), np.sum(energy)])
    return fn.reshape(f.shape), gn.reshape(g.shape), change


@njit(cache=True, fastmath=False, inline='always')
def _correct_mass_momentum(fp, c, w, rho, px, py, denom_x, denom_y):
    mass = 0.
    for a in range(fp.size):
        mass += fp[a]
    for a in range(fp.size):
        fp[a] += (rho-mass)*w[a]
    current_x = current_y = 0.
    for a in range(fp.size):
        current_x += fp[a]*c[a, 0]
        current_y += fp[a]*c[a, 1]
    dx, dy = px-current_x, py-current_y
    for a in range(fp.size):
        fp[a] += dx*w[a]*c[a, 0]/denom_x
        fp[a] += dy*w[a]*c[a, 1]/denom_y


@njit(cache=True, fastmath=False, nogil=True, error_model='numpy')
def _collide_cells(stress, q_raw, momentum, fe, ge, rho, u, energy, tau21, tau22, retention,
                   c, w, second, heat, first, trace_scale, f_fraction):
    cells, q = fe.shape
    result_f, result_g = np.empty_like(fe), np.empty_like(ge)
    denom_x = denom_y = 0.
    for a in range(q):
        denom_x += w[a]*c[a, 0]*c[a, 0]
        denom_y += w[a]*c[a, 1]*c[a, 1]
    for cell in prange(cells):
        ux, uy = u[cell, 0], u[cell, 1]
        xx, xy, yy = stress[cell, 0], stress[cell, 1], stress[cell, 2]
        px, py = momentum[cell, 0], momentum[cell, 1]
        trace = trace_scale*(1.-1./tau22[cell])*(xx+yy)
        dev = (1.-1./tau21[cell])*(xx-yy)
        shear = (1.-1./tau21[cell])*xy
        dxx, dyy = .5*(trace+dev), .5*(trace-dev)
        fp, gp = result_f[cell], result_g[cell]
        for a in range(q):
            fp[a] = fe[cell, a] + dyy*second[0, a] + shear*second[1, a] + dxx*second[2, a]
        _correct_mass_momentum(fp, c, w, rho[cell], px, py, denom_x, denom_y)
        # Subtract the already present central flux before adding the projection.
        present_fx = present_fy = present_gx = present_gy = 0.
        for a in range(q):
            x, y = c[a, 0]-ux, c[a, 1]-uy
            speed2 = x*x+y*y
            present_fx += fp[a]*speed2*x
            present_fy += fp[a]*speed2*y
            present_gx += ge[cell, a]*x
            present_gy += ge[cell, a]*y
        target_x = retention[cell]*q_raw[cell, 0]
        target_y = retention[cell]*q_raw[cell, 1]
        dfx = f_fraction*target_x-.5*present_fx
        dfy = f_fraction*target_y-.5*present_fy
        dgx = (1.-f_fraction)*target_x-present_gx
        dgy = (1.-f_fraction)*target_y-present_gy
        for a in range(q):
            fp[a] += dfy*heat[0, a] + dfx*heat[1, a]
            gp[a] = ge[cell, a] + dgy*first[0, a] + dgx*first[1, a]
        _correct_mass_momentum(fp, c, w, rho[cell], px, py, denom_x, denom_y)
        # Same central total-energy definition as recover_macro, at the corrected
        # post-collision velocity; no raw-moment or heat-flux substitution.
        mass = momentum_x = momentum_y = 0.
        for a in range(q):
            mass += fp[a]
            momentum_x += fp[a]*c[a, 0]
            momentum_y += fp[a]*c[a, 1]
        vx, vy = momentum_x/mass, momentum_y/mass
        kinetic = internal = 0.
        for a in range(q):
            x, y = c[a, 0]-vx, c[a, 1]-vy
            kinetic += fp[a]*(x*x+y*y)
            internal += gp[a]
        speed = math.sqrt(vx*vx+vy*vy)
        correction = energy[cell]-(.5*mass*speed*speed + (.5*kinetic+internal))
        for a in range(q):
            gp[a] += correction*(fe[cell, a]/rho[cell])
    return result_f, result_g


@njit(cache=True, fastmath=False, nogil=True)
def _measure_cells(f, g, fe, u, c):
    cells, q = f.shape
    stress = np.empty((cells, 3))
    q_raw = np.empty((cells, 2))
    momentum = np.empty((cells, 2))
    for cell in prange(cells):
        ux, uy = u[cell, 0], u[cell, 1]
        xx = xy = yy = px = py = 0.
        qfx = qfy = qgx = qgy = 0.
        for a in range(q):
            x, y = c[a, 0]-ux, c[a, 1]-uy
            speed2 = x*x+y*y
            neq = f[cell, a]-fe[cell, a]
            xx += neq*x*x
            xy += neq*x*y
            yy += neq*y*y
            px += f[cell, a]*c[a, 0]
            py += f[cell, a]*c[a, 1]
            qfx += f[cell, a]*speed2*x
            qfy += f[cell, a]*speed2*y
            qgx += g[cell, a]*x
            qgy += g[cell, a]*y
        stress[cell, 0], stress[cell, 1], stress[cell, 2] = xx, xy, yy
        q_raw[cell, 0], q_raw[cell, 1] = .5*qfx+qgx, .5*qfy+qgy
        momentum[cell, 0], momentum[cell, 1] = px, py
    return stress, q_raw, momentum


# Compile the same arithmetic twice, so serial and parallel paths cannot drift.
# The parallel dispatchers deliberately do not share the serial disk cache.
_measure_cells_parallel = njit(fastmath=False, nogil=True, parallel=True)(_measure_cells.py_func)
_collide_cells_parallel = njit(fastmath=False, nogil=True, parallel=True,
    error_model='numpy')(_collide_cells.py_func)


def _parallel_field(cell_count):
    return cell_count >= PARALLEL_MIN_CELLS and get_num_threads() > 1


def collide_prepared(f, g, fe, ge, macro, transport, mapping, lattice, projectors):
    """Adapt public arrays to the compiled arithmetic, without changing inputs."""
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != 37:
        raise ValueError('compiled D2Q37 collision requires equal f/g shapes ending in 37')
    populations = lambda values: np.ascontiguousarray(values).reshape(-1, 37)
    measure = _measure_cells_parallel if _parallel_field(f.size//37) else _measure_cells
    measured = measure(populations(f), populations(g), populations(fe),
        np.ascontiguousarray(macro.u).reshape(-1, 2), lattice.c)
    return collide_measured(*measured, fe, ge, macro, transport, mapping, lattice, projectors)


def collide_measured(stress, q_raw, momentum, fe, ge, macro, transport, mapping, lattice, projectors):
    """Common compiled reconstruction, accepting measured stress xx,xy,yy."""
    flat = lambda values: np.ascontiguousarray(values).reshape(-1)
    populations = lambda values: np.ascontiguousarray(values).reshape(-1, 37)
    collide = _collide_cells_parallel if _parallel_field(fe.size//37) else _collide_cells
    fp, gp = collide(
        np.ascontiguousarray(stress).reshape(-1, 3),
        np.ascontiguousarray(q_raw).reshape(-1, 2),
        np.ascontiguousarray(momentum).reshape(-1, 2), populations(fe), populations(ge),
        flat(macro.rho), np.ascontiguousarray(macro.u).reshape(-1, 2), flat(macro.E_tot),
        flat(transport.tau21), flat(transport.tau22), flat(transport.heat_flux_retention),
        lattice.c, lattice.w, *projectors, mapping.collision.trace_bulk_scale,
        mapping.collision.regularized_heat_flux_f_fraction)
    return fp.reshape(fe.shape), gp.reshape(ge.shape)
