"""Compiled collision arithmetic; mapping, equilibrium and checks stay public.

No transport law or SI/LU conversion is implemented here. The NumPy path supplies
local relaxation coefficients and the same fixed moment projectors. All kernels
use float64, serial cell loops and strict floating-point arithmetic (no fastmath).
"""
import math

import numpy as np
from numba import njit


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
def _collide_cells(f, g, fe, ge, rho, u, energy, tau21, tau22, retention,
                   c, w, second, heat, first, trace_scale, f_fraction):
    cells, q = f.shape
    result_f, result_g = np.empty_like(f), np.empty_like(g)
    denom_x = denom_y = 0.
    for a in range(q):
        denom_x += w[a]*c[a, 0]*c[a, 0]
        denom_y += w[a]*c[a, 1]*c[a, 1]
    for cell in range(cells):
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
        target_x = retention[cell]*(.5*qfx+qgx)
        target_y = retention[cell]*(.5*qfy+qgy)
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


def collide_prepared(f, g, fe, ge, macro, transport, mapping, lattice, projectors):
    """Adapt public arrays to the compiled arithmetic, without changing inputs."""
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != 37:
        raise ValueError('compiled D2Q37 collision requires equal f/g shapes ending in 37')
    flat = lambda values: np.ascontiguousarray(values).reshape(-1)
    populations = lambda values: np.ascontiguousarray(values).reshape(-1, 37)
    fp, gp = _collide_cells(populations(f), populations(g), populations(fe), populations(ge),
        flat(macro.rho), np.ascontiguousarray(macro.u).reshape(-1, 2), flat(macro.E_tot),
        flat(transport.tau21), flat(transport.tau22), flat(transport.heat_flux_retention),
        lattice.c, lattice.w, *projectors, mapping.collision.trace_bulk_scale,
        mapping.collision.regularized_heat_flux_f_fraction)
    return fp.reshape(f.shape), gp.reshape(g.shape)
