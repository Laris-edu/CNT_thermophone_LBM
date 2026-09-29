"""Nine sufficient raw moments of the existing regularized D2Q37 collision.

Order: rho, jx, jy, E, Mxx, Mxy, Myy, JEx, JEy. E and JE contain both
translational and internal energy. This is a factorization of the population
collision, not a new closure or a population lifting. The 65-dimensional
kernel of extraction is erased by collision, not by free streaming.
"""
from __future__ import annotations

import numpy as np

from core.lattice import make_lattice
from core.macroscopic import MacroState
from core.equilibrium import equilibrium_fg
from core.unit_mapping import local_transport_coefficients

MOMENT_NAMES = ('rho', 'jx', 'jy', 'E', 'Mxx', 'Mxy', 'Myy', 'JEx', 'JEy')


def extract_moments(f, g, lattice=None):
    """Extract the nine raw moments from genuine populations (..., Q)."""
    lattice = lattice or make_lattice()
    f, g = np.asarray(f, dtype=float), np.asarray(g, dtype=float)
    if f.shape != g.shape or f.ndim < 1 or f.shape[-1] != lattice.q:
        raise ValueError('f and g must have equal shapes ending in lattice.q')
    x, y = lattice.c.T
    energy = .5*f*(x*x+y*y)+g
    return np.stack((np.sum(f,axis=-1), f@x, f@y, np.sum(energy,axis=-1),
                     f@(x*x), f@(x*y), f@(y*y), energy@x, energy@y),axis=-1)


def recover_moment_macro(z, mapping):
    """Recover MacroState without assigning the unresolved population modes."""
    z = np.asarray(z, dtype=float)
    if z.ndim < 1 or z.shape[-1] != 9:
        raise ValueError('moment state must end in nine components')
    rho = z[..., 0]
    u = z[..., 1:3]/rho[..., None]
    speed2 = np.sum(u*u,axis=-1)
    bulk = .5*rho*speed2
    internal = z[..., 3]-bulk
    theta = 2*internal/((mapping.lattice.D+mapping.lattice.S)*rho)
    p = rho*theta
    gamma = 1+2/(mapping.lattice.D+mapping.lattice.S)
    c_s = np.sqrt(gamma*theta)
    speed = np.sqrt(speed2)
    mach = np.divide(speed,c_s,out=np.zeros_like(speed),where=c_s!=0)
    K_tr = .5*(z[..., 4]+z[..., 6])-bulk
    G_int = z[..., 3]-.5*(z[..., 4]+z[..., 6])
    return MacroState(rho,u,theta,p,internal/rho,gamma,mach,K_tr,G_int,z[...,3])


def _measured_state(z, macro):
    """Measured nonequilibrium stress and raw central total-energy flux."""
    rho,u,p = macro.rho,macro.u,macro.p
    Pxx = z[...,4]-rho*u[...,0]**2
    Pxy = z[...,5]-rho*u[...,0]*u[...,1]
    Pyy = z[...,6]-rho*u[...,1]**2
    stress = np.stack((np.stack((Pxx-p,Pxy),axis=-1),
                       np.stack((Pxy,Pyy-p),axis=-1)),axis=-2)
    Pu = np.stack((Pxx*u[...,0]+Pxy*u[...,1],
                   Pxy*u[...,0]+Pyy*u[...,1]),axis=-1)
    q_raw = z[...,7:9]-z[...,3,None]*u-Pu
    return stress, q_raw


def heat_flux_moments(z, mapping):
    """Conductive LU heat flux, with the public local pre-collision readout.

    This moment diagnostic is not wall energy exchange or an acoustic pressure.
    """
    z = np.asarray(z, dtype=float)
    macro = recover_moment_macro(z, mapping)
    _, q_raw = _measured_state(z, macro)
    transport = local_transport_coefficients(mapping, macro.rho, macro.theta)
    return q_raw * transport.heat_flux_readout[..., None]


def collide_moments(z, mapping, lattice=None, backend='numpy'):
    """Return real post-collision f,g from the nine sufficient raw moments.

    Positivity is not imposed on moments or on a fictitious pre-collision lift.
    Callers must inspect the reconstructed and streamed populations themselves.
    Transport, equilibrium, projections and conservation corrections are shared
    with collide_fg. Equivalence is to float64 roundoff, not bitwise identity.
    """
    from core.collision_smrt import (
        _fixed_zero_lower_projector, _reconstruct_collision, validate_collision_backend)
    validate_collision_backend(backend)
    lattice = lattice or make_lattice()
    z = np.asarray(z,dtype=float)
    macro = recover_moment_macro(z,mapping)
    stress, q_raw = _measured_state(z, macro)
    transport = local_transport_coefficients(mapping,macro.rho,macro.theta)
    fe,ge = equilibrium_fg(macro.rho,macro.u,macro.theta,mapping.lattice.S,lattice)
    if backend == 'numba':
        from core.collision_numba import collide_measured
        key,weights = tuple(map(tuple,lattice.c)),tuple(lattice.w)
        projectors = tuple(_fixed_zero_lower_projector(key,weights,kind)
                           for kind in ('second','heat','first'))
        compact = np.stack((stress[...,0,0],stress[...,0,1],stress[...,1,1]),axis=-1)
        return collide_measured(compact,q_raw,z[...,1:3],fe,ge,macro,transport,mapping,lattice,projectors)
    return _reconstruct_collision(macro,stress,q_raw,z[...,1:3],fe,ge,transport,mapping,lattice)

def pressure_entropy_coordinates(z, mapping):
    """Reversible coordinates [p, ux, uy, sigma, Pi_xx, Pi_xy, Pi_yy, qraw_x, qraw_y].

    sigma = log(p/p0) - gamma*log(rho/rho0), referenced to the actual
    mapping state p0=rho_ref_lu*theta_ref_lu. These coordinates do not define
    independent autonomous thermal/acoustic fields or alter the collision.
    qraw is the central energy-flux moment, without transport readout factors.
    Nonpositive density or pressure is rejected, never clipped.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim < 1 or z.shape[-1] != 9:
        raise ValueError('moment state must end in nine components')
    rho = z[..., 0]
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError('rho must be finite and positive')
    internal = z[..., 3] - .5*np.sum(z[..., 1:3]**2, axis=-1)/rho
    if np.any(~np.isfinite(internal)) or np.any(internal <= 0):
        raise ValueError('pressure must be finite and positive')
    if np.any(~np.isfinite(z)):
        raise ValueError('moment state must be finite')
    macro = recover_moment_macro(z, mapping)
    stress, qraw = _measured_state(z, macro)
    rho0 = mapping.lattice.rho_ref_lu
    p0 = rho0*mapping.theta_ref_lu
    sigma = np.log(macro.p/p0) - macro.gamma*np.log(rho/rho0)
    return np.stack((macro.p, macro.u[..., 0], macro.u[..., 1], sigma,
                     stress[..., 0, 0], stress[..., 0, 1], stress[..., 1, 1],
                     qraw[..., 0], qraw[..., 1]), axis=-1)


def moments_from_pressure_entropy(coords, mapping):
    """Invert pressure/velocity/entropy/stress/qraw coordinates into nine moments.

    This is an algebraic coordinate change, with no population lifting, clipping
    or transport-law substitution. Positive p and recovered rho are required.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim < 1 or coords.shape[-1] != 9:
        raise ValueError('pressure entropy coordinates must end in nine components')
    p = coords[..., 0]
    if np.any(~np.isfinite(p)) or np.any(p <= 0):
        raise ValueError('pressure must be finite and positive')
    if np.any(~np.isfinite(coords)):
        raise ValueError('pressure entropy coordinates must be finite')
    rho0 = mapping.lattice.rho_ref_lu
    p0 = rho0*mapping.theta_ref_lu
    gamma = 1 + 2/(mapping.lattice.D + mapping.lattice.S)
    # Combine the two exponentials to avoid unnecessary intermediate overflow.
    with np.errstate(over='ignore', under='ignore'):
        rho = rho0*np.exp((np.log(p/p0)-coords[..., 3])/gamma)
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0):
        raise ValueError('recovered rho must be finite and positive')
    ux, uy = coords[..., 1], coords[..., 2]
    E = p/(gamma-1) + .5*rho*(ux*ux+uy*uy)
    Pxx, Pxy, Pyy = p+coords[..., 4], coords[..., 5], p+coords[..., 6]
    return np.stack((rho, rho*ux, rho*uy, E,
                     rho*ux*ux+Pxx, rho*ux*uy+Pxy, rho*uy*uy+Pyy,
                     E*ux+Pxx*ux+Pxy*uy+coords[..., 7],
                     E*uy+Pxy*ux+Pyy*uy+coords[..., 8]), axis=-1)
