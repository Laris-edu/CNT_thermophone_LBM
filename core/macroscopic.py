"""Macroscopic recovery and energy diagnostics for Phase 2 f-g distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.lattice import Lattice, make_lattice
from core.unit_mapping import UnitMapping, local_transport_coefficients


ENERGY_CLOSURE_DEFINITION = "central: E_tot = 0.5*rho*|u|^2 + K_tr + G_int"


@dataclass
class MacroState:
    rho: np.ndarray
    u: np.ndarray
    theta: np.ndarray
    p: np.ndarray
    e: np.ndarray
    gamma: float
    mach: np.ndarray
    K_tr: np.ndarray
    G_int: np.ndarray
    E_tot: np.ndarray


def density(f: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(f, dtype=float), axis=-1)


def velocity(f: np.ndarray, rho: np.ndarray | None = None, lattice: Lattice | None = None) -> np.ndarray:
    lattice = lattice or make_lattice()
    f = np.asarray(f, dtype=float)
    rho = density(f) if rho is None else np.asarray(rho, dtype=float)
    momentum = np.einsum("...a,ai->...i", f, lattice.c)
    return momentum / rho[..., None]


def translational_internal_energy(
    f: np.ndarray,
    u: np.ndarray,
    lattice: Lattice | None = None,
) -> np.ndarray:
    lattice = lattice or make_lattice()
    peculiar = lattice.c - np.asarray(u, dtype=float)[..., None, :]
    c2 = np.sum(peculiar**2, axis=-1)
    return 0.5 * np.sum(np.asarray(f, dtype=float) * c2, axis=-1)


def internal_energy_scalar(
    f: np.ndarray,
    g: np.ndarray,
    u: np.ndarray,
    lattice: Lattice | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_tr = translational_internal_energy(f, u, lattice)
    G_int = np.sum(np.asarray(g, dtype=float), axis=-1)
    return K_tr + G_int, K_tr, G_int


def recover_macro(
    f: np.ndarray,
    g: np.ndarray,
    *,
    D: int = 2,
    S: int = 3,
    lattice: Lattice | None = None,
) -> MacroState:
    lattice = lattice or make_lattice()
    rho = density(f)
    u = velocity(f, rho, lattice)
    internal, K_tr, G_int = internal_energy_scalar(f, g, u, lattice)
    theta = 2.0 * internal / ((D + S) * rho)
    p = rho * theta
    gamma = 1.0 + 2.0 / (D + S)
    speed = np.sqrt(np.sum(u * u, axis=-1))
    c_s = np.sqrt(gamma * theta)
    mach = np.divide(speed, c_s, out=np.zeros_like(speed, dtype=float), where=c_s != 0.0)
    e = internal / rho
    E_tot = 0.5 * rho * speed**2 + internal
    return MacroState(
        rho=rho,
        u=u,
        theta=theta,
        p=p,
        e=e,
        gamma=gamma,
        mach=mach,
        K_tr=K_tr,
        G_int=G_int,
        E_tot=E_tot,
    )


def central_energy_flux_lu(
    f: np.ndarray,
    g: np.ndarray,
    u: np.ndarray | None = None,
    lattice: Lattice | None = None,
) -> np.ndarray:
    """Return raw central energy-flux moment with component order ``(q_x, q_y)``."""

    lattice = lattice or make_lattice()
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    if u is None:
        u = velocity(f, lattice=lattice)
    u = np.asarray(u, dtype=float)
    peculiar = lattice.c - u[..., None, :]
    c2 = np.sum(peculiar**2, axis=-1)
    translational = 0.5 * np.einsum("...a,...a,...ai->...i", f, c2, peculiar)
    internal = np.einsum("...a,...ai->...i", g, peculiar)
    return translational + internal


def heat_flux_lu(
    f: np.ndarray,
    g: np.ndarray,
    u: np.ndarray | None = None,
    lattice: Lattice | None = None,
    mapping: UnitMapping | None = None,
) -> np.ndarray:
    """Return conductive lattice heat flux with component order ``(q_x, q_y)``.

    The raw central energy-flux moment remains available through
    :func:`central_energy_flux_lu`.  When a :class:`UnitMapping` is supplied,
    the configured conductive heat-flux moment factor is applied so exported
    ``q_lu`` fields use the Fourier-law/Phase_3 handoff convention.
    """

    q_raw = central_energy_flux_lu(f, g, u=u, lattice=lattice)
    if mapping is None:
        return q_raw
    if mapping.transport.mode == "local":
        state = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
        coefficients = local_transport_coefficients(mapping, state.rho, state.theta)
        # 此关系读取碰撞前布居；是离散时间半步修正，不是标定增益。
        return q_raw * coefficients.heat_flux_readout[..., None]
    q_conductive = q_raw * mapping.collision.conductive_heat_flux_moment_factor
    correction_factor = mapping.collision.conductive_heat_flux_galilean_correction_factor
    if correction_factor == 0.0:
        return q_conductive
    state = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    theta_prime = state.theta - mapping.theta_ref_lu
    return q_conductive + correction_factor * state.u * theta_prime[..., None]
