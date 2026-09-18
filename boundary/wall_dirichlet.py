"""Level A prescribed-temperature wall helpers for Phase_3.

The first Phase_3 wall implementation is intentionally explicit: it clamps the
bottom wall row to a no-slip equilibrium state with the requested thermodynamic
wall temperature. It does not recompute transport relaxation parameters and it
does not use clipping or population floors.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from core.solver import GasSolver2D
from phase3_interfaces.wall_state_contract import (
    wall_state_from_temperature,
    wall_state_from_theta,
)


BOTTOM_WALL_ROW = 0
LEVEL_A_WALL_NORMAL_CONVENTION = "upper gas half-domain: wall_normal=+e_y"
LEVEL_A_HEAT_FLUX_SIGN_CONVENTION = "q_g''=-k_g*dT/dy|0+ positive from film into upper gas"
RhoWallPolicy = Literal["pressure_preserving", "constant_density"]


@dataclass(frozen=True)
class DirichletWallDiagnostics:
    row: int
    theta_wall_lu: float
    recovered_theta_wall_lu: float
    T_wall_K: float
    rho_wall_lu: float
    rho_wall_policy: str
    velocity_set: str
    Q: int
    max_theta_error_lu: float
    max_velocity_lu: float
    mass_before_lu: float
    mass_after_lu: float
    mass_delta_lu: float
    wall_normal_convention: str = LEVEL_A_WALL_NORMAL_CONVENTION
    heat_flux_sign_convention: str = LEVEL_A_HEAT_FLUX_SIGN_CONVENTION

    @property
    def finite(self) -> bool:
        values = (
            self.theta_wall_lu,
            self.recovered_theta_wall_lu,
            self.T_wall_K,
            self.rho_wall_lu,
            self.max_theta_error_lu,
            self.max_velocity_lu,
            self.mass_before_lu,
            self.mass_after_lu,
            self.mass_delta_lu,
        )
        return all(math.isfinite(float(value)) for value in values)


def wall_theta_from_inputs(
    solver: GasSolver2D,
    *,
    theta_wall_lu: float | None = None,
    T_wall_K: float | None = None,
) -> float:
    """Resolve a wall temperature from either lattice or SI units."""

    if (theta_wall_lu is None) == (T_wall_K is None):
        raise ValueError("provide exactly one of theta_wall_lu or T_wall_K")
    if theta_wall_lu is not None:
        return float(wall_state_from_theta(theta_wall_lu, solver.config)["theta_wall_lu"])
    return float(wall_state_from_temperature(T_wall_K, solver.config)["theta_wall_lu"])


def wall_density_lu(
    solver: GasSolver2D,
    theta_wall_lu: float,
    *,
    rho_policy: RhoWallPolicy = "pressure_preserving",
) -> float:
    """Return the density used for the clamped wall row."""

    if rho_policy == "pressure_preserving":
        p_ref_lu = solver.mapping.lattice.rho_ref_lu * solver.mapping.theta_ref_lu
        return float(p_ref_lu / float(theta_wall_lu))
    if rho_policy == "constant_density":
        return float(solver.mapping.lattice.rho_ref_lu)
    raise ValueError(f"unknown rho_policy: {rho_policy}")


def apply_bottom_dirichlet_wall(
    solver: GasSolver2D,
    *,
    theta_wall_lu: float | None = None,
    T_wall_K: float | None = None,
    rho_policy: RhoWallPolicy = "pressure_preserving",
    row: int = BOTTOM_WALL_ROW,
) -> DirichletWallDiagnostics:
    """Clamp one bottom-wall row to no-slip and prescribed temperature."""

    if row != BOTTOM_WALL_ROW:
        raise ValueError("P3-1 supports the bottom wall row only")
    solver.get_macro()  # lazily initializes the solver state if needed
    assert solver.f is not None and solver.g is not None

    theta = wall_theta_from_inputs(
        solver,
        theta_wall_lu=theta_wall_lu,
        T_wall_K=T_wall_K,
    )
    rho_w = wall_density_lu(solver, theta, rho_policy=rho_policy)

    mass_before = float(np.sum(solver.f))
    rho = np.full((1, solver.nx), rho_w, dtype=float)
    u = np.zeros((1, solver.nx, 2), dtype=float)
    theta_field = np.full((1, solver.nx), theta, dtype=float)
    f_wall, g_wall = equilibrium_fg(
        rho,
        u,
        theta_field,
        solver.mapping.lattice.S,
        solver.lattice,
    )
    solver.f[row : row + 1] = f_wall
    solver.g[row : row + 1] = g_wall

    macro = recover_macro(
        solver.f[row : row + 1],
        solver.g[row : row + 1],
        D=solver.mapping.lattice.D,
        S=solver.mapping.lattice.S,
        lattice=solver.lattice,
    )
    mass_after = float(np.sum(solver.f))
    state = wall_state_from_theta(theta, solver.config)
    return DirichletWallDiagnostics(
        row=row,
        theta_wall_lu=theta,
        recovered_theta_wall_lu=float(np.mean(macro.theta)),
        T_wall_K=float(state["T_wall_K"]),
        rho_wall_lu=rho_w,
        rho_wall_policy=rho_policy,
        velocity_set=solver.mapping.lattice.velocity_set,
        Q=int(solver.lattice.q),
        max_theta_error_lu=float(np.max(np.abs(macro.theta - theta))),
        max_velocity_lu=float(np.max(np.linalg.norm(macro.u, axis=-1))),
        mass_before_lu=mass_before,
        mass_after_lu=mass_after,
        mass_delta_lu=mass_after - mass_before,
    )


def advance_with_bottom_dirichlet_wall(
    solver: GasSolver2D,
    *,
    theta_wall_lu: float | None = None,
    T_wall_K: float | None = None,
    rho_policy: RhoWallPolicy = "pressure_preserving",
    n_steps: int = 1,
) -> DirichletWallDiagnostics:
    """Advance the solver while keeping the bottom wall clamped."""

    diag = apply_bottom_dirichlet_wall(
        solver,
        theta_wall_lu=theta_wall_lu,
        T_wall_K=T_wall_K,
        rho_policy=rho_policy,
    )
    for _ in range(int(n_steps)):
        solver.step(1)
        diag = apply_bottom_dirichlet_wall(
            solver,
            theta_wall_lu=theta_wall_lu,
            T_wall_K=T_wall_K,
            rho_policy=rho_policy,
        )
    return diag


def sinusoidal_wall_temperature_lu(
    t_si,
    *,
    theta0_lu: float,
    theta_hat_lu: complex,
    frequency_hz: float,
):
    """Return ``theta0 + Re[theta_hat exp(i Omega t)]``."""

    t = np.asarray(t_si, dtype=float)
    omega = 2.0 * math.pi * float(frequency_hz)
    return float(theta0_lu) + np.real(complex(theta_hat_lu) * np.exp(1j * omega * t))

