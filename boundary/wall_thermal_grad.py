"""Grad / regularized wet-node thermal Dirichlet wall for D2Q37 (doc §5.4, stabilized).

Both prior attempts failed: the ABB prototype over-injects energy for the polyatomic f/g
pair, and the min-norm moment reconstruction excites an unstable near-wall momentum ghost
mode. This module reconstructs the whole bottom wall row 0 as a *regularized* distribution

    f_row0 = feq(rho_w, 0, theta_w) + f1_neq
    g_row0 = geq(rho_w, 0, theta_w) + g1_neq + delta

where ``f1_neq/g1_neq`` are the NON-equilibrium parts extrapolated from the first interior
gas row (row 1) -- a physically regularized (RR) interior state, so no arbitrary ghost
content is injected (the source of the min-norm instability). Because ``feq_hermite4``
matches mass/momentum exactly, ``f1_neq`` has zero mass/momentum, so ``u = 0`` and
``rho = rho_w`` hold exactly; the uniform ``delta`` on ``g`` pins the total internal energy
to ``0.5 (D+S) rho_w theta_w`` (i.e. ``theta = theta_w`` exactly) without changing the
heat flux (``sum_a c_a = 0``). The retained ``f1_neq`` shear + f/g heat-flux non-equilibrium
lets the near-wall conductive flux persist (the equilibrium clamp killed it).

No clipping / floor / positivity repair is used.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from core.solver import GasSolver2D

from .wall_common import BOTTOM_WALL_ROW, bottom_wall_stencil, pressure_preserving_rho


def _grad_reconstruct_bottom_row(
    solver: GasSolver2D,
    f_stream: np.ndarray,
    g_stream: np.ndarray,
    theta_w: float,
    *,
    rho_policy: str,
    extrap: str,
    fill_deep_links: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared Grad/regularized row-0 reconstruction at ``theta_w`` (in place)."""

    lattice = solver.lattice
    D = int(solver.mapping.lattice.D)
    S = int(solver.mapping.lattice.S)
    q = int(lattice.q)
    c2 = np.sum(np.asarray(lattice.c, dtype=float) ** 2, axis=-1)  # (q,)

    if theta_w <= 0.0:
        raise ValueError("wall temperature must be positive")
    if rho_policy == "pressure_preserving":
        rho_w = pressure_preserving_rho(theta_w, solver.mapping)
    elif rho_policy == "constant_density":
        rho_w = float(solver.mapping.lattice.rho_ref_lu)
    else:
        raise ValueError(f"unknown rho_policy: {rho_policy}")

    f_w, g_w = equilibrium_fg(
        np.full((1, 1), rho_w), np.zeros((1, 1, 2)), np.full((1, 1), theta_w),
        S, lattice,
    )
    feq_w = np.asarray(f_w[0, 0], dtype=float)  # (q,)
    geq_w = np.asarray(g_w[0, 0], dtype=float)

    def interior_neq(j):
        m = recover_macro(f_stream[j:j + 1], g_stream[j:j + 1], D=D, S=S, lattice=lattice)
        feq_j, geq_j = equilibrium_fg(m.rho, m.u, m.theta, S, lattice)
        return f_stream[j:j + 1] - feq_j, g_stream[j:j + 1] - geq_j  # (1,nx,q)

    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    f1n, g1n = interior_neq(1)
    if extrap == "linear" and solver.ny >= 3:
        f2n, g2n = interior_neq(2)
        f_neq = 2.0 * f1n - f2n
        g_neq = 2.0 * g1n - g2n
    else:
        f_neq, g_neq = f1n, g1n

    f0 = feq_w[None, None, :] + f_neq                      # (1,nx,q); rho_w & u=0 exact
    k_tr = 0.5 * np.sum(f0 * c2, axis=-1)                  # (1,nx)  (u=0 -> |c-u|^2=|c|^2)
    g_partial = float(np.sum(geq_w)) + np.sum(g_neq, axis=-1)   # (1,nx)
    target_int = 0.5 * (D + S) * rho_w * theta_w
    delta = (target_int - k_tr - g_partial) / q           # (1,nx)
    g0 = geq_w[None, None, :] + g_neq + delta[..., None]

    f_stream[0:1] = f0
    g_stream[0:1] = g0

    # Optional: deep below-wall links for rows 1..max_cy-1 zero-gradient from the wall row.
    # Off by default -- forcing rows 1,2 specific directions to the wall value injects an
    # odd/even wall-normal mode; leaving them as the periodic-top (~far-field mean) stream
    # is smoother for small perturbations.
    if fill_deep_links:
        st = bottom_wall_stencil(lattice)
        f0row = f_stream[0].copy()
        g0row = g_stream[0].copy()
        for a in st.incoming:
            for y in range(1, int(st.cy[a])):
                f_stream[y, :, a] = f0row[:, a]
                g_stream[y, :, a] = g0row[:, a]
    return f_stream, g_stream


def make_bottom_grad_wall_callback(
    theta_wall_lu: float | Callable[[GasSolver2D], float],
    *,
    rho_policy: str = "pressure_preserving",
    extrap: str = "row1",
    fill_deep_links: bool = False,
    row: int = BOTTOM_WALL_ROW,
):
    """Return a ``solver.step`` boundary_callback imposing a Grad/regularized bottom wall.

    ``extrap`` selects the near-wall non-equilibrium extrapolation: ``"row1"`` (0th order,
    copy row 1) or ``"linear"`` (``2*neq1 - neq2``).
    """

    if row != BOTTOM_WALL_ROW:
        raise ValueError("only the bottom wall (row 0) is supported")

    def _callback(*, solver, f_post, g_post, f_stream, g_stream):
        theta_w = float(theta_wall_lu(solver) if callable(theta_wall_lu) else theta_wall_lu)
        return _grad_reconstruct_bottom_row(
            solver, f_stream, g_stream, theta_w,
            rho_policy=rho_policy, extrap=extrap, fill_deep_links=fill_deep_links,
        )

    return _callback


def neumann_theta_wall_lu(
    solver: GasSolver2D,
    f_stream: np.ndarray,
    g_stream: np.ndarray,
    q_wall_si: float,
    *,
    kg_si: float,
    flux_stencil: str = "second_order",
) -> float:
    """Wall temperature that pins the one-sided finite-difference gradient to ``-q/k``.

    KEPT AS A DOCUMENTED NEGATIVE RESULT / FORMULA HELPER: pinning the FD temperature
    gradient is a *refuted* Level B flux controller -- the near-wall temperature gradient
    systematically under-represents the conduction-moment flux (the certified q_g''
    measure), so this delivers ~2.5x the intended moment flux (P3-6 run ``2566fe52...``,
    Phase3_STATUS.md). The committed Level B script servos the moment extraction instead
    (``scripts/phase3_levelb_admittance.py``).

    Converts the prescribed one-sided gas heat flux ``q_g''=-k_g dT/dy|0+`` (SI, positive
    from film into the upper gas) into the wet-node wall temperature via the one-sided
    stencil on the post-stream interior rows:

        second_order: T_0 = (4 T_1 - T_2 + 2 dx q/k) / 3
        first_order:  T_0 = T_1 + dx q/k

    All conversions stay in SI against the reference ``kg_si`` (the same conductivity the
    analytic admittance reference uses); no lattice transport quantity is re-derived here.
    """

    D = int(solver.mapping.lattice.D)
    S = int(solver.mapping.lattice.S)
    scale = float(solver.mapping.temperature_scale)
    dx_m = float(solver.mapping.lattice.dx_m)

    def row_T(j: int) -> float:
        m = recover_macro(f_stream[j:j + 1], g_stream[j:j + 1], D=D, S=S, lattice=solver.lattice)
        return float(np.mean(np.asarray(m.theta, dtype=float))) * scale

    dT = dx_m * float(q_wall_si) / float(kg_si)
    if flux_stencil == "second_order":
        if solver.ny < 3:
            raise ValueError("second_order flux stencil needs ny >= 3")
        T_w = (4.0 * row_T(1) - row_T(2) + 2.0 * dT) / 3.0
    elif flux_stencil == "first_order":
        T_w = row_T(1) + dT
    else:
        raise ValueError(f"unknown flux_stencil: {flux_stencil}")
    return T_w / scale




def apply_bottom_grad_wall_inplace(
    solver: GasSolver2D,
    theta_wall_lu: float,
    *,
    rho_policy: str = "pressure_preserving",
    extrap: str = "linear",
) -> None:
    """Reconstruct the bottom wall row 0 in place on the current solver state.

    Reuses the callback with the current ``solver.f/g`` as the post-stream field (row 1's
    non-equilibrium is read from the live state). Used by the Level C coupler to re-impose
    the wall temperature at the corrected film temperature outside a streaming step (row 1
    is unaffected, so the extracted near-wall q_g is unchanged).
    """

    cb = make_bottom_grad_wall_callback(
        float(theta_wall_lu), rho_policy=rho_policy, extrap=extrap, fill_deep_links=False
    )
    cb(solver=solver, f_post=solver.f, g_post=solver.g, f_stream=solver.f, g_stream=solver.g)
