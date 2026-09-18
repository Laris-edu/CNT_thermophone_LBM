"""Mass-neutral Grad/regularized thermal Dirichlet wall (Phase_5 WP1-3, contract §6.1).

G1-W production-wall candidate. Identical Grad/regularized reconstruction to
``wall_thermal_grad`` — ``f0 = feq(rho_w, 0, theta_w) + interior-neq copy`` with the
uniform ``g`` delta pinning the internal energy, so ``theta = theta_w`` and ``u = 0``
hold exactly and the physical non-equilibrium heat flux is retained (NOT an
equilibrium clamp) — with exactly ONE change:

    rho_w is not prescribed. The wall row keeps the post-stream row-0 density
    column by column, so the wall operation is **mass-neutral by construction**
    (its net change of total mass is identically zero up to floating-point
    summation noise).

Rationale (G1-W risk basis, contract §6.1): the frozen ``pressure_preserving``
policy sets ``rho_w = p_ref/theta_w`` each step; under ``theta_w = theta0 (1 +
eps cos)`` this writes an O(eps) 1f plus O(eps^2) DC/2f density source into the
domain before the equations of motion act. Keeping the streamed density instead
makes the wall pressure ``p_w = rho_w * theta_w`` a *dynamic* quantity that
responds to the incident field — physically the correct rigid-wall behaviour
(the acoustic pressure at a rigid wall oscillates; it is not clamped).

Default OFF everywhere: no frozen config or production path imports this
module; the M3/M4 stacks stay byte-identical. Production adoption is decided by
the G1-W gate (WP2), including the small-amplitude admittance regression
(amplitude <=5%, phase <=5 deg) and the old-vs-new difference audit.
No clipping / floor / positivity repair is used.

STATUS (2026-07-21, candidate v1, NOT certified): mass metrics are perfect at
production scale (10 kHz dx2p6 ny=48: flux components <=1.7e-17 vs the 1e-10
gate, 2-period drift 2.3e-13) and the run is stable, BUT the Level-A-style
admittance preview reads amplitude -37.9% vs the analytic half-space reference
(phase +2.65 deg; ratio to the frozen wall 0.656 @ +0.45 deg) — far outside
the <=5% regression row, so v1 must NOT enter G1a as-is. Open diagnosis
(Phase5_STATUS §3): (a) instrument — the conduction-moment export factor is a
narrow-band (tau,k) point calibration made on the pressure-clamped stack, and
a changed near-wall field structure can misread by O(10%) easily (FD-gradient
dual-channel discriminator designed); (b) physics — with mass conserved the
y-periodic rig becomes a genuinely closed box heated from both wrap sides
(pressure-release vs rigid acoustic character); (c) wall-model deficiency.
The frozen pressure-preserving wall reproduces the M3 admittance exactly in
the same probe (-5.32%/+2.198 deg), so the probe chain itself is validated.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from core.solver import GasSolver2D

from .wall_common import BOTTOM_WALL_ROW, bottom_wall_stencil


def _mass_neutral_reconstruct_row(
    solver: GasSolver2D,
    f_stream: np.ndarray,
    g_stream: np.ndarray,
    theta_w: float,
    *,
    row: int,
    interior_sign: int,
    extrap: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Mass-neutral reconstruction of ``row`` at ``theta_w`` (in place).

    ``interior_sign`` points from the wall row into the gas (+1 bottom wall,
    -1 top lid); the non-equilibrium is copied/extrapolated from that side.
    """

    lattice = solver.lattice
    D = int(solver.mapping.lattice.D)
    S = int(solver.mapping.lattice.S)
    q = int(lattice.q)
    c2 = np.sum(np.asarray(lattice.c, dtype=float) ** 2, axis=-1)  # (q,)

    if theta_w <= 0.0:
        raise ValueError("wall temperature must be positive")
    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    nx = f_stream.shape[1]
    # column-by-column post-stream density is KEPT -> zero net mass change
    rho_w = np.sum(f_stream[row:row + 1], axis=-1)  # (1, nx)
    if not np.all(rho_w > 0.0):
        raise RuntimeError("non-positive streamed wall-row density")
    theta_row = np.full((1, nx), float(theta_w))
    feq_w, geq_w = equilibrium_fg(
        rho_w, np.zeros((1, nx, 2)), theta_row, S, lattice
    )  # (1, nx, q); mass = rho_w and momentum = 0 exactly

    def interior_neq(j):
        m = recover_macro(f_stream[j:j + 1], g_stream[j:j + 1], D=D, S=S, lattice=lattice)
        feq_j, geq_j = equilibrium_fg(m.rho, m.u, m.theta, S, lattice)
        return f_stream[j:j + 1] - feq_j, g_stream[j:j + 1] - geq_j  # (1,nx,q)

    j1 = row + interior_sign
    f1n, g1n = interior_neq(j1)
    if extrap == "linear" and solver.ny >= 3:
        f2n, g2n = interior_neq(row + 2 * interior_sign)
        f_neq = 2.0 * f1n - f2n
        g_neq = 2.0 * g1n - g2n
    else:
        f_neq, g_neq = f1n, g1n

    f0 = feq_w + f_neq                                     # rho_w & u=0 exact
    k_tr = 0.5 * np.sum(f0 * c2, axis=-1)                  # (1,nx) (u=0 -> |c-u|^2=|c|^2)
    g_partial = np.sum(geq_w, axis=-1) + np.sum(g_neq, axis=-1)   # (1,nx)
    target_int = 0.5 * (D + S) * rho_w * theta_w
    delta = (target_int - k_tr - g_partial) / q            # (1,nx)
    g0 = geq_w + g_neq + delta[..., None]

    f_stream[row:row + 1] = f0
    g_stream[row:row + 1] = g0
    return f_stream, g_stream


def _mass_neutral_reconstruct_bottom_row(
    solver: GasSolver2D,
    f_stream: np.ndarray,
    g_stream: np.ndarray,
    theta_w: float,
    *,
    extrap: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared mass-neutral row-0 reconstruction at ``theta_w`` (in place)."""

    return _mass_neutral_reconstruct_row(
        solver, f_stream, g_stream, theta_w,
        row=BOTTOM_WALL_ROW, interior_sign=+1, extrap=extrap,
    )


def make_bottom_mass_neutral_wall_callback(
    theta_wall_lu: float | Callable[[GasSolver2D], float],
    *,
    extrap: str = "row1",
    row: int = BOTTOM_WALL_ROW,
):
    """Return a ``solver.step`` boundary_callback imposing the mass-neutral wall.

    Same signature family as ``make_bottom_grad_wall_callback``; there is no
    ``rho_policy`` — mass neutrality *is* the density policy.
    """

    if row != BOTTOM_WALL_ROW:
        raise ValueError("only the bottom wall (row 0) is supported")
    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    def _callback(*, solver, f_post, g_post, f_stream, g_stream):
        theta_w = float(theta_wall_lu(solver) if callable(theta_wall_lu) else theta_wall_lu)
        return _mass_neutral_reconstruct_bottom_row(
            solver, f_stream, g_stream, theta_w, extrap=extrap
        )

    return _callback


def _mass_neutral_reconstruct_row0_symmetric(
    solver: GasSolver2D,
    f_stream: np.ndarray,
    g_stream: np.ndarray,
    theta_w: float,
    *,
    extrap: str,
    row: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """v1.1 symmetric two-sided mass-neutral reconstruction of row 0.

    In the y-periodic sealed rig the wall row serves BOTH sides (rows 1.. above
    and, via the wrap, rows ny-1.. below). v1 copied the non-equilibrium from
    row 1 only, which makes the wrap side a malformed boundary (measured
    top/bottom moment-flux asymmetry 35 vs 308, Phase5_STATUS 2026-07-22).
    Here the non-equilibrium is built per direction: populations with cy>0
    (feeding the upper gas) take the row-1 side, cy<0 (feeding the wrap side)
    take the row ny-1 side, grazing directions take the average. The blended
    non-equilibrium no longer has exactly zero mass/momentum, so both are
    removed EXACTLY by subtracting the pure equilibrium increment
    ``feq(rho_w+drho, dj/(rho_w+drho), theta_w) - feq(rho_w, 0, theta_w)``
    (smoothest possible removal — no ghost content, the min-norm lesson).
    Mass neutrality, u=0 and theta=theta_w remain machine-exact.

    ``row`` generalizes the band position on the periodic lattice (G4a tent
    rig: hot band at row 0, ambient sink band at row ny//2). Neighbours are
    taken modulo ny (up = row+1, down = row-1); ``row=0`` reproduces the
    original v1.1 wall byte-identically (row-1 == ny-1 by the wrap).
    """

    lattice = solver.lattice
    D = int(solver.mapping.lattice.D)
    S = int(solver.mapping.lattice.S)
    q = int(lattice.q)
    c = np.asarray(lattice.c, dtype=float)  # (q, 2)
    c2 = np.sum(c**2, axis=-1)

    if theta_w <= 0.0:
        raise ValueError("wall temperature must be positive")
    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")
    if solver.ny < 5:
        raise ValueError("symmetric two-sided wall needs ny >= 5")

    ny = int(solver.ny)
    row = int(row) % ny
    nx = f_stream.shape[1]
    rho_w = np.sum(f_stream[row:row + 1], axis=-1)  # (1, nx) kept -> mass neutral
    if not np.all(rho_w > 0.0):
        raise RuntimeError("non-positive streamed wall-row density")
    theta_row = np.full((1, nx), float(theta_w))
    zeros_u = np.zeros((1, nx, 2))
    feq_w, geq_w = equilibrium_fg(rho_w, zeros_u, theta_row, S, lattice)

    def interior_neq(j):
        m = recover_macro(f_stream[j:j + 1], g_stream[j:j + 1], D=D, S=S, lattice=lattice)
        feq_j, geq_j = equilibrium_fg(m.rho, m.u, m.theta, S, lattice)
        return f_stream[j:j + 1] - feq_j, g_stream[j:j + 1] - geq_j

    up_f, up_g = interior_neq((row + 1) % ny)
    dn_f, dn_g = interior_neq((row - 1) % ny)
    if extrap == "linear":
        up2_f, up2_g = interior_neq((row + 2) % ny)
        dn2_f, dn2_g = interior_neq((row - 2) % ny)
        up_f, up_g = 2.0 * up_f - up2_f, 2.0 * up_g - up2_g
        dn_f, dn_g = 2.0 * dn_f - dn2_f, 2.0 * dn_g - dn2_g

    st = bottom_wall_stencil(lattice)
    f_neq = np.empty_like(up_f)
    g_neq = np.empty_like(up_g)
    f_neq[..., st.incoming] = up_f[..., st.incoming]
    g_neq[..., st.incoming] = up_g[..., st.incoming]
    f_neq[..., st.outgoing] = dn_f[..., st.outgoing]
    g_neq[..., st.outgoing] = dn_g[..., st.outgoing]
    f_neq[..., st.grazing] = 0.5 * (up_f + dn_f)[..., st.grazing]
    g_neq[..., st.grazing] = 0.5 * (up_g + dn_g)[..., st.grazing]

    # exact mass/momentum removal via the equilibrium increment
    drho = np.sum(f_neq, axis=-1)                       # (1, nx)
    dj = np.einsum("yxq,qd->yxd", f_neq, c)             # (1, nx, 2)
    rho_pert = rho_w + drho
    if not np.all(rho_pert > 0.0):
        raise RuntimeError("blended non-equilibrium exceeds wall density")
    u_pert = dj / rho_pert[..., None]
    feq_pert, geq_pert = equilibrium_fg(rho_pert, u_pert, theta_row, S, lattice)
    f_neq = f_neq - (feq_pert - feq_w)
    g_neq = g_neq - (geq_pert - geq_w)

    f0 = feq_w + f_neq                                  # rho_w & u=0 exact
    k_tr = 0.5 * np.sum(f0 * c2, axis=-1)
    g_partial = np.sum(geq_w, axis=-1) + np.sum(g_neq, axis=-1)
    target_int = 0.5 * (D + S) * rho_w * theta_w
    delta = (target_int - k_tr - g_partial) / q
    g0 = geq_w + g_neq + delta[..., None]

    f_stream[row:row + 1] = f0
    g_stream[row:row + 1] = g0
    return f_stream, g_stream


def make_symmetric_mass_neutral_band_callback(
    theta_band_lu: float | Callable[[GasSolver2D], float],
    row: int,
    *,
    extrap: str = "row1",
):
    """G4a tent band: the v1.1 symmetric reconstruction at an arbitrary row.

    Same certified construction as ``make_symmetric_mass_neutral_wall_callback``
    (per-direction neq, equilibrium-increment cleanup, exact energy pinning),
    placed at ``row`` on the periodic lattice. Used for the tent rig's ambient
    sink band (row ny//2) and, composed with the row-0 wall, keeps the field
    free of first-order jumps (the WP1-3 lid-rig death variable).
    """

    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    def _callback(*, solver, f_post, g_post, f_stream, g_stream):
        theta_b = float(theta_band_lu(solver) if callable(theta_band_lu) else theta_band_lu)
        return _mass_neutral_reconstruct_row0_symmetric(
            solver, f_stream, g_stream, theta_b, extrap=extrap, row=int(row)
        )

    return _callback


def make_symmetric_mass_neutral_wall_callback(
    theta_wall_lu: float | Callable[[GasSolver2D], float],
    *,
    extrap: str = "row1",
):
    """v1.1: symmetric two-sided mass-neutral wall for the y-periodic sealed rig."""

    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    def _callback(*, solver, f_post, g_post, f_stream, g_stream):
        theta_w = float(theta_wall_lu(solver) if callable(theta_wall_lu) else theta_wall_lu)
        return _mass_neutral_reconstruct_row0_symmetric(
            solver, f_stream, g_stream, theta_w, extrap=extrap
        )

    return _callback


def make_top_mass_neutral_lid_callback(
    theta_lid_lu: float | Callable[[GasSolver2D], float],
    *,
    extrap: str = "row1",
):
    """Return a ``solver.step`` boundary_callback imposing a mass-neutral
    isothermal lid on the TOP row — the canonical A2a heat sink
    ``T(H_s)=T_ambient`` (contract §3.3) in mass-neutral form.

    Same reconstruction as the bottom wall with the interior on the other
    side; compose with the bottom wall over disjoint rows.  The open-boundary
    ``compose_boundary_callbacks`` helper is not part of this repository (it
    lives in the legacy JFM2024 repository's ``boundary/open_cbc.py``).
    """

    if extrap not in {"linear", "row1"}:
        raise ValueError(f"unsupported non-equilibrium extrapolation: {extrap}")

    def _callback(*, solver, f_post, g_post, f_stream, g_stream):
        theta_l = float(theta_lid_lu(solver) if callable(theta_lid_lu) else theta_lid_lu)
        return _mass_neutral_reconstruct_row(
            solver, f_stream, g_stream, theta_l,
            row=solver.ny - 1, interior_sign=-1, extrap=extrap,
        )

    return _callback


def apply_bottom_mass_neutral_wall_inplace(
    solver: GasSolver2D,
    theta_wall_lu: float,
    *,
    extrap: str = "row1",
) -> None:
    """Reconstruct the bottom wall row 0 in place on the current solver state.

    Mirror of ``apply_bottom_grad_wall_inplace`` for future Level C coupling:
    row 1 is unaffected, so the extracted near-wall q_g is unchanged.
    """

    cb = make_bottom_mass_neutral_wall_callback(float(theta_wall_lu), extrap=extrap)
    cb(solver=solver, f_post=solver.f, g_post=solver.g, f_stream=solver.f, g_stream=solver.g)
