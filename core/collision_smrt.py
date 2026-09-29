"""Conservative central-Hermite SMRT-style collision for Phase 2.

本项目实际运行且经回归覆盖的一条碰撞路径：

* 二阶中心矩闭合（``second_order``，唯一保留的闭合阶数）；
* 偏应力按实测非平衡中心矩以 ``(1 - 1/tau21)`` 松弛（``measured`` 口径）；
* 迹通道按 ``(1 - 1/tau22)`` 松弛，``nu_b = 0`` 对应 tau22 = 1/2、保留因子 -1；
* 热流按局部 ``1 - 1/tau32`` 做最小范数投影；
* 逐格质量、动量和所选总能量守恒，不做正性裁剪。

迁移时删除了不在这条路径上的分支（解析应变率偏应力、四阶闭合、
``ghost_orthogonal_local*`` 迹策略族、谱投影与 ``acoustic_simplified`` 开关）；
其中解析应变率分支的符号与本文件的实测口径相反，等效剪切黏性为负，
详见旧仓 ``docs/CORE_REVIEW_20260917.md``。松弛时间只能由
``core.unit_mapping`` 通过 :class:`core.unit_mapping.UnitMapping` 提供。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from core.equilibrium import equilibrium_fg
from core.lattice import Lattice, make_lattice
from core.macroscopic import ENERGY_CLOSURE_DEFINITION, central_energy_flux_lu, recover_macro
from core.unit_mapping import LocalTransport, UnitMapping, local_transport_coefficients


@dataclass
class CollisionDiagnostics:
    mass_residual: np.ndarray
    momentum_residual: np.ndarray
    energy_residual: np.ndarray
    min_f_post: float
    min_g_post: float
    clipping_used: bool = False


def validate_collision_backend(backend):
    if backend not in ('numpy', 'numba'):
        raise ValueError('collision_backend must be numpy or numba')
    return backend


@lru_cache(maxsize=12)
def _fixed_zero_lower_projector(c_values, weights, kind):
    """零低阶矩时，中心矩约束等价于原始矩约束；无需逐格解 Gram 系统。

    平移热流基只增加二阶及以下多项式，其增量矩全为零。因此这是原最小
    范数投影的代数等价实现，不将实际流速近似为零。缓存按格子内容区分。
    """
    c, w = np.asarray(c_values), np.asarray(weights)
    x, y = c[:, 0], c[:, 1]
    rows = [np.ones_like(x), y, x]
    start = 1
    if kind != 'first':
        rows += [y*y, x*y, x*x]
        start = 3
    if kind == 'heat':
        rows += [.5*(x*x+y*y)*y, .5*(x*x+y*y)*x]
        start = 6
    basis = np.stack(rows)
    wb = basis*w
    return np.linalg.solve(wb @ basis.T, wb)[start:]


def _project_zero_lower(delta, lattice, kind, start):
    projector = _fixed_zero_lower_projector(tuple(map(tuple, lattice.c)), tuple(lattice.w), kind)
    return np.einsum('...i,ia->...a', delta[..., start:], projector)


def _central_second_moment_matrix(lattice: Lattice, u: np.ndarray) -> np.ndarray:
    """Return rows for ``1, xi_y, xi_x, xi_y^2, xi_x xi_y, xi_x^2``."""

    xi = lattice.c - u[..., None, :]
    xi_x = xi[..., 0]
    xi_y = xi[..., 1]
    return np.stack(
        (
            np.ones_like(xi_x),
            xi_y,
            xi_x,
            xi_y * xi_y,
            xi_x * xi_y,
            xi_x * xi_x,
        ),
        axis=-2,
    )


def _weighted_minimum_norm_second_order_delta(
    delta_moments: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    """Project second-order central moments back to populations.

    This is the binomial/central-Hermite regularization step used for the
    current Phase 2 collision.  It reconstructs only the non-equilibrium
    second-order stress content, with zero density and momentum increments,
    while minimizing the weighted population norm set by the lattice weights.
    """

    if not np.any(delta_moments[..., :3]):
        return _project_zero_lower(delta_moments, lattice, 'second', 3)
    basis = _central_second_moment_matrix(lattice, u)
    weighted_basis = basis * lattice.w
    gram = np.einsum("...ia,...ja->...ij", weighted_basis, basis)
    multipliers = np.linalg.solve(gram, delta_moments[..., None])[..., 0]
    return np.einsum("...ia,...i,a->...a", basis, multipliers, lattice.w)


def _central_first_moment_matrix(lattice: Lattice, u: np.ndarray) -> np.ndarray:
    """Return rows for ``1, xi_y, xi_x``."""

    xi = lattice.c - u[..., None, :]
    xi_x = xi[..., 0]
    xi_y = xi[..., 1]
    return np.stack((np.ones_like(xi_x), xi_y, xi_x), axis=-2)


def _weighted_minimum_norm_first_order_delta(
    delta_moments: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    """Project g first-order central internal-energy flux to populations."""

    if not np.any(delta_moments[..., :1]):
        return _project_zero_lower(delta_moments, lattice, 'first', 1)
    basis = _central_first_moment_matrix(lattice, u)
    weighted_basis = basis * lattice.w
    gram = np.einsum("...ia,...ja->...ij", weighted_basis, basis)
    multipliers = np.linalg.solve(gram, delta_moments[..., None])[..., 0]
    return np.einsum("...ia,...i,a->...a", basis, multipliers, lattice.w)


def _central_heat_flux_moment_matrix(lattice: Lattice, u: np.ndarray) -> np.ndarray:
    """Return rows fixing lower f moments plus third-order heat flux.

    Row order is ``1, xi_y, xi_x, xi_y^2, xi_x xi_y, xi_x^2,
    0.5|xi|^2 xi_y, 0.5|xi|^2 xi_x``.  The last two rows correspond to the
    translational central energy flux components, while the preceding rows
    force zero density, momentum and second-order stress increments.
    """

    xi = lattice.c - u[..., None, :]
    xi_x = xi[..., 0]
    xi_y = xi[..., 1]
    xi2 = xi_x * xi_x + xi_y * xi_y
    return np.stack(
        (
            np.ones_like(xi_x),
            xi_y,
            xi_x,
            xi_y * xi_y,
            xi_x * xi_y,
            xi_x * xi_x,
            0.5 * xi2 * xi_y,
            0.5 * xi2 * xi_x,
        ),
        axis=-2,
    )


def _weighted_minimum_norm_heat_flux_delta(
    delta_moments: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    """Project f third-order translational heat flux to populations."""

    if not np.any(delta_moments[..., :6]):
        return _project_zero_lower(delta_moments, lattice, 'heat', 6)
    basis = _central_heat_flux_moment_matrix(lattice, u)
    weighted_basis = basis * lattice.w
    gram = np.einsum("...ia,...ja->...ij", weighted_basis, basis)
    multipliers = np.linalg.solve(gram, delta_moments[..., None])[..., 0]
    return np.einsum("...ia,...i,a->...a", basis, multipliers, lattice.w)


def _central_translational_energy_flux_lu(
    f: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    cx = lattice.c[:, 0] - u[..., 0, None]
    cy = lattice.c[:, 1] - u[..., 1, None]
    xi2 = cx*cx + cy*cy
    return .5 * np.stack((np.einsum('...a,...a,...a->...', f, xi2, cx),
        np.einsum('...a,...a,...a->...', f, xi2, cy)), axis=-1)


def _central_internal_energy_flux_lu(
    g: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    cx = lattice.c[:, 0] - u[..., 0, None]
    cy = lattice.c[:, 1] - u[..., 1, None]
    return np.stack((np.einsum('...a,...a->...', g, cx),
        np.einsum('...a,...a->...', g, cy)), axis=-1)


def _nonequilibrium_central_stress(
    f: np.ndarray,
    f_eq: np.ndarray,
    u: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    cx = lattice.c[:, 0] - u[..., 0, None]
    cy = lattice.c[:, 1] - u[..., 1, None]
    neq = f - f_eq
    xx = np.einsum('...a,...a,...a->...', neq, cx, cx)
    xy = np.einsum('...a,...a,...a->...', neq, cx, cy)
    yy = np.einsum('...a,...a,...a->...', neq, cy, cy)
    return np.stack((np.stack((xx, xy), axis=-1), np.stack((xy, yy), axis=-1)), axis=-2)


def _regularized_f_collision(
    f: np.ndarray,
    f_eq: np.ndarray,
    u: np.ndarray,
    mapping: UnitMapping,
    lattice: Lattice,
    *,
    transport: LocalTransport,
) -> np.ndarray:
    """Relax central f moments through second order.

    Deviatoric stress follows the measured central-moment content scaled by
    ``(1 - 1/tau21)``; the trace follows ``(1 - 1/tau22)``.  Higher-order
    non-equilibrium content is not reconstructed, so the minimum-norm step
    carries only the density, momentum and second-order stress rows.
    """

    stress = _nonequilibrium_central_stress(f, f_eq, u, lattice)
    return _reconstruct_stress(stress, f_eq, u, mapping, lattice, transport)


def _reconstruct_stress(stress, f_eq, u, mapping, lattice, transport):
    """Shared measured-stress reconstruction for populations and nine moments."""
    stress_xy = stress[..., 0, 1]
    stress_dev = stress[..., 0, 0] - stress[..., 1, 1]
    tau21 = transport.tau21
    tau22 = transport.tau22
    shear_factor = 1.0 - 1.0 / tau21
    trace = stress[..., 0, 0] + stress[..., 1, 1]
    # nu_b = 0 对应 tau22 = 1/2，迹保留因子 (1 - 1/tau22) = -1，即零体耗散。
    trace_post = mapping.collision.trace_bulk_scale * (1.0 - 1.0 / tau22) * trace
    dev_post = shear_factor * stress_dev
    xy_post = shear_factor * stress_xy

    delta_xx = 0.5 * (trace_post + dev_post)
    delta_yy = 0.5 * (trace_post - dev_post)
    zeros = np.zeros_like(delta_xx)
    delta_moments_second = np.stack(
        (
            zeros,
            zeros,
            zeros,
            delta_yy,
            xy_post,
            delta_xx,
        ),
        axis=-1,
    )
    return f_eq + _weighted_minimum_norm_second_order_delta(delta_moments_second, u, lattice)


def _regularized_heat_flux_collision(
    f_post: np.ndarray,
    g_eq: np.ndarray,
    f_before: np.ndarray,
    g_before: np.ndarray,
    u: np.ndarray,
    mapping: UnitMapping,
    lattice: Lattice,
    transport: LocalTransport,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply regularized total heat-flux content before energy correction.

    ``f`` carries the third-order central translational energy flux and ``g``
    carries the first-order central internal-energy flux.  The split is stored
    in :mod:`core.unit_mapping` so the collision implementation does not hide a
    transport mapping decision.
    """

    total_heat_flux = central_energy_flux_lu(f_before, g_before, u=u, lattice=lattice)
    return _reconstruct_heat_flux(f_post, g_eq, total_heat_flux, u, mapping, lattice, transport)


def _reconstruct_heat_flux(f_post, g_eq, total_heat_flux, u, mapping, lattice, transport):
    """Shared projection of measured total central energy flux."""
    heat_flux_factor = transport.heat_flux_retention[..., None]
    target_heat_flux = heat_flux_factor * total_heat_flux
    f_fraction = mapping.collision.regularized_heat_flux_f_fraction
    f_heat_flux = f_fraction * target_heat_flux - _central_translational_energy_flux_lu(
        f_post, u, lattice,
    )
    g_heat_flux = (1.0 - f_fraction) * target_heat_flux - _central_internal_energy_flux_lu(
        g_eq, u, lattice,
    )

    zeros = np.zeros_like(f_heat_flux[..., 0])
    f_delta_moments = np.stack(
        (
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            zeros,
            f_heat_flux[..., 1],
            f_heat_flux[..., 0],
        ),
        axis=-1,
    )
    f_with_heat_flux = f_post + _weighted_minimum_norm_heat_flux_delta(f_delta_moments, u, lattice)

    g_delta_moments = np.stack((zeros, g_heat_flux[..., 1], g_heat_flux[..., 0]), axis=-1)
    g_with_heat_flux = g_eq + _weighted_minimum_norm_first_order_delta(g_delta_moments, u, lattice)
    return f_with_heat_flux, g_with_heat_flux


def _correct_f_conserved_moments(
    f_post: np.ndarray,
    f_before: np.ndarray,
    lattice: Lattice,
) -> np.ndarray:
    rho_before = np.sum(f_before, axis=-1)
    momentum_before = np.einsum("...a,ai->...i", f_before, lattice.c)
    return _correct_f_targets(f_post, rho_before, momentum_before, lattice)


def _correct_f_targets(f_post, rho_before, momentum_before, lattice):
    rho_after = np.sum(f_post, axis=-1)
    corrected = f_post + (rho_before - rho_after)[..., None] * lattice.w
    momentum_after = np.einsum("...a,ai->...i", corrected, lattice.c)
    dm = momentum_before - momentum_after
    cx = lattice.c[:, 0]
    cy = lattice.c[:, 1]
    denom_x = float(np.sum(lattice.w * cx * cx))
    denom_y = float(np.sum(lattice.w * cy * cy))
    corrected = corrected + (dm[..., 0, None] * lattice.w * cx / denom_x)
    return corrected + (dm[..., 1, None] * lattice.w * cy / denom_y)


def _reconstruct_collision(macro, stress, q_raw, momentum, f_eq, g_eq, transport, mapping, lattice):
    """Reconstruct one collision from its sufficient measured moments."""
    f_post = _reconstruct_stress(stress, f_eq, macro.u, mapping, lattice, transport)
    f_post = _correct_f_targets(f_post, macro.rho, momentum, lattice)
    f_post, g_shape = _reconstruct_heat_flux(
        f_post, g_eq, q_raw, macro.u, mapping, lattice, transport)
    f_post = _correct_f_targets(f_post, macro.rho, momentum, lattice)
    macro_mid = recover_macro(f_post, g_shape, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    delta_G = macro.E_tot - macro_mid.E_tot
    g_post = g_shape + delta_G[..., None] * (f_eq / macro.rho[..., None])
    return f_post, g_post


def collide_fg(
    f: np.ndarray,
    g: np.ndarray,
    mapping: UnitMapping,
    *,
    lattice: Lattice | None = None,
    return_diagnostics: bool = False,
    backend: str = 'numpy',
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, CollisionDiagnostics]:
    """Collide ``f`` and ``g`` while conserving mass, momentum and total energy.

    Per-cell algorithm:

    1. Recover pre-collision macro state and selected central total energy.
    2. Build ``f_eq`` and ``g_eq`` from the same macro state.
    3. Relax central-Hermite f stress content with ``tau21``/``tau22``.
    4. Reconstruct f third-order central translational energy flux and g
       first-order central internal-energy flux from the local tau32.
    5. Correct the zero moment of g with ``delta_G * f_eq/rho`` so the selected
       central total energy is conserved without adding a comoving heat flux.
    """

    validate_collision_backend(backend)
    lattice = lattice or make_lattice()
    f = np.asarray(f, dtype=float)
    g = np.asarray(g, dtype=float)
    macro_before = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    transport = local_transport_coefficients(mapping, macro_before.rho, macro_before.theta)
    f_eq, g_eq = equilibrium_fg(
        macro_before.rho,
        macro_before.u,
        macro_before.theta,
        mapping.lattice.S,
        lattice,
    )

    if backend == 'numba':
        from core.collision_numba import collide_prepared
        key, weights = tuple(map(tuple, lattice.c)), tuple(lattice.w)
        projectors = tuple(_fixed_zero_lower_projector(key, weights, kind)
                           for kind in ('second', 'heat', 'first'))
        f_post, g_post = collide_prepared(f, g, f_eq, g_eq, macro_before, transport,
                                        mapping, lattice, projectors)
    else:
        stress = _nonequilibrium_central_stress(f, f_eq, macro_before.u, lattice)
        q_raw = central_energy_flux_lu(f, g, u=macro_before.u, lattice=lattice)
        momentum = np.einsum('...a,ai->...i', f, lattice.c)
        f_post, g_post = _reconstruct_collision(
            macro_before, stress, q_raw, momentum, f_eq, g_eq, transport, mapping, lattice)

    if not return_diagnostics:
        return f_post, g_post

    macro_after = recover_macro(f_post, g_post, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    mass_residual = np.sum(f_post - f, axis=-1)
    momentum_residual = np.einsum("...a,ai->...i", f_post - f, lattice.c)
    energy_residual = macro_after.E_tot - macro_before.E_tot
    diagnostics = CollisionDiagnostics(
        mass_residual=mass_residual,
        momentum_residual=momentum_residual,
        energy_residual=energy_residual,
        min_f_post=float(np.min(f_post)),
        min_g_post=float(np.min(g_post)),
        clipping_used=False,
    )
    return f_post, g_post, diagnostics


def assert_collision_conservation(
    f: np.ndarray,
    g: np.ndarray,
    mapping: UnitMapping,
    *,
    tol: float = 1.0e-12,
    lattice: Lattice | None = None,
) -> None:
    _, _, diagnostics = collide_fg(f, g, mapping, lattice=lattice, return_diagnostics=True)
    if np.max(np.abs(diagnostics.mass_residual)) > tol:
        raise AssertionError("collision mass conservation failed")
    if np.max(np.abs(diagnostics.momentum_residual)) > tol:
        raise AssertionError("collision momentum conservation failed")
    if np.max(np.abs(diagnostics.energy_residual)) > tol:
        raise AssertionError(f"{ENERGY_CLOSURE_DEFINITION} conservation failed")
