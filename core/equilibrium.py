"""以格子求积权重构造有限阶 Hermite 平衡态，不做无权配矩或正性裁剪。

f 使用四阶展开（需要 D2Q37 的八阶求积），g 默认使用二阶展开。
参照：Scagliarini et al., Phys. Fluids 22, 055101 (2010), Appendix A,
https://arxiv.org/abs/1005.3639 。热力学 theta 与固定的 theta_q 分别保留。
"""

from __future__ import annotations

import numpy as np

from core.lattice import Lattice, make_lattice


def monomial_exponents(max_order: int) -> list[tuple[int, int]]:
    """返回总次数不超过 ``max_order`` 的二维单项指数。"""

    return [(m, total - m) for total in range(max_order + 1) for m in range(total + 1)]


class EquilibriumDomainError(ValueError):
    """输入不属于当前截断平衡态可接受的温度/速度范围。"""


def _quadrature_equilibrium(rho_like, u, theta, order, lattice, *, check_positive=True):
    """Hermite 张量收缩的旋转不变量形式；系数由 Maxwellian 解析给出。"""
    if order not in (2, 3, 4):
        raise ValueError('equilibrium order must be 2, 3, or 4')
    if lattice.d != 2 or lattice.q != 37:
        raise ValueError('unsupported equilibrium quadrature')
    rho_like=np.asarray(rho_like,dtype=float)
    theta=np.asarray(theta,dtype=float)
    u=np.asarray(u,dtype=float)
    if u.ndim == 0 or u.shape[-1] != lattice.d:
        raise ValueError('equilibrium velocity must have a final axis of length 2')
    if (not np.all(np.isfinite(rho_like)) or not np.all(np.isfinite(theta))
            or not np.all(np.isfinite(u))):
        raise EquilibriumDomainError('equilibrium inputs must be finite')
    if np.any(rho_like < 0) or np.any(theta <= 0):
        raise EquilibriumDomainError('equilibrium density must be nonnegative and temperature positive')
    tq=float(lattice.theta_q)
    if not np.isfinite(tq) or tq <= 0:
        raise ValueError('quadrature temperature must be positive and finite')
    # dot = (c.u)/theta_q, u2 = |u|²/theta_q, c2 = |c|²/theta_q.
    dot=np.einsum('...i,ai->...a',u,lattice.c)/tq
    u2=np.sum(u*u,axis=-1)[...,None]/tq
    c2=np.sum(lattice.c*lattice.c,axis=-1)/tq
    delta=(theta/tq-1)[...,None]
    d=lattice.d
    dot2=dot*dot
    polynomial=1+dot+.5*(dot2-u2+delta*(c2-d))
    if order >= 3:
        polynomial += dot*(dot2-3*u2+3*delta*(c2-d-2))/6
    if order == 4:
        polynomial += (dot2*dot2-6*dot2*u2+3*u2*u2
            +6*delta*(dot2*(c2-d-4)-u2*(c2-d-2))
            +3*delta*delta*(c2*c2-2*(d+2)*c2+d*(d+2)))/24
    populations=rho_like[...,None]*lattice.w*polynomial
    if not np.all(np.isfinite(populations)):
        raise EquilibriumDomainError('equilibrium expansion produced non-finite populations')
    if check_positive and np.any(populations < 0):
        raise EquilibriumDomainError(
            f'equilibrium H{order} is not nonnegative: min={float(np.min(populations)):.6g}, '
            f'theta/theta_q=[{float(np.min(theta)/tq):.6g}, {float(np.max(theta)/tq):.6g}]. '
            'State is outside this truncated equilibrium positivity domain; check unit mapping/temperature/velocity. '
            'Populations are not clipped.')
    return populations


def feq_hermite4(
    rho: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    lattice: Lattice | None = None,
    *,
    check_positive: bool = True,
) -> np.ndarray:
    """标准求积权重的四阶 Hermite 平衡态；默认检查逐布居非负性。

    ``theta`` is the thermodynamic lattice temperature.  The lattice
    quadrature temperature remains available from ``lattice.theta_q`` and is
    not used as a thermodynamic substitute.
    """

    lattice = lattice or make_lattice()
    return _quadrature_equilibrium(rho,u,theta,4,lattice,check_positive=check_positive)


def geq_polyatomic(
    rho: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    S: float,
    lattice: Lattice | None = None,
    order: int = 2,
    *,
    check_positive: bool = True,
) -> np.ndarray:
    """Polyatomic internal-energy equilibrium for ``g``.

    The zero moment is ``(S/2) rho theta`` and the default moment recovery is
    second order, as required by the Phase 2 contract.
    """

    if order < 2 or order > 4:
        raise ValueError("g equilibrium order must be 2, 3, or 4")
    if not np.isfinite(S) or S < 0:
        raise ValueError('internal degrees S must be finite and nonnegative')
    lattice = lattice or make_lattice()
    rho=np.asarray(rho,dtype=float)
    if not np.all(np.isfinite(rho)) or np.any(rho < 0):
        raise EquilibriumDomainError('equilibrium density must be finite and nonnegative')
    e_int_extra = 0.5 * float(S) * rho * np.asarray(theta, dtype=float)
    return _quadrature_equilibrium(e_int_extra,u,theta,order,lattice,check_positive=check_positive)


def equilibrium_fg(
    rho: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    S: float,
    lattice: Lattice | None = None,
    *,
    check_positive: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    lattice = lattice or make_lattice()
    return (feq_hermite4(rho,u,theta,lattice,check_positive=check_positive),
            geq_polyatomic(rho,u,theta,S,lattice,check_positive=check_positive))


def raw_moments(distribution: np.ndarray, lattice: Lattice | None = None, max_order: int = 4) -> dict[tuple[int, int], np.ndarray]:
    lattice = lattice or make_lattice()
    f = np.asarray(distribution, dtype=float)
    cx = lattice.c[:, 0]
    cy = lattice.c[:, 1]
    out: dict[tuple[int, int], np.ndarray] = {}
    for m, n in monomial_exponents(max_order):
        out[(m, n)] = np.sum(f * (cx**m * cy**n), axis=-1)
    return out
