"""平面单侧热声验证的链接边界，物理 z 对应数组 y。

底部是零净质量流的全容纳漫反射热壁（壁面位于首格中心下方 dx/2）；
有限网格上的气体首格温度不是壁温。顶部采用线性声学特征量投影，保留
熵扰动及非平衡分布；不是完整非线性、多维 LODI。所有越界链接均显式填入。
"""
from __future__ import annotations

import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro


def crossing_totals(f, g, lattice, *, top, incoming=False):
    """实际跨边界布居的质量与总能量，每单位横向格数；非连续矩近似。"""
    mass = energy = 0.0
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        selected = (cy < 0 if top else cy > 0) if incoming else (cy > 0 if top else cy < 0)
        if selected:
            rows = slice(-abs(cy), None) if top else slice(0, abs(cy))
            fa, ga = f[rows, :, a], g[rows, :, a]
            mass += float(np.sum(fa)) / f.shape[1]
            energy += float(np.sum(.5 * np.dot(lattice.c[a], lattice.c[a]) * fa + ga)) / f.shape[1]
    return mass, energy


def diffuse_temperature_wall(f_post, g_post, f_new, g_new, mapping, lattice, temperature_K):
    """全容纳热壁；壁密度由离散出入质量相等确定，绝不固定壁压。"""
    if not np.isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("wall temperature must be finite and positive")
    fw, gw = equilibrium_fg(np.ones(f_new.shape[1]), np.zeros((f_new.shape[1], 2)),
        np.full(f_new.shape[1], temperature_K / mapping.temperature_scale), mapping.lattice.S, lattice)
    outgoing = np.zeros(f_new.shape[1])
    denominator = np.zeros_like(outgoing)
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        if cy < 0:
            outgoing += np.sum(f_post[:-cy, :, a], axis=0)
        elif cy > 0:
            denominator += cy * fw[:, a]
    rho_wall = outgoing / denominator
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        if cy > 0:
            f_new[:cy, :, a] = rho_wall * fw[:, a]
            g_new[:cy, :, a] = rho_wall * gw[:, a]


def characteristic_top(f_post, g_post, f_new, g_new, mapping, lattice):
    """静止均匀背景的小扰动上出口：W-=p'-rho0*c0*u_y=0。

    W+ 和熵模态 rho'-p'/c0² 从最上方气体格读出；保持其非平衡分量。
    入射平衡态随声波同时改变 rho、u、T，不将温度或总压钉死。
    适用范围是正入射平面小振幅波，反射率必须由独立脉冲测量。
    """
    m = recover_macro(f_post[-1], g_post[-1], D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    rho0, theta0, c0 = mapping.lattice.rho_ref_lu, mapping.theta_ref_lu, mapping.c0_lu
    dp = m.p - rho0 * theta0
    outgoing = dp + rho0 * c0 * m.u[..., 1]
    entropy = m.rho - rho0 - dp / c0**2
    p_target = rho0 * theta0 + .5 * outgoing
    rho_target = rho0 + entropy + .5 * outgoing / c0**2
    u_target = m.u.copy()
    u_target[..., 1] = .5 * outgoing / (rho0 * c0)
    fe, ge = equilibrium_fg(m.rho, m.u, m.theta, mapping.lattice.S, lattice)
    ft, gt = equilibrium_fg(rho_target, u_target, p_target/rho_target, mapping.lattice.S, lattice)
    ft += f_post[-1] - fe
    gt += g_post[-1] - ge
    for a, (_, cy) in enumerate(lattice.c.astype(int)):
        if cy < 0:
            f_new[cy:, :, a] = ft[:, a]
            g_new[cy:, :, a] = gt[:, a]
