"""平面单侧热声验证的链接边界，物理 z 对应数组 y。

底部是零净质量流的全容纳漫反射热壁（壁面位于首格中心下方 dx/2）；
有限网格上的气体首格温度不是壁温。顶部采用线性声学特征量投影，保留
熵扰动及非平衡分布；不是完整非线性、多维 LODI。所有越界链接均显式填入。
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro


@lru_cache(maxsize=16)
def _crossing_links(c_values, top, incoming):
    """Explicit (row, velocity) pairs, including every multispeed link."""
    rows, directions, kinetic = [], [], []
    for a, (cx, cy) in enumerate(c_values):
        cy = int(cy)
        selected = (cy < 0 if top else cy > 0) if incoming else (cy > 0 if top else cy < 0)
        if selected:
            for row in (range(-abs(cy), 0) if top else range(abs(cy))):
                rows.append(row)
                directions.append(a)
                kinetic.append(.5 * (cx*cx + cy*cy))
    arrays = (np.asarray(rows), np.asarray(directions), np.asarray(kinetic)[:, None])
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _link_totals(f_links, g_links, kinetic):
    nx = f_links.shape[1]
    return float(np.sum(f_links) / nx), float(np.sum(kinetic*f_links + g_links) / nx)


def crossing_totals(f, g, lattice, *, top, incoming=False):
    """实际跨边界布居的质量与总能量，每单位横向格数；非连续矩近似。"""
    rows, directions, kinetic = _crossing_links(tuple(map(tuple, lattice.c)), top, incoming)
    return _link_totals(f[rows, :, directions], g[rows, :, directions], kinetic)


class PreparedDiffuseWall:
    """One collided state's fixed outgoing links and temperature-dependent inflow.

    Trial energies come from the same discrete incoming populations later written
    to the streamed state. No conductive moment or analytic admittance is used.
    """

    def __init__(self, f_post, g_post, mapping, lattice):
        self.mapping, self.lattice = mapping, lattice
        key = tuple(map(tuple, lattice.c))
        rows, directions, kinetic = _crossing_links(key, False, False)
        outgoing_f, outgoing_g = f_post[rows, :, directions], g_post[rows, :, directions]
        self.outgoing_mass = np.sum(outgoing_f, axis=0)
        self.outgoing_energy = _link_totals(outgoing_f, outgoing_g, kinetic)[1]
        self.rows, self.directions, self.kinetic = _crossing_links(key, False, True)
        self._temperature = None

    def _incoming(self, temperature_K):
        if not np.isfinite(temperature_K) or temperature_K <= 0:
            raise ValueError("wall temperature must be finite and positive")
        if temperature_K != self._temperature:
            nx = self.outgoing_mass.size
            fw, gw = equilibrium_fg(np.ones(nx), np.zeros((nx, 2)),
                np.full(nx, temperature_K / self.mapping.temperature_scale),
                self.mapping.lattice.S, self.lattice)
            f_links, g_links = fw[:, self.directions].T, gw[:, self.directions].T
            rho_wall = self.outgoing_mass / np.sum(f_links, axis=0)
            self._f, self._g = f_links * rho_wall, g_links * rho_wall
            self._temperature = temperature_K
        return self._f, self._g

    def energy_exchange_lu(self, temperature_K):
        f_links, g_links = self._incoming(temperature_K)
        return _link_totals(f_links, g_links, self.kinetic)[1] - self.outgoing_energy

    def write(self, temperature_K, f_new, g_new):
        f_links, g_links = self._incoming(temperature_K)
        f_new[self.rows, :, self.directions] = f_links
        g_new[self.rows, :, self.directions] = g_links


def diffuse_temperature_wall(f_post, g_post, f_new, g_new, mapping, lattice, temperature_K):
    """全容纳热壁；壁密度由离散出入质量相等确定，绝不固定壁压。"""
    PreparedDiffuseWall(f_post, g_post, mapping, lattice).write(temperature_K, f_new, g_new)


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
