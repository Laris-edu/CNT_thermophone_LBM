"""碰撞核的符号约定与守恒回归：固定实测偏应力口径，并给守恒入口首个调用者。

本文件两条检查各自独立取证，不依赖被测实现的中间量：

* 实测偏应力必须满足 Navier–Stokes 符号（Π^neq 与应变率反号），且逐格一致；
* 带非平凡扰动的 local 态上，碰撞必须逐格守恒质量、动量和所选总能量。

已退役的解析应变率分支（符号与实测口径相反、等效负黏性）不在本仓，
其证据见旧仓 docs/CORE_REVIEW_20260917.md。
"""
from copy import deepcopy
from pathlib import Path

import numpy as np

from config_io import load_config
from core.collision_smrt import assert_collision_conservation, collide_fg
from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from core.solver import GasSolver2D

ROOT = Path(__file__).resolve().parents[1]
ANALYTIC = ROOT / 'configs/gas_air_10k_d2q37_qmatched_analytic_stress.yaml'
LOCAL = ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


def _central_deviatoric(f, g, mapping, lattice):
    """Return ``(Pi_xx - Pi_yy, Pi_xy, du_x/dx - du_y/dy, du_x/dy + du_y/dx)``."""

    macro = recover_macro(f, g, D=mapping.lattice.D, S=mapping.lattice.S, lattice=lattice)
    xi = lattice.c - macro.u[..., None, :]
    neq = f - equilibrium_fg(macro.rho, macro.u, macro.theta, mapping.lattice.S, lattice)[0]
    xx = np.einsum('...a,...a,...a->...', neq, xi[..., 0], xi[..., 0])
    yy = np.einsum('...a,...a,...a->...', neq, xi[..., 1], xi[..., 1])
    xy = np.einsum('...a,...a,...a->...', neq, xi[..., 0], xi[..., 1])
    u = macro.u
    dux_dy = 0.5 * (np.roll(u[..., 0], -1, axis=0) - np.roll(u[..., 0], 1, axis=0))
    duy_dx = 0.5 * (np.roll(u[..., 1], -1, axis=1) - np.roll(u[..., 1], 1, axis=1))
    return xx - yy, xy, dux_dy + duy_dx


def _shear_state(solver, amplitude, mode_index=1):
    """Initialize a transverse shear wave with variation along y."""

    y = np.arange(solver.ny, dtype=float)
    phase = 2.0 * np.pi * mode_index * y / solver.ny
    u = np.zeros((solver.ny, solver.nx, 2), dtype=float)
    u[..., 0] = amplitude * np.sin(phase)[:, None]
    solver.initialize_from_macro(np.ones((solver.ny, solver.nx)), u, solver.mapping.theta_ref_lu)


def test_measured_deviatoric_stress_keeps_the_navier_stokes_sign():
    """实测口径：Π_xy^neq 必须与剪切率反号，且比值在格点间一致。

    f 平衡态与流步给出非平衡应力的实际符号；如果平衡态或松弛符号被改动，
    这里会先于动态回归失败，并直接指出是符号约定变了。
    """

    config = deepcopy(load_config(ANALYTIC))
    config['numerics'] = {'nx': 4, 'ny': 32}
    solver = GasSolver2D(config)
    _shear_state(solver, amplitude=1.0e-6)
    solver.step(60)

    _dev, xy, shear_rate = _central_deviatoric(
        solver.f, solver.g, solver.mapping, solver.lattice
    )
    mask = np.abs(shear_rate) > 1.0e-10
    assert np.count_nonzero(mask) > 16
    ratio = xy[mask] / shear_rate[mask]
    assert np.median(ratio) < 0.0, '实测非平衡剪切应力必须与应变率反号'
    spread = (np.max(ratio) - np.min(ratio)) / abs(np.median(ratio))
    assert spread < 1.0e-3, '同一构型内比值应逐格一致，出现各向异性说明闭合被改坏'


def test_collision_conserves_mass_momentum_and_energy_on_perturbed_local_state():
    """非平凡扰动下的逐格守恒；调用 core 自带的守恒断言入口。"""

    config = load_config(LOCAL)
    config['numerics'].update(nx=32, ny=32)
    solver = GasSolver2D(config)
    lattice = solver.lattice
    x = np.arange(solver.nx) / solver.nx
    y = np.arange(solver.ny) / solver.ny
    grid_x, grid_y = np.meshgrid(x, y)
    rho = 1.0 + 1.0e-3 * np.sin(2.0 * np.pi * grid_x)
    theta = solver.mapping.theta_ref_lu * (1.0 + 2.0e-3 * np.cos(2.0 * np.pi * grid_y))
    u = np.stack(
        (
            3.0e-3 * np.sin(2.0 * np.pi * grid_y),
            2.0e-3 * np.sin(2.0 * np.pi * grid_x),
        ),
        axis=-1,
    )
    f, g = equilibrium_fg(rho, u, theta, solver.mapping.lattice.S, lattice)

    assert_collision_conservation(f, g, solver.mapping, tol=1.0e-13, lattice=lattice)
    f_post, g_post = collide_fg(f, g, solver.mapping, lattice=lattice)
    assert np.min(f_post) > 0.0 and np.min(g_post) > 0.0
