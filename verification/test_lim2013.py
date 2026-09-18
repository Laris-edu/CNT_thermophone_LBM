"""Lim 2013 参照模型的回归：精确解、近似解的适用范围，以及与仓库链路的恒等关系。

只测解析参照，不跑 LBM（LBM 对照见 ``scripts/verify_lim2013.py --mode lbm``）。
断言都带独立含义：特征根代回四次方程、声分支趋于 Omega/c0、精确解与紧致热层恒等式
互证、Rayleigh 距离复现论文自述数字、以及本仓链路与论文 Eq. (22) 的比值等于一个
闭式常数（这是两条独立实现之间的定量绑定，见下）。

参照：Lim, Tong & Li, JSV 332 (2013) 6329-6342。
"""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from reference.analytical_models import pressure_profile_from_exponential_temperature
from reference.constants import PhysicalParams
from reference.film_ode import level_c_closed_form
from reference.lim2013 import (
    TABLE1_AIR,
    areal_drive_from_power,
    characteristic_roots,
    conductive_heat_flux_hat,
    exact_pressure_hat,
    exact_thermal_admittance,
    far_field_pressure_rms,
    film_temperature_hat,
    near_field_pressure_hat_from_flux,
    near_field_pressure_rms_approximate,
    near_field_pressure_rms_exact,
    rayleigh_distance_m,
    with_alpha_from_kappa,
)

# 论文 §4.1 的算例。
AREA_M2 = 9.0e-4
POWER_W = 0.85
BETA0 = 16.0
C_S = 1.0e-3
PROBE_X_M = 0.115


def _paper_case(f_hz: float, gas=TABLE1_AIR, beta0: float = BETA0) -> complex:
    """论文算例的膜温复幅（面功率口径，含 Eq. (1) 的负号）。"""

    return film_temperature_hat(
        f_hz, -areal_drive_from_power(POWER_W, AREA_M2), C_S,
        beta0_per_side_W_m2K=beta0, gas=gas,
    )


@pytest.mark.parametrize("f_hz", [10.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6])
def test_characteristic_roots_solve_the_quartic_and_acoustic_branch_is_omega_over_c0(f_hz):
    """sigma1^2/sigma2^2 必须满足 Eq. (10) 的四次方程，且声分支趋于 j*Omega/c0。"""

    gas = TABLE1_AIR
    sigma1, sigma2, d1, d2 = characteristic_roots(f_hz, gas)
    w = 2.0 * math.pi * f_hz
    a = 1j * w / gas.alpha_m2_s
    b = -1j * w / gas.kappa_W_mK
    c = gas.rho0_kg_m3 * w * w / gas.T0_K
    d = -gas.rho0_kg_m3 * w * w / gas.p0_Pa
    for sigma in (sigma1, sigma2):
        residual = sigma**4 - (a + d) * sigma**2 + (a * d - b * c)
        assert abs(residual) < 1e-6 * max(abs(a * d), abs(b * c), 1.0)
        assert sigma.real > 0.0
    # d_i = (sigma_i^2 - a)/b，即 Eq. (9) 的压力-温度比例系数。
    assert d1 == pytest.approx((sigma1**2 - a) / b, rel=1e-12)
    assert d2 == pytest.approx((sigma2**2 - a) / b, rel=1e-12)
    # 声分支：sigma2 -> j*Omega/c0（论文 Eq. A.5）。低频端偏差来自耦合，容许 2%。
    assert abs(sigma2 / (1j * w / gas.c0_m_s) - 1.0) < 0.02
    # 热分支量级：|sigma1| ~ sqrt(Omega/alpha)。
    assert abs(sigma1) / math.sqrt(w / gas.alpha_m2_s) == pytest.approx(1.0, rel=0.05)


def test_exact_admittance_approaches_sqrt_jOmega_over_alpha():
    """论文 Eq. (18) 的 M ~= sqrt(j Omega/alpha) 在三档频率上的偏离上限。"""

    gas = TABLE1_AIR
    bounds = {10.0: 1.0e-4, 1.0e4: 5.0e-3, 1.0e6: 2.0e-2}
    for f_hz, bound in bounds.items():
        w = 2.0 * math.pi * f_hz
        approx = cmath.sqrt(1j * w / gas.alpha_m2_s)
        deviation = abs(exact_thermal_admittance(f_hz, gas) / approx - 1.0)
        assert deviation < bound, (f_hz, deviation)


def test_eq22_agrees_with_the_exact_solution_over_the_paper_range_and_fails_above_1MHz():
    """论文 §4.4 的适用范围：Eq. (22) 在 10 Hz-100 kHz 内有效，1 MHz 以上失效。"""

    gas = TABLE1_AIR
    worst = 0.0
    for f_hz in np.logspace(1.0, 5.0, 25):
        approx = near_field_pressure_rms_approximate(
            float(f_hz), POWER_W, AREA_M2, C_S, beta0_per_side_W_m2K=BETA0, gas=gas)
        exact = near_field_pressure_rms_exact(
            float(f_hz), _paper_case(float(f_hz)), x_m=PROBE_X_M, gas=gas)
        worst = max(worst, abs(exact / approx - 1.0))
    assert worst < 0.015, worst  # 实测 1.04%

    approx_1mhz = near_field_pressure_rms_approximate(
        1.0e6, POWER_W, AREA_M2, C_S, beta0_per_side_W_m2K=BETA0, gas=gas)
    exact_1mhz = near_field_pressure_rms_exact(1.0e6, _paper_case(1.0e6), x_m=PROBE_X_M, gas=gas)
    assert exact_1mhz / approx_1mhz < 0.7  # 实测 0.62，论文 Fig. 6(b) 的尖锐下降


@pytest.mark.parametrize(
    "f_hz,phase_bound_deg",
    [(1.0e3, 1.5), (1.0e4, 3.5), (1.0e5, 10.0)],
)
def test_compact_layer_identity_reproduces_the_full_exact_solution(f_hz, phase_bound_deg):
    """(gamma-1)Q0/c0（Eq. 19/A.14）必须等于 Eq. (9) 的远场极限。

    幅值偏差恒为 1.03%（与频率无关，实测 |ident|/|full| = 0.989684），相位偏差则随频率
    增长（1 kHz 0.87deg -> 100 kHz 8.68deg），后者就是 Eq. (22) 近似本身的误差，故按频率
    分别设界。这个恒等式是"用壁面热流换声压"的全部依据。
    """

    gas = TABLE1_AIR
    T_s = _paper_case(f_hz)
    q = conductive_heat_flux_hat(f_hz, T_s, gas=gas)
    identity = near_field_pressure_hat_from_flux(q, gas)
    delta_T = math.sqrt(2.0 * gas.alpha_m2_s / (2.0 * math.pi * f_hz))
    full = exact_pressure_hat(f_hz, T_s, 10.0 * delta_T, gas=gas)
    assert abs(identity) / abs(full) == pytest.approx(0.9897, rel=0.005)
    assert abs(math.degrees(float(np.angle(identity / full)))) < phase_bound_deg


def test_rayleigh_distance_reproduces_the_paper_numbers():
    """R0 = s/lambda 必须复现论文自述的 44 kHz、0.04 m、0.3 m 三处数字。"""

    # "for s = 3x3 cm^2 and at 11.5 cm, plane wave for f > 44 kHz"
    crossing = PAPER_AREA = 9.0e-4
    f_cross = 0.115 * TABLE1_AIR.c0_m_s / crossing
    assert f_cross == pytest.approx(44.3e3, rel=0.01)
    # "for 10 cm^2, 100 kHz -> 0.3 m ; 10 kHz -> 0.04 m"
    assert rayleigh_distance_m(1.0e5, 1.0e-3, TABLE1_AIR) == pytest.approx(0.3, rel=0.05)
    assert rayleigh_distance_m(1.0e4, 1.0e-3, TABLE1_AIR) == pytest.approx(0.03, rel=0.35)
    # 近场值乘 R0/x 即远场值（Eq. 23），在 x = R0 处两者相等。
    near = near_field_pressure_rms_approximate(
        1.0e4, POWER_W, AREA_M2, C_S, beta0_per_side_W_m2K=BETA0, gas=TABLE1_AIR)
    R0 = rayleigh_distance_m(1.0e4, AREA_M2, TABLE1_AIR)
    far_at_R0 = far_field_pressure_rms(
        1.0e4, POWER_W, AREA_M2, C_S, R0, beta0_per_side_W_m2K=BETA0, gas=TABLE1_AIR)
    assert far_at_R0 == pytest.approx(near, rel=1e-12)
    assert far_field_pressure_rms(
        1.0e4, POWER_W, AREA_M2, C_S, 2.0 * R0,
        beta0_per_side_W_m2K=BETA0, gas=TABLE1_AIR) == pytest.approx(near / 2.0, rel=1e-12)


@pytest.mark.parametrize("f_hz", [1.0e3, 1.0e4, 5.0e4, 1.0e5])
def test_repo_chain_over_lim_eq22_is_the_closed_form_eos_ratio(f_hz):
    """本仓链路 / 论文 Eq. (22) = gamma*P0 / [(gamma-1)*rho0*cp*T0]，与频率无关。

    两条实现独立（一条走 p'' 方程的传播解，一条走 Lim 的 sigma/d 代数解），
    比值恒定说明差异只是常数口径，不是随频率变化的物理分歧。
    """

    gas = TABLE1_AIR
    params = PhysicalParams(
        rho0=gas.rho0_kg_m3, cp=gas.cp_J_kgK, kg=gas.kappa_W_mK, c0=gas.c0_m_s,
        gamma=gas.gamma, p0=gas.p0_Pa, T0=gas.T0_K, beta0=BETA0,
    )
    drive = areal_drive_from_power(POWER_W, AREA_M2)
    closed = level_c_closed_form(f_hz, complex(drive), C_A=C_S, params=params)
    p_hat, _ = pressure_profile_from_exponential_temperature(
        f_hz, closed.T_s_hat, np.array([PROBE_X_M]), params=params)
    repo = abs(p_hat[0]) / math.sqrt(2.0)
    lim = near_field_pressure_rms_approximate(
        f_hz, POWER_W, AREA_M2, C_S, beta0_per_side_W_m2K=BETA0, gas=gas)
    expected = gas.gamma * gas.p0_Pa / ((gas.gamma - 1.0) * gas.rho0_kg_m3 * gas.cp_J_kgK * gas.T0_K)
    assert repo / lim == pytest.approx(expected, rel=1.0e-3)  # 实测 1.012995 vs 1.01302


def test_heat_capacity_crossover_is_kappa_sqrt_2_over_alpha_omega():
    """论文 Fig. 4 的形状：c_s 低于交叉点平坦，高于交叉点近似按 1/c_s 下降。

    交叉点由 omega*c_s/2 = kappa*sqrt(omega/2alpha) 给出，即 c_s* = kappa*sqrt(2/(alpha*omega))。
    （论文 §4.3 的文字说高 c_s 趋于常数，与其 Fig. 4 相反；此处按公式与图断言。）
    """

    gas = TABLE1_AIR
    f_hz = 1.0e4
    w = 2.0 * math.pi * f_hz
    c_star = gas.kappa_W_mK * math.sqrt(2.0 / (gas.alpha_m2_s * w))
    assert c_star == pytest.approx(0.03116, rel=0.02)

    def pressure(c_s: float) -> float:
        return near_field_pressure_rms_approximate(
            f_hz, 1.0, 1.0e-3, c_s, beta0_per_side_W_m2K=BETA0, gas=gas)

    # 交叉点以下平坦：c_s = 0.1*c_s* 相对 c_s -> 0 的极限只降 5% 以内（实测 4.8%）。
    assert pressure(0.1 * c_star) / pressure(1.0e-6 * c_star) == pytest.approx(1.0, rel=0.05)
    # 充分高于交叉点后进入纯 1/c_s 律：10*c_s* -> 100*c_s* 的落差落在 15% 以内（实测 0.110）。
    assert pressure(100.0 * c_star) / pressure(10.0 * c_star) == pytest.approx(0.1, rel=0.15)
    # 但 c_s* -> 10*c_s* 这一段仍被 beta0 与 kappa 项托住，落差明显小于十倍（实测 0.203）。
    assert pressure(10.0 * c_star) / pressure(c_star) > 0.15


def test_lim_parameter_mapping_rejects_invalid_inputs():
    """负/零输入必须显式报错，不静默返回。"""

    from reference.lim2013 import LimGas

    with pytest.raises(ValueError):
        areal_drive_from_power(1.0, 0.0)
    with pytest.raises(ValueError):
        rayleigh_distance_m(1.0e4, -1.0)
    with pytest.raises(ValueError):
        film_temperature_hat(1.0e4, 1.0 + 0j, 0.0, beta0_per_side_W_m2K=BETA0)
    with pytest.raises(ValueError):
        film_temperature_hat(1.0e4, 1.0 + 0j, C_S, beta0_per_side_W_m2K=-1.0)
    with pytest.raises(ValueError):
        characteristic_roots(0.0)
    with pytest.raises(ValueError):
        exact_pressure_hat(1.0e4, 1.0 + 0j, -1.0)
    with pytest.raises(ValueError):
        far_field_pressure_rms(1.0e4, POWER_W, AREA_M2, C_S, 0.0, beta0_per_side_W_m2K=BETA0)
    with pytest.raises(ValueError):
        LimGas(rho0_kg_m3=-1.0, cp_J_kgK=1006.0, kappa_W_mK=0.0262, c0_m_s=347.0,
               alpha_m2_s=2.25e-5, gamma=1.4)
    with pytest.raises(ValueError):
        LimGas(rho0_kg_m3=1.16, cp_J_kgK=1006.0, kappa_W_mK=0.0262, c0_m_s=347.0,
               alpha_m2_s=2.25e-5, gamma=1.0)


def test_alpha_consistency_helper_removes_the_table1_gap():
    """Table 1 的 alpha 与 kappa/(rho0*cp) 差 0.2%；辅助函数必须消掉这个差。"""

    gas = TABLE1_AIR
    assert gas.alpha_m2_s == pytest.approx(2.25e-5)
    assert gas.alpha_consistent_m2_s == pytest.approx(2.2451e-5, rel=1e-4)
    assert with_alpha_from_kappa(gas).alpha_m2_s == gas.alpha_consistent_m2_s
