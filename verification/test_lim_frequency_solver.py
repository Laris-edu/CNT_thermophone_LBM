"""Lim 公共频域入口：驱动符号、双侧因子、声分支与 Rayleigh 外推。"""

import cmath
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.linalg import eig

from reference import lim2013 as lim


def _solve(**overrides):
    inputs = dict(
        f_hz=1.0e4,
        drive_areal_hat=-0.85 / 9.0e-4,
        c_s_J_m2K=1.0e-3,
        area_m2=9.0e-4,
        x_m=0.115,
        beta0_per_side_W_m2K=16.0,
    )
    inputs.update(overrides)
    return lim.solve_lim2013_frequency(**inputs)


def test_near_complex_pressure_retains_eq21_denominator_phase_and_negative_drive():
    """去掉复分母相位、翻转热流符号或漏掉双侧因子都会失败。"""
    w = 2.0 * math.pi * 1.0e4
    gas = lim.TABLE1_AIR
    h = gas.kappa_W_mK * math.sqrt(w / (2.0 * gas.alpha_m2_s))
    D = (16.0 + h) + 1j * (h + w * 1.0e-3 / 2.0)
    expected = (
        0.85 / (2.0 * 9.0e-4) * (gas.gamma - 1.0) / D
        * math.sqrt(w / gas.alpha_m2_s) * gas.kappa_W_mK / gas.c0_m_s
        * cmath.exp(-1j * (w * 0.115 / gas.c0_m_s + 3.0 * math.pi / 4.0))
    )
    result = _solve()
    assert result.p_hat_near_approximate_Pa == pytest.approx(expected, rel=1e-12)
    assert result.p_rms_near_approximate_Pa == pytest.approx(abs(expected) / math.sqrt(2.0))


def test_solver_exact_fields_match_independent_coupled_matrix_eigenproblem():
    """用独立矩阵特征向量及边界线性方程检查膜温、单侧热流、完整压和声分支。"""
    gas = lim.TABLE1_AIR
    w = 2.0 * math.pi * 1.0e4
    matrix = np.array([
        [-gas.rho0_kg_m3 * w**2 / gas.p0_Pa, gas.rho0_kg_m3 * w**2 / gas.T0_K],
        [-1j * w / gas.kappa_W_mK, 1j * w / gas.alpha_m2_s],
    ])
    eigenvalues, vectors = eig(matrix)
    sigma = np.sqrt(eigenvalues)
    # 状态向量为 (p,T)，独立构造 T(0)=1、p'(0)=0 的边界条件。
    coefficients = np.linalg.solve(
        np.array([vectors[1], -sigma * vectors[0]]), [1.0, 0.0]
    )
    q_per_K = gas.kappa_W_mK * np.sum(sigma * vectors[1] * coefficients)
    T_s = (-0.85 / 9.0e-4) / (32.0 + 2.0 * q_per_K + 1j * w * 1.0e-3)
    pressure_modes = vectors[0] * coefficients * T_s * np.exp(-sigma * 0.115)
    result = _solve()
    assert result.T_s_hat_exact_K == pytest.approx(T_s, rel=1e-9)
    assert result.q_one_sided_hat_exact_W_m2 == pytest.approx(q_per_K * T_s, rel=1e-9)
    assert result.p_hat_planar_exact_Pa == pytest.approx(sum(pressure_modes), rel=1e-9)
    assert result.p_hat_acoustic_exact_Pa == pytest.approx(pressure_modes[np.argmin(abs(sigma))], rel=1e-9)


def test_eq24_acoustic_mode_differs_from_full_pressure_inside_thermal_layer():
    """误用完整近壁压力代替 Eq24 的声分支会失败。"""
    gas = lim.TABLE1_AIR
    T_s = 1.0 + 0.2j
    full_wall = lim.exact_pressure_hat(1.0e4, T_s, 0.0)
    acoustic_wall = lim.acoustic_pressure_hat_exact(1.0e4, T_s, 0.0)
    s1, s2, *_ = lim.characteristic_roots(1.0e4, gas)
    # p'(0)=0 给出 p_thermal(0)/p_acoustic(0)=-sigma2/sigma1。
    assert full_wall / acoustic_wall == pytest.approx(1.0 - s2 / s1)
    assert abs(full_wall - acoustic_wall) > 1.0e-3 * abs(acoustic_wall)
    # Eq24 的距离衰减由声分支实部决定，与相位和热分支无关。
    acoustic_far = lim.acoustic_pressure_hat_exact(1.0e4, T_s, 0.115)
    assert abs(acoustic_far / acoustic_wall) == pytest.approx(math.exp(-s2.real * 0.115))
    assert acoustic_far / acoustic_wall == pytest.approx(cmath.exp(-s2 * 0.115))
    result = _solve()
    assert result.p_rms_acoustic_exact_Pa == pytest.approx(abs(result.p_hat_acoustic_exact_Pa) / math.sqrt(2.0))


def test_near_phase_propagates_outward_and_arbitrary_drive_phase_is_preserved():
    f = 1.0e4
    quarter_wavelength = lim.TABLE1_AIR.c0_m_s / (4.0 * f)
    a = lim.near_field_pressure_hat_approximate(
        f, 1000.0, 1.0e-3, 0.0, beta0_per_side_W_m2K=16.0
    )
    b = lim.near_field_pressure_hat_approximate(
        f, 1000.0, 1.0e-3, quarter_wavelength, beta0_per_side_W_m2K=16.0
    )
    assert b / a == pytest.approx(-1j)
    rotated = _solve(drive_areal_hat=-0.85 / 9.0e-4 * 1j)
    base = _solve()
    for field in ("T_s_hat_exact_K", "q_one_sided_hat_exact_W_m2", "p_hat_planar_exact_Pa", "p_hat_near_approximate_Pa"):
        assert getattr(rotated, field) == pytest.approx(1j * getattr(base, field))
    assert rotated.p_rms_far_Pa == pytest.approx(base.p_rms_far_Pa)


def test_rayleigh_distance_and_far_rms_match_at_crossover_then_decay_as_inverse_distance():
    # 用公共入口给出的浮点边界，避免两种等价写法的一个舍入位差。
    R0 = lim.rayleigh_distance_m(1.0e4, 9.0e-4)
    at_R0 = _solve(x_m=R0)
    farther = _solve(x_m=2.0 * R0)
    assert at_R0.rayleigh_distance_m == pytest.approx(0.02593659942363112)
    assert at_R0.in_near_field
    assert not farther.in_near_field
    assert at_R0.p_rms_far_Pa == pytest.approx(at_R0.p_rms_near_approximate_Pa)
    assert farther.p_rms_far_Pa == pytest.approx(at_R0.p_rms_far_Pa / 2.0)
    assert farther.p_rms_near_approximate_Pa == pytest.approx(at_R0.p_rms_near_approximate_Pa)


def test_fixed_total_power_gives_inverse_area_near_pressure_and_area_independent_far_pressure():
    base = _solve()
    larger = _solve(area_m2=1.8e-3, drive_areal_hat=-0.85 / 1.8e-3)
    assert larger.p_rms_near_approximate_Pa == pytest.approx(base.p_rms_near_approximate_Pa / 2.0)
    assert larger.p_rms_far_Pa == pytest.approx(base.p_rms_far_Pa)


def test_custom_gas_and_high_frequency_attenuation_are_not_replaced_by_table_constants():
    gas = replace(lim.TABLE1_AIR, c0_m_s=340.0)
    result = _solve(f_hz=1.0e6, gas=gas)
    assert result.rayleigh_distance_m == pytest.approx(9.0e-4 * 1.0e6 / 340.0)
    assert result.acoustic_decay_exponent > 0.4
    assert result.p_rms_acoustic_exact_Pa / result.p_rms_near_approximate_Pa < 0.7
    assert result.delta_T_m == pytest.approx(math.sqrt(2.0 * gas.alpha_m2_s / (2.0 * math.pi * 1.0e6)))


def test_zero_drive_returns_zero_response_without_relative_error_division():
    result = _solve(drive_areal_hat=0.0)
    assert result.T_s_hat_exact_K == 0.0
    assert result.q_one_sided_hat_exact_W_m2 == 0.0
    assert result.p_rms_planar_exact_Pa == 0.0
    assert result.p_rms_near_approximate_Pa == 0.0
    assert result.p_rms_far_Pa == 0.0


@pytest.mark.parametrize("overrides", [
    {"f_hz": 0.0}, {"f_hz": -1.0}, {"f_hz": math.nan},
    {"drive_areal_hat": complex(1.0, math.inf)},
    {"area_m2": 0.0}, {"area_m2": math.nan},
    {"x_m": 0.0}, {"x_m": -1.0}, {"x_m": math.inf},
    {"c_s_J_m2K": 0.0}, {"c_s_J_m2K": math.nan},
    {"beta0_per_side_W_m2K": -1.0}, {"beta0_per_side_W_m2K": math.nan},
])
def test_frequency_solver_rejects_invalid_physical_inputs(overrides):
    with pytest.raises(ValueError):
        _solve(**overrides)
