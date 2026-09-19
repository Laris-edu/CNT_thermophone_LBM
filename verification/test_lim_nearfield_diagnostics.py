"""守护 Lim 对照的 EOS、区间热流相位及压力相位符号。"""
import numpy as np
import pytest

from config_io import load_config
from core.solver import GasSolver2D
from reference.lim2013 import characteristic_roots, conductive_heat_flux_hat
from simulation.study import BASE
from postproc.diagnostics import fit_window
from reference.lim2013 import matched_gas


def test_reference_uses_actual_lbm_equilibrium_pressure():
    solver = GasSolver2D(load_config(BASE))
    gas = matched_gas(solver.mapping)
    actual = solver.get_macro().p[0,0]*solver.mapping.pressure_scale
    assert gas.p0_Pa == pytest.approx(actual, rel=1e-12)
    assert gas.c0_m_s**2 == pytest.approx(gas.gamma*gas.p0_Pa/gas.rho0_kg_m3, rel=1e-12)
    assert gas.cp_J_kgK == pytest.approx(
        gas.gamma*gas.p0_Pa/((gas.gamma-1)*gas.rho0_kg_m3*gas.T0_K), rel=1e-12)


def test_interval_average_flux_fit_has_midpoint_phase():
    frequency = 5e4
    omega = 2*np.pi*frequency
    t = np.arange(3*32+1)/(32*frequency)
    midpoint = .5*(t[1:]+t[:-1])
    target = 4*np.exp(.4j)
    # 独立解析积分给出每个区间的热流均值；另加常量与线性漂移。
    means = np.real(target*(np.exp(1j*omega*t[1:])-np.exp(1j*omega*t[:-1]))/(1j*omega*np.diff(t)))
    means += 7+1e3*midpoint
    got, _ = fit_window(midpoint, means, frequency, 2, 3)
    assert got == pytest.approx(target*np.sinc(1/32), rel=1e-12)
    wrong, _ = fit_window(t[1:], means, frequency, 2, 3)
    assert abs(np.angle(wrong/target, deg=True)) > 5


def test_outgoing_exact_branch_has_positive_heat_flux_pressure_sign():
    gas = matched_gas(GasSolver2D(load_config(BASE)).mapping)
    frequency = 5e4
    s1,s2,d1,d2 = characteristic_roots(frequency,gas)
    outgoing_per_K = d2*d1*s1/(d1*s1-d2*s2)
    q_per_K = conductive_heat_flux_hat(frequency,1,gas=gas)
    # 从原文 (9)--(12) 直接构造出射系数，检查 (19) 印刷负号的 180° 陷阱。
    ratio = outgoing_per_K/(q_per_K*(gas.gamma-1)/gas.c0_m_s)
    assert abs(ratio-1) < 1e-3
