"""捕获膜气能量不闭合与热声参照漏掉gamma的回归；显式local基线。"""
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from core.solver import GasSolver2D
from coupling.conjugate import run_levelc_predictor_corrector
from coupling.film_ode import FilmOdeParams
from reference.analytical_models import pressure_profile_from_exponential_temperature
from reference.constants import PhysicalParams
from reference.thermal_admittance import thermal_admittance_halfspace

BASE = Path(__file__).resolve().parents[1]/'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


def make_solver(nx=1):
    config=load_config(BASE)
    config['numerics'].update(nx=nx,ny=32)
    return GasSolver2D(config)


@pytest.mark.parametrize('power',[1.,-1.])
def test_default_coupling_conserves_film_plus_actual_gas_energy(power):
    solver=make_solver()
    mp=solver.mapping
    params=FilmOdeParams(C_A_si=7e-4,T_ref_K=300,gas_flux_factor=2)
    result=run_levelc_predictor_corrector(solver=solver,params=params,drive=power,
        n_steps=256,wall_bc='mass_neutral',grad_extrap='row1')
    # 初态由明确的均匀平衡状态独立确定，不使用被测审计器的返回值。
    gas_initial=solver.ny*mp.lattice.dx_m*mp.pressure_scale*mp.theta_ref_lu*2.5
    gas_final=np.sum(solver.get_macro().E_tot)*mp.pressure_scale*mp.lattice.dx_m
    film_change=params.C_A_si*(result.T_s_K[-1]-300)
    work=power*result.t_si[-1]
    assert abs(gas_final-gas_initial+film_change-work)/abs(work)<2e-6


def test_default_coupling_wall_and_film_are_the_same_state():
    solver=make_solver()
    result=run_levelc_predictor_corrector(solver=solver,
        params=FilmOdeParams(C_A_si=7e-4),drive=100.,n_steps=32,
        wall_bc='mass_neutral',grad_extrap='row1')
    assert np.max(abs(result.T_s_K-result.T_wall_K))<1e-10


def test_unforced_equilibrium_is_not_a_false_energy_failure():
    solver=make_solver()
    result=run_levelc_predictor_corrector(solver=solver,params=FilmOdeParams(),drive=0.,n_steps=128)
    assert np.max(abs(result.T_s_K-300))<1e-9
    assert result.passed_smoke


def test_explicit_legacy_scheme_keeps_its_original_default_wall():
    result=run_levelc_predictor_corrector(solver=make_solver(),params=FilmOdeParams(),
        drive=1.,n_steps=2,scheme='heun_picard1')
    assert result.wall_bc == 'equilibrium_clamp'
    assert result.q_extraction == 'moment_row'


def test_interval_flux_closes_energy_with_leak_and_multiple_x_columns():
    solver=make_solver(nx=4)
    params=FilmOdeParams(C_A_si=7e-4,linear_leak_conductance_si=500.)
    result=run_levelc_predictor_corrector(solver=solver,params=params,drive=100.,n_steps=256)
    dt=solver.mapping.lattice.dt_s
    leak=np.trapezoid(500*(result.T_s_K-300),result.t_si)
    input_work=100*result.t_si[-1]
    wall_heat=2*np.sum(result.q_g_one_sided_si[1:])*dt
    film_storage=7e-4*(result.T_s_K[-1]-300)
    assert abs(input_work-leak-wall_heat-film_storage)/input_work<1e-7
    gas_change=(np.mean(solver.get_macro().E_tot,axis=1).sum()-solver.ny*2.5*solver.mapping.theta_ref_lu)
    gas_change*=solver.mapping.pressure_scale*solver.mapping.lattice.dx_m
    assert abs(input_work-leak-gas_change-film_storage)/input_work<2e-6
    assert np.max(abs(result.mass_lu/result.mass_lu[0]-1))<1e-11


@pytest.mark.parametrize('kwargs',[{'q_feedback_scale':2.},{'q_feedback_relax':.5},
    {'q_extraction':'moment_row'},{'wall_bc':'thermal_grad'},{'dt_si':1e-6}])
def test_conservative_entry_rejects_nonconservative_combinations(kwargs):
    with pytest.raises(ValueError):
        run_levelc_predictor_corrector(solver=make_solver(),params=FilmOdeParams(),
            drive=1.,n_steps=2,**kwargs)


def test_exponential_thermal_mode_radiation_obeys_energy_limit():
    mp=make_solver().mapping
    ph=mp.physical
    params=PhysicalParams(T0=ph.T0_K,rho0=ph.rho0_kg_m3,c0=ph.c0_m_s,
        p0=mp.pressure_scale*mp.theta_ref_lu,gamma=ph.gamma,cp=ph.cp_J_kgK,
        kg=ph.kg_W_mK,nu0=ph.nu0_m2_s)
    frequency=10000.
    omega=2*np.pi*frequency
    delta=np.sqrt(2*params.alpha0/omega)
    distance=10*delta
    p,u=pressure_profile_from_exponential_temperature(frequency,1+0j,np.array([0,distance]),params)
    q=thermal_admittance_halfspace(frequency,params)
    # 热膨胀体积速度积分 + p=rho*c*u 给出 (gamma-1)/c，独立于被测源项。
    expected=(params.gamma-1)/params.c0
    outgoing=p[1]*np.exp(1j*omega/params.c0*distance)
    assert abs(outgoing/q/expected-1)<1e-4
    assert abs(u[0])<1e-12
