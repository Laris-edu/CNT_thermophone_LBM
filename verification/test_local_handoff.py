"""新 local 基线的热流接口检查；不代替动态热壁/完整耦合验证。"""
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from core.solver import GasSolver2D
from core.macroscopic import central_energy_flux_lu
from phase3_interfaces.heat_flux_extraction import (
    UPPER_GAS_WALL_NORMAL, convert_heat_flux_lu_to_phys,
    extract_wall_heat_flux, normal_heat_flux_lu,
)


def test_local_thermal_gradient_readout_and_si_conversion():
    config=load_config(Path(__file__).resolve().parents[1]/'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')
    config['numerics'].update(nx=16,ny=16)
    solver=GasSolver2D(config)
    factor=1+1e-5*np.sin(2*np.pi*np.arange(16)/16)[:,None]
    solver.initialize_from_macro(1/factor,np.zeros(2),solver.mapping.theta_ref_lu*factor)
    solver.step(5)
    measured=extract_wall_heat_flux(solver.f,solver.g,config=config)
    expected=normal_heat_flux_lu(solver.get_heat_flux_lu(),UPPER_GAS_WALL_NORMAL)
    assert np.isfinite(measured).all() and np.max(abs(measured))>1e-10
    np.testing.assert_allclose(measured,expected,rtol=1e-10,atol=1e-20)
    physical=extract_wall_heat_flux(solver.f,solver.g,config=config,return_physical=True)
    np.testing.assert_allclose(physical,convert_heat_flux_lu_to_phys(measured,config),rtol=1e-12)


def test_unconfigured_heat_flux_defaults_to_d2q37_raw_moment():
    config=load_config(Path(__file__).resolve().parents[1]/'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')
    config['numerics'].update(nx=2,ny=2)
    solver=GasSolver2D(config)
    solver.initialize_from_macro(1.,np.zeros(2),solver.mapping.theta_ref_lu)
    g=solver.g+1e-6*solver.lattice.w*solver.lattice.c[:,1]
    raw=normal_heat_flux_lu(central_energy_flux_lu(solver.f,g,lattice=solver.lattice))
    assert np.max(abs(raw))>1e-8
    np.testing.assert_allclose(extract_wall_heat_flux(solver.f,g),raw)
    # 缺配置只能读原始矩，不能静默冒充已修正的物理传导热流。
    conductive=extract_wall_heat_flux(solver.f,g,config=config)
    assert not np.allclose(conductive,raw,rtol=1e-3,atol=0.)
    with pytest.raises(ValueError,match='config is required'):
        extract_wall_heat_flux(solver.f,g,return_physical=True)
    with pytest.raises(ValueError,match='D2Q21'):
        extract_wall_heat_flux(solver.f,g,velocity_set='D2Q21')
