"""新 local 基线的热流接口检查；不代替动态热壁/完整耦合验证。"""
from pathlib import Path

import numpy as np

from config_io import load_config
from core.solver import GasSolver2D
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
