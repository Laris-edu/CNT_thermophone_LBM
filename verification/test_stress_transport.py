"""局部输运基线的实测应力回归：检查声波、剪切波与热扩散的真实演化。"""
from pathlib import Path

import numpy as np
import pytest

from config_io import load_config
from core.solver import GasSolver2D
from core.collision_smrt import collide_fg
from verification.acoustic_wave_measurement import AcousticWaveSettings, measure_acoustic_wave_direction
from verification.shear_wave_measurement import ShearWaveSettings, measure_shear_wave_direction
from verification.thermal_diffusion_measurement import ThermalDiffusionSettings, measure_thermal_diffusion_direction


CONFIG = Path(__file__).resolve().parents[1]/'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


def test_small_k_acoustic_attenuation_matches_configured_zero_bulk_target():
    # Every lattice step is sampled: the legacy Prony helper's rate is per sample.
    result=measure_acoustic_wave_direction(load_config(CONFIG),'x',
        AcousticWaveSettings(nx=128,ny=4,steps=300,fit_start=40,sample_interval=1,amplitude=1e-6))
    measured=result['acoustic_attenuation_measured_lu']
    target=result['acoustic_attenuation_reference_lu']
    assert np.isfinite(measured) and measured>0
    assert abs(measured/target-1)<.03, result
    assert result['sound_speed_relative_error']<.002, result


@pytest.mark.parametrize('direction,shape',[('x',(128,4)),('y',(4,128))])
def test_shear_viscosity_matches_the_same_physical_mapping(direction,shape):
    nx,ny=shape
    result=measure_shear_wave_direction(load_config(CONFIG),direction,
        ShearWaveSettings(nx=nx,ny=ny,steps=180,fit_start=40,amplitude=1e-6))
    assert result['relative_error']<.005,result


def test_trace_repair_preserves_thermal_diffusion():
    result=measure_thermal_diffusion_direction(load_config(CONFIG),'x',
        ThermalDiffusionSettings(nx=128,ny=4,steps=260,fit_start=50,amplitude=1e-6))
    assert result['relative_error']<.02,result


@pytest.mark.parametrize('component',['normal','shear'])
def test_nonzero_stress_relaxes_even_when_velocity_gradient_is_zero(component):
    # A uniform non-equilibrium stress has zero velocity gradient. Its finite
    # relaxation must come from the stress itself, not a gradient replacement.
    s=GasSolver2D(load_config(CONFIG))
    s.nx=s.ny=1
    s.initialize_from_macro(1.,np.zeros(2),s.mapping.theta_ref_lu)
    cx,cy=s.lattice.c.T
    basis=cx*cx-cy*cy if component=='normal' else cx*cy
    f=s.f+1e-6*s.lattice.w*basis
    g=s.g.copy()
    fpost,gpost=collide_fg(f,g,s.mapping,lattice=s.lattice)
    before=float(np.sum((f-s.f)*basis))
    after=float(np.sum((fpost-s.f)*basis))
    assert after/before==pytest.approx(1-1/s.mapping.tau21,abs=2e-8)
    assert np.min(fpost)>0 and np.min(gpost)>0
