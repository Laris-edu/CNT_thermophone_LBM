from pathlib import Path

import numpy as np
import pytest

from simulation.finite_strip import prepare_strip_case, run_strip_case, compare_pulse_cases


def small_case():
    return prepare_strip_case(dict(name='tiny', kind='pulse', dx_m=1e-5,
        width_m=12e-5, height_m=8e-5, film_width_m=0., duration_s=6e-8,
        pulse_sigma_m=2e-5, probe_radius_m=2e-5, sponge_width_m=0., sponge_rate=0.))


def test_geometry_and_clock_preserve_mapping_and_requested_sizes():
    c=prepare_strip_case(dict(name='target', kind='power', dx_m=1.305875e-6,
        width_m=.002, height_m=.0005, film_width_m=.001))
    assert c['numerics']['nx'] == 1532
    assert c['numerics']['ny'] == 383
    assert c['strip']['steps'] == 322560
    assert c['strip']['requested_height_m'] == .0005
    assert c['strip']['actual_height_m'] == pytest.approx(383*1.305875e-6)
    assert c['lattice']['dx_m']/c['lattice']['dt_s'] == pytest.approx(351.0192)


@pytest.mark.parametrize('change', [dict(dx_m=-1),dict(film_width_m=.003),dict(kind='typo')])
def test_invalid_cases_fail_before_creating_outputs(change):
    with pytest.raises(ValueError):
        prepare_strip_case(dict(name='bad', **change))


def test_short_pulse_writes_four_files_and_refuses_overwrite(tmp_path):
    output=tmp_path/'case'
    result=run_strip_case(small_case(), output)
    assert {p.name for p in output.iterdir()} == {'effective.yaml','manifest.json','trajectory.npz','summary.json'}
    assert result['status'] == 'completed'
    assert result['validation_passed'] is None  # no boundary reference yet
    assert result['state_validation_passed']
    with pytest.raises(FileExistsError):
        run_strip_case(small_case(), output)


def test_pulse_comparison_does_not_call_small_difference_a_reflection_measurement():
    t=np.linspace(0,1e-5,101)
    ref=np.exp(-((t-1e-6)/3e-7)**2)[:,None]*np.array([[1.,.5]])
    trial=ref.copy(); trial[t>5e-6,0]+=.025
    args=dict(t_s=t,pressure_perturbation_Pa=ref,probe_x_m=np.array([0.,1.]),probe_y_m=np.array([1.,0.]))
    actual={**args,'pressure_perturbation_Pa':trial}
    result=compare_pulse_cases(actual,args,return_start_s=4e-6,limit=.01)
    assert not result['boundary_validation_passed']
    assert result['max_return_error_ratio'] == pytest.approx(.025)
    assert result['frequency_20kHz_validated'] is False
