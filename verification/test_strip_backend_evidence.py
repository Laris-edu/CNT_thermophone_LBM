"""Bind boundary evidence to the backend that will advance the power case."""
import numpy as np
import pytest
import yaml

from Compute.finite_strip.run import compare_batch, validate_boundary_evidence
from simulation.finite_strip import compare_pulse_cases, prepare_strip_case
from verification.test_strip_boundary_gate import write_pulse


def _set_backend(config, backend):
    if backend is None:
        config['numerics'].pop('collision_backend', None)
    else:
        config['numerics']['collision_backend'] = backend


def _pulse_comparison(trial_backend, reference_backend):
    common = dict(kind='pulse', dx_m=1e-5, sponge_width_m=0., sponge_rate=0.,
                  pml_width_m=.0002)
    trial = prepare_strip_case(dict(common, name='trial', width_m=.002, height_m=.0005))
    reference = prepare_strip_case(dict(common, name='reference', width_m=.0042, height_m=.0021))
    _set_backend(trial, trial_backend)
    _set_backend(reference, reference_backend)
    time = np.arange(1, 282)*trial['lattice']['dt_s']
    trace = dict(t_s=time, probe_x_m=np.array([0.]), probe_y_m=np.array([.00025]),
                 pressure_perturbation_Pa=np.exp(-((time-1e-6)/3e-7)**2)[:, None])
    return compare_pulse_cases(trace, trace, return_start_s=1.8e-6,
                               trial_effective=trial, reference_effective=reference)


@pytest.mark.parametrize('trial,reference', [('cupy', 'numba'), ('numba', 'cupy'),
                                            (None, 'cupy'), ('cupy', None)])
def test_pulse_reference_rejects_different_execution_backend(trial, reference):
    with pytest.raises(ValueError, match='collision_backend'):
        _pulse_comparison(trial, reference)


@pytest.mark.parametrize('trial,reference', [(None, 'numba'), ('numba', None),
                                            (None, None), ('cupy', 'cupy')])
def test_pulse_reference_matches_legacy_numba_and_explicit_same_backend(trial, reference):
    assert _pulse_comparison(trial, reference)['boundary_validation_passed']


def _proof(root, backend):
    for name, width, height in [('pulse_candidate_dx10um', .002, .0005),
        ('pulse_reference_dx10um', .0042, .0021), ('pulse_sides_dx10um', .002, .002),
        ('pulse_top_dx10um', .004, .0005)]:
        write_pulse(root, name, width, height, 0.)
        path = root/'raw'/name/'effective.yaml'
        config = yaml.safe_load(path.read_text())
        _set_backend(config, backend)
        path.write_text(yaml.safe_dump(config))
    compare_batch(root, dict(validation=dict(pulse_return_start_s=1.8e-6,
                                            pulse_return_error_limit=.01)))
    return prepare_strip_case(dict(name='power', kind='power', sponge_width_m=0.,
                                   sponge_rate=0., pml_width_m=.0002))


@pytest.mark.parametrize('pulse,power', [('numba', 'cupy'), ('cupy', 'numba'),
                                        (None, 'cupy'), ('cupy', None)])
def test_boundary_evidence_rejects_power_backend_substitution(tmp_path, pulse, power):
    config = _proof(tmp_path, pulse)
    _set_backend(config, power)
    with pytest.raises(ValueError, match='collision_backend'):
        validate_boundary_evidence(tmp_path, config, source_id='test-source')


@pytest.mark.parametrize('pulse,power', [(None, 'numba'), ('numba', None),
                                        (None, None), ('cupy', 'cupy')])
def test_boundary_evidence_accepts_matching_backend_including_legacy(tmp_path, pulse, power):
    config = _proof(tmp_path, pulse)
    _set_backend(config, power)
    assert validate_boundary_evidence(tmp_path, config, source_id='test-source')['boundary_validation_passed']
