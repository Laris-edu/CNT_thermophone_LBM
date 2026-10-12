"""A good combined trace cannot bypass a failed or missing side check."""
import json

import numpy as np
import pytest
import yaml

from Compute.finite_strip.run import compare_batch
from simulation.finite_strip import prepare_strip_case


def write_pulse(root,name,width,height,error):
    c=prepare_strip_case(dict(name=name,kind='pulse',dx_m=1e-5,
        width_m=width,height_m=height,sponge_width_m=0,sponge_rate=0,
        pml_width_m=.0002))
    out=root/'raw'/name; out.mkdir(parents=True)
    (out/'effective.yaml').write_text(yaml.safe_dump(c))
    (out/'summary.json').write_text(json.dumps(dict(status='completed',state_validation_passed=True)))
    (out/'manifest.json').write_text(json.dumps(dict(status='completed',source_id='test-source',
        source_clean=True,git_commit='test-commit')))
    t=np.arange(1,282)*c['lattice']['dt_s']
    p=np.exp(-((t-1e-6)/3e-7)**2)
    p[t>2e-6]+=error
    np.savez(out/'trajectory.npz',t_s=t,probe_x_m=[0.],probe_y_m=[.00025],
             pressure_perturbation_Pa=p[:,None])


def test_failed_side_component_blocks_combined_boundary_gate(tmp_path):
    for name,w,h,error in [('pulse_candidate_dx10um',.002,.0005,.001),
        ('pulse_reference_dx10um',.0042,.0021,0.),
        ('pulse_sides_dx10um',.002,.002,.02),('pulse_top_dx10um',.004,.0005,.001)]:
        write_pulse(tmp_path,name,w,h,error)
    compare_batch(tmp_path,dict(validation=dict(pulse_return_start_s=1.8e-6,pulse_return_error_limit=.01)))
    result=json.loads((tmp_path/'final/boundary_validation.json').read_text())
    assert result['boundary_validation_passed'] is False
    assert result['components']['combined']['boundary_validation_passed'] is True
    assert result['components']['sides']['boundary_validation_passed'] is False
    assert result['max_return_error_ratio']==pytest.approx(.02,rel=.003)


def test_missing_separate_boundary_evidence_is_not_a_pass(tmp_path):
    write_pulse(tmp_path,'pulse_candidate_dx10um',.002,.0005,.001)
    write_pulse(tmp_path,'pulse_reference_dx10um',.0042,.0021,0.)
    with pytest.raises(FileNotFoundError):
        compare_batch(tmp_path,dict(validation=dict(pulse_return_start_s=1.8e-6,pulse_return_error_limit=.01)))
    assert not (tmp_path/'final/boundary_validation.json').exists()


def test_modified_boundary_data_cannot_authorize_a_long_power_run(tmp_path,monkeypatch):
    from Compute.finite_strip.run import validate_boundary_evidence
    import simulation.provenance
    for name,w,h in [('pulse_candidate_dx10um',.002,.0005),
        ('pulse_reference_dx10um',.0042,.0021),('pulse_sides_dx10um',.002,.002),
        ('pulse_top_dx10um',.004,.0005)]:
        write_pulse(tmp_path,name,w,h,0.)
    compare_batch(tmp_path,dict(validation=dict(pulse_return_start_s=1.8e-6,pulse_return_error_limit=.01)))
    monkeypatch.setattr(simulation.provenance,'capture_source',lambda root: {'source_id':'test-source'})
    power=prepare_strip_case(dict(name='power',kind='power',sponge_width_m=0,sponge_rate=0,pml_width_m=.0002))
    validate_boundary_evidence(tmp_path,power)
    target=tmp_path/'raw/pulse_top_dx10um/trajectory.npz'
    target.write_bytes(target.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='changed'):
        validate_boundary_evidence(tmp_path,power)
