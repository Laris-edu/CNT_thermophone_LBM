"""Expansion and lifecycle contracts for synchronized two-level experiments."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from simulation.study import ROOT, prepare_case, validate_effective

BASE = ROOT/'Compute/open_planar/e1_long_air.yaml'


def parameters(**changes):
    result = dict(name='step',mode='thermal_step',refine=4,length_m=.00100030025,
        probes_m=[.00012,.0003,.0006],amplitude_K=.1,
        thermal_step=dict(duration_s=5e-6,interface_m=20.894e-6),
        grid=dict(type='two_level',ratio=2,interface_m=20.894e-6),
        sampling=dict(probe_interpolation='linear',sync_fine_steps=2))
    result.update(changes)
    return result


def test_expand_resolves_physical_cells_and_work_without_changing_requested_interface():
    config=prepare_case(parameters(),BASE)
    layout=config['grid_layout']
    assert (layout['fine_cells'],layout['coarse_cells'],layout['shadow_cells']) == (32,750,6)
    assert config['numerics']['steps']==2688
    assert layout['sync_steps']==1344
    assert layout['cell_updates']==1344*(2*32+750)
    assert layout['interface_m']==20.894e-6
    assert layout['actual_length_m']==pytest.approx(.00100030025)
    assert config['level_mappings']['coarse']['dt_s']==pytest.approx(2*config['level_mappings']['fine']['dt_s'])
    validate_effective(config)
    altered=deepcopy(config); altered['grid_layout']['fine_cells']+=2
    with pytest.raises(ValueError,match='layout'):
        validate_effective(altered)


@pytest.mark.parametrize('update',[
    {'grid':dict(type='two_level',ratio=3,interface_m=20.894e-6)},
    {'grid':dict(type='two_level',ratio=2,interface_m=21e-6)},
    {'sampling':dict(probe_interpolation='cubic',sync_fine_steps=2)},
    {'sampling':dict(probe_interpolation='linear',sync_fine_steps=1)},
    {'thermal_step':dict(duration_s=-1,interface_m=20.894e-6)},
])
def test_invalid_two_level_parameters_are_rejected(update):
    with pytest.raises(ValueError):
        prepare_case(parameters(**update),BASE)


def test_uniform_reference_keeps_the_same_sync_times():
    reference=parameters(); reference.pop('grid')
    fine=prepare_case(reference,BASE)
    two=prepare_case(parameters(),BASE)
    assert fine['numerics']['steps']==two['numerics']['steps']
    assert fine['numerics']['ny']==1532


def test_two_level_grid_implies_sync_before_sampling_validation():
    config=prepare_case(parameters(sampling=dict(probe_interpolation='linear')),BASE)
    assert config['run']['sampling']['sync_fine_steps']==2


def test_unknown_grid_fields_and_incompatible_legacy_pulse_cadence_are_rejected():
    spec=parameters(); spec['grid']['unexpected']=123
    with pytest.raises(ValueError,match='grid'):
        prepare_case(spec,BASE)
    with pytest.raises(ValueError,match='pulse'):
        prepare_case(dict(name='pulse',mode='pulse',sampling=dict(sync_fine_steps=2)),BASE)


def test_new_four_file_case_can_be_reanalyzed_without_writing(tmp_path):
    from simulation import run_case,analyze_case
    config=prepare_case(parameters(thermal_step=dict(duration_s=20e-9,interface_m=20.894e-6)),BASE)
    output=tmp_path/'case'
    result=run_case(config,output)
    assert result['status']=='completed'
    assert set(p.name for p in output.iterdir())=={'effective.yaml','manifest.json','trajectory.npz','summary.json'}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    repeated=analyze_case(output)
    assert repeated['validation_passed']==result['validation_passed']
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    with np.load(output/'trajectory.npz') as data:
        assert len(data['heat_time_s'])==2*len(data['t_s'])
        assert result['cell_updates']==len(data['t_s'])*(2*32+750)
    with pytest.raises(FileExistsError):
        run_case(config,output)
