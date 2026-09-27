"""Configuration expansion, non-overwrite, provenance and execution status contracts."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml'


def write_study(tmp_path, **values):
    path = tmp_path / 'study.yaml'
    path.write_text(yaml.safe_dump(dict(base_config=str(BASE), **values)), encoding='utf-8')
    return path


def test_study_expands_sweep_and_resolves_complete_configuration(tmp_path):
    from simulation.study import load_study
    path = write_study(tmp_path, defaults={'refine': 1, 'cycles': 6},
        cases=[{'name': 'pulse', 'mode': 'pulse'}],
        sweeps=[{'name': 'thermal', 'parameters': {'frequency_Hz': [50000, 75000], 'refine': [2, 4]}}])
    cases = load_study(path)
    assert len(cases) == 5
    assert len({c['case']['name'] for c in cases}) == 5
    assert cases[0]['run']['mode'] == 'pulse'
    for case in cases[1:]:
        run = case['run']
        assert case['lattice']['dx_m'] == pytest.approx(2.61175e-6 / run['refine'])
        assert run['probes_m'] == [.00012, .0003, .0006]
        assert run['analysis']['harmonics'] == 3
        assert run['sampling']['profile_cycles'] == 2
        assert case['numerics']['boundary'] == 'planar_open_y'
        assert case['numerics']['steps'] > 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ['study.yaml']


@pytest.mark.parametrize('bad', [
    {'name': '../outside'}, {'name': 'CON'}, {'name': 'x', 'refine': 1.5},
    {'name': 'x', 'frequency_Hz': float('nan')}, {'name': 'x', 'probes_m': [.002]},
    {'name': 'x', 'amplitdue_K': .2}, {'name': 'x', 'cycles': 3, 'ramp_cycles': 2},
    {'name': 'x', 'analysis': {'detrend_order': 1.0}},
    {'name': 'x', 'mode': 'pulse', 'refine': 1, 'pulse': {'window_fraction': .0001}},
])
def test_invalid_cases_are_rejected_before_creating_outputs(tmp_path, bad):
    from simulation.study import load_study
    path = write_study(tmp_path, cases=[bad])
    with pytest.raises(ValueError):
        load_study(path)
    assert len(list(tmp_path.iterdir())) == 1


def test_duplicate_case_names_and_batch_collisions_are_rejected_before_run(tmp_path):
    from simulation import run_study
    from simulation.study import load_study
    path = write_study(tmp_path, cases=[{'name': 'same'}, {'name': 'Same'}])
    with pytest.raises(ValueError, match='duplicate'):
        load_study(path)
    path = write_study(tmp_path, defaults={'mode': 'pulse', 'refine': 1},
                       cases=[{'name': 'first'}, {'name': 'existing'}])
    existing = tmp_path / 'batch/raw/existing'
    existing.mkdir(parents=True)
    (existing/'keep.txt').write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        run_study(path, 'batch')
    assert not (tmp_path/'batch/raw/first').exists()
    assert (existing/'keep.txt').read_text() == 'keep'


def test_pulse_case_has_four_files_and_separate_execution_and_validation_status(tmp_path):
    from simulation import prepare_case, run_case
    effective = prepare_case({'name': 'pulse', 'mode': 'pulse', 'refine': 1})
    output = tmp_path/'batch/raw/pulse'
    result = run_case(effective, output, source_cache=tmp_path/'sources')
    assert {p.name for p in output.iterdir()} == {'trajectory.npz','summary.json','effective.yaml','manifest.json'}
    assert result['reflection_ratio'] == pytest.approx(.00490664292097417, rel=1e-10)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status'] == 'completed'
    assert manifest['git_commit']
    assert isinstance(manifest['git_dirty'], bool)
    assert manifest['source_sha256']
    assert not (tmp_path/'sources').exists()
    assert not list(tmp_path.rglob('*.zip'))
    assert manifest['finished_utc'] and manifest['elapsed_s'] > 0
    assert result['validation_passed'] is True
    with np.load(output/'trajectory.npz') as data:
        assert data['t_s'].size == result['steps']
        assert bool(data['run_completed'])
    with pytest.raises(FileExistsError):
        run_case(effective, output, source_cache=tmp_path/'sources')


@pytest.mark.parametrize('exception,status', [(RuntimeError('test failure'),'failed'), (KeyboardInterrupt(),'interrupted')])
def test_failure_keeps_partial_trajectory_and_marks_existing_records(tmp_path, monkeypatch, exception, status):
    from simulation import prepare_case, run_case
    from core.planar_open_solver import PlanarOpenSolver
    original = PlanarOpenSolver.step_temperature
    def step(self, temperature):
        if self.t_lu == 2:
            raise exception
        return original(self, temperature)
    monkeypatch.setattr(PlanarOpenSolver, 'step_temperature', step)
    output = tmp_path/'pulse'
    effective = prepare_case({'name': 'pulse', 'mode': 'pulse', 'refine': 1})
    with pytest.raises(type(exception)):
        run_case(effective, output, source_cache=tmp_path/'sources')
    assert {p.name for p in output.iterdir()} == {'trajectory.npz','summary.json','effective.yaml','manifest.json'}
    assert json.loads((output/'manifest.json').read_text())['status'] == status
    assert json.loads((output/'summary.json').read_text())['validation_passed'] is None
    with np.load(output/'trajectory.npz') as data:
        assert len(data['t_s']) == 2
        assert not bool(data['run_completed'])


def test_source_metadata_without_git_is_read_only_and_tracks_content(tmp_path):
    from simulation.provenance import capture_source
    root = tmp_path/'project'
    (root/'core').mkdir(parents=True)
    (root/'core/example.py').write_text('value = 1\n', encoding='utf-8')
    cache = tmp_path/'sources'
    first = capture_source(root, cache)
    second = capture_source(root, cache)
    assert not cache.exists()
    assert first['git_commit'] is None and first['git_dirty'] is None
    assert first['source_clean'] is False
    assert first['source_id'] == second['source_id']
    assert first['source_sha256'] == second['source_sha256']
    (root/'core/example.py').write_text('value = 2\n', encoding='utf-8')
    third = capture_source(root, cache)
    assert third['source_id'] != first['source_id']
    assert not cache.exists()
    assert list(root.rglob('*')) == [root/'core', root/'core/example.py']


@pytest.mark.parametrize('change', ['modified', 'staged', 'deleted', 'untracked', 'case_script'])
def test_git_records_clean_and_dirty_states_without_archiving(tmp_path, change):
    from simulation.provenance import capture_source
    root=tmp_path/'repo'; root.mkdir()
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True).stdout.strip()
    git('init')
    (root/'core').mkdir()
    source=root/'core/example.py'; source.write_text('value = 1\n',encoding='utf-8')
    (root/'config_io.py').write_text('value = 2\n',encoding='utf-8')
    git('add','core/example.py','config_io.py')
    git('-c','user.name=Compute test','-c','user.email=test@example.invalid','commit','-m','fixture')
    cache=tmp_path/'sources'
    clean=capture_source(root,cache)
    assert clean['source_clean'] and clean['git_dirty'] is False
    assert clean['git_commit'] == git('rev-parse','HEAD')
    assert not cache.exists()
    if change == 'deleted':
        source.unlink()
    elif change == 'untracked':
        (root/'core/new.py').write_text('value = 3\n',encoding='utf-8')
    elif change == 'case_script':
        (root/'Compute/topic').mkdir(parents=True)
        (root/'Compute/topic/run.py').write_text('value = 3\n',encoding='utf-8')
    else:
        source.write_text('value = 3\n',encoding='utf-8')
        if change == 'staged':
            git('add','core/example.py')
    dirty=capture_source(root,cache)
    assert dirty['git_commit'] == clean['git_commit']
    assert dirty['git_dirty'] is True
    assert dirty['source_clean'] is (change == 'case_script')
    assert not cache.exists()


def test_thermal_custom_probes_and_read_only_reanalysis(tmp_path):
    from simulation import prepare_case, run_case, analyze_case
    output=tmp_path/'thermal'
    config=prepare_case(dict(name='thermal',refine=1,length_m=.00015,frequency_Hz=200000.,
        probes_m=[.00003,.00008],cycles=2.,ramp_cycles=0.,sampling={'profile_samples_per_cycle':32}))
    result=run_case(config,output,source_cache=tmp_path/'sources')
    # 历史 manifest 中的快照路径仅为记录；重分析不要求归档存在，也不重建归档。
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['source_snapshot']='../missing_historical_source.zip'
    manifest_path.write_text(json.dumps(manifest),encoding='utf-8')
    import hashlib
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    replay=analyze_case(output)
    assert replay['probes']==result['probes']
    assert replay['heat_flux']==result['heat_flux']
    assert len(replay['probes'])==2
    assert hashes=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}


@pytest.mark.parametrize('interrupt_after', [None, 2])
def test_thermal_velocity_samples_match_committed_normal_velocity(tmp_path, monkeypatch, interrupt_after):
    from core.planar_open_solver import PlanarOpenSolver
    from simulation import prepare_case
    from simulation.planar import collect_thermal

    config=prepare_case(dict(name='velocity',refine=1,length_m=.00015,frequency_Hz=200000.,
        probes_m=[.00003,.00008],cycles=2.,ramp_cycles=0.,
        sampling={'profile_samples_per_cycle':4096}))
    solver=PlanarOpenSolver(config)
    mapping=solver.mapping
    initial=solver.get_macro()
    velocity=initial.u.copy()
    velocity[...,0]=3e-5/mapping.velocity_scale
    velocity[...,1]=-1e-4/mapping.velocity_scale
    solver.initialize_from_macro(initial.rho,velocity,initial.theta)
    expected=[]
    original=solver.step_temperature
    def step(temperature):
        if interrupt_after is not None and len(expected)==interrupt_after:
            raise RuntimeError('interrupted sampling')
        audit=original(temperature)
        expected.append(solver.get_macro().u[:,0,1].copy()*mapping.velocity_scale)
        return audit
    monkeypatch.setattr(solver,'step_temperature',step)
    run=config['run'].copy()
    run['cycles']=4*mapping.lattice.dt_s*run['frequency_Hz']
    if interrupt_after is None:
        collect_thermal(solver,tmp_path,run)
    else:
        with pytest.raises(RuntimeError,match='interrupted sampling'):
            collect_thermal(solver,tmp_path,run)
    with np.load(tmp_path/'trajectory.npz') as data:
        rows=[np.argmin(abs(solver.z_m-z)) for z in run['probes_m']]
        np.testing.assert_array_equal(data['velocity_m_s'],np.asarray(expected)[:,rows])
        np.testing.assert_array_equal(data['profile_velocity_m_s'],expected)
        np.testing.assert_array_equal(data['profile_t_s'],data['t_s'])
        np.testing.assert_allclose(data['t_s'],np.arange(1,len(expected)+1)*mapping.lattice.dt_s)
        assert bool(data['run_completed']) is (interrupt_after is None)


def test_failed_numerical_gate_is_still_a_completed_run(tmp_path):
    from simulation import prepare_case, run_case
    config=prepare_case(dict(name='pulse',mode='pulse',refine=1,analysis={'reflection_ratio':1e-12}))
    result=run_case(config,tmp_path/'case',source_cache=tmp_path/'sources')
    assert result['validation_passed'] is False
    assert json.loads((tmp_path/'case/manifest.json').read_text())['status']=='completed'


def test_existing_source_archive_is_preserved(tmp_path):
    from simulation.provenance import capture_source
    root=tmp_path/'project'
    (root/'core').mkdir(parents=True)
    (root/'core/example.py').write_text('value = 1\n',encoding='utf-8')
    cache=root/'Compute/_sources'; cache.mkdir(parents=True)
    archived=cache/'historical.zip'
    archived.write_bytes(b'historical source archive')
    capture_source(root,cache)
    assert list(cache.iterdir()) == [archived]
    assert archived.read_bytes() == b'historical source archive'


def test_validation_plot_selects_cases_by_parameters_and_requires_full_group():
    from scripts.plot_planar_open import validation_groups
    thermal=lambda name,r,H: (Path(name),dict(kind='prescribed_temperature_direct_pressure',
        frequency_Hz=50000,wall_amplitude_K=.1,probes=[{}, {}, {}]),dict(refine=r,length=H))
    grid=[thermal(f'grid__refine_{r}',r,.001) for r in (1,2,4)]
    long=thermal('domain_sweep__long',2,.002)
    pulses=[(Path(f'pulse{r}'),dict(kind='outgoing_gaussian_pulse'),dict(refine=r)) for r in (1,4)]
    selected_grid,selected_pulses,domain=validation_groups([long,*pulses,*reversed(grid)])
    assert selected_grid==grid and selected_pulses==pulses
    assert domain==(grid[1],long)
    with pytest.raises(ValueError,match='pulse'):
        validation_groups([*grid,long])
    with pytest.raises(ValueError,match='grid'):
        validation_groups([*grid[:2],*pulses,long])


def test_selected_sweep_case_matches_legacy_command(tmp_path, monkeypatch):
    from simulation import run_study
    from scripts.verify_planar_open import main
    path=write_study(tmp_path,defaults={'mode':'pulse','refine':1},
        sweeps=[{'name':'pulse','parameters':{'refine':[1,2]}}])
    run_study(path,'batch',selected=['pulse__refine_1'],source_cache=tmp_path/'sources')
    raw=tmp_path/'batch/raw'
    assert [p.name for p in raw.iterdir()]==['pulse__refine_1']
    monkeypatch.setattr('sys.argv',['verify_planar_open','--mode','pulse','--refine','1',
        '--output',str(tmp_path/'legacy'),'--case','pulse'])
    main()
    new_path=raw/'pulse__refine_1'
    old_path=tmp_path/'legacy/raw/pulse'
    assert json.loads((new_path/'summary.json').read_text())==json.loads((old_path/'summary.json').read_text())
    with np.load(new_path/'trajectory.npz') as new, np.load(old_path/'trajectory.npz') as old:
        for key in new.files:
            if key!='elapsed_s':
                np.testing.assert_array_equal(new[key],old[key])
    assert not (tmp_path/'sources').exists()
    assert not list(tmp_path.rglob('*.zip'))
    assert not list(tmp_path.rglob('*.md'))
    assert not list(tmp_path.rglob('_work'))


@pytest.mark.parametrize('entry', ['verify_nearfilm_chain', 'verify_lim_nearfield'])
def test_legacy_diagnostics_record_git_without_source_archives(tmp_path, monkeypatch, entry):
    from importlib import import_module
    module=import_module('scripts.'+entry)
    def stop_calculation(*args, **kwargs):
        raise RuntimeError('stop before numerical work')
    if entry == 'verify_nearfilm_chain':
        monkeypatch.setattr(module,'run_levelc_predictor_corrector',stop_calculation)
        extra=['--mode','C']
        failure=SystemExit
        record='manifest.json'
    else:
        monkeypatch.setattr(module,'run_case',stop_calculation)
        extra=[]
        failure=RuntimeError
        record='summary.json'
    output=tmp_path/'batch'
    monkeypatch.setattr('sys.argv',[entry,'--output',str(output),*extra])
    with pytest.raises(failure):
        module.main()
    assert not list(output.rglob('*.zip'))
    metadata=json.loads((output/'raw'/record).read_text(encoding='utf-8'))
    assert metadata['git_commit']
    assert isinstance(metadata['git_dirty'],bool)
    assert metadata['source_sha256']
