"""Manufactured Hankel fixtures for analysis; these do not validate LBM data."""
import importlib
import importlib.util
import json

import numpy as np
import pytest
import yaml
from scipy.io import loadmat
from scipy.special import hankel2

from core.unit_mapping import create_unit_mapping
from farfield.strip_surface import upper_rectangle
from simulation.finite_strip import prepare_strip_case


def analysis_api():
    assert importlib.util.find_spec('Compute.finite_strip.analyze') is not None, 'nearfield analysis is missing'
    return importlib.import_module('Compute.finite_strip.analyze')


@pytest.fixture
def hankel_case():
    config = prepare_strip_case(dict(name='manufactured_only', kind='power', dx_m=2e-6,
        sponge_width_m=0., sponge_rate=0., pml_width_m=.0002))
    m = create_unit_mapping(config)
    f = config['strip']['frequency_Hz']; w = 2*np.pi*f; k = w/m.physical.c0_m_s
    t = (np.arange(1, 321)/32+.37)/f  # Deliberately nonzero absolute phase origin.
    heat_t = t-.5*m.lattice.dt_s
    factor = .02*np.exp(.63j)

    def hats(x, y):
        # Source at y=0 has exactly the required even/odd mirror symmetry.
        rx = np.asarray(x)-.0001; ry = np.asarray(y)
        r = np.hypot(rx, ry)
        p = factor*hankel2(0, k*r)
        radial = -1j*factor*hankel2(1, k*r)/(m.physical.rho0_kg_m3*m.physical.c0_m_s)
        return p, radial*rx/r, radial*ry/r

    surfaces = [upper_rectangle(a, b, 5e-6) for a, b in
                ((.00065, .00015), (.0007, .0002), (.00075, .00025))]
    geometry = {key: np.concatenate([s[key] for s in surfaces]) for key in surfaces[0]}
    sizes = np.array([len(s['x_m']) for s in surfaces]); ends = np.cumsum(sizes)
    slices = np.column_stack((np.r_[0, ends[:-1]], ends))
    p, ux, uy = hats(geometry['x_m'], geometry['y_m'])
    wave = np.exp(1j*w*t)[:, None]
    rho0, T0 = m.physical.rho0_kg_m3, m.physical.T0_K
    fields = np.stack((np.real(wave*p), np.real(wave*ux), np.real(wave*uy),
        T0+np.zeros_like(wave.real*p.real), rho0+np.real(wave*p)/m.physical.c0_m_s**2), axis=-1)
    # An inner circular observation is retained, but must never validate all Gamma.
    px = np.array([0., 0., -.00078, .00078]); py = np.array([.0001, .00028, .0001, .0001])
    direct, _, _ = hats(px, py)
    T_hat = np.array([.03+.02j, .02-.01j, .015+.009j, .025-.005j])
    q_hat = np.array([4+3j, 2-3j, 3+1j, 5-.8j])
    data = dict(t_s=t, heat_time_s=heat_t, audit_t_s=np.array([10.47/f]),
        surface_t_s=t, surface_fields=fields, surface_slices=slices,
        surface_channels=np.array(['pressure_perturbation_Pa','ux_m_s','uy_m_s',
                                   'temperature_K','rho_kg_m3']),
        pressure_perturbation_Pa=np.real(wave*direct), probe_x_m=px, probe_y_m=py,
        film_temperature_K=T0+.2+np.real(wave*T_hat),
        film_wall_energy_J_m2=m.lattice.dt_s*(10+np.real(np.exp(1j*w*heat_t)[:,None]*q_hat)),
        film_x_m=np.linspace(-.0004,.0004,4), run_completed=True)
    data.update({f'surface_{key}': value for key,value in geometry.items()})
    summary=dict(status='completed',state_validation_passed=True,
                 state_checks={'finite':True,'positive':True,'energy_conservation':True})
    boundary=dict(boundary_validation_passed=True,limit=.01,max_return_error_ratio=.0005)
    return data, config, summary, boundary, dict(p=p,ux=ux,uy=uy,direct=direct,T=T_hat,q=q_hat)


def test_absolute_phase_midpoint_heat_and_all_contours(hankel_case):
    api=analysis_api()
    data,config,summary,boundary,want=hankel_case
    result,arrays=api.analyze_data(data,config,summary,boundary)
    assert result['fit_window']['end_cycle']==10
    assert result['fit_window']['start_cycle']==8
    assert result['fit_window']['time_origin_shift_s']==0.
    np.testing.assert_allclose(arrays['surface_p_hat_Pa'],want['p'],rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(arrays['film_T_hat_K'],want['T'],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(arrays['film_q_hat_W_m2'],want['q'],rtol=1e-10,atol=1e-10)
    np.testing.assert_array_equal(arrays['validation_probe_indices'],[1,2,3])
    np.testing.assert_allclose(arrays['kirchhoff_probe_p_hat_Pa'],np.tile(want['direct'][1:],(3,1)),rtol=.002)
    assert len(result['contours'])==3
    assert all(row['stationarity']['checked_surface_points'] > 100 for row in result['contours'])
    assert result['source_validation_passed']
    assert result['propagation_validation_passed'] is None


def test_one_unselected_film_bin_and_one_surface_node_prevent_certification(hankel_case):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    f=config['strip']['frequency_Hz']
    # Historically only bins 0, N//2, N-1 were checked; bin 1 must also count.
    data['film_temperature_K'][data['t_s']*f>8,1] += .02*np.cos(2*np.pi*f*data['t_s'][data['t_s']*f>8])
    data['surface_fields'][data['surface_t_s']*f>8,17,0] *= 1.1
    result,_=api.analyze_data(data,config,summary,boundary)
    assert not result['checks']['film_stationarity']
    assert not result['checks']['surface_stationarity']
    assert not result['source_validation_passed']


@pytest.mark.parametrize('defect',['boundary','wrong_velocity','warm_background','no_exterior_probes'])
def test_failed_nearfield_gate_blocks_mat_creation(tmp_path,hankel_case,defect):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    if defect=='boundary':
        boundary['boundary_validation_passed']=False
    elif defect=='wrong_velocity':
        data['surface_fields'][:,:,1:3] *= -1
    elif defect=='warm_background':
        data['surface_fields'][:,:,3] += 5.
    else:
        data['probe_x_m'][:]=0.; data['probe_y_m'][:]=.0001
    result,arrays=api.analyze_data(data,config,summary,boundary)
    assert not result['source_validation_passed']
    output=tmp_path/'invalid.mat'
    with pytest.raises(ValueError,match='source validation'):
        api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')
    assert not output.exists()
    assert result['propagation_validation_passed'] is None


def test_valid_export_uses_measured_p_and_u_and_reports_snapped_observers(tmp_path,hankel_case):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    result,arrays=api.analyze_data(data,config,summary,boundary)
    output=tmp_path/'source.mat'
    api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')
    actual=loadmat(output,squeeze_me=True)
    assert actual['source_validation_passed']==1
    assert actual['dx_m']==.00005 and actual['pml_size']==32
    assert actual['evaluation_cycles']>=8.25
    assert actual['domain_half_width_m']==.004 and actual['domain_half_height_m']==.049
    assert np.max(abs(actual['p_hat_Pa']-actual['rho0_kg_m3']*actual['c0_m_s']*actual['un_hat_m_s']))>.01
    assert .045 in np.atleast_1d(actual['observer_y_m'])
    assert not np.array_equal(actual['observer_y_m'],actual['expected_grid_observer_y_m'])
    assert np.count_nonzero(actual['observer_role']=='interior_cancel') >= 1
    with pytest.raises(FileExistsError):
        api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')


def test_missing_first_part_of_eight_cycle_observation_does_not_certify(hankel_case):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    keep=data['t_s']*config['strip']['frequency_Hz']>3
    for key in ('t_s','surface_t_s','heat_time_s','surface_fields',
                'film_temperature_K','film_wall_energy_J_m2','pressure_perturbation_Pa'):
        data[key]=data[key][keep]
    result,_=api.analyze_data(data,config,summary,boundary)
    assert not result['checks']['observation_time_coverage']
    assert not result['validation_passed']
    assert not result['source_validation_passed']


def test_surface_stationarity_is_reported_separately_from_film(hankel_case):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    select=data['t_s']*config['strip']['frequency_Hz']>8
    data['film_temperature_K'][select,0] += .02*np.cos(2*np.pi*config['strip']['frequency_Hz']*data['t_s'][select])
    result,_=api.analyze_data(data,config,summary,boundary)
    assert result['checks']['surface_stationarity']
    assert not result['checks']['film_stationarity']
    assert not result['source_validation_passed']


@pytest.mark.parametrize('boundary_flag',[False,True])
def test_failed_cli_preserves_diagnostics_and_raw_hashes_without_mat(tmp_path,hankel_case,boundary_flag):
    api=analysis_api()
    data,config,summary,boundary,_=hankel_case
    # Reordered channels must be indexed by their saved names, not column position.
    order=[4,2,0,3,1]
    data['surface_channels']=data['surface_channels'][order]
    data['surface_fields']=data['surface_fields'][:,:,order]
    boundary['boundary_validation_passed']=boundary_flag
    case=tmp_path/'raw'/'manufactured_only'; case.mkdir(parents=True)
    final=tmp_path/'final'; final.mkdir()
    (case/'effective.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    (case/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (case/'manifest.json').write_text(json.dumps({'fixture':'manufactured, not LBM','status':'completed'}),encoding='utf-8')
    np.savez_compressed(case/'trajectory.npz',**data)
    (final/'boundary_validation.json').write_text(json.dumps(boundary),encoding='utf-8')
    assert api.main(['--batch',str(tmp_path),'--case','manufactured_only','--export-kwave'])==2
    saved=json.loads((final/'manufactured_only_nearfield.json').read_text(encoding='utf-8'))
    assert saved['validation_passed']  # State/periodicity and source are different gates.
    assert not saved['source_validation_passed']
    assert not saved['checks']['boundary_evidence_bound']
    assert saved['propagation_validation_passed'] is None
    assert len(saved['provenance']['raw_sha256'])==4
    assert all(len(v)==64 for v in saved['provenance']['raw_sha256'].values())
    assert saved['provenance']['boundary_sha256']==api._sha256(final/'boundary_validation.json')
    assert (final/'manufactured_only_nearfield.npz').exists()
    assert not (final/'manufactured_only_kwave_input.mat').exists()
    with pytest.raises(FileExistsError):
        api.main(['--batch',str(tmp_path),'--case','manufactured_only','--export-kwave'])
