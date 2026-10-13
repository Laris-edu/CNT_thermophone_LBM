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
    return _manufactured_hankel_case()


def _manufactured_hankel_case(**parameters):
    config = prepare_strip_case(dict(name='manufactured_only', kind='power', dx_m=2e-6,
        sponge_width_m=0., sponge_rate=0., **{'pml_width_m':.0002,**parameters}))
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

    surfaces = [upper_rectangle(a, b, 5e-6) for a, b in config['strip']['handoff_contours_m']]
    geometry = {key: np.concatenate([s[key] for s in surfaces]) for key in surfaces[0]}
    sizes = np.array([len(s['x_m']) for s in surfaces]); ends = np.cumsum(sizes)
    slices = np.column_stack((np.r_[0, ends[:-1]], ends))
    p, ux, uy = hats(geometry['x_m'], geometry['y_m'])
    wave = np.exp(1j*w*t)[:, None]
    rho0, T0 = m.physical.rho0_kg_m3, m.physical.T0_K
    fields = np.stack((np.real(wave*p), np.real(wave*ux), np.real(wave*uy),
        T0+np.zeros_like(wave.real*p.real), rho0+np.real(wave*p)/m.physical.c0_m_s**2), axis=-1)
    # An inner circular observation is retained, but must never validate all Gamma.
    exterior = np.asarray(config['strip']['validation_probes_m'])
    px = np.r_[0.,exterior[:,0]]; py = np.r_[.0001,exterior[:,1]]
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
    assert actual['diagnostic_mode']==0
    assert actual['source_validation_bypassed']==0
    assert actual['dx_m']==.00005 and actual['pml_size']==32
    assert actual['evaluation_cycles']>=8.25
    assert actual['domain_half_width_m']==.004 and actual['domain_half_height_m']==.049
    assert np.max(abs(actual['p_hat_Pa']-actual['rho0_kg_m3']*actual['c0_m_s']*actual['un_hat_m_s']))>.01
    assert .045 in np.atleast_1d(actual['observer_y_m'])
    assert not np.array_equal(actual['observer_y_m'],actual['expected_grid_observer_y_m'])
    assert np.count_nonzero(actual['observer_role']=='interior_cancel') >= 1
    with pytest.raises(FileExistsError):
        api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')


def _phase_mismatched_export(hankel_case):
    data,config,summary,boundary,want=hankel_case
    omega=2*np.pi*config['strip']['frequency_Hz']
    data['pressure_perturbation_Pa']=np.real(
        np.exp(1j*omega*data['t_s'])[:,None]*want['direct']*np.exp(.05j))
    result,arrays=analysis_api().analyze_data(data,config,summary,boundary)
    result['checks']['boundary_evidence_bound']=True
    assert result['validation_passed'] is True
    assert [name for name,passed in result['checks'].items() if not passed]==['kirchhoff_vs_direct']
    assert result['source_validation_passed'] is False
    return result,arrays


def test_diagnostic_export_keeps_failed_source_gate_in_real_mat(tmp_path,hankel_case):
    api=analysis_api()
    result,arrays=_phase_mismatched_export(hankel_case)
    original=json.dumps(result,sort_keys=True)
    blocked=tmp_path/'normal_blocked.mat'
    with pytest.raises(ValueError,match='source validation'):
        api.export_kwave_input(blocked,result,arrays,toolbox_path=tmp_path/'toolbox')
    assert not blocked.exists()
    output=tmp_path/'diagnostic.mat'
    api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox',diagnostic=True)
    actual=loadmat(output,squeeze_me=True)
    assert actual['source_validation_passed']==0
    assert actual['diagnostic_mode']==1
    assert actual['source_validation_bypassed']==1
    assert actual['propagation_validation_performed']==0
    assert 'kirchhoff_vs_direct' in str(actual['diagnostic_scope'])
    assert 'not validated' in str(actual['diagnostic_scope'])
    stored=json.loads(str(actual['source_analysis_json']))
    assert stored['source_validation_passed'] is False
    assert stored['checks']['kirchhoff_vs_direct'] is False
    assert json.dumps(result,sort_keys=True)==original
    a,b=arrays['surface_slices'][1]
    np.testing.assert_array_equal(actual['p_hat_Pa'][:b-a],arrays['surface_p_hat_Pa'][a:b])


@pytest.mark.parametrize('defect',['other_gate','missing_gate','empty_checks','nonmapping_checks',
                                 'state_failed','state_missing','status_failed','source_unknown'])
def test_diagnostic_export_rejects_other_or_missing_prerequisites(tmp_path,hankel_case,defect):
    result,arrays=_phase_mismatched_export(hankel_case)
    if defect=='other_gate':
        result['checks']['geometry']=False
    elif defect=='missing_gate':
        del result['checks']['geometry']
    elif defect=='empty_checks':
        result['checks']={}
    elif defect=='nonmapping_checks':
        result['checks']=None
    elif defect=='state_failed':
        result['validation_passed']=False
    elif defect=='state_missing':
        del result['validation_passed']
    elif defect=='status_failed':
        result['status']='failed'
    else:
        result['source_validation_passed']=None
    output=tmp_path/'rejected_diagnostic.mat'
    with pytest.raises(ValueError,match='source validation'):
        analysis_api().export_kwave_input(output,result,arrays,
            toolbox_path=tmp_path/'toolbox',diagnostic=True)
    assert not output.exists()


def test_reduced_export_separates_propagation_observers_from_direct_lbm_probes(tmp_path):
    api=analysis_api()
    data,config,summary,boundary,_=_manufactured_hankel_case(height_m=.0002,
        pml_width_m=.00008,probe_radius_m=.000075,
        handoff_contours_m=[[.00085,.00009],[.00090,.00010],[.00091,.00011]],
        validation_probes_m=[[0.,.000115],[-.000915,.00005],[.000915,.00005]])
    result,arrays=api.analyze_data(data,config,summary,boundary)
    assert result['source_validation_passed']
    result['provenance']={'fixture':'manufactured, not LBM','raw_sha256':{'trajectory.npz':'a'*64}}
    original_analysis=json.dumps(result,sort_keys=True)
    original_arrays={key:value.copy() for key,value in arrays.items()}
    output=tmp_path/'reduced_source.mat'
    api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')
    actual=loadmat(output,squeeze_me=True)
    dx=float(actual['dx_m'])
    assert dx==.00005
    sx=np.atleast_1d(actual['expected_grid_observer_x_m'])
    sy=np.atleast_1d(actual['expected_grid_observer_y_m'])
    roles=np.atleast_1d(actual['observer_role'])
    exterior=roles!='interior_cancel'
    width,height=config['strip']['handoff_contours_m'][1]
    requested_gap=np.hypot(np.maximum(abs(np.atleast_1d(actual['observer_x_m'])[:3])-width,0.),
        np.maximum(abs(np.atleast_1d(actual['observer_y_m'])[:3])-height,0.))
    assert np.all(requested_gap>=3*dx-1e-12)
    gap=np.hypot(np.maximum(abs(sx)-width,0.),np.maximum(abs(sy)-height,0.))
    assert np.all(gap[exterior]>=2*dx-1e-12), 'snapped exterior observers overlap or approach the source curve'
    assert roles[:3].tolist()==['exterior_propagation_check']*3
    assert roles[3]=='farfield_45mm' and np.isclose(sy[3],.045)
    np.testing.assert_allclose(sx[4:],[0.,-.0004,.0004],rtol=0,atol=1e-14)
    np.testing.assert_allclose(sy[4:],0.,rtol=0,atol=1e-14)
    reference=np.atleast_1d(actual['expected_grid_kirchhoff_p_hat_Pa'])
    assert np.isfinite(reference).all() and np.all(abs(reference[exterior])>1e-6)
    r=np.hypot(sx[exterior]-.0001,sy[exterior])
    expected=.02*np.exp(.63j)*hankel2(0,2*np.pi*actual['frequency_Hz']/actual['c0_m_s']*r)
    np.testing.assert_allclose(reference[exterior],expected,rtol=.002,atol=1e-8)
    # Independently enumerate all pressure and staggered-velocity deposition
    # stencil indices from the MATLAB adapter, even zero-weight neighbours.
    x0,y0=-float(actual['domain_half_width_m']),-float(actual['domain_half_height_m'])
    observer_indices=set(zip(np.rint((sx[exterior]-x0)/dx).astype(int),
                             np.rint((sy[exterior]-y0)/dx).astype(int)))
    for shift_x,shift_y in ((0.,0.),(.5*dx,0.),(0.,.5*dx)):
        ix=np.floor((actual['x_m']-x0-shift_x)/dx).astype(int)
        iy=np.floor((actual['y_m']-y0-shift_y)/dx).astype(int)
        support={(int(x+i),int(y+j)) for x,y in zip(ix,iy) for i,j in ((0,0),(0,1),(1,0),(1,1))}
        assert observer_indices.isdisjoint(support)
    selected=arrays['validation_probe_indices']
    np.testing.assert_array_equal(actual['nearfield_direct_x_m'],data['probe_x_m'][selected])
    np.testing.assert_array_equal(actual['nearfield_direct_y_m'],data['probe_y_m'][selected])
    np.testing.assert_array_equal(actual['nearfield_direct_p_hat_Pa'],arrays['direct_probe_p_hat_Pa'][selected])
    assert json.loads(str(actual['source_analysis_json']))['provenance']==result['provenance']
    assert 'not direct LBM observations' in str(actual['observer_reference_note'])
    assert json.dumps(result,sort_keys=True)==original_analysis
    for key,value in arrays.items():
        np.testing.assert_array_equal(value,original_arrays[key])


def test_export_keeps_exterior_positions_already_clear_of_source_deposition(tmp_path):
    api=analysis_api()
    probes=[[0.,.0004],[-.0009,.0001],[.0009,.0001]]
    data,config,summary,boundary,_=_manufactured_hankel_case(pml_width_m=0.,
        validation_probes_m=probes)
    result,arrays=api.analyze_data(data,config,summary,boundary)
    assert result['source_validation_passed']
    output=tmp_path/'clear_source.mat'
    api.export_kwave_input(output,result,arrays,toolbox_path=tmp_path/'toolbox')
    actual=loadmat(output,squeeze_me=True)
    np.testing.assert_array_equal(np.column_stack((actual['observer_x_m'][:3],
        actual['observer_y_m'][:3])),probes)


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


@pytest.mark.parametrize('custom_valid',[False,True])
def test_custom_boundary_file_is_the_evidence_actually_bound(tmp_path,hankel_case,custom_valid):
    """Default and selected proofs disagree; the selected proof must decide."""
    from Compute.finite_strip.run import compare_batch
    from verification.test_strip_boundary_gate import write_pulse
    api=analysis_api()
    data,config,summary,_,_=hankel_case
    # Saved manufactured pulses only: these fixtures test provenance, not LBM.
    for name,w,h in [('pulse_candidate_dx10um',.002,.0005),
        ('pulse_reference_dx10um',.0042,.0021),('pulse_sides_dx10um',.002,.002),
        ('pulse_top_dx10um',.004,.0005)]:
        write_pulse(tmp_path,name,w,h,0.)
    compare_batch(tmp_path,dict(validation=dict(pulse_return_start_s=1.8e-6,pulse_return_error_limit=.01)))
    default=tmp_path/'final/boundary_validation.json'
    selected=tmp_path/'final/selected_boundary_proof.json'
    proof=json.loads(default.read_text(encoding='utf-8'))
    if custom_valid:
        selected.write_text(json.dumps(proof),encoding='utf-8')
        proof['boundary_validation_passed']=False
        default.write_text(json.dumps(proof),encoding='utf-8')
    else:
        proof['pulse_provenance']['combined']['sha256']['trajectory.npz']='0'*64
        selected.write_text(json.dumps(proof),encoding='utf-8')
    case=tmp_path/'raw/manufactured_only'; case.mkdir()
    (case/'effective.yaml').write_text(yaml.safe_dump(config),encoding='utf-8')
    (case/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (case/'manifest.json').write_text(json.dumps(dict(status='completed',source_id='test-source',
        fixture='manufactured, not LBM')),encoding='utf-8')
    np.savez_compressed(case/'trajectory.npz',**data)
    result,_=api.analyze_saved(case,selected)
    assert result['checks']['boundary_evidence_bound'] is custom_valid
    assert result['source_validation_passed'] is custom_valid
    assert result['provenance']['boundary_path']==str(selected.resolve())
    assert result['provenance']['boundary_sha256']==api._sha256(selected)
