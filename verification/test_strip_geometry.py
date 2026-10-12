"""Reject invalid sampled geometry before allocating or advancing a gas solver."""
import json

import numpy as np
import pytest

from simulation.finite_strip import collect_strip, prepare_strip_case, run_strip_case


def reduced_geometry(**updates):
    parameters=dict(name='reduced_geometry',kind='power',dx_m=1.305875e-6,
        width_m=.002,height_m=.0002,film_width_m=.001,probe_radius_m=.000075,
        pml_width_m=.00008,sponge_width_m=0.,sponge_rate=0.,
        handoff_contours_m=[[.00085,.00009],[.00090,.00010],[.00091,.00011]],
        validation_probes_m=[[0.,.000115],[-.000915,.00005],[.000915,.00005]])
    parameters.update(updates)
    return parameters


def test_reduced_r2_geometry_is_checked_without_constructing_solver(monkeypatch):
    from core.finite_strip_solver import FiniteStripSolver

    def forbidden(*args,**kwargs):
        raise AssertionError('geometry preparation must not allocate the gas solver')
    monkeypatch.setattr(FiniteStripSolver,'__init__',forbidden)
    c=prepare_strip_case(reduced_geometry())
    assert (c['numerics']['nx'],c['numerics']['ny'])==(1532,153)
    assert c['strip']['handoff_contours_m'][1]==[.00090,.00010]
    assert c['strip']['validation_probes_m']==reduced_geometry()['validation_probes_m']


@pytest.mark.parametrize('updates',[
    {'handoff_contours_m':[[.00085,.00009],[.0009,.0001]]},
    {'handoff_contours_m':[[.00085,.00009],[.0009,.0001],[.00089,.00011]]},
    {'handoff_contours_m':[[.00085,.00009],[.0009,.00009],[.00091,.00011]]},
    {'handoff_contours_m':[[.0005001,.00009],[.0009,.0001],[.00091,.00011]]},
    {'handoff_contours_m':[[.00085,.00009],[.0009,np.nan],[.00091,.00011]]},
    {'validation_probes_m':[[0.,.000105],[-.000915,.00005],[.000915,.00005]]},
    {'validation_probes_m':[[0.,.000115],[-.000915,.00005]]},
    {'validation_probes_m':[[0.,.000115],[-.000915,0.],[.000915,.00005]]},
])
def test_prepare_rejects_invalid_contours_and_validation_probes(updates):
    with pytest.raises(ValueError):
        prepare_strip_case(reduced_geometry(**updates))


@pytest.mark.parametrize('feature',['contour','probe'])
def test_sample_inside_physical_pml_face_is_rejected_if_interpolation_uses_pml(feature):
    p=reduced_geometry()
    dx=p['dx_m'];ny=round(p['height_m']/dx);padding=round(p['pml_width_m']/dx)
    # Geometric point is outside the absorbing layer, but its upper bilinear
    # neighbour is inside. Checking only the nominal coordinate misses this.
    inside_face=(ny-padding)*dx-.25*dx
    if feature=='contour':
        p['handoff_contours_m'][2][1]=inside_face
        p['validation_probes_m'][0]=[.000915,.00005]
    else:
        p['validation_probes_m'][0][1]=inside_face
    with pytest.raises(ValueError,match='interpolation|template'):
        prepare_strip_case(p)


def test_collector_uses_configured_nested_surfaces_and_exterior_probes():
    from core.finite_strip_solver import FiniteStripSolver
    from farfield.strip_surface import upper_rectangle

    c=prepare_strip_case(reduced_geometry(dx_m=5e-6))
    run=dict(c['strip'],steps=1)
    gas=FiniteStripSolver(c,film_width_m=run['film_width_m'],pml_cells=run['pml_cells'])
    result=collect_strip(gas,run)
    np.testing.assert_array_equal(np.column_stack((result['probe_x_m'][13:],
        result['probe_y_m'][13:])),run['validation_probes_m'])
    for index,(width,height) in enumerate(run['handoff_contours_m']):
        expected=upper_rectangle(width,height,2*run['dx_m'])
        low,high=result['surface_slices'][index]
        for name in expected:
            np.testing.assert_array_equal(result['surface_'+name][low:high],expected[name])
    assert result['surface_fields'].shape[-1]==5


def test_old_default_geometry_remains_explicit_and_pulse_ignores_power_surfaces():
    c=prepare_strip_case(dict(name='legacy_geometry',kind='power'))
    assert c['strip']['handoff_contours_m']==[[.00065,.00015],[.0007,.0002],[.00075,.00025]]
    assert c['strip']['validation_probes_m']==[[0.,.00028],[-.00078,.0001],[.00078,.0001]]
    pulse=prepare_strip_case(dict(name='small_pulse',kind='pulse',dx_m=1e-5,
        width_m=.0004,height_m=.00016,probe_radius_m=.00004,
        sponge_width_m=0.,sponge_rate=0.))
    assert pulse['strip']['film_width_m']==0.


@pytest.mark.parametrize('updates',[
    {'pml_width_m':2e-6},
    {'pml_width_m':.0002},
    {'pml_width_m':0.,'sponge_width_m':.00021},
])
def test_prepare_rejects_absorber_dimensions_before_solver_allocation(updates):
    with pytest.raises(ValueError,match='cells'):
        prepare_strip_case(reduced_geometry(**updates))


@pytest.mark.parametrize('failed',[False,True])
def test_manifest_records_runtime_before_and_after_run(tmp_path,monkeypatch,failed):
    import numba
    import core.collision_numba as collision
    import simulation.finite_strip as strip
    import simulation.provenance as provenance

    c=prepare_strip_case(dict(name='runtime',kind='pulse',dx_m=1e-5,
        width_m=.0004,height_m=.00016,probe_radius_m=.00004,
        sponge_width_m=0.,sponge_rate=0.))
    c['strip']['steps']=1
    calls=[]
    actual_capture=provenance.capture_numba_runtime
    def capture(cells):
        result=actual_capture(cells)
        calls.append(result)
        return result
    monkeypatch.setattr(provenance,'capture_numba_runtime',capture)
    monkeypatch.setattr(strip,'capture_source',lambda _: {})
    if failed:
        def fail(*args):
            raise RuntimeError('deliberate collector failure')
        monkeypatch.setattr(strip,'collect_strip',fail)
        with pytest.raises(RuntimeError,match='deliberate collector failure'):
            run_strip_case(c,tmp_path/'runtime')
    else:
        run_strip_case(c,tmp_path/'runtime')
    manifest=json.loads((tmp_path/'runtime/manifest.json').read_text())
    assert manifest['status']==('failed' if failed else 'completed')
    assert len(calls)==2
    assert manifest['runtime_started']==calls[0]
    assert manifest['runtime']==calls[1]
    for runtime in calls:
        assert runtime['cell_count']==40*16
        assert runtime['numba_threads']==numba.get_num_threads()
        assert runtime['parallel_selected']==collision._parallel_field(40*16)
        assert runtime['threading_layer']==numba.threading_layer()
