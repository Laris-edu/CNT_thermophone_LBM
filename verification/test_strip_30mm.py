"""Wide-strip export and backend contracts; synthetic fields are not validation."""
import numpy as np
import pytest
from scipy.io import loadmat

from Compute.finite_strip.analyze import export_kwave_input
from Compute.finite_strip.chain import validate_propagation
from farfield.strip_surface import upper_rectangle
from verification.test_strip_chain import fixture as propagation_fixture


def _export_fixture(half_width=.0154, outer_probe=.015415):
    surfaces = [upper_rectangle(half_width+d, h, 50e-6)
                for d, h in ((-.00005, .00009), (0., .00010), (.00001, .00011))]
    sizes = np.array([len(s['x_m']) for s in surfaces])
    ends = np.cumsum(sizes)
    arrays = {'surface_'+key: np.concatenate([s[key] for s in surfaces])
              for key in surfaces[0]}
    arrays.update(surface_slices=np.column_stack((np.r_[0, ends[:-1]], ends)),
                  surface_p_hat_Pa=np.full(ends[-1], .01+.02j),
                  surface_ux_hat_m_s=np.zeros(ends[-1], complex),
                  surface_uy_hat_m_s=np.full(ends[-1], .0001j),
                  validation_probe_indices=np.arange(3),
                  probe_x_m=np.array([0., -outer_probe, outer_probe]),
                  probe_y_m=np.array([.000115, .00005, .00005]),
                  direct_probe_p_hat_Pa=np.full(3, .01+.02j))
    result = dict(source_validation_passed=True, checks={'fixture_gate': True},
                  frequency_Hz=20000., air={'rho0_kg_m3': 1.16, 'c0_m_s': 347.})
    return result, arrays


@pytest.mark.parametrize('probe', [.015415, .022])
def test_wide_export_covers_curve_and_exterior_observers(tmp_path, probe):
    result, arrays = _export_fixture(outer_probe=probe)
    path = tmp_path/'source.mat'
    export_kwave_input(path, result, arrays, toolbox_path=tmp_path)
    saved = loadmat(path, squeeze_me=True)
    half_width, dx = saved['domain_half_width_m'], saved['dx_m']
    assert half_width >= .0185-1e-14
    assert half_width >= np.max(abs(saved['expected_grid_observer_x_m']))+dx-1e-14
    assert half_width/dx == pytest.approx(round(half_width/dx))
    assert np.max(abs(saved['x_m']))+dx < half_width
    np.testing.assert_array_equal(saved['nearfield_direct_x_m'], arrays['probe_x_m'])


def test_export_records_explicit_gpu_request_without_changing_source(tmp_path):
    result, arrays = _export_fixture()
    cpu, gpu = tmp_path/'cpu.mat', tmp_path/'gpu.mat'
    export_kwave_input(cpu, result, arrays, toolbox_path=tmp_path)
    export_kwave_input(gpu, result, arrays, toolbox_path=tmp_path, kwave_backend='matlab-gpu')
    first, second = (loadmat(path, squeeze_me=True) for path in (cpu, gpu))
    assert first['kwave_backend_requested'] == 'cpu'
    assert second['kwave_backend_requested'] == 'matlab-gpu'
    for name in ('p_hat_Pa', 'un_hat_m_s', 'x_m', 'y_m', 'ds_m',
                 'expected_grid_kirchhoff_p_hat_Pa'):
        np.testing.assert_array_equal(first[name], second[name])


def test_export_rejects_unknown_backend_before_writing(tmp_path):
    result, arrays = _export_fixture()
    path = tmp_path/'invalid.mat'
    with pytest.raises(ValueError, match='backend'):
        export_kwave_input(path, result, arrays, toolbox_path=tmp_path, kwave_backend='fallback')
    assert not path.exists()


def test_propagation_gate_rejects_silent_cpu_fallback():
    source, output = propagation_fixture()
    source['kwave_backend_requested'] = 'matlab-gpu'
    output.update(kwave_backend_requested='matlab-gpu', kwave_backend_effective='cpu')
    with pytest.raises(ValueError, match='backend'):
        validate_propagation(source, output)


def test_propagation_gate_reports_effective_gpu_and_provenance():
    source, output = propagation_fixture()
    source['kwave_backend_requested'] = 'matlab-gpu'
    output.update(kwave_backend_requested='matlab-gpu', kwave_backend_effective='matlab-gpu',
                  gpu_provenance_json='{"enabled":true,"name":"fixture GPU"}')
    result = validate_propagation(source, output)
    assert result['kwave_backend_effective'] == 'matlab-gpu'
    assert result['gpu_provenance'] == {'enabled': True, 'name': 'fixture GPU'}
    assert result['propagation_validation_passed']


def test_chain_routes_explicit_gpu_request_to_export_and_preflight(tmp_path, monkeypatch):
    import json
    from scipy.io import savemat
    from verification.test_strip_chain import _saved_stage

    chain, root, toolbox, args = _saved_stage(tmp_path, monkeypatch)
    source, output = propagation_fixture()
    source['observer_role'] = source['observer_role'].astype(object)
    source['kwave_backend_requested'] = 'matlab-gpu'
    output.update(kwave_backend_requested='matlab-gpu', kwave_backend_effective='matlab-gpu',
                  gpu_provenance_json='{"enabled":true,"name":"fixture GPU"}')

    def analyze(argv):
        assert argv[argv.index('--kwave-backend')+1] == 'matlab-gpu'
        (root/'final').mkdir()
        savemat(root/'final/power_r2_kwave_input.mat', source)
        return 0

    def matlab(command, **kwargs):
        if 'kwave_handoff(' in command[-1]:
            savemat(root/'final/power_r2_kwave_output.mat', output)
        else:
            assert 'gpuDevice' in command[-1]
            assert 'Distrib_Computing_Toolbox' in command[-1]

    monkeypatch.setattr(chain, 'analyze_main', analyze)
    monkeypatch.setattr(chain.subprocess, 'run', matlab)
    assert chain.main(args+['--kwave-backend', 'matlab-gpu']) == 0
    saved = json.loads((root/'final/propagation_validation.json').read_text())
    assert saved['kwave_backend_requested'] == saved['kwave_backend_effective'] == 'matlab-gpu'
    assert saved['gpu_provenance']['enabled'] is True
