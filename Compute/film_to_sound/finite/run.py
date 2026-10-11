"""LBM outgoing harmonic -> free-space finite monopole sheet -> k-Wave CW.

The effective source is uniform over the user-selected square. Its strength
comes from LBM, not from the reference integral. The latter independently
checks propagation of the SAME assumed source, not full device accuracy.
All complex amplitudes use exp(+i omega t), retaining the original clock.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.io import loadmat
import yaml

from Compute.film_to_sound.analyze import analyze_batch, sha256
from core.unit_mapping import create_unit_mapping

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def pack(value):
    value = np.asarray(value)
    return dict(re=value.real.tolist(), im=value.imag.tolist(),
                amplitude=np.abs(value).tolist(), phase_deg=np.angle(value, deg=True).tolist())


def unpack(value):
    return np.asarray(value['re']) + 1j*np.asarray(value['im'])


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def tree_hash(directory):
    files = {str(p.relative_to(directory)): sha256(p) for p in sorted(directory.rglob('*.m'))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return dict(sha256=digest, files=files)


def prepare(config_path):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    source_batch = (config_path.parent/cfg['source_batch']).resolve()
    toolbox = (config_path.parent/cfg['toolbox_path']).resolve()
    if not (toolbox/'acousticFieldPropagator.m').is_file():
        raise ValueError('k-Wave acousticFieldPropagator.m is missing')
    if not Path(cfg['matlab_executable']).is_file():
        raise ValueError('MATLAB executable is missing')
    source_result = analyze_batch(source_batch)
    if not source_result['propagation_validation_passed']:
        raise ValueError('upstream LBM validation or propagation checks failed')
    raw_cfg = yaml.safe_load((source_batch/'raw/power_f20k_r2/effective.yaml').read_text())
    ph = create_unit_mapping(raw_cfg).physical
    film = cfg['film']
    if (film['radiation'] != 'free_space_symmetric_monopole_sheet'
            or film['amplitude_profile'] != 'uniform' or film['phase_profile'] != 'uniform'
            or cfg['medium'] != 'homogeneous_lossless'):
        raise ValueError('this study supports only the stated uniform free-space CW model')
    if not np.isclose(film['width_m']*film['height_m'], raw_cfg['run']['power']['area_m2'], rtol=1e-12):
        raise ValueError('finite-film area must equal the LBM power normalization area')
    if raw_cfg['run']['film']['gas_flux_factor'] != 2:
        raise ValueError('requires symmetric two-sided LBM film balance')
    probe = next(p for p in source_result['probes'] if p['nominal_z_m'] == cfg['handoff_nominal_m'])
    frequency = source_result['frequency_Hz']
    k = 2*np.pi*frequency/ph.c0_m_s
    source_pressure = complex(unpack(probe['outgoing']))*np.exp(1j*k*probe['z_m'])
    source = dict(film_width_m=film['width_m'], film_height_m=film['height_m'],
        frequency_Hz=frequency, c0_m_s=ph.c0_m_s, rho0_kg_m3=ph.rho0_kg_m3,
        handoff_z_m=probe['z_m'], handoff_outgoing_pressure=probe['outgoing'],
        source_pressure=pack(source_pressure), input_power=source_result['channels']['power_W_m2'],
        volume_injection_per_area_m_s=pack(2*source_pressure/(ph.rho0_kg_m3*ph.c0_m_s)),
        film_heat_capacity_assumed_J_m2K=raw_cfg['run']['film']['C_A_J_m2K'],
        phase_convention='exp(+i omega t)', source_batch=str(source_batch),
        upstream_status=source_result['power_case']['status'],
        upstream_validation_passed=source_result['power_case']['validation_passed'],
        upstream_propagation_validation_passed=source_result['propagation_validation_passed'],
        upstream_raw_sha256=source_result['raw_sha256'])
    inputs = dict(frequency_Hz=frequency, c0_m_s=ph.c0_m_s, rho0_kg_m3=ph.rho0_kg_m3,
        source_re=float(source_pressure.real), source_im=float(source_pressure.imag),
        width_m=film['width_m'], height_m=film['height_m'], toolbox_path=str(toolbox), cases=cfg['cases'])
    # Check both exchange formats before creating any output directory.
    yaml.safe_dump(dict(source=inputs, study=cfg), sort_keys=False)
    json.dumps(dict(source=source, inputs=inputs), allow_nan=False)
    return cfg, source, inputs, toolbox


def reference_pressure(points, source, order):
    """Independent Gauss-Legendre surface integral of the free-space Green kernel."""
    nodes, weights = leggauss(order)
    sx = nodes*source['film_width_m']/2
    sy = nodes*source['film_height_m']/2
    sx, sy = np.meshgrid(sx, sy, indexing='ij')
    area_weights = np.outer(weights, weights)*source['film_width_m']*source['film_height_m']/4
    k = 2*np.pi*source['frequency_Hz']/source['c0_m_s']
    strength = 1j*k*complex(unpack(source['source_pressure']))/(2*np.pi)
    output = []
    for x, y, z in np.asarray(points):
        radius = np.sqrt((x-sx)**2+(y-sy)**2+z*z)
        if np.any(radius == 0):
            raise ValueError('reference integral is not evaluated on the source sheet')
        output.append(strength*np.sum(area_weights*np.exp(-1j*k*radius)/radius))
    return np.asarray(output)


def compare_arrays(label, got, expected, limits):
    got, expected = np.asarray(got), np.asarray(expected)
    resolved = (got.shape == expected.shape and got.size > 0
        and np.isfinite(got).all() and np.isfinite(expected).all()
        and np.all(abs(expected) > 1e-10) and np.all(abs(got) > 1e-10))
    if not resolved:
        return dict(label=label, amplitude_error_percent=None, phase_error_deg=None,
                    passed=False, reason='unresolved or mismatched signals')
    amp = float(np.max(abs(100*(abs(got)/abs(expected)-1))))
    phase = float(np.max(abs(np.angle(got/expected, deg=True))))
    return dict(label=label, amplitude_error_percent=amp, phase_error_deg=phase,
        sample_count=int(got.size), passed=amp <= limits['amplitude_percent'] and phase <= limits['phase_deg'])


def nearest(axis, target):
    index = int(np.argmin(abs(axis-target)))
    if abs(axis[index]-target) > 1e-10:
        raise ValueError(f'coordinate {target} is not a grid node')
    return index


def axis_data(data):
    return data['pressure_Pa'][nearest(data['x_m'], 0), nearest(data['y_m'], 0), :]


def analyze(batch, cfg, source):
    if (batch/'final/finite_film.json').exists():
        raise FileExistsError('refusing to overwrite an existing finite-film analysis')
    data = {}
    for case in cfg['cases']:
        raw = batch/'raw'/case['name']
        manifest = json.loads((raw/'manifest.json').read_text())
        if manifest.get('study_configuration') != cfg or manifest.get('upstream') != source:
            raise ValueError('current study/source does not match saved case provenance')
        if (manifest['status'] != 'completed' or sha256(raw/'trajectory.npz') != manifest['trajectory_sha256']
                or sha256(raw/'effective.yaml') != manifest['effective_sha256']):
            raise ValueError(f'case provenance mismatch: {case["name"]}')
        with np.load(raw/'trajectory.npz') as archive:
            data[case['name']] = {key: archive[key] for key in archive.files}
    fine = data['finite_fine']; z = fine['z_m']; limits = cfg['validation']
    selected = (z >= limits['minimum_axial_distance_m']-1e-12) & (z <= cfg['target_m']+1e-12)
    za = z[selected]; pa = axis_data(fine)[selected]
    x = fine['x_m']
    lateral = fine['pressure_Pa'][:, nearest(fine['y_m'], 0), nearest(z, cfg['target_m'])]
    points = np.concatenate([np.column_stack([np.zeros_like(za), np.zeros_like(za), za]),
        np.column_stack([x, np.zeros_like(x), np.full_like(x, cfg['target_m'])])])
    low_order, high_order = limits['quadrature_orders']
    ref_low = reference_pressure(points, source, low_order)
    ref = reference_pressure(points, source, high_order)
    measured = np.concatenate([pa, lateral])
    comparisons = [compare_arrays('Green integral', measured, ref, limits),
                   compare_arrays('Quadrature', ref_low, ref, limits)]
    # Compare grids only at shared physical nodes: 3 mm multiples for 1 and .75 mm.
    common_z = np.arange(6, 46, 3)*1e-3
    fine_common = axis_data(fine)[[nearest(z, zz) for zz in common_z]]
    coarse = data['finite_coarse']
    comparisons.append(compare_arrays('Grid spacing',
        axis_data(coarse)[[nearest(coarse['z_m'], zz) for zz in common_z]], fine_common, limits))
    for name, label in [('finite_time', 'Evaluation time'), ('finite_padding', 'FFT padding')]:
        check = data[name]
        check_values = np.concatenate([axis_data(check)[selected],
            check['pressure_Pa'][:, nearest(check['y_m'], 0), nearest(check['z_m'], cfg['target_m'])]])
        comparisons.append(compare_arrays(label, check_values, measured, limits))
    lower = axis_data(fine)[[nearest(z, -zz) for zz in za]]
    comparisons.append(compare_arrays('Two-side symmetry', lower, pa, limits))
    plane = data['plane_check']; zp = plane['z_m']
    select_p = abs(zp) >= limits['minimum_axial_distance_m']-1e-12
    k = 2*np.pi*source['frequency_Hz']/source['c0_m_s']
    expected = complex(unpack(source['source_pressure']))*np.exp(-1j*k*abs(zp[select_p]))
    comparisons.append(compare_arrays('Plane transfer', plane['pressure_Pa'].ravel()[select_p], expected, limits))
    case_areas = {name: float(d['source_area_m2']) for name,d in data.items() if name != 'plane_check'}
    area_ok = all(np.isclose(area, source['film_width_m']*source['film_height_m'], rtol=1e-12) for area in case_areas.values())
    source_dir = Path(source['source_batch'])
    unchanged = all(sha256(source_dir/'raw'/case/file) == digest
        for case, files in source['upstream_raw_sha256'].items() for file,digest in files.items())
    gates = {row['label']: row['passed'] for row in comparisons}
    gates.update(source_area=bool(area_ok), upstream_raw_unchanged=unchanged,
                 upstream_validation=source['upstream_validation_passed'],
                 upstream_propagation=source['upstream_propagation_validation_passed'])
    target = axis_data(fine)[nearest(z, cfg['target_m'])]
    output = dict(status='completed', validation_passed=all(gates.values()),
        finite_propagation_validation_passed=all(gates.values()), source=source, limits=limits,
        gates=gates, metrics=dict(comparisons=comparisons, source_areas_m2=case_areas),
        target=dict(z_m=cfg['target_m'], pressure=pack(target), reference=pack(ref[len(za)-1]),
            rms_Pa=float(abs(target)/np.sqrt(2)), SPL_dB_re_20uPa=float(20*np.log10(abs(target)/np.sqrt(2)/20e-6))),
        curves=dict(axis_z_m=za.tolist(), kwave_axis_pressure=pack(pa), reference_axis_pressure=pack(ref[:len(za)]),
            lateral_x_m=x.tolist(), kwave_lateral_pressure=pack(lateral), reference_lateral_pressure=pack(ref[len(za):])),
        cases={case['name']: json.loads((batch/'raw'/case['name']/'summary.json').read_text()) for case in cfg['cases']},
        raw_sha256={case['name']: {name: sha256(batch/'raw'/case['name']/name)
            for name in ('effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json')} for case in cfg['cases']},
        scope='Uniform in-phase 30 x 30 mm zero-thickness monopole sheet, free-space symmetric radiation; '
              'one-way coupling from planar LBM; homogeneous lossless air; no edge heat transfer, supports, '
              'finite-film acoustic feedback or experimental device validation. Film heat capacity is assumed.')
    dump(batch/'final/finite_film.json', output)
    return output


def run_case(batch, case, cfg, inputs, provenance):
    name = case['name']; raw = batch/'raw'/name
    raw.mkdir(parents=True, exist_ok=False)
    effective = dict(case=case, source=inputs, model=cfg['film'], medium=cfg['medium'])
    (raw/'effective.yaml').write_text(yaml.safe_dump(effective, sort_keys=False), encoding='utf-8')
    manifest = dict(status='running', **provenance, effective_sha256=sha256(raw/'effective.yaml'),
                    started_utc=datetime.now(timezone.utc).isoformat())
    dump(raw/'manifest.json', manifest)
    mat_path = batch/'_work'/f'{name}.mat'
    def mq(value):
        return str(value).replace("'", "''")
    command = f"addpath('{mq(HERE)}'); kwave_finite('{mq(batch/'_work/input.json')}','{mq(mat_path)}','{name}');"
    started = time.perf_counter()
    print(f'Running {name}: dx={case["dx_m"]*1e3:g} mm, cycles={case["evaluation_cycles"]}', flush=True)
    proc = subprocess.run([cfg['matlab_executable'], '-wait', '-batch', command], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace',
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    (batch/'_work'/f'{name}.log').write_text(proc.stdout, encoding='utf-8')
    if proc.returncode:
        manifest.update(status='failed', returncode=proc.returncode)
        dump(raw/'manifest.json', manifest)
        dump(raw/'summary.json', dict(status='failed', validation_passed=False))
        raise RuntimeError(f'{name}: MATLAB failed; see {batch / "_work" / (name+".log")}')
    mat = loadmat(mat_path, squeeze_me=True)
    pressure = np.asarray(mat['pressure'])
    arrays = dict(x_m=np.atleast_1d(mat.get('x_m', 0)), y_m=np.atleast_1d(mat.get('y_m', 0)),
        z_m=np.atleast_1d(mat['z_m']), pressure_Pa=pressure,
        source_weights=np.asarray(mat['source_weights']), source_area_m2=np.asarray(mat['source_area_m2']))
    finite = all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(raw/'trajectory.npz', **arrays)
    summary = dict(status='completed', validation_passed=finite, finite_state=finite,
        elapsed_s=float(mat['elapsed_s']), wall_elapsed_s=time.perf_counter()-started,
        dx_m=case['dx_m'], evaluation_time_s=float(mat['evaluation_time_s']),
        grid_expansion_factor=case['grid_expansion_factor'], source_area_m2=float(mat['source_area_m2']),
        shape=list(pressure.shape), matlab_version=str(mat['matlab_version']), kwave_version=str(mat['kwave_version']))
    dump(raw/'summary.json', summary)
    manifest.update(status='completed', trajectory_sha256=sha256(raw/'trajectory.npz'))
    dump(raw/'manifest.json', manifest)
    print(f'Completed {name}: {summary["elapsed_s"]:.2f} s solver, finite={finite}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'study.yaml')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--analyze-only', type=Path)
    args = parser.parse_args()
    cfg, source, inputs, toolbox = prepare(args.config)
    batch = args.config.resolve().parent/cfg['batch_name']
    if args.preflight:
        print(json.dumps(dict(batch=str(batch), batch_exists=batch.exists(), source=source,
            matlab_input=inputs, version=git('rev-parse', 'HEAD')), indent=2))
        return
    if args.analyze_only:
        result = analyze(args.analyze_only.resolve(), cfg, source)
    else:
        batch.mkdir(exist_ok=False)
        (batch/'_work').mkdir(); (batch/'final').mkdir()
        dump(batch/'_work/input.json', inputs)
        scripts = {p.name: sha256(p) for p in HERE.iterdir() if p.suffix in ('.py', '.m', '.yaml')}
        provenance = dict(git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')),
            study_clean=not bool(git('status', '--porcelain', '--', str(HERE))),
            study_sha256=scripts, study_configuration=cfg, kwave_source=tree_hash(toolbox), upstream=source)
        for case in cfg['cases']:
            run_case(batch, case, cfg, inputs, provenance)
        result = analyze(batch, cfg, source)
    print(json.dumps(dict(status=result['status'], validation_passed=result['validation_passed'],
        target=result['target'], comparisons=result['metrics']['comparisons']), indent=2))


if __name__ == '__main__':
    main()
