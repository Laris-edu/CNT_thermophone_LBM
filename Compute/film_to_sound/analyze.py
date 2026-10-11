"""Read-only film/power fits and lossless planar outgoing-wave propagation."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from postproc.diagnostics import compare, fit_window, record
from simulation import analyze_case
from simulation.study import validate_effective

ROOT = Path(__file__).resolve().parents[2]
FILES = ('effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json')
SOURCES_M = (.00012, .00016, .00020)
TARGETS_M = (.00030, .00060)
LIMITS = dict(amplitude_percent=1., phase_deg=1., incoming_ratio=.01)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _hashes(path):
    return {name: sha256(Path(path) / name) for name in FILES}


def _load(path, mode):
    path = Path(path).resolve()
    if not all((path / name).is_file() for name in FILES):
        raise ValueError(f'missing four-file raw contract: {path}')
    config = yaml.safe_load((path / 'effective.yaml').read_text(encoding='utf-8'))
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'completed' or summary.get('status') != 'completed':
        raise ValueError(f'raw run is not completed: {path}')
    if config['run']['mode'] != mode or 'grid' in config:
        raise ValueError(f'requires a uniform {mode} run: {path}')
    if sha256(path / 'effective.yaml') != manifest.get('effective_sha256'):
        raise ValueError(f'effective configuration hash changed: {path}')
    validate_effective(config)
    with np.load(path / 'trajectory.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if not bool(data.get('run_completed', False)) or not all(
            np.isfinite(a).all() for a in data.values() if a.dtype.kind in 'biufc'):
        raise ValueError(f'incomplete or nonfinite raw arrays: {path}')
    mapping = create_unit_mapping(config); dt = mapping.lattice.dt_s
    t = data['t_s']; n = len(t)
    if (n == 0 or t.ndim != 1 or not np.allclose(t, np.arange(1, n+1)*dt, rtol=0, atol=dt*1e-5)
            or data['audit'].shape != (n, 6) or summary.get('steps') != n):
        raise ValueError(f'raw time/audit/summary shape mismatch: {path}')
    if mode == 'pulse' and n != config['numerics']['steps']:
        raise ValueError('pulse trajectory does not cover the full planned duration')
    if mode == 'pulse' and any(data[key].shape != (n,) for key in ('pressure_Pa', 'outgoing_Pa', 'incoming_Pa')):
        raise ValueError('pulse pressure/decomposition arrays do not share endpoint times')
    if mode == 'power':
        count = len(config['run']['probes_m'])
        for key in ('pressure_Pa', 'velocity_m_s'):
            if data[key].shape != (n, count):
                raise ValueError(f'invalid {key} shape')
        for key in ('power_W_m2', 'film_temperature_K', 'wall_flux_W_m2', 'heat_time_s'):
            if data[key].shape != (n,):
                raise ValueError(f'invalid {key} shape')
        if not np.allclose(data['heat_time_s'], t-dt/2, rtol=0, atol=dt*1e-5):
            raise ValueError('power and wall flux must use interval-midpoint timestamps')
        if (n > config['numerics']['steps'] or (n < config['numerics']['steps']
                and str(data.get('stop_reason')) != 'periodic_stable') or t[-1]*config['run']['frequency_Hz'] < 2):
            raise ValueError('power trajectory does not contain a completed two-cycle window')
        if data['probe_z_m'].shape != (count,):
            raise ValueError('invalid probe coordinate shape')
    z = (np.arange(config['numerics']['ny'])+.5)*mapping.lattice.dx_m
    nominal = config['run']['probes_m'] if mode == 'power' else [config['run']['pulse']['probe_fraction']*len(z)*mapping.lattice.dx_m]
    expected = np.array([z[np.argmin(abs(z-p))] for p in nominal])
    if not np.allclose(np.atleast_1d(data['probe_z_m']), expected, rtol=0, atol=1e-15):
        raise ValueError('saved probe positions disagree with the effective grid')
    if 'z_m' in data and not np.allclose(data['z_m'], z, rtol=0, atol=1e-15):
        raise ValueError('saved grid coordinates disagree with the effective grid')
    if (not np.isclose(summary['actual_length_m'], len(z)*mapping.lattice.dx_m, rtol=0, atol=1e-15)
            or ('actual_length_m' in data and not np.isclose(float(data['actual_length_m']), summary['actual_length_m'], rtol=0, atol=1e-15))):
        raise ValueError('raw/summary domain length disagrees with the effective grid')
    return config, manifest, summary, data


def load_power_case(batch):
    return _load(Path(batch) / 'raw/power_f20k_r2', 'power')


def _provenance(manifest):
    hashes = manifest.get('source_sha256', {})
    source_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode('utf-8')).hexdigest()
    return bool(hashes and manifest.get('git_commit') and manifest.get('source_clean')
                and manifest.get('source_id') == source_id
                and all((ROOT/name).is_file() and sha256(ROOT/name) == digest for name, digest in hashes.items()))


def _public(path, summary):
    recomputed = analyze_case(path)
    mismatches = [key for key, value in recomputed.items() if summary.get(key) != value]
    consistent = not mismatches
    return dict(saved_validation_passed=summary['validation_passed'],
                recomputed_validation_passed=recomputed['validation_passed'], consistent=consistent,
                summary_mismatches=mismatches,
                saved_gates=summary.get('gates', {}), recomputed_gates=recomputed.get('gates', {}),
                passed=bool(summary['validation_passed'] and recomputed['validation_passed'] and consistent))


def _metadata(path, manifest, summary):
    return dict(path=str(Path(path).resolve()), status=manifest['status'],
                validation_passed=summary['validation_passed'],
                **{key: manifest.get(key) for key in ('git_commit', 'git_dirty', 'source_clean', 'source_id')})


def pulse_preflight(batch):
    path = Path(batch) / 'raw/pulse_r2'; before = _hashes(path)
    config, manifest, summary, data = _load(path, 'pulse')
    public = _public(path, summary); a = config['run']['analysis']; audit = data['audit']
    metrics = dict(reflection_ratio=float(summary['reflection_ratio']),
        max_mass_residual_lu=float(np.max(abs(audit[:, 4]))),
        max_energy_residual_J_m2=float(np.max(abs(audit[:, 5]))))
    gates = dict(public_validation=public['passed'], finite_raw=True,
        reflection=metrics['reflection_ratio'] < min(.01, a['reflection_ratio']),
        step_mass=metrics['max_mass_residual_lu'] < a['mass_residual_lu'],
        step_gas_energy=metrics['max_energy_residual_J_m2'] < a['energy_residual_J_m2'],
        public_source_matches=_provenance(manifest))
    for name, state, initial, columns, threshold in (
            ('mass', 'mass_lu', 'initial_mass_lu', (0, 1), 'cumulative_mass_lu'),
            ('gas_energy', 'gas_energy_J_m2', 'initial_gas_energy_J_m2', (2, 3), 'cumulative_energy_J_m2')):
        available = state in data and initial in data
        metrics[f'independent_cumulative_{name}_available'] = available
        if available:
            value = float(np.max(abs(data[state]-float(data[initial])-np.cumsum(audit[:, columns].sum(axis=1)))))
            metrics[f'cumulative_{name}_residual'] = value
            gates[f'independent_cumulative_{name}'] = value < a[threshold]
    gates['raw_files_unchanged'] = before == _hashes(path)
    return dict(**_metadata(path, manifest, summary), public_validation=public, **metrics,
                cumulative_assessed=all(metrics[f'independent_cumulative_{name}_available'] for name in ('mass', 'gas_energy')),
                gates=gates, preflight_passed=all(gates.values()), raw_sha256=before)


def _failed(error):
    return dict(re=None, im=None, amplitude=None, phase_deg=None, resolved=False, error=str(error))


def _fit(t, x, frequency, window):
    try:
        t, x = np.asarray(t), np.asarray(x)
        mask = (t*frequency >= window[0]-1e-10) & (t*frequency < window[1]-1e-10)
        values = x[mask]
        if len(values) < 10 or not np.isfinite(values).all():
            raise ValueError('insufficient or nonfinite fit samples')
        with np.errstate(divide='ignore', invalid='ignore'):
            hat, quality = fit_window(t, x, frequency, *window, harmonics=3, detrend_order=1)
        resolution = 256*np.finfo(float).eps*max(1., float(np.max(abs(values))))
        if not np.isfinite(hat) or abs(hat) <= resolution or quality['condition_number'] > 1e8:
            raise ValueError('zero/unresolved harmonic or degenerate fit')
        if not all(np.isfinite(value) for value in quality.values()):
            raise ValueError('nonfinite fit quality')
        return hat, dict(**record(hat), resolved=True, fit=quality)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError) as exc:
        return None, _failed(exc)


def decompose(pressure, velocity, impedance):
    return (pressure+impedance*velocity)/2, (pressure-impedance*velocity)/2


def propagate(outgoing, frequency, distance, sound_speed):
    distance = np.asarray(distance)
    if np.any(distance < 0):
        raise ValueError('propagation target must lie above the source')
    return outgoing*np.exp(-2j*np.pi*frequency*distance/sound_speed)


def checked(got, expected):
    if got is None or expected is None or abs(expected) == 0 or abs(got) == 0:
        return dict(amplitude_error_percent=None, phase_error_deg=None,
                    complex_relative_error=None, passed=False, error='missing/unresolved harmonic')
    item = compare(got, expected)
    item['passed'] = bool(abs(item['amplitude_error_percent']) <= LIMITS['amplitude_percent']
                          and abs(item['phase_error_deg']) <= LIMITS['phase_deg'])
    return item


def _record(hat):
    return dict(**record(hat), resolved=True) if hat is not None else _failed('missing/unresolved harmonic')


def analyze_batch(batch):
    batch = Path(batch).resolve(); path = batch / 'raw/power_f20k_r2'; before = _hashes(path)
    config, manifest, summary, data = load_power_case(batch)
    pulse = pulse_preflight(batch)
    pc, pm, _, _ = _load(batch / 'raw/pulse_r2', 'pulse')
    compatible = all(pc[key] == config[key] for key in ('physical', 'lattice', 'collision', 'transport'))
    compatible &= {k:v for k,v in pc['numerics'].items() if k != 'steps'} == {k:v for k,v in config['numerics'].items() if k != 'steps'}
    compatible &= all(pc['run'][key] == config['run'][key] for key in ('refine', 'length_m'))
    compatible &= pm['source_sha256'] == manifest['source_sha256'] and pm['git_commit'] == manifest['git_commit']
    public = _public(path, summary); ph = create_unit_mapping(config).physical
    frequency = config['run']['frequency_Hz']; end = int(np.floor(data['t_s'][-1]*frequency+1e-10))
    window = (end-2, end); channels = {}; hats = {}; probes = []
    for name, time in (('power_W_m2', 'heat_time_s'), ('film_temperature_K', 't_s'), ('wall_flux_W_m2', 'heat_time_s')):
        _, channels[name] = _fit(data[time], data[name], frequency, window)
    for j, nominal in enumerate(config['run']['probes_m']):
        p, pr = _fit(data['t_s'], data['pressure_Pa'][:, j], frequency, window)
        u, ur = _fit(data['t_s'], data['velocity_m_s'][:, j], frequency, window)
        outgoing, incoming = decompose(p, u, ph.rho0_kg_m3*ph.c0_m_s) if p is not None and u is not None else (None, None)
        resolved = outgoing is not None and abs(outgoing) > 256*np.finfo(float).eps*max(1., abs(p))
        if not resolved:
            outgoing = None
        ratio = float(abs(incoming/outgoing)) if resolved else None
        hats[nominal] = (p, outgoing, float(data['probe_z_m'][j]))
        probes.append(dict(nominal_z_m=nominal, z_m=float(data['probe_z_m'][j]), pressure=pr,
            velocity=ur, outgoing=_record(outgoing), incoming=_record(incoming), incoming_over_outgoing=ratio))
    propagation = []; predictions = {}
    for source in SOURCES_M:
        _, outgoing, source_z = hats.get(source, (None, None, None))
        for target in TARGETS_M:
            direct, _, target_z = hats.get(target, (None, None, None))
            predicted = propagate(outgoing, frequency, target_z-source_z, ph.c0_m_s) if outgoing is not None and target_z is not None else None
            predictions[source, target] = predicted
            propagation.append(dict(source_nominal_z_m=source, source_z_m=source_z,
                target_nominal_z_m=target, target_z_m=target_z, predicted=_record(predicted),
                direct=_record(direct), comparison=checked(predicted, direct)))
    sensitivity = [dict(source_nominal_z_m=i, source_z_m=hats.get(i, (None, None, None))[2],
        reference_source_nominal_z_m=j, reference_source_z_m=hats.get(j, (None, None, None))[2],
        target_nominal_z_m=.00060, target_z_m=hats.get(.00060, (None, None, None))[2],
        comparison=checked(predictions[i, .00060], predictions[j, .00060])) for i,j in combinations(SOURCES_M, 2)]
    _, primary, z = hats.get(.00020, (None, None, None))
    exterior = propagate(primary, frequency, .003-z, ph.c0_m_s) if primary is not None else None
    ratios = [p['incoming_over_outgoing'] for p in probes if p['nominal_z_m'] in SOURCES_M]
    gates = dict(public_validation=public['passed'], pulse_preflight=pulse['preflight_passed'],
        pulse_power_compatible=bool(compatible), public_source_matches=_provenance(manifest),
        resolved_harmonics=all(c['resolved'] for c in channels.values()) and len(hats) >= 5
            and all(n in hats and hats[n][0] is not None and hats[n][1] is not None for n in (*SOURCES_M, *TARGETS_M)),
        incoming_over_outgoing=len(ratios) == 3 and all(r is not None and r <= .01 for r in ratios),
        propagation_vs_direct_total=all(row['comparison']['passed'] for row in propagation),
        interface_sensitivity=all(row['comparison']['passed'] for row in sensitivity),
        raw_files_unchanged=before == _hashes(path) and pulse['raw_sha256'] == _hashes(batch / 'raw/pulse_r2'))
    return dict(status='completed', frequency_Hz=frequency, window_cycles=list(window),
        phase_convention='exp(+i omega t)', air=dict(rho0_kg_m3=ph.rho0_kg_m3, c0_m_s=ph.c0_m_s),
        power_case=_metadata(path, manifest, summary), public_validation=public, pulse=pulse,
        channels=channels, probes=probes, propagation=propagation, interface_sensitivity=sensitivity,
        extrapolation=dict(source_z_m=z, target_z_m=.003, pressure=_record(exterior),
            description='Lossless planar outgoing-wave model outside the simulated domain; no 1/r spreading. Not directly validated at 3 mm.'),
        limits=LIMITS, gates=gates, propagation_validation_passed=all(gates.values()),
        raw_sha256=dict(power_f20k_r2=before, pulse_r2=pulse['raw_sha256']),
        scope='Uniform r2 open planar, symmetric single-temperature film, local transport and normal incidence. No finite-film device or geometric far-field validation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args(); output = args.batch / 'final/propagation.json'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    result = analyze_batch(args.batch)
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(payload)
    print(json.dumps(dict(status=result['status'], propagation_validation_passed=result['propagation_validation_passed'], gates=result['gates']), indent=2))


if __name__ == '__main__':
    main()
