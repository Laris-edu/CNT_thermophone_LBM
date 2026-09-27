"""Compare prescribed-temperature planar LBM domains without refitting propagation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from postproc.diagnostics import compare, fit_window, record


ROOT = Path(__file__).resolve().parents[2]
WINDOW = (4, 6)
TIME_WINDOWS = ((3, 4), (4, 5), (5, 6))
INTERFACES_M = (0.00008, 0.00012, 0.00016)
TARGETS_M = (0.00030, 0.00060)
AMPLITUDE_PERCENT, PHASE_DEG, REFLECTION_RATIO = 1., 1., .01
TIME_AMPLITUDE_PERCENT, TIME_PHASE_DEG = .1, .1
FILES = ('effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(got, expected, *, temporal=False):
    if abs(expected) == 0:
        raise ValueError('a zero reference amplitude cannot define a relative comparison')
    result = compare(got, expected)
    amplitude = TIME_AMPLITUDE_PERCENT if temporal else AMPLITUDE_PERCENT
    phase = TIME_PHASE_DEG if temporal else PHASE_DEG
    result['passed'] = bool(abs(result['amplitude_error_percent']) <= amplitude
                            and abs(result['phase_error_deg']) <= phase)
    return result


def decompose(pressure, velocity, impedance):
    # SI pressure and normal velocity have the same exp(+i omega t) time origin.
    return (pressure + impedance * velocity) / 2, (pressure - impedance * velocity) / 2


def propagate(outgoing, frequency, distance, sound_speed):
    if distance < 0:
        raise ValueError('handoff targets must lie above their interface')
    return outgoing * np.exp(-2j * np.pi * frequency * distance / sound_speed)


def load_case(path):
    path = Path(path).resolve()
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
    config = yaml.safe_load((path / 'effective.yaml').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or config['run']['mode'] != 'thermal':
        raise ValueError(f'requires a completed prescribed-temperature case: {path}')
    if sha256(path / 'effective.yaml') != manifest['effective_sha256']:
        raise ValueError(f'effective configuration hash changed: {path}')
    keys = ('t_s', 'heat_time_s', 'wall_flux_W_m2', 'first_cell_temperature_K',
            'pressure_Pa', 'velocity_m_s', 'probe_z_m', 'run_completed')
    with np.load(path / 'trajectory.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in keys}
    if (not bool(data['run_completed']) or not all(np.all(np.isfinite(a)) for a in data.values())
            or data['t_s'][-1] * config['run']['frequency_Hz'] < WINDOW[1] - 1e-8):
        raise ValueError(f'incomplete or nonfinite trajectory: {path}')
    return dict(path=path, manifest=manifest, summary=summary, config=config, data=data,
                hashes={name: sha256(path / name) for name in FILES})


def fit_case(case, window, order=1):
    data, config = case['data'], case['config']
    ph = create_unit_mapping(config).physical
    channels = {'wall_flux_W_m2': (data['heat_time_s'], data['wall_flux_W_m2']),
                'first_cell_temperature_K': (data['t_s'], data['first_cell_temperature_K'])}
    for index in range(len(data['probe_z_m'])):
        for field in ('pressure_Pa', 'velocity_m_s'):
            channels[f'{field}_{index}'] = (data['t_s'], data[field][:, index])
    hats, details = {}, {}
    for name, (time, values) in channels.items():
        hats[name], quality = fit_window(time, values, config['run']['frequency_Hz'],
                                         *window, detrend_order=order)
        details[name] = dict(**record(hats[name]), fit=quality)
    for index in range(len(data['probe_z_m'])):
        outgoing, incoming = decompose(hats[f'pressure_Pa_{index}'], hats[f'velocity_m_s_{index}'],
                                       ph.rho0_kg_m3 * ph.c0_m_s)
        hats[f'outgoing_Pa_{index}'] = outgoing
        details[f'outgoing_Pa_{index}'] = record(outgoing)
        details[f'incoming_Pa_{index}'] = record(incoming)
        details[f'incoming_Pa_{index}']['incoming_over_outgoing'] = abs(incoming / outgoing)
    return hats, details


def stationarity(case):
    fits = [fit_case(case, window)[0] for window in TIME_WINDOWS]
    comparisons = [dict(cycles=window, reference_cycles=TIME_WINDOWS[-1],
        channels={name: checked(fit[name], fits[-1][name], temporal=True) for name in fit})
        for window, fit in zip(TIME_WINDOWS[:-1], fits[:-1])]
    sensitivity = []
    for window in (*TIME_WINDOWS, WINDOW):
        linear = fit_case(case, window)[0]
        constant = fit_case(case, window, order=0)[0]
        sensitivity.append(dict(cycles=window, channels={name: checked(constant[name], linear[name],
            temporal=True) for name in linear}))
    return dict(comparisons=comparisons, detrend_sensitivity=sensitivity,
                passed=all(item['passed'] for block in comparisons + sensitivity
                           for item in block['channels'].values()))


def probe_index(case, nominal):
    requested = case['config']['run']['probes_m']
    if nominal not in requested:
        raise ValueError(f'missing nominal probe {nominal}: {case["path"]}')
    return requested.index(nominal)


def metadata(case):
    config, manifest = case['config'], case['manifest']
    mapping = create_unit_mapping(config)
    return dict(case_dir=str(case['path']), status=manifest['status'],
        validation_passed=case['summary']['validation_passed'], public_gates=case['summary']['gates'],
        git_commit=manifest['git_commit'], git_dirty=manifest['git_dirty'],
        source_clean=manifest['source_clean'], source_id=manifest['source_id'],
        raw_sha256=case['hashes'], length_m=config['numerics']['ny'] * mapping.lattice.dx_m,
        ny=config['numerics']['ny'], steps=config['numerics']['steps'],
        cycles=config['run']['cycles'], elapsed_s=manifest['elapsed_s'])


def analyze(reference_dir, short_dirs):
    reference = load_case(reference_dir)
    shorts = [load_case(path) for path in short_dirs]
    for name, digest in reference['manifest']['source_sha256'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'current public source differs from the saved reference: {name}')
    config = reference['config']
    ph = create_unit_mapping(config).physical
    frequency = config['run']['frequency_Hz']
    # Only geometry, runtime, sampling and probe lists may differ.
    for case in shorts:
        if case['manifest']['source_sha256'] != reference['manifest']['source_sha256']:
            raise ValueError('long and short runs used different public runtime sources')
        for key in ('physical', 'lattice', 'collision', 'transport'):
            if case['config'][key] != config[key]:
                raise ValueError(f'mismatched {key}')
        for key, value in config['numerics'].items():
            if key not in ('ny', 'steps') and case['config']['numerics'][key] != value:
                raise ValueError(f'mismatched numerics.{key}')
        for key, value in config['run'].items():
            if key not in ('length_m', 'cycles', 'probes_m', 'sampling') and case['config']['run'][key] != value:
                raise ValueError(f'mismatched run.{key}')
        for nominal in INTERFACES_M:
            if case['data']['probe_z_m'][probe_index(case, nominal)] != reference['data']['probe_z_m'][probe_index(reference, nominal)]:
                raise ValueError('interface cells differ between domains')

    long_hats, long_details = fit_case(reference, WINDOW)
    propagation_control, results = [], []
    for nominal in INTERFACES_M:
        index = probe_index(reference, nominal)
        interface_z = float(reference['data']['probe_z_m'][index])
        for target in TARGETS_M:
            target_index = probe_index(reference, target)
            target_z = float(reference['data']['probe_z_m'][target_index])
            predicted = propagate(long_hats[f'outgoing_Pa_{index}'], frequency, target_z - interface_z, ph.c0_m_s)
            propagation_control.append(dict(interface_z_m=interface_z, target_z_m=target_z,
                predicted_Pa=record(predicted),
                vs_long_total=checked(predicted, long_hats[f'pressure_Pa_{target_index}']),
                vs_long_outgoing=checked(predicted, long_hats[f'outgoing_Pa_{target_index}'])))

    for case in shorts:
        hats, details = fit_case(case, WINDOW)
        time_check = stationarity(case)
        truncation, handoff, independence, predicted_by_interface = [], [], [], []
        for nominal in INTERFACES_M:
            index, long_index = probe_index(case, nominal), probe_index(reference, nominal)
            z = float(case['data']['probe_z_m'][index])
            truncation.append(dict(interface_z_m=z, channels={field: checked(hats[f'{field}_{index}'],
                long_hats[f'{field}_{long_index}']) for field in ('pressure_Pa', 'velocity_m_s', 'outgoing_Pa')}))
            predictions = []
            for target in TARGETS_M:
                target_index = probe_index(reference, target)
                target_z = float(reference['data']['probe_z_m'][target_index])
                predicted = propagate(hats[f'outgoing_Pa_{index}'], frequency, target_z - z, ph.c0_m_s)
                predictions.append(predicted)
                handoff.append(dict(interface_z_m=z, target_z_m=target_z,
                    predicted_Pa=record(predicted), reference_total_Pa=record(long_hats[f'pressure_Pa_{target_index}']),
                    vs_long_total=checked(predicted, long_hats[f'pressure_Pa_{target_index}']),
                    vs_long_outgoing=checked(predicted, long_hats[f'outgoing_Pa_{target_index}'])))
            predicted_by_interface.append(predictions)
        for i in range(len(INTERFACES_M)):
            for j in range(i):
                independence.append(dict(interface_nominal_m=INTERFACES_M[i], reference_interface_nominal_m=INTERFACES_M[j],
                    # Common propagation factor cancels; one common target suffices.
                    target_nominal_m=TARGETS_M[0], comparison=checked(predicted_by_interface[i][0], predicted_by_interface[j][0])))
        wall = checked(hats['wall_flux_W_m2'], long_hats['wall_flux_W_m2'])
        reflections = [details[f'incoming_Pa_{i}']['incoming_over_outgoing'] for i in range(len(INTERFACES_M))]
        gates = dict(public_validation=bool(case['summary']['validation_passed']),
            harmonic_stationarity=time_check['passed'],
            interface_truncation=all(v['passed'] for row in truncation for v in row['channels'].values()),
            wall_flux_change=wall['passed'], incoming_over_outgoing=all(r <= REFLECTION_RATIO for r in reflections),
            handoff_vs_total=all(row['vs_long_total']['passed'] for row in handoff),
            handoff_vs_outgoing=all(row['vs_long_outgoing']['passed'] for row in handoff),
            interface_independence=all(row['comparison']['passed'] for row in independence))
        results.append(dict(**metadata(case), fits=details, stationarity=time_check,
            wall_flux_vs_long=wall, interface_truncation=truncation, handoff=handoff,
            interface_independence=independence, gates=gates, handoff_passed=all(gates.values()),
            cell_step_cost_relative_to_equal_cycles_long=case['config']['numerics']['ny'] / config['numerics']['ny']))

    preflight_path = reference['path'].parent.parent / 'final/time_convergence.json'
    preflight = json.loads(preflight_path.read_text(encoding='utf-8'))
    if (Path(preflight['case_dir']).resolve() != reference['path'] or
            preflight['git_commit'] != reference['manifest']['git_commit']):
        raise ValueError('time-convergence preflight does not identify this long case')
    gates = dict(long_public_validation=bool(reference['summary']['validation_passed']),
        long_time_convergence=bool(preflight['time_convergence_passed']),
        long_propagation_control=all(row[key]['passed'] for row in propagation_control
                                    for key in ('vs_long_total', 'vs_long_outgoing')),
        long_incoming_over_outgoing=all(value['incoming_over_outgoing'] <= REFLECTION_RATIO
            for key, value in long_details.items() if key.startswith('incoming_Pa_')),
        short_cases=all(result['handoff_passed'] for result in results))
    for case in [reference, *shorts]:
        if case['hashes'] != {name: sha256(case['path'] / name) for name in FILES}:
            raise RuntimeError('raw files changed during analysis')
    return dict(status='completed', handoff_validation_passed=all(gates.values()), gates=gates,
        frequency_Hz=frequency, fit_cycles=WINDOW, phase_convention='exp(+i omega t)',
        propagation='p_plus=(p+rho*c*u_normal)/2; p(z)=p_plus(z_interface)*exp(-i*omega*distance/c)',
        sound_speed_m_s=ph.c0_m_s, impedance_Pa_s_m=ph.rho0_kg_m3 * ph.c0_m_s,
        limits=dict(amplitude_percent=AMPLITUDE_PERCENT, phase_deg=PHASE_DEG,
            incoming_over_outgoing=REFLECTION_RATIO, time_amplitude_percent=TIME_AMPLITUDE_PERCENT,
            time_phase_deg=TIME_PHASE_DEG),
        reference=dict(**metadata(reference), fits=long_details,
            time_convergence_file=str(preflight_path), time_convergence_sha256=sha256(preflight_path)),
        propagation_control=propagation_control, short_cases=results,
        analysis_sources_sha256={str(path.relative_to(ROOT)): sha256(path) for path in
            (Path(__file__), Path(__file__).with_name('handoff_study.yaml'), Path(__file__).with_name('gas.yaml'))},
        raw_files_unchanged=True, current_public_source_matches_runs=True,
        scope='Small-amplitude prescribed-temperature planar normal incidence. The long LBM domain is '
              'a numerical reference, not exact truth. No finite-film geometric spreading, power-driven '
              'film/substrate validation, full-field equilibrium claim, or reduction of the LBM time step count. '
              'No fitted propagation speed, gain, phase shift, or heat-flux source replacement. '
              'Public validation and additional handoff gates are reported separately.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--short', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True, help='New final output directory')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'refusing to overwrite {args.output}')
    result = analyze(args.reference, args.short)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'handoff.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
        allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'handoff_validation_passed', 'gates')}, indent=2))


if __name__ == '__main__':
    main()
