"""Read-only time-convergence analysis; run as python -m Compute.lbm_acoustic_handoff.analyze."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import CubicSpline

from postproc.diagnostics import compare, record
from postproc.multiharmonic_fit import fit_multiharmonic


WINDOWS = ((4, 6), (10, 12), (18, 20))
AMPLITUDE_PERCENT = 0.1
PHASE_DEG = 0.1


def fit_series(t, values, frequency, window, order):
    start, end = window
    mask = (t * frequency >= start - 1e-10) & (t * frequency < end - 1e-10)
    fit = fit_multiharmonic(t[mask], values[mask], 2 * np.pi * frequency,
                           n_harmonics=3, detrend_order=order)
    return fit.harmonic(1), dict(**record(fit.harmonic(1)),
        sample_mean=float(np.mean(values[mask])),
        trend_at_midpoint=float(fit.trend_coeffs_scaled[0]),
        drift_per_s=float(fit.trend_coeffs_raw[1]),
        residual_rms=float(fit.residual_rms), n_samples=fit.window.n_samples)


def check(got, reference):
    result = compare(got, reference)
    result['passed'] = bool(abs(result['amplitude_error_percent']) <= AMPLITUDE_PERCENT
                            and abs(result['phase_error_deg']) <= PHASE_DEG)
    return result


def analyze(case_dir):
    case_dir = Path(case_dir).resolve()
    manifest = json.loads((case_dir / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((case_dir / 'summary.json').read_text(encoding='utf-8'))
    config = yaml.safe_load((case_dir / 'effective.yaml').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or config['run']['mode'] != 'thermal':
        raise ValueError('requires a completed prescribed-temperature case')
    with np.load(case_dir / 'trajectory.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    frequency = config['run']['frequency_Hz']
    if (not bool(data['run_completed']) or data['t_s'][-1] * frequency < 20 - 1e-8):
        raise ValueError('a complete trajectory of at least 20 cycles is required')
    channels = {'wall_flux_W_m2': (data['heat_time_s'], data['wall_flux_W_m2']),
                'first_cell_temperature_K': (data['t_s'], data['first_cell_temperature_K'])}
    for index in range(len(data['probe_z_m'])):
        for name in ('pressure_Pa', 'velocity_m_s'):
            channels[f'{name}_{index}'] = (data['t_s'], data[name][:, index])
    windows, amplitudes = [], []
    for window in WINDOWS:
        records, hats = {}, {}
        for name, (t, values) in channels.items():
            hat, fitted = fit_series(t, values, frequency, window, order=1)
            constant_hat, _ = fit_series(t, values, frequency, window, order=0)
            records[name] = dict(**fitted, detrend_sensitivity=check(constant_hat, hat))
            hats[name] = hat
        windows.append(dict(cycles=window, channels=records))
        amplitudes.append(hats)
    comparisons = []
    for index in range(len(WINDOWS) - 1):
        metrics = {name: check(amplitudes[index][name], amplitudes[-1][name]) for name in channels}
        comparisons.append(dict(candidate_cycles=WINDOWS[index], reference_cycles=WINDOWS[-1],
                                channels=metrics, passed=all(v['passed'] for v in metrics.values())))

    # The collector rounds the sampling stride to integer steps; cycle phases
    # need not repeat. Interpolate actual timestamps onto common phases without
    # extrapolation, retaining means and transients (no fitted phase correction).
    # These are state comparisons, not energy integrals or thermal-equilibrium gates.
    phase_count = config['run']['sampling']['profile_samples_per_cycle']
    phase_clock = data['profile_t_s'] * frequency
    profiles = {}
    native_counts = [int(np.count_nonzero((phase_clock >= start - 1e-10)
                                         & (phase_clock < end - 1e-10))) for start, end in WINDOWS]
    for name in ('temperature_K', 'pressure_Pa', 'velocity_m_s'):
        interpolate = CubicSpline(phase_clock, data[f'profile_{name}'], axis=0, extrapolate=False)
        blocks = []
        for start, end in WINDOWS:
            targets = np.arange(start, end)[:, None] + np.arange(phase_count)[None, :] / phase_count
            block = interpolate(targets)
            if not np.all(np.isfinite(block)):
                raise ValueError('saved profile timestamps do not cover the requested phase grid')
            blocks.append(block)
        profiles[name] = blocks
    profile_comparisons, mean_columns = [], {'z_m': data['z_m']}
    for name, blocks in profiles.items():
        late = blocks[-1].mean(axis=0)
        late_mean = late.mean(axis=0)
        scale = float(np.max(abs(late - late_mean)))
        for window, block in zip(WINDOWS, blocks):
            mean_columns[f'{name}_cycles_{window[0]}_{window[1]}'] = block.mean(axis=(0, 1))
        for index, block in enumerate(blocks):
            waveform = block.mean(axis=0)
            difference = waveform - late
            mean_difference = waveform.mean(axis=0) - late_mean
            oscillation_difference = difference - mean_difference
            raw_max = float(np.max(abs(difference)))
            profile_comparisons.append(dict(field=name, cycles=WINDOWS[index],
                reference_cycles=WINDOWS[-1],
                max_mean_difference=float(np.max(abs(mean_difference))),
                max_raw_same_phase_difference=raw_max,
                max_mean_removed_same_phase_difference=float(np.max(abs(oscillation_difference))),
                max_adjacent_cycle_difference=float(np.max(abs(block[1] - block[0]))),
                reference_max_phase_deviation=scale,
                raw_difference_percent_of_reference_phase_deviation=100 * raw_max / scale if scale else None))
    harmonic_passed = all(item['passed'] for item in comparisons)
    sensitivity_passed = all(channel['detrend_sensitivity']['passed']
        for window in windows for channel in window['channels'].values())
    result = dict(case_dir=str(case_dir), status=manifest['status'],
        validation_passed=summary['validation_passed'], public_gates=summary['gates'],
        git_commit=manifest['git_commit'], git_dirty=manifest['git_dirty'],
        source_clean=manifest['source_clean'],
        frequency_Hz=frequency, probe_z_m=data['probe_z_m'].tolist(),
        windows=windows, comparisons=comparisons, profile_comparisons=profile_comparisons,
        profile_sampling=dict(native_samples_in_windows=native_counts,
            common_samples_per_cycle=phase_count, interpolation='cubic spline on recorded timestamps; no extrapolation',
            phases='0, 1/N, ..., (N-1)/N'),
        limits=dict(amplitude_percent=AMPLITUDE_PERCENT, phase_deg=PHASE_DEG),
        harmonic_time_convergence_passed=harmonic_passed,
        detrend_sensitivity_passed=sensitivity_passed,
        time_convergence_passed=harmonic_passed and sensitivity_passed,
        full_field_periodic_steady_state_established=None,
        scope='Finite-window harmonic convergence; profile means and raw waveforms reported separately. '
              'No claim of infinite-time equilibrium, short-domain accuracy, or finite-film radiation.',
        sampling='Pressure/normal velocity/temperature at step end; actual wall exchange at interval midpoint. '
                 'Profile means use a common interpolated phase grid; no energy integral is inferred.',
        phase_convention='exp(+i omega t), shared original time axis; no phase or gain correction')
    return result, mean_columns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New directory; defaults to the batch final directory')
    args = parser.parse_args()
    output = args.output or args.case_dir.resolve().parent.parent / 'final'
    if output.exists():
        raise FileExistsError(f'refusing to overwrite analysis directory: {output}')
    result, means = analyze(args.case_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'time_convergence.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'mean_profiles.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(means)
        writer.writerows(zip(*means.values()))
    print(json.dumps({key: result[key] for key in ('status', 'validation_passed',
        'harmonic_time_convergence_passed', 'detrend_sensitivity_passed', 'time_convergence_passed')}, indent=2))


if __name__ == '__main__':
    main()
