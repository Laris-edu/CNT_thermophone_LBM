"""Two controlled diagnostics for the saved finite-strip boundary failure.

This does not change acceptance thresholds or the production boundary. Each
run changes one feature of the saved candidate and uses a new output folder.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from simulation.finite_strip import run_strip_case
from simulation.output import dump
from simulation.study import valid_name
from core.unit_mapping import create_unit_mapping


TOPIC = Path(__file__).resolve().parent
ORIGINAL = TOPIC / '20261012_strip_validation'
CASE_NAMES = ('pulse_no_sponge_dx10um', 'pulse_tall_dx10um')


def configuration(name):
    c = yaml.safe_load((ORIGINAL / 'raw/pulse_candidate_dx10um/effective.yaml').read_text())
    c = deepcopy(c)
    c['case']['name'] = name
    c['strip']['name'] = name
    if name == CASE_NAMES[0]:
        c['strip']['sponge_rate'] = 0.
        change = 'sponge_rate only: 0.05 -> 0; geometry and outer characteristic unchanged'
    elif name == CASE_NAMES[1]:
        c['numerics']['ny'] = 200
        for key in ('height_m', 'requested_height_m', 'actual_height_m'):
            c['strip'][key] = .002
        change = 'top height only: 0.5 -> 2 mm; width and sponge profile unchanged'
    else:
        raise ValueError(name)
    c['boundary_diagnosis'] = dict(source_case=str(ORIGINAL / 'raw/pulse_candidate_dx10um'),
                                   controlled_change=change, certifies_20kHz=False)
    return c


def load_case(path):
    summary = json.loads((path / 'summary.json').read_text())
    if summary['status'] != 'completed' or not summary['state_validation_passed']:
        raise ValueError(f'Case failed state checks: {path}')
    with np.load(path / 'trajectory.npz') as data:
        return {key: data[key] for key in data.files}


def difference_metrics(actual, reference, *, start_s=1.8e-6, stop_s=np.inf):
    for key in ('t_s', 'probe_x_m', 'probe_y_m'):
        if not np.array_equal(actual[key], reference[key]):
            raise ValueError(f'Unmatched {key}')
    t = actual['t_s']
    scale = np.max(abs(reference['pressure_perturbation_Pa']), axis=0)
    error = actual['pressure_perturbation_Pa'] - reference['pressure_perturbation_Pa']
    selected = np.flatnonzero((t >= start_s) & (t <= stop_s))
    normalized = abs(error[selected]) / scale
    local, probe = np.unravel_index(np.argmax(normalized), normalized.shape)
    index = int(selected[local])
    return dict(max_relative=float(normalized[local, probe]), probe=int(probe),
        peak_time_s=float(t[index]), signed_difference_Pa=float(error[index, probe]),
        reference_peak_Pa=float(scale[probe]),
        actual_pressure_Pa=float(actual['pressure_perturbation_Pa'][index, probe]),
        reference_pressure_Pa=float(reference['pressure_perturbation_Pa'][index, probe]),
        probe_x_m=float(actual['probe_x_m'][probe]), probe_y_m=float(actual['probe_y_m'][probe]),
        per_probe_relative=np.max(normalized, axis=0).tolist())


def analyze(root):
    source = ORIGINAL / 'raw'
    reference = load_case(source / 'pulse_reference_dx10um')
    candidate = load_case(source / 'pulse_candidate_dx10um')
    no_sponge = load_case(root / 'raw' / CASE_NAMES[0])
    tall = load_case(root / 'raw' / CASE_NAMES[1])
    metrics = {}
    for name, data in (('candidate', candidate), ('no_sponge', no_sponge), ('tall', tall)):
        metrics[name] = dict(
            full=difference_metrics(data, reference),
            early=difference_metrics(data, reference, start_s=0., stop_s=1.2e-6),
            before_side_return=difference_metrics(data, reference, stop_s=4e-6))
    t = candidate['t_s']
    difference = candidate['pressure_perturbation_Pa'] - reference['pressure_perturbation_Pa']
    scale = np.max(abs(reference['pressure_perturbation_Pa']), axis=0)
    index, probe = np.unravel_index(np.argmax(abs(difference) / scale), difference.shape)
    original_peak = dict(time_s=float(t[index]), probe=int(probe),
        candidate_minus_reference_Pa=float(difference[index, probe]),
        candidate_minus_tall_Pa=float(candidate['pressure_perturbation_Pa'][index, probe]
                                      - tall['pressure_perturbation_Pa'][index, probe]),
        tall_minus_reference_Pa=float(tall['pressure_perturbation_Pa'][index, probe]
                                      - reference['pressure_perturbation_Pa'][index, probe]),
        no_sponge_minus_reference_Pa=float(no_sponge['pressure_perturbation_Pa'][index, probe]
                                           - reference['pressure_perturbation_Pa'][index, probe]))
    output = root / 'final/boundary_cause.json'
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(scope='Controlled broadband boundary diagnosis; no thermal-source certification',
        original_batch=str(ORIGINAL), diagnostic_batch=str(root),
        metrics=metrics, decomposition_at_original_peak=original_peak,
        acceptance_limit_unchanged=.01, frequency_20kHz_validated=False,
        propagation_validation_passed=None)
    dump(output, result)
    print(json.dumps(result, indent=2), flush=True)


def characteristic_compatibility(root):
    """Independent free-space acoustic test of the local plane-wave condition.

    Fourier-Bessel evolution of a Gaussian initial pressure with zero velocity.
    No computed reference values enter the LBM or change its boundary condition.
    """
    from scipy.integrate import simpson
    from scipy.special import j0, j1
    c = configuration(CASE_NAMES[0])
    mapping = create_unit_mapping(c)
    sigma = c['strip']['pulse_sigma_m']
    radius = c['strip']['actual_height_m']
    amplitude = c['strip']['pulse_amplitude_Pa']
    sound_speed = mapping.physical.c0_m_s
    dt = mapping.lattice.dt_s
    t = np.arange(int(np.ceil(3e-6 / dt)) + 1) * dt
    solutions = []
    for points in (3001, 6001):
        q = np.linspace(0., 12., points)  # q = sigma*k
        weight = q * np.exp(-q*q/2)
        phase = np.outer(t * sound_speed / sigma, q)
        pressure = amplitude * simpson(weight*j0(q*radius/sigma)*np.cos(phase), x=q, axis=1)
        impedance_velocity = amplitude * simpson(weight*j1(q*radius/sigma)*np.sin(phase), x=q, axis=1)
        solutions.append((pressure, impedance_velocity))
    pressure, impedance_velocity = solutions[-1]
    refinement_change = max(float(np.max(abs(a-b))) for a, b in zip(*solutions))
    exact_initial = amplitude * np.exp(-.5*(radius/sigma)**2)
    assert abs(pressure[0]-exact_initial) < 1e-9
    assert refinement_change < 1e-8
    outgoing = pressure + impedance_velocity
    local_incoming = pressure - impedance_velocity
    target_pressure = outgoing / 2
    right = amplitude * np.exp(-.5*((radius-sound_speed*t)/sigma)**2)
    left = amplitude * np.exp(-.5*((radius+sound_speed*t)/sigma)**2)
    plane_pressure, plane_impedance_velocity = .5*(right+left), .5*(right-left)
    result = dict(
        scope='Independent homogeneous inviscid acoustic counterpart; not an LBM validation or imposed source',
        initial_condition='p=A exp(-r^2/(2 sigma^2)), velocity=0, free space',
        pressure_formula='p/A = integral_0^infinity q exp(-q^2/2) J0(q*r/sigma) cos(q*c*t/sigma) dq',
        velocity_formula='rho*c*u_r/A = integral_0^infinity q exp(-q^2/2) J1(q*r/sigma) sin(q*c*t/sigma) dq',
        reference_config=str(ORIGINAL/'raw/pulse_candidate_dx10um/effective.yaml'),
        radius_m=radius, sigma_m=sigma, amplitude_Pa=amplitude, sound_speed_m_s=sound_speed,
        time_window_s=[float(t[0]), float(t[-1])], sample_dt_s=dt,
        extrema_scope='Maxima over the saved discrete times, not continuous-time extrema',
        quadrature_points=[3001, 6001], quadrature_upper_q=12., refinement_change_Pa=refinement_change,
        cylindrical_pressure_peak_Pa=float(np.max(abs(pressure))),
        cylindrical_local_incoming_peak_Pa=float(np.max(abs(local_incoming))),
        target_pressure_change_relative_to_local_peak=float(np.max(abs(target_pressure-pressure))/np.max(abs(pressure))),
        plane_local_incoming_peak_Pa=float(np.max(abs(plane_pressure-plane_impedance_velocity))),
        interpretation='An outgoing cylindrical field need not have p-rho*c*u_r=0. The projection mismatch is not a measured reflection coefficient or a decomposition of the 5.52% return error.',
        reference_injected_into_lbm=False)
    output = root/'final/characteristic_compatibility.json'
    arrays = root/'_work/characteristic_compatibility.npz'
    if output.exists() or arrays.exists():
        raise FileExistsError('Characteristic diagnostic already exists')
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays, t_s=t, cylindrical_pressure_Pa=pressure,
        cylindrical_impedance_velocity_Pa=impedance_velocity,
        local_target_pressure_Pa=target_pressure,
        plane_pressure_Pa=plane_pressure, plane_impedance_velocity_Pa=plane_impedance_velocity)
    dump(output, result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', required=True)
    parser.add_argument('--case', choices=CASE_NAMES)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--characteristic-check', action='store_true')
    args = parser.parse_args()
    root = TOPIC / valid_name(args.batch)
    if args.characteristic_check:
        characteristic_compatibility(root)
        return
    if args.analyze:
        analyze(root)
        return
    if not args.case:
        parser.error('--case is required when not analyzing')
    c = configuration(args.case)
    output = root / 'raw' / args.case
    print(json.dumps(dict(name=args.case, output=str(output), exists=output.exists(),
        nx=c['numerics']['nx'], ny=c['numerics']['ny'], dt_s=c['lattice']['dt_s'],
        steps=c['strip']['steps'], change=c['boundary_diagnosis']['controlled_change']), indent=2), flush=True)
    if output.exists():
        raise FileExistsError(output)
    if not args.dry_run:
        run_strip_case(c, output)


if __name__ == '__main__':
    main()
