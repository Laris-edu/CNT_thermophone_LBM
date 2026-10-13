"""Saved finite-strip nearfield checks and gated closed-curve k-Wave export.

Run from the repository root with ``python -m Compute.finite_strip.analyze``.
This entry never advances LBM or k-Wave. A valid source is not a validated
45 mm propagation result, a three-dimensional square film, or grid convergence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat
import yaml

from core.unit_mapping import create_unit_mapping
from farfield.kirchhoff_2d import (GREEN_OUTGOING, dpdn_from_velocity,
                                  kirchhoff_2d_frequency)
from farfield.strip_surface import mirror_strip_surface
from postproc.diagnostics import compare
from postproc.multiharmonic_fit import fit_multiharmonic
from postproc.power import stationarity_at_cycle
from simulation.output import dump
from simulation.provenance import capture_source
from simulation.study import valid_name

ROOT = Path(__file__).resolve().parents[2]
LIMIT = .01
PHASE_LIMIT_DEG = 1.
GEOMETRY_KEYS = ('x_m', 'y_m', 'normal_x', 'normal_y', 'ds_m')


def _window(t, frequency, start, end):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('timestamps must be finite and strictly increasing')
    return (t*frequency >= start-1e-10) & (t*frequency < end-1e-10)


def _fit_columns(t, values, frequency, start, end):
    """Use the public absolute-clock fitter, including resolved zero components."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(t):
        raise ValueError('fit signals must have shape (time, channel)')
    mask = _window(t, frequency, start, end)
    out = {key: [] for key in ('hat', 'mean', 'residual_rms', 'condition_number', 'resolution')}
    for x in values.T:
        fit = fit_multiharmonic(np.asarray(t)[mask], x[mask], 2*np.pi*frequency,
                               n_harmonics=3, detrend_order=1)
        out['hat'].append(fit.harmonic(1))
        out['mean'].append(float(fit.trend_coeffs_scaled[0]))
        out['residual_rms'].append(float(fit.residual_rms))
        out['condition_number'].append(float(fit.condition_number))
        out['resolution'].append(256*np.finfo(float).eps*max(1., float(np.max(abs(x[mask])))))
    return {key: np.asarray(value) for key, value in out.items()}


def _time_coverage(t, frequency, end, audit_end):
    """The public fitter alone can fit a sinusoid across an incomplete window."""
    t = np.asarray(t, dtype=float)
    differences = np.diff(t)
    if not len(differences) or not np.isfinite(t).all() or np.any(differences <= 0):
        return dict(passed=False, reason='invalid observation clock')
    cadence = float(np.median(differences))
    tolerance = 1e-10/frequency
    return dict(first_s=float(t[0]), last_s=float(t[-1]), cadence_s=cadence,
        passed=bool(t[0] <= (end-8)/frequency+cadence+tolerance
            and t[-1] >= end/frequency-cadence-tolerance and t[-1] <= audit_end+tolerance
            and np.max(differences) <= cadence*(1+1e-8)
            and cadence*frequency < 1/6),
        criterion='eight complete cycles up to one saved-sample spacing at either edge; '
            'no missing cadence and sampling above third-harmonic Nyquist')


def _fit_diagnostic(fit):
    resolved = abs(fit['hat']) > fit['resolution']
    ratios = np.divide(fit['residual_rms'], abs(fit['hat']),
                       out=np.zeros_like(fit['residual_rms']), where=resolved)
    return dict(max_residual_rms=float(np.max(fit['residual_rms'], initial=0.)),
                max_residual_over_resolved_fundamental=float(np.max(ratios, initial=0.)),
                unresolved_indices=np.flatnonzero(~resolved).tolist(),
                max_condition_number=float(np.max(fit['condition_number'], initial=0.)),
                residual_is_acceptance_gate=False)


def _stationarity_input(t, film_t, heat_t, film_T, film_q, p, un):
    return dict(t_s=t, film_t_s=film_t, heat_time_s=heat_t,
                film_temperature_K=film_T, wall_flux_W_m2=film_q,
                pressure_Pa=p, velocity_m_s=un)


def _stationarity(data, frequency, end):
    checked = stationarity_at_cycle(data, frequency, end,
        dict(stationary_relative=LIMIT, stationary_phase_deg=PHASE_LIMIT_DEG))
    # Keep every tested point and its sensitivities; the full time records remain raw.
    keep = ('passed', 'error', 'amplitude_span_relative', 'phase_span_deg',
            'duration_sensitivity', 'detrend_sensitivity')
    checked['channels'] = {name: {key: item[key] for key in keep if key in item}
                           for name, item in checked['channels'].items()}
    return checked


def _comparisons(actual, expected):
    rows = []
    for got, want in zip(np.asarray(actual).ravel(), np.asarray(expected).ravel()):
        if not np.isfinite(got) or not np.isfinite(want) or abs(want) <= 256*np.finfo(float).eps:
            rows.append(dict(passed=False, error='nonfinite or unresolved comparison phasor'))
        else:
            item = compare(got, want)
            item['passed'] = bool(abs(item['amplitude_error_percent']) < 100*LIMIT
                                 and abs(item['phase_error_deg']) < PHASE_LIMIT_DEG)
            rows.append(item)
    return dict(passed=bool(rows and all(row['passed'] for row in rows)), points=rows)


def _propagate(surface, x, y, *, frequency, rho0, c0):
    return kirchhoff_2d_frequency(
        surface_x_m=surface['x_m'], surface_y_m=surface['y_m'],
        normal_x=surface['normal_x'], normal_y=surface['normal_y'], ds_m=surface['ds_m'],
        p_hat_Pa=surface['p_hat_Pa'],
        dpdn_hat_Pa_m=dpdn_from_velocity(surface['un_hat_m_s'],
            omega_rad_s=2*np.pi*frequency, rho0_kg_m3=rho0),
        observer_x_m=x, observer_y_m=y, omega_rad_s=2*np.pi*frequency,
        c0_m_s=c0, green_convention=GREEN_OUTGOING)


def analyze_data(data, effective, summary, boundary):
    """Return numerical diagnostics and phasors; never certify untested propagation.

    This pure interface also admits labelled manufactured test fixtures. Only
    ``analyze_saved`` attaches the actual input-file and runtime provenance.
    """
    mapping = create_unit_mapping(effective)
    run, gas = effective['strip'], mapping.physical
    if run['kind'] != 'power':
        raise ValueError('nearfield analysis requires a power-driven finite-strip case')
    frequency = float(run['frequency_Hz'])
    audit_t = np.asarray(data['audit_t_s'], dtype=float)
    if not len(audit_t) or not np.isfinite(audit_t).all() or np.any(np.diff(audit_t) <= 0):
        raise ValueError('actual completed audit timestamps are required')
    end = int(np.floor(audit_t[-1]*frequency+1e-10))
    if end < 2:
        raise ValueError('two complete cycles are required to fit a source')
    start = end-2
    t = np.asarray(data['t_s'], dtype=float)
    surface_t = np.asarray(data['surface_t_s'], dtype=float)
    heat_t = np.asarray(data['heat_time_s'], dtype=float)
    coverage = {name: _time_coverage(times, frequency, end, audit_t[-1])
                for name, times in (('endpoint', t), ('surface_endpoint', surface_t), ('heat_midpoint', heat_t))}
    if heat_t.shape != t.shape or not np.allclose(heat_t, t-.5*mapping.lattice.dt_s,
                                                rtol=0., atol=1e-14*mapping.lattice.dt_s):
        raise ValueError('actual wall exchange must use the single-step interval midpoint')
    film_T = np.asarray(data['film_temperature_K'], dtype=float)
    # Saved wall energy is one accepted lattice interval, even under observation decimation.
    film_q = np.asarray(data['film_wall_energy_J_m2'], dtype=float)/mapping.lattice.dt_s
    if film_T.shape != film_q.shape or film_T.ndim != 2 or film_T.shape[1] == 0:
        raise ValueError('film temperature and interval wall exchange shapes differ')
    channels = [str(v) for v in data['surface_channels']]
    required = ('pressure_perturbation_Pa', 'ux_m_s', 'uy_m_s', 'temperature_K', 'rho_kg_m3')
    if any(channels.count(name) != 1 for name in required):
        raise ValueError('surface channels must include unique pressure, velocity, temperature, rho_kg_m3')
    fields = np.asarray(data['surface_fields'], dtype=float)
    if fields.ndim != 3 or fields.shape[0] != len(surface_t) or fields.shape[2] != len(channels):
        raise ValueError('surface_fields shape differs from timestamps or channel metadata')
    surface_values = {name: fields[:, :, channels.index(name)] for name in required}
    geometry = {key: np.asarray(data['surface_'+key], dtype=float) for key in GEOMETRY_KEYS}
    count = fields.shape[1]
    if (any(value.shape != (count,) or not np.isfinite(value).all() for value in geometry.values())
            or np.any(geometry['ds_m'] <= 0)
            or not np.allclose(np.hypot(geometry['normal_x'], geometry['normal_y']), 1., rtol=0, atol=1e-10)):
        raise ValueError('invalid saved surface geometry')
    slices = np.asarray(data['surface_slices'], dtype=int)
    if (slices.shape != (3, 2) or slices[0, 0] != 0 or slices[-1, 1] != count
            or np.any(slices[:, 1] <= slices[:, 0]) or np.any(slices[1:, 0] != slices[:-1, 1])):
        raise ValueError('three contiguous nonempty contour slices are required')
    p = _fit_columns(surface_t, surface_values[required[0]], frequency, start, end)
    ux = _fit_columns(surface_t, surface_values['ux_m_s'], frequency, start, end)
    uy = _fit_columns(surface_t, surface_values['uy_m_s'], frequency, start, end)
    temperature = _fit_columns(surface_t, surface_values['temperature_K'], frequency, start, end)
    rho = _fit_columns(surface_t, surface_values['rho_kg_m3'], frequency, start, end)
    Tfit = _fit_columns(t, film_T, frequency, start, end)
    qfit = _fit_columns(heat_t, film_q, frequency, start, end)
    direct = _fit_columns(t, data['pressure_perturbation_Pa'], frequency, start, end)
    px, py = (np.asarray(data[key], dtype=float) for key in ('probe_x_m', 'probe_y_m'))
    if px.shape != py.shape or px.shape != direct['hat'].shape or not np.isfinite([px, py]).all():
        raise ValueError('direct probe coordinates and channels differ')
    margin = max(run.get('pml_cells', 0), run.get('sponge_cells', 0))*mapping.lattice.dx_m
    inner_x, inner_y = run['actual_width_m']/2-margin, run['actual_height_m']-margin
    exterior = (abs(px) < inner_x) & (py > 0) & (py < inner_y)
    geometry_pass = True
    upper_surfaces = []
    for a, b in slices:
        upper = {key: value[a:b] for key, value in geometry.items()}
        half_width, height = float(np.max(abs(upper['x_m']))), float(np.max(upper['y_m']))
        exterior &= (abs(px) > half_width+1e-14) | (py > height+1e-14)
        film_extent = max(abs(run.get('actual_film_x_min_m', -run['film_width_m']/2)),
                          abs(run.get('actual_film_x_max_m', run['film_width_m']/2)))
        geometry_pass &= bool(half_width > film_extent and half_width < inner_x
                              and height < inner_y and np.all(upper['y_m'] > 0))
        upper_surfaces.append(upper)
    probe_indices = np.flatnonzero(exterior)
    external_count_pass = len(probe_indices) >= 3
    un_series = (surface_values['ux_m_s']*geometry['normal_x']
                 + surface_values['uy_m_s']*geometry['normal_y'])
    arrays = dict(surface_p_hat_Pa=p['hat'], surface_ux_hat_m_s=ux['hat'],
                  surface_uy_hat_m_s=uy['hat'], surface_un_hat_m_s=
                  ux['hat']*geometry['normal_x']+uy['hat']*geometry['normal_y'],
                  surface_T_mean_K=temperature['mean'], surface_rho_mean_kg_m3=rho['mean'],
                  surface_slices=slices, film_T_hat_K=Tfit['hat'], film_q_hat_W_m2=qfit['hat'],
                  film_x_m=np.asarray(data['film_x_m']), direct_probe_p_hat_Pa=direct['hat'],
                  probe_x_m=px, probe_y_m=py, validation_probe_indices=probe_indices)
    arrays.update({'surface_'+key: value for key, value in geometry.items()})
    for name, fit in (('surface_p', p), ('surface_ux', ux), ('surface_uy', uy),
                      ('surface_T', temperature), ('surface_rho', rho), ('film_T', Tfit), ('film_q', qfit)):
        arrays[name+'_residual_rms'] = fit['residual_rms']
    contours, reconstructed = [], []
    pressure0 = mapping.lattice.rho_ref_lu*mapping.theta_ref_lu*mapping.pressure_scale
    for index, ((a, b), upper) in enumerate(zip(slices, upper_surfaces)):
        closed = mirror_strip_surface(upper, p['hat'][a:b], ux['hat'][a:b], uy['hat'][a:b])
        closure = np.hypot(np.sum(closed['normal_x']*closed['ds_m']),
                           np.sum(closed['normal_y']*closed['ds_m']))/np.sum(closed['ds_m'])
        geometry_pass &= bool(closure < 1e-8)
        reconstructed.append(_propagate(closed, px[probe_indices], py[probe_indices],
            frequency=frequency, rho0=gas.rho0_kg_m3, c0=gas.c0_m_s))
        stable = _stationarity(_stationarity_input(surface_t, t, heat_t,
            film_T[:, 0], film_q[:, 0], surface_values[required[0]][:, a:b], un_series[:, a:b]),
            frequency, end)
        # The shared API requires film channels; their independent all-bin check
        # below owns those results, so a film failure does not mislabel Gamma.
        stable['channels'] = {name: item for name, item in stable['channels'].items()
                              if name.startswith(('pressure_', 'velocity_'))}
        stable['passed'] = bool(stable['eligible'] and all(item['passed'] for item in stable['channels'].values()))
        stable['checked_surface_points'] = int(b-a)
        stable['checked_quantities'] = ['pressure', 'normal_velocity']
        weights = upper['ds_m']/np.sum(upper['ds_m'])
        background = dict(mean_temperature_K=float(weights@temperature['mean'][a:b]),
                          mean_density_kg_m3=float(weights@rho['mean'][a:b]),
            max_point_temperature_departure_relative=float(np.max(abs(temperature['mean'][a:b]/gas.T0_K-1))),
            max_point_density_departure_relative=float(np.max(abs(rho['mean'][a:b]/gas.rho0_kg_m3-1))),
            max_pressure_amplitude_over_p0=float(np.max(abs(p['hat'][a:b]))/pressure0),
            max_velocity_amplitude_over_c0=float(np.max(np.hypot(abs(ux['hat'][a:b]), abs(uy['hat'][a:b])))/gas.c0_m_s),
            max_temperature_amplitude_over_T0=float(np.max(abs(temperature['hat'][a:b]))/gas.T0_K),
            max_density_amplitude_over_rho0=float(np.max(abs(rho['hat'][a:b]))/gas.rho0_kg_m3))
        background['max_mean_velocity_over_c0'] = float(np.max(np.hypot(ux['mean'][a:b], uy['mean'][a:b]))/gas.c0_m_s)
        background['max_mean_pressure_departure_over_p0'] = float(np.max(abs(p['mean'][a:b]))/pressure0)
        background['passed'] = bool(all(value < LIMIT for key, value in background.items()
                                        if key.startswith('max_')))
        background['criterion'] = 'each listed ambient departure or perturbation < 1%; homogeneous linear acoustic applicability'
        contours.append(dict(index=index+1, half_width_m=float(np.max(abs(upper['x_m']))),
            height_m=float(np.max(upper['y_m'])), closed_normal_residual=float(closure),
            stationarity=stable, background=background,
            direct_probe_comparison=_comparisons(reconstructed[-1], direct['hat'][probe_indices])))
    arrays['kirchhoff_probe_p_hat_Pa'] = np.asarray(reconstructed)
    sensitivities = [dict(contours=[a+1, b+1], **_comparisons(reconstructed[a], reconstructed[b]))
                     for a, b in ((0, 1), (0, 2), (1, 2))]
    empty = np.empty((len(t), 0))
    film_checks = [_stationarity(_stationarity_input(t, t, heat_t, film_T[:, j], film_q[:, j], empty, empty),
                                frequency, end) for j in range(film_T.shape[1])]
    fit_checks = (p, ux, uy, temperature, rho, Tfit, qfit, direct)
    checks = dict(run_completed=bool(np.asarray(data.get('run_completed', False)).item())
                  and summary.get('status') == 'completed',
        state_validation=bool(summary.get('state_validation_passed') is True
            and summary.get('state_checks') and all(summary['state_checks'].values())),
        boundary_prerequisite=bool(boundary.get('boundary_validation_passed') is True
            and 0 < boundary.get('limit', float('inf')) <= LIMIT
            and 0 <= boundary.get('max_return_error_ratio', float('inf')) < LIMIT),
        complete_eight_cycle_observation=end >= 8,
        observation_time_coverage=all(item['passed'] for item in coverage.values()),
        fitted_signals_finite_and_conditioned=bool(all(np.isfinite(f['hat']).all()
            and np.all(f['condition_number'] <= 1e8) for f in fit_checks)),
        geometry=bool(geometry_pass), exterior_probes=bool(external_count_pass),
        film_stationarity=all(row['passed'] for row in film_checks),
        surface_stationarity=all(row['stationarity']['passed'] for row in contours),
        homogeneous_linear_acoustic_applicability=all(row['background']['passed'] for row in contours),
        kirchhoff_vs_direct=all(row['direct_probe_comparison']['passed'] for row in contours),
        contour_sensitivity=all(row['passed'] for row in sensitivities))
    result = dict(status='completed', kind='saved_2D_infinite_strip_nearfield_analysis',
        phase_convention='Re(hat * exp(+i omega t)); original LBM time origin',
        frequency_Hz=frequency, air=dict(rho0_kg_m3=gas.rho0_kg_m3, c0_m_s=gas.c0_m_s,
                                       T0_K=gas.T0_K, p0_Pa=float(pressure0)),
        fit_window=dict(start_cycle=start, end_cycle=end, start_s=start/frequency,
            end_s=end/frequency, actual_audit_end_s=float(audit_t[-1]), time_origin_shift_s=0.,
            pressure_and_temperature='step endpoint', wall_heat_flux='single-step interval midpoint',
            harmonics=3, detrend_order=1),
        tolerances=dict(amplitude_relative=LIMIT, phase_deg=PHASE_LIMIT_DEG,
                        physical_small_perturbation_relative=LIMIT),
        validation_probe_indices=probe_indices.tolist(), selected_export_contour=2,
        contours=contours, contour_sensitivity=sensitivities, observation_time_coverage=coverage,
        film_stationarity=dict(checked_bins=film_T.shape[1], bins=film_checks),
        fit_diagnostics={name: _fit_diagnostic(fit) for name, fit in
            (('surface_pressure', p), ('surface_ux', ux), ('surface_uy', uy),
             ('surface_temperature', temperature), ('surface_density', rho),
             ('film_temperature', Tfit), ('single_side_actual_wall_heat_flux', qfit), ('direct_pressure', direct))},
        checks=checks, source_validation_passed=bool(all(checks.values())),
        periodic_response_stable=bool(checks['complete_eight_cycle_observation']
            and checks['observation_time_coverage'] and checks['film_stationarity'] and checks['surface_stationarity']),
        validation_passed=bool(checks['run_completed'] and checks['state_validation']
            and checks['complete_eight_cycle_observation'] and checks['observation_time_coverage'] and checks['film_stationarity']
            and checks['surface_stationarity']),
        validation_passed_scope='completed run, public state checks, and all-film/all-contour finite-window stationarity only',
        propagation_validation_passed=None, grid_independence_assessed=False,
        thermal_equilibrium_assessed=False,
        validation_scope='saved nearfield state, finite-window stationarity, homogeneous linear applicability, '
            'three closed curves versus direct exterior probes; no k-Wave or 45 mm validation',
        boundary_validation=boundary)
    return result, arrays


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def analyze_saved(case_dir, boundary_path=None):
    case_dir = Path(case_dir).resolve()
    boundary_path = (Path(boundary_path).resolve() if boundary_path is not None
                     else case_dir.parent.parent/'final/boundary_validation.json')
    paths = {name: case_dir/name for name in ('effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json')}
    effective = yaml.safe_load(paths['effective.yaml'].read_text(encoding='utf-8'))
    summary = json.loads(paths['summary.json'].read_text(encoding='utf-8'))
    manifest = json.loads(paths['manifest.json'].read_text(encoding='utf-8'))
    boundary = json.loads(boundary_path.read_text(encoding='utf-8'))
    with np.load(paths['trajectory.npz'], allow_pickle=False) as raw:
        result, arrays = analyze_data(raw, effective, summary, boundary)
    # A copied boolean from another batch is not evidence for this solver.
    # Compare against the simulation runtime, not a later analysis checkout.
    from Compute.finite_strip.run import validate_boundary_evidence
    try:
        if not manifest.get('source_id'):
            raise ValueError('raw simulation source identity is missing')
        validate_boundary_evidence(boundary_path.parent.parent,effective,
                                   source_id=manifest['source_id'],proof_path=boundary_path)
        binding=dict(passed=True)
    except (OSError,ValueError,KeyError) as exc:
        binding=dict(passed=False,reason=str(exc))
    result['boundary_evidence_binding']=binding
    result['checks']['boundary_evidence_bound']=binding['passed']
    result['source_validation_passed']=bool(all(result['checks'].values()))
    result['provenance'] = dict(case_directory=str(case_dir),
        raw_sha256={name: _sha256(path) for name, path in paths.items()},
        raw_manifest=manifest, boundary_path=str(boundary_path),
        boundary_sha256=_sha256(boundary_path), analysis_script_sha256=_sha256(__file__),
        analysis_runtime_source=capture_source(ROOT),
        note='Runtime hashes describe this analysis; raw manifest describes simulation. Hashes do not restore uncommitted code.')
    return result, arrays


def export_kwave_input(path, result, arrays, *, toolbox_path, diagnostic=False,
                       kwave_backend='cpu', unvalidated=False):
    """Export Gamma 2; opt-in unvalidated results retain every failed check."""
    if kwave_backend not in ('cpu', 'matlab-gpu'):
        raise ValueError('k-Wave backend must be cpu or matlab-gpu')
    checks = result.get('checks', {})
    if unvalidated:
        usable=('run_completed','fitted_signals_finite_and_conditioned','geometry')
        if not all(checks.get(name) is True for name in usable):
            raise ValueError('direct export requires completed usable finite source data and geometry')
    elif diagnostic:
        required = {'run_completed', 'state_validation', 'boundary_prerequisite',
            'complete_eight_cycle_observation', 'observation_time_coverage',
            'fitted_signals_finite_and_conditioned', 'geometry', 'exterior_probes',
            'film_stationarity', 'surface_stationarity',
            'homogeneous_linear_acoustic_applicability', 'kirchhoff_vs_direct',
            'contour_sensitivity', 'boundary_evidence_bound'}
        allowed = (result.get('status') == 'completed' and result.get('validation_passed') is True
            and result.get('source_validation_passed') is False and isinstance(checks, dict)
            and required.issubset(checks) and checks['kirchhoff_vs_direct'] is False
            and all(passed is True for name, passed in checks.items() if name != 'kirchhoff_vs_direct'))
        if not allowed:
            raise ValueError('diagnostic source validation permits only kirchhoff_vs_direct failure '
                             'after completed state and every other source check pass')
    elif result.get('source_validation_passed') is not True or not all(checks.values()):
        raise ValueError('source validation must pass before a k-Wave MAT is created')
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    a, b = np.asarray(arrays['surface_slices'], dtype=int)[1]
    upper = {key: np.asarray(arrays['surface_'+key])[a:b] for key in GEOMETRY_KEYS}
    closed = mirror_strip_surface(upper, arrays['surface_p_hat_Pa'][a:b],
                                 arrays['surface_ux_hat_m_s'][a:b], arrays['surface_uy_hat_m_s'][a:b])
    indices = np.asarray(arrays['validation_probe_indices'], dtype=int)
    px, py = arrays['probe_x_m'][indices], arrays['probe_y_m'][indices]
    dx, half_y = .00005, .049
    # Direct LBM probes may lie closer to Gamma than the propagation grid can
    # resolve. Keep their measured data below; move only propagation checks
    # along the outward vector from the nearest point on the Gamma 2 rectangle.
    width, height = np.max(abs(upper['x_m'])), np.max(upper['y_m'])
    anchor_x, anchor_y = np.clip(px, -width, width), np.clip(py, -height, height)
    outward_x, outward_y = px-anchor_x, py-anchor_y
    clearance = np.hypot(outward_x, outward_y)
    if np.any(clearance <= 0):
        raise ValueError('propagation checks require direct probes exterior to Gamma 2')
    outward_scale = np.maximum(1., 3*dx/clearance)
    exterior_x, exterior_y = anchor_x+outward_scale*outward_x, anchor_y+outward_scale*outward_y
    ox, oy = np.r_[exterior_x, 0., 0., -.0004, .0004], np.r_[exterior_y, .045, 0., 0., 0.]
    # Keep the historical 4 mm minimum and 3.1 mm source-to-domain margin.
    # Wider films need a wider acoustic grid; retain a cell after snapping
    # even when an exterior validation receiver extends beyond that margin.
    requested_half_x = max(.004, width+.0031, float(np.max(abs(ox)))+2*dx)
    half_x = float(np.ceil(requested_half_x/dx-1e-12)*dx)
    # Match MATLAB round for nonnegative grid indices, not Python's ties-to-even.
    sx = -half_x+np.floor((ox+half_x)/dx+.5)*dx
    sy = -half_y+np.floor((oy+half_y)/dx+.5)*dx
    snapped_clearance = np.hypot(np.maximum(abs(sx[:len(indices)])-width, 0.),
                                 np.maximum(abs(sy[:len(indices)])-height, 0.))
    # Bilinear deposition reaches one cell on each staggered lattice. The
    # three-cell requested margin leaves at least two after observer snapping.
    if np.any(snapped_clearance < 2*dx-1e-14):
        raise ValueError('snapped propagation observers approach the deposited source')
    rho0, c0 = result['air']['rho0_kg_m3'], result['air']['c0_m_s']
    reference = _propagate(closed, sx, sy, frequency=result['frequency_Hz'], rho0=rho0, c0=c0)
    payload = dict(**closed, rho0_kg_m3=rho0, c0_m_s=c0, frequency_Hz=result['frequency_Hz'],
        dx_m=dx, domain_half_width_m=half_x, domain_half_height_m=half_y,
        pml_size=32, evaluation_cycles=8.25, ramp_cycles=3., record_cycles=2., cfl=.3,
        toolbox_path=str(Path(toolbox_path).resolve()),
        kwave_backend_requested=kwave_backend,
        source_validation_passed=int(result['source_validation_passed']),
        diagnostic_mode=int(diagnostic or unvalidated),
        source_validation_bypassed=int((diagnostic or unvalidated) and not result['source_validation_passed']),
        diagnostic_scope=('User-requested direct propagation without boundary/source certification; '
            'all actual check results are retained. This is an unvalidated numerical result.' if unvalidated else
            'Diagnostic export bypasses kirchhoff_vs_direct only; source remains not validated. '
            'This is not a validated LBM-to-k-Wave chain or 45 mm result.' if diagnostic else
            'Source checks passed; propagation has not yet been validated.'),
        propagation_validation_performed=0, observer_x_m=ox, observer_y_m=oy,
        observer_role=np.array(['exterior_propagation_check']*len(indices)+['farfield_45mm']+['interior_cancel']*3, dtype=object),
        exterior_observer_requested_clearance_m=3*dx, exterior_observer_minimum_snapped_clearance_m=2*dx,
        expected_grid_observer_x_m=sx, expected_grid_observer_y_m=sy,
        expected_grid_kirchhoff_p_hat_Pa=reference,
        interior_cancel_target_Pa=0., nearfield_direct_x_m=px, nearfield_direct_y_m=py,
        nearfield_direct_p_hat_Pa=arrays['direct_probe_p_hat_Pa'][indices],
        nearfield_direct_probe_indices=indices, contour_index=2,
        phase_convention='exp(+i omega t), original LBM clock, no gain or phase adjustment',
        observer_reference_note='Exterior propagation checks are independent Kirchhoff propagation comparisons, '
            'not direct LBM observations. The reference uses snapped coordinates; '
            'nearfield_direct coordinates and phasors remain the saved actual LBM measurements.',
        normalization_note='Measured p and normal u; adapter applies source normalization. No extra factor two.',
        source_analysis_json=json.dumps(result, ensure_ascii=False, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        savemat(handle, payload, do_compression=True, oned_as='column')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--boundary', type=Path)
    parser.add_argument('--export-kwave', action='store_true')
    parser.add_argument('--toolbox', type=Path, default=ROOT/'k-wave-toolbox-version-1.4.1')
    parser.add_argument('--kwave-backend', choices=('cpu', 'matlab-gpu'), default='cpu')
    args = parser.parse_args(argv)
    name = valid_name(args.case)
    destination = args.batch.resolve()/'final'
    paths = [destination/(name+'_nearfield.json'), destination/(name+'_nearfield.npz')]
    mat = destination/(name+'_kwave_input.mat')
    for path in paths+([mat] if args.export_kwave else []):
        if path.exists():
            raise FileExistsError(path)
    result, arrays = analyze_saved(args.batch/'raw'/name, args.boundary)
    result['kwave_export'] = dict(requested=args.export_kwave, created=False, path=None)
    destination.mkdir(parents=True, exist_ok=True)
    if args.export_kwave and result['source_validation_passed']:
        export_kwave_input(mat, result, arrays, toolbox_path=args.toolbox, kwave_backend=args.kwave_backend)
        result['kwave_export'].update(created=True, path=str(mat))
    elif args.export_kwave:
        result['kwave_export']['reason'] = 'source validation failed; no MAT file created'
    dump(paths[0], result)
    with paths[1].open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    print(json.dumps(dict(status=result['status'], source_validation_passed=result['source_validation_passed'],
        propagation_validation_passed=None, failed_checks=[key for key, ok in result['checks'].items() if not ok],
        diagnostic_json=str(paths[0]), diagnostic_npz=str(paths[1]), kwave_export=result['kwave_export']), indent=2))
    return 0 if result['source_validation_passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
