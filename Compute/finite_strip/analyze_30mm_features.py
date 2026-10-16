"""Read saved strip data: edge response, acoustic load, and near-wall diagnostics.

No solver is advanced. Peak phasors retain exp(+i omega t); wall heat is
actual one-sided exchange. Scalar q/T is a response ratio for this case.
Figures are deterministic diagnostics without statistical error bars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from postproc.multiharmonic_fit import fit_multiharmonic
from reference.lim2013 import matched_gas, solve_lim2013_frequency


def record(z):
    return dict(amplitude=float(abs(z)), phase_deg=float(np.angle(z, deg=True)),
                real=float(np.real(z)), imag=float(np.imag(z)))


def fit(t, values, frequency, start, end, order=1):
    mask = (t*frequency >= start-1e-10) & (t*frequency < end-1e-10)
    return fit_multiharmonic(t[mask], values[mask], 2*np.pi*frequency,
                            n_harmonics=3, detrend_order=order)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path, required=True)
    args = parser.parse_args()
    raw, final = args.batch/'raw/power_r2', args.batch/'final'
    config = yaml.safe_load((raw/'effective.yaml').read_text(encoding='utf-8'))
    analysis = json.loads((final/'power_r2_nearfield.json').read_text(encoding='utf-8'))
    summary = json.loads((raw/'summary.json').read_text(encoding='utf-8'))
    with np.load(final/'power_r2_nearfield.npz', allow_pickle=False) as saved:
        z = {k: saved[k] for k in saved.files}
    m = create_unit_mapping(config)
    run = config['strip']; f = run['frequency_Hz']; gas = matched_gas(m)
    x, T, q = (z[k] for k in ('film_x_m', 'film_T_hat_K', 'film_q_hat_W_m2'))
    dx = m.lattice.dx_m; centre = int(np.argmin(abs(x)))
    distance = run['film_width_m']/2-abs(x)
    Q = np.sum(q)*dx; Q0 = q[centre]*run['film_width_m']
    replaced = Q-dx*(q[0]+q[-1]-2*q[centre])
    metrics = dict(status=summary['status'], phase_convention=analysis['phase_convention'],
        fit_window=analysis['fit_window'], raw_validation_passed=summary['validation_passed'],
        nearfield_validation_passed=analysis['validation_passed'],
        nearfield_validation_scope=analysis['validation_passed_scope'],
        source_validation_passed=analysis['source_validation_passed'],
        grid_independence_assessed=analysis['grid_independence_assessed'],
        interpretation='Diagnostic analysis of one 2D infinite-z strip; no new simulation or validation gates.',
        centre={k: record(v[centre]) for k, v in [('T_K', T), ('q_W_m2', q), ('q_over_T_W_m2K', q/T)]},
        end={k: record(v[-1]) for k, v in [('T_K', T), ('q_W_m2', q), ('q_over_T_W_m2K', q/T)]},
        end_T_ratio=float(abs(T[-1]/T[centre])), end_q_ratio=float(abs(q[-1]/q[centre])),
        next_bin_q_ratio=float(abs(q[-2]/q[centre])),
        integrated_one_sided_Q_W_m=record(Q), uniform_centre_Q_W_m=record(Q0),
        whole_strip_amplitude_change_percent=float(100*(abs(Q/Q0)-1)),
        whole_strip_phase_change_deg=float(np.angle(Q/Q0, deg=True)),
        end_replacement_amplitude_change_percent=float(100*(abs(replaced/Q)-1)),
        end_replacement_phase_change_deg=float(np.angle(replaced/Q, deg=True)),
        left_right_T_error=float(np.max(abs(T-T[::-1]))/np.max(abs(T))),
        left_right_q_error=float(np.max(abs(q-q[::-1]))/np.max(abs(q))),
        scales=dict(thermal_um=float(np.sqrt(gas.alpha_m2_s/(np.pi*f))*1e6),
            viscous_um=float(np.sqrt(m.physical.nu0_m2_s/(np.pi*f))*1e6),
            film_diffusion_um=float(np.sqrt(run['sheet_conductance_W_K']/(run['C_A_J_m2K']*np.pi*f))*1e6),
            wavelength_mm=gas.c0_m_s/f*1e3), edge_bands=[], contours=[])
    for width in (4, 20, 50, 100, 200):
        weights = np.clip((width*1e-6-distance+dx/2)/dx, 0, 1)*dx
        band = np.sum(q*weights); extra = np.sum((q-q[centre])*weights)
        metrics['edge_bands'].append(dict(each_edge_width_um=width,
            Q_W_m=record(band), excess_vs_centre_W_m=record(extra),
            amplitude_fraction_percent=float(100*abs(band/Q)),
            excess_amplitude_fraction_percent=float(100*abs(extra/Q))))
    tops = []
    for a, b in z['surface_slices']:
        ind = np.arange(a, b); top = ind[z['surface_normal_y'][ind] > .999]
        xx = z['surface_x_m'][top]; p = z['surface_p_hat_Pa'][top]
        uy = z['surface_uy_hat_m_s'][top]; ux = z['surface_ux_hat_m_s'][top]
        intensity = .5*p*np.conj(uy); lateral = .5*p*np.conj(ux)
        c = int(np.argmin(abs(xx))); j = int(np.argmax(abs(p)))
        jI = int(np.argmax(intensity.real)); internal = abs(xx) < .014
        ds = z['surface_ds_m'][ind]
        flux = np.sum(.5*z['surface_p_hat_Pa'][ind]*np.conj(z['surface_un_hat_m_s'][ind])*ds)
        metrics['contours'].append(dict(height_um=float(np.max(z['surface_y_m'][ind])*1e6),
            centre_pressure=record(p[c]), maximum_pressure=record(p[j]),
            pressure_max_x_mm=float(xx[j]*1e3), pressure_max_over_centre=float(abs(p[j]/p[c])),
            intensity_max_x_mm=float(xx[jI]*1e3),
            intensity_max_over_centre=float(intensity[jI].real/intensity[c].real),
            centre_normalized_impedance=record(p[c]/(gas.rho0_kg_m3*gas.c0_m_s*uy[c])),
            interior_uy_span_percent=float(100*np.ptp(abs(uy[internal]))/np.mean(abs(uy[internal]))),
            contour_pressure_velocity_flux_W_m=record(flux)))
        tops.append((xx, p, uy, intensity, lateral, c))
    probes = np.flatnonzero((z['probe_x_m'] >= -1e-12) & (abs(z['probe_x_m']) < .0001))
    probes = probes[np.argsort(z['probe_y_m'][probes])]
    results = [solve_lim2013_frequency(f, -run['mean_power_W_m2'], run['C_A_J_m2K'],
        area_m2=config['material']['nominal_projected_area_m2'], x_m=float(y),
        beta0_per_side_W_m2K=run['G_W_m2K']/2, gas=gas) for y in z['probe_y_m'][probes]]
    pref = np.array([r.p_hat_planar_exact_Pa for r in results])
    pac = np.array([r.p_hat_acoustic_exact_Pa for r in results])
    metrics['central_probes'] = [dict(index=int(i), x_um=float(z['probe_x_m'][i]*1e6),
        y_um=float(z['probe_y_m'][i]*1e6), measured=record(z['direct_probe_p_hat_Pa'][i]),
        planar_reference=record(pr), planar_thermal_over_acoustic=float(abs((pr-pa)/pa)))
        for i, pr, pa in zip(probes, pref, pac)]
    r = results[-1]
    metrics['planar_centre_comparison'] = {name: dict(reference=record(ref),
        amplitude_error_percent=float(100*(abs(actual/ref)-1)),
        phase_error_deg=float(np.angle(actual/ref, deg=True))) for name, actual, ref in
        [('T', T[centre], r.T_s_hat_exact_K), ('q', q[centre], r.q_one_sided_hat_exact_W_m2),
         ('q_over_T', q[centre]/T[centre], r.q_one_sided_hat_exact_W_m2/r.T_s_hat_exact_K)]}
    with np.load(raw/'trajectory.npz', allow_pickle=False) as data:
        assert bool(data['run_completed'])
        t = data['t_s']; ht = data['heat_time_s']; ft = data['film_temperature_K']
        fq = data['film_wall_energy_J_m2']/m.lattice.dt_s; pp = data['pressure_perturbation_Pa']
        assert np.allclose(ht, t-.5*m.lattice.dt_s, rtol=0, atol=1e-14*m.lattice.dt_s)
        start, end = (analysis['fit_window'][k] for k in ('start_cycle', 'end_cycle'))
        metrics['refits'] = {}
        for name, clock, values, expected in [('centre_T', t, ft[:, centre], T[centre]),
            ('end_T', t, ft[:, -1], T[-1]), ('centre_q', ht, fq[:, centre], q[centre]),
            ('end_q', ht, fq[:, -1], q[-1]), ('nearwall_p', t, pp[:, probes[0]], z['direct_probe_p_hat_Pa'][probes[0]])]:
            fits = [fit(clock, values, f, lo, end, order) for lo, order in
                    [(start, 1), (end-4, 1), (start, 2)]]
            assert np.allclose(fits[0].harmonic(1), expected, rtol=2e-10, atol=1e-12)
            metrics['refits'][name] = dict(fundamental=[record(v.harmonic(1)) for v in fits],
                trend_slope_per_s=float(fits[0].trend_coeffs_raw[1]),
                residual_rms=float(fits[0].residual_rms),
                second_over_first=float(fits[0].amplitude(2)/fits[0].amplitude(1)))
        metrics['film_last_snapshot_K'] = dict(centre=float(ft[-1, centre]), end=float(ft[-1, -1]))
        metrics['observations'] = len(t)
    source = [raw/'effective.yaml', raw/'trajectory.npz', final/'power_r2_nearfield.npz']
    metrics['sha256'] = {p.relative_to(args.batch).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    expected = json.loads((final/'main_figures_20261014/figure_data_manifest.json').read_text(encoding='utf-8'))['source_sha256']
    assert all(value == expected[key] for key, value in metrics['sha256'].items())
    metrics['inputs_match_existing_source_manifest'] = True
    metrics['analysis_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metrics['analysis_git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    metrics['analysis_git_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    metrics['analysis_script_tracked_clean'] = not bool(subprocess.check_output(
        ['git', 'status', '--porcelain', '--', str(Path(__file__))], text=True).strip())
    metrics['reference_source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path('reference/lim2013.py'), Path('core/unit_mapping.py'), Path('postproc/multiharmonic_fit.py')]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'features.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 10, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.spines.right': False, 'axes.spines.top': False, 'legend.frameon': False})

    def save(fig, name):
        fig.tight_layout()
        require_matplotlib_panel_alignment(fig, json_out=args.output/(name+'.alignment.json'), strict=True)
        fig.savefig(args.output/(name+'.pdf'))
        fig.savefig(args.output/(name+'.svg'))
        fig.savefig(args.output/(name+'.png'), dpi=600)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2047244094, 3.3))  # 183 mm diagnostic figure
    select = (x > 0) & (distance < .0002)
    for value, label, color in [(T, 'Temperature', '#0072B2'), (q, 'Wall heat flux', '#D55E00'),
                                (q/T, 'Response ratio q/T', '#757575')]:
        ax.plot(distance[select]*1e6, abs(value[select]/value[centre]), '.-', ms=3, lw=1, label=label, color=color)
    ax.axhline(1, lw=.6, color='#AAAAAA', zorder=0)
    ax.set(xlabel='Distance inward from right film edge (µm)', ylabel='Amplitude / centre amplitude',
           title='Large terminal response is confined to one cell', xlim=(0, 200))
    ax.legend(loc='upper right'); save(fig, 'edge_response')
    fig, ax = plt.subplots(figsize=(7.2047244094, 3.3))
    xx, p, uy, intensity, lateral, c = tops[1]
    for val, label, color in [(abs(p)/abs(p[c]), 'Pressure', '#0072B2'),
        (abs(uy)/abs(uy[c]), 'Normal velocity', '#757575'),
        (intensity.real/intensity[c].real, 'Upward active intensity', '#D55E00')]:
        ax.plot(xx*1e3, val, lw=1, label=label, color=color)
    ax.set(xlabel='Position along film (mm)', ylabel='Value / centre value',
        title='Acoustic load varies despite nearly uniform wall heat flux; y = 100 µm')
    ax.legend(loc='lower center', ncol=3); save(fig, 'acoustic_load')
    fig, ax = plt.subplots(figsize=(7.2047244094, 3.3))
    height = z['probe_y_m'][probes]*1e6
    ax.plot(height, np.angle(-z['direct_probe_p_hat_Pa'][probes], deg=True), 'o-',
            color='#0072B2', label='Saved central LBM probes', lw=1, ms=4)
    ax.plot(height, np.angle(-pref, deg=True), '--', color='#757575', label='Independent planar full solution', lw=1)
    ax.set(xlabel='Probe height (µm); central arc |x| ≤ 75 µm', ylabel='Pressure phase relative to AC power (°)',
        title='Near-wall phase anomaly is absent from the planar reference')
    ax.legend(loc='upper right'); save(fig, 'nearwall_phase')
    print(json.dumps({k: metrics[k] for k in ['status', 'observations', 'end_T_ratio', 'end_q_ratio',
        'whole_strip_amplitude_change_percent', 'end_replacement_amplitude_change_percent', 'refits']}, indent=2))


if __name__ == '__main__':
    main()
