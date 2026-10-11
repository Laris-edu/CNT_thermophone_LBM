"""Three evidence figures from saved data, without advancing a solver.

Contract: (1) cold-start energy input produces the measured local chain;
(2) the resolved thermal layer feeds a planar acoustic wave, whose exterior
extension is a model; (3) conservation, stationarity and propagation are checked
separately. Quantitative grids, deterministic trajectories (no replicate CI),
editable PDF plus 300 dpi PNG. All reconstruction uses the original clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from Compute.film_to_sound.analyze import load_power_case, propagate
from Compute.open_planar.plot_e1 import stationarity_ratio
from postproc.multiharmonic_fit import fit_multiharmonic
from postproc.power import energy_balances

BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#777777'


def phasor(item):
    if item.get('re') is None or item.get('im') is None:
        raise ValueError('unresolved harmonic cannot be plotted as a measured wave')
    return complex(item['re'], item['im'])


def style():
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'lines.linewidth': 1.2, 'legend.frameon': False,
        'savefig.dpi': 300, 'path.simplify': False})


def layout(rows, title, result, footer):
    fig, axes = plt.subplots(rows, 2, figsize=(9.5, 9.4 if rows == 4 else 7.8))
    fig.subplots_adjust(left=.115, right=.96, bottom=.13, top=.855,
                        hspace=.60, wspace=.39)
    fig.suptitle(title, x=.115, y=.966, ha='left', fontsize=15, fontweight='bold')
    state = (f"Run: {result['power_case']['status']}  |  Public validation: "
             f"{result['power_case']['validation_passed']}  |  Propagation: "
             f"{result['propagation_validation_passed']}")
    fig.text(.115, .918, state, fontsize=9)
    fig.text(.115, .035, footer, fontsize=8, va='bottom', linespacing=1.6)
    for letter, ax in zip('abcdefgh', axes.flat):
        ax.text(-.25, 1.10, letter, transform=ax.transAxes, fontsize=11, fontweight='bold')
        ax.tick_params(direction='out', length=3, width=.7)
    return fig, axes


def chain_figure(config, summary, data, result):
    run = config['run']; frequency = run['frequency_Hz']
    start, end = result['window_cycles']; omega = 2*np.pi*frequency
    probe = int(np.argmin(abs(data['probe_z_m']-.0006)))
    t0 = run['film']['T_ref_K']; p0 = summary['air']['p0_Pa']
    fig, axes = layout(4, 'Heat input drives a local acoustic response', result,
        'Left: complete cold-start records; raw pressure includes mean drift. Right: fitted fundamental only.\n'
        'One upper gas domain; symmetric lower-side heat exchange enters the film equation with factor 2.')
    series = [
        (data['heat_time_s'], data['power_W_m2'], 'Areal input (W/m²)', result['channels']['power_W_m2']),
        (data['t_s'], data['film_temperature_K']-t0, 'Film excess (K)', result['channels']['film_temperature_K']),
        (data['heat_time_s'], data['wall_flux_W_m2'], 'One-side heat flux (W/m²)', result['channels']['wall_flux_W_m2']),
        (data['t_s'], data['pressure_Pa'][:, probe]-p0, 'Gas pressure excess (Pa)', result['probes'][probe]['pressure'])]
    time = np.linspace(start/frequency, end/frequency, 600)
    for row, (clock, values, label, amplitude) in enumerate(series):
        axes[row, 0].plot(clock*1e3, values, color=BLUE, rasterized=True)
        axes[row, 0].axvspan(start/frequency*1e3, end/frequency*1e3, color=GREY, alpha=.12)
        axes[row, 0].set_ylabel(label)
        axes[row, 1].plot(time*frequency, np.real(phasor(amplitude)*np.exp(1j*omega*time)), color=ORANGE)
        axes[row, 1].set_ylabel(label.replace('excess', 'fundamental'))
        axes[row, 1].set_xlim(start, end)
    axes[0, 0].set_title('Cold start', loc='left')
    axes[0, 1].set_title(f'Final complete cycles {start:g}–{end:g}', loc='left')
    axes[-1, 0].set_xlabel('Time (ms)')
    axes[-1, 1].set_xlabel('Cycle since original start')
    axes[-1, 0].set_title(f'Direct gas pressure at {data["probe_z_m"][probe]*1e3:.4f} mm', loc='left')
    return fig, list(axes.flat)


def profile_fits(config, data, window):
    frequency = config['run']['frequency_Hz']; t = data['profile_t_s']
    mask = (t*frequency >= window[0]-1e-10) & (t*frequency < window[1]-1e-10)
    if mask.sum() < 10:
        raise ValueError('insufficient profile samples in the analysis window')
    fits = [fit_multiharmonic(t[mask], column, 2*np.pi*frequency,
                             n_harmonics=3, detrend_order=1)
            for column in data['profile_pressure_Pa'][mask].T]
    return np.array([fit.harmonic(1) for fit in fits]), mask


def field_figure(config, summary, data, result):
    frequency = result['frequency_Hz']; window = result['window_cycles']
    hats, mask = profile_fits(config, data, window)
    primary = next(p for p in result['probes'] if np.isclose(p['nominal_z_m'], .0002))
    z = data['z_m']; length = float(data['actual_length_m'])
    target = np.linspace(primary['z_m'], .003, 400)
    modeled = propagate(phasor(primary['outgoing']), frequency, target-primary['z_m'], result['air']['c0_m_s'])
    fig, axes = layout(2, 'Thermal layer, direct sound and planar extension', result,
        'Direct gas profiles: last-two-cycle fits and phase-sample mean temperature. No geometric spreading.\n'
        'Beyond the shaded domain edge: lossless planar model; exterior pressure is not directly validated.')
    near = z <= .0003
    axes[0, 0].plot(z[near]*1e6, np.mean(data['profile_temperature_K'][mask], axis=0)[near]-summary['air']['T0_K'], color=ORANGE)
    axes[0, 0].set(xlabel='Height above film (µm)', ylabel='Mean gas temperature excess (K)')
    axes[0, 0].set_title('Resolved thermal layer', loc='left')
    t = np.linspace(window[0]/frequency, window[1]/frequency, 600)
    for nominal, color in ((.0002, BLUE), (.0006, GREEN)):
        p = next(p for p in result['probes'] if np.isclose(p['nominal_z_m'], nominal))
        axes[0, 1].plot(t*frequency, np.real(phasor(p['pressure'])*np.exp(2j*np.pi*frequency*t)),
                        color=color, label=f"Direct {p['z_m']*1e3:.3f} mm")
    axes[0, 1].plot(t*frequency, np.real(phasor(result['extrapolation']['pressure'])*np.exp(2j*np.pi*frequency*t)),
                    '--', color=ORANGE, label='Model 3 mm')
    axes[0, 1].set(xlabel='Cycle since original start', ylabel='Pressure fundamental (Pa)')
    axes[0, 1].set_title('Propagation delay', loc='left')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(.5, 1.01), ncol=1)
    axes[0, 1].set_ylim(-2.5*max(np.max(abs(hats)), abs(modeled[-1])), 2.5*max(np.max(abs(hats)), abs(modeled[-1])))
    axes[1, 0].plot(z*1e3, abs(hats), color=BLUE, label='Direct gas')
    axes[1, 0].plot(target*1e3, abs(modeled), '--', color=ORANGE, label='From 0.20 mm')
    power = phasor(result['channels']['power_W_m2'])
    axes[1, 1].plot(z*1e3, np.unwrap(np.angle(hats/power))*180/np.pi, color=BLUE)
    axes[1, 1].plot(target*1e3, np.unwrap(np.angle(modeled/power))*180/np.pi, '--', color=ORANGE)
    axes[1, 0].set_ylabel('Pressure amplitude (Pa)')
    axes[1, 1].set_ylabel('Phase relative to input power (deg)')
    for ax in axes[1]:
        ax.axvspan(length*1e3, 3., color=GREY, alpha=.10)
        ax.axvline(length*1e3, color=GREY, linewidth=.8, linestyle=':')
        ax.set(xlabel='Height above film (mm)', xlim=(0, 3))
    axes[1, 0].legend(loc='upper right')
    return fig, list(axes.flat)


def validation_figure(config, summary, data, result):
    fig, axes = layout(2, 'Independent checks bound the demonstration', result,
        'Each criterion is reported separately. Harmonic stationarity does not establish thermal equilibrium.\n'
        'Grid/domain independence and finite-film radiation are not assessed by these checks.')
    diagnostics = [d for d in summary['stationarity'] if d['eligible']]
    axes[0, 0].plot([d['cycle'] for d in diagnostics],
                    [stationarity_ratio(d, config['run']['analysis']) for d in diagnostics], 'o-', color=BLUE, ms=3)
    axes[0, 0].axhline(1, color=GREY, linestyle='--')
    axes[0, 0].set(xlabel='Complete cycles', ylabel='Maximum stationarity / limit')
    axes[0, 0].set_title('Finite-window periodic stability', loc='left')
    balances = energy_balances(data, config['run']['film'])
    for key, label, color in (('film_cumulative', 'Film', ORANGE), ('gas_cumulative', 'Upper gas', BLUE),
                              ('total_cumulative', 'Film + both gas sides', GREEN)):
        axes[0, 1].plot(data['t_s']*1e3, balances[key], color=color, label=label, rasterized=True)
    axes[0, 1].set_yscale('symlog', linthresh=1e-12)
    limit = config['run']['analysis']['cumulative_energy_J_m2']
    for sign in (-1, 1):
        axes[0, 1].axhline(sign*limit, color=GREY, linestyle='--', linewidth=.8)
    axes[0, 1].set(xlabel='Time (ms)', ylabel='Cumulative energy residual (J/m²)')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(.5, .85))
    interfaces = [p for p in result['probes'] if p['nominal_z_m'] in (.00012, .00016, .0002)]
    labels = ['Pulse'] + [f"{p['nominal_z_m']*1e3:.2f} mm" for p in interfaces]
    ratios = [result['pulse']['reflection_ratio']] + [p['incoming_over_outgoing'] for p in interfaces]
    axes[1, 0].plot(range(4), np.asarray(ratios)*100, 'o', color=BLUE)
    axes[1, 0].axhline(1, color=GREY, linestyle='--')
    axes[1, 0].set_xticks(range(4), labels)
    axes[1, 0].set(ylabel='Reflected or incoming / outgoing (%)', yscale='log')
    axes[1, 0].set_title('Outlet pulse and harmonic decomposition', loc='left')
    rows = result['propagation']
    x = np.arange(len(rows))
    for field, label, color, marker in (('amplitude_error_percent', 'Amplitude / 1%', BLUE, 'o'),
                                      ('phase_error_deg', 'Phase / 1 deg', ORANGE, 's')):
        values = [abs(r['comparison'][field]) if r['comparison'][field] is not None else np.nan for r in rows]
        axes[1, 1].plot(x, values, marker, color=color, label=label, ms=4)
    axes[1, 1].axhline(1, color=GREY, linestyle='--')
    axes[1, 1].set_xticks(x, [f"{r['source_nominal_z_m']*1e3:.2f}\nto {r['target_nominal_z_m']*1e3:.2f}" for r in rows])
    axes[1, 1].set(xlabel='Source to direct target (mm)', ylabel='Absolute error / acceptance limit')
    axes[1, 1].legend(loc='upper center', bbox_to_anchor=(.5, .95))
    return fig, list(axes.flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--alignment-script', type=Path, help='Optional external rendered-layout auditor')
    args = parser.parse_args(); batch = args.batch.resolve(); output = batch/'final'
    config, manifest, summary, data = load_power_case(batch)
    result = json.loads((output/'propagation.json').read_text(encoding='utf-8'))
    # The analysis must identify exactly the raw data being plotted.
    if Path(result['power_case']['path']).resolve() != batch/'raw'/'power_f20k_r2':
        raise ValueError('propagation analysis identifies a different power case')
    for case_name in ('power_f20k_r2', 'pulse_r2'):
        for name, digest in result['raw_sha256'][case_name].items():
            with (batch/'raw'/case_name/name).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'raw file changed since propagation analysis: {case_name}/{name}')
    names = ('chain', 'fields', 'validation')
    paths = [output/'film_to_sound.pdf', *(output/f'{name}.png' for name in names)]
    if any(p.exists() for p in paths):
        raise FileExistsError('figures already exist; use a new batch')
    require_matplotlib_panel_alignment = None
    if args.alignment_script:
        require_matplotlib_panel_alignment = runpy.run_path(str(args.alignment_script))['require_matplotlib_panel_alignment']
        qa = batch/'_work'/'figure_qa'; qa.mkdir(parents=True, exist_ok=True)
    style()
    makers = (chain_figure, field_figure, validation_figure)
    with PdfPages(paths[0]) as pdf:
        for name, maker, png in zip(names, makers, paths[1:]):
            fig, axes = maker(config, summary, data, result)
            fig.canvas.draw()
            if require_matplotlib_panel_alignment:
                require_matplotlib_panel_alignment(fig, axes=axes, json_out=qa/f'{name}.alignment.json',
                    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
            pdf.savefig(fig)
            fig.savefig(png, dpi=300)
            plt.close(fig)
    print(paths[0])


if __name__ == '__main__':
    main()
