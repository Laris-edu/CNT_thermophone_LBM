"""Read-only matched-window E1 grid comparison; reuse public fits and balances.

The figure asks whether a coarser grid preserves the first 20-cycle response.
Four panels show mean film drift, spatial thermal bias, harmonic amplitude bias,
and harmonic phase bias. Differences are deterministic, not uncertainty bands.
Original numerical gates and exploratory grid criteria remain separate.
"""
import argparse
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import runpy
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config_io import load_config
from core.unit_mapping import create_unit_mapping
from postproc.diagnostics import compare, fit_window, record
from postproc.power import energy_balances
from simulation.output import dump
from Compute.open_planar.analyze_e1_long import mean_state


END_CYCLE = 20
MEAN_WINDOW = (10, 20)
FIT_WINDOW = (18, 20)
LIMIT_PERCENT = 5.
LIMIT_PHASE_DEG = 5.


def load_prefix(case):
    config = load_config(case / 'effective.yaml')
    manifest = json.loads((case / 'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((case / 'summary.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'completed' or summary['status'] != 'completed':
        raise ValueError('A running or failed case is not a comparison result')
    mapping = create_unit_mapping(config)
    steps = round(END_CYCLE / config['run']['frequency_Hz'] / mapping.lattice.dt_s)
    scalar_keys = ('t_s', 'heat_time_s', 'film_temperature_K', 'wall_flux_W_m2',
        'gas_energy_J_m2', 'mass_lu', 'input_energy_J_m2', 'leak_energy_J_m2',
        'outlet_temperature_K', 'outlet_density_kg_m3', 'outlet_velocity_m_s',
        'outlet_pressure_Pa', 'pressure_Pa', 'velocity_m_s', 'audit')
    constant_keys = ('initial_gas_energy_J_m2', 'initial_film_temperature_K',
        'initial_mass_lu', 'probe_z_m', 'z_m')
    with np.load(case / 'trajectory.npz') as saved:
        if not bool(saved['run_completed']) or saved['t_s'].size < steps:
            raise ValueError('Completed trajectory must cover 20 cycles')
        data = {key: saved[key][:steps].copy() for key in scalar_keys}
        data.update({key: saved[key].copy() for key in constant_keys})
        pt = saved['profile_t_s']
        keep = pt <= END_CYCLE / config['run']['frequency_Hz'] + 1e-14
        data['profile_t_s'] = pt[keep]
        for key in ('temperature_K', 'density_kg_m3', 'velocity_m_s', 'pressure_Pa'):
            data['profile_' + key] = saved['profile_' + key][keep].copy()
    if not np.isclose(data['t_s'][-1] * config['run']['frequency_Hz'], END_CYCLE):
        raise ValueError('Incorrect prefix duration')
    return dict(case=str(case.resolve()), config=config, manifest=manifest,
        summary=summary, mapping=mapping, data=data)


def summarize(item):
    data, config, mapping = item['data'], item['config'], item['mapping']
    run = config['run']; frequency = run['frequency_Hz']; area = run['power']['area_m2']
    dt = mapping.lattice.dt_s; t = np.r_[0., data['t_s']]
    Ts = np.r_[float(data['initial_film_temperature_K']), data['film_temperature_K']]
    start, end = np.array(MEAN_WINDOW) / frequency
    channels = [('film_temperature', data['t_s'], data['film_temperature_K']),
                ('wall_flux', data['heat_time_s'], data['wall_flux_W_m2'])]
    channels += [(f'pressure_{j}', data['t_s'], data['pressure_Pa'][:, j])
                 for j in range(data['pressure_Pa'].shape[1])]
    fits = {}
    for name, times, values in channels:
        z, diagnostic = fit_window(times, values, frequency, *FIT_WINDOW)
        fits[name] = dict(**record(z), diagnostic=diagnostic)
    lo, hi = [int(round(v / frequency / dt)) for v in MEAN_WINDOW]
    audit = data['audit']; interval = slice(lo, hi)
    means = dict(film_temperature_K=float(mean_state(t, Ts, start, end)))
    for key in ('outlet_temperature_K', 'outlet_density_kg_m3',
                'outlet_velocity_m_s', 'outlet_pressure_Pa'):
        means[key] = float(mean_state(data['t_s'], data[key], start, end))
    profiles = {key: mean_state(data['profile_t_s'], data['profile_' + key], start, end)
                for key in ('temperature_K', 'density_kg_m3', 'velocity_m_s', 'pressure_Pa')}
    means['mean_gas_temperature_K'] = float(profiles['temperature_K'].mean())
    mean_power = dict(input_W=float(data['input_energy_J_m2'][interval].sum()*area/(end-start)),
        two_side_wall_W=float(2*audit[interval, 2].sum()*area/(end-start)),
        leak_W=float(data['leak_energy_J_m2'][interval].sum()*area/(end-start)),
        two_side_outlet_export_W=float(-2*audit[interval, 3].sum()*area/(end-start)))
    balances = energy_balances(data, run['film'])
    film_rate = (balances['film_storage'][hi-1]-balances['film_storage'][lo-1])*area/(end-start)
    gas_rate = 2*(balances['gas_storage'][hi-1]-balances['gas_storage'][lo-1])*area/(end-start)
    mean_power.update(film_storage_rate_W=float(film_rate), two_side_gas_storage_rate_W=float(gas_rate),
        independent_balance_residual_W=float(mean_power['input_W']-mean_power['leak_W']
            -mean_power['two_side_outlet_export_W']-film_rate-gas_rate))
    maxima = {key: float(np.max(abs(balances[key]))) for key in
        ('film_step', 'gas_step', 'total_step', 'film_cumulative', 'gas_cumulative', 'total_cumulative')}
    maxima['cumulative_mass_lu'] = float(np.max(abs(data['mass_lu']
        -float(data['initial_mass_lu'])-np.cumsum(audit[:, 0]+audit[:, 1]))))
    cv = mapping.physical.cp_J_kgK/mapping.physical.gamma
    rho = data['profile_density_kg_m3']; temp = data['profile_temperature_K']
    velocity = data['profile_velocity_m_s']
    reconstructed = (rho*cv*temp + .5*rho*velocity**2).sum(axis=1)*mapping.lattice.dx_m
    recorded = np.interp(data['profile_t_s'], t,
        np.r_[float(data['initial_gas_energy_J_m2']), data['gas_energy_J_m2']])
    maxima['profile_energy_reconstruction_J_m2'] = float(np.max(abs(reconstructed-recorded)))
    cycles = [dict(cycle=c, film_temperature_K=float(mean_state(t, Ts, (c-1)/frequency, c/frequency)))
              for c in range(1, END_CYCLE+1)]
    full_energy = dict(input_J=float(data['input_energy_J_m2'].sum()*area),
        film_storage_J=float(balances['film_storage'][-1]*area),
        two_side_gas_storage_J=float(2*balances['gas_storage'][-1]*area),
        leak_J=float(data['leak_energy_J_m2'].sum()*area),
        two_side_wall_J=float(2*audit[:, 2].sum()*area),
        two_side_outlet_export_J=float(-2*audit[:, 3].sum()*area),
        independent_balance_residual_J=float(balances['total_cumulative'][-1]*area))
    return dict(fits=fits, means=means, profiles=profiles, mean_power=mean_power,
                audit_maxima=maxima, cycles=cycles, energy=full_energy)


def analyze(candidate, reference):
    cc, rc = candidate['config'], reference['config']
    for key in ('physical', 'transport', 'collision'):
        if cc[key] != rc[key]:
            raise ValueError(f'Different physics: {key}')
    cr, rr = deepcopy(cc['run']), deepcopy(rc['run'])
    for run in (cr, rr):
        run.pop('cycles'); run.pop('refine')
    if cr != rr:
        raise ValueError('Runs differ beyond refinement and duration')
    if candidate['manifest']['source_sha256'] != reference['manifest']['source_sha256']:
        raise ValueError('Public sources differ; comparison requires explicit review')
    c, r = summarize(candidate), summarize(reference)
    harmonic_rows = []
    for name, fit in c['fits'].items():
        other = r['fits'][name]
        error = compare(complex(fit['re'], fit['im']), complex(other['re'], other['im']))
        harmonic_rows.append(dict(channel=name, candidate_amplitude=fit['amplitude'],
            reference_amplitude=other['amplitude'], **error,
            passed=abs(error['amplitude_error_percent'])<=LIMIT_PERCENT
                   and abs(error['phase_error_deg'])<=LIMIT_PHASE_DEG))
    z, rz = candidate['data']['z_m'], reference['data']['z_m']
    if not (np.all(np.diff(rz)>0) and z[0]>=rz[0] and z[-1]<=rz[-1]):
        raise ValueError('Reference profile interpolation must not extrapolate')
    aligned = {key: np.interp(z, rz, values) for key, values in r['profiles'].items()}
    difference = {key: c['profiles'][key]-values for key, values in aligned.items()}
    T0 = candidate['mapping'].physical.T0_K
    film_difference = c['means']['film_temperature_K']-r['means']['film_temperature_K']
    film_percent = 100*film_difference/abs(r['means']['film_temperature_K']-T0)
    profile_K = float(np.max(abs(difference['temperature_K'])))
    profile_percent = 100*profile_K/float(np.max(abs(r['profiles']['temperature_K']-T0)))
    gates = dict(harmonics=all(row['passed'] for row in harmonic_rows),
        mean_film_temperature=abs(film_percent)<=LIMIT_PERCENT,
        mean_temperature_profile=profile_percent<=LIMIT_PERCENT)
    result = dict(candidate_case=candidate['case'], reference_case=reference['case'],
        status=candidate['summary']['status'], validation_passed=candidate['summary']['validation_passed'],
        numerical_gates=candidate['summary']['gates'],
        periodic_response_stable=candidate['summary']['periodic_response_stable'],
        mean_thermal_equilibrium_assessed=False, mean_thermal_stable=None,
        reason='20 cycles only; this is a grid comparison, not a long-time thermal-equilibrium test',
        public_source_hashes_equal=True, mean_window_cycles=MEAN_WINDOW, harmonic_window_cycles=FIT_WINDOW,
        comparison_limits=dict(amplitude_percent=LIMIT_PERCENT, phase_deg=LIMIT_PHASE_DEG,
            mean_temperature_excess_percent=LIMIT_PERCENT),
        grid_comparison_gates=gates, grid_comparison_passed=all(gates.values()),
        exploratory_feasibility_passed=bool(candidate['summary']['validation_passed'] and all(gates.values())),
        probe_coordinates_m=dict(candidate=candidate['data']['probe_z_m'].tolist(),
                                 reference=reference['data']['probe_z_m'].tolist()),
        harmonics=harmonic_rows, mean_film_difference_K=float(film_difference),
        mean_film_difference_percent_of_reference_excess=float(film_percent),
        mean_profile_max_difference_K=profile_K, mean_profile_difference_percent_of_reference_excess=profile_percent,
        mean_velocity_profile_max_difference_m_s=float(np.max(abs(difference['velocity_m_s']))),
        mean_power_differences_W={key:c['mean_power'][key]-r['mean_power'][key] for key in c['mean_power']},
        cases={label:dict(refine=item['config']['run']['refine'],
            run_commit=item['manifest']['git_commit'], steps_analyzed=len(item['data']['t_s']),
            final_case_elapsed_s=item['manifest']['elapsed_s'], means=stats['means'], mean_power=stats['mean_power'],
            energy_0_20_cycles=stats['energy'], audit_maxima_0_20_cycles=stats['audit_maxima'])
            for label,item,stats in [('candidate',candidate,c),('reference',reference,r)]},
        limits=['r4 is a numerical comparator, not an exact solution',
            'Native pressure probes differ slightly; no fitted phase correction is applied',
            'Mean profiles linearly interpolate r4 values to r2 cell centers without extrapolation',
            'This does not establish warm-background, long-time, domain or continuum-grid independence'])
    return result, c, r, aligned, difference


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def plot(result, candidate, c, r, difference, output, qa, alignment_script):
    # Quantitative grid: matched temporal, spatial, amplitude and phase differences.
    # One deterministic trajectory per grid; no confidence interval is implied.
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.linewidth':.8, 'pdf.fonttype':42, 'svg.fonttype':'none', 'legend.frameon':False})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.5))
    fig.subplots_adjust(left=.12, right=.97, bottom=.14, top=.85, wspace=.52, hspace=.68)
    fig.suptitle('E1: matched 20-cycle grid comparison', y=.98, fontsize=12)
    fig.text(.5,.92,'Differences: candidate minus r4; harmonics use cycles 18-20', ha='center', fontsize=8)
    for letter, ax in zip('abcd', axes.flat):
        ax.annotate(letter, xy=(0,1), xycoords='axes fraction', xytext=(-27,9),
                    textcoords='offset points', fontweight='bold', fontsize=11)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    axes[0,0].plot(range(1,21),[a['film_temperature_K']-b['film_temperature_K']
        for a,b in zip(c['cycles'],r['cycles'])], color='#0072B2', marker='o', ms=3)
    axes[0,0].set(xlabel='Acoustic cycle', ylabel=r'Cycle-mean $\Delta T_s$ (K)', title='Mean film-temperature bias')
    axes[0,1].plot(candidate['data']['z_m']*1e3,difference['temperature_K'],color='#0072B2')
    axes[0,1].set(xlabel='Distance from film (mm)', ylabel=r'Mean-profile $\Delta T$ (K)', title='Gas-temperature bias, cycles 10-20')
    labels=['Film','Wall','P1','P2','P3']; xx=np.arange(len(labels))
    for ax,key,title,ylabel in [(axes[1,0],'amplitude_error_percent','Fundamental amplitude', 'Amplitude difference (%)'),
                               (axes[1,1],'phase_error_deg','Fundamental phase', 'Phase difference (deg)')]:
        values=[v[key] for v in result['harmonics']]
        ax.scatter(xx,values,color='#0072B2',s=24,zorder=3)
        ax.vlines(xx,0,values,color='#0072B2',lw=1)
        ax.set_xticks(xx,labels); ax.set(xlim=(-.5,4.5),title=title,ylabel=ylabel)
        ax.axhline(0,color='#777777',lw=.7)
        ax.axhline(5,color='#D55E00',ls='--',lw=.8)
        ax.axhline(-5,color='#D55E00',ls='--',lw=.8)
        extent=max(6.,max(abs(np.asarray(values)))*1.3)
        ax.set_ylim(-extent,extent)
    fig.text(.5,.035,'Dashed lines: exploratory limits of 5% and 5 degrees; r4 is not an exact solution.',
             ha='center',fontsize=8)
    require_matplotlib_panel_alignment=runpy.run_path(str(alignment_script))['require_matplotlib_panel_alignment']
    require_matplotlib_panel_alignment(fig,json_out=qa/'comparison.alignment.json',strict=True,
                                      tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(output/'e1_r2_r4_comparison.pdf')
    fig.savefig(output/'e1_r2_r4_comparison.svg')
    fig.savefig(output/'e1_r2_r4_comparison.png',dpi=600)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--reference-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--alignment-script',type=Path,required=True)
    args=parser.parse_args()
    output=args.output or args.case_dir.parent.parent/'final'
    output.mkdir(parents=True,exist_ok=False)
    candidate,reference=load_prefix(args.case_dir),load_prefix(args.reference_dir)
    result,c,r,aligned,difference=analyze(candidate,reference)
    result['analysis_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['input_metadata_sha256']={label:{name:hashlib.sha256((Path(item['case'])/name).read_bytes()).hexdigest()
        for name in ('effective.yaml','manifest.json','summary.json')} for label,item in [('candidate',candidate),('reference',reference)]}
    dump(output/'grid_comparison.json',result)
    write_csv(output/'harmonic_comparison.csv',result['harmonics'])
    write_csv(output/'cycle_means.csv',[dict(cycle=a['cycle'],candidate_K=a['film_temperature_K'],
        reference_K=b['film_temperature_K'],difference_K=a['film_temperature_K']-b['film_temperature_K'])
        for a,b in zip(c['cycles'],r['cycles'])])
    write_csv(output/'mean_profiles.csv',[dict(z_m=float(z),**{f'{label}_{key}':float(values[key][j])
        for label,values in [('candidate',c['profiles']),('reference_interpolated',aligned),('difference',difference)]
        for key in values}) for j,z in enumerate(candidate['data']['z_m'])])
    write_csv(output/'mean_energy.csv',[dict(quantity=key,candidate_W=value,reference_W=r['mean_power'][key],
        difference_W=value-r['mean_power'][key]) for key,value in c['mean_power'].items()])
    qa=output.parent/'_work'/'grid_figure_qa' if args.output is None else output/'qa'
    qa.mkdir(parents=True,exist_ok=True)
    plot(result,candidate,c,r,difference,output,qa,args.alignment_script)
    print(json.dumps({key:result[key] for key in ('status','validation_passed','grid_comparison_passed',
        'exploratory_feasibility_passed','mean_film_difference_K','mean_profile_max_difference_K')},indent=2))


if __name__=='__main__':
    main()
