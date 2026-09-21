"""Read-only E1 figures: startup, stationarity, energy and spatial fields.

Four evidence pages, one PDF and four PNG previews. No data are recalculated
by advancing the solver. Full time-series marks are rasterized in the PDF;
axes and labels remain editable. The source is trajectory.npz + summary.json.
"""
import argparse
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from config_io import load_config
from postproc.power import energy_balances

COLORS=('#0072B2','#D55E00','#009E73','#777777')


def stationarity_ratio(diagnostic, analysis):
    """An invalid channel makes the whole metric unavailable, never passing."""
    if not diagnostic['eligible']:
        return np.nan
    checks=[]
    for channel in diagnostic['channels'].values():
        if 'amplitude_span_relative' not in channel:
            return np.nan
        checks.extend([channel['amplitude_span_relative']/analysis['stationary_relative'],
                       channel['phase_span_deg']/analysis['stationary_phase_deg']])
        for kind in ('duration_sensitivity','detrend_sensitivity'):
            checks.extend([abs(channel[kind]['amplitude_error_percent'])/(100*analysis['stationary_relative']),
                           abs(channel[kind]['phase_error_deg'])/analysis['stationary_phase_deg']])
    return max(checks) if checks and np.all(np.isfinite(checks)) else np.nan


def figures(data,summary,run):
    """The four pages support distinct claims; uncertainty is numerical only."""
    t=data['t_s']*1e3; tq=data['heat_time_s']*1e3
    p0=summary['air']['p0_Pa']; T0=summary['air']['T0_K']
    diagnostics=summary['stationarity']
    frequency=run['frequency_Hz']
    fig,axes=plt.subplots(4,1,figsize=(7.2,8.4),sharex=True,layout='constrained')
    axes[0].plot(tq,data['power_W_m2']*run['power']['area_m2'],color=COLORS[0],rasterized=True)
    axes[0].set_ylabel('Input power (W)')
    axes[1].plot(t,data['film_temperature_K']-T0,color=COLORS[0],rasterized=True,label='Film temperature')
    if diagnostics:
        good=[h for h in diagnostics if 'mean' in h['channels']['film_temperature']]
        axes[1].plot([(h['cycle']-1)/frequency*1e3 for h in good],
            [h['channels']['film_temperature']['mean']-T0 for h in good],
            'o-',color=COLORS[1],ms=3,label='Two-cycle mean')
        axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    axes[1].set_ylabel(r'$T_s-T_0$ (K)')
    axes[2].plot(tq,data['wall_flux_W_m2'],color=COLORS[0],rasterized=True)
    axes[2].set_ylabel('One-side wall flux\n(W m$^{-2}$)')
    for j,z in enumerate(data['probe_z_m']):
        axes[3].plot(t,data['pressure_Pa'][:,j]-p0,color=COLORS[j%len(COLORS)],
                     lw=.8,label=f'{z*1e3:.3f} mm',rasterized=True)
    axes[3].set_ylabel(r'$p-p_0$ (Pa)'); axes[3].set_xlabel('Time (ms)')
    axes[3].legend(loc='upper left',bbox_to_anchor=(1.01,1.),title='Probe distance')
    fig.suptitle('E1 local source: cold-start power, heating and sound',fontsize=11)
    yield 'startup',fig,axes

    fig,axes=plt.subplots(2,2,figsize=(7.2,5.7),layout='constrained')
    time=np.array([h['time_s'] for h in diagnostics])*1e3
    def metric(channel,key):
        return np.array([h['channels'][channel].get(key,np.nan) for h in diagnostics])
    axes[0,0].plot(time,metric('film_temperature','amplitude'),'o-',color=COLORS[0],ms=3)
    axes[0,0].set_ylabel(r'Film $|\widehat{T}_{s,1}|$ (K)')
    for j,z in enumerate(data['probe_z_m']):
        axes[0,1].plot(time,metric(f'pressure_{j}','amplitude'),'o-',ms=3,
                      color=COLORS[j%len(COLORS)],label=f'{z*1e3:.3f} mm')
    axes[0,1].set_ylabel(r'$|\widehat{p}_1|$ (Pa)')
    axes[0,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    for j,(channel,label) in enumerate((('film_temperature','Film'),('wall_flux','Wall flux'))):
        phases=metric(channel,'phase_relative_power_deg')
        phases=np.degrees(np.unwrap(np.radians(phases)))
        axes[1,0].plot(time,phases,'o-',ms=3,color=COLORS[j],label=label)
    axes[1,0].set_ylabel('Phase relative to power (deg)')
    axes[1,0].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    ratios=np.array([stationarity_ratio(h,run['analysis']) for h in diagnostics])
    axes[1,1].plot(time,ratios,'o-',color=COLORS[0],ms=3)
    invalid=np.array([h['eligible'] for h in diagnostics],dtype=bool)&~np.isfinite(ratios)
    if invalid.any():
        axes[1,1].plot(time[invalid],np.full(invalid.sum(),.95),'x',color=COLORS[1],
                      transform=axes[1,1].get_xaxis_transform(),label='Invalid fit (fails)')
    axes[1,1].axhline(1,color=COLORS[1],ls='--',lw=1,label='Acceptance limit')
    axes[1,1].set_ylabel('Worst criterion / limit')
    axes[1,1].set_yscale('log')
    axes[1,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    for ax in axes.flat:
        ax.set_xlabel('Window end time (ms)')
    fig.suptitle('Finite-window periodic response; thermal equilibrium not assessed',fontsize=10)
    yield 'stationarity',fig,axes.flat

    b=energy_balances(data,run['film'])
    fig,axes=plt.subplots(2,1,figsize=(7.2,5.8),sharex=True,layout='constrained')
    quantities=((np.cumsum(data['input_energy_J_m2']),'Input'),
        (b['film_storage']+2*b['gas_storage'],'Film + two-side gas storage'),
        (np.cumsum(data['leak_energy_J_m2']),'Additional heat leak'),
        (-2*np.cumsum(data['audit'][:,3]),'Two-side outlet export'))
    for j,(values,label) in enumerate(quantities):
        axes[0].plot(t,values,color=COLORS[j],label=label,rasterized=True)
    axes[0].set_ylabel('Cumulative energy (J m$^{-2}$)')
    axes[0].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    for j,name in enumerate(('film','gas','total')):
        axes[1].plot(t,b[name+'_cumulative'],color=COLORS[j],label=name.capitalize(),rasterized=True)
    tolerance=run['analysis']['cumulative_energy_J_m2']
    axes[1].axhline(tolerance,color=COLORS[3],ls='--',lw=.8)
    axes[1].axhline(-tolerance,color=COLORS[3],ls='--',lw=.8)
    axes[1].set_ylabel('Storage-balance residual (J m$^{-2}$)')
    axes[1].set_xlabel('Time (ms)'); axes[1].set_yscale('symlog',linthresh=1e-12)
    axes[1].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    fig.suptitle('Independent storage and boundary-exchange audit',fontsize=11)
    yield 'energy',fig,axes

    fig,axes=plt.subplots(2,2,figsize=(7.2,6.0),layout='constrained')
    tp=data['profile_t_s']*1e3; z=data['z_m']*1e3
    meshes=[]
    for ax,values,cmap,label in (
        (axes[0,0],data['profile_temperature_K']-T0,'magma',r'$T-T_0$ (K)'),
        (axes[0,1],data['profile_pressure_Pa']-p0,'RdBu_r',r'$p-p_0$ (Pa)')):
        mesh=ax.pcolormesh(tp,z,values.T,shading='auto',cmap=cmap,rasterized=True)
        meshes.append(mesh)
        ax.set_xlabel('Time (ms)'); ax.set_ylabel('Distance (mm)')
        fig.colorbar(mesh,ax=ax,label=label,orientation='horizontal',pad=.12)
    desired=[1/frequency,6/frequency,float(data['profile_t_s'][-1])]
    indices=sorted({int(np.argmin(abs(data['profile_t_s']-value))) for value in desired})
    for j,index in enumerate(indices):
        label=f'{data["profile_t_s"][index]*1e3:.3f} ms'
        axes[1,0].plot(z,data['profile_temperature_K'][index]-T0,color=COLORS[j],label=label)
        axes[1,1].plot(z,data['profile_velocity_m_s'][index],color=COLORS[j],label=label)
    axes[1,0].set_ylabel(r'$T-T_0$ (K)'); axes[1,1].set_ylabel('Normal velocity (m s$^{-1}$)')
    for ax in axes[1]:
        ax.set_xlabel('Distance (mm)')
    axes[1,1].legend(loc='upper left',bbox_to_anchor=(1.01,1.))
    fig.suptitle('Spatial development inside the simulated planar domain',fontsize=11)
    yield 'fields',fig,axes.flat


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--alignment-script',type=Path,
                        help='optional external rendered-layout audit; reports go to batch/_work/figure_qa')
    args=parser.parse_args()
    case=args.case_dir.resolve(); batch=case.parent.parent
    summary=json.loads((case/'summary.json').read_text(encoding='utf-8'))
    if summary.get('status')!='completed' or summary.get('kind')!='power_driven_open_planar_local_source':
        raise ValueError('requires a completed power case; failed validation is shown, not concealed')
    config=load_config(case/'effective.yaml'); output=batch/'final'
    output.mkdir(parents=True,exist_ok=True)
    targets=[output/'e1_local_source.pdf',*[output/f'e1_{name}.png' for name in ('startup','stationarity','energy','fields')]]
    if any(path.exists() for path in targets):
        raise FileExistsError('E1 figures already exist; choose a new output batch')
    require_matplotlib_panel_alignment=None
    if args.alignment_script:
        require_matplotlib_panel_alignment=runpy.run_path(str(args.alignment_script))['require_matplotlib_panel_alignment']
        qa=batch/'_work/figure_qa'; qa.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'lines.linewidth':1.,'legend.frameon':False,'savefig.dpi':300})
    with np.load(case/'trajectory.npz') as data, PdfPages(targets[0]) as pdf:
        for name,fig,axes in figures(data,summary,config['run']):
            state=(f"Run: {summary['status']} | Validation: "
                   f"{'PASS' if summary['validation_passed'] else 'FAIL'} | Periodic response: "
                   f"{'stable' if summary['periodic_response_stable'] else 'not stable'}")
            fig.suptitle(fig._suptitle.get_text()+'\n'+state,fontsize=10)
            axes=list(axes)
            for letter,ax in zip('abcd',axes):
                ax.text(0.,1.04,letter,transform=ax.transAxes,fontweight='bold',fontsize=10)
            fig.canvas.draw()
            if require_matplotlib_panel_alignment:
                require_matplotlib_panel_alignment(fig,json_out=str(qa/f'{name}.alignment.json'),strict=True,
                          tolerance_pt=1.5,gutter_tolerance_pt=1.5)
            fig.savefig(output/f'e1_{name}.png',dpi=300)
            pdf.savefig(fig,dpi=300)
            plt.close(fig)
    print(output/'e1_local_source.pdf')


if __name__=='__main__':
    main()
