"""Five evidence pages for a completed E1 run; Python/matplotlib only.

Heating, acoustic response, mean profiles, energy accounting and outlet state
answer separate questions. Envelopes include every saved sample in each cycle;
they are deterministic ranges, not confidence intervals. Energy curves show
exact cumulative interval sums at every cycle boundary. No solver is advanced.
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

from Compute.open_planar.analyze_e1_long import inspect_case, assess
from Compute.open_planar.plot_e1 import stationarity_ratio
from core.unit_mapping import create_unit_mapping


COLORS=('#0072B2','#D55E00','#009E73','#777777','#CC79A7')


def page(title):
    fig,axes=plt.subplots(2,2,figsize=(7.2,6.3))
    fig.subplots_adjust(left=.115,right=.965,bottom=.10,top=.82,wspace=.48,hspace=.85)
    fig.suptitle(title,fontsize=11,y=.99)
    for letter,ax in zip('abcd',axes.flat):
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-26,8),
            textcoords='offset points',fontweight='bold',fontsize=10)
        ax.ticklabel_format(axis='both',style='plain',useOffset=False)
    return fig,axes


def legend(ax,**kwargs):
    return ax.legend(loc='lower left',bbox_to_anchor=(0,1.025),borderaxespad=0,
                     fontsize=7.3,handlelength=1.5,**kwargs)


def envelope(ax,data,key,cycles,frequency,dt,offset=0.,color=COLORS[0],column=None):
    values=data[key] if column is None else data[key][:,column]
    lows=[]; highs=[]
    for c in cycles:
        a=int(round((c['cycle']-1)/frequency/dt)); b=int(round(c['cycle']/frequency/dt))
        lows.append(float(values[a:b].min())-offset); highs.append(float(values[a:b].max())-offset)
    times=[c['time_ms'] for c in cycles]
    ax.fill_between(times,lows,highs,color=color,alpha=.16)
    ax.plot(times,lows,color=color,lw=.6); ax.plot(times,highs,color=color,lw=.6)


def figures(data,config,summary,result,windows,profiles,cycles,balances):
    run=config['run']; m=create_unit_mapping(config); dt=m.lattice.dt_s
    frequency=run['frequency_Hz']; area=run['power']['area_m2']
    T0=summary['air']['T0_K']; p0=summary['air']['p0_Pa']; rho0=summary['air']['rho0_kg_m3']
    times=np.array([c['time_ms'] for c in cycles]); wt=np.array([w['mid_time_ms'] for w in windows])
    endpoints=np.r_[0,[int(round(c['cycle']/frequency/dt)) for c in cycles]].astype(int)
    et=endpoints*dt*1e3
    wvalues=lambda key:np.array([w[key] for w in windows])
    fig,axes=page('Mean heating and gas storage are different observables')
    ax=axes[0,0]
    ax.fill_between(times,[c['film_min_K']-T0 for c in cycles],
        [c['film_max_K']-T0 for c in cycles],color=COLORS[0],alpha=.16,label='Cycle range')
    ax.plot(times,[c['film_mean_K']-T0 for c in cycles],color=COLORS[0],label='Cycle mean')
    ax.plot(wt,wvalues('film_temperature_K')-T0,'o',color=COLORS[1],ms=3,label='10-cycle mean')
    ax.set_ylabel(r'$T_s-T_0$ (K)'); legend(ax,ncol=2)
    ax=axes[0,1]
    e=data['gas_energy_J_m2']-float(data['initial_gas_energy_J_m2'])
    envelope(ax,{'energy':e},'energy',cycles,frequency,dt)
    ax.plot(wt,wvalues('gas_energy_change_from_cold_J_m2'),'o-',color=COLORS[1],ms=3)
    ax.set_ylabel(r'One-side $\Delta E_g$ (J m$^{-2}$)')
    ax.set_title('Total storage: cycle range and 10-cycle mean',fontsize=8,pad=9)
    ax=axes[1,0]
    cv=m.physical.cp_J_kgK/m.physical.gamma
    sensible=(data['profile_density_kg_m3']*cv*(data['profile_temperature_K']-T0)).sum(axis=1)*m.lattice.dx_m
    ax.plot(data['profile_t_s']*1e3,sensible,color=COLORS[1])
    ax.set_ylabel(r'$\int \rho c_v(T-T_0)\,dz$ (J m$^{-2}$)')
    ax.set_title('Sensible excess: warming diagnostic',fontsize=8,pad=9)
    ax=axes[1,1]
    gasmass=data['profile_density_kg_m3'].sum(axis=1)*m.lattice.dx_m
    ax.plot(data['profile_t_s']*1e3,(gasmass/gasmass[0]-1)*100,color=COLORS[0])
    ax.set_ylabel(r'Gas mass change (%)')
    ax.set_title('Mass leaves the open control volume',fontsize=8,pad=9)
    for ax in axes.flat: ax.set_xlabel('Time (ms)')
    yield 'heating',fig

    fig,axes=page('Acoustic response from startup through the full observation')
    diagnostics=summary['stationarity']; at=np.array([s['time_s'] for s in diagnostics])*1e3
    for j,z in enumerate(data['probe_z_m']):
        color=COLORS[j]; label=f'{z*1e3:.3f} mm'
        envelope(axes[0,0],data,'pressure_Pa',cycles,frequency,dt,p0,color,j)
        axes[0,0].plot([],[],color=color,label=label)
        amp=np.array([s['channels'][f'pressure_{j}'].get('amplitude',np.nan) for s in diagnostics])
        phase=np.array([s['channels'][f'pressure_{j}'].get('phase_relative_power_deg',np.nan) for s in diagnostics])
        axes[0,1].plot(at,amp,color=color,label=label)
        axes[1,0].plot(at,np.degrees(np.unwrap(np.radians(phase))),color=color,label=label)
    axes[0,0].set_ylabel(r'$p-p_0$ range (Pa)'); legend(axes[0,0],ncol=2)
    axes[0,1].set_ylabel(r'$|\widehat{p}_1|$ (Pa)'); axes[0,1].set_title('Two-cycle harmonic fit',fontsize=8,pad=9)
    axes[1,0].set_ylabel('Phase relative to power (deg)')
    axes[1,0].set_title(r'$\exp(+i\omega t)$ convention',fontsize=8,pad=9)
    ratio=np.array([stationarity_ratio(s,run['analysis']) for s in diagnostics])
    axes[1,1].plot(at,ratio,color=COLORS[0],label='All response channels')
    axes[1,1].axhline(1,color=COLORS[1],ls='--',label='Acceptance limit')
    axes[1,1].set_yscale('log'); axes[1,1].set_ylabel('Worst criterion / limit')
    legend(axes[1,1])
    for ax in axes.flat: ax.set_xlabel('Time / fit-window end (ms)')
    yield 'acoustics',fig

    fig,axes=page('Ten-cycle means test continued thermal and flow evolution')
    z=data['z_m']*1e3
    cmap=plt.get_cmap('viridis')
    for j,(w,profile) in enumerate(zip(windows,profiles)):
        color=cmap(j/max(1,len(windows)-1)); label=f"{w['start_cycle']}-{w['end_cycle']}"
        axes[0,0].plot(z,profile['temperature_K']-T0,color=color,label=label)
        axes[0,1].plot(z,profile['velocity_m_s'],color=color)
    axes[0,0].set_ylabel(r'$\langle T\rangle-T_0$ (K)'); axes[0,0].set_xlabel('Distance (mm)')
    legend(axes[0,0],ncol=5,columnspacing=.5)
    axes[0,1].set_ylabel(r'$\langle u_y\rangle$ (m s$^{-1}$)'); axes[0,1].set_xlabel('Distance (mm)')
    axes[0,1].set_title('Same window colours as temperature',fontsize=8,pad=9)
    transitions=result['all_window_transitions']; tt=[s['current_window'][1]/frequency*1e3 for s in transitions]
    for key,label,color in [('film','Film',COLORS[0]),('profile','Temperature profile',COLORS[1])]:
        vals=[s['film_difference_K']/s['film_limit_K'] if key=='film' else s['profile_max_difference_K']/s['profile_limit_K'] for s in transitions]
        axes[1,0].plot(tt,vals,'o-',ms=3,color=color,label=label)
    axes[1,0].axhline(1,color=COLORS[3],ls='--'); axes[1,0].set_yscale('log')
    axes[1,0].set_ylabel('Mean-temperature change / limit'); legend(axes[1,0])
    axes[1,0].set_xlabel('Window end time (ms)')
    vals=[s['velocity_max_difference_m_s']/s['velocity_limit_m_s'] for s in transitions]
    axes[1,1].plot(tt,vals,'o-',ms=3,color=COLORS[0],label='Mean velocity profile')
    axes[1,1].axhline(1,color=COLORS[3],ls='--',label='Acceptance limit')
    axes[1,1].set_yscale('log'); axes[1,1].set_ylabel('Mean-velocity change / limit')
    axes[1,1].set_xlabel('Window end time (ms)'); legend(axes[1,1])
    yield 'mean_profiles',fig

    fig,axes=page('Actual energy exchanges close against independent storage')
    quantities=[(np.cumsum(data['input_energy_J_m2']),'Input'),
        (balances['film_storage']+2*balances['gas_storage'],'Film + gas storage'),
        (np.cumsum(data['leak_energy_J_m2']),'Additional leak'),
        (-2*np.cumsum(data['audit'][:,3]),'Outlet export')]
    for j,(value,label) in enumerate(quantities):
        axes[0,0].plot(et,np.r_[0,value][endpoints]*area*1e3,color=COLORS[j],label=label)
    axes[0,0].set_ylabel('Cumulative energy (mJ)'); legend(axes[0,0],ncol=2)
    for j,(key,label) in enumerate([('input_W','Input'),('two_side_wall_W','Wall, two sides'),
        ('leak_W','Additional leak'),('two_side_outlet_export_W','Outlet, two sides')]):
        axes[0,1].plot(wt,wvalues(key),'o-',ms=3,color=COLORS[j],label=label)
    axes[0,1].set_ylabel('10-cycle mean power (W)'); legend(axes[0,1],ncol=2)
    for j,(key,label) in enumerate([('film_storage_rate_W','Film'),('two_side_gas_storage_rate_W','Gas, two sides')]):
        axes[1,0].plot(wt,wvalues(key),'o-',ms=3,color=COLORS[j],label=label)
    threshold=.01*run['power']['mean_W']
    axes[1,0].axhline(threshold,color=COLORS[3],ls='--'); axes[1,0].axhline(-threshold,color=COLORS[3],ls='--')
    axes[1,0].set_ylabel('Independent storage rate (W)'); legend(axes[1,0],ncol=2)
    for j,key in enumerate(('film','gas','total')):
        vals=balances[key+'_cumulative']; lo=[]; hi=[]
        for a,b in zip(endpoints[:-1],endpoints[1:]):
            lo.append(float(vals[a:b].min())); hi.append(float(vals[a:b].max()))
        axes[1,1].fill_between(times,lo,hi,color=COLORS[j],alpha=.12)
        axes[1,1].plot(times,lo,color=COLORS[j],lw=.7,label=key.capitalize())
        axes[1,1].plot(times,hi,color=COLORS[j],lw=.7)
    tol=run['analysis']['cumulative_energy_J_m2']
    axes[1,1].axhline(tol,color=COLORS[3],ls='--'); axes[1,1].axhline(-tol,color=COLORS[3],ls='--')
    axes[1,1].set_yscale('symlog',linthresh=1e-12)
    axes[1,1].set_yticks([-tol,-tol/100,0,tol/100,tol])
    axes[1,1].set_ylabel(r'Balance residual (J m$^{-2}$)'); legend(axes[1,1],ncol=3)
    for ax in axes.flat: ax.set_xlabel('Time (ms)')
    yield 'energy',fig

    fig,axes=page('Outlet background: changes relative to the cold reference')
    for ax,key,offset,label in zip(axes.flat,
        ('outlet_temperature_K','outlet_density_kg_m3','outlet_velocity_m_s','outlet_pressure_Pa'),
        (T0,rho0,0.,p0),
        (r'$T_{out}-T_0$ (K)',r'$\rho_{out}-\rho_0$ (kg m$^{-3}$)',r'$u_{out}$ (m s$^{-1}$)',r'$p_{out}-p_0$ (Pa)')):
        envelope(ax,data,key,cycles,frequency,dt,offset)
        ax.plot(wt,wvalues(key)-offset,'o-',color=COLORS[1],ms=3,label='10-cycle mean')
        ax.plot([],[],color=COLORS[0],label='Cycle range')
        ax.set_xlabel('Time (ms)'); ax.set_ylabel(label); legend(ax,ncol=2)
    yield 'outlet',fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--alignment-script',type=Path,required=True)
    args=parser.parse_args(); case=args.case_dir.resolve()
    output=args.output_dir or case.parent.parent/'final'
    if (output/'e1_long_thermal.pdf').exists():
        raise FileExistsError('figures exist; use a new output directory')
    data,config,summary,manifest=inspect_case(case)
    result,windows,profiles,cycles,balances=assess(data,config,summary,manifest)
    output.mkdir(parents=True,exist_ok=True)
    qa=(case.parent.parent/'_work/figure_qa') if args.output_dir is None else output/'qa'
    qa.mkdir(parents=True,exist_ok=True)
    require_matplotlib_panel_alignment=runpy.run_path(str(args.alignment_script))['require_matplotlib_panel_alignment']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8.5,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'lines.linewidth':1.1,'legend.frameon':False})
    state=(f"Run: {manifest['status']} | Validation: {summary['validation_passed']} | "
           f"Periodic: {summary['periodic_response_stable']} | Mean thermal: {result['mean_thermal_stable']}")
    with PdfPages(output/'e1_long_thermal.pdf') as pdf:
        for name,fig in figures(data,config,summary,result,windows,profiles,cycles,balances):
            fig.text(.5,.935,state,ha='center',fontsize=7.4)
            fig.canvas.draw()
            require_matplotlib_panel_alignment(fig,json_out=qa/f'{name}.alignment.json',strict=True,
                      tolerance_pt=1.5,gutter_tolerance_pt=1.5,require_panel_labels=True)
            fig.savefig(output/f'e1_long_{name}.png',dpi=300)
            fig.savefig(output/f'e1_long_{name}.svg')
            pdf.savefig(fig,dpi=300)
            plt.close(fig)
    print(output/'e1_long_thermal.pdf')


if __name__=='__main__':
    main()
