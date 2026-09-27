"""Plot saved paired interface and thermal responses without rerunning the gas.

Quantitative grids test three bounded questions: transmission/reflection at the
interface, thermal-step profile and actual exchange differences, and agreement
of driven responses. Deterministic traces have no statistical error bars.
All plotted data come from raw four-file cases; no time or phase alignment is
applied. The step temperature panel explicitly zooms into the interface region.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from Compute.multi_grid.compare import load_case
from core.unit_mapping import create_unit_mapping
from reference.lim2013 import matched_gas


BLUE='#2477A8'
GRAY='#777777'
GOLD='#BD8636'
METHODS=(('Uniform fine',GRAY,'--'),('Two level',BLUE,'-'))


def _pair(batch,name):
    paths=[batch/'raw'/f'{name}_{kind}' for kind in ('uniform','two_level')]
    if not all((path/'manifest.json').exists() for path in paths):
        return None
    if not all(json.loads((path/'manifest.json').read_text(encoding='utf-8')).get('status')=='completed' for path in paths):
        return None
    cases=[load_case(path) for path in paths]
    if not np.array_equal(cases[0][3]['t_s'],cases[1][3]['t_s']):
        raise ValueError('paired curves require identical recorded times; no alignment is permitted')
    return cases


def _figure(rows):
    fig,axes=plt.subplots(rows,2,figsize=(180/25.4,(64*rows+12)/25.4),
                          layout='constrained',squeeze=False)
    for j,ax in enumerate(axes.flat):
        ax.set_label(chr(ord('a')+j))
        ax.tick_params(direction='out',length=3,width=.7)
        ax.grid(False)
    return fig,axes


def _method_legend(fig):
    handles=[Line2D([],[],color=color,linestyle=style,lw=1.2,label=label)
             for label,color,style in METHODS]
    fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)


def _export(fig,batch,name,audit_tools):
    final=batch/'final'; final.mkdir(exist_ok=True)
    fig.canvas.draw()
    if audit_tools is not None:
        spec=importlib.util.spec_from_file_location('audit_panel_alignment',audit_tools/'audit_panel_alignment.py')
        module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module
        spec.loader.exec_module(module)
        qa=batch/'_work'/'qa'; qa.mkdir(parents=True,exist_ok=True)
        module.require_matplotlib_panel_alignment(fig,json_out=qa/f'{name}.alignment.json',
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
    pdf=final/f'{name}.pdf'
    fig.savefig(pdf)
    fig.savefig(final/f'{name}.png',dpi=300)
    if audit_tools is not None:
        subprocess.run([sys.executable,str(audit_tools/'audit_pdf_text.py'),str(pdf),'--min-pt','5'],check=True)
        subprocess.run([sys.executable,str(audit_tools/'audit_figure_collisions.py'),str(pdf),
            '--json-out',str(qa/f'{name}.collision.json'),
            '--overlay-pdf',str(qa/f'{name}.collision.pdf')],check=True)
    else:
        print(f'  {name}: exported; rendered alignment/collision QA not requested')
    plt.close(fig)
    return pdf


def plot_pulses(batch,audit_tools=None):
    pairs=[(name,_pair(batch,name)) for name in ('pulse_up','pulse_down')]
    pairs=[item for item in pairs if item[1] is not None]
    if not pairs:
        return None
    fig,axes=_figure(len(pairs)); _method_legend(fig)
    for row,(name,cases) in enumerate(pairs):
        config=cases[0][0]; run=config['run']; pulse=run['interface_pulse']
        gas=matched_gas(create_unit_mapping(config)); direction=pulse['direction']
        source=pulse['center_fraction']*float(cases[0][3]['actual_length_m'])
        face=pulse['interface_m']; incident=0 if direction==1 else 1; transmitted=1-incident
        backward=[]
        for case,(label,color,style) in zip(cases,METHODS):
            data=case[3]; t=data['t_s']; z=data['probe_z_m']
            dp=data['pressure_Pa']-gas.p0_Pa
            rhocu=gas.rho0_kg_m3*gas.c0_m_s*data['velocity_m_s']
            forward=.5*(dp+direction*rhocu); backward.append(.5*(dp-direction*rhocu))
            axes[row,0].plot(t*1e6,forward[:,transmitted],color=color,ls=style,lw=1.1)
        axes[row,1].plot(t*1e6,1e3*(backward[1][:,incident]-backward[0][:,incident]),color=BLUE,lw=1.1)
        axes[row,1].axhline(0,color=GRAY,ls=':',lw=.7)
        half=4*pulse['width_fraction']*float(cases[0][3]['actual_length_m'])/gas.c0_m_s
        return_time=(abs(face-source)+abs(face-z[incident]))/gas.c0_m_s
        transmission_time=abs(z[transmitted]-source)/gas.c0_m_s
        for ax,center in zip(axes[row],(transmission_time,return_time)):
            ax.axvspan((center-half)*1e6,(center+half)*1e6,color=BLUE,alpha=.07,lw=0)
            ax.set_xlabel('Time (µs)')
        direction_label='Fine → coarse' if direction==1 else 'Coarse → fine'
        axes[row,0].set_title(f'{chr(97+2*row)}  {direction_label}: transmission',loc='left',fontweight='bold')
        axes[row,1].set_title(f'{chr(98+2*row)}  Reverse wave: two level − uniform',loc='left',fontweight='bold')
        axes[row,0].set_ylabel('Forward pressure (Pa)')
        axes[row,1].set_ylabel('Reverse pressure difference (mPa)')
    return _export(fig,batch,'interface_pulses',audit_tools)


def plot_step(batch,audit_tools=None):
    cases=_pair(batch,'step')
    if cases is None:
        return None
    fig,axes=_figure(1); a,b=axes[0]; _method_legend(fig)
    config=cases[0][0]; gas=matched_gas(create_unit_mapping(config))
    face=config['run']['thermal_step']['interface_m']; uniform=cases[0][3]; two_level=cases[1][3]
    for case,(label,color,style) in zip(cases,METHODS):
        data=case[3]
        a.plot(data['z_m']*1e6,(data['profile_temperature_K'][-1]-gas.T0_K)*1e3,
               color=color,ls=style,lw=1.1)
    a.axvline(face*1e6,color=GOLD,ls=':',lw=.9)
    a.set_xlim(0,4*face*1e6)
    a.set_xlabel('Distance from film (µm)'); a.set_ylabel('Gas temperature increment (mK)')
    a.set_title('a  Final temperature: interface region',loc='left',fontweight='bold')
    ends=two_level['sync_fine_step'].astype(int)-1; t=two_level['t_s']*1e6
    wall=(np.cumsum(two_level['wall_energy_J_m2'])[ends]-np.cumsum(uniform['wall_energy_J_m2'])[ends])
    interface=np.cumsum(two_level['interface_exchange'][:,1,3]-uniform['interface_exchange'][:,1,3])
    b.plot(t,1e3*wall,color=BLUE,lw=1.1,label='Wall exchange')
    b.plot(t,1e3*interface,color=GOLD,lw=1.1,label='Into upper block')
    b.axhline(0,color=GRAY,ls=':',lw=.7)
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2,frameon=False)
    b.set_xlabel('Time (µs)'); b.set_ylabel('Cumulative energy difference (mJ/m²)')
    b.set_title('b  Actual exchange: two level − uniform',loc='left',fontweight='bold')
    return _export(fig,batch,'thermal_step',audit_tools)


def plot_responses(batch,audit_tools=None):
    pairs=[(name,_pair(batch,name)) for name in ('thermal','power')]
    pairs=[item for item in pairs if item[1] is not None]
    if not pairs:
        return None
    fig,axes=_figure(len(pairs)); _method_legend(fig)
    for row,(name,cases) in enumerate(pairs):
        config=cases[0][0]; run=config['run']; gas=matched_gas(create_unit_mapping(config))
        probe=min(1,len(cases[0][3]['probe_z_m'])-1)
        for case,(label,color,style) in zip(cases,METHODS):
            data=case[3]
            if name=='thermal':
                axes[row,0].plot(data['t_s']*1e6,data['pressure_Pa'][:,probe]-gas.p0_Pa,color=color,ls=style,lw=1.1)
                axes[row,1].plot(data['heat_time_s']*1e6,data['wall_flux_W_m2'],color=color,ls=style,lw=1.1)
                for ax in axes[row]:
                    ax.set_xlim((run['cycles']-2)/run['frequency_Hz']*1e6,run['cycles']/run['frequency_Hz']*1e6)
                    ax.set_xlabel('Time (µs)')
            else:
                axes[row,0].plot(data['film_t_s']*1e3,data['film_temperature_K'],color=color,ls=style,lw=1.1)
                axes[row,1].plot(data['t_s']*1e3,data['pressure_Pa'][:,probe]-gas.p0_Pa,color=color,ls=style,lw=1.1)
                for ax in axes[row]:
                    ax.set_xlabel('Time (ms)')
        position=cases[0][3]['probe_z_m'][probe]*1e6
        if name=='thermal':
            titles=(f'Given temperature: pressure at {position:g} µm','Given temperature: actual wall heat flux')
            labels=('Direct gas pressure increment (Pa)','Single-side wall heat flux (W/m²)')
        else:
            titles=('Power cold start: film temperature',f'Power cold start: pressure at {position:g} µm')
            labels=('Film temperature (K)','Direct gas pressure increment (Pa)')
        for j,(ax,title,label) in enumerate(zip(axes[row],titles,labels)):
            ax.set_title(f'{chr(97+2*row+j)}  {title}',loc='left',fontweight='bold')
            ax.set_ylabel(label)
    return _export(fig,batch,'driven_responses',audit_tools)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--audit-tools',type=Path,help='Directory containing rendered figure audit scripts')
    args=parser.parse_args()
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.titlesize':8,'axes.labelsize':8,'legend.fontsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False})
    for plot in (plot_pulses,plot_step,plot_responses):
        output=plot(args.batch,args.audit_tools)
        if output is not None:
            print(output)


if __name__=='__main__':
    main()
