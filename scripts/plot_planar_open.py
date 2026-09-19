"""单侧直接压力验证图：实测波形、网格误差与独立反射测量。

图形合同：验证 50 kHz 平面给温声压能否随网格加密趋近 Lim 精确解；
三面板分别展示主比较、离散误差、开放边界的独立控制。其他频率与域长
算例全部导出 CSV，并在报告中列出；图中只选同物理域的 50 kHz 网格组。
确定性模拟，无统计重复或误差条；7.1 x 3.0 inch，PDF/SVG 可编辑文字。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from postproc.multiharmonic_fit import fit_multiharmonic
from reference.lim2013 import characteristic_roots
from simulation.output import dump
from postproc.diagnostics import fit_window, compare
from reference.lim2013 import matched_gas
from config_io import load_config
from core.unit_mapping import create_unit_mapping
from simulation.output import OutputLayout, input_raw


def validation_groups(records):
    """Select the documented validation group by physics, independent of case names."""
    thermal=[r for r in records if r[1]['kind']=='prescribed_temperature_direct_pressure']
    grid=sorted([r for r in thermal if r[1]['frequency_Hz']==50000 and r[2]['length']==.001],
                key=lambda r:r[2]['refine'])
    if [r[2]['refine'] for r in grid] != [1,2,4]:
        raise ValueError('require complete 50 kHz r=1,2,4 grid group at length=0.001 m')
    if any(len(r[1]['probes'])<2 for r in grid):
        raise ValueError('validation waveform requires at least two pressure probes')
    pulses=sorted([r for r in records if r[1]['kind']=='outgoing_gaussian_pulse'],
                  key=lambda r:r[2]['refine'])
    if not {1,4}.issubset({r[2]['refine'] for r in pulses}):
        raise ValueError('require independent pulse cases at r=1 and r=4')
    short=grid[1]
    long=next((r for r in thermal if r[1]['frequency_Hz']==50000 and r[2]['refine']==2
        and r[2]['length']==.002 and r[1]['wall_amplitude_K']==short[1]['wall_amplitude_K']),None)
    return grid,pulses,(short,long) if long else None


def supplementary_checks(records, output, domain_pair=None):
    """相同时间窗的域长比较，以及独立读取气体温度剖面的误差。"""
    profiles=[]
    for path,s,a in records:
        if s['kind']!='prescribed_temperature_direct_pressure':
            continue
        data=np.load(path/'trajectory.npz')
        gas=matched_gas(create_unit_mapping(load_config(path/'effective.yaml')))
        frequency=s['frequency_Hz'];amp=s['wall_amplitude_K'];cycles=s['cycles']
        s1,s2,d1,d2=characteristic_roots(frequency,gas)
        c1=-d2*s2/(d1*s1-d2*s2)*(-1j*amp)
        c2=d1*s1/(d1*s1-d2*s2)*(-1j*amp)
        delta=np.sqrt(gas.alpha_m2_s/(np.pi*frequency))
        indices=np.flatnonzero(data['z_m']<=5*delta)
        errors=[]
        for index in indices:
            z=data['z_m'][index]
            got,_=fit_window(data['profile_t_s'],data['profile_temperature_K'][:,index],
                frequency,cycles-1,cycles)
            expected=c1*np.exp(-s1*z)+c2*np.exp(-s2*z)
            errors.append(abs(got-expected)/amp)
        profiles.append(dict(case=path.name,points=int(len(indices)),delta_T_m=float(delta),
            max_temperature_complex_error_over_wall_amplitude=float(max(errors))))
    domain=[]
    if domain_pair:
        short,long=domain_pair
        ds=np.load(short[0]/'trajectory.npz');dl=np.load(long[0]/'trajectory.npz')
        np.testing.assert_array_equal(ds['probe_z_m'],dl['probe_z_m'])
        end=min(short[1]['cycles'],long[1]['cycles']);frequency=short[1]['frequency_Hz']
        for j,z in enumerate(ds['probe_z_m']):
            hs,_=fit_window(ds['t_s'],ds['pressure_Pa'][:,j],frequency,end-1,end)
            hl,_=fit_window(dl['t_s'],dl['pressure_Pa'][:,j],frequency,end-1,end)
            domain.append(dict(z_m=float(z),start_cycle=end-1,end_cycle=end,**compare(hl,hs)))
    dump(output/'supplementary_checks.json',dict(temperature_profiles=profiles,
        domain_length_comparison=domain,
        domain_complex_change_below_one_percent=bool(domain and max(
            p['complex_relative_error'] for p in domain)<.01),
        source_summaries={p.name:hashlib.sha256((p/'summary.json').read_bytes()).hexdigest()
            for p,_,_ in records}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True,help='实验批次或 raw/ 目录')
    parser.add_argument('--output',type=Path,required=True,help='输出批次根目录，图表写入 final/')
    parser.add_argument('--qa-tools',type=Path,help='optional layout audit tools; omitted for ordinary plots')
    parser.add_argument('--overwrite',action='store_true',help='允许重新生成已有图表输出目录')
    parser.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf'])
    args=parser.parse_args()
    records=[]
    raw=input_raw(args.input)
    paths=sorted(raw.glob('*/summary.json'))
    if (raw/'summary.json').is_file():
        paths.insert(0,raw/'summary.json')
    for path in paths:
        summary=json.loads(path.read_text(encoding='utf8'))
        if summary.get('kind') not in ('prescribed_temperature_direct_pressure','outgoing_gaussian_pulse'):
            continue
        manifest=json.loads((path.parent/'manifest.json').read_text(encoding='utf8'))
        records.append((path.parent,summary,manifest['args']))
    if not records:
        raise ValueError('no completed cases')
    grid,pulses,domain_pair=validation_groups(records)
    layout=OutputLayout(args.output)
    layout.prepare_figures(overwrite=args.overwrite,qa=args.qa_tools is not None)
    plot_data=layout.raw/'plot_data'
    plot_data.mkdir(parents=True,exist_ok=True)
    supplementary_checks(records,plot_data,domain_pair)
    rows=[]
    for path,s,a in records:
        if s['kind']=='prescribed_temperature_direct_pressure':
            for p in s['probes']:
                rows.append(dict(case=path.name,frequency_Hz=s['frequency_Hz'],refine=a['refine'],
                    length_requested_m=a['length'],dx_m=s['mapping']['dx_m'],probe_z_m=p['z_m'],
                    pressure_amplitude_Pa=p['lbm']['amplitude'],pressure_rms_Pa=p['lbm']['amplitude']/np.sqrt(2),
                    pressure_phase_deg=p['lbm']['phase_deg'],lim_amplitude_Pa=p['lim']['amplitude'],
                    lim_phase_deg=p['lim']['phase_deg'],**p['comparison'],cycle_change=p['cycle_change']))
    with (layout.final/'source_data.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    fine_path,fine,_=grid[-1]
    data=np.load(fine_path/'trajectory.npz')
    freq=fine['frequency_Hz'];cycles=fine['cycles']
    mask=(data['t_s']*freq>=cycles-1)&(data['t_s']*freq<cycles)
    t=data['t_s'][mask];p=data['pressure_Pa'][mask,1]
    fit=fit_multiharmonic(t,p,2*np.pi*freq,n_harmonics=3,detrend_order=1)
    p=p-np.polynomial.polynomial.polyval(t,fit.trend_coeffs_raw)
    phat=fine['probes'][1]['lim'];phat=complex(phat['re'],phat['im'])
    exact=np.real(phat*np.exp(2j*np.pi*freq*t))
    np.savetxt(plot_data/'waveform.csv',np.column_stack((t,p,exact)),delimiter=',',
        header='time_s,pressure_detrended_Pa,lim_Pa',comments='')
    np.savetxt(plot_data/'reflection.csv',np.array([[s['mapping']['dx_m'],s['reflection_ratio']]
        for _,s,_ in pulses]),delimiter=',',header='dx_m,reflection_ratio',comments='')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False})
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.0))
    fig.subplots_adjust(left=.085,right=.98,bottom=.24,top=.74,wspace=.55)
    phase=t*freq-(cycles-1)
    axes[0].plot(phase,p,color='#287D8E',lw=1.7,label='Direct LBM pressure')
    axes[0].plot(phase,exact,color='#333333',lw=1.2,ls='--',label='Lim, prescribed T')
    axes[0].set(xlabel='Time / period',ylabel='Acoustic pressure (Pa)',xticks=[0,.5,1])
    axes[0].legend(loc='lower left',bbox_to_anchor=(-.08,1.04),fontsize=7,handlelength=2.)
    axes[1].plot([s['mapping']['dx_m']*1e6 for _,s,_ in grid],
        [100*max(p['comparison']['complex_relative_error'] for p in s['probes']) for _,s,_ in grid],
        color='#287D8E',marker='o',ms=4,lw=1.1)
    # 为接近零但非零的精细网格圆点留出物理尺寸余量，避免被坐标框裁掉。
    axes[1].set(xlabel='Grid spacing (µm)',ylabel='Pressure complex error (%)',ylim=(-.8,None))
    axes[2].plot([s['mapping']['dx_m']*1e6 for _,s,_ in pulses],
        [s['reflection_ratio']*100 for _,s,_ in pulses],color='#C47A29',marker='s',ms=4,lw=1.1)
    axes[2].set(xlabel='Grid spacing (µm)',ylabel='Pulse reflection (%)',ylim=(0,None))
    for ax,letter in zip(axes,'abc'):
        ax.tick_params(direction='out',length=3)
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-27,8),
            textcoords='offset points',fontweight='bold',fontsize=10)
    if args.qa_tools is not None:
        sys.path.insert(0,str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(layout.qa/'validation.alignment.json'),
            overlay_svg=str(layout.qa/'validation.alignment.svg'),tolerance_pt=1.5,strict=True)
    for extension in dict.fromkeys(args.formats):
        fig.savefig(layout.final/f'validation.{extension}',dpi=600)
    plt.close(fig)


if __name__=='__main__':
    main()
