"""绘制 verify_lim_nearfield 输出：声压吻合与膜温复幅误差共同展示。

确定性网格实验，无统计重复/置信区间；所有算例均保留。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reference.lim2013 import LimGas, TABLE1_AIR, near_field_pressure_rms_approximate
from simulation.output import OutputLayout, input_raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, nargs='+', required=True, help='批次目录、raw/ 或 summary.json')
    parser.add_argument('--output', type=Path, required=True, help='输出批次根目录，图表写入 final/')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--formats', nargs='+', choices=['png','pdf','svg'], default=['png','pdf'])
    parser.add_argument('--qa-tools', type=Path,
                        help='可选：nature-figure 的 scripts 目录；日常绘图无需排版审计')
    args = parser.parse_args()
    inputs = [input_raw(path)/'summary.json' if path.is_dir() else path for path in args.input]
    runs = [json.loads(path.read_text(encoding='utf-8')) for path in inputs]
    if not all(r['completed'] for r in runs):
        raise ValueError('只绘制已完整结束的运行；失败算例必须先核查')
    cases = [case for r in runs for case in r['cases']]
    settings = runs[0]['args']
    gas = LimGas(**cases[0]['gas'])
    if any(c['gas'] != cases[0]['gas'] for c in cases):
        raise ValueError('曲线比较需要相同气体常数')
    for run in runs:
        if any(run['args'][k]!=settings[k] for k in ('area','power','capacity','beta0','distance')):
            raise ValueError('曲线比较需要相同物理工况')
    if len({(c['frequency_Hz'],c['refine']) for c in cases}) != len(cases):
        raise ValueError('重复的频率/网格点，需明确选择独立运行')
    layout = OutputLayout(args.output)
    layout.prepare_figures(overwrite=args.overwrite,qa=args.qa_tools is not None)
    fields = ['frequency_Hz','refine','dx_m','ny','p_from_q_rms_Pa','p_Eq22_matched_rms_Pa',
              'p_Eq22_table1_rms_Pa','p_Eq22_error_percent','T_amplitude_error_percent',
              'T_phase_error_deg','q_phase_error_deg','q_window_change','total_energy_relative']
    with (layout.final/'source_data.csv').open('w', encoding='utf-8', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for c in cases:
            row = {k:c[k] for k in fields if k in c}
            row.update(T_amplitude_error_percent=c['T_comparison']['amplitude_error_percent'],
                T_phase_error_deg=c['T_comparison']['phase_error_deg'],
                q_phase_error_deg=c['q_comparison']['phase_error_deg'],
                q_window_change=c['window_complex_relative_change']['q'])
            writer.writerow(row)
    plt.rcParams.update({'font.family':'Arial', 'font.size':9, 'axes.labelsize':9,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.8,'legend.frameon':False})
    fig, axes = plt.subplots(1,2,figsize=(7.1,3.55))
    fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.72,wspace=.38)
    f = np.linspace(min(c['frequency_Hz'] for c in cases),max(c['frequency_Hz'] for c in cases),201)
    def curve(g):
        return [near_field_pressure_rms_approximate(fi,settings['power'],settings['area'],settings['capacity'],
            beta0_per_side_W_m2K=settings['beta0'],gas=g) for fi in f]
    axes[0].plot(f/1000,curve(gas),color='#444444',lw=1.4,label='Eq. 22, matched gas')
    axes[0].plot(f/1000,curve(TABLE1_AIR),color='#888888',lw=1,ls='--',label='Eq. 22, Table 1 gas')
    colors = ['#287D8E','#C47A29','#775B9C','#47794C']
    markers = ['o','s','^','D']
    refinements = sorted({c['refine'] for c in cases})
    if len(refinements)>len(colors):
        raise ValueError('此图支持至多四组网格')
    for i,r in enumerate(refinements):
        group = sorted([c for c in cases if c['refine']==r],key=lambda c:c['frequency_Hz'])
        x = [c['frequency_Hz']/1000 for c in group]
        label = f'LBM heat flux, dx/{r}'
        style = dict(color=colors[i],marker=markers[i],mfc='none',ms=5.8,mew=1.1)
        axes[0].plot(x,[c['p_from_q_rms_Pa'] for c in group],ls='none',label=label,**style)
        axes[1].plot(x,[100*c['T_comparison']['complex_relative_error'] for c in group],lw=1,**style)
    axes[1].axhline(0,color='#BBBBBB',lw=.7)
    axes[0].set_ylabel('Pressure from heat flux (Pa RMS)')
    axes[1].set_ylabel('Film temperature complex error (%)')
    for ax,letter in zip(axes,'ab'):
        ax.set_xlabel('Acoustic frequency (kHz)')
        ax.set_xticks(sorted({c['frequency_Hz']/1000 for c in cases}))
        ax.tick_params(direction='out',length=3)
        ax.annotate(letter,xy=(0,1),xycoords='axes fraction',xytext=(-26,8),
            textcoords='offset points',fontweight='bold',fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.99),ncol=2,fontsize=8,
        columnspacing=1.5,handlelength=2.2)
    if args.qa_tools is not None:
        sys.path.insert(0,str(args.qa_tools))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=str(layout.qa/'comparison.alignment.json'),
            overlay_svg=str(layout.qa/'comparison.alignment.svg'),tolerance_pt=1.5,strict=True)
    for extension in dict.fromkeys(args.formats):
        fig.savefig(layout.final/f'comparison.{extension}',dpi=600)
    plt.close(fig)


if __name__ == '__main__':
    main()
