"""Parameter-grounded E1 domain schematic; no solver is advanced.

Evidence: physical film context -> full numerical column -> first eight cells.
Panel a is schematic; panel b resolves all ny cell boundaries with its width
enlarged; panel c uses square cells. Chinese labels distinguish the represented
film area from the single periodic numerical column. No simulated data, sampled
grid lines, or additional physics are introduced. Python/vector drawing only.
"""
import argparse
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from config_io import load_config
from core.unit_mapping import create_unit_mapping


BLUE = '#0072B2'
LIGHT = '#D9EFF8'
ORANGE = '#D55E00'
DARK = '#303C46'
GRAY = '#687680'


def arrow(ax, start, end, color=BLUE, both=False, lw=1.0):
    ax.annotate('', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='<->' if both else '->', color=color,
                        lw=lw, shrinkA=0, shrinkB=0, mutation_scale=9))


def draw(case_dir, output, alignment_script, overwrite=False):
    config = load_config(case_dir / 'effective.yaml')
    summary = json.loads((case_dir / 'summary.json').read_text(encoding='utf-8'))
    mapping = create_unit_mapping(config)
    nx, ny = config['numerics']['nx'], config['numerics']['ny']
    dx = mapping.lattice.dx_m
    height = ny * dx
    area = config['run']['power']['area_m2']
    probes = np.asarray(summary['probe_z_m'])
    rows = np.rint(probes / dx - .5).astype(int)
    if nx != 1 or ny < 8 or config['run']['refine'] != 2:
        raise ValueError('This schematic depicts the current E1 r2 single-column case')
    np.testing.assert_allclose(probes, (rows + .5) * dx, atol=1e-15, rtol=0)
    assert abs(height - summary['actual_length_m']) < 1e-14
    assert abs(area - 9e-4) < 1e-15
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / ('e1_r2_domain.' + ext) for ext in ('pdf', 'svg', 'png')]
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Schematic already exists; use --overwrite for this figure only')
    plt.rcParams.update({'font.family':'sans-serif',
        'font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':10, 'pdf.fonttype':42, 'svg.fonttype':'none',
        'axes.unicode_minus':False, 'axes.spines.top':False, 'axes.spines.right':False})
    # Technical illustration, 305 x 163 mm; no journal submission size is claimed.
    fig, axes = plt.subplots(1, 3, figsize=(12, 6.4))
    fig.subplots_adjust(left=.035, right=.975, bottom=.19, top=.81, wspace=.19)
    fig.suptitle('E1 · r2：薄膜、计算域与网格划分', fontsize=18, y=.965, color=DARK)
    fig.text(.5, .897, '空间仅沿膜法向变化；每个气体格点仍保留完整 D2Q37 速度分量',
             ha='center', fontsize=11, color=GRAY)
    titles = ['薄膜与代表性计算列', '完整计算域与探针', '近膜网格放大：前 8 层']
    for letter, title, ax in zip('abc', titles, axes):
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_axis_off()
        ax.text(0, 1.04, letter, transform=ax.transAxes, fontweight='bold', fontsize=13, color=DARK)
        ax.text(.09, 1.04, title, transform=ax.transAxes, fontsize=12, color=DARK)

    ax = axes[0]
    film = np.array([[.09,.23],[.65,.23],[.91,.41],[.35,.41]])
    ax.add_patch(Polygon(film, closed=True, facecolor='#515B63', edgecolor=DARK, lw=1))
    # The prism is a representative numerical column; its drawn width is enlarged.
    base = np.array([[.465,.315],[.515,.315],[.545,.335],[.495,.335]])
    top = base + [0,.43]
    ax.add_patch(Polygon(np.array([base[0],base[1],top[1],top[0]]),closed=True,
        facecolor=LIGHT,edgecolor=BLUE,lw=1.2,alpha=.9))
    ax.add_patch(Polygon(np.array([base[1],base[2],top[2],top[1]]),closed=True,
        facecolor='#A9D8EA',edgecolor=BLUE,lw=1.2,alpha=.9))
    ax.add_patch(Polygon(top,closed=True,facecolor=LIGHT,edgecolor=BLUE,lw=1.2))
    ax.text(.04,.89,'膜上侧气体的\n代表性计算列',fontsize=11,color=BLUE,va='top')
    arrow(ax,(.34,.82),(.485,.73))
    arrow(ax,(.77,.50),(.77,.77))
    ax.text(.81,.76,r'$z$',fontsize=13,color=BLUE)
    ax.text(.73,.46,'膜法向',fontsize=10,color=BLUE)
    ax.text(.43,.277,'CNT 薄膜',fontsize=10,color='white',ha='center')
    arrow(ax,(.09,.165),(.65,.165),GRAY,both=True)
    ax.text(.37,.105,'30 mm',ha='center',fontsize=10,color=DARK)
    arrow(ax,(.70,.205),(.96,.385),GRAY,both=True)
    ax.text(.86,.19,'30 mm',ha='center',fontsize=10,color=DARK)
    ax.text(.04,.06,'位置关系示意，非等比例\n横向均匀假设；膜边缘未建模',
            fontsize=9.5,color=GRAY,linespacing=1.8,va='top')

    ax = axes[1]
    x0, width, y0, h = .31,.17,.13,.73
    ax.add_patch(Rectangle((x0,y0),width,h,facecolor=LIGHT,edgecolor=BLUE,lw=1))
    edges = y0 + h * np.arange(ny+1) / ny
    # Draw every physical cell boundary, including both ends: 767 lines for r2.
    ax.hlines(edges,x0,x0+width,color=BLUE,lw=.10,alpha=.55)
    arrow(ax,(.12,y0),(.12,y0+h),GRAY,both=True)
    ax.text(.065,y0+h/2,f'H = {height*1e3:.8f} mm',rotation=90,
            ha='center',va='center',fontsize=10,color=DARK)
    ax.plot([x0-.035,x0+width+.035],[y0+h,y0+h],color=BLUE,lw=1.7)
    for x in (x0+.045,x0+width-.045):
        arrow(ax,(x,y0+h+.015),(x,y0+h+.095))
    ax.text(.59,.90,'开放出口',fontsize=10,color=BLUE)
    ax.text(x0+width/2,.985,f'{ny} 层',ha='center',fontsize=10,color=BLUE)
    for index,(z,row) in enumerate(zip(probes,rows),1):
        yy=y0+h*z/height
        ax.plot(x0+width/2,yy,'o',color=ORANGE,ms=4.5)
        ax.plot([x0+width/2+.018,.56],[yy,yy],color=ORANGE,lw=.9)
        ax.text(.59,yy+.017,f'P{index}',fontsize=10,fontweight='bold',color=ORANGE)
        ax.text(.59,yy-.032,f'{z*1e3:.6f} mm',fontsize=9.5,color=DARK)
    ax.plot([x0-.04,x0+width+.04],[y0,y0],color=ORANGE,lw=3)
    ax.text(.59,.113,'膜面  z = 0',fontsize=10,color=ORANGE)
    ax.text(.04,.055,'单列：$n_x = 1$；横向两侧周期\n横向宽度放大显示；图示全部网格层',
            fontsize=9.5,color=GRAY,linespacing=1.8,va='top')

    ax = axes[2]
    bottom, cell_h, left = .13,.083,.18
    # Set physical square-cell aspect despite the surrounding annotation canvas.
    box=ax.get_position()
    cell_w=cell_h*(box.height*fig.get_figheight())/(box.width*fig.get_figwidth())
    for j in range(8):
        y=bottom+j*cell_h
        ax.add_patch(Rectangle((left,y),cell_w,cell_h,facecolor=LIGHT if j%2==0 else '#F1F8FC',
                              edgecolor=BLUE,lw=.85))
        ax.plot(left+cell_w/2,y+cell_h/2,'o',color=ORANGE if j==0 else BLUE,ms=3.2)
        ax.text(left-.055,y+cell_h/2,str(j),ha='right',va='center',fontsize=9,color=GRAY)
    ax.text(left-.07,bottom+8*cell_h+.045,'行号 j',ha='center',fontsize=9.5,color=GRAY)
    ax.text(.46,bottom+8*cell_h+.01,f'z = {8*dx*1e6:.6f} μm',fontsize=10,color=DARK)
    ax.text(.46,.91,'放大后每个方格\n对应一个气体格点单元',fontsize=10,color=GRAY,linespacing=1.6)
    cy=bottom+cell_h/2
    ax.plot([left+cell_w/2+.015,.43],[cy,cy],color=ORANGE,lw=.9)
    ax.text(.46,cy+.025,'首格中心',fontsize=10,color=ORANGE)
    ax.text(.46,cy-.026,f'z = {dx*.5*1e6:.7f} μm',fontsize=9.5,color=DARK)
    ydim=bottom+3*cell_h
    arrow(ax,(.40,ydim),(.40,ydim+cell_h),GRAY,both=True)
    ax.text(.46,ydim+cell_h/2,f'Δz = {dx*1e6:.6f} μm',va='center',fontsize=10,color=DARK)
    ax.plot([left-.045,left+cell_w+.045],[bottom,bottom],color=ORANGE,lw=3)
    arrow(ax,(left,.065),(left+cell_w,.065),GRAY,both=True)
    ax.text(.46,.03,'Δx = Δz\n膜面位于首格中心下方半格',fontsize=9.5,color=GRAY,linespacing=1.6)
    ax.text(.02,-.045,'每格：37 个 $f_i$ + 37 个 $g_i$；圆点表示格心',fontsize=9.5,color=GRAY)

    fig.text(.045,.107,'计算范围：只推进膜上侧气体；下侧通过对称双侧因子进入膜能量平衡。',fontsize=10,color=DARK)
    fig.text(.045,.056,'网格与探针来自已完成的 E1 r2、20 周期算例；蓝色表示气体计算域，橙色表示膜面或探针。',
             fontsize=9.5,color=GRAY)
    qa=case_dir.parent.parent/'_work'/'domain_figure_qa'
    qa.mkdir(parents=True,exist_ok=True)
    require_matplotlib_panel_alignment=runpy.run_path(str(alignment_script))['require_matplotlib_panel_alignment']
    require_matplotlib_panel_alignment(fig,json_out=qa/'domain.alignment.json',strict=True,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(output/'e1_r2_domain.pdf')
    fig.savefig(output/'e1_r2_domain.svg')
    fig.savefig(output/'e1_r2_domain.png',dpi=600)
    plt.close(fig)
    print(json.dumps(dict(nx=nx,ny=ny,dx_m=dx,height_m=height,film_area_m2=area,
        probes_m=probes.tolist(),probe_rows_zero_based=rows.tolist(),grid_boundaries_drawn=len(edges),
        output=str(output.resolve())),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--alignment-script',type=Path,required=True)
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    draw(args.case_dir,args.output or args.case_dir.parent.parent/'final',args.alignment_script,args.overwrite)
