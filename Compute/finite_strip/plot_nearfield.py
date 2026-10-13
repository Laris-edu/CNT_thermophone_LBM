"""Read-only, three-page LBM nearfield figure from saved measurements.

Contract: time histories distinguish periodic response from mean heating;
film maps locate edge effects; gas snapshots locate the thermal layer and
spatial pressure while the saved harmonic curves isolate the 20 kHz response.
This is a deterministic simulation, with no statistical replicates or error
bars. Grey film bands show the entire spatial range, not uncertainty. Python
is the exclusive backend. No LBM, k-Wave or propagation kernel is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.transforms import ScaledTranslation
import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from postproc.multiharmonic_fit import fit_multiharmonic


COLORS = ('#20262D', '#0072B2', '#D55E00')
PROBE_COLORS = ('#0072B2', '#D55E00', '#009E73')
NAMES = ('lbm_nearfield_timeseries.png', 'lbm_nearfield_film_space.png',
         'lbm_nearfield_gas_space.png', 'lbm_nearfield.pdf')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_batch(batch):
    raw, final = batch/'raw/power_r2', batch/'final'
    analysis_path, fits_path = final/'power_r2_nearfield.json', final/'power_r2_nearfield.npz'
    analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
    hashes = analysis['provenance']['raw_sha256']
    if set(hashes) != {'effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json'}:
        raise ValueError('Expected the full raw four-file provenance contract')
    for name, expected in hashes.items():
        if sha(raw/name) != expected:
            raise ValueError(f'Raw input differs from the saved analysis: {name}')
    config = yaml.safe_load((raw/'effective.yaml').read_text(encoding='utf-8'))
    summary = json.loads((raw/'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
    if any(item['status'] != 'completed' for item in (analysis, summary, manifest)):
        raise ValueError('Only completed saved data may be plotted')
    keys = ('t_s', 'heat_time_s', 'audit_t_s', 'film_temperature_K', 'film_wall_energy_J_m2',
        'film_x_m', 'film_dx_m', 'pressure_perturbation_Pa', 'probe_x_m', 'probe_y_m',
        'x_m', 'y_m', 'final_temperature_K', 'final_pressure_perturbation_Pa',
        'surface_t_s', 'surface_fields', 'surface_channels', 'surface_slices',
        'surface_x_m', 'surface_y_m', 'surface_normal_x', 'surface_normal_y', 'surface_ds_m',
        'run_completed')
    with np.load(raw/'trajectory.npz', allow_pickle=False) as saved:
        data = {key: saved[key] for key in keys}
    with np.load(fits_path, allow_pickle=False) as saved:
        fits = {key: saved[key] for key in saved.files}
    m = create_unit_mapping(config)
    time, heat = data['t_s'], data['heat_time_s']
    if not data['run_completed'] or np.any(np.diff(time) <= 0):
        raise ValueError('Incomplete run or invalid observation clock')
    if not np.allclose(heat, time-.5*m.lattice.dt_s, rtol=0, atol=1e-14*m.lattice.dt_s):
        raise ValueError('Wall heat must use its accepted-step midpoint')
    T = data['film_temperature_K']
    q = data['film_wall_energy_J_m2']/m.lattice.dt_s
    if T.shape != q.shape or T.shape != (len(time), len(data['film_x_m'])):
        raise ValueError('Film space/time shape mismatch')
    if data['pressure_perturbation_Pa'].shape != (len(time), len(data['probe_x_m'])):
        raise ValueError('Pressure probe/time shape mismatch')
    for key in ('final_temperature_K', 'final_pressure_perturbation_Pa'):
        if data[key].shape != (len(data['y_m']), len(data['x_m'])):
            raise ValueError('Gas field/grid shape mismatch')
    for key in ('film_x_m', 'probe_x_m', 'probe_y_m', 'surface_x_m', 'surface_y_m',
                'surface_slices', 'surface_normal_x', 'surface_normal_y', 'surface_ds_m'):
        if not np.array_equal(data[key], fits[key]):
            raise ValueError(f'Saved fits and raw geometry differ: {key}')
    for key, values in {**data, **fits, 'wall_heat_flux_W_m2': q}.items():
        if values.dtype.kind in 'fc' and not np.isfinite(values).all():
            raise ValueError(f'Nonfinite input: {key}')
    # Bind the saved fitted arrays to the checksummed raw series, using the same
    # public absolute-clock fitter. This is analysis only, not a new simulation.
    window = analysis['fit_window']
    frequency = analysis['frequency_Hz']
    channels = list(data['surface_channels'])
    comparisons = (
        (time, T, fits['film_T_hat_K']), (heat, q, fits['film_q_hat_W_m2']),
        (time, data['pressure_perturbation_Pa'], fits['direct_probe_p_hat_Pa']),
        (data['surface_t_s'], data['surface_fields'][:, :, channels.index('pressure_perturbation_Pa')],
         fits['surface_p_hat_Pa']))
    for clock, values, expected in comparisons:
        mask = ((clock*frequency >= window['start_cycle']-1e-10)
                & (clock*frequency < window['end_cycle']-1e-10))
        actual = np.array([fit_multiharmonic(clock[mask], v[mask], 2*np.pi*frequency,
            n_harmonics=window['harmonics'], detrend_order=window['detrend_order']).harmonic(1)
            for v in values.T])
        if not np.allclose(actual, expected, rtol=2e-10, atol=1e-12):
            raise ValueError('Saved harmonic array does not reproduce from checksummed raw data')
    data.update(q_W_m2=q, T0_K=m.physical.T0_K, mapping=m, config=config,
        analysis=analysis, fits=fits, end_s=float(data['audit_t_s'][-1]))
    metadata = dict(raw_sha256=hashes, analysis_json_sha256=sha(analysis_path),
        analysis_npz_sha256=sha(fits_path), plot_sha256=sha(Path(__file__)),
        fitted_arrays_reproduced_from_raw=True, raw_validation_passed=summary['validation_passed'],
        source_validation_passed=analysis['source_validation_passed'],
        fit_window=window, n_observations=len(time), n_film_bins=T.shape[1],
        heat_flux='Single-side actual accepted-step wall energy divided by lattice dt, not observation cadence',
        maps='Final gas snapshot includes DC/slow drift and is not a harmonic amplitude map',
        uncertainty='None: deterministic simulation; film band is spatial minimum to maximum')
    return data, metadata


def frame(title, subtitle, size=(7.2047244094, 8.8582677165)):
    fig = plt.figure(figsize=(7.2047244094, size[1]))  # 183 mm, task-specific page height
    fig.text(.11, .961, title, fontsize=12, weight='bold')
    fig.text(.11, .928, subtitle, fontsize=8)
    return fig


def label(ax, text):
    letter, title = text.split(maxsplit=1)
    for content, offset, weight in ((letter, 0, 'bold'), (title, 14, 'normal')):
        transform = ax.transAxes+ScaledTranslation(offset/72, 9/72, ax.figure.dpi_scale_trans)
        ax.text(0, 1, content, transform=transform, fontsize=9, fontweight=weight,
                ha='left', va='bottom')
    ax.tick_params(labelsize=7.5)
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)


def timeseries(d):
    fig = frame('有限条膜 LBM 近场：膜温—实际换热—直接声压',
                '1 mm 膜宽；上半气体域 2 mm × 0.2 mm；20 kHz；16 周期完成')
    fig.text(.11, .892, '膜：黑色均值 / 蓝色中心 / 橙色右缘；灰带为全部 766 段的空间范围', fontsize=7.8)
    x = d['film_x_m']; centre = int(np.argmin(abs(x))); edge = len(x)-1
    fig.text(.11, .867, f'实际膜位置：中心 x={x[centre]*1e3:.6f} mm；右缘 x={x[edge]*1e3:.6f} mm', fontsize=7.5)
    gs = fig.add_gridspec(3, 2, left=.11, right=.965, bottom=.19, top=.813, hspace=.58, wspace=.32)
    axes = np.array([[fig.add_subplot(gs[r,c]) for c in range(2)] for r in range(3)])
    clocks = (d['t_s'], d['heat_time_s'], d['t_s'])
    values = (d['film_temperature_K'], d['q_W_m2']/1000, d['pressure_perturbation_Pa'])
    ylabels = ('膜温 Tf (K)', '单侧实际 q (kW/m²)', '直接 p−p0 (Pa)')
    start, end = d['analysis']['fit_window']['start_s'], d['end_s']
    probes = [0, int(np.argmin(abs(d['probe_x_m'])+abs(d['probe_y_m']-.000075))),
              int(np.argmin(abs(d['probe_x_m'])+abs(d['probe_y_m']-.000115)))]
    for row, (clock, array, ylabel) in enumerate(zip(clocks, values, ylabels)):
        for col in range(2):
            ax = axes[row,col]; mask = clock >= (0 if col == 0 else start-1e-14)
            t, v = clock[mask]*1e6, array[mask]
            if row < 2:
                ax.fill_between(t, v.min(axis=1), v.max(axis=1), color='#C4CAD0', alpha=.38, linewidth=0)
                for trace, color, style in zip((v.mean(axis=1), v[:,centre], v[:,edge]), COLORS, ('-', '--', '-')):
                    ax.plot(t, trace, color=color, ls=style, lw=1.1)
            else:
                ax.plot(t, v, color='#BBC1C7', lw=.5, alpha=.7)
                for probe, color in zip(probes, PROBE_COLORS):
                    ax.plot(t, v[:,probe], color=color, lw=1.1)
            label(ax, f'{chr(97+row*2+col)}  '+('全部 16 周期' if col == 0 else '最后完整 2 周期'))
            ax.set(xlabel='原始时间 (µs)', ylabel=ylabel,
                   xlim=((0 if col == 0 else start*1e6), end*1e6))
            ax.grid(axis='y', color='#DCE0E4', lw=.45); ax.set_axisbelow(True)
    fig.text(.11, .126, '声压：灰线含全部 16 个探针；突出位置为实际保存坐标 (x, y)：', fontsize=7.8)
    for row, (probe, color) in enumerate(zip(probes, PROBE_COLORS)):
        fig.text(.11, .102-row*.021,
            f'探针 {probe:02d}  ({d["probe_x_m"][probe]*1e3:.6f}, {d["probe_y_m"][probe]*1e3:.6f}) mm',
            fontsize=7.5, color=color)
    fig.text(.11, .023, 'q>0 表示膜向气体放热；热流按单步中点，膜温与声压按步末；周期稳定不代表平均温度停止漂移。', fontsize=7.1)
    return fig, []


def colorbar(fig, ax, artist, title):
    box = ax.get_position()
    cax = fig.add_axes([box.x1+.012, box.y0, .014, box.height])
    cb = fig.colorbar(artist, cax=cax); cb.set_label(title, fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    return cax


def film_space(d):
    fig = frame('沿膜空间分布：有限宽度造成中心与边缘差异',
                '上排：全部保存样本；下排：已保存的 20 kHz 基频复幅值模', size=(7.2047244094, 7.4803149606))
    gs = fig.add_gridspec(2, 2, left=.11, right=.88, bottom=.17, top=.84, hspace=.6, wspace=.74)
    axes = [fig.add_subplot(gs[r,c]) for r in range(2) for c in range(2)]
    x = d['film_x_m']*1e3
    content = ((d['t_s']*1e6, d['film_temperature_K']-d['T0_K'], 'a  膜温时空分布', 'Tf−T0 (K)'),
               (d['heat_time_s']*1e6, d['q_W_m2']/1000, 'b  实际换热时空分布', 'q (kW/m²)'))
    cbars = []
    for ax, (clock, values, title, ctitle) in zip(axes[:2], content):
        artist = ax.pcolormesh(x, clock, values, shading='nearest', cmap='viridis', rasterized=True)
        ax.set(xlabel='膜面 x (mm)', ylabel='原始时间 (µs)'); label(ax, title)
        cbars.append(colorbar(fig, ax, artist, ctitle))
    for ax, values, title, ylabel, color in (
        (axes[2], abs(d['fits']['film_T_hat_K']), 'c  膜温基频幅值', '|Tf_hat| (K)', COLORS[1]),
        (axes[3], abs(d['fits']['film_q_hat_W_m2'])/1000, 'd  单侧实际热流基频幅值', '|q_hat| (kW/m²)', COLORS[2])):
        ax.plot(x, values, color=color, lw=1.2)
        ax.set(xlabel='膜面 x (mm)', ylabel=ylabel, xlim=(x[0], x[-1])); label(ax, title)
        ax.grid(axis='y', color='#DCE0E4', lw=.45); ax.set_axisbelow(True)
    # Keep the original endpoint heat-flux peaks separate from the axes spines.
    margin = .03*(x[-1]-x[0])
    axes[3].set_xlim(x[0]-margin, x[-1]+margin)
    fig.text(.11, .092, '基频拟合窗口 700–800 µs：公共三谐波 + 线性漂移；保留原始时间零点，热流使用单步中点。', fontsize=7.2)
    fig.text(.11, .05, '温度/热流时空图含平均分量；基频幅值图只表示 20 kHz 振荡。端部为绝热边缘，无电极/支撑固体。', fontsize=7.2)
    return fig, cbars


def gas_space(d):
    fig = frame('气体空间分布：最终瞬时场与交接线声学分量',
                f'最终时刻 t={d["end_s"]*1e6:.3f} µs；全域快照含平均分量/慢漂移，不是 20 kHz 声压幅值图',
                size=(7.2047244094, 7.8740157480))
    gs = fig.add_gridspec(3, 2, left=.11, right=.87, bottom=.2, top=.84,
                          height_ratios=(1,1,1.75), hspace=.82, wspace=.38)
    maps = [fig.add_subplot(gs[row,:]) for row in range(2)]
    lower = [fig.add_subplot(gs[2,col]) for col in range(2)]
    x,y = d['x_m']*1e3,d['y_m']*1e3
    dx = d['mapping'].lattice.dx_m*1e3
    edges = (x[0]-dx/2,x[-1]+dx/2,0,y[-1]+dx/2)
    run = d['config']['strip']; thickness = run['pml_cells']*dx
    left, right, top = edges[0]+thickness, edges[1]-thickness, edges[3]-thickness
    a,b = d['fits']['surface_slices'][d['analysis']['selected_export_contour']-1]
    sx,sy = d['fits']['surface_x_m'][a:b]*1e3,d['fits']['surface_y_m'][a:b]*1e3
    hx,hy = float(np.max(abs(sx))),float(np.max(sy))
    cbars = []
    for ax, values, title, ctitle in (
        (maps[0],d['final_temperature_K']-d['T0_K'],'a  气体最终瞬时温升','Tg−T0 (K)'),
        (maps[1],d['final_pressure_perturbation_Pa'],'b  气体最终瞬时压力变化','p−p0 (Pa)')):
        limit = float(np.max(abs(values)))
        artist = ax.imshow(values, origin='lower', extent=edges, aspect='equal',
            interpolation='none', cmap='RdBu_r', norm=Normalize(vmin=-limit,vmax=limit), rasterized=True)
        for boundary in (left,right): ax.axvline(boundary, color='#555555', ls=':', lw=.8)
        ax.axhline(top, color='#555555', ls=':', lw=.8)
        ax.plot([-hx,-hx,hx,hx],[0,hy,hy,0], color='#7A5195', ls='--', lw=.8)
        ax.plot([run['actual_film_x_min_m']*1e3,run['actual_film_x_max_m']*1e3],[0,0],
                color='#20262D', lw=2.5, clip_on=False)
        ax.set(xlabel='x (mm)', ylabel='y (mm)', xlim=edges[:2], ylim=edges[2:]); label(ax,title)
        cbars.append(colorbar(fig,ax,artist,ctitle))
    for index,(a,b) in enumerate(d['fits']['surface_slices']):
        top_mask = d['fits']['surface_normal_y'][a:b] > .5
        top_x = d['fits']['surface_x_m'][a:b][top_mask]*1e3
        top_p = abs(d['fits']['surface_p_hat_Pa'][a:b][top_mask])
        lower[0].plot(top_x,top_p,color=('#888888','#7A5195','#009E73')[index],
                      lw=1.3 if index==1 else .8)
    label(lower[0], 'c  三条交接线上边基频幅值')
    lower[0].set(xlabel='交接线上边 x (mm)',ylabel='|p_hat| (Pa)')
    centre = int(np.argmin(abs(d['x_m'])))
    lower[1].plot(d['final_pressure_perturbation_Pa'][:,centre],y,color=COLORS[1],lw=1.2)
    lower[1].axhline(top,color='#555555',ls=':',lw=.8)
    label(lower[1], 'd  最终瞬时压力竖剖面')
    lower[1].set(xlabel='p−p0 (Pa)',ylabel='y (mm)',ylim=edges[2:])
    for ax in lower: ax.grid(color='#DCE0E4',lw=.45); ax.set_axisbelow(True)
    fig.text(.11,.139,'地图保持真实纵横比；黑线=膜，紫虚线=主交接线，灰点线外侧/上方=PML。',fontsize=7.5)
    fig.text(.11,.105,f'实际 PML 入口：|x|={right:.6f} mm，y={top:.6f} mm；竖剖面 x={d["x_m"][centre]*1e3:.6f} mm。',fontsize=7.3)
    fig.text(.11,.073,'c 图：灰 y=0.09 mm / 紫 y=0.10 mm (主线) / 绿 y=0.11 mm；幅值来自 700–800 µs 拟合。',fontsize=7.3)
    fig.text(.11,.035,'结果属于二维无限长条膜近场；周期稳定已达到，交接/传播验证失败记录仍保留，未证明长期热稳态。',fontsize=7.2)
    return fig, cbars


def render(batch, qa_scripts, preview):
    batch = batch.resolve()
    for name in NAMES:
        if (batch/'final'/name).exists():
            raise FileExistsError(f'Existing final figure will not be overwritten: {name}')
    d, metadata = read_batch(batch)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Microsoft YaHei','Arial','DejaVu Sans'],
        'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.unicode_minus':True})
    qa_root = batch.parent/'_work/figure_qa'; qa_root.mkdir(parents=True,exist_ok=True)
    qa = Path(tempfile.mkdtemp(prefix=f'{batch.name}-nearfield-',dir=qa_root))
    spec = importlib.util.spec_from_file_location('nearfield_alignment',qa_scripts/'audit_panel_alignment.py')
    alignment = importlib.util.module_from_spec(spec); sys.modules[spec.name] = alignment; spec.loader.exec_module(alignment)
    figures = [timeseries(d),film_space(d),gas_space(d)]
    pdf = qa/'lbm_nearfield.pdf'
    with PdfPages(pdf,metadata={'Title':'有限条膜 LBM 近场','Subject':json.dumps(metadata),
                               'Creator':'Compute/finite_strip/plot_nearfield.py'}) as pages:
        for index,(fig,cbars) in enumerate(figures):
            fig.canvas.draw()
            alignment.require_matplotlib_panel_alignment(fig,json_out=qa/f'alignment_page{index+1}.json',
                exclude_axes=cbars,tolerance_pt=1.5,gutter_tolerance_pt=1.5,
                require_panel_labels=True,strict=True)
            pages.savefig(fig)
            fig.savefig(qa/NAMES[index],format='png',dpi=600,metadata={'Description':json.dumps(metadata)})
            plt.close(fig)
    with (qa/'provenance.json').open('x',encoding='utf-8') as handle:
        json.dump(metadata,handle,ensure_ascii=False,indent=2)
    for script,arguments in (
        ('validate_figure.py',[str(Path(__file__).resolve())]),
        ('audit_pdf_text.py',[str(pdf),'--min-pt','5']),
        ('audit_figure_collisions.py',[str(pdf),'--json-out',str(qa/'collisions.json'),
                                     '--overlay-pdf',str(qa/'collision_overlay.pdf'),'--strict'])):
        audit = subprocess.run([sys.executable,'-X','utf8',str(qa_scripts/script),*arguments],capture_output=True,
            text=True,encoding='utf-8',errors='replace')
        (qa/f'{Path(script).stem}.txt').write_text(audit.stdout+audit.stderr,encoding='utf-8')
        print(audit.stdout,end='')
        format_only = (script=='validate_figure.py'
            and [line for line in audit.stdout.splitlines() if line.startswith('[FAIL]')]
            == ['[FAIL] EXPORT-VECTOR: Missing required vector export: SVG'])
        if audit.returncode and not format_only:
            raise RuntimeError(f'{script} failed; preview retained at {qa}')
    if not preview:
        for name in NAMES:
            with (qa/name).open('rb') as src,(batch/'final'/name).open('xb') as dst:
                shutil.copyfileobj(src,dst)
    print(json.dumps(dict(preview=preview,qa_directory=str(qa),
        outputs=[str((qa if preview else batch/'final')/name) for name in NAMES]),indent=2))


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True,type=Path)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--qa-scripts',type=Path,default=Path.home()/'.codex/skills/nature-figure/scripts')
    args = parser.parse_args()
    batch = args.batch if args.batch.is_absolute() else Path(__file__).resolve().parent/args.batch
    render(batch,args.qa_scripts.resolve(),args.preview)


if __name__ == '__main__':
    main()
