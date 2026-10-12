"""Plot saved pulse boundary screening; never run or alter a simulation.

Figure contract: do the candidate boundaries reproduce the larger-domain
pulse at the saved grid? Three raw traces establish waveform agreement,
paired differences and all probe angles expose the acceptance metric, and
the final field locates that evidence. This deterministic numerical figure
has no statistical replicates or confidence intervals. It is not a thermal
layer, 20 kHz, or finite-film validation. Python is the exclusive backend.

Run from any directory with --batch 20261012_strip_validation. --preview
keeps both renders in _work/figure_qa for visual inspection before delivery.
Only new boundary_validation.png and .pdf are published to final/; existing
figures are never overwritten. The skill QA scripts may be supplied with
--qa-scripts when they are installed at a different location.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from matplotlib.patches import Rectangle
import numpy as np
import yaml


CASE_NAMES = ('pulse_candidate_dx10um', 'pulse_reference_dx10um')
SELECTED_COLORS = ('#0072B2', '#D55E00', '#7A5195')
REFERENCE_COLOR = '#555555'


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_batch(batch):
    """Check the saved comparison against every raw probe, without refitting."""
    cases, inputs = [], []
    keys = ('t_s', 'probe_x_m', 'probe_y_m', 'pressure_perturbation_Pa',
            'x_m', 'y_m', 'final_pressure_perturbation_Pa')
    for name in CASE_NAMES:
        directory = batch/'raw'/name
        paths = [directory/filename for filename in
                 ('effective.yaml', 'manifest.json', 'trajectory.npz', 'summary.json')]
        effective = yaml.safe_load(paths[0].read_text(encoding='utf-8'))
        manifest, summary = _json(paths[1]), _json(paths[3])
        if manifest['status'] != 'completed' or summary['status'] != 'completed':
            raise ValueError(f'{name} is not a completed saved run')
        if effective['strip']['kind'] != 'pulse':
            raise ValueError(f'{name} is not a pulse case')
        with np.load(paths[2], allow_pickle=False) as archive:
            if not bool(archive['run_completed']):
                raise ValueError(f'{name} trajectory is incomplete')
            data = {key: np.asarray(archive[key]).copy() for key in keys}
        if not all(np.all(np.isfinite(value)) for value in data.values()):
            raise ValueError(f'{name} contains nonfinite plotted observations')
        t, px, py = (data[key] for key in keys[:3])
        if (t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0)
                or px.shape != (13,) or py.shape != px.shape
                or data['pressure_perturbation_Pa'].shape != (len(t), len(px))):
            raise ValueError('expected 13 probes and increasing endpoint times')
        if data['final_pressure_perturbation_Pa'].shape != (len(data['y_m']), len(data['x_m'])):
            raise ValueError('final pressure field does not match saved coordinates')
        cases.append(dict(data=data, effective=effective, manifest=manifest, summary=summary))
        inputs.extend(paths)
    small, large = (case['data'] for case in cases)
    for key in ('t_s', 'probe_x_m', 'probe_y_m'):
        if small[key].shape != large[key].shape or not np.allclose(small[key], large[key], rtol=0, atol=1e-14):
            raise ValueError(f'candidate and reference differ in {key}')
    for case in cases:
        dx = float(case['effective']['lattice']['dx_m'])
        if not np.isclose(dx, 1e-5, rtol=0, atol=1e-15):
            raise ValueError('this figure is explicitly scoped to the 10 micrometre screening pair')
        for key in ('x_m', 'y_m'):
            if not np.allclose(np.diff(case['data'][key]), dx, rtol=0, atol=1e-14):
                raise ValueError('saved field coordinates do not match the effective grid')
    comparison_path = batch/'final/boundary_validation.json'
    saved_comparison = _json(comparison_path)
    comparison = saved_comparison.get('components',{}).get('combined',saved_comparison)
    inputs.append(comparison_path)
    limit, start = float(comparison['limit']), float(comparison['return_start_s'])
    if not np.isfinite(limit) or limit <= 0 or not np.isfinite(start):
        raise ValueError('invalid saved acceptance window or limit')
    mask = small['t_s'] >= start
    peak = np.max(np.abs(large['pressure_perturbation_Pa']), axis=0)
    if not mask.any() or np.any(peak < 1e-12):
        raise ValueError('unresolved reference pulse or empty return window')
    difference = np.abs(small['pressure_perturbation_Pa']-large['pressure_perturbation_Pa'])/peak
    maxima = np.max(difference[mask], axis=0)
    if (not np.allclose(maxima, comparison['return_error_ratios'], rtol=1e-10, atol=1e-12)
            or not np.isclose(np.max(maxima), comparison['max_return_error_ratio'], rtol=1e-10, atol=1e-12)):
        raise ValueError('saved validation metric is inconsistent with the current raw trajectories')
    expected_pass = bool(np.max(maxima) < limit and comparison.get('reference_configuration_checked') is True)
    if comparison['boundary_validation_passed'] is not expected_pass:
        raise ValueError('saved boundary status contradicts its metric or reference checks')
    provenance = {str(path.relative_to(batch)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in inputs}
    return cases, comparison, difference, maxima, provenance


def _panel(axis, label, title):
    axis.set_label(label)
    axis.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(-24, 9),
                  textcoords='offset points', fontsize=9, fontweight='bold',
                  ha='left', va='bottom', annotation_clip=False)
    axis.set_title(title, loc='left', pad=9, fontsize=8.2)
    axis.tick_params(labelsize=7, width=.65, length=3)


def make_figure(cases, comparison, difference, maxima):
    """Use all 13 probes; highlight the nearest 0, +45, and +90 degree probes."""
    plt.rcParams.update({'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], 'font.size': 7.5,
        'axes.labelsize': 7.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'legend.frameon': False, 'legend.fontsize': 7,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'savefig.dpi': 600})
    small, large = (case['data'] for case in cases)
    t = small['t_s']*1e6
    angle = np.degrees(np.arctan2(small['probe_x_m'], small['probe_y_m']))
    selected = [int(np.argmin(np.abs(angle-target))) for target in (0, 45, 90)]
    if len(set(selected)) != 3:
        raise ValueError('three distinct normal, oblique and near-tangent probes are required')
    titles = ('Normal', 'Oblique', 'Near tangent')
    start = comparison['return_start_s']*1e6
    threshold = comparison['limit']*100
    # 183 x 218 mm, with enough physical height for the 4:1 domain map.
    fig = plt.figure(figsize=(7.2047244094, 8.5826771654), facecolor='white')
    grid = fig.add_gridspec(3, 3, left=.10, right=.965, bottom=.20, top=.835,
                           hspace=.65, wspace=.48, height_ratios=(1, 1, 1.5))
    trace_axes = [fig.add_subplot(grid[0, j]) for j in range(3)]
    difference_axis = fig.add_subplot(grid[1, :2])
    angle_axis = fig.add_subplot(grid[1, 2])
    field_axis = fig.add_subplot(grid[2, :])
    pressure_scale = max(float(np.max(np.abs(small['pressure_perturbation_Pa'][:, selected]))),
                         float(np.max(np.abs(large['pressure_perturbation_Pa'][:, selected]))))
    trace_lo = min(float(np.min(small['pressure_perturbation_Pa'][:, selected])),
                   float(np.min(large['pressure_perturbation_Pa'][:, selected])))
    trace_hi = max(float(np.max(small['pressure_perturbation_Pa'][:, selected])),
                   float(np.max(large['pressure_perturbation_Pa'][:, selected])))
    for j, (axis, probe, color, title) in enumerate(zip(trace_axes, selected, SELECTED_COLORS, titles)):
        _panel(axis, chr(ord('a')+j), f'{title} | θ = {angle[probe]:.1f}°')
        axis.axvspan(t[0], start, color='#EEEEEE', linewidth=0, zorder=0)
        axis.plot(t, large['pressure_perturbation_Pa'][:, probe], color=REFERENCE_COLOR,
                  linewidth=1.3, linestyle=(0, (4, 2)), label='Reference')
        axis.plot(t, small['pressure_perturbation_Pa'][:, probe], color=color,
                  linewidth=.9, label='Candidate')
        axis.set(xlim=(0, t[-1]), ylim=(trace_lo-.09*pressure_scale, trace_hi+.15*pressure_scale),
                 xlabel='Time (µs)')
        axis.set_xticks([0, 2, 4, 6, 8])
        if j == 0:
            axis.set_ylabel('Pressure perturbation (Pa)')
            axis.legend(loc='upper right', handlelength=2.0)
        else:
            axis.tick_params(labelleft=False)
    _panel(difference_axis, 'd', 'Paired error at every probe')
    difference_axis.axvspan(t[0], start, color='#EEEEEE', linewidth=0, zorder=0)
    others = [j for j in range(13) if j not in selected]
    difference_axis.plot(t, 100*difference[:, others], color='#C4C4C4', linewidth=.55)
    for probe, color, title in zip(selected, SELECTED_COLORS, titles):
        difference_axis.plot(t, 100*difference[:, probe], color=color, linewidth=.9,
                             label=f'{title} ({angle[probe]:.1f}°)')
    difference_axis.axhline(threshold, color='#A33131', linestyle='--', linewidth=.85)
    difference_axis.set(xlim=(0, t[-1]), ylim=(0, 1.18*max(threshold, float(np.max(100*difference)))),
                        xlabel='Time (µs)', ylabel='Normalized error (%)')
    difference_axis.set_xticks([0, 2, 4, 6, 8])
    _panel(angle_axis, 'e', 'Maximum return-window error')
    order = np.argsort(angle)
    angle_axis.plot(angle[order], 100*maxima[order], color='#555555', marker='o',
                    markersize=3, linewidth=.75)
    for probe, color in zip(selected, SELECTED_COLORS):
        angle_axis.plot(angle[probe], 100*maxima[probe], color=color, marker='o', markersize=4)
    angle_axis.axhline(threshold, color='#A33131', linestyle='--', linewidth=.85)
    angle_axis.set(xlim=(-95, 95), ylim=(0, 1.18*max(threshold, float(np.max(100*maxima)))),
                   xlabel='Angle from +y (°)', ylabel='Peak error (%)')
    angle_axis.set_xticks([-90, 0, 90])
    _panel(field_axis, 'f', f'Candidate final pressure field | t = {t[-1]:.3f} µs')
    dx = float(cases[0]['effective']['lattice']['dx_m'])
    x, y = small['x_m']*1e3, small['y_m']*1e3
    extent = [x[0]-.5*dx*1e3, x[-1]+.5*dx*1e3, y[0]-.5*dx*1e3, y[-1]+.5*dx*1e3]
    field = small['final_pressure_perturbation_Pa']
    amplitude = float(np.max(np.abs(field)))
    if amplitude <= 0:
        raise ValueError('final pressure field is identically zero')
    image = field_axis.imshow(field, origin='lower', extent=extent, cmap='RdBu_r',
                              vmin=-amplitude, vmax=amplitude, interpolation='nearest',
                              aspect='equal', rasterized=True)
    field_axis.scatter(small['probe_x_m']*1e3, small['probe_y_m']*1e3, s=8,
                       facecolors='white', edgecolors='#444444', linewidths=.5, zorder=4)
    for probe, color in zip(selected, SELECTED_COLORS):
        field_axis.scatter(small['probe_x_m'][probe]*1e3, small['probe_y_m'][probe]*1e3,
                           s=18, color=color, edgecolors='white', linewidths=.6, zorder=5)
    margin = max(cases[0]['effective']['strip']['sponge_cells'],
                 cases[0]['effective']['strip'].get('pml_cells',0))*dx*1e3
    if margin > 0:
        field_axis.plot([extent[0]+margin, extent[0]+margin, extent[1]-margin, extent[1]-margin],
                        [0, extent[3]-margin, extent[3]-margin, 0],
                        color='#333333', linestyle=(0, (3, 2)), linewidth=.65)
    field_axis.add_patch(Rectangle((extent[0], 0), extent[1]-extent[0], extent[3],
                                  fill=False, linewidth=.75, edgecolor='#222222', clip_on=False))
    field_axis.set(xlim=extent[:2], ylim=extent[2:], xlabel='x (mm)', ylabel='y (mm)')
    # A separate colorbar preserves the field's physical aspect and plot width.
    fig.canvas.draw()
    field_box = field_axis.get_position()
    colorbar_axis = fig.add_axes([field_box.x0+.16*field_box.width, field_box.y0-.075,
                                 .68*field_box.width, .014], label='pressure_colorbar')
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label('Pressure perturbation (Pa)', fontsize=7)
    colorbar.ax.tick_params(labelsize=7, length=2, pad=2)
    from matplotlib.ticker import MaxNLocator
    colorbar.locator = MaxNLocator(nbins=5)
    colorbar.update_ticks()
    passed = 'PASS' if comparison['boundary_validation_passed'] else 'FAIL'
    state = ['PASS' if case['summary'].get('state_validation_passed') is True else 'FAIL' for case in cases]
    fig.text(.10, .975, 'Open-boundary pulse screening', fontsize=12, fontweight='bold', va='top')
    fig.text(.10, .945, f'Runs: completed / completed     State checks: {state[0]} / {state[1]}',
             fontsize=8.2, va='top')
    fig.text(.10, .922, f'Combined pulse: {passed}   |   maximum error {100*np.max(maxima):.3f}%   |   limit {threshold:g}%',
             fontsize=8.2, va='top', color='#00705C' if passed == 'PASS' else '#A33131')
    width_small, height_small = (cases[0]['effective']['strip'][key]*1e3 for key in ('actual_width_m', 'actual_height_m'))
    width_large, height_large = (cases[1]['effective']['strip'][key]*1e3 for key in ('actual_width_m', 'actual_height_m'))
    fig.text(.10, .896, f'Candidate: {width_small:g} × {height_small:g} mm   |   reference: {width_large:g} × {height_large:g} mm   |   dx = 10 µm',
             fontsize=7.5, va='top')
    fig.text(.10, .075, f'Error = |candidate − reference| / peak |reference| at each probe. Criterion: t ≥ {start:g} µs (unshaded).',
             fontsize=7, va='top')
    fig.text(.10, .057, f'Red dashed: {threshold:g}% limit. Colored traces / markers match a–c; all 13 probes shown. Field dashed: absorber edge.',
             fontsize=7, va='top')
    fig.text(.10, .031, '10 µm coarse-grid boundary screening only; no 20 kHz or membrane thermal-layer certification.',
             fontsize=7.3, fontweight='bold', va='top')
    return fig, colorbar_axis


def _qa_module(directory, name):
    source = directory/f'{name}.py'
    if not source.is_file():
        raise FileNotFoundError(f'figure QA helper is required: {source}')
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def render(batch, qa_scripts, *, preview=False):
    batch = batch.resolve()
    destinations = [batch/'final'/f'boundary_validation.{suffix}' for suffix in ('png', 'pdf')]
    for path in destinations:
        if path.exists():
            raise FileExistsError(f'existing final figure will not be overwritten: {path}')
    cases, comparison, difference, maxima, provenance = read_batch(batch)
    fig, colorbar_axis = make_figure(cases, comparison, difference, maxima)
    qa_root = batch/'_work/figure_qa'
    qa_root.mkdir(parents=True, exist_ok=True)
    qa = Path(tempfile.mkdtemp(prefix='boundary-', dir=qa_root))
    # Final drawn axes rectangles, including unequal grid spans, must align.
    alignment = _qa_module(qa_scripts, 'audit_panel_alignment')
    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(fig, json_out=qa/'alignment.json',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True,
        exclude_axes=[colorbar_axis], strict=True)
    metadata = dict(batch=batch.name, created_utc=datetime.now(timezone.utc).isoformat(),
                    input_sha256=provenance, scope=comparison['scope'],
                    plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    format_contract='Requested PDF and PNG; generic SVG/TIFF recommendations do not apply.')
    pdf, png = qa/'boundary_validation.pdf', qa/'boundary_validation.png'
    fig.savefig(pdf, metadata={'Title': 'Open-boundary pulse screening',
                               'Subject': json.dumps(metadata, ensure_ascii=True),
                               'Creator': 'Compute/finite_strip/plot.py'})
    fig.savefig(png, dpi=600, metadata={'Description': json.dumps(metadata, ensure_ascii=True)})
    plt.close(fig)
    for script, arguments in (
        ('validate_figure.py', [str(Path(__file__).resolve())]),
        ('audit_pdf_text.py', [str(pdf), '--min-pt', '5']),
        ('audit_figure_collisions.py', [str(pdf), '--json-out', str(qa/'collisions.json'),
                                       '--overlay-pdf', str(qa/'collision_overlay.pdf')])):
        result = subprocess.run([sys.executable, str(qa_scripts/script), *arguments],
                                capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(result.stdout, end='')
        if result.stderr:
            print(result.stderr, file=sys.stderr, end='')
        format_only = (script == 'validate_figure.py'
            and [line for line in result.stdout.splitlines() if line.startswith('[FAIL]')]
            == ['[FAIL] EXPORT-VECTOR: Missing required vector export: SVG'])
        if format_only:
            print('Format-contract exception: this task requests editable PDF and PNG, not an additional SVG.')
        elif result.returncode:
            raise RuntimeError(f'{script} failed; diagnostic render retained at {qa}')
    if not preview:
        # Exclusive creation also protects an output written after the initial check.
        for source, destination in ((png, destinations[0]), (pdf, destinations[1])):
            with source.open('rb') as incoming, destination.open('xb') as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    print(json.dumps(dict(preview=preview, png=str(png if preview else destinations[0]),
        pdf=str(pdf if preview else destinations[1]), qa_directory=str(qa)), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', required=True, help='batch name beside this plotting script, or an absolute path')
    parser.add_argument('--preview', action='store_true', help='render and audit in _work without publishing final files')
    parser.add_argument('--qa-scripts', type=Path, default=Path.home()/'.codex/skills/nature-figure/scripts')
    args = parser.parse_args()
    batch = Path(args.batch)
    if not batch.is_absolute():
        batch = Path(__file__).resolve().parent/batch
    render(batch, args.qa_scripts.resolve(), preview=args.preview)


if __name__ == '__main__':
    main()
