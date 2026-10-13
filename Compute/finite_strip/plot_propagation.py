"""Plot the saved diagnostic k-Wave pressure at the 45 mm observer.

Figure contract: one quantitative time-series panel shows what the 2D
infinite-strip diagnostic run actually recorded. It is not validation evidence.
Use every saved sample at the observer labelled farfield_45mm, preserving time
and pressure without interpolation, normalization or phase adjustment. This
deterministic simulation has no statistical replicates or confidence interval.
Python is the exclusive backend; exports are an editable PDF and a 600 dpi PNG.
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
import numpy as np
from scipy.io import loadmat


def read_saved(batch):
    final = batch/'final'
    incoming = final/'power_r2_kwave_diagnostic_input.mat'
    outgoing = final/'power_r2_kwave_diagnostic_output.mat'
    summary = json.loads((final/'propagation_diagnostic.json').read_text(encoding='utf-8'))
    source = loadmat(incoming, simplify_cells=True)
    result = loadmat(outgoing, simplify_cells=True)
    if (summary['status'] != 'completed' or summary['diagnostic_mode'] is not True
            or summary['source_validation_passed'] is not False
            or summary['propagation_validation_passed'] is not False):
        raise ValueError('Expected a completed, explicitly unvalidated diagnostic run')
    for path, key in ((incoming, 'input_sha256'), (outgoing, 'output_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest() != summary[key]:
            raise ValueError(f'Saved propagation checksum differs: {path.name}')
    for data in (source, result):
        if data['diagnostic_mode'] != 1 or data['source_validation_passed'] != 0:
            raise ValueError('MAT diagnostic/source validation flags differ from summary')
    roles = np.asarray(source['observer_role']).ravel().astype(str)
    selected = np.flatnonzero(roles == 'farfield_45mm')
    if len(selected) != 1:
        raise ValueError('Exactly one farfield_45mm observer is required')
    index = int(selected[0])
    time = np.asarray(result['time_s'], dtype=float).ravel()
    pressure = np.asarray(result['observer_pressure_Pa'], dtype=float)
    x = np.asarray(result['observer_x_m'], dtype=float).ravel()
    y = np.asarray(result['observer_y_m'], dtype=float).ravel()
    if (pressure.shape != (len(roles), len(time)) or len(time) < 2
            or not np.isfinite(time).all() or np.any(np.diff(time) <= 0)
            or not np.isfinite(pressure).all()):
        raise ValueError('Invalid saved pressure/time arrays')
    for axis, actual in (('x', x), ('y', y)):
        if not np.allclose(actual, np.asarray(summary[f'observer_{axis}_m']), rtol=0, atol=1e-12):
            raise ValueError('Saved observer coordinates differ from summary')
    peak = abs(np.asarray(result['observer_p_hat_Pa']).ravel()[index])
    if not np.isclose(peak, summary['pressure_peak_at_45mm_Pa'][0], rtol=1e-12, atol=0):
        raise ValueError('Saved fundamental amplitude differs from summary')
    frequency = float(result['frequency_Hz'])
    if frequency != float(source['frequency_Hz']):
        raise ValueError('Source and propagated frequencies differ')
    return time, pressure[index], x[index], y[index], peak, frequency, summary


def render(batch, qa_scripts):
    batch = batch.resolve()
    outputs = [batch/'final'/name for name in
               ('propagation_45mm_diagnostic.png', 'propagation_45mm_diagnostic.pdf')]
    for path in outputs:
        if path.exists():
            raise FileExistsError(f'Existing figure will not be overwritten: {path}')
    time, pressure, x, y, peak, frequency, summary = read_saved(batch)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .8, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(7.2047244094, 3.9370078740))  # 183 x 100 mm
    fig.subplots_adjust(left=.12, right=.975, bottom=.25, top=.77)
    ax.plot(time*1e6, pressure*1e3, color='#0072B2', linewidth=1.4)
    ax.set(xlabel='Saved k-Wave time (µs)', ylabel='Pressure (mPa)',
           xlim=(time[0]*1e6, time[-1]*1e6))
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    ax.grid(axis='y', linewidth=.45, color='#D7DCE0')
    ax.set_axisbelow(True)
    fig.text(.12, .93, 'k-Wave pressure at 45 mm', fontsize=12, weight='bold')
    fig.text(.12, .86, f'2D infinite strip · diagnostic propagation · {frequency/1000:g} kHz', fontsize=9)
    fig.text(.12, .13, f'Actual observer: x = {x*1e3:.4g} mm, y = {y*1e3:.4g} mm'
             f'    |    Fundamental peak: {peak*1e3:.5g} mPa', fontsize=8)
    fig.text(.12, .07, 'Source validation: false    |    Propagation validation: false',
             fontsize=8, color='#555555')
    qa_root = batch.parent/'_work/figure_qa'
    qa_root.mkdir(parents=True, exist_ok=True)
    qa = Path(tempfile.mkdtemp(prefix=f'{batch.name}-propagation-', dir=qa_root))
    spec = importlib.util.spec_from_file_location('figure_alignment', qa_scripts/'audit_panel_alignment.py')
    alignment = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = alignment
    spec.loader.exec_module(alignment)
    fig.canvas.draw()
    alignment.require_matplotlib_panel_alignment(fig, json_out=qa/'alignment.json', strict=True)
    provenance = dict(input_sha256=summary['input_sha256'], output_sha256=summary['output_sha256'],
        plot_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sample_count=len(time), source_validation_passed=False, propagation_validation_passed=False,
        selection='All saved samples for the unique farfield_45mm observer',
        transformations='Only s to microseconds and Pa to mPa; original saved time origin retained',
        scope=summary['scope'], uncertainty='None: one deterministic simulated time series')
    png, pdf = qa/outputs[0].name, qa/outputs[1].name
    fig.savefig(pdf, format='pdf', metadata={'Title': '45 mm diagnostic k-Wave propagation',
        'Subject': json.dumps(provenance), 'Creator': 'Compute/finite_strip/plot_propagation.py'})
    fig.savefig(png, format='png', dpi=600, metadata={'Description': json.dumps(provenance)})
    plt.close(fig)
    for script, arguments in (
        ('validate_figure.py', [str(Path(__file__).resolve())]),
        ('audit_pdf_text.py', [str(pdf), '--min-pt', '5']),
        ('audit_figure_collisions.py', [str(pdf), '--json-out', str(qa/'collisions.json'),
                                      '--overlay-pdf', str(qa/'collision_overlay.pdf'), '--strict'])):
        audit = subprocess.run([sys.executable, str(qa_scripts/script), *arguments],
            capture_output=True, text=True, encoding='utf-8', errors='replace')
        (qa/f'{Path(script).stem}.txt').write_text(audit.stdout+audit.stderr, encoding='utf-8')
        print(audit.stdout, end='')
        # This task requests PDF and PNG. The generic SVG preference is inapplicable.
        pdf_png_only = (script == 'validate_figure.py'
            and [line for line in audit.stdout.splitlines() if line.startswith('[FAIL]')]
            == ['[FAIL] EXPORT-VECTOR: Missing required vector export: SVG'])
        if audit.returncode and not pdf_png_only:
            raise RuntimeError(f'{script} failed; QA render retained at {qa}')
    for temporary, final in zip((png, pdf), outputs):
        with temporary.open('rb') as src, final.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    print(json.dumps(dict(png=str(outputs[0]), pdf=str(outputs[1]), qa_directory=str(qa)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', required=True, type=Path)
    parser.add_argument('--qa-scripts', type=Path, default=Path.home()/'.codex/skills/nature-figure/scripts')
    args = parser.parse_args()
    batch = args.batch if args.batch.is_absolute() else Path(__file__).resolve().parent/args.batch
    render(batch, args.qa_scripts.resolve())


if __name__ == '__main__':
    main()
