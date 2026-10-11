"""Plot the saved finite-film field and its numerical validation.

Figure contract: a uniform, free-space monopole sheet extends the local LBM
source into a finite-aperture acoustic field; separate checks establish source
normalization, propagation accuracy and numerical sensitivity. Two quantitative
grids, deterministic complex amplitudes (no replicate confidence intervals),
editable PDF and 300 dpi PNG. Phase retains the original input-power clock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from Compute.film_to_sound.plot import style

BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#777777'


def complex_values(value):
    """Read the explicit complex-number representation used by the analysis."""
    answer = np.asarray(value['re']) + 1j*np.asarray(value['im'])
    if not np.all(np.isfinite(answer)):
        raise ValueError('non-finite complex pressure cannot be plotted')
    return answer


def layout(title, result, footer):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 8.2))
    fig.subplots_adjust(left=.105, right=.965, bottom=.145, top=.825,
                        hspace=.64, wspace=.52)
    fig.suptitle(title, x=.105, y=.962, ha='left', fontsize=15, fontweight='bold')
    fig.text(.105, .91,
             f"Run: {result['status']}  |  Finite-source validation: "
             f"{result['validation_passed']}  |  Uniform 30 × 30 mm suspended film",
             fontsize=9)
    fig.text(.105, .035, footer, fontsize=8, va='bottom', linespacing=1.6)
    for letter, ax in zip('abcd', axes.flat):
        ax.annotate(letter, xy=(0, 1), xycoords='axes fraction',
                    xytext=(-39, 18), textcoords='offset points',
                    fontsize=11, fontweight='bold', annotation_clip=False)
        ax.tick_params(direction='out', length=3, width=.7)
    return fig, axes


def colorbar_in_gutter(fig, ax, image, label):
    box = ax.get_position()
    cax = fig.add_axes([box.x1+.014, box.y0+.02, .011, box.height-.04])
    bar = fig.colorbar(image, cax=cax)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.set_label(label, fontsize=8)


def phase_relative(values, power):
    """Unwrap only; neither fit nor compensate any propagation phase."""
    if np.any(np.abs(values) == 0) or abs(power) == 0:
        raise ValueError('phase is undefined for a zero-amplitude signal')
    return np.rad2deg(np.unwrap(np.angle(values/power)))


def get_curves(result):
    curves = result['curves']
    z = np.asarray(curves['axis_z_m'])
    lateral = np.asarray(curves['lateral_x_m'])
    if not np.all(np.diff(z) > 0) or not np.all(np.diff(lateral) > 0):
        raise ValueError('saved curve coordinates must be strictly increasing')
    return (z, complex_values(curves['kwave_axis_pressure']),
            complex_values(curves['reference_axis_pressure']), lateral,
            complex_values(curves['kwave_lateral_pressure']),
            complex_values(curves['reference_lateral_pressure']))


def field_figure(result, data):
    source = result['source']
    z, axis, reference, lateral, lateral_p, lateral_ref = get_curves(result)
    pressure = complex_values(source['source_pressure'])
    power_item = source.get('input_power', source.get('input_power_W_m2'))
    if power_item is None:
        raise ValueError('original input-power phasor is required for phase plotting')
    power = complex_values(power_item)
    fig, axes = layout('A finite thermal source radiates into both half-spaces', result,
        'One-way coupling: local LBM source → uniform free-space monopole sheet → homogeneous lossless air.\n'
        'Finite-aperture propagation is modeled; edge thermal feedback is absent and film heat capacity remains assumed.')
    x_field, y_field, z_field = (np.asarray(data[k]) for k in ('x_m', 'y_m', 'z_m'))
    field = np.asarray(data['pressure_Pa'])
    expected = (len(x_field), len(y_field), len(z_field))
    if field.shape != expected:
        raise ValueError(f'pressure_Pa shape {field.shape} does not match {expected}')
    central_y = np.argmin(abs(y_field))
    im = axes[0, 0].pcolormesh(x_field*1e3, z_field*1e3,
                              abs(field[:, central_y, :]).T,
                              shading='auto', cmap='viridis', rasterized=True)
    half = source['film_width_m']*500
    axes[0, 0].plot([-half, half], [0, 0], color='white', linewidth=2.8)
    axes[0, 0].plot([-half, half], [0, 0], color='black', linewidth=.8)
    axes[0, 0].set(xlabel='Lateral coordinate x (mm)', ylabel='Height z (mm)')
    axes[0, 0].set_title('Central plane; film at z = 0', loc='left')
    colorbar_in_gutter(fig, axes[0, 0], im, 'Pressure amplitude (Pa)')
    axes[0, 1].plot(z*1e3, abs(axis), color=BLUE, label='k-Wave')
    axes[0, 1].plot(z*1e3, abs(reference), '--', color=ORANGE, label='Green-function integral')
    axes[0, 1].axhline(float(abs(pressure)), linestyle=':', color=GREY,
                       label='Infinite-plane amplitude')
    axes[0, 1].set(xlabel='Height above film (mm)', ylabel='Pressure amplitude (Pa)',
                   xlim=(0, 45))
    axes[0, 1].set_title('On-axis finite-aperture response', loc='left')
    axes[0, 1].legend(loc='upper center', bbox_to_anchor=(.5, 1.0), fontsize=7.5)
    ymax = max(float(abs(pressure)), np.max(abs(axis)), np.max(abs(reference)))
    axes[0, 1].set_ylim(0, ymax*1.38)
    axes[1, 0].plot(lateral*1e3, abs(lateral_p), color=BLUE, label='k-Wave')
    axes[1, 0].plot(lateral*1e3, abs(lateral_ref), '--', color=ORANGE,
                    label='Green-function integral')
    axes[1, 0].set(xlabel='Lateral coordinate x (mm)', ylabel='Pressure amplitude (Pa)')
    axes[1, 0].set_title('Lateral profile at z = 45 mm', loc='left')
    axes[1, 1].plot(z*1e3, phase_relative(axis, power), color=BLUE)
    axes[1, 1].plot(z*1e3, phase_relative(reference, power), '--', color=ORANGE)
    axes[1, 1].set(xlabel='Height above film (mm)',
                   ylabel='Phase relative to input power (deg)', xlim=(0, 45))
    axes[1, 1].set_title('Original-clock phase; no fitted offset', loc='left')
    return fig, list(axes.flat)


def validation_figure(result, data):
    z, pressure, reference, *_ = get_curves(result)
    fig, axes = layout('Separate numerical checks bound the propagation model', result,
        'Deterministic runs; symbols show numerical differences, not statistical uncertainty. Limits: 1% and 1°.\n'
        'Agreement validates this source and propagation model; it does not establish measured-device accuracy.')
    resolved = abs(reference) > 0
    if not np.all(resolved):
        raise ValueError('reference pressure contains unresolved amplitudes')
    amp = (abs(pressure)/abs(reference)-1)*100
    phase = np.rad2deg(np.angle(pressure/reference))
    axes[0, 0].plot(z*1e3, abs(amp), color=BLUE, label='Amplitude / 1%')
    axes[0, 0].plot(z*1e3, abs(phase), '--', color=ORANGE, label='Phase / 1°')
    axes[0, 0].axhline(1, color=GREY, linestyle=':')
    axes[0, 0].set(xlabel='Height above film (mm)', ylabel='Absolute difference / limit',
                   xlim=(float(z[0]*1e3), 45))
    axes[0, 0].set_title('k-Wave versus independent integral', loc='left')
    axes[0, 0].legend(loc='upper right', fontsize=7.5)
    comparisons = result['metrics']['comparisons']
    index = np.arange(len(comparisons))
    for key, color, marker, label in (
            ('amplitude_error_percent', BLUE, 'o', 'Amplitude / 1%'),
            ('phase_error_deg', ORANGE, 's', 'Phase / 1°')):
        values = [abs(row[key]) for row in comparisons]
        axes[0, 1].plot(index, values, marker, color=color, ms=4, label=label)
    axes[0, 1].axhline(1, color=GREY, linestyle=':')
    axes[0, 1].set_xticks(index, [row['label'].replace(' ', '\n') for row in comparisons])
    axes[0, 1].set_ylabel('Absolute difference / limit')
    axes[0, 1].set_title('Normalization and numerical sensitivity', loc='left')
    axes[0, 1].legend(loc='upper right', fontsize=7.5)
    x_field, y_field, z_field = (np.asarray(data[k]) for k in ('x_m', 'y_m', 'z_m'))
    center = np.asarray(data['pressure_Pa'])[np.argmin(abs(x_field)), np.argmin(abs(y_field)), :]
    positive = np.flatnonzero(z_field > 0)
    opposite = np.array([np.argmin(abs(z_field+z_field[i])) for i in positive])
    if not np.allclose(z_field[opposite], -z_field[positive], rtol=0, atol=1e-10):
        raise ValueError('field grid does not support matching upper/lower samples')
    if np.any(abs(center[positive]) == 0):
        raise ValueError('zero on-axis pressure prevents symmetry assessment')
    residual = abs(center[positive]-center[opposite])/abs(center[positive])*100
    axes[1, 0].plot(z_field[positive]*1e3, residual, color=GREEN)
    axes[1, 0].set(xlabel='Distance from film |z| (mm)',
                   ylabel='Upper/lower complex difference (%)')
    axes[1, 0].set_title('Free-space two-sided symmetry', loc='left')
    weights = np.asarray(data['source_weights'])
    if weights.shape != (len(x_field), len(y_field)):
        raise ValueError('source_weights must be the x-y aperture weights')
    dx, dy = float(np.diff(x_field)[0]), float(np.diff(y_field)[0])
    area = float(weights.sum()*dx*dy)
    im = axes[1, 1].pcolormesh(x_field*1e3, y_field*1e3, weights.T,
                              shading='auto', cmap='Greys', vmin=0, vmax=1,
                              rasterized=True)
    axes[1, 1].set(xlabel='Lateral coordinate x (mm)', ylabel='Lateral coordinate y (mm)')
    axes[1, 1].set_title(f'Weighted aperture area: {area*1e6:.3f} mm²', loc='left')
    return fig, list(axes.flat)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--alignment-script', type=Path,
                        help='Rendered-layout auditor; mandatory for final delivery')
    args = parser.parse_args()
    batch = args.batch.resolve(); output = batch/'final'
    result = json.loads((output/'finite_film.json').read_text(encoding='utf-8'))
    for case, files in result.get('raw_sha256', {}).items():
        for filename, digest in files.items():
            with (batch/'raw'/case/filename).open('rb') as stream:
                if hashlib.file_digest(stream, 'sha256').hexdigest() != digest:
                    raise ValueError(f'raw data changed after analysis: {case}/{filename}')
    paths = [output/'finite_film.pdf', output/'finite_field.png', output/'finite_validation.png']
    if any(path.exists() for path in paths):
        raise FileExistsError('figures already exist; select a new output batch')
    require_matplotlib_panel_alignment = None
    if args.alignment_script:
        require_matplotlib_panel_alignment = runpy.run_path(str(args.alignment_script))['require_matplotlib_panel_alignment']
        qa = batch/'_work'/'figure_qa'; qa.mkdir(parents=True, exist_ok=True)
    style()
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    with np.load(batch/'raw'/'finite_fine'/'trajectory.npz') as data, PdfPages(paths[0]) as pdf:
        for maker, path in zip((field_figure, validation_figure), paths[1:]):
            fig, axes = maker(result, data)
            fig.canvas.draw()
            if require_matplotlib_panel_alignment:
                require_matplotlib_panel_alignment(fig, axes=axes, json_out=qa/f'{path.stem}.alignment.json',
                    tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
            pdf.savefig(fig)
            fig.savefig(path, dpi=300)
            plt.close(fig)
    print(paths[0])


if __name__ == '__main__':
    main()
