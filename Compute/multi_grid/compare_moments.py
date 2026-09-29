"""Compare saved uniform nine-moment/population runs; no simulation or time shifts."""
import argparse
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np

from Compute.multi_grid.compare import PAIRS, load_case, pulse_comparison, step_comparison, harmonic_comparison
from simulation.output import dump


KINDS = ('moments9', 'populations')
TIMINGS = {'warmup_s', 'advance_s', 'elapsed_s'}
STATE_METADATA = {'moment_state_final', 'state_storage_bytes', 'representation_equivalence_only'}
AXES = {'t_s', 'heat_time_s', 'film_t_s', 'profile_t_s', 'probe_z_m', 'z_m',
        'cell_width_m', 'sync_fine_step', 'level_index'}


def physical_config(config):
    config = deepcopy(config)
    config['case'].pop('name', None)
    config['numerics'].pop('state_representation', None)
    return config


def array_difference(a, b, rtol, atol):
    if a.shape != b.shape:
        raise ValueError(f'array shapes differ: {a.shape}, {b.shape}')
    if a.dtype.kind not in 'biufc' or b.dtype.kind not in 'biufc':
        return dict(equal=bool(np.array_equal(a, b)), passed=bool(np.array_equal(a, b)))
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite saved numeric data')
    delta = a.astype(float) - b.astype(float)
    norm = float(np.linalg.norm(b.astype(float).ravel()))
    error = float(np.linalg.norm(delta.ravel()))
    # Centering changes neither sample times nor phases; expose cancellation by DC.
    centered = b.astype(float) - (np.mean(b, axis=0) if b.ndim and b.size else b)
    fluctuation = float(np.linalg.norm(centered.ravel()))
    return dict(shape=list(a.shape), max_absolute_error=float(np.max(abs(delta), initial=0)),
        rms_error=float(np.sqrt(np.mean(delta**2))) if delta.size else 0.,
        unshifted_relative_l2=error/norm if norm else None,
        unshifted_error_over_reference_fluctuation_l2=error/fluctuation if fluctuation else None,
        passed=bool(np.allclose(a, b, rtol=rtol, atol=atol)))


def final_moment_check(data, config, rtol, atol):
    from core.kinetic_moments import recover_moment_macro
    from core.unit_mapping import create_unit_mapping
    required={'moment_state_final','profile_t_s','t_s','profile_density_kg_m3',
        'profile_pressure_Pa','profile_velocity_m_s','profile_temperature_K','gas_energy_J_m2','cell_width_m'}
    if required-data.keys():
        raise ValueError(f'missing final-state observations: {sorted(required-data.keys())}')
    z = data['moment_state_final']
    expected = (config['numerics']['ny'], config['numerics']['nx'], 9)
    if z.shape != expected or not np.isfinite(z).all():
        raise ValueError('final moment state must be finite and have shape (ny,nx,9)')
    mapping = create_unit_mapping(config)
    macro = recover_moment_macro(z, mapping)
    if data['profile_t_s'][-1] != data['t_s'][-1]:
        raise ValueError('last profile does not describe the final moment state')
    fields = dict(profile_density_kg_m3=macro.rho[:,0]*mapping.rho_scale,
        profile_pressure_Pa=macro.p[:,0]*mapping.pressure_scale,
        profile_velocity_m_s=macro.u[:,0,1]*mapping.velocity_scale,
        profile_temperature_K=macro.theta[:,0]*mapping.temperature_scale)
    checks = {key:array_difference(value, data[key][-1], rtol, atol)
              for key,value in fields.items()}
    energy = np.asarray(np.sum(macro.E_tot[:,0]*data['cell_width_m'])*mapping.pressure_scale)
    checks['gas_energy_J_m2'] = array_difference(energy, data['gas_energy_J_m2'][-1], rtol, atol)
    return dict(shape=list(z.shape), comparison='moments9 final state against its saved physical observations; no population final state saved',
                checks=checks, passed=all(v['passed'] for v in checks.values()))


def compare_batch(batch, rtol=1e-10, atol=1e-12, pairs=PAIRS):
    batch = Path(batch)
    results, curves = {}, {}
    for pair in pairs:
        paths = [batch/'raw'/f'{pair}_{kind}' for kind in KINDS]
        if not all(path.is_dir() for path in paths):
            raise FileNotFoundError(f'missing complete pair: {pair}')
        cases = [load_case(path) for path in paths]
        (ca, sa, ma, a), (cb, sb, mb, b) = cases
        for path, (config, summary, manifest, data), kind in zip(paths, cases, KINDS):
            if config['numerics'].get('state_representation', 'populations') != kind:
                raise ValueError(f'incorrect representation in {path}')
            if 'grid' in config or config['run']['sampling']['sync_fine_steps'] != 2:
                raise ValueError('comparison requires uniform grids and sync_fine_steps=2')
            digest = hashlib.sha256((path/'effective.yaml').read_bytes()).hexdigest()
            if digest != manifest.get('effective_sha256'):
                raise ValueError(f'effective config digest mismatch: {path}')
            if not bool(data['run_completed']) or summary['status'] != 'completed':
                raise ValueError(f'incomplete result: {path}')
        for key in ('git_commit', 'source_id', 'source_sha256'):
            if not ma.get(key) or ma[key] != mb.get(key):
                raise ValueError(f'paired source mismatch or missing {key}: {pair}')
        if physical_config(ca) != physical_config(cb):
            raise ValueError(f'physical configurations differ: {pair}')
        if a.keys()-STATE_METADATA != b.keys()-STATE_METADATA:
            raise ValueError(f'saved trajectory fields differ: {pair}')
        fields = {}
        for key in a:
            if key in STATE_METADATA:
                continue
            if key in AXES and not np.array_equal(a[key], b[key]):
                raise ValueError(f'physical observation axes differ: {pair}/{key}')
            if key == 'interface_exchange' and not bool(a['interface_exchange_available']) and not bool(b['interface_exchange_available']):
                if a[key].shape != b[key].shape or not np.array_equal(a[key], b[key], equal_nan=True):
                    raise ValueError('unavailable interface exchange markers differ')
                fields[key] = dict(available=False, passed=True)
                continue
            fields[key] = array_difference(a[key], b[key], rtol, atol)
            if key in TIMINGS:
                fields[key]['informational_only'] = True
        wall = [float(np.sum(d['wall_energy_J_m2'])) for d in (a, b)]
        moment_check = final_moment_check(a, ca, rtol, atol)
        physical_compare = (pulse_comparison if ca['run']['mode']=='interface_pulse' else
            step_comparison if ca['run']['mode']=='thermal_step' else harmonic_comparison)
        def label_operands(value):
            if isinstance(value,dict):
                return {key.replace('two_level','moments9').replace('uniform','populations'):
                    label_operands(item) for key,item in value.items()}
            return value
        physical = label_operands(physical_compare(a,b,ca))
        results[pair] = dict(fields=fields,
            physical_comparison=physical,
            final_moment_state=moment_check,
            state_metadata={kind:{key:np.asarray(case[3][key]).tolist()
                for key in STATE_METADATA-{'moment_state_final'} if key in case[3]}
                for kind,case in zip(KINDS,cases)},
            wall_cumulative_energy_J_m2=dict(zip(KINDS, wall)),
            wall_cumulative_difference_J_m2=wall[0]-wall[1],
            equivalence_passed=moment_check['passed'] and all(v['passed'] for k, v in fields.items() if k not in TIMINGS),
            validation_passed={k:c[1]['validation_passed'] for k,c in zip(KINDS,cases)},
            status={k:c[2]['status'] for k,c in zip(KINDS,cases)},
            provenance={k:{name:c[2].get(name) for name in
                ('git_commit','git_dirty','source_id','source_clean')} for k,c in zip(KINDS,cases)},
            timings_s={k:{name:float(c[3][name]) for name in TIMINGS} for k,c in zip(KINDS,cases)})
        curves[pair] = (a['t_s'], a['pressure_Pa']-b['pressure_Pa'])
    return dict(representation_equivalence_only=True, independent_field_grids=False,
        checked_pairs=list(pairs),unchecked_pairs=[pair for pair in PAIRS if pair not in pairs],
        operand_order=list(KINDS),
        gate_scope='same-grid representation equivalence only; not C/D or physical validation',
        comparison_passed=all(v['equivalence_passed'] for v in results.values()),
        tolerances=dict(rtol=rtol, atol=atol, rule='elementwise abs(a-b) <= atol + rtol*abs(populations)'),
        phase_convention='exp(+i omega t); no time shift, fitted gain, or phase compensation',
        channel_meanings=dict(pressure_Pa='direct gas pressure',
            wall_energy_J_m2='actual interval wall exchange; cumulative uses interval sum',
            film_temperature_K='film state', profile_conductive_heat_flux_W_m2='conductive moment readout'),
        pairs=results), curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--pair', choices=PAIRS, nargs='+', default=list(PAIRS),
                        help='Explicitly compare only completed pairs; absent groups remain untested')
    args = parser.parse_args()
    final = args.batch/'final'
    outputs = [final/'comparison.json', final/'representation_difference.png']
    if any(path.exists() for path in outputs):
        raise FileExistsError('comparison output already exists; preserve historical results')
    result, curves = compare_batch(args.batch,pairs=tuple(args.pair))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(args.pair), 1, figsize=(9, 2*len(args.pair)), constrained_layout=True,squeeze=False)
    for ax, pair in zip(axes[:,0], args.pair):
        t, difference = curves[pair]
        ax.plot(t*1e6, difference)
        ax.set(title=pair, xlabel='Time (us)', ylabel='Pressure difference (Pa)')
        ax.grid(alpha=.25)
    fig.suptitle('Same-grid representation diagnostic: moments9 minus populations')
    final.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[1], dpi=150)
    plt.close(fig)
    dump(outputs[0], result)
    print('same-grid representation equivalence:', result['comparison_passed'])


if __name__ == '__main__':
    main()
