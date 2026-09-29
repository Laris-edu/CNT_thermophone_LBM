"""Audit saved thermal-step evidence; optionally run one moved-interface pair.

Default operation only reads existing data and writes one new audit JSON. Run
from the repository root with ``python -m Compute.multi_grid.audit_interface``.
Raw moments reconstructed from saved physical heat flux are explicitly labelled;
they are not an independent population-level check.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import yaml

from Compute.multi_grid.compare import load_case, step_comparison
from core.unit_mapping import create_unit_mapping, local_transport_coefficients
from simulation import prepare_case, run_case


HERE = Path(__file__).resolve().parent


def _point_readout(config, data, time_index, row):
    """Invert the public local readout, using saved density and temperature."""
    level = int(data['level_index'][row])
    local_config = deepcopy(config)
    if level == 1:
        for key in ('dx_m', 'dt_s'):
            local_config['lattice'][key] *= 2
    elif level != 0:
        raise ValueError('expected level 0 or 1')
    mapping = create_unit_mapping(local_config)
    rho = data['profile_density_kg_m3'][time_index, row] / mapping.rho_scale
    theta = data['profile_temperature_K'][time_index, row] / mapping.temperature_scale
    coefficients = local_transport_coefficients(mapping, rho, theta)
    factor = float(coefficients.heat_flux_readout)
    physical = float(data['profile_conductive_heat_flux_W_m2'][time_index, row])
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError('invalid local heat-flux readout')
    return dict(level=level, row_zero_based=int(row), z_m=float(data['z_m'][row]),
        density_kg_m3=float(rho*mapping.rho_scale),
        temperature_K=float(theta*mapping.temperature_scale),
        tau32=float(coefficients.tau32), local_readout_factor=factor,
        heat_flux_scale=mapping.heat_flux_scale, physical_q_W_m2=physical,
        reconstructed_raw_central_moment_lu=physical/mapping.heat_flux_scale/factor,
        raw_moment_evidence='inverse of saved physical q using public local mapping; populations unavailable')


def audit_pair(batch):
    batch = Path(batch).resolve()
    paths = [batch/'raw'/f'step_{kind}' for kind in ('two_level', 'uniform')]
    ca, sa, ma, a = load_case(paths[0])
    cb, sb, mb, b = load_case(paths[1])
    if ma['git_commit'] != mb['git_commit'] or ma['source_id'] != mb['source_id']:
        raise ValueError('paired saved results have different source provenance')
    if not np.array_equal(a['profile_t_s'], b['profile_t_s']):
        raise ValueError('saved profiles must have identical physical times')
    metrics = step_comparison(a, b, ca)
    qa = a['profile_conductive_heat_flux_W_m2']
    qb = np.array([np.interp(a['z_m'], b['z_m'], row)
                   for row in b['profile_conductive_heat_flux_W_m2']])
    difference = qa-qb
    ti, row = np.unravel_index(np.argmax(abs(difference)), difference.shape)
    reference_row = int(np.argmin(abs(b['z_m']-a['z_m'][row])))
    same_position = bool(b['z_m'][reference_row] == a['z_m'][row])
    nf = int(np.count_nonzero(a['level_index'] == 0))
    point = dict(snapshot_index=int(ti), time_s=float(a['profile_t_s'][ti]),
        two_level=_point_readout(ca, a, ti, row),
        nearest_uniform=_point_readout(cb, b, ti, reference_row),
        exactly_same_position=same_position,
        nearest_position_difference_m=float(b['z_m'][reference_row]-a['z_m'][row]),
        fine_cell_count=nf, fine_rows_after_peak=nf-1-int(row) if row < nf else None,
        comparison_uniform_q_W_m2=float(qb[ti, row]),
        signed_difference_W_m2=float(difference[ti, row]),
        normalization_reference_global_max_W_m2=float(np.max(abs(qb))))
    start, stop = max(0, nf-6), min(len(a['z_m']), nf+4)
    neighborhood = [dict(row_zero_based=j, level=int(a['level_index'][j]),
        z_m=float(a['z_m'][j]), two_level_q_W_m2=float(qa[ti, j]),
        interpolated_uniform_q_W_m2=float(qb[ti, j])) for j in range(start, stop)]
    provenance = [dict(case=str(path), git_commit=manifest['git_commit'],
        source_id=manifest['source_id'], git_dirty=manifest.get('git_dirty'),
        source_clean=manifest.get('source_clean'), status=manifest['status'],
        validation_passed=summary['validation_passed'])
        for path, manifest, summary in zip(paths, (ma, mb), (sa, sb))]
    return dict(batch=str(batch), saved_case_provenance=provenance,
        metrics=metrics, peak=point, peak_time_interface_neighborhood=neighborhood,
        snapshot_times_s=a['profile_t_s'].tolist(),
        limitations=['Maximum covers saved snapshots, not every synchronized step.',
            'Reconstructed raw moments do not independently validate saved q or collision.',
            'Conservation and temperature agreement do not establish local conductive-moment accuracy.',
            'Interface-localized evidence alone cannot distinguish algorithmic truncation from an implementation defect.'])


def moved_configs(interface_m):
    """Expand the existing study's step parameters through the public API."""
    study = yaml.safe_load((HERE/'study.yaml').read_text(encoding='utf-8'))
    base = (HERE/study['base_config']).resolve()
    cases = {case['name']: case for case in study['cases']}
    configs = []
    for kind in ('two_level', 'uniform'):
        parameters = deepcopy(study['defaults'])
        parameters.update(deepcopy(cases[f'step_{kind}']))
        parameters['thermal_step'] = dict(duration_s=5e-6, interface_m=interface_m)
        if kind == 'two_level':
            if interface_m == cases['step_two_level']['grid']['interface_m']:
                raise ValueError('moved interface must differ from the baseline interface')
            parameters['grid']['interface_m'] = interface_m
        configs.append(prepare_case(parameters, base))
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='Existing baseline batch with final/comparison.json')
    parser.add_argument('--step-refinement', type=Path, help='Optional existing finer thermal-step batch')
    parser.add_argument('--output-batch', type=Path, required=True, help='New, nonexistent batch directory')
    parser.add_argument('--run-moved-interface', action='store_true', help='Explicitly enable one new 5 us paired computation')
    parser.add_argument('--interface-m', type=float, help='New exact common cell face; required with --run-moved-interface')
    args = parser.parse_args()
    if args.run_moved_interface != (args.interface_m is not None):
        parser.error('--run-moved-interface and --interface-m must be supplied together')
    output = args.output_batch.resolve()
    if output.exists():
        raise FileExistsError(f'output batch already exists: {output}')
    comparison_path = args.baseline.resolve()/'final/comparison.json'
    comparison = json.loads(comparison_path.read_text(encoding='utf-8'))
    result = dict(scope='Read-only baseline audit plus explicitly requested short interface experiment',
        saved_comparison=dict(path=str(comparison_path),
            comparison_passed=comparison['comparison_passed'],
            step=comparison['pairs']['step']),
        baseline=audit_pair(args.baseline),
        diagnosis='Peak coordinate, mapping and localization evidence; no threshold or population modification.')
    if args.step_refinement:
        result['step_refinement'] = audit_pair(args.step_refinement)
    configs = moved_configs(args.interface_m) if args.run_moved_interface else []
    # Expand and display the actual work and destinations before any computation.
    for config in configs:
        print(json.dumps(dict(name=config['case']['name'],
            fine_steps=config['numerics']['steps'], grid=config.get('grid'),
            output=str(output/'raw'/config['case']['name'])), ensure_ascii=False), flush=True)
    output.mkdir(parents=True, exist_ok=False)
    if configs:
        for config in configs:
            run_case(config, output/'raw'/config['case']['name'])
        result['moved_interface'] = audit_pair(output)
    final = output/'final'
    final.mkdir()
    path = final/'baseline_audit.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(path)


if __name__ == '__main__':
    main()
