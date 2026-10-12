"""Plan/run finite-strip cases; do not launch power before boundary evidence."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from simulation.finite_strip import prepare_strip_case,run_strip_case,compare_pulse_cases
from simulation.output import dump
from simulation.study import valid_name


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_boundary_evidence(root,config,*,source_id=None,proof_path=None):
    """Bind the gate to actual saved pulses, absorber geometry and runtime."""
    from simulation.provenance import capture_source
    from simulation.study import ROOT
    proof_file=Path(proof_path) if proof_path is not None else root/'final/boundary_validation.json'
    proof=json.loads(proof_file.read_text(encoding='utf-8'))
    if proof.get('boundary_validation_passed') is not True or proof.get('limit',1)>.01:
        raise ValueError('boundary validation has not passed the 1% gate')
    for label in ('combined','top','sides'):
        if proof.get('components',{}).get(label,{}).get('boundary_validation_passed') is not True:
            raise ValueError(f'missing passed {label} boundary evidence')
    expected_source_id=source_id or capture_source(ROOT)['source_id']
    provenance=proof.get('pulse_provenance',{})
    for label in ('combined','top','sides','reference'):
        item=provenance.get(label)
        if item is None:
            raise ValueError(f'missing {label} boundary provenance')
        directory=root/'raw'/valid_name(item['name'])
        for name,digest in item['sha256'].items():
            if name not in ('effective.yaml','manifest.json','trajectory.npz'):
                raise ValueError('unsupported boundary evidence file')
            if _digest(directory/name)!=digest:
                raise ValueError(f'boundary evidence changed: {label}/{name}')
        if (item['source_id']!=expected_source_id or not item['source_clean']
                or not item['git_commit']):
            raise ValueError('boundary evidence does not match the committed public runtime')
    candidate=yaml.safe_load((root/'raw'/provenance['combined']['name']/'effective.yaml').read_text())
    for key in ('width_m','height_m','pml_width_m','pml_target_return','pml_alpha_Hz','sponge_rate'):
        if candidate['strip'][key]!=config['strip'][key]:
            raise ValueError(f'boundary geometry/absorber differs from power run: {key}')
    for key in ('physical','transport','collision'):
        if candidate[key]!=config[key]:
            raise ValueError(f'boundary physics differs from power run: {key}')
    return proof


def load_cases(path):
    spec=yaml.safe_load(path.read_text(encoding='utf-8'))
    cases=[prepare_strip_case({**spec.get('defaults',{}),**p},path.parent/spec['base_config']) for p in spec['cases']]
    for case in cases:
        case['material']=spec['material']
    return spec,{c['case']['name']:c for c in cases}


def compare_batch(root,spec):
    small=root/'raw/pulse_candidate_dx10um'; big=root/'raw/pulse_reference_dx10um'
    cases={'combined':small,'top':root/'raw/pulse_top_dx10um','sides':root/'raw/pulse_sides_dx10um'}
    configurations={}
    for directory in (*cases.values(),big):
        summary=json.loads((directory/'summary.json').read_text())
        if summary['status']!='completed' or not summary['state_validation_passed']:
            raise ValueError('all boundary pulses must complete with state and conservation checks')
        configurations[directory]=yaml.safe_load((directory/'effective.yaml').read_text())
    components={}
    for label,path in cases.items():
        with np.load(path/'trajectory.npz') as a,np.load(big/'trajectory.npz') as b:
            component=compare_pulse_cases(a,b,return_start_s=spec['validation']['pulse_return_start_s'],
                limit=spec['validation']['pulse_return_error_limit'],
                trial_effective=configurations[path],reference_effective=configurations[big])
            if label!='combined':
                from core.unit_mapping import create_unit_mapping
                cfg=configurations[path]; run=cfg['strip']
                distance=run['actual_width_m']/2 if label=='top' else run['actual_height_m']
                distance-=max(run['sponge_cells'],run.get('pml_cells',0))*run['dx_m']
                earliest=(2*distance-run['probe_radius_m']-6*run['pulse_sigma_m'])/create_unit_mapping(cfg).physical.c0_m_s
                if earliest<=float(a['t_s'][-1]):
                    raise ValueError(f'{label} isolation contaminated by the other boundary in this window')
                component['other_boundary_return_lower_bound_s']=earliest
            components[label]=component
    provenance={}
    for label,path in {**cases,'reference':big}.items():
        manifest=json.loads((path/'manifest.json').read_text())
        provenance[label]=dict(name=path.name,
            sha256={name:_digest(path/name) for name in ('effective.yaml','manifest.json','trajectory.npz')},
            source_id=manifest['source_id'],git_commit=manifest['git_commit'],source_clean=manifest['source_clean'])
    result=dict(components=components,pulse_provenance=provenance,
        boundary_validation_passed=all(c['boundary_validation_passed'] for c in components.values()),
        max_return_error_ratio=max(c['max_return_error_ratio'] for c in components.values()),
        limit=spec['validation']['pulse_return_error_limit'],frequency_20kHz_validated=False,
        propagation_validation_passed=None,
        scope='Combined, isolated top, and isolated sides broadband pulse; not a 20 kHz certificate')
    root.joinpath('final').mkdir(exist_ok=True)
    destination=root/'final/boundary_validation.json'
    if destination.exists():
        raise FileExistsError(destination)
    dump(destination,result)
    summary=json.loads((small/'summary.json').read_text())
    summary.update(boundary_validation_passed=result['boundary_validation_passed'],
        validation_passed=result['boundary_validation_passed'],boundary_comparison=result)
    dump(small/'summary.json',summary)
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).with_name('study.yaml'))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--case')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--compare-pulses',action='store_true')
    args=parser.parse_args()
    spec,cases=load_cases(args.study.resolve())
    root=args.study.resolve().parent/valid_name(args.batch)
    if args.compare_pulses:
        compare_batch(root,spec); return
    selected=[cases[args.case]] if args.case else list(cases.values())
    for config in selected:
        output=root/'raw'/config['case']['name']; run=config['strip']
        print(json.dumps(dict(name=config['case']['name'],nx=config['numerics']['nx'],ny=config['numerics']['ny'],
            dx_m=config['lattice']['dx_m'],dt_s=config['lattice']['dt_s'],steps=run['steps'],
            output=str(output),exists=output.exists(),power_W_per_m=run['mean_power_W_m2']*run['film_width_m']),indent=2))
        if output.exists():
            raise FileExistsError(output)
        if args.dry_run:
            continue
        if run['kind']=='power':
            validate_boundary_evidence(root,config)
            # Passing this prerequisite permits nearfield testing only. The
            # resulting 20 kHz field still needs grid/domain/contour validation.
        run_strip_case(config,output)


if __name__=='__main__':
    main()
