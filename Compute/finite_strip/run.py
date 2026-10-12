"""Plan/run finite-strip cases; do not launch power before boundary evidence."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from simulation.finite_strip import prepare_strip_case,run_strip_case,compare_pulse_cases
from simulation.output import dump
from simulation.study import valid_name


def load_cases(path):
    spec=yaml.safe_load(path.read_text(encoding='utf-8'))
    cases=[prepare_strip_case({**spec.get('defaults',{}),**p},path.parent/spec['base_config']) for p in spec['cases']]
    for case in cases:
        case['material']=spec['material']
    return spec,{c['case']['name']:c for c in cases}


def compare_batch(root,spec):
    small=root/'raw/pulse_candidate_dx10um'; big=root/'raw/pulse_reference_dx10um'
    results=[]
    for directory in (small,big):
        summary=json.loads((directory/'summary.json').read_text())
        if summary['status']!='completed' or not summary['state_validation_passed']:
            raise ValueError('both pulses must complete with state and conservation checks')
        results.append(yaml.safe_load((directory/'effective.yaml').read_text()))
    with np.load(small/'trajectory.npz') as a,np.load(big/'trajectory.npz') as b:
        result=compare_pulse_cases(a,b,return_start_s=spec['validation']['pulse_return_start_s'],
            limit=spec['validation']['pulse_return_error_limit'],trial_effective=results[0],reference_effective=results[1])
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
            evidence=root/'final/boundary_validation.json'
            if not evidence.exists():
                raise RuntimeError('Power run gated: the preliminary boundary pulse must be compared first')
            proof=json.loads(evidence.read_text())
            if proof.get('boundary_validation_passed') is not True:
                raise RuntimeError('Preliminary boundary validation has not passed; power run not started')
            # Passing this prerequisite permits nearfield testing only. The
            # resulting 20 kHz field still needs grid/domain/contour validation.
        run_strip_case(config,output)


if __name__=='__main__':
    main()
