"""Compatibility CLI for open-planar cases; implementation lives in simulation."""
from __future__ import annotations

import argparse
from pathlib import Path

from simulation import prepare_case, run_case
from simulation.planar import make_solver, pulse_case, thermal_case
from simulation.study import valid_name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pulse','thermal'],required=True)
    parser.add_argument('--refine',type=int,default=4)
    parser.add_argument('--length',type=float,default=1e-3)
    parser.add_argument('--frequency',type=float,default=50000.)
    parser.add_argument('--amplitude',type=float,default=.1)
    parser.add_argument('--cycles',type=float,default=6)
    parser.add_argument('--ramp-cycles',type=float,default=2)
    parser.add_argument('--probes',type=float,nargs='+',default=[.12e-3,.30e-3,.60e-3])
    parser.add_argument('--output',type=Path,required=True,help='batch root; case data go in raw/')
    parser.add_argument('--case',help='optional case name under raw/')
    args=parser.parse_args()
    try:
        name=valid_name(args.case) if args.case else 'case'
        effective=prepare_case(dict(name=name,mode=args.mode,refine=args.refine,length_m=args.length,
            frequency_Hz=args.frequency,amplitude_K=args.amplitude,cycles=args.cycles,
            ramp_cycles=args.ramp_cycles,probes_m=args.probes))
        output=args.output/'raw'/name if args.case else args.output/'raw'
        result=run_case(effective,output)
    except (ValueError,FileExistsError) as exc:
        parser.exit(2,f'{exc}\n')
    if args.mode=='pulse':
        print(f"reflection={result['reflection_ratio']:.6%}, passed={result['pass_reflection']}")
    else:
        print('gates:',result['gates'])


if __name__=='__main__':
    main()
