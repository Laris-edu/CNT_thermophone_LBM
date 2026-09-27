"""Run selected cases or parameter sweeps from Compute/<topic>/study.yaml."""
import argparse
from pathlib import Path
from simulation.runner import plan_study, run_study


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--batch',required=True,help='new batch name, e.g. 20260919_baseline')
    parser.add_argument('--case',nargs='+',help='optional case names; otherwise run the whole study')
    parser.add_argument('--dry-run',action='store_true',help='validate and list jobs without writing files')
    args=parser.parse_args()
    try:
        jobs=plan_study(args.config,args.batch,args.case)
        if args.dry_run:
            for config,output in jobs:
                layout=config.get('grid_layout')
                detail=(f", coarse/sync steps={layout['sync_steps']}, physical cells="
                        f"{layout['fine_cells']}+{layout['coarse_cells']}, shadows={layout['shadow_cells']}, "
                        f"cell updates={layout['cell_updates']}" if layout else '')
                print(f"{config['case']['name']}: {config['run']['mode']}, fine steps={config['numerics']['steps']}{detail} -> {output}")
            return 0
        run_study(args.config,args.batch,selected=args.case)
    except (ValueError,FileExistsError) as exc:
        parser.exit(2,f'{exc}\n')
    except KeyboardInterrupt:
        parser.exit(130,'Interrupted; partial data and status retained.\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
