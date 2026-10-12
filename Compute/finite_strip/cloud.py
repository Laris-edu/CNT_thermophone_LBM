"""Launch finite-strip LBM stages in fresh processes with explicit CPU settings."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
BOUNDARY_CASES = ('pulse_candidate_dx10um', 'pulse_reference_dx10um',
                  'pulse_sides_dx10um', 'pulse_top_dx10um')


def available_cpus():
    """Respect the process CPU affinity where the platform exposes it."""
    count = (getattr(os, 'process_cpu_count', os.cpu_count)() or 1)
    if hasattr(os, 'sched_getaffinity'):
        count = min(count, len(os.sched_getaffinity(0)))
    return count


def _positive_integer(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--study', type=Path, default=Path(__file__).with_name('study.yaml'))
    parser.add_argument('--batch', required=True)
    parser.add_argument('--stage', choices=('boundary', 'power'), default='boundary')
    parser.add_argument('--threads', type=_positive_integer, default=64)
    parser.add_argument('--parallel-min-cells', type=_positive_integer, default=131072)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    cpus = available_cpus()
    if not args.dry_run and args.threads > cpus:
        raise ValueError(f'requested {args.threads} threads but only {cpus} CPUs are available')
    imported = sys.modules.get('numba')
    if imported is not None and (imported.config.NUMBA_NUM_THREADS != args.threads
                                or imported.get_num_threads() != args.threads):
        raise RuntimeError('Numba already has a different thread pool; launch cloud in a fresh Python process')
    settings = dict(NUMBA_NUM_THREADS=str(args.threads),
                    CNT_LBM_PARALLEL_MIN_CELLS=str(args.parallel_min_cells),
                    OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', BLIS_NUM_THREADS='1',
                    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
                    OMP_NUM_THREADS=str(args.threads), OMP_DYNAMIC='FALSE')
    environment = {**os.environ, **settings}
    study = args.study.resolve()
    print(json.dumps(dict(stage=args.stage, dry_run=args.dry_run, available_cpus=cpus,
        requested_threads_available=args.threads <= cpus, environment=settings), indent=2), flush=True)
    command = [sys.executable, '-m', 'Compute.finite_strip.run',
               '--study', str(study), '--batch', args.batch]
    selected = BOUNDARY_CASES if args.stage == 'boundary' else ('power_r2',)
    # The existing entry point expands physical configurations, checks all
    # requested output names and prints grid/steps/paths without running LBM.
    for case in selected:
        subprocess.run([*command, '--case', case, '--dry-run'], cwd=ROOT,
                       env=environment, check=True)
    if args.stage == 'boundary':
        proof = study.parent/args.batch/'final/boundary_validation.json'
        if proof.exists():
            raise FileExistsError(proof)
    if args.dry_run:
        return 0
    for case in selected:
        subprocess.run([*command, '--case', case], cwd=ROOT, env=environment, check=True)
    if args.stage == 'boundary':
        subprocess.run([*command, '--compare-pulses'], cwd=ROOT, env=environment, check=True)
        result = json.loads(proof.read_text(encoding='utf-8'))
        limit, error = result.get('limit'), result.get('max_return_error_ratio')
        passed = (result.get('boundary_validation_passed') is True
                  and all(result.get('components', {}).get(label, {}).get('boundary_validation_passed') is True
                          for label in ('combined', 'top', 'sides'))
                  and type(limit) in (int, float) and math.isfinite(limit) and 0 < limit <= .01
                  and type(error) in (int, float) and math.isfinite(error) and 0 <= error < limit)
        print(json.dumps(dict(stage='boundary', run_status='completed',
            boundary_validation_passed=passed, limit=limit, max_return_error_ratio=error,
            evidence=str(proof)), indent=2), flush=True)
    else:
        raw = study.parent/args.batch/'raw/power_r2'
        manifest = json.loads((raw/'manifest.json').read_text(encoding='utf-8'))
        summary = json.loads((raw/'summary.json').read_text(encoding='utf-8'))
        # The raw runner deliberately leaves public validation unset. Keep it
        # distinct from this stage's state/periodicity gate and the later local
        # all-contour source analysis; normal process exit proves none of them.
        passed = (manifest.get('status') == summary.get('status') == 'completed'
                  and summary.get('state_validation_passed') is True
                  and summary.get('periodic_response_stable') is True
                  and summary.get('validation_passed') is not False)
        print(json.dumps(dict(stage='power', run_status=manifest.get('status'),
            summary_status=summary.get('status'), validation_passed=summary.get('validation_passed'),
            state_validation_passed=summary.get('state_validation_passed'),
            periodic_response_stable=summary.get('periodic_response_stable'),
            nearfield_state_periodic_gate_passed=passed, source_validation_passed=None,
            propagation_validation_passed=None,
            scope='Saved state and periodic response only; source and propagation checks require the local chain.'),
            indent=2), flush=True)
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
