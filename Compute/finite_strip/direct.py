"""User-requested direct power run and unvalidated k-Wave propagation; no pulses."""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import subprocess
import time
import os

import numpy as np
from scipy.io import loadmat

from Compute.finite_strip.run import load_cases
from simulation.finite_strip import run_strip_case
from simulation.study import ROOT, valid_name


def atomic_json(path, value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--study',type=Path,default=Path(__file__).with_name('study_30mm.yaml'))
    parser.add_argument('--batch',required=True)
    parser.add_argument('--matlab',type=Path,required=True)
    parser.add_argument('--toolbox',type=Path,required=True)
    args=parser.parse_args(argv)
    _,cases=load_cases(args.study.resolve())
    config=cases['power_r2']
    if config['strip']['kind']!='power' or config['numerics']['collision_backend']!='cupy':
        raise ValueError('direct entry requires a power-driven GPU case')
    if not args.matlab.is_file() or not (args.toolbox/'kspaceFirstOrder2D.m').is_file():
        raise FileNotFoundError('MATLAB/k-Wave installation is missing')
    batch=args.study.resolve().parent/valid_name(args.batch)
    batch.mkdir(exist_ok=False)
    work=batch/'_work';work.mkdir()
    def status(stage,**extra):
        value=dict(stage=stage,updated_utc=datetime.now(timezone.utc).isoformat(),
            validation_passed=None,boundary_validation='skipped_by_user_request',**extra)
        atomic_json(work/'run_status.json',value)
        print(json.dumps(value),flush=True)
    try:
        # Query free memory only; never signal or modify another GPU job.
        while True:
            output=subprocess.check_output(['nvidia-smi',
                '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            available=[(int(free.strip()),uuid.strip()) for uuid,free in
                       (line.split(',') for line in output.strip().splitlines())]
            free,uuid=max(available)
            if free>=20000:
                break
            status('waiting_for_gpu',maximum_free_MiB=free,required_free_MiB=20000)
            time.sleep(30)
        os.environ['CUDA_VISIBLE_DEVICES']=uuid
        config['execution']=dict(boundary_validation='skipped_by_user_request',
            purpose='Direct requested power result; no pulse, grid or boundary certification.')
        raw=batch/'raw/power_r2'
        status('lbm_running',gpu_uuid=uuid,free_MiB=free,steps=config['strip']['steps'])
        def progress(data):
            temporary=work/'progress.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,**data)
            temporary.replace(work/'progress.npz')
            temperatures=data['film_temperature_K'][-1]
            value=dict(accepted_steps=int(data['accepted_steps']),
                physical_time_s=float(data['physical_time_s']),elapsed_s=float(data['elapsed_s']),
                cycles=float(data['physical_time_s']*data['frequency_Hz']),
                film_temperature_min_K=float(np.min(temperatures)),
                film_temperature_max_K=float(np.max(temperatures)),
                maximum_observed_pressure_Pa=float(np.max(abs(data['pressure_perturbation_Pa'][-1]))),
                run_completed=False,validation_passed=None)
            atomic_json(work/'progress.json',value)
            print('NEARFIELD_PROGRESS '+json.dumps(value),flush=True)
        summary=run_strip_case(config,raw,progress_callback=progress)
        status('lbm_completed',summary=summary)
        # The actual mathematical checks are retained, but are not gates for
        # this explicitly requested unvalidated result route.
        from Compute.finite_strip.analyze import analyze_data,export_kwave_input
        from Compute.finite_strip.chain import validate_propagation
        import yaml
        effective=yaml.safe_load((raw/'effective.yaml').read_text())
        with np.load(raw/'trajectory.npz',allow_pickle=False) as data:
            analysis,arrays=analyze_data(data,effective,summary,
                dict(boundary_validation_passed=None,status='skipped_by_user_request'))
        final=batch/'final';final.mkdir()
        atomic_json(final/'power_r2_nearfield.json',analysis)
        np.savez_compressed(final/'power_r2_nearfield.npz',**arrays)
        incoming=final/'power_r2_kwave_unvalidated_input.mat'
        outgoing=final/'power_r2_kwave_unvalidated_output.mat'
        export_kwave_input(incoming,analysis,arrays,toolbox_path=args.toolbox,
            kwave_backend='matlab-gpu',unvalidated=True)
        # Release the LBM allocator cache before MATLAB starts on this GPU.
        import cupy as cp
        gc.collect();cp.get_default_memory_pool().free_all_blocks()
        quote=lambda p:"'"+Path(p).resolve().as_posix().replace("'","''")+"'"
        command=(f'maxNumCompThreads(4);addpath({quote(Path(__file__).parent)});'
                 f'kwave_handoff({quote(incoming)},{quote(outgoing)})')
        status('kwave_running',source_validation_passed=analysis['source_validation_passed'])
        subprocess.run([str(args.matlab),'-batch',command],cwd=ROOT,check=True)
        result=validate_propagation(loadmat(incoming,squeeze_me=True),loadmat(outgoing,squeeze_me=True))
        result['execution_scope']='User-requested unvalidated direct result; no boundary pulses performed.'
        atomic_json(final/'propagation_unvalidated.json',result)
        status('completed',propagation_validation_passed=result['propagation_validation_passed'],
            pressure_peak_at_45mm_Pa=result['pressure_peak_at_45mm_Pa'])
    except BaseException as exc:
        status('interrupted' if isinstance(exc,(KeyboardInterrupt,SystemExit)) else 'failed',
            error=dict(type=type(exc).__name__,message=str(exc)))
        raise


if __name__=='__main__':
    main()
