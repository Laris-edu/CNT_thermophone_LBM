"""Run the nearfield/propagation chain, with an explicit saved-source diagnostic mode.

This is immediate case orchestration, not a scheduler. Existing cases and final
outputs are never overwritten. A failed state, nearfield, or propagation gate
stops the normal chain without changing tolerances or launching another scan.
Diagnostic propagation preserves failed checks and uses separate output names.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml
from scipy.io import loadmat

from Compute.finite_strip.analyze import main as analyze_main,analyze_saved,export_kwave_input
from Compute.finite_strip.run import load_cases,validate_boundary_evidence
from farfield.kirchhoff_2d import kirchhoff_2d_frequency,dpdn_from_velocity,GREEN_OUTGOING
from postproc.multiharmonic_fit import fit_multiharmonic
from simulation.finite_strip import run_strip_case
from simulation.output import dump
from simulation.study import ROOT,valid_name


def validate_propagation(source,result):
    """Compare actual k-Wave coordinates/phasors with independent Kirchhoff."""
    backend_metadata = {}
    if 'kwave_backend_requested' in source:
        requested = str(source['kwave_backend_requested'])
        if (requested not in ('cpu', 'matlab-gpu')
                or str(result.get('kwave_backend_requested')) != requested
                or str(result.get('kwave_backend_effective')) != requested):
            raise ValueError('k-Wave backend differs from the explicit requested backend')
        provenance = json.loads(str(result.get('gpu_provenance_json', '{}')))
        if not isinstance(provenance, dict) or provenance.get('enabled') is not (requested == 'matlab-gpu'):
            raise ValueError('k-Wave backend GPU provenance is missing or inconsistent')
        backend_metadata = dict(kwave_backend_requested=requested,
            kwave_backend_effective=requested, gpu_provenance=provenance)
    diagnostic=bool(source.get('diagnostic_mode',0)==1)
    source_passed=bool(source['source_validation_passed']==1)
    if diagnostic and (result.get('diagnostic_mode')!=1
                       or result.get('source_validation_passed')!=int(source_passed)
                       or result.get('source_validation_bypassed')!=int(not source_passed)):
        raise ValueError('k-Wave diagnostic/source validation metadata differs from input')
    x,y=(np.atleast_1d(result['observer_'+a+'_m']).ravel() for a in ('x','y'))
    for a,actual in zip(('x','y'),(x,y)):
        if not np.allclose(actual,np.atleast_1d(source['expected_grid_observer_'+a+'_m']).ravel(),
                           rtol=0,atol=1e-12):
            raise ValueError('k-Wave observer coordinates differ from the exported grid')
    roles=np.asarray(source['observer_role']).ravel().astype(str)
    exterior=roles!='interior_cancel'; interior=~exterior
    if not exterior.any() or not interior.any() or not np.any(roles=='farfield_45mm'):
        raise ValueError('nearfield, 45 mm, and interior validation observers are required')
    frequency=float(source['frequency_Hz']); omega=2*np.pi*frequency
    rho,c=float(source['rho0_kg_m3']),float(source['c0_m_s'])
    predicted=np.asarray(result['observer_p_hat_Pa'],dtype=complex).ravel()
    reference=kirchhoff_2d_frequency(surface_x_m=source['x_m'],surface_y_m=source['y_m'],
        normal_x=source['normal_x'],normal_y=source['normal_y'],ds_m=source['ds_m'],
        p_hat_Pa=source['p_hat_Pa'],dpdn_hat_Pa_m=dpdn_from_velocity(source['un_hat_m_s'],
            omega_rad_s=omega,rho0_kg_m3=rho),observer_x_m=x,observer_y_m=y,
        omega_rad_s=omega,c0_m_s=c,green_convention=GREEN_OUTGOING)
    if np.any(abs(reference[exterior])<1e-12):
        raise ValueError('unresolved exterior propagation reference')
    ratio=predicted[exterior]/reference[exterior]
    amp=float(np.max(abs(abs(ratio)-1)))
    phase=float(np.max(abs(np.angle(ratio,deg=True))))
    leakage=float(np.max(abs(predicted[interior]))/np.max(abs(reference[exterior])))
    t=np.atleast_1d(result['time_s']); signal=np.asarray(result['observer_pressure_Pa'])
    if signal.shape!=(len(x),len(t)):
        raise ValueError('propagation time series dimensions do not match observers')
    if t.ndim!=1 or len(t)<32 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('propagation time must be finite and strictly increasing')
    cadence=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),cadence,rtol=1e-7,atol=1e-13):
        raise ValueError('propagation time has missing or irregular samples')
    # Endpoint samples of two periods span 2T-dt. Having enough points for
    # two least-squares fits alone does not establish complete coverage.
    if t[-1]-t[0]+cadence<2/frequency-1e-13:
        raise ValueError('propagation time must cover two complete periods')
    # Independent two consecutive one-period fits catch residual ramp/PML
    # transients; keep the absolute source clock, with no phase correction.
    fitted=[]
    for low,high in ((t[-1]-2/frequency,t[-1]-1/frequency),(t[-1]-1/frequency,t[-1])):
        mask=(t>low+1e-13)&(t<=high+1e-13)
        if mask.sum()<16:
            raise ValueError('two full propagated periods must be recorded')
        fitted.append(np.array([fit_multiharmonic(t[mask],v[mask],omega,
            n_harmonics=3,detrend_order=1).harmonic(1) for v in signal]))
    if np.any(abs(fitted[1][exterior])<1e-12):
        raise ValueError('unresolved propagated periodic response')
    drift=fitted[0][exterior]/fitted[1][exterior]
    window_amp=float(np.max(abs(abs(drift)-1)))
    window_phase=float(np.max(abs(np.angle(drift,deg=True))))
    signal_ratio=(fitted[0][exterior]+fitted[1][exterior])/(2*predicted[exterior])
    signal_amp=float(np.max(abs(abs(signal_ratio)-1)))
    signal_phase=float(np.max(abs(np.angle(signal_ratio,deg=True))))
    signal_leak=float(np.max(abs(np.asarray(fitted)[:,interior]))/np.max(abs(reference[exterior])))
    checks=dict(source_validation=bool(source['source_validation_passed']==1),
        finite=bool(np.isfinite(predicted).all() and np.isfinite(signal).all()),
        exterior_amplitude=amp<=.01,exterior_phase=phase<=1.,
        interior_cancellation=leakage<=.01,
        recorded_period_amplitude=window_amp<=.01,recorded_period_phase=window_phase<=1.,
        timeseries_matches_reported_amplitude=signal_amp<=.01,
        timeseries_matches_reported_phase=signal_phase<=1.,timeseries_interior_cancellation=signal_leak<=.01)
    far=np.flatnonzero(roles=='farfield_45mm')
    return dict(status='completed',source_validation_passed=checks['source_validation'],
        **backend_metadata,
        diagnostic_mode=diagnostic,source_validation_bypassed=bool(diagnostic and not source_passed),
        propagation_validation_passed=all(checks.values()),checks=checks,
        max_amplitude_error_relative=amp,max_phase_error_deg=phase,interior_leakage_relative=leakage,
        period_amplitude_change_relative=window_amp,period_phase_change_deg=window_phase,
        timeseries_reported_amplitude_difference=signal_amp,timeseries_reported_phase_difference_deg=signal_phase,
        timeseries_interior_leakage_relative=signal_leak,
        observer_x_m=x.tolist(),observer_y_m=y.tolist(),observer_role=roles.tolist(),
        pressure_hat_real_Pa=predicted.real.tolist(),pressure_hat_imag_Pa=predicted.imag.tolist(),
        kirchhoff_hat_real_Pa=reference.real.tolist(),kirchhoff_hat_imag_Pa=reference.imag.tolist(),
        pressure_peak_at_45mm_Pa=abs(predicted[far]).tolist(),
        phase_at_45mm_deg=np.angle(predicted[far],deg=True).tolist(),
        tolerances=dict(amplitude_relative=.01,phase_deg=1.,interior_relative=.01),
        scope='2D infinite strip, homogeneous lossless k-Wave propagation to 45 mm; '
              'independent Kirchhoff numerical comparison, no direct LBM or experimental field at 45 mm')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',required=True)
    parser.add_argument('--study',type=Path,default=Path(__file__).with_name('study.yaml'))
    parser.add_argument('--stage',choices=('all','propagate'),default='all')
    parser.add_argument('--matlab',type=Path,default=Path('E:/Matlab/bin/matlab.exe'))
    parser.add_argument('--toolbox',type=Path,default=ROOT/'k-wave-toolbox-version-1.4.1')
    parser.add_argument('--kwave-backend',choices=('cpu','matlab-gpu'),default='cpu',
        help='Use CPU double precision or MATLAB gpuArray double precision; no fallback.')
    parser.add_argument('--diagnostic',action='store_true',
        help='Propagate a saved stable source despite its direct-pressure comparison failure; retain failed validation.')
    parser.add_argument('--matlab-threads',type=int,
        help='Requested MATLAB computational threads; does not change the numerical grid or source.')
    args=parser.parse_args(argv)
    if args.diagnostic and args.stage!='propagate':
        parser.error('--diagnostic requires --stage propagate; it cannot start LBM')
    if args.matlab_threads is not None and args.matlab_threads<1:
        parser.error('--matlab-threads must be positive')
    root=args.study.resolve().parent/valid_name(args.batch)
    case=root/'raw/power_r2'
    if args.stage=='all':
        _,cases=load_cases(args.study.resolve())
        config=cases['power_r2']
        validate_boundary_evidence(root,config)
    else:
        # A downloaded run is defined by its saved configuration and runtime,
        # not by the current study or this machine's public-source identity.
        for name in ('effective.yaml','manifest.json','trajectory.npz','summary.json'):
            if not (case/name).is_file():
                raise FileNotFoundError(case/name)
        config=yaml.safe_load((case/'effective.yaml').read_text(encoding='utf-8'))
        manifest=json.loads((case/'manifest.json').read_text(encoding='utf-8'))
        summary=json.loads((case/'summary.json').read_text(encoding='utf-8'))
        if manifest.get('status')!='completed' or summary.get('status')!='completed':
            raise ValueError('propagation requires a completed saved nearfield run')
        if not manifest.get('source_id'):
            raise ValueError('saved nearfield runtime source identity is missing')
        if manifest.get('source_clean') is not True or not manifest.get('git_commit'):
            raise ValueError('propagation requires a committed clean simulation runtime')
        validate_boundary_evidence(root,config,source_id=manifest['source_id'])
    # Fail dependency and overwrite checks before the many-hour LBM run.
    if not args.matlab.is_file() or not (args.toolbox/'kspaceFirstOrder2D.m').is_file():
        raise FileNotFoundError('MATLAB executable or k-Wave entry point is missing')
    prefix='power_r2_kwave_diagnostic' if args.diagnostic else 'power_r2_kwave'
    incoming=root/'final'/f'{prefix}_input.mat'
    outgoing=root/'final'/f'{prefix}_output.mat'
    target=root/'final'/('propagation_diagnostic.json' if args.diagnostic else 'propagation_validation.json')
    outputs=[incoming,outgoing,target]
    if not args.diagnostic:
        outputs += [root/'final'/name for name in ('power_r2_nearfield.json','power_r2_nearfield.npz')]
    for path in ([case] if args.stage=='all' else [])+outputs:
        if path.exists():
            raise FileExistsError(path)
    quote=lambda path:"'"+Path(path).resolve().as_posix().replace("'","''")+"'"
    thread_setting='' if args.matlab_threads is None else f'maxNumCompThreads({args.matlab_threads});'
    gpu_probe=("assert(license('test','Distrib_Computing_Toolbox'),'MATLAB Parallel Computing Toolbox is required');"
               "gpuDevice;" if args.kwave_backend=='matlab-gpu' else '')
    probe=thread_setting+gpu_probe+f"addpath({quote(args.toolbox)});assert(~isempty(which('kspaceFirstOrder2D')));disp('MATLAB/k-Wave preflight passed')"
    if args.stage=='all':
        subprocess.run([str(args.matlab),'-batch',probe],cwd=ROOT,check=True)
        print(json.dumps(dict(stage='power',output=str(case),exists=case.exists(),
            nx=config['numerics']['nx'],ny=config['numerics']['ny'],steps=config['strip']['steps'],
            dt_s=config['lattice']['dt_s']),indent=2),flush=True)
        run_strip_case(config,case)
    if args.diagnostic:
        # Re-read the immutable saved run. Existing failed analysis remains intact.
        analysis,arrays=analyze_saved(case)
        export_kwave_input(incoming,analysis,arrays,toolbox_path=args.toolbox,diagnostic=True,
                           kwave_backend=args.kwave_backend)
        print(json.dumps(dict(stage='diagnostic_propagation',
            source_validation_passed=analysis['source_validation_passed'],
            failed_source_checks=[key for key,ok in analysis['checks'].items() if not ok],
            input=str(incoming),output=str(outgoing)),indent=2),flush=True)
    else:
        status=analyze_main(['--batch',str(root),'--case','power_r2','--export-kwave','--toolbox',str(args.toolbox),
                             '--kwave-backend',args.kwave_backend])
        if status:
            print('Nearfield/source gate failed; diagnostics saved and propagation not started.',flush=True)
            return status
    if args.stage=='propagate':
        subprocess.run([str(args.matlab),'-batch',probe],cwd=ROOT,check=True)
    if outgoing.exists():
        raise FileExistsError(outgoing)
    command=thread_setting+f'addpath({quote(Path(__file__).parent)});kwave_handoff({quote(incoming)},{quote(outgoing)})'
    subprocess.run([str(args.matlab),'-batch',command],cwd=ROOT,check=True)
    result=validate_propagation(loadmat(incoming,squeeze_me=True),loadmat(outgoing,squeeze_me=True))
    result['input_sha256']=hashlib.sha256(incoming.read_bytes()).hexdigest()
    result['output_sha256']=hashlib.sha256(outgoing.read_bytes()).hexdigest()
    result['adapter_sha256']=hashlib.sha256(Path(__file__).with_name('kwave_handoff.m').read_bytes()).hexdigest()
    result['matlab_threads_requested']=args.matlab_threads
    if args.diagnostic:
        result['execution_scope']='User-requested diagnostic propagation; failed source/interior checks remain failed.'
    if target.exists():
        raise FileExistsError(target)
    dump(target,result)
    print(json.dumps(result,indent=2),flush=True)
    return 0 if args.diagnostic or result['propagation_validation_passed'] else 3


if __name__=='__main__':
    raise SystemExit(main())
