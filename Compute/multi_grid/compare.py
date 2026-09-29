"""Read saved paired experiments; keep new research gates separate from validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from core.unit_mapping import create_unit_mapping
from postproc.diagnostics import compare, fit_window
from reference.lim2013 import matched_gas
from simulation.output import dump


PAIRS=('pulse_up','pulse_down','step','thermal','power')


def load_case(path):
    config=yaml.safe_load((path/'effective.yaml').read_text(encoding='utf-8'))
    summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status']!='completed':
        raise ValueError(f'incomplete case: {path.name}')
    with np.load(path/'trajectory.npz') as saved:
        data={key:saved[key] for key in saved.files}
    return config,summary,manifest,data


def relative_l2(a,b):
    denominator=np.linalg.norm(b)
    if denominator==0:
        raise ValueError('zero reference signal; relative error is unresolved')
    return float(np.linalg.norm(a-b)/denominator)


def pulse_comparison(a,b,config):
    gas=matched_gas(create_unit_mapping(config)); p=config['run']['interface_pulse']
    z=a['probe_z_m']; source=p['center_fraction']*float(a['actual_length_m']); face=p['interface_m']
    direction=p['direction']; incident_row=0 if direction==1 else 1; transmitted_row=1-incident_row
    half=4*p['width_fraction']*float(a['actual_length_m'])/gas.c0_m_s
    ti=abs(z[incident_row]-source)/gas.c0_m_s
    tr=(abs(face-source)+abs(face-z[incident_row]))/gas.c0_m_s
    tt=abs(z[transmitted_row]-source)/gas.c0_m_s
    t=a['t_s']; mi=abs(t-ti)<half; mr=abs(t-tr)<half; mt=abs(t-tt)<half
    def waves(data):
        dp=data['pressure_Pa']-gas.p0_Pa
        rhocu=gas.rho0_kg_m3*gas.c0_m_s*data['velocity_m_s']
        return (dp+direction*rhocu)/2,(dp-direction*rhocu)/2
    af,ab=waves(a); bf,bb=waves(b)
    peak=lambda x:float(np.max(abs(x)))
    incident=peak(bf[mi,incident_row])
    reflection=peak(ab[mr,incident_row])/peak(af[mi,incident_row])
    extra=peak(ab[mr,incident_row]-bb[mr,incident_row])/incident
    transmit=abs(peak(af[mt,transmitted_row])/peak(bf[mt,transmitted_row])-1)
    l2=relative_l2(af[mt,transmitted_row],bf[mt,transmitted_row])
    peak_time=lambda x:float(t[mt][np.argmax(abs(x[mt,transmitted_row]))])
    gates=dict(reflection=reflection<.01,additional_reverse=extra<.005,
               transmission_peak=transmit<.01,transmission_unshifted_l2=l2<.01)
    return dict(reflection_ratio=reflection,reference_reflection_ratio=peak(bb[mr,incident_row])/incident,
        additional_reverse_peak_ratio=extra,transmission_peak_relative_error=transmit,
        transmission_unshifted_l2=l2,transmission_peak_time_difference_s=peak_time(af)-peak_time(bf),
        windows_s=dict(incident=[ti-half,ti+half],reflection=[tr-half,tr+half],transmission=[tt-half,tt+half]),
        gates=gates)


def step_comparison(a,b,config):
    # Common observation coordinates; interpolation never changes the gas state.
    assert np.all(np.diff(b['z_m'])>0)
    if not np.allclose(a['profile_t_s'],b['profile_t_s'],rtol=0,atol=1e-18):
        raise ValueError('profile times must match without time shifting')
    temperature=np.array([np.interp(a['z_m'],b['z_m'],row) for row in b['profile_temperature_K']])
    error=float(np.max(abs(a['profile_temperature_K']-temperature))/config['run']['amplitude_K'])
    wall_a=float(np.sum(a['wall_energy_J_m2'])); wall_b=float(np.sum(b['wall_energy_J_m2']))
    iface_a=float(np.sum(a['interface_exchange'][:,1,3])); iface_b=float(np.sum(b['interface_exchange'][:,1,3]))
    wall_error=abs(wall_a/wall_b-1); interface_error=abs(iface_a/iface_b-1)
    qa=a['profile_conductive_heat_flux_W_m2']
    qb=np.array([np.interp(a['z_m'],b['z_m'],row) for row in b['profile_conductive_heat_flux_W_m2']])
    moment_l2=relative_l2(qa,qb)
    moment_max=float(np.max(abs(qa-qb))/np.max(abs(qb)))
    difference=qa-qb; widths=a['cell_width_m']
    near=abs(a['z_m']-config['run']['thermal_step']['interface_m'])<3*np.max(widths)
    weighted_error=difference*difference*widths
    peak=np.unravel_index(np.argmax(abs(difference)),difference.shape)
    localization=dict(width_weighted_snapshot_l2=float(np.sqrt(np.sum(weighted_error)/np.sum(qb*qb*widths))),
        strip_halfwidth_m=float(3*np.max(widths)),
        squared_error_fraction_in_strip=(float(np.sum(weighted_error[:,near])/np.sum(weighted_error))
            if np.sum(weighted_error)>0 else None),
        max_error_outside_strip_W_m2=float(np.max(abs(difference[:,~near]))),
        peak_time_s=float(a['profile_t_s'][peak[0]]),peak_z_m=float(a['z_m'][peak[1]]),
        peak_two_level_W_m2=float(qa[peak]),peak_uniform_W_m2=float(qb[peak]),
        peak_relative_to_local_reference=(float(difference[peak]/qb[peak]) if qb[peak]!=0 else None))
    return dict(max_temperature_error_over_wall_increment=error,
        wall_energy_J_m2=[wall_a,wall_b],interface_energy_into_upper_J_m2=[iface_a,iface_b],
        wall_cumulative_relative_error=wall_error,interface_cumulative_relative_error=interface_error,
        conductive_moment_relative_l2=moment_l2,conductive_moment_normalized_max_error=moment_max,
        conductive_moment_localization=localization,
        gates=dict(temperature=error<.01,wall_energy=wall_error<.01,interface_energy=interface_error<.01,
                   conductive_moment=moment_max<.05))


def _aligned_harmonic_axes(a,b,power):
    fields={'t_s':'pressure_Pa','heat_time_s':'wall_flux_W_m2',
            'profile_t_s':'profile_temperature_K'}
    if power:
        fields['film_t_s']='film_temperature_K'
    for key,field in fields.items():
        ta,tb=np.asarray(a[key]),np.asarray(b[key])
        if (ta.ndim!=1 or tb.ndim!=1 or not np.array_equal(ta,tb)
                or not np.isfinite(ta).all() or np.any(np.diff(ta)<=0)):
            raise ValueError(f'{key} must contain identical finite increasing physical times')
        if any(len(data[field])!=len(data[key]) for data in (a,b)):
            raise ValueError(f'{field} must match its {key} axis')


def harmonic_comparison(a,b,config):
    run=config['run']; f=run['frequency_Hz']; end=run['cycles']; channels={}
    _aligned_harmonic_axes(a,b,run['mode']=='power')
    specs=[('wall_flux','heat_time_s','wall_flux_W_m2',None)]
    specs.extend((f'pressure_{j}','t_s','pressure_Pa',j) for j in range(a['pressure_Pa'].shape[1]))
    if run['mode']=='power':
        specs.append(('film_temperature','film_t_s','film_temperature_K',None))
    for name,time_key,key,index in specs:
        av=a[key] if index is None else a[key][:,index]
        bv=b[key] if index is None else b[key][:,index]
        ah,_=fit_window(a[time_key],av,f,end-2,end,harmonics=3,detrend_order=1)
        bh,_=fit_window(b[time_key],bv,f,end-2,end,harmonics=3,detrend_order=1)
        item=compare(ah,bh)
        item['passed']=abs(item['amplitude_error_percent'])<1 and abs(item['phase_error_deg'])<1
        item['cold_start_max_absolute_difference']=float(np.max(abs(av-bv)))
        item['cold_start_rms_difference']=float(np.sqrt(np.mean((av-bv)**2)))
        channels[name]=item
    # Mean temperatures use common saved right endpoints in (end-2,end].
    # These are endpoint sample means, not time integrals or harmonic DC terms.
    def right_endpoints(time_key):
        cycles=np.asarray(a[time_key])*f
        mask=(cycles>end-2+1e-10)&(cycles<=end+1e-10)
        if not np.any(mask):
            raise ValueError(f'{time_key} has no common endpoints in the last two cycles')
        return mask
    profile_mask=right_endpoints('profile_t_s')
    mean_a=np.mean(a['profile_temperature_K'][profile_mask],axis=0)
    mean_b=np.mean(b['profile_temperature_K'][profile_mask],axis=0)
    assert np.all(np.diff(b['z_m'])>0)
    mean_difference=mean_a-np.interp(a['z_m'],b['z_m'],mean_b)
    gas=matched_gas(create_unit_mapping(config))
    def volume_mean_rise(data,mean):
        widths=np.asarray(data['cell_width_m'])
        if widths.shape!=mean.shape or not np.isfinite(widths).all() or np.any(widths<=0):
            raise ValueError('cell_width_m must contain a positive width for every temperature cell')
        return float(np.average(mean-gas.T0_K,weights=widths))
    def values_and_difference(x,y):
        return dict(two_level=float(x),uniform=float(y),two_level_minus_uniform=float(x-y))
    profile_times=np.asarray(a['profile_t_s'])[profile_mask]
    window=dict(start_s=float((end-2)/f),end_s=float(end/f),endpoint_rule='start < t <= end',
        averaging='equal weight over common saved endpoints; gas volume weighted by physical cell width',
        profile_samples=int(len(profile_times)),profile_first_s=float(profile_times[0]),
        profile_last_s=float(profile_times[-1]))
    extra=dict(mean_temperature_window=window,gas_reference_temperature_K=gas.T0_K,
        gas_volume_mean_temperature_rise_last_two_cycles_K=values_and_difference(
            volume_mean_rise(a,mean_a),volume_mean_rise(b,mean_b)))
    if run['mode']=='power':
        mask=right_endpoints('film_t_s'); film_times=np.asarray(a['film_t_s'])[mask]
        means=[float(np.mean(data['film_temperature_K'][mask])) for data in (a,b)]
        reference=run['film']['T_ref_K']
        extra['film_mean_temperature_last_two_cycles_K']=means
        extra['film_reference_temperature_K']=reference
        extra['film_mean_temperature_rise_last_two_cycles_K']=values_and_difference(
            means[0]-reference,means[1]-reference)
        window.update(film_samples=int(len(film_times)),film_first_s=float(film_times[0]),
                      film_last_s=float(film_times[-1]))
    return dict(channels=channels,gates={key:item['passed'] for key,item in channels.items()},
        mean_gas_temperature_max_difference_K=float(np.max(abs(mean_difference))),**extra)


def compare_batch(batch):
    batch=Path(batch); results={}
    for pair in PAIRS:
        paths=[batch/'raw'/f'{pair}_{kind}' for kind in ('two_level','uniform')]
        if not all(path.exists() for path in paths):
            continue
        ca,sa,ma,a=load_case(paths[0]); cb,sb,mb,b=load_case(paths[1])
        if ma['git_commit']!=mb['git_commit'] or ma['source_id']!=mb['source_id']:
            raise ValueError('paired results must use the same commit and public source')
        if not np.array_equal(a['t_s'],b['t_s']) or not np.array_equal(a['probe_z_m'],b['probe_z_m']):
            raise ValueError('paired observations must share physical times and positions')
        item=(pulse_comparison(a,b,ca) if pair.startswith('pulse') else
              step_comparison(a,b,ca) if pair=='step' else harmonic_comparison(a,b,ca))
        item.update(comparison_passed=bool(all(item['gates'].values())),
            validation_passed=dict(two_level=sa['validation_passed'],uniform=sb['validation_passed']),
            status=dict(two_level=sa['status'],uniform=sb['status']),git_commit=ma['git_commit'],
            performance=dict(cell_updates=[int(a['cell_updates']),int(b['cell_updates'])],
                ideal_update_speedup=float(b['cell_updates']/a['cell_updates']),
                warmup_s=[float(a['warmup_s']),float(b['warmup_s'])],
                advance_s=[float(a['advance_s']),float(b['advance_s'])],
                total_s=[ma['elapsed_s'],mb['elapsed_s']],
                advance_speedup=float(b['advance_s']/a['advance_s']),
                total_speedup=float(mb['elapsed_s']/ma['elapsed_s'])))
        if pair=='power':
            item['periodic_response_stable']=[sa['periodic_response_stable'],sb['periodic_response_stable']]
        results[pair]=item
    return dict(scope='nx=1, small amplitude, ratio 2; paired accuracy gates independent of validation_passed',
        phase_convention='exp(+i omega t); no time shifts or phase compensation',
        comparison_passed=bool(len(results)==5 and all(item['comparison_passed'] for item in results.values())),
        pairs=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,required=True)
    parser.add_argument('--step-refinement',type=Path,help='Optional saved finer step pair used to diagnose the interface error')
    args=parser.parse_args()
    result=compare_batch(args.batch)
    if args.step_refinement:
        ca,sa,ma,a=load_case(args.step_refinement/'raw/step_two_level')
        cb,sb,mb,b=load_case(args.step_refinement/'raw/step_uniform')
        diagnostic=step_comparison(a,b,ca)
        diagnostic.update(refine=ca['run']['refine'],git_commit=ma['git_commit'],
            validation_passed=[sa['validation_passed'],sb['validation_passed']],
            cases=[str(args.step_refinement/'raw'/name) for name in ('step_two_level','step_uniform')])
        result['step_refinement_diagnostic']=diagnostic
    final=args.batch/'final'; final.mkdir(exist_ok=True)
    dump(final/'comparison.json',result)
    for pair,item in result['pairs'].items():
        print(pair,item['comparison_passed'],item['gates'])


if __name__=='__main__':
    main()
