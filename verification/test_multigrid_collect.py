"""Synchronized observations retain independent fine film intervals."""
from types import SimpleNamespace

import numpy as np
import pytest
from copy import deepcopy
from dataclasses import replace

from postproc.power import _channels, energy_balances
from simulation.planar import make_solver
from simulation.study import DEFAULTS, POWER_DEFAULTS


def test_conserved_observation_interpolates_momentum_and_energy():
    from simulation.multigrid import observe_conserved
    state=SimpleNamespace(rho=np.array([[1.],[3.]]),
        u=np.array([[[0.,0.]],[[0.,2.]]]),
        E_tot=np.array([[5.],[21.]]),gamma=1.4)
    observed=observe_conserved(state,np.array([0.,2.]),np.array([1.]))
    assert observed['rho'][0]==2.
    assert observed['u'][0,1]==1.5
    assert observed['E_tot'][0]==13.
    assert observed['p'][0]==pytest.approx(.4*(13.-.5*2.*1.5**2))
    with pytest.raises(ValueError,match='inside'):
        observe_conserved(state,np.array([0.,2.]),np.array([3.]))


def test_power_dual_axes_preserve_fine_film_residuals():
    data=dict(t_s=np.array([2.,4.]),film_t_s=np.arange(1.,5.),
        heat_time_s=np.arange(.5,4.),film_temperature_K=np.array([301.,302.,303.,304.]),
        wall_flux_W_m2=np.ones(4),pressure_Pa=np.zeros((2,1)),velocity_m_s=np.zeros((2,1)),
        gas_energy_J_m2=np.array([2.,4.]),initial_gas_energy_J_m2=0.,
        initial_film_temperature_K=300.,input_energy_J_m2=np.full(4,3.),
        leak_energy_J_m2=np.zeros(4),wall_energy_J_m2=np.ones(4),
        sync_fine_step=np.array([2,4]),audit=np.array([[0.,0.,2.,0.,0.,0.]]*2))
    film=dict(C_A_J_m2K=1.,gas_flux_factor=2.)
    assert len(next(_channels(data))[1])==4
    balances=energy_balances(data,film)
    assert balances['film_step'].shape==(4,)
    np.testing.assert_array_equal(balances['film_step'],0.)
    np.testing.assert_array_equal(balances['total_step'],0.)
    data['film_temperature_K'][0]+=.1
    changed=energy_balances(data,film)
    assert max(abs(changed['film_step']))==pytest.approx(.1)
    np.testing.assert_allclose(changed['total_step'],0.,atol=1e-12)


def _step_run(solver,steps=4):
    return dict(mode='thermal_step',frequency_Hz=50000.,amplitude_K=.1,
        thermal_step=dict(duration_s=steps*solver.mapping.lattice.dt_s,
                          interface_m=8*solver.mapping.lattice.dx_m),
        probes_m=[4*solver.mapping.lattice.dx_m,12*solver.mapping.lattice.dx_m],
        sampling=dict(sync_fine_steps=2,probe_interpolation='linear',profile_samples_per_cycle=32))


def test_uniform_collector_two_fine_steps_and_actual_midpoints(tmp_path):
    from simulation.multigrid import collect_synchronized
    solver=make_solver(refine=1,length=50e-6)
    dt=solver.mapping.lattice.dt_s
    data=collect_synchronized(solver,tmp_path,_step_run(solver))
    np.testing.assert_allclose(data['t_s'],[2*dt,4*dt])
    np.testing.assert_allclose(data['heat_time_s'],(np.arange(4)+.5)*dt)
    np.testing.assert_allclose(data['wall_energy_J_m2'],data['wall_flux_W_m2']*dt)
    np.testing.assert_array_equal(data['sync_fine_step'],[2,4])
    assert data['run_completed']
    assert data['profile_t_s'][0]==0.
    assert data['profile_t_s'][-1]==data['t_s'][-1]


def test_failed_second_uniform_substep_discards_partial_sync_samples(tmp_path,monkeypatch):
    from simulation.multigrid import collect_synchronized
    solver=make_solver(refine=1,length=50e-6)
    original=solver.step_temperature
    def failing(temperature):
        if solver.t_lu==3:
            raise FloatingPointError('injected failure after a fine substep')
        return original(temperature)
    monkeypatch.setattr(solver,'step_temperature',failing)
    with pytest.raises(FloatingPointError,match='injected'):
        collect_synchronized(solver,tmp_path,_step_run(solver))
    with np.load(tmp_path/'trajectory.npz') as data:
        assert len(data['t_s'])==1
        assert len(data['heat_time_s'])==2
        assert not data['run_completed']
        assert data['sync_fine_step'][-1]==2


@pytest.mark.parametrize('direction',[1,-1])
def test_interface_pulse_uses_interface_return_not_outlet(direction):
    from postproc.interface import analyze_interface_pulse
    from reference.lim2013 import matched_gas
    solver=make_solver(); mapping=solver.mapping; gas=matched_gas(mapping)
    H=1e-3; interface=H/2; probes=np.array([.375,.625])*H
    source=(.25 if direction==1 else .75)*H; sigma=.025*H
    t=np.arange(0,.5*H/gas.c0_m_s,1e-9)
    outgoing=np.zeros((len(t),2)); incoming=np.zeros_like(outgoing)
    for j,z in enumerate(probes):
        primary=np.exp(-.5*((t-abs(z-source)/gas.c0_m_s)/(sigma/gas.c0_m_s))**2)
        (outgoing if direction==1 else incoming)[:,j]=primary
    j=0 if direction==1 else 1
    reflected_time=abs(2*interface-source-probes[j])/gas.c0_m_s
    (incoming if direction==1 else outgoing)[:,j]=.004*np.exp(-.5*((t-reflected_time)/(sigma/gas.c0_m_s))**2)
    data=dict(t_s=t,pressure_Pa=gas.p0_Pa+outgoing+incoming,
        velocity_m_s=(outgoing-incoming)/(gas.rho0_kg_m3*gas.c0_m_s),
        probe_z_m=probes,actual_length_m=H,audit=np.zeros((len(t),6)),
        mass_lu=np.zeros(len(t)),initial_mass_lu=0.,gas_energy_J_m2=np.zeros(len(t)),
        initial_gas_energy_J_m2=0.,min_f=1.,min_g=1.)
    run=dict(interface_pulse=dict(direction=direction,center_fraction=source/H,
        width_fraction=sigma/H,interface_m=interface),analysis=DEFAULTS['analysis'])
    result=analyze_interface_pulse(data,mapping,run)
    assert result['reflection_ratio']==pytest.approx(.004,rel=2e-4)
    assert result['pass_reflection']
    assert result['reflected_time_s']==pytest.approx(reflected_time)


@pytest.mark.parametrize('mode',['thermal_step','power'])
def test_real_two_level_collector_has_independent_storage_and_fine_wall_intervals(tmp_path,mode):
    from core.two_level_planar_solver import TwoLevelPlanarOpenSolver
    from simulation.multigrid import collect_synchronized
    config=deepcopy(make_solver(refine=1,length=24*2.61175e-6).config)
    config['grid']=dict(type='two_level',ratio=2,interface_m=8*config['lattice']['dx_m'])
    solver=TwoLevelPlanarOpenSolver(config); dt=solver.mapping.lattice.dt_s
    if mode=='power':
        run=deepcopy(POWER_DEFAULTS)
        run.update(cycles=4*dt*run['frequency_Hz'],probes_m=[4*solver.mapping.lattice.dx_m,16*solver.mapping.lattice.dx_m])
        run['stopping']['stop_when_stable']=False
        run['sampling'].update(sync_fine_steps=2,probe_interpolation='linear')
    else:
        run=_step_run(solver)
    data=collect_synchronized(solver,tmp_path,run)
    assert len(data['t_s'])==2
    assert len(data['heat_time_s'])==4
    assert data['interface_exchange_available']
    assert data['block_conserved'].shape==(2,2,4)
    np.testing.assert_allclose(np.sum(data['block_conserved'][:,:,3],axis=1),data['gas_energy_J_m2'],rtol=1e-14)
    np.testing.assert_allclose(data['audit'][:,2],data['wall_energy_J_m2'].reshape(2,2).sum(axis=1),rtol=0,atol=1e-15)
    np.testing.assert_allclose(data['interface_exchange'][:,:2,:].sum(axis=1),0,rtol=0,atol=1e-13)
    if mode=='power':
        assert len(data['film_t_s'])==4
        balances=energy_balances(data,run['film'])
        assert np.max(abs(balances['film_step']))<1e-10
        assert np.max(abs(balances['gas_step']))<1e-10


def test_bad_film_endpoint_stops_before_committing_the_next_sync(tmp_path,monkeypatch):
    import simulation.multigrid as collect
    solver=make_solver(refine=1,length=50e-6); dt=solver.mapping.lattice.dt_s
    run=deepcopy(POWER_DEFAULTS)
    run.update(cycles=4*dt*run['frequency_Hz'],probes_m=[4*solver.mapping.lattice.dx_m,12*solver.mapping.lattice.dx_m])
    run['stopping']['stop_when_stable']=False
    run['sampling'].update(sync_fine_steps=2,probe_interpolation='linear')
    original=collect.step_open_film
    def bad_endpoint(**kwargs):
        should_corrupt=kwargs['solver'].t_lu==2
        result=original(**kwargs)
        return replace(result,T_s_K=result.T_s_K+.001) if should_corrupt else result
    monkeypatch.setattr(collect,'step_open_film',bad_endpoint)
    with pytest.raises(FloatingPointError,match='film/total'):
        collect.collect_synchronized(solver,tmp_path,run)
    with np.load(tmp_path/'trajectory.npz') as data:
        assert len(data['t_s'])==1
        assert len(data['film_t_s'])==2
        assert not data['run_completed']


def test_final_profile_failure_keeps_complete_profile_arrays_and_failed_status(tmp_path,monkeypatch):
    from simulation.multigrid import collect_synchronized
    solver=make_solver(refine=1,length=50e-6)
    original=solver.get_heat_flux_lu; calls=0
    def fail_final_profile():
        nonlocal calls
        calls+=1
        if calls==3:
            raise RuntimeError('injected final profile failure')
        return original()
    monkeypatch.setattr(solver,'get_heat_flux_lu',fail_final_profile)
    with pytest.raises(RuntimeError,match='final profile'):
        collect_synchronized(solver,tmp_path,_step_run(solver,steps=2))
    with np.load(tmp_path/'trajectory.npz') as data:
        assert not data['run_completed']
        assert len(data['profile_t_s'])==len(data['profile_temperature_K'])==1
        assert len(data['t_s'])==1
