from pathlib import Path
import numpy as np
import pytest
from config_io import load_config
from core.unit_mapping import create_unit_mapping, local_transport_coefficients
from core.lattice import make_lattice
from core.equilibrium import equilibrium_fg
from core.macroscopic import recover_macro
from core.collision_smrt import collide_fg
from core.kinetic_moments import extract_moments, recover_moment_macro, collide_moments
from core.streaming import pull_stream_fg

ROOT = Path(__file__).resolve().parents[1]

def state(model='sutherland'):
    cfg = load_config(ROOT / 'configs/gas_air_10k_d2q37_qmatched_local_transport.yaml')
    cfg['transport']['model'] = model
    mapping = create_unit_mapping(cfg)
    rng = np.random.default_rng(491)
    rho = 1 + .03*rng.normal(size=(8, 5))
    u = .025*rng.normal(size=(8, 5, 2)) + [.06, -.03]
    theta = mapping.theta_ref_lu*(1+.02*rng.normal(size=rho.shape))
    f,g = equilibrium_fg(rho,u,theta,mapping.lattice.S,make_lattice())
    f *= 1 + .01*rng.uniform(-1,1,f.shape)
    g *= 1 + .01*rng.uniform(-1,1,g.shape)
    assert min(f.min(),g.min()) > 0
    return mapping,f,g

@pytest.mark.parametrize('model',['constant','sutherland'])
@pytest.mark.parametrize('backend',['numpy','numba'])
def test_nine_moments_suffice_for_collision(model,backend):
    mapping,f,g=state(model)
    z=extract_moments(f,g)
    macro=recover_moment_macro(z,mapping)
    original=recover_macro(f,g)
    for name in ('rho','u','theta','p','K_tr','G_int','E_tot'):
        np.testing.assert_allclose(getattr(macro,name),getattr(original,name),rtol=2e-14,atol=2e-14)
    actual=collide_moments(z,mapping,backend=backend)
    expected=collide_fg(f,g,mapping,backend=backend)
    for a,b in zip(actual,expected):
        np.testing.assert_allclose(a,b,rtol=2e-13,atol=2e-14)
    post=extract_moments(*actual)
    np.testing.assert_allclose(post[...,:4],z[...,:4],rtol=2e-14,atol=2e-14)
    # Independently verify trace relaxation and raw total energy.
    tau=local_transport_coefficients(mapping,macro.rho,macro.theta).tau22
    trace=z[...,4]+z[...,6]-macro.rho*np.sum(macro.u**2,axis=-1)-2*macro.p
    trace_post=post[...,4]+post[...,6]-macro.rho*np.sum(macro.u**2,axis=-1)-2*macro.p
    assert np.max(np.abs(trace)) > 1e-5
    np.testing.assert_allclose(trace_post,mapping.collision.trace_bulk_scale*(1-1/tau)*trace,atol=3e-14)
    c=make_lattice().c
    raw_energy=.5*np.sum(actual[0]*np.sum(c*c,axis=1),axis=-1)+np.sum(actual[1],axis=-1)
    np.testing.assert_allclose(raw_energy,z[...,3],atol=2e-14)

def test_kernel_of_nine_moments_is_erased():
    mapping,f,g=state()
    eye=np.eye(74)
    B=extract_moments(eye[:,:37],eye[:,37:]).T
    assert np.linalg.matrix_rank(B)==9
    rng=np.random.default_rng(72)
    h=rng.normal(size=74)
    h-=B.T@np.linalg.solve(B@B.T,B@h)
    h*=.1*min(f.min(),g.min())/np.max(np.abs(h))
    np.testing.assert_allclose(B@h,0,atol=1e-18)
    for backend in ('numpy','numba'):
        a=collide_fg(f,g,mapping,backend=backend)
        b=collide_fg(f+h[:37],g+h[37:],mapping,backend=backend)
        for ai,bi in zip(a,b):
            np.testing.assert_allclose(ai,bi,atol=2e-14,rtol=2e-13)

@pytest.mark.parametrize('channel', ['heat_flux', 'stress'])
def test_four_conserved_moments_do_not_close_streaming(channel):
    mapping,f,g=state()
    # Vary the internal-energy flux without changing any local conserved moment.
    c=make_lattice().c
    wave=np.sin(2*np.pi*np.arange(f.shape[0])/f.shape[0])[:,None,None]
    if channel == 'heat_flux':
        df=0
        dg=1e-5*wave*make_lattice().w*c[:,1]
    else:
        df=1e-5*wave*make_lattice().w*(c[:,0]**2-c[:,1]**2)
        dg=0
    assert min((f+df).min(), (g+dg).min()) > 0
    np.testing.assert_allclose(extract_moments(f,g)[...,:4],extract_moments(f+df,g+dg)[...,:4],atol=2e-14)
    a=extract_moments(*pull_stream_fg(*collide_fg(f,g,mapping)))
    b=extract_moments(*pull_stream_fg(*collide_fg(f+df,g+dg,mapping)))
    assert np.max(np.abs(a[...,:4]-b[...,:4]))>1e-8


def test_moment_heat_flux_matches_population_readout():
    from core.kinetic_moments import heat_flux_moments
    from core.macroscopic import heat_flux_lu
    for model in ('constant', 'sutherland'):
        mapping,f,g=state(model)
        np.testing.assert_allclose(heat_flux_moments(extract_moments(f,g),mapping),
                                   heat_flux_lu(f,g,mapping=mapping),rtol=2e-12,atol=2e-14)


@pytest.mark.parametrize('model', ['constant', 'sutherland'])
def test_pressure_entropy_coordinates_roundtrip(model):
    from core.kinetic_moments import pressure_entropy_coordinates, moments_from_pressure_entropy, heat_flux_moments
    from core.macroscopic import central_energy_flux_lu
    mapping,f,g=state(model)
    z=extract_moments(f,g)
    coords=pressure_entropy_coordinates(z,mapping)
    restored=moments_from_pressure_entropy(coords,mapping)
    np.testing.assert_allclose(restored,z,rtol=2e-13,atol=2e-14)
    a,b=recover_moment_macro(z,mapping),recover_moment_macro(restored,mapping)
    for name in ('rho','p','u','theta','E_tot','K_tr','G_int'):
        np.testing.assert_allclose(getattr(a,name),getattr(b,name),rtol=2e-13,atol=2e-14)
    np.testing.assert_allclose(coords[...,7:9],central_energy_flux_lu(f,g),atol=2e-14)
    np.testing.assert_allclose(heat_flux_moments(restored,mapping),heat_flux_moments(z,mapping),atol=2e-14)
    rho0=mapping.lattice.rho_ref_lu
    p0=rho0*mapping.theta_ref_lu
    np.testing.assert_allclose(coords[...,3],np.log(a.p/p0)-a.gamma*np.log(a.rho/rho0),atol=2e-14)

@pytest.mark.parametrize('bad', [0., -1.])
def test_pressure_entropy_coordinates_reject_nonpositive_density_or_pressure(bad):
    from core.kinetic_moments import pressure_entropy_coordinates, moments_from_pressure_entropy
    mapping,f,g=state()
    z=extract_moments(f,g)
    invalid=z.copy()
    invalid[...,0]=bad
    with pytest.raises(ValueError,match='rho'):
        pressure_entropy_coordinates(invalid,mapping)
    invalid=z.copy()
    invalid[...,3]=.5*np.sum(z[...,1:3]**2,axis=-1)/z[...,0]+bad
    with pytest.raises(ValueError,match='pressure'):
        pressure_entropy_coordinates(invalid,mapping)
    coords=pressure_entropy_coordinates(z,mapping)
    coords[...,0]=bad
    with pytest.raises(ValueError,match='pressure'):
        moments_from_pressure_entropy(coords,mapping)
