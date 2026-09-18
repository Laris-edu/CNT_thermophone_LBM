"""局部输运的动态读数及独立一维变物性 NSF 参照。"""
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from core.solver import GasSolver2D
from core.macroscopic import heat_flux_lu
from config_io import load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs/gas_air_10k_d2q37_qmatched_local_transport.yaml"


def material_ratios(T, T0=300., model="sutherland"):
    T = np.asarray(T)
    if model == "constant":
        return np.ones_like(T), np.ones_like(T)
    return ((T / T0)**1.5 * (T0 + 111.) / (T + 111.),
            (T / T0)**1.5 * (T0 + 194.) / (T + 194.))


def measure_uniform(T=300., density_ratio=1., model="sutherland", n=128, steps=500, speed=0., return_series=False):
    cfg = load_config(CONFIG)
    cfg["transport"]["model"] = model
    cfg["numerics"].update(nx=n, ny=4)
    s = GasSolver2D(cfg)
    m = s.mapping
    theta = T / m.temperature_scale
    k = 2 * np.pi / n
    phase = k * np.arange(n)
    transform = np.exp(-1j * phase)
    eps = 1e-6
    u = np.zeros((4, n, 2))
    u[..., 0] = speed
    u[..., 1] = eps * np.cos(phase)
    s.initialize_from_macro(density_ratio, u, theta)
    shear = []
    for step in range(steps + 1):
        shear.append(np.mean(s.get_macro().u[..., 1] * transform))
        if step < steps:
            s.step()
    factor = 1 + eps * np.cos(phase)
    s.initialize_from_macro(density_ratio / factor, np.array([speed, 0.]), theta * factor)
    initial = s.get_macro()
    energy0 = np.sum(initial.E_tot)
    mass0 = np.sum(initial.rho)
    entropy, q_actual, q_fourier = [], [], []
    cp = (m.lattice.D + m.lattice.S + 2) / 2
    for step in range(steps + 1):
        state = s.get_macro()
        entropy.append(np.mean(((state.theta - theta) / theta
                    - (state.gamma - 1) * (state.rho - density_ratio) / density_ratio) * transform))
        if step >= 100:
            q = heat_flux_lu(s.f, s.g, lattice=s.lattice, mapping=m)
            q_actual.append(np.mean(q[..., 0] * transform))
            theta_mode = np.mean(state.theta * transform)
            k_ratio = material_ratios(T, model=model)[1]
            q_fourier.append(-m.alpha_lu * cp * k_ratio * 1j * k * theta_mode)
        if step < steps:
            s.step()
    time = np.arange(steps + 1)
    fit = time >= 100
    nu_measured = -np.polyfit(time[fit], np.log(np.abs(shear)[fit]), 1)[0] / k**2
    alpha_measured = -np.polyfit(time[fit], np.log(np.abs(entropy)[fit]), 1)[0] / k**2
    mu_ratio, k_ratio = material_ratios(T, model=model)
    nu_expected = m.nu_lu * mu_ratio / density_ratio
    alpha_expected = m.alpha_lu * k_ratio / density_ratio
    q_actual, q_fourier = np.asarray(q_actual), np.asarray(q_fourier)
    q_ratio = np.vdot(q_fourier, q_actual) / np.vdot(q_fourier, q_fourier)
    final = s.get_macro()
    row = dict(T_K=T, density_ratio=density_ratio, model=model, n=n, speed_lu=speed,
                nu_ratio=float(nu_measured / nu_expected), alpha_ratio=float(alpha_measured / alpha_expected),
                q_ratio_real=float(q_ratio.real), q_ratio_imag=float(q_ratio.imag),
                mass_relative_error=float(abs(np.sum(final.rho) / mass0 - 1)),
                energy_relative_error=float(abs(np.sum(final.E_tot) / energy0 - 1)),
                min_f=float(s.f.min()), min_g=float(s.g.min()))
    if return_series:
        row["arrays"] = dict(time_lu=time, shear=np.asarray(shear), entropy=np.asarray(entropy),
                             q_actual=q_actual, q_fourier=q_fourier)
    return row


def measure_nonuniform(model="sutherland", n=128, steps=300, amplitude=.1, refine=1, return_series=False):
    """270–330 K 周期等压初态；与独立守恒形式 NSF 的完整演化比较。"""
    cfg = load_config(CONFIG)
    cfg["transport"]["model"] = model
    cfg["lattice"]["dx_m"] /= refine
    cfg["lattice"]["dt_s"] /= refine
    n *= refine
    steps *= refine
    cfg["numerics"].update(nx=n, ny=4)
    s = GasSolver2D(cfg)
    m = s.mapping
    theta0 = m.theta_ref_lu
    cp, cv = 3.5, 2.5
    theta_initial = theta0 * (1 + amplitude * np.cos(2 * np.pi * np.arange(n) / n))
    rho_initial = theta0 / theta_initial
    s.initialize_from_macro(rho_initial, np.zeros(2), theta_initial)
    initial = s.get_macro()
    energy0 = np.sum(initial.E_tot)
    wave = 2 * np.pi * np.fft.fftfreq(n)

    def derivative(a):
        return np.fft.ifft(1j * wave * np.fft.fft(a)).real

    def rhs(t, conserved):
        rho, momentum, energy = conserved.reshape(3, n)
        u = momentum / rho
        theta = (energy / rho - .5 * u**2) / cv
        mu_ratio, k_ratio = material_ratios(theta * m.temperature_scale, model=model)
        # D=2：纵向偏应力系数为 mu（非三维的 4mu/3）；本配置 bulk=0。
        stress = m.nu_lu * mu_ratio * derivative(u)
        q = -m.alpha_lu * cp * k_ratio * derivative(theta)
        pressure = rho * theta
        return -np.concatenate((derivative(momentum), derivative(momentum * u + pressure - stress),
                                derivative((energy + pressure) * u - stress * u + q)))

    state0 = np.stack((rho_initial, np.zeros(n), cv * rho_initial * theta_initial))
    ref = solve_ivp(rhs, (0, steps), state0.ravel(), method="DOP853", rtol=2e-11, atol=2e-13, max_step=1.)
    if not ref.success:
        raise RuntimeError(ref.message)
    rho_ref, momentum_ref, energy_ref = ref.y[:, -1].reshape(3, n)
    theta_ref = (energy_ref / rho_ref - .5 * (momentum_ref / rho_ref)**2) / cv
    s.step(steps)
    actual = s.get_macro()
    theta_lbm = actual.theta.mean(axis=0)
    q_lbm = heat_flux_lu(s.f, s.g, lattice=s.lattice, mapping=m)[..., 0].mean(axis=0)
    q_ref = -m.alpha_lu * cp * material_ratios(theta_ref * m.temperature_scale, model=model)[1] * derivative(theta_ref)
    row = dict(model=model, n=n, steps=steps, amplitude=amplitude,
                refine=refine,
                temperature_evolution_error=float(np.linalg.norm(theta_lbm - theta_ref) / np.linalg.norm(theta_ref - theta_initial)),
                temperature_perturbation_error=float(np.linalg.norm(theta_lbm - theta_ref) / np.linalg.norm(theta_initial - theta0)),
                temperature_max_error_K=float(np.max(np.abs(theta_lbm - theta_ref)) * m.temperature_scale),
                heat_flux_relative_error=float(np.linalg.norm(q_lbm - q_ref) / np.linalg.norm(q_ref)),
                mass_relative_error=float(abs(np.sum(actual.rho) / np.sum(initial.rho) - 1)),
                energy_relative_error=float(abs(np.sum(actual.E_tot) / energy0 - 1)),
                min_f=float(s.f.min()), min_g=float(s.g.min()))
    if return_series:
        row["arrays"] = dict(theta_initial=theta_initial, theta_lbm=theta_lbm, theta_nsf=theta_ref,
                             rho_lbm=actual.rho.mean(axis=0), rho_nsf=rho_ref, q_lbm=q_lbm, q_nsf=q_ref)
    return row
