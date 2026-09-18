"""周期双侧热带的逐步保守膜气交换；区间热流不作滞后反馈。"""
from __future__ import annotations

import numpy as np

from boundary.wall_thermal_mass_neutral import make_symmetric_mass_neutral_wall_callback
from .drive import evaluate_drive
from .energy_audit import EnergyAuditResult


def run_conservative_conjugate(*, solver, params, drive, n_steps, dt_si=None,
                              T_initial_K=None, energy_tolerance=1e-2,
                              probe=None, grad_extrap="row1"):
    """热壁重构与膜储能在同一步联立；只支持封闭周期域、对称双侧。

    gas_energy 包含湿壁节点，热带两侧合计对应一张膜的面积。湿壁密度由
    流步决定且重构保持不变，因此其零速度总能量严格线性于新膜温。输入
    功端点梯形、漏热中点；无需差分箱压力、非线性迭代或经验反馈系数。
    每步独立记录气体实际能量，碰撞/流步漂移不吸收进膜交换来掩盖错误。
    """
    from .conjugate import LevelCCouplingResult

    if not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    if params.gas_flux_factor != 2.0:
        raise ValueError("symmetric periodic gas represents two sides; gas_flux_factor must be 2")
    if solver.ny < 5 or grad_extrap not in {"row1", "linear"}:
        raise ValueError("requires ny>=5 and grad_extrap row1 or linear")
    dt = solver.mapping.lattice.dt_s
    if dt_si is not None and not np.isclose(dt_si, dt, rtol=1e-12, atol=0):
        raise ValueError("dt_si must equal the LBM time step")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("energy_tolerance must be positive and finite")
    m = solver.mapping
    T_initial = params.T_ref_K if T_initial_K is None else float(T_initial_K)
    if not np.isfinite(T_initial) or T_initial <= 0:
        raise ValueError("initial temperature must be positive and finite")
    py, px = (1, solver.nx//2) if probe is None else probe
    if not (0 <= py < solver.ny and 0 <= px < solver.nx):
        raise ValueError("probe is outside the gas domain")

    c2half = .5*np.sum(solver.lattice.c**2, axis=1)
    energy_scale = m.pressure_scale*m.lattice.dx_m
    cv_lu = .5*(m.lattice.D+m.lattice.S)

    def energy(f, g):
        return float(np.sum(f*c2half+g)/solver.nx*energy_scale)

    def wall_callback(T):
        return make_symmetric_mass_neutral_wall_callback(T/m.temperature_scale, extrap=grad_extrap)

    solver.initialize_from_macro(m.lattice.rho_ref_lu, np.zeros(2), m.theta_ref_lu)
    # 初态显式包括同温湿壁；不调用会改变质量的 pressure-preserving 初始化。
    cb = wall_callback(T_initial)
    solver.f, solver.g = cb(solver=solver, f_post=solver.f, g_post=solver.g,
                           f_stream=solver.f, g_stream=solver.g)
    t = np.arange(n_steps+1)*dt
    Ts, Tw, theta = np.empty_like(t), np.empty_like(t), np.empty_like(t)
    p, temp, mass, pmean, Eg = (np.empty_like(t) for _ in range(5))
    P = np.array([evaluate_drive(drive, ti) for ti in t])
    if not np.isfinite(P).all():
        raise ValueError("drive must be finite")
    win, wleak, exchange = (np.zeros_like(t) for _ in range(3))
    Ts[0] = T_initial

    def sample(i):
        macro = solver.get_macro()
        theta[i] = np.mean(macro.theta[0])
        Tw[i] = theta[i]*m.temperature_scale
        p[i] = macro.p[py,px]*m.pressure_scale
        temp[i] = macro.theta[py,px]*m.temperature_scale
        mass[i] = np.sum(macro.rho)
        pmean[i] = np.mean(macro.p)*m.pressure_scale
        Eg[i] = energy(solver.f,solver.g)
        if not np.isfinite(Eg[i]) or not np.isfinite(Tw[i]):
            raise FloatingPointError("non-finite coupled state; no clipping applied")

    sample(0)
    for i in range(n_steps):
        old_T = Ts[i]
        win[i+1] = .5*dt*(P[i]+P[i+1])

        def advance_wall(*, solver, f_post, g_post, f_stream, g_stream):
            row_f, row_g = f_stream[:1].copy(), g_stream[:1].copy()
            rho_wall = np.mean(np.sum(row_f, axis=-1))
            Cw = cv_lu*rho_wall*energy_scale/m.temperature_scale
            row_energy = energy(row_f,row_g)
            G = params.linear_leak_conductance_si
            delta_T = (win[i+1]-G*dt*(old_T-params.T_ref_K)
                       + row_energy-Cw*old_T)/(params.C_A_si+Cw+.5*dt*G)
            Ts[i+1] = old_T+delta_T
            if not np.isfinite(Ts[i+1]) or Ts[i+1] <= 0:
                raise FloatingPointError("non-positive or non-finite film temperature; no clipping applied")
            f_new,g_new = wall_callback(Ts[i+1])(solver=solver,f_post=f_post,g_post=g_post,
                                                f_stream=f_stream,g_stream=g_stream)
            exchange[i+1] = energy(f_new[:1]-row_f,g_new[:1]-row_g)
            wleak[i+1] = G*dt*(.5*(old_T+Ts[i+1])-params.T_ref_K)
            return f_new,g_new

        solver.step(1, boundary_callback=advance_wall)
        sample(i+1)

    storage = params.C_A_si*(Ts-Ts[0])
    # 无驱动平衡态不能用约1e-14 J/m²的舍入交换量作分母而产生假失败。
    # 仅为归一化下限；不改变分布、交换量或残差。受迫验证的尺度由输入功主导。
    roundoff_scale = 128*np.finfo(float).eps*(n_steps+1)*max(abs(Eg[0]),1.)
    scale = max(float(np.sum(abs(win))),float(np.sum(abs(exchange))),
                float(np.max(abs(storage))),roundoff_scale)

    def audit(residual):
        maximum = float(np.max(abs(residual)))
        return EnergyAuditResult(residual_si=residual,scale_si=scale,max_abs_residual_si=maximum,
            max_relative_residual=maximum/scale,final_residual_si=float(residual[-1]),
            tolerance=energy_tolerance)

    film_audit = audit(np.cumsum(win-wleak-exchange)-storage)
    total_audit = audit(np.cumsum(win-wleak)-storage-(Eg-Eg[0]))
    dTdt = np.r_[0.,np.diff(Ts)/dt]
    return LevelCCouplingResult(t_si=t,T_s_K=Ts,P_in_si=P,q_g_one_sided_si=exchange/(2*dt),
        dT_s_dt_K_s=dTdt,theta_wall_lu=theta,T_wall_K=Tw,pressure_probe_Pa=p,
        temperature_probe_K=temp,energy_audit=film_audit,coupling_scheme="conservative",
        picard_iterations=0,predictor_corrector_delta_K=np.zeros_like(t),
        wall_temperature_error_K=abs(Tw-Ts),finite=bool(np.isfinite(solver.f).all() and
        np.isfinite(solver.g).all() and np.isfinite(Ts).all()),wall_bc="mass_neutral",
        mass_lu=mass,p_box_mean_Pa=pmean,q_extraction="wall_exchange",
        q_sample_kind="preceding_interval_mean",gas_energy_J_m2=Eg,
        wall_energy_increment_J_m2=exchange,input_work_increment_J_m2=win,
        leak_work_increment_J_m2=wleak,total_energy_audit=total_audit)
